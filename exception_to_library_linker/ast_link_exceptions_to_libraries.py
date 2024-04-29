"""
The goal of this script is to link notebook exceptions to (machine learning) libraries.
It uses the Python abstract syntax tree for this.
"""

import ast
from typing import List, Iterator, Set, Tuple
from pathlib import Path
import copy
from uuid import UUID

from wmutils.collections.list_access import flatten
from wmutils.collections.safe_dict import SafeDict

from exception_to_library_linker.notebook_exception import (
    get_raw_notebook_exceptions_from,
    try_parse_notebook_exception,
    get_root_exception,
    get_cause_iterator,
    NotebookStacktraceEntry,
    RawNotebookException,
    NotebookException,
    FileStacktraceEntry,
)
from exception_to_library_linker.objects import (
    PackageImport,
    ComponentImport,
    get_library_from_package,
)
from exception_to_library_linker.notebook_to_python_conversion import (
    NotebookToPythonMapper,
)
from exception_to_library_linker.get_components_from_libraries import (
    get_lib_classes_no_singleton,
)
import config


LOG_UNSUPPORTED_STATEMENTS = False
DELETE_PY_FILE_ON_EXIT = True


def link_many_nb_exceptions_to_ml_libraries(
    notebook_paths: Iterator[Path],
) -> Iterator[List[List[Tuple[UUID, PackageImport | ComponentImport]]]]:
    for notebook_path in notebook_paths:
        yield link_exceptions_to_ml_libraries(notebook_path)


def link_exceptions_to_ml_libraries(
    notebook_path: Path, python_path: Path | None = None
) -> List[List[Tuple[UUID, PackageImport | ComponentImport]]]:
    try:
        with NotebookToPythonMapper(
            notebook_path, python_path, delete_py_file_on_exit=DELETE_PY_FILE_ON_EXIT
        ) as nb_mapper:
            # Collects all libraries and removes everything that has no ML relevance.
            libraries_per_exception = __link_exceptions_to_libraries(nb_mapper)
            ml_libraries = [
                (exception, __filter_non_ml_imports(libs))
                for exception, libs in libraries_per_exception
            ]
            ml_libraries = list(
                (exception, list(libs)) for exception, libs in ml_libraries
            )
            return ml_libraries
    except SyntaxError:
        print(f"Couldn't create AST in file {notebook_path=}, {python_path=}")
        return []


def __link_exceptions_to_libraries(
    nb_mapper: NotebookToPythonMapper,
) -> Iterator[Tuple[UUID, Set[PackageImport | ComponentImport]]]:

    exception_exclusion_list = set(config.builtin_exps_excluded)

    try:
        py_ast = __build_python_ast(nb_mapper.mapping.python_path)
    except SyntaxError:
        print(f"Couldn't create AST in file {nb_mapper.mapping.notebook_path=}")
        return

    # Loads imports from AST.
    package_imports = list(__get_package_imports(py_ast))
    component_imports = list(__get_component_imports(py_ast))
    # HACK: This only works when you use methods that they both have, like `get_used_alias`.
    imports: List[ComponentImport | PackageImport] = [
        *component_imports,
        *package_imports,
    ]

    raw_excs = get_raw_notebook_exceptions_from(nb_mapper.mapping.notebook_path)
    for raw_exc in raw_excs:

        if raw_exc.exception_name.lower() in exception_exclusion_list:
            continue

        success, exc = try_parse_notebook_exception(raw_exception=raw_exc)
        if not success:
            yield raw_exc.exception_id, []
            continue

        # TODO: Yield these somehow without yielding duplicates.
        used_libraries = __find_library_in_exception(
            exc, component_imports, package_imports
        )
        used_libraries = set(used_libraries)

        # Finds dependencies related to the statement itself.
        # TODO: We could probably filter this list, removing Python standard library functions / constants etc.
        vars = __get_exc_statement_variables(raw_exc, exc, nb_mapper, py_ast)
        used_vars = set()
        for var in vars:
            has_imp = False
            for imp in imports:
                if var == imp.get_used_alias():
                    used_libraries.add(imp)
                    has_imp = True
            if not has_imp:
                used_vars.add(var)

        root_cause = get_root_exception(exc)
        if isinstance(root_cause, NotebookStacktraceEntry):
            py_line_number = __get_py_line_number(root_cause, nb_mapper)

            # Attempts to link the identified variables to libraries through assignments in the notebook.
            new_libraries = __find_libraries_in_var_assignments(
                py_ast, used_vars, py_line_number, component_imports, package_imports
            )
            used_libraries = used_libraries.union(new_libraries)

        yield exc.exception_id, used_libraries


def __get_package_imports(py_ast: ast.Module) -> Iterator[PackageImport]:
    imports = (imp for imp in py_ast.body if isinstance(imp, ast.Import))
    imports = (
        (
            PackageImport(get_library_from_package(name.name), name.name, name.asname)
            for name in imp.names
        )
        for imp in imports
    )
    imports = flatten(imports)
    yield from imports


def __get_component_imports(py_ast: ast.Module) -> Iterator[ComponentImport]:
    # Loads all the import froms.
    importfroms = (imp for imp in py_ast.body if isinstance(imp, ast.ImportFrom))
    # HACK: idk why, but this solves multithreading issues.
    q = list(importfroms)
    importfroms = (
        (
            ComponentImport(
                get_library_from_package(entry.module),
                entry.module,
                name.name,
                name.asname,
            )
            for name in entry.names
        )
        for entry in importfroms
    )
    importfroms = flatten(importfroms)

    # Replaces all `import * from` lib with all their components.
    lib_classes = get_lib_classes_no_singleton()
    lib_classes = SafeDict(initial_mapping=lib_classes, default_value=list)
    importfroms = (
        (
            imp
            if imp.component != "*"
            else (
                ComponentImport(imp.library, imp.package, comp)
                for comp in lib_classes[imp.library]
            )
        )
        for imp in importfroms
    )
    importfroms = flatten(importfroms)

    yield from importfroms


def __filter_non_ml_imports(
    imports: Iterator[PackageImport | ComponentImport],
) -> Iterator[PackageImport | ComponentImport]:
    inclusion_list = set(config.top_lib_names)
    yield from [imp for imp in imports if imp.library in inclusion_list]


def __build_python_ast(python_path: Path) -> ast.Module:
    with open(python_path, "r", encoding="utf-8") as python_file:
        python_code = python_file.read()
    tree = ast.parse(python_code)
    return tree


def __find_library_in_exception(
    exc: NotebookException,
    comp_imports: List[ComponentImport],
    pack_imports: List[PackageImport],
) -> Iterator[str]:
    # TODO: yield the import objects instead.
    # used_libraries = set(comp.library.lower() for comp in comp_imports)
    # used_libraries = used_libraries.union(pack.library.lower() for pack in pack_imports)

    root_cause = get_root_exception(exc)
    cause_iterator = get_cause_iterator(exc)

    for cause in cause_iterator:
        if cause == root_cause:
            break

    for cause in cause_iterator:
        if not isinstance(cause, FileStacktraceEntry):
            continue

        cause_path = Path(cause.file_path)
        parts = [part.lower() for part in cause_path.parts]

        # Tests for relevant dependencies.
        for part in parts:
            # TODO: This is currently a little trigger happy as all components will trigger, however, I doubt this is an issue.
            for comp in comp_imports:
                if comp.library.lower() in part:
                    yield comp
            for pack in pack_imports:
                if pack.library.lower() in part:
                    yield pack


def __get_exc_statement_variables(
    raw_exc: RawNotebookException,
    exc: NotebookException,
    nb_mapper: NotebookToPythonMapper,
    py_ast: ast.Module,
) -> Iterator[str]:
    # Searches for the root cause.
    root_cause = get_root_exception(exc)

    if root_cause is None or not isinstance(root_cause, NotebookStacktraceEntry):
        return

    py_line_number = __get_py_line_number(root_cause, nb_mapper)

    if py_line_number is None:
        return

    # Identifies the current exception node and its leaf nodes.
    exception_node = __find_exception_node(py_ast, py_line_number)
    leafs = __iterate_through_leafs(exception_node)

    yield from leafs


def __get_py_line_number(
    root_cause: NotebookStacktraceEntry,
    nb_mapper: NotebookToPythonMapper,
) -> int:
    """Helper method to do get pyhon line number magic."""
    # TODO: Refactor this to not require 3 objects.

    cell_index = root_cause.cell_index
    if root_cause.cell_index is None:
        # TODO: It would be better if this result is stored somewhere for reuse.
        cell_index = __find_my_cell_index(root_cause, nb_mapper)

    if cell_index is None or not isinstance(cell_index, int):
        return None

    try:
        py_line_number = nb_mapper.get_python_line_number(
            cell_index, root_cause.exception_line_number - 1
        )
        return py_line_number
    except:
        print(
            f"Couldn't find python line number {nb_mapper.mapping.notebook_path=}, {cell_index=}, {root_cause.exception_line_number=}"
        )
        return None


def __find_my_cell_index(
    root_cause: NotebookStacktraceEntry, nb_mapper: NotebookToPythonMapper
) -> int | None:
    # HACK: The principle of this whole method is hacky. Optimally, there would be a better way to infer cell index from the stacktrace directly.

    nb_cells_indices = nb_mapper.mapping.nb_to_py_line_mapping.keys()
    for cell_index in nb_cells_indices:

        is_match = True
        for line_index, line in root_cause.previewed_lines.items():

            try:
                py_line = nb_mapper.get_python_line(cell_index, line_index - 1)
            except:
                is_match = False
                break

            py_line = py_line.strip()
            if py_line != line:
                is_match = False
                break

        if is_match:
            return cell_index

    return None


def __find_exception_node(py_ast: ast.Module, exception_line_number: int) -> ast.expr:
    if "body" in py_ast.__dict__:
        for element in py_ast.body:
            if element.end_lineno <= exception_line_number:
                continue
            return __find_exception_node(element, exception_line_number)
    return py_ast


def __iterate_through_leafs(expr: ast.expr) -> Iterator[str]:
    """
    Performs an in-order iteration through the expression nodes,
    yielding the AST's leave nodes.
    """

    if "value" in expr.__dict__:
        expr = expr.value
    expr_type = type(expr)
    children: List[ast.expr] = []
    match (expr_type):
        case ast.BinOp:
            # Binary operation: e.g., x = a + b
            children = [expr.left, expr.right]
        case ast.IfExp:
            # Ternary expression: x = a if p else b
            children = [expr.test, expr.body, expr.orelse]
        case ast.Call:
            # Function call: f(x, y, z, ...)
            children = [expr.func, *expr.args]
        case ast.Attribute:
            # Subscript: a[x]
            # Allows nesting.
            if "id" in expr.value.__dict__:
                yield expr.value.id
            else:
                children = [expr.value]
        case ast.Subscript:
            # Subscript: a[x]
            # Allows nesting.
            if "id" in expr.value.__dict__:
                yield expr.value.id
            else:
                children = [expr.value]
        case ast.Name:
            yield expr.id
        case ast.Constant:
            # Constants are ignored.
            pass
        case _:
            if LOG_UNSUPPORTED_STATEMENTS:
                print(f"Found unsupported expression type: {expr_type}.")

    # Yields the results of its children.
    for child in children:
        result = __iterate_through_leafs(child)
        yield from result


def __find_libraries_in_var_assignments(
    py_ast: ast.Module,
    variables: Iterator[str],
    exception_line_number: int,
    comp_imports: List[ComponentImport],
    pack_import: List[PackageImport],
) -> Iterator[str]:

    variables: Set[str] = set(variables)
    if len(variables) == 0:
        return

    # HACK: This only works when you use methods that they both have, like `get_used_alias`.
    imports: List[ComponentImport | PackageImport] = [*comp_imports, *pack_import]

    assignments = __find_assignments_in_ast_in_reversed_order(
        py_ast, exception_line_number
    )
    for assignment in assignments:

        # A copy is made to overwrite variables with their sources,
        # making it possible to search nested assignments; i.e., when
        # var `a` is assigned using variable `b`, which is assigned at
        # some earlier stage.
        updated_variables: Set[str] = copy.copy(variables)

        for var_name in variables:
            leafs = __find_relevant_assignment_sources(assignment, var_name)

            # Compares the leafs with the imports to identify a relationship.
            new_variables = set()
            for leaf in leafs:
                has_import = False
                for imp in imports:
                    if leaf == imp.get_used_alias():
                        yield imp
                        has_import = True
                if not has_import:
                    # If it is not a package name, it must be something assigned
                    # in the code itself, which we will search for.
                    new_variables.add(leaf)

            if len(new_variables) > 0:
                # Updates the target variable and adds the new ones.
                # If the variable was assigned using itself, it will be present
                # in the set of new variables. This is only necessary when the
                # assignment was relevant.
                updated_variables.remove(var_name)
                updated_variables = updated_variables.union(new_variables)

        # If there are no new variables, we stop searching.
        if len(updated_variables) == 0:
            break

        variables = updated_variables


def __find_assignments_in_ast_in_reversed_order(
    py_ast: ast.Module, start_line: int
) -> Iterator[ast.Assign | ast.AnnAssign]:
    """Yields the assignment objects in the AST in reversed order,
    starting at the line number that was provided."""
    for obj in py_ast.body[::-1]:
        if obj.end_lineno <= start_line:
            yield from __find_assignments(obj)


def __find_assignments(
    stmt: ast.stmt | ast.NamedExpr,
) -> Iterator[ast.Assign | ast.AnnAssign]:
    stmt_type = type(stmt)

    children = []
    match (stmt_type):
        case ast.For:
            children = stmt.body
        case ast.While:
            children = stmt.body
        case ast.If:
            children = [*stmt.body, *stmt.orelse]
        case ast.With:
            children = stmt.body
        case ast.Try:
            children = [*stmt.body, *stmt.handlers, *stmt.orelse, *stmt.finalbody]
        case ast.TryStar:
            children = [*stmt.body, *stmt.handlers, *stmt.orelse, *stmt.finalbody]
        case ast.Assign:
            yield stmt
        case ast.AnnAssign:
            yield stmt
        case ast.NamedExpr:
            # TODO: This method currently ignores the walrus operator (i.e., ':=' / ast.NamedExpr).
            pass
        case _:
            if LOG_UNSUPPORTED_STATEMENTS:
                print(f"Found unsupported statement type {stmt_type}.")
            pass

    for child in children:
        child_assignments = __find_assignments(child)
        yield from child_assignments


def __find_relevant_assignment_sources(
    assignment: ast.Assign | ast.AnnAssign, var_name: str
) -> Iterator[str]:
    if not isinstance(assignment, ast.AnnAssign) and len(assignment.targets) > 1:
        if LOG_UNSUPPORTED_STATEMENTS:
            print(f"Found multiple assignments {assignment}; this isn't supported.")
        return

    if isinstance(assignment, ast.AnnAssign):
        target = assignment.target
    else:
        target = assignment.targets[0]

    # Identifies whether the saught after variable is assigned.
    is_relevant_assignment = isinstance(target, ast.Name) and target.id == var_name
    source_index = -1
    if not is_relevant_assignment and isinstance(target, ast.Tuple):
        for i, var in enumerate(target.elts):
            if not isinstance(var, ast.Name) or var.id != var_name:
                continue
            source_index = i
            break
        is_relevant_assignment = source_index != -1

    if not is_relevant_assignment:
        return

    # Identifies the relevant source of the assignment.
    leafs = []

    if isinstance(assignment.value, ast.Tuple):
        if source_index == -1:
            leafs = [
                __iterate_through_leafs(ele)
                for ele in assignment.value.elts
                if isinstance(ele, ast.expr)
            ]
            leafs = __flatten_once(leafs)
        else:
            leafs = __iterate_through_leafs(assignment.value.elts[source_index])
    elif isinstance(assignment.value, ast.expr):
        leafs = __iterate_through_leafs(assignment.value)
    else:
        if LOG_UNSUPPORTED_STATEMENTS:
            print(f"Found unsupported assignment type: {assignment.value}.")
        return

    yield from leafs


def __flatten_once(items: Iterator[Iterator]) -> Iterator:
    """Helper method that yields the elements of a nested list with depth 1."""
    for sub_items in items:
        yield from sub_items


if __name__ == "__main__":
    from os import path
    from wmutils.file import iterate_through_files_in_nested_folders
    import tqdm

    # Having a static path helps with debugging a specific notebook.
    static_path = None
    static_path = "data/harddrive/GitHub/nbdata_error_g/nbdata_g_error_100-199/00100-7-ch21-mongodb.ipynb"
    if not static_path:
        # Loads all notebooks.
        base_folder = "./data/notebooks/nbdata_err_kaggle/nbdata_err_kaggle/nbdata_k_error/nbdata_k_error/"
        files = (
            file
            for file in iterate_through_files_in_nested_folders(
                base_folder,
                10_000,
            )
            if path.isfile(file)
        )
        DELETE_PY_FILE_ON_EXIT = True
    else:
        DELETE_PY_FILE_ON_EXIT = False
        files = [static_path]
    files = (Path(p) for p in files)

    # Links the exceptions to ML libraries.
    # And tests for how many NBs / exceptions it succeeded.
    total_nbs = 0
    total_nbs_with_exc = 0
    total_exc = 0

    tot_nbs_with_partial_links = 0
    tot_nbs_with_all_links = 0
    tot_exc_with_links = 0

    for nb_path in tqdm.tqdm(files):
        # Exceptions are traced for debugging reasons.
        try:
            ml_links = link_exceptions_to_ml_libraries(nb_path)
        except:
            print(nb_path)
            raise

        print(ml_links)
        ml_links = [link[1] for link in ml_links]

        total_nbs += 1

        if len(ml_links) > 0:
            total_nbs_with_exc += 1

            if all(len(link) > 0 for link in ml_links):
                tot_nbs_with_all_links += 1

            if any(len(link) > 0 for link in ml_links):
                tot_nbs_with_partial_links += 1

        total_exc += len(ml_links)
        tot_exc_with_links += sum(1 if len(links) > 0 else 0 for links in ml_links)

    print("\nResults:")

    def __safe_cal_perc(count, total):
        if total == 0:
            return 0
        return count / total * 100

    print(
        f"Notebooks with relevant exceptions and that can be parsed {total_nbs_with_exc}/{total_nbs} ({__safe_cal_perc(total_nbs_with_exc, total_nbs):.2f}%)"
    )

    print(
        f"Notebooks with partial ML links {tot_nbs_with_partial_links}/{total_nbs_with_exc} ({__safe_cal_perc(tot_nbs_with_partial_links, total_nbs_with_exc):.2f}%)"
    )

    print(
        f"Notebooks with all ML links {tot_nbs_with_all_links}/{total_nbs_with_exc} ({__safe_cal_perc(tot_nbs_with_all_links, total_nbs_with_exc):.2f}%)"
    )

    print(
        f"Exceptions with ML links {tot_exc_with_links}/{total_exc} ({__safe_cal_perc(tot_exc_with_links, total_exc):.2f}%)"
    )
