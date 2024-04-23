"""
The goal of this script is to link notebook exceptions to (machine learning) libraries.
It uses the Python abstract syntax tree for this.
"""

import ast
from typing import List, Iterator, Set
from pathlib import Path

from wmutils.collections.list_access import flatten

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


def link_exceptions_to_libraries(notebook_path: Path) -> Iterator[str | None]:
    with NotebookToPythonMapper(
        notebook_path, delete_py_file_on_exit=False
    ) as nb_mapper:
        return __link_exceptions_to_libraries(nb_mapper)


def __link_exceptions_to_libraries(
    nb_mapper: NotebookToPythonMapper,
) -> Iterator[Set[str]]:

    py_ast = __build_python_ast(nb_mapper.mapping.python_path)

    # Loads imports from AST.
    package_imports = list(__get_package_imports(py_ast))
    component_imports = list(__get_component_imports(py_ast))

    raw_excs = get_raw_notebook_exceptions_from(nb_mapper.mapping.notebook_path)
    for raw_exc in raw_excs:
        success, exc = try_parse_notebook_exception(raw_exception=raw_exc)
        if not success:
            continue

        # TODO: Yield these somehow without yielding duplicates.
        used_libraries = __find_library_in_exception(
            exc, component_imports, package_imports
        )
        used_libraries = set(used_libraries)

        # TODO: We could probably filter this list, removing Python standard library functions / constants etc.
        vars = __get_exc_statement_variables(raw_exc, exc, nb_mapper, py_ast)
        if not vars is None:
            for var in vars:
                # TODO: The performance of this process can probably be optimized by iterating through the assignments first instead of the varnames.
                new_libraries = __find_libraries_in_var_assignments(
                    py_ast, var, component_imports, package_imports
                )
                used_libraries = used_libraries.union(new_libraries)

        yield used_libraries


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
    importfroms = (imp for imp in py_ast.body if isinstance(imp, ast.ImportFrom))
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
    yield from importfroms


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

    py_line_number = nb_mapper.get_python_line_number(
        raw_exc.cell_index, root_cause.exception_line_number - 1
    )

    if py_line_number is None:
        return

    # Identifies the current exception node and its leaf nodes.
    exception_node = __find_exception_node(py_ast, py_line_number)
    leafs = __iterate_through_leafs(exception_node)

    yield from leafs


def __find_exception_node(py_ast: ast.Module, exception_line_number: int) -> ast.expr:
    if "body" in py_ast.__dict__:
        for element in py_ast.body:
            if element.end_lineno < exception_line_number:
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
            print(f"Found unsupported type: {expr_type}.")

    # Yields the results of its children.
    for child in children:
        result = __iterate_through_leafs(child)
        yield from result


def __find_libraries_in_var_assignments(
    py_ast: ast.Module,
    var_name: str,
    comp_imports: List[ComponentImport],
    pack_import: List[PackageImport],
) -> Iterator[str]:
    for assignment in __find_assignments_in_ast(py_ast):
        leafs = __find_relevant_assignment_sources(assignment, var_name)

        # Compares the leafs with the imports to identify a relationship.
        for leaf in leafs:
            for comp in comp_imports:
                if leaf == comp.get_used_alias():
                    yield comp

            for pack in pack_import:
                if leaf == pack.get_used_alias():
                    yield pack

        # TODO: Continue here. The var_name + assignment loops must be flipped. After, the var_name that was just assigned with some other var_name should be replaced.

def __find_assignments_in_ast(
    py_ast: ast.Module,
) -> Iterator[ast.Assign | ast.AnnAssign]:
    for obj in py_ast.body:
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
            print(f"Found unsupported statement type {stmt_type}.")
            pass

    for child in children:
        child_assignments = __find_assignments(child)
        yield from child_assignments


def __find_relevant_assignment_sources(
    assignment: ast.Assign | ast.AnnAssign, var_name: str
) -> Iterator[str]:
    if len(assignment.targets) > 1:
        print(f"Found multiple assignments {assignment}")
        return

    # Identifies whether the saught after variable is assigned.
    is_relevant_assignment = (
        isinstance(assignment.targets[0], ast.Name)
        and assignment.targets[0].id == var_name
    )
    source_index = -1
    if not is_relevant_assignment and isinstance(assignment.targets[0], ast.Tuple):
        for i, var in enumerate(assignment.targets[0].elts):
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
        print(f"Found unsupported assignment type: {assignment.value}.")
        return

    yield from leafs


def __flatten_once(items: Iterator[Iterator]) -> Iterator:
    for sub_items in items:
        yield from sub_items


if __name__ == "__main__":

    p = "/workspaces/jupyter_nbs_empirical/data/notebooks/nbdata_err_kaggle/nbdata_err_kaggle/nbdata_k_error/nbdata_k_error/2304/younggeng_notebookaa0e30a6b5.ipynb"
    p = Path(p)

    q = link_exceptions_to_libraries(p)
    print(list(q))
