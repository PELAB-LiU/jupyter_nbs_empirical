# NOTE: THIS IS OLD

"""
Implements two scripts that derive data from notebooks using AST.
- `get_assignment_source_id`
- `get_imports`
"""

import ast
from enum import Enum
from pathlib import Path
from typing import Tuple, List

from wmutils.collections.list_access import flatten
from wmutils.collections.safe_dict import SafeDict

from exception_to_library_linker.translate_nb_exception_to_python import (
    PythonNotebookToPythonConverter,
)

from exception_to_library_linker.objects import (
    PackageImport,
    ComponentImport,
    get_library_from_package,
)

from get_components_from_libraries import get_lib_classes


class AssignmentReturn(Enum):
    Value = 1
    Method = 2
    MultipleValue = 3
    MultipleMethod = 4
    Mixed = 5
    Other = 6


class AssignmentSourceException(Exception):
    pass


def get_assignment_source_id(
    line: str,
) -> None | Tuple[AssignmentReturn, str | List[str]]:
    tree = ast.parse(line)

    if len(tree.body) == 0:
        raise AssignmentSourceException(
            "No assignment found, you're probably parsing a comment."
        )

    assignment = tree.body[0]

    if not isinstance(assignment, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        raise AssignmentSourceException(f'Line "{line}" is not an assignment.')

    value = assignment.value
    results = __inner_get_assignment_source_id(value)

    return results


def __inner_get_assignment_source_id(expr: ast.expr):
    """Helper method to allow recursion."""
    if isinstance(expr, (ast.Attribute, ast.Subscript)):
        if 'id' in expr.value.__dict__:
            return AssignmentReturn.Value, expr.value.id
        return __inner_get_assignment_source_id(expr.value)
    elif isinstance(expr, ast.Call):
        # Finds the root of the function.
        # This is commonly the underlying package
        id = None
        current = expr.func
        while id is None:
            if "id" not in current.__dict__:
                if 'value' in current.__dict__:
                    current = current.value
                else:
                    current = current.func
                continue
            id = current.id
        return AssignmentReturn.Method, id
    elif isinstance(expr, ast.Name):
        return AssignmentReturn.Value, expr.id

    inner_expr = None
    if isinstance(expr, ast.BinOp):
        inner_expr = [expr.left, expr.right]
    elif isinstance(expr, ast.IfExp):
        inner_expr = [expr.body, expr.orelse]

    if inner_expr is None:
        # print(f"Unsupported expression: {expr}")
        return AssignmentReturn.Other, None

    inner_values = [__inner_get_assignment_source_id(entry) for entry in inner_expr]
    all_values = all(entry[0] == AssignmentReturn.Value for entry in inner_values)
    all_methods = all(entry[0] == AssignmentReturn.Method for entry in inner_values)

    key = (
        AssignmentReturn.MultipleValue
        if all_values
        else (
            AssignmentReturn.MultipleMethod if all_methods else AssignmentReturn.Mixed
        )
    )
    inner_values = [entry[1] for entry in inner_values]
    inner_values = flatten(inner_values)
    inner_values = [entry for entry in inner_values if not entry is None]

    if len(inner_values) == 0:
        return AssignmentReturn.Other, None

    return key, inner_values


def get_imports(
    python_code: str,
) -> Tuple[List[PackageImport], List[ComponentImport]]:
    tree = ast.parse(python_code)

    # Generates the package imports.
    imports = (imp for imp in tree.body if isinstance(imp, ast.Import))
    imports = (
        (
            PackageImport(get_library_from_package(name.name), name.name, name.asname)
            for name in imp.names
        )
        for imp in imports
    )
    imports = flatten(imports)
    imports = list(imports)

    # Generates the component imports.
    importfroms = (imp for imp in tree.body if isinstance(imp, ast.ImportFrom))
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

    # if any of the importfroms is a *, it loads all the components of that library.
    lib_classes = get_lib_classes()
    lib_classes = SafeDict(initial_mapping=lib_classes, default_value=list)
    importfroms = [
        (
            entry
            if entry.component != "*"
            else (
                ComponentImport(entry.library, entry.package, comp)
                for comp in lib_classes[entry.library]
            )
        )
        for entry in importfroms
    ]
    importfroms = flatten(importfroms)
    importfroms = list(importfroms)

    return imports, importfroms


def main():
    with PythonNotebookToPythonConverter(Path("./test_nb.ipynb")) as nb_converter:
        py_path = nb_converter.get_python_file_name()

        with open(py_path, "r", encoding="utf-8") as py_file:
            python_code = py_file.readlines()
            python_code = list(python_code)

            tests = [5, 7, 9, 12, 14, 16, 19, 21, 23, 25, 27, 30, 32, 34, 40]
            tests = [x + 18 for x in tests]

            for test in tests:
                my_line = python_code[test].strip()
                print(my_line)
                results = get_assignment_source_id(my_line)
                print(f'Line {test + 1}: "{my_line}"')
                print(results)
                print()


def main2():
    # Generates an AST from the python file corresponding
    # to the provided notebook path.

    with PythonNotebookToPythonConverter(Path("./test_nb.ipynb")) as nb_converter:
        py_path = nb_converter.get_python_file_name()

        with open(py_path, "r", encoding="utf-8") as py_file:
            python_code = py_file.read()

    imports = get_imports(python_code)
    print(imports)


if __name__ == "__main__":
    main()
    main2()
