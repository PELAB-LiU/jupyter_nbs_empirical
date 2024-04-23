"""
Contains all logic necessary to link the exception that is raised in some notebook
to an underlying library. To the outside world, the only relevant method in this
script is ``link_exception_to_library``.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

from exception_to_library_linker.get_components_from_libraries import get_lib_classes

import regex as re
from wmutils.iterators import RepeatingIterator

import util
from exception_to_library_linker.translate_nb_exception_to_python import (
    PythonNotebookToPythonConverter,
    set_default_delete_on_exit,
)
from exception_to_library_linker.ast_get_assignment_source import (
    get_imports,
    get_assignment_source_id,
    AssignmentSourceException,
    AssignmentReturn,
)
from exception_to_library_linker.objects import (
    PackageImport,
    ComponentImport,
    CompositeComponentImport,
    NotebookException,
    get_library_from_package,
)

DELETE_PY_FILE = True

if not DELETE_PY_FILE:
    print("FILES ARE NOT BEING CLEANED UP. YOU DON'T WANT THIS.")


def link_exception_to_library(
    notebook_path: Path,
) -> Dict[int, PackageImport | ComponentImport]:
    """
    Attempts to link exceptions thrown by a notebook to an underlying package.
    This is done in using three methods, such that if the former failed, the following will be applying:
    1. Easy errors: Test if the libary is mentioned in the error message.
    2. Library function errors: Test whether the line on which the error is throw references a library,
       or a component imported from a library.
    3. Variable errors: Test whether any line preceding the error line in which the variable is used,
       references a library or component. Or, in case said variable is assigned using some other variable,
       that variable is associated with a library or imported component.
    """

    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook = json.loads(notebook_file.read())

    if ("nbformat" in notebook) and notebook["nbformat"] < 4:
        raise ValueError("Invalid notebook format.")

    exceptions_per_cell = _find_cells_with_exception(notebook)

    try:
        pack_dependencies_per_cell, comp_dependencies_per_cell = (
            _find_dependencies_for_all_cells(notebook)
        )
    except SyntaxError:
        return {}

    # The tree methods used to identify underlying packages.
    methods = [
        _match_evalue_with_package_dependencies,
        _match_error_line_with_dependencies,
        _match_error_with_dependencies_through_preceding_code,
    ]

    culprits = {}
    for method in methods:
        # Applies the current identification method.
        culprit_libraries = _match_all_exceptions_with_library(
            pack_dependencies_per_cell,
            comp_dependencies_per_cell,
            exceptions_per_cell,
            match_method=method,
            notebook=notebook,
            notebook_path=notebook_path,
        )

        culprits.update(culprit_libraries)

        # Removes resolved instances from the list.
        exceptions_per_cell = {
            key: value
            for key, value in exceptions_per_cell.items()
            if culprit_libraries[key] is None
        }

    return culprits


# Meta-data generation.


def _find_dependencies_for_all_cells(
    notebook: dict,
) -> Tuple[Dict[int, PackageImport], Dict[int, ComponentImport]]:
    """
    Returns a dictionary where the index of a cell refers to all of the
    imports that have been imported up until that point.
    """

    pack_dependencies_per_cell = dict()
    comp_dependencies_per_cell = dict()

    pack_dependencies: Set[PackageImport] = set()
    comp_dependencies: Set[ComponentImport] = set()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        # Has two formatting types.
        source: List[str] = cell["source"]
        if isinstance(source, list):
            source = cell["source"]
        elif isinstance(source, str):
            source = source.split("\n")

        new_pack_dependencies, new_comp_dependencies = get_imports("\n".join(source))

        pack_dependencies = pack_dependencies.union(new_pack_dependencies)
        comp_dependencies = comp_dependencies.union(new_comp_dependencies)

        pack_dependencies_per_cell[index] = pack_dependencies.copy()
        comp_dependencies_per_cell[index] = comp_dependencies.copy()

    to_replace = dict()
    for i, comp_import in enumerate(list(comp_dependencies)[:-1]):
        for other in list(comp_dependencies)[i + 1 :]:
            if (
                comp_import.library != other.library
                and comp_import.get_used_alias() == other.get_used_alias()
            ):
                merged_source = comp_import
                if comp_import in to_replace:
                    merged_source = to_replace[comp_import]
                composite_comp_import = CompositeComponentImport(merged_source, other)
                # print(f"Creating composite component import: {composite_comp_import}")
                for ele in composite_comp_import.inner_component_imports:
                    to_replace[ele] = composite_comp_import

    for old, new in to_replace.items():
        comp_dependencies.remove(old)
        comp_dependencies.add(new)

    return pack_dependencies_per_cell, comp_dependencies_per_cell


def _find_cells_with_exception(
    notebook: dict,
) -> Dict[int, Dict[str, NotebookException]]:
    """Returns dictionary where the key is the cell index and the
    value a triple containing the ename, evalue and traceback of
    the thrown exception."""

    exceptions_per_cell = dict()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue

        for output in cell["outputs"]:
            if output["output_type"] != "error":
                continue

            # Parses traceback to have a human readable format.
            traceback = output["traceback"]
            if not isinstance(traceback, list):
                traceback = [traceback]
            traceback = [util.parse_traceback_2(line) for line in traceback]

            new_exception = NotebookException(
                output["ename"], output["evalue"], traceback
            )
            if new_exception.ename == "KeyboardInterrupt":
                continue

            exceptions_per_cell[index] = new_exception

    return exceptions_per_cell


# Exception-to-library linking methods.


def _match_all_exceptions_with_library(
    pack_dependencies_per_cell: Dict[int, List[PackageImport]],
    comp_dependencies_per_cell: Dict[int, List[ComponentImport]],
    exceptions_per_cell: Dict[int, Dict[str, str]],
    match_method: Callable[
        [int, List[PackageImport], List[ComponentImport], Dict[str, str]], str | None
    ],
    **kwargs,
) -> Dict[int, str | None]:
    """Per provided exception, it finds the underlying fault library, or
    None if there is no underlying library."""

    culprit_libraries = dict()

    for cell_index, exception in exceptions_per_cell.items():
        pack_dependencies = pack_dependencies_per_cell[cell_index]
        comp_dependencies = comp_dependencies_per_cell[cell_index]

        culprit_library = match_method(
            cell_index=cell_index,
            pack_dependencies=pack_dependencies,
            comp_dependencies=comp_dependencies,
            exception=exception,
            **kwargs,
        )
        culprit_libraries[cell_index] = culprit_library

    return culprit_libraries


def _match_evalue_with_package_dependencies(
    pack_dependencies: List[PackageImport], exception: NotebookException, **_
) -> str | None:
    culprit = None

    for dependency in pack_dependencies:
        used_alias = dependency.get_used_alias()

        pat = rf".*{used_alias}.*"
        pat = re.compile(pat)

        traceback = exception.traceback
        traceback = "".join(traceback)

        # TODO: this should look at the last line of the stack trace only. if this refers to anywhere inside the script, we should look at the one above. If it points outside the script, that is the culprit of the exception. We should not consider cascading dependencies as those dependencies are of a different type.
        if re.match(pattern=pat, string=exception.evalue):
            culprit = dependency
            break

        if not culprit is None:
            break

    return culprit


def _match_error_line_with_dependencies(
    cell_index,
    pack_dependencies: List[PackageImport],
    comp_dependencies: List[ComponentImport],
    exception: NotebookException,
    notebook: dict,
    **_,
) -> PackageImport | ComponentImport | None:
    """Performs xx.yy analysis, where xx matches either a library alias or an imported component."""

    line_number = __find_line_number_in_traceback(exception)
    if line_number is None:
        return None

    exception_line = __find_line_in_notebook(notebook, cell_index, line_number)

    pack_dep = __has_pack_dependency(exception_line, pack_dependencies)
    if not pack_dep is None:
        return pack_dep

    comp_dep = __has_comp_dependency(exception_line, comp_dependencies)
    if not comp_dep is None:
        return comp_dep

    return None


def _match_error_with_dependencies_through_preceding_code(
    cell_index,
    pack_dependencies: List[PackageImport],
    comp_dependencies: List[ComponentImport],
    exception: NotebookException,
    notebook_path: Path,
    notebook: dict,
    **_,
) -> PackageImport | ComponentImport | None:
    line_number = __find_line_number_in_traceback(exception)
    if line_number is None:
        return None

    with PythonNotebookToPythonConverter(notebook_path) as nb_converter:
        py_line_number, py_line = nb_converter.get_python_line_from_cell(
            cell_index, line_number
        )
        expected_line = __find_line_in_notebook(notebook, cell_index, line_number)
        if py_line != expected_line:
            raise ValueError(
                "Generated line does not match expected line. The notebook was likely updated after the exception was raised.",
                py_line_number,
                py_line,
                expected_line,
            )

        pattern = r"[^a-zA-Z0-9._]"
        results = re.split(pattern, py_line)
        terms = (entry for entry in results if "." in entry)
        terms = (entry.split(".")[0] for entry in terms)

        # TODO: These terms could potentially be further filtered using `evalue`.
        terms = set(terms)

        rep_iterator = RepeatingIterator(range(py_line_number, 0, -1))
        for i in rep_iterator:
            previous_line = nb_converter.get_python_line(i)

            new_terms = None

            for term in terms:
                if not term in previous_line:
                    continue

                # TODO: This cannot deal with functions.

                # pack_dep = __has_pack_dependency(previous_line, pack_dependencies)
                # if not pack_dep is None:
                #     return pack_dep

                # comp_dep = __has_comp_dependency(previous_line, comp_dependencies)
                # if not comp_dep is None:
                #     return comp_dep

                # Tests if the line is an assignment to the traced variable.
                # This allows sieving for cascading dependencies, ultimately
                # leading to a reference ot a package.

                try:
                    return_type, source = get_assignment_source_id(
                        previous_line.strip()
                    )
                except AssignmentSourceException:
                    continue
                except SyntaxError:
                    # TODO: This doesn't deal with multi-line statements, ending with "\"
                    # print(f'Line contains syntax errors: "{previous_line}".')
                    continue
                except:
                    print(previous_line)
                    raise

                match return_type:
                    case AssignmentReturn.Other:
                        continue
                    case AssignmentReturn.Value | AssignmentReturn.Method:
                        new_terms = [source]
                    case (
                        AssignmentReturn.MultipleMethod
                        | AssignmentReturn.MultipleValue
                        | AssignmentReturn.Mixed
                    ):
                        new_terms = source
                break

            if new_terms:
                for new_term in new_terms:
                    if not new_term in terms:
                        rep_iterator.set_repeat_last(True)
                        terms.add(new_term)
                if not term in new_terms:
                    terms.remove(term)

                for term in terms:
                    pack_dep = __has_pack_dependency(term, pack_dependencies)
                    if not pack_dep is None:
                        return pack_dep

                    comp_dep = __has_comp_dependency(term, comp_dependencies)
                    if not comp_dep is None:
                        return comp_dep

        return None


# Helper methods.


def __has_pack_dependency(line: str, pack_dependencies: List[PackageImport]):
    # Tests package dependencies.
    for package in pack_dependencies:
        used_alias = package.get_used_alias()
        pat = rf"\b{used_alias}\b"
        pat = rf"(?<![a-zA-Z0-9_]){used_alias}(?![a-zA-Z0-9_])"
        pat = re.compile(pat)
        if re.match(pat, line):
            return package
    return None


def __has_comp_dependency(line: str, comp_dependencies: List[ComponentImport]):
    # Tests component dependencies
    for component in comp_dependencies:
        used_alias = component.component_alias
        pat = rf"\b{component}\b"
        pat = rf"(?<![a-zA-Z0-9_]){used_alias}(?![a-zA-Z0-9_])"
        pat = re.compile(pat)
        if re.match(pat, line):
            return component
    return None


def __find_line_number_in_traceback(exception: NotebookException) -> int | None:
    """Proxy method that identifies applies identification methods in the right order."""
    line_number = None
    for line in exception.traceback:
        # Identifies the line at which the error is thrown.
        new_line_number = __find_line_number_in_traceback_using_line(line)
        if not new_line_number is None:
            line_number = new_line_number

            # TODO: In general, you want to yield the lowest line number, as that is where the error actually happened. However, this error could then actually have occurred in another cell. We do not account for this. Therefore, for now, we return the first line, which is certain to be in the current cell.
            return line_number

    if not line_number is None:
        return line_number

    for line in exception.traceback:
        # Identifies the line at which the error is thrown.
        line_number = __find_line_number_in_traceback_using_arrow(line)
        if not line_number is None:
            return line_number

    return None


def __find_line_number_in_traceback_using_line(traceback: str) -> int | None:
    """Finds the line number of the traceback."""
    # Searches for lines that mention the line number.
    pat = r"(?<=line\s)\d+"
    pat = re.compile(pat)
    line_number = re.findall(pat, traceback)
    if len(line_number) == 0:
        line_number = [None]
    if len(line_number) > 1:
        raise ValueError("Multiple lines referenced in traceback.")
    line_number = line_number[0]

    if not line_number is None:
        return int(line_number)

    return None


def __find_line_number_in_traceback_using_arrow(traceback: str) -> int | None:
    # Alternative method is to search for an '--->'.
    pat = r".*-+> (\d+)"
    pat = re.compile(pat)
    line_number = re.findall(pat, traceback)
    if len(line_number) == 0:
        line_number = [None]
    # if len(line_number) > 1:
    #     raise ValueError("Multiple lines referenced in traceback.")
    line_number = line_number[0]

    if not line_number is None:
        return int(line_number)

    return None


def __find_line_in_notebook(notebook: dict, cell_index: int, line_number: int) -> str:
    """Returns the line corresponding notebook/cell.s"""
    cell = notebook["cells"][cell_index]
    if cell["cell_type"] != "code":
        raise ValueError("Specified cell does not contain code.")
    source: str = cell["source"]
    if not isinstance(source, list):
        source = source.split("\n")
    line = source[line_number - 1]
    return line


# Test main function.


if __name__ == "__main__":
    set_default_delete_on_exit(False)

    p = Path(
        "/workspaces/jupyter_nbs_empirical/data/notebooks/nbdata_err_kaggle/nbdata_err_kaggle/nbdata_k_error/nbdata_k_error/2310/alfredovillegas_alfredo-villegas-salas-first-python-notebook.ipynb"
    )
    pypath = p.with_suffix(".py")
    print(f"{pypath=}")
    culprits = link_exception_to_library(p)
    print(f"{culprits=}")
