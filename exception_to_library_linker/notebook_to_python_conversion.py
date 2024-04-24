from pathlib import Path
import nbformat
from nbconvert import PythonExporter
import regex as re
from typing import Iterator, Dict, List
from dataclasses import dataclass
import json
import os
from numbers import Number


class NotebookToPythonMapper:
    def __init__(
        self, notebook_path: Path, delete_py_file_on_exit: bool = True
    ) -> None:
        self.__notebook_path = notebook_path
        self.__delete_py_file_on_exit = delete_py_file_on_exit
        self.mapping: NotebookToPythonMapping | None = None

    def __enter__(self) -> "NotebookToPythonMapper":
        self.mapping = convert_notebook_to_python(self.__notebook_path)
        if not self.__delete_py_file_on_exit:
            print(f"{self.mapping.notebook_path=}")
            print(f"{self.mapping.python_path=}")
        return self

    def __exit__(self, *_, **__):
        if self.__delete_py_file_on_exit:
            os.remove(self.mapping.python_path)

    def get_python_line_number(self, cell_index: int, cell_line_number: int) -> int:
        return self.mapping.nb_to_py_line_mapping[cell_index][cell_line_number]

    def get_python_line(self, cell_index: int, cell_line_number: int) -> str:
        line_number = self.get_python_line_number(cell_index, cell_line_number)
        line = self.mapping.python_lines[line_number]
        return line


@dataclass(frozen=True)
class NotebookToPythonMapping:
    notebook_path: Path
    python_path: Path
    nb_to_py_line_mapping: Dict[int, Dict[int, int]]
    python_lines: List[str]
    notebook: Dict


@dataclass(frozen=True)
class CellLineRange:
    cell_index: int
    cell_run_index: int
    cell_line_range_start: int
    cell_line_range_end: int


def convert_notebook_to_python(notebook_path: Path) -> NotebookToPythonMapping:
    python_path = __convert(notebook_path)
    cell_line_ranges = __find_cell_line_ranges(python_path)
    nb_to_py_lines_mapping, notebook, python_code = __map_all_nb_to_py_lines(
        python_path, notebook_path, cell_line_ranges
    )
    mapping = NotebookToPythonMapping(
        notebook_path=notebook_path,
        python_path=python_path,
        nb_to_py_line_mapping=nb_to_py_lines_mapping,
        python_lines=python_code,
        notebook=notebook,
    )
    return mapping


def __convert(notebook_path: Path) -> Path:
    # Create an instance of PythonExporter
    exporter = PythonExporter()

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook_content = f.read()

    # Convert notebook to Python script
    notebook = nbformat.reads(notebook_content, as_version=4)
    (python_script, _) = exporter.from_notebook_node(notebook)

    # Write the Python script to a file
    python_output_path = notebook_path.with_suffix(".py")
    with open(python_output_path, "w", encoding="utf-8") as f:
        f.write(python_script)

    return python_output_path


def __find_cell_line_ranges(python_path: Path) -> Iterator[CellLineRange]:
    with open(python_path, "r", encoding="utf-8") as python_file:
        python_code = python_file.readlines()

    cell_line_range_start = 0
    code_cell_index = -1
    previous_cell_run_index = None
    cell_start_pattern = re.compile(r"^#\s*In\[(\d+|\s+)\]:$")
    for line_number, line in enumerate(python_code):

        # When `nbconvert` converts python files, the cell of each
        # code is prefixed with a line `# In[X]` where X is the index
        # of the cell in the notebook's execution order or some spaces
        # if it wasn't run.
        cell_run_index = re.match(cell_start_pattern, line)
        if not cell_run_index:
            continue

        # The notebook is prefixed with a shabang and metadata.
        # This is ignored.
        if code_cell_index > -1:
            line_range = CellLineRange(
                cell_index=code_cell_index,
                cell_run_index=previous_cell_run_index,
                cell_line_range_start=cell_line_range_start + 1,
                cell_line_range_end=line_number,
            )
            yield line_range

        cell_run_index = cell_run_index.group(1)
        if len(cell_run_index.strip()) > 0:
            cell_run_index = int(cell_run_index)
        previous_cell_run_index = cell_run_index

        cell_line_range_start = line_number
        code_cell_index += 1

    # Yields the last entry.
    line_range = CellLineRange(
        cell_index=code_cell_index,
        cell_run_index=cell_run_index,
        cell_line_range_start=cell_line_range_start + 1,
        cell_line_range_end=line_number,
    )
    yield line_range


def __map_all_nb_to_py_lines(
    python_path: Path, notebook_path: Path, cell_ranges: Iterator[CellLineRange]
) -> Dict[int, Dict[int, int]]:
    with open(python_path, "r", encoding="utf-8") as python_file:
        python_code = python_file.readlines()

    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        nb = json.loads(notebook_file.read())

    nb_to_py_mapping = {}
    code_cell_index = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        cell_range = next(cell_ranges)

        cell_range_start, cell_range_end = (
            cell_range.cell_line_range_start,
            cell_range.cell_line_range_end,
        )
        python_code_range = python_code[cell_range_start:cell_range_end]

        nb_code = cell["source"]

        if isinstance(nb_code, str):
            nb_code = re.split(re.compile(r"\r?\n|\r\n?"), nb_code)

        mapping = __map_nb_to_py_lines(python_code_range, nb_code)
        mapping = {
            key: __safe_add(value, cell_range.cell_line_range_start)
            for key, value in mapping.items()
        }
        nb_to_py_mapping[code_cell_index] = mapping

        code_cell_index += 1

    return nb_to_py_mapping, nb, python_code


def __safe_add(a: Number | None, b: Number) -> Number | None:
    if a is None:
        return None
    return a + b


def __map_nb_to_py_lines(
    python_lines: List[str], notebook_lines: List[str]
) -> Dict[int, int]:
    # You might want to clean the lines, but don't do this; it messes up indexing.
    # python_lines = [line.strip() for line in python_lines if len(line.strip()) > 0]
    # notebook_lines = [line.strip() for line in notebook_lines if len(line.strip()) > 0]

    # The dict is initialized with `None` as not all NB lines can
    # be mapped to python lines (e.g., magic NB methods).
    mapping = {nb_line_index: None for nb_line_index in range(len(notebook_lines))}

    last_py_line = 0
    for nb_line_index, nb_line in enumerate(notebook_lines):
        for py_line_index, py_line in enumerate(
            python_lines[last_py_line:], start=last_py_line
        ):
            # The stripped string is used for comparison because of newlines
            # which might be present in one line but not in the other.
            is_match = py_line.strip() == nb_line.strip()
            if is_match:
                last_py_line = py_line_index + 1
                mapping[nb_line_index] = py_line_index
                break

    return mapping


if __name__ == "__main__":
    """
    Tests the python notebook conversion and line mapping creation.
    """

    p = "/workspaces/jupyter_nbs_empirical/data/notebooks/nbdata_err_kaggle/nbdata_err_kaggle/nbdata_k_error/nbdata_k_error/2304/younggeng_notebookaa0e30a6b5.ipynb"
    p = Path(p)

    with NotebookToPythonMapper(p) as nb_mapper:
        x = nb_mapper.get_python_line_number(1, 1)
        print(x)
