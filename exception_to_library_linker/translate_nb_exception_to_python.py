from pathlib import Path
import json
from io import TextIOWrapper
import os
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class CodeCell:
    cell_start: int = -1
    cell_end: int = -1
    code_cell_index: int = -1


DEFAULT_DELETE_ON_EXIT = True


def set_default_delete_on_exit(value: bool):
    global DEFAULT_DELETE_ON_EXIT
    print("Don't do this.")
    DEFAULT_DELETE_ON_EXIT = value


class PythonNotebookToPythonConverter:

    def __init__(
        self, notebook_path: Path, delete_on_exit: bool = DEFAULT_DELETE_ON_EXIT
    ) -> None:
        self.__notebook_path = notebook_path
        self.__delete_on_exit = delete_on_exit
        self.__python_code = []
        self.__line_mapping = None
        self.__output_path = self.get_python_file_name()
        if not self.__delete_on_exit:
            print(f"{self.__output_path=}")

    def __enter__(self) -> "PythonNotebookToPythonConverter":
        self.convert()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        if self.__delete_on_exit and os.path.exists(self.__output_path):
            os.remove(self.__output_path)

    def convert(self) -> None:
        with open(self.__notebook_path, "r", encoding="utf-8") as notebook_file:
            notebook = json.loads(notebook_file.read())

        with open(self.__output_path, "w+", encoding="utf-8") as output_file:
            mapping = self.__generate_python(output_file, notebook)
            self.__line_mapping = mapping

    def __generate_python(
        self, output_file: TextIOWrapper, notebook: dict
    ) -> Dict[int, Tuple[int, int]]:
        line_counter = 0

        cell_to_lines_mapping = dict()

        code_cell_index = 0
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue

            source: str = cell["source"]
            if not isinstance(source, list):
                source = source.split("\n")
            source_line_count = len(source)

            # Line indices
            cell_start = line_counter
            cell_end = cell_start + source_line_count
            # The mapping boundaries are inclusive.
            cell_to_lines_mapping[cell_index] = CodeCell(
                cell_start, cell_end, code_cell_index
            )
            code_cell_index += 1
            line_counter = cell_end + 1

            for line in source:
                output_file.write(f"{line}\n")
                self.__python_code.append(line)

        return cell_to_lines_mapping

    def cell_line_number_to_python_line_number(
        self, cell_index: int, line_number: int
    ) -> int:
        """
        Generates line number in the parsed python file using
        notebook cell information.
        :param cell_index: zero-indexed, index of the addressed cell.
        :param line_number: one-indexed, line number within the cell.
        """
        cell: CodeCell = self.__line_mapping[cell_index]
        python_line = cell.cell_start + line_number - cell.code_cell_index - 1
        if python_line > cell.cell_end:
            raise ValueError(
                "Provided line number is falls not within cell's line range."
            )
        return python_line

    def get_python_line_from_cell(
        self, cell_index: int, cell_line_number: int
    ) -> Tuple[int, str]:
        line_number = self.cell_line_number_to_python_line_number(
            cell_index, cell_line_number
        )
        line = self.__python_code[line_number]
        return line_number, line

    def get_python_line(self, line_index: int) -> str:
        return self.__python_code[line_index]

    def get_python_file_name(self) -> Path:
        return self.__notebook_path.with_suffix(".py")
