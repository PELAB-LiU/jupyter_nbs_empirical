from pathlib import Path
from typing import Dict
import json
import pandas as pd
from wmutils.file import iterate_through_files_in_nested_folders
from wmutils.multithreading import parallelize_tasks
import config

from exception_to_library_linker.link_exception_to_library_nb import (
    link_exception_to_library,
    DELETE_PY_FILE,
)
from exception_to_library_linker.get_components_from_libraries import (
    init as lib_classes_init,
)


def link_erroneous_notebook_exceptions_to_library(
    dataframe_path: Path, notebook_directory: Path
):
    lib_classes_init()

    # Loads data.
    print(f"{dataframe_path=}")
    df = pd.read_excel(dataframe_path, header=0)
    mapping = __build_notebook_to_path_mapping(notebook_directory)
    df_len = len(df)

    # Filters entries that have no notebook.
    df["fpath"] = df["fname"].map(mapping)
    mask = df["fpath"].notna()
    df = df[mask]

    print(
        f"Removed {df_len - len(mask)}/{df_len} entries ({(df_len - len(mask)) / df_len * 100:.2f}%) that do note have a `.ipynb` file."
    )

    notebooks = df["fpath"].unique().tolist()

    # culprits = parallelize_tasks(
    #     tasks=notebooks,
    #     on_task_received=__parallelized_link_exception_to_library,
    #     thread_count=9,
    #     return_results=True,
    #     use_early_return_results=True,
    # )

    culprits = []
    for index, task in enumerate(notebooks):
        a = __parallelized_link_exception_to_library(task, index, len(notebooks))
        culprits.append(a)

    culprits = list(culprits)
    culprits = [json.loads(culprit) for culprit in culprits]
    print(f"{culprits=}")

    errors = 0
    linked_errors = 0
    for element in culprits:
        for key, value in element.items():
            errors += 1
            if value != None:
                linked_errors += 1
    print(
        f"Found {linked_errors}/{errors} ({linked_errors / errors * 100:.2f}%) error to library links."
    )


def __parallelized_link_exception_to_library(
    task, task_id: int, total_tasks: int, *args, **kwargs
):
    if task_id % 100 == 0:
        print(f"Progress {task_id}/{total_tasks} ({task_id / total_tasks * 100:.2f}%).")

    try:
        culprits = link_exception_to_library(task)
    except Exception as ex:
        print(f'Notebook "{task}" failed with message: "{ex}"')
        culprits = {}

    output = {
        key: value.to_dictionary() if value else None for key, value in culprits.items()
    }
    output = json.dumps(output)

    return output


def __build_notebook_to_path_mapping(notebook_directory: Path) -> Dict[str, Path]:
    mapping = dict()
    for file in iterate_through_files_in_nested_folders(notebook_directory, 10_000):
        file_path = Path(file).absolute()
        mapping[file_path.name] = file_path
    return mapping


def main():
    if not DELETE_PY_FILE:
        raise ValueError(
            "DELETE_PY_FILE in `link_exception_to_library_nb` can't be set to false."
        )

    dataframe_path = Path("./data/nberror_k.xlsx")
    notebook_directory = Path("./data/notebooks/")

    link_erroneous_notebook_exceptions_to_library(
        dataframe_path=dataframe_path, notebook_directory=notebook_directory
    )


if __name__ == "__main__":
    main()
