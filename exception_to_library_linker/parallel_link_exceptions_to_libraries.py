"""
Parallelizes linking exceptions to libraries by distirbuting files across threads.
This script simply wraps `ast_link_exceptions_to_libraries`, so for any implementation
logic refer to that.
"""

from typing import Iterator
from pathlib import Path
import os
from wmutils.multithreading import parallelize_tasks
from wmutils.collections.safe_dict import SafeDict
import json
import traceback

from exception_to_library_linker.ast_link_exceptions_to_libraries import (
    link_exceptions_to_ml_libraries,
)
from exception_to_library_linker.nb_python_path_mapper import (
    many_quick_create_notebook_to_python_mapping,
)
import config


def parallel_link_exceptions_to_ml_libraries(
    notebook_paths: Iterator[Path],
    thread_count: int = 8,
    count_only: bool = False,
    use_cached_py_files: bool = False,
) -> Path:
    """
    Does all the things `link_many_nb_exceptions_to_ml_libraries`
    does, but in parallel, so it's faster and it outputs the results
    to a JSON lines file.
    """
    tmp_data_dir = config.path_default.joinpath("tmp").joinpath("link_exc_to_lib")
    real_output_path = config.path_default.joinpath(
        "links_exceptions_to_libraries.jsonl"
    )

    if not count_only:
        __setup_output_files(thread_count, tmp_data_dir, real_output_path)

        # Sets standard tasks.
        tasks = {
            nb_path: {"nb_path": nb_path, "py_path": None} for nb_path in notebook_paths
        }

        # Overwrites standard tasks.
        if use_cached_py_files:
            tasks = __get_tasks_for_cached_py_files()
            # HACK: This is to deal with debuggin artefacts. Sometimes the .py file isn't removed.
            tasks = {
                key: value
                for key, value in tasks.items()
                if not value["nb_path"].endswith(".py")
            }

        parallelize_tasks(
            tasks=tasks.items(),
            on_task_received=__parallel_link_exceptions_to_ml_libraries,
            thread_count=thread_count,
            return_results=False,
            tmp_data_dir=tmp_data_dir,
        )

        __cleanup_output_files(thread_count, tmp_data_dir, real_output_path)

    count_output(real_output_path)

    print(f'Exception to library links stored in: "{real_output_path}".')

    return real_output_path


def __setup_output_files(thread_count: int, tmp_data_dir: Path, real_output_path: Path):
    if not os.path.exists(tmp_data_dir):
        os.makedirs(tmp_data_dir)

    # Creates outputfiles for each worker.
    for i in range(thread_count):
        worker_output_path = tmp_data_dir.joinpath(str(i))
        with open(worker_output_path, "w+") as worker_output_file:
            worker_output_file.seek(0)
            worker_output_file.write("")
            worker_output_file.truncate()

    with open(real_output_path, "w+", encoding="utf-8") as real_output_file:
        real_output_file.seek(0)
        real_output_file.write("")
        real_output_file.truncate()


def __get_tasks_for_cached_py_files():
    base_dir = config.path_default.joinpath("nb_to_py_mappings")
    py_dirs = [
        config.path_drive.joinpath("Kaggle").joinpath("nbdata_k_error"),
        config.path_drive.joinpath("GitHub").joinpath("nbdata_error_g"),
    ]
    nb_dirs = [
        config.path_drive.joinpath("Kaggle").joinpath("k_error_pys"),
        config.path_drive.joinpath("GitHub").joinpath("g_error_pys"),
    ]
    mapping: SafeDict = many_quick_create_notebook_to_python_mapping(
        py_dirs, nb_dirs, base_dir
    )
    return mapping


def __parallel_link_exceptions_to_ml_libraries(
    task: Path, tmp_data_dir: Path, worker_id: int, task_id: int, *args, **kwargs
) -> None:
    task_name, task = task

    if task_id % 100 == 0:
        print(f"Starting task {task_id}: '{task_name}' ({task})")

    nb_path = Path(task["nb_path"])
    py_path = Path(task["py_path"])

    used_cached_file = os.path.exists(py_path)
    if not used_cached_file:
        py_path = None

    try:
        ml_links = link_exceptions_to_ml_libraries(nb_path, py_path)
    except:
        tb = traceback.format_exc()
        print(f'{task=}\n{tb}')
        return
        
    ml_links = [
        (
            {
                "exception_id:": str(exception_id),
                "libraries": [imp.to_dictionary() for imp in link],
            }
        )
        for exception_id, link in ml_links
    ]
    json_entry = {
        "notebook": str(nb_path),
        "python_code": str(py_path),
        "used_cached_file": used_cached_file,
        "exc_to_lib_links": ml_links,
    }
    j_data = json.dumps(json_entry)

    output_path = tmp_data_dir.joinpath(str(worker_id))
    with open(output_path, "a", encoding="utf-8") as output_file:
        output_file.write(f"{j_data}\n")


def __cleanup_output_files(
    thread_count: int, tmp_data_dir: Path, real_output_path: Path
):
    # Creates outputfiles for each worker.
    # with open(real_output_path, 'w+', encoding='utf-8') as output_file:
    for i in range(thread_count):
        worker_output_path = tmp_data_dir.joinpath(str(i))
        __append_file(worker_output_path, real_output_path)
        os.remove(worker_output_path)

    os.removedirs(tmp_data_dir)


def __append_file(source_file: Path, target_file: Path):
    # Open both files in binary mode for efficient reading and writing
    with open(source_file, "rb") as src, open(target_file, "ab") as dest:
        # Read and write in chunks for efficiency
        for chunk in iter(lambda: src.read(4096), b""):
            dest.write(chunk)


def count_output(output_path: Path):
    with open(output_path, "r", encoding="utf-8") as output_file:
        data = output_file.readlines()

    # Links the exceptions to ML libraries.
    # And tests for how many NBs / exceptions it succeeded.
    total_nbs = 0
    total_nbs_with_exc = 0
    total_exc = 0

    tot_nbs_with_partial_links = 0
    tot_nbs_with_all_links = 0
    tot_exc_with_links = 0

    for line in data:
        j_data = json.loads(line)
        ml_links = j_data["exc_to_lib_links"]

        total_nbs += 1

        if len(ml_links) > 0:
            total_nbs_with_exc += 1

            if all(len(link["libraries"]) > 0 for link in ml_links):
                tot_nbs_with_all_links += 1

            if any(len(link["libraries"]) > 0 for link in ml_links):
                tot_nbs_with_partial_links += 1

        total_exc += len(ml_links)
        tot_exc_with_links += sum(
            1 if len(links["libraries"]) > 0 else 0 for links in ml_links
        )

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


if __name__ == "__main__":
    from wmutils.file import iterate_through_files_in_nested_folders

    files = (
        Path(file)
        for file in iterate_through_files_in_nested_folders(
            "./data/notebooks/nbdata_err_kaggle/nbdata_err_kaggle/nbdata_k_error/nbdata_k_error/",
            10_000,
        )
        if file.endswith(".ipynb") and os.path.isfile(file)
    )

    count_only = False
    output_path = parallel_link_exceptions_to_ml_libraries(
        files, thread_count=9, count_only=count_only, use_cached_py_files=True
    )
