from typing import Iterator
from pathlib import Path
import os
from wmutils.multithreading import parallelize_tasks
import json

from exception_to_library_linker.ast_link_exceptions_to_libraries import (
    link_exceptions_to_ml_libraries,
)
import config


def parallel_link_exceptions_to_ml_libraries(
    notebook_paths: Iterator[Path], thread_count: int = 8
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

    __setup_output_files(thread_count, tmp_data_dir, real_output_path)

    parallelize_tasks(
        tasks=notebook_paths,
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
            worker_output_file.write("")

    with open(real_output_path, "w+", encoding="utf-8") as real_output_file:
        real_output_file.seek(0)
        real_output_file.write("")
        real_output_file.truncate()


def __parallel_link_exceptions_to_ml_libraries(
    task: Path, tmp_data_dir: Path, worker_id: int, task_id: int, *args, **kwargs
) -> None:
    if task_id % 100 == 0:
        print(f"Starting task {task_id}: '{task}'")

    nb_path = task

    ml_links = link_exceptions_to_ml_libraries(nb_path)
    ml_links = [[imp.to_dictionary() for imp in link] for link in ml_links]
    json_entry = {"notebook": str(nb_path), "ml_links": ml_links}
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
        data = output_file.readlines

        j_data = json.loads(line)

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
        ml_links = j_data["ml_links"]

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

    output_path = parallel_link_exceptions_to_ml_libraries(files, thread_count=8)
