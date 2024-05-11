from pathlib import Path
import os
from typing import Dict, List
import json
import hashlib

from tqdm import tqdm

from wmutils.file import iterate_through_files_in_nested_folders
from wmutils.collections.safe_dict import SafeDict


def many_quick_create_notebook_to_python_mapping(
    nb_dirs: List[Path], py_dirs: List[Path], base_folder: Path
) -> SafeDict[Path, None | Dict[str, Path]]:
    if len(nb_dirs) != len(py_dirs):
        raise ValueError("Different number of notebook and python directories.")

    all_mappings = SafeDict(default_value=None, delete_when_default=False)

    for nb_dir, py_dir in zip(nb_dirs, py_dirs):
        mapping = quick_create_notebook_to_python_mapping(nb_dir, py_dir, base_folder)
        if not mapping is None: 
            all_mappings.update(mapping)

    print(f"Loaded {len(all_mappings)} mappings from {len(nb_dirs)} directories.")
    return all_mappings


def quick_create_notebook_to_python_mapping(
    notebook_directory: Path, python_directory: Path, base_folder: Path
) -> SafeDict[Path, None | Dict[str, Path]]:
    """Loads a stored copy of the NB to python mapping if it exists, otherwise,
    it builds it from scratch and stores it in a file. Using a cached version
    speed up the process quite a lot."""

    if not os.path.exists(base_folder):
        os.path.makedirs(base_folder)

    m = hashlib.md5()
    m.update(str(notebook_directory).encode())
    m.update(str(python_directory).encode())
    h = m.hexdigest()

    file_name = f"nb_to_py_map_{h}.json"
    quick_data_path = base_folder.joinpath(file_name)

    if os.path.exists(quick_data_path):
        print(f"Loading cached mapping from '{quick_data_path}'.")
        # Loads the quick data.
        with open(quick_data_path, "r", encoding="utf-8") as quick_data_file:
            data = quick_data_file.read()
        mapping = json.loads(data)
    else:
        print(f"Coulnd't find '{quick_data_path}', making a new one.")
        # Builds the mapping and writes it to a quick access file.
        mapping = create_notebook_to_python_mapping(
            notebook_directory, python_directory
        )
        with open(quick_data_path, "w+", encoding="utf-8") as quick_data_file:
            quick_data_file.write(json.dumps(mapping))

    mapping = SafeDict(initial_mapping=mapping, default_value=None, delete_when_default=False)
    return mapping


def create_notebook_to_python_mapping(
    notebook_directory: Path, python_directory: Path
) -> Dict[Path, Dict[str, Path]]:
    """Creates a NB to python file mapping from scratch."""

    # Loads notebook paths
    nb_paths = (
        Path(name)
        for name in tqdm(
            iterate_through_files_in_nested_folders(
                notebook_directory, max_depth=10_000
            )
        )
    )
    nb_paths = {file.stem: file for file in nb_paths}

    # Loads python paths.
    py_paths = (
        Path(name)
        for name in tqdm(
            iterate_through_files_in_nested_folders(python_directory, max_depth=10_000)
        )
    )
    py_paths = {file.stem: file for file in py_paths}

    # Builds mapping.
    mapping = {}
    for nb_name, nb_path in nb_paths.items():
        entry = {
            "nb_path": str(nb_path), 
            "py_path": str(py_paths[nb_name]) if nb_name in py_paths else None
        }
        mapping[str(nb_name)] = entry

    print(f"Loaded {len(mapping)} mappings from {len(nb_paths)} NBs and {len(py_paths)} Python files.")

    return mapping


def replace_prefix(data_path: Path, old: Path, new: Path):
    # NOTE: There is no reason for anyone to ever use this.
    with open(data_path, 'r', encoding='utf-8') as data_file:
        j_data = json.loads(data_file.read())
    
    def __posixify(path: Path):
        parts = str(path).split("\\")
        return str(Path(*parts))

    old = __posixify(old)
    new = str(new)

    def __replace(path: str | None) -> str | None:
        if path is None:
            return None
        path = __posixify(path)
        path = path[len(old):]
        path= f'{new}/{path}'
        path = Path(path).absolute()
        return str(path)

    j_data = {key: {'nb_path': __replace(value['nb_path']), "py_path": __replace(value['py_path'])} for key, value in j_data.items()}

    with open(data_path.with_suffix(".json.tmp"), 'w+', encoding='utf-8') as data_file:
        data_file.write(json.dumps(j_data))
    
    

if __name__ == "__main__":
    base_folder = Path("/workspaces/jupyter_nbs_empirical/data/nb_to_py_mappings")

    py_paths = [
        Path("/workspaces/jupyter_nbs_empirical/data/harddrive/Kaggle/k_error_pys"),
        Path("/workspaces/jupyter_nbs_empirical/data/harddrive/GitHub/g_error_pys"),
    ]

    nb_paths = [
        Path("/workspaces/jupyter_nbs_empirical/data/harddrive/Kaggle/nbdata_k_error"),
        Path("/workspaces/jupyter_nbs_empirical/data/harddrive/GitHub/nbdata_error_g"),
    ]

    mapping = many_quick_create_notebook_to_python_mapping(
        nb_paths, py_paths, base_folder
    )

    # import config
    # replace_prefix(Path('data/nb_to_py_mappings/nb_to_py_map_00a30b636bd0a88aa5debc7792defbd9.json'), config.willems_path_drive_2, config.willems_path_drive)
    # replace_prefix(Path('data/nb_to_py_mappings/nb_to_py_map_0367b1e6b4459d5197dd5e7f586b072c.json'), config.willems_path_drive_2, config.willems_path_drive)
