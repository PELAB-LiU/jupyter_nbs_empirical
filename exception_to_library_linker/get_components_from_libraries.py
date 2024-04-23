# lib2 parsing config
import pickle
import config
from pathlib import Path
from typing import Dict, List

# TODO: Currently the pickle file doesn't respect subpackages and just returns everything. Doing this might make the process quite a bit faster.
PICKLE_PATH = config.path_default.joinpath("lib_classes.pickle")
__LIB_CLASSES_DICT: None | Dict[str, List[str]] = None


def init(pickle_path: Path = PICKLE_PATH):
    global __LIB_CLASSES_DICT
    print(f'Initializing lib classes from "{pickle_path}"')
    with open(pickle_path, "rb") as f:
        __LIB_CLASSES_DICT = pickle.load(f)


def get_lib_classes() -> Dict[str, List[str]]:
    if __LIB_CLASSES_DICT is None:
        init()
    return __LIB_CLASSES_DICT


if __name__ == "__main__":
    lib_classes = get_lib_classes()
    print(lib_classes.keys())
