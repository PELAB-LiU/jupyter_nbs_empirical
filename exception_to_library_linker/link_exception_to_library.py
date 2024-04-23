import config
from pathlib import Path
from typing import List, Iterator
import os
import pandas as pd

import pickle
from wmutils.file import iterate_through_files_in_nested_folders
from wmutils.multithreading import parallelize_tasks

import util
import imports_parser


def __main():
    notebook_path = config.path_default.joinpath("notebooks")
    notebook_entry_path = config.path_default.joinpath("nberror_g_all.xlsx")

    link_exceptions_to_library_2(notebook_path, notebook_entry_path)


def link_exception_to_library(notebook_path: Path) -> str | None:
    pass


def link_exceptions_to_library(notebook_directory: Path, notebook_names: Iterator[str]):
    """Links all exceptions in the provided notebooks to their underlying library."""
    skipped = 0
    processed = 0
    for name in notebook_names:
        notebook_path = notebook_directory.joinpath(name)
        if not os.path.exists(notebook_path):
            skipped += 1
            continue
        link_exception_to_library(notebook_path)
        processed += 1

    print(
        "Skipped {0:.2f}% of the notebooks".format(
            skipped / (skipped + processed) * 100
        )
    )


def link_exceptions_to_library_2(notebook_directory: Path, dataframe_path: Path):
    df = pd.read_excel(dataframe_path, header=0)
    
    def has_notebook_file(notebook_name:str) -> bool:
        return os.path.exists(notebook_directory.joinpath(notebook_name))
    mask = df['fname'].apply(has_notebook_file)
    df = df[mask]

    print(f"{df.columns=}, {len(df)}")

    # Applies method 1
    df["lib"] = df["traceback"].apply(util.extract_lib)
    null_count = 100 * (len(df) - len(df[df["lib"].isnull()])) / len(df)
    print(
        f"Extracted {null_count:.2f}% of the underlying packages using simple method."
    )

    # Applies method 2
    pickle_path = config.path_default.join('lib_classes.pickle')
    if not os.path.exists(pickle_path):
        library_classes = util.export_classes_from_modules(config.top_lib_names, pickle_path)
    else:
        with open(pickle_path, "rb") as pickle_file:
            library_classes = pickle.load(pickle_file)
    print("Loaded pickled class data.")
    nb_imports: pd.DataFrame = construct_imports_df(notebook_directory, df)
    print('Loaded nb imports.')
    kwargs = {
        "lib_names": config.top_lib_names,
        "df_imports": nb_imports,
        "lib_classes_dict": library_classes,
    }
    df["lib2"] = df["traceback"].apply(util.extract_lib_2, **kwargs)
    null_count = 100 * (len(df) - len(df[df["lib2"].isnull()])) / len(df)
    print(
        f"Extracted {null_count:.2f}% of the underlying packages using the less simple method."
    )


def construct_imports_df(notebook_directory: Path, df: pd.DataFrame) -> pd.DataFrame:
    def __get_imports(notebook: str) -> str:
        """Returns the imports of the specified notebook."""
        notebook_path = notebook_directory.joinpath(notebook)
        imports = imports_parser.get_imports_nbs_static(
            notebook_path, get_imports_func=imports_parser.get_imports_line_outermost
        )

        import_str = ",".join(imports)
        return import_str

    df = df[["fname"]].copy()
    df["imports"] = df["fname"].apply(__get_imports)

    return df


# util.extract_lib
# util.extract_lib_2


# util.simple_lib_parser


if __name__ == "__main__":
    __main()
