from pathlib import Path

path_drive_kaggle = r"D:\Data_NBs\Kaggle"

yirans_path_default = r"C:\Users\yirwa29\Downloads\Dataset-Nb"
willems_path_default = "/workspaces/jupyter_nbs_empirical/data"
path_default = Path(yirans_path_default)

print(f"{path_default=}")

path_github_error_process = path_default.joinpath("nbdata_g_error")
path_github_error_analysis = path_github_error_process.joinpath("analysis_gerr")
path_github_error_ast = path_default.joinpath("nbdata_g_error_ast")

path_kaggle_error_process = path_default.joinpath("nbdata_k_error")
path_kaggle_error_analysis = path_kaggle_error_process.joinpath("analysis_kerr")
path_kaggle_error_ast = path_default.joinpath("nbdata_k_error_ast")

top_lib_names = [
    "pandas",
    "numpy",
    "matplotlib",
    "sklearn",
    "seaborn",
    "tensorflow",
    "torch",
    "xgboost",
    "scipy",
    "plotly",
    "cv2",
    "keras",
    "lightgbm",
    "torchvision",
    "nltk",
    "transformers",
    "catboost",
    "statsmodels",
    "imblearn",
    "wordcloud",
    "missingno",
    "optuna",
    "skimage",
    "datasets",
]

builtin_exps_excluded = ["keyboardinterrupt"]

builtin_exps_uninteresting = ["nameerror", "memoryerror", "keyboardinterrupt", "timeouterror", "windowserror",
                         "pendingdeprecationwarning", "deprecationwarning", 
                         "userwarning", "warning", "runtimewarning"
                         "modulenotfounderror", "importerror", "notimplementederror", 
                         "filenotfounderror", "fileexistserror", "isadirectoryerror", "notadirectoryerror","eoferror",  "permissionerror", "unicodeerror","unicodedecodeerror", "unicodeencodeerror"
                         "oserror", "ioerror", "brokenpipeerror", "blockingioerror",
                         "connectionerror", "connectionreseterror", "connectionrefusederror", "connectionabortederror", "systemexit"]

err_lib_percent_cutoff_k = 0.5
err_lib_count_cutoff_k = 100

err_lib_percent_cutoff_g = 0.5
err_lib_count_cutoff_g = 400

exception_list_python = [
    "generatorexit",
    "keyboardinterrupt",
    "systemexit",
    "arithmeticerror",
    "assertionerror",
    "attributeerror",
    "buffererror",
    "eoferror",
    "importerror",
    "lookuperror",
    "memoryerror",
    "nameerror",
    "oserror",
    "referenceerror",
    "runtimeerror",
    "stopasynciteration",
    "stopiteration",
    "syntaxerror",
    "systemerror",
    "typeerror",
    "valueerror",
    "warning",
    "floatingpointerror",
    "overflowerror",
    "zerodivisionerror",
    "byteswarning",
    "deprecationwarning",
    "encodingwarning",
    "futurewarning",
    "importwarning",
    "pendingdeprecationwarning",
    "resourcewarning",
    "runtimewarning",
    "syntaxwarning",
    "unicodewarning",
    "userwarning",
    "blockingioerror",
    "childprocesserror",
    "connectionerror",
    "fileexistserror",
    "filenotfounderror",
    "interruptederror",
    "isadirectoryerror",
    "notadirectoryerror",
    "permissionerror",
    "processlookuperror",
    "timeouterror",
    "indentationerror",
    "indexerror",
    "keyerror",
    "modulenotfounderror",
    "notimplementederror",
    "recursionerror",
    "unboundlocalerror",
    "unicodeerror",
    "brokenpipeerror",
    "connectionabortederror",
    "connectionrefusederror",
    "connectionreseterror",
    "taberror",
    "unicodedecodeerror",
    "unicodeencodeerror",
    "unicodetranslateerror",
    "environmenterror",
    "ioerror",
    "windowserror",
]

exclude_base_exceptions = [
    "BaseException",
    "BaseExceptionGroup",
    "Exception",
    "ExceptionGroup",
]

check_lan_list = [
    "assembly",
    "batchfile",
    "c",
    "c#",
    "c++",
    "clojure",
    "cmake",
    "cobol",
    "coffeescript",
    "css",
    "csv",
    "dart",
    "dm",
    "dockerfile",
    "elixir",
    "erlang",
    "fortran",
    "go",
    "groovy",
    "haskell",
    "html",
    "ini",
    "java",
    "javascript",
    "json",
    "julia",
    "kotlin",
    "lisp",
    "lua",
    "makefile",
    "markdown",
    "matlab",
    "objective-c",
    "ocaml",
    "pascal",
    "perl",
    "php",
    "powershell",
    "prolog",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "shell",
    "sql",
    "swift",
    "tex",
    "toml",
    "typescript",
    "verilog",
    "visual basic",
    "xml",
    "yaml",
]
