from pathlib import Path

path_drive_kaggle = r"D:\Data_NBs\Kaggle"

yirans_path_default = Path(r"C:\Users\yirwa29\Downloads\Dataset-Nb")
willems_path_default = Path(".").absolute().joinpath("data")

path_default = yirans_path_default

willems_path_drive = path_default.joinpath("harddrive")
willems_path_drive_2 = Path(r"D:\Data_NBs")
path_drive = willems_path_drive


print(f"{path_default=}")

path_github_error_process = path_default.joinpath("nbdata_g_error")
path_github_error_analysis = path_github_error_process.joinpath("analysis_gerr")
path_github_cluster_valueerr = path_github_error_analysis.joinpath("cluster_selection_valueerror_2/df_mlerr_g_mlbugs_valueerr_dedup.xlsx")

path_kaggle_error_process = path_default.joinpath("nbdata_k_error")
path_kaggle_error_analysis = path_kaggle_error_process.joinpath("analysis_kerr")
path_kaggle_cluster_valueerr = path_kaggle_error_analysis.joinpath("cluster_selection_valueerror_2/df_mlerr_k_mlbugs_valueerr_dedup.xlsx")


path_w2v_models = path_default.joinpath("retrained_embeddings")

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

NB_SOURCE = {
    "kaggle": 1,
    "github": 2
}
# summarize manual label config (categories)------------------------------
summed_label_names = ["label_root_cause", "label_ML_pipeline", "label_if_ML_bug", "label_refined_exp_type"]

# sum1
label_root_cause = {"misunderstanding of data":["misunderstanding of data structure", "misunderstanding of ML model"],
                    "API misuse":["misunderstanding of APIs","misunderstanding of libraries", "misunderstanding of types of objects", "invalid argument"],
                    "nb":["nb specific - execution order","nb specific - previous cell error","nb specific - need execute future cells"],
                    "wrong implementation":["did not import", "undefined variable", "undefined function", "typo", "wrong implementation", "uninitializated"],
                   "insufficient resources":["insufficient resources"],
                   "unknown": ["unknown"],
                   "environment problem":["module not installed", "change of environment", "file/path not found or exist", "library versions incompatible", "settings(permission, environment)", "external control (window closed)"],
                   "library issue":["API change","error inside library"]}
# sum2
label_ML_pipeline = {"environment setup":["environment setup (module not found, file/path not found)"],
                    "data prepare":["data preparation/preprocessing"],
                    "data visualization": ["data visualization"],
                    "model construction":["model construction (include compilation and visualization/summary)"],
                    "model training":["training/validation (grid search)"],
                    "model eval":["evaluation/inference (history plot, metric visualization)"],
                    "not applicable":["not-applicable (sub-labels needed, e.g., tutorials, physics simulation, ..)","not applicable - tutorial notebook","not applicable - physics","not applicable - education"],
                    "unknown":["unknown"]}
# sum3
label_if_ML_bug = {"ML bug":["ML/data science library related (ML imports, error raised by library)"],
                  "python bug":["general code error"],
                  "unknown":["unknown"]}
# sum4
label_refined_exp_type = {"indexerror":["indexerror-nd","indexerror-1d", "indexerror"],
                         "nameerror":["module not found", "variable not found", "function not found ", "class not found"],
                         "attributeerror":["attributeerror"],
                         "other":["zerodivisionerror","incompleteparseerror","systemerror","systemexit","assertionerror","requesterror","syntaxerror","constraint violation (database)","indentationerror", "executablenotfound"],
                         "valueerror":["valueerror - data value violation", "valueerror - row count mismatch", "valueerror - feature name mismatch", "tensor shape mismatch","valueerror", "valueerror - data range mismatch"],
                         "ioerror":["fileexistserror","unsupported file type (read file)", "file permission", "oserror","filenotfounderror","jsondecodeerror"],
                         "unknown":["unknown"],
                         "API argument violation":["wrong arguments to API"],
                         "out of resources":["out of space (disk)", "out of memory (OOM)"],
                         "keyerror":["keyerror","notfounderror"],
                         "runtimeerror":["initialization error (call mul-times, wrong order)","runtimeerror"],
                         "typeerror":["typeerror", "typeerror-notsubscriptable", "typeerror-op", "typeerror-notiterable", "typeerror-unhashable","typeerror-notcallable", "cast exception", "unsupported broadcast"],
                         "environment": ["environment setup", "importerror"]}

# summarize manual label config (categories) end-------------------------

cluster_size_cuttoff_k = 10
cluster_size_cuttoff_g = 100

# err_lib_percent_cutoff_k = 0.5
# err_lib_count_cutoff_k = 100

# err_lib_percent_cutoff_g = 0.5
# err_lib_count_cutoff_g = 400

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
