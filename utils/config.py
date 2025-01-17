from pathlib import Path

path_drive_kaggle = r"D:\Data_NBs\Kaggle"

yirans_path_default = Path(r"C:\Users\yirwa29\Downloads\data_jupyter_nbs_empirical")
willems_path_default = Path(".").absolute().joinpath("data")

path_default = yirans_path_default

willems_path_drive = path_default.joinpath("harddrive")
willems_path_drive_2 = Path(r"D:\Data_NBs")
path_drive = willems_path_drive

print(f"{path_default=}")

path_github_error_process = path_default.joinpath("nbdata_g_error")
path_kaggle_error_process = path_default.joinpath("nbdata_k_error")

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

NB_SOURCE = {
    "kaggle": 1,
    "github": 2
}
# summarize manual label config (categories)------------------------------
summed_label_names = ["label_root_cause", "label_ML_pipeline", "label_if_ML_bug", "label_refined_exp_type"]

# sum1 ---------------------------------------------------------------------------------------------------------------------
label_root_cause = {"data":["misunderstanding of data structure", "misunderstanding of ML model"],
                    "API":["misunderstanding of APIs","misunderstanding of libraries", "invalid argument"],
                    "NB specific":["nb specific - execution order","nb specific - previous cell error","nb specific - need execute future cells"],
                    "implementation":["did not import", "undefined variable", "undefined function", "typo", "wrong implementation"],
                   "resources":["insufficient resources"],
                   "unknown": ["unknown"],
                   "environment":["module not installed", "change of environment", "file/path not found or exist", "library versions incompatible", "settings(permission, environment)", "external control (window closed)"],
                    "library":["API change", "error inside library"], 
                    "intentional":["intentional"]}
# sum2 ---------------------------------------------------------------------------------------------------------------------
label_ML_pipeline = {"environment setup":["environment setup (module not found, file/path not found)"],
                    "data preparation":["data preparation/preprocessing"],
                    "data visualization": ["data visualization"],
                    "model construction":["model construction (include compilation and visualization/summary)"],
                    "training":["training/validation (grid search)"],
                    "evaluation/prediction":["evaluation/inference (history plot, metric visualization)"],
                    "no ML pipeline":["not-applicable (sub-labels needed, e.g., tutorials, physics simulation, ..)","not applicable - tutorial notebook","not applicable - physics","not applicable - education","unknown"]}
# sum3 ---------------------------------------------------------------------------------------------------------------------
label_if_ML_bug = {"ML bug":["ML/data science library related (ML imports, error raised by library)"],
                  "python bug":["general code error"],
                  "unknown":["unknown"]}
# sum4 ---------------------------------------------------------------------------------------------------------------------
label_refined_exp_type = {"index":["indexerror-nd","indexerror-1d", "indexerror"],
                         "name":["module not found", "variable not found", "function not found ", "class not found","nameerror"],
                         "attribute":["attributeerror"], 
                          "assertion":["assertionerror"], 
                          "request" : ["requesterror"], 
                          "syntax":["syntaxerror","indentationerror"],
                         "other":["zerodivisionerror","incompleteparseerror","systemerror","systemexit","constraint violation (database)", "executablenotfound", "incompleteparseerror", "illegalmoveerror", "qiskiterror", "nosuchwindowexception"],
                         "value":["valueerror - data value violation", "valueerror - feature name mismatch", "tensor shape mismatch","valueerror", "valueerror - data range mismatch"],
                         "io":["fileexistserror","unsupported file type (read file)", "file permission", "filenotfounderror","jsondecodeerror"],
                         "unknown":["unknown"],
                         "API arg":["wrong arguments to API"],
                         "resource":["out of space (disk)", "out of memory (OOM)"],
                         "key":["keyerror","notfounderror"],
                         "runtime":["initialization error (call mul-times, wrong order)","runtimeerror"],
                         "type":["typeerror", "typeerror-notsubscriptable", "typeerror-op", "typeerror-notiterable", "typeerror-unhashable","typeerror-notcallable", "cast exception", "unsupported broadcast"],
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
