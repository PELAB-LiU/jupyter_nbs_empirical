from pathlib import Path

path_drive_kaggle = r"D:\Data_NBs\Kaggle"

yirans_path_default = Path(r"C:\Users\yirwa29\Downloads\data_jupyter_nbs_empirical")
willems_path_default = Path(".").absolute().joinpath("data")

path_default = yirans_path_default
path_plot_default = Path(r"C:\Users\yirwa29\OneDrive - Linköpings universitet\SAProject\Paper2\figures_v2")

willems_path_drive = path_default.joinpath("harddrive")
willems_path_drive_2 = Path(r"D:\Data_NBs")
path_drive = willems_path_drive

print(f"{path_default=}")

path_github_error_process = path_default.joinpath("GitHub")
path_kaggle_error_process = path_default.joinpath("Kaggle")

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

dl_lib_names = [
    "tensorflow",
    "torch",
    "keras",
    "torchvision",
    "transformers",
]

builtin_exps_excluded = ["keyboardinterrupt"]

NB_SOURCE = {
    "kaggle": 1,
    "github": 2
}
# summarize manual label config (categories)------------------------------
summed_label_names = ["label_root_cause", "label_ML_pipeline", "label_if_ML_bug", "label_refined_exp_type"]

# sum1 ---------------------------------------------------------------------------------------------------------------------
label_root_cause = {"data confusion":["misunderstanding of data structure"],
                    "ML model confusion": ["misunderstanding of ML model"],
                    "API misuse":["misunderstanding of APIs","misunderstanding of libraries", "invalid argument"],
                    "NB specific":["nb specific - execution order","nb specific - previous cell error","nb specific - need execute future cells"],
                    "implementation error":["did not import", "undefined variable", "undefined function", "typo", "wrong implementation"],
                   "insufficient resource":["insufficient resources"],
                   "unknown": ["unknown"],
                   "environment setting":["module not installed", "change of environment", "file/path not found or exist", "library versions incompatible", "settings(permission, environment)", "external control (window closed)"],
                    "library cause":["API change", "error inside library"]}
# sum2 ---------------------------------------------------------------------------------------------------------------------
label_ML_pipeline = {"environment setup":["environment setup (module not found, file/path not found)"],
                    "data preparation":["data preparation/preprocessing"],
                    "data visualization": ["data visualization"],
                    "model construction":["model construction (include compilation and visualization/summary)"],
                    "training":["training/validation (grid search)"],
                    "evaluation/prediction":["evaluation/inference (history plot, metric visualization)"]}
# sum3 ---------------------------------------------------------------------------------------------------------------------
label_if_ML_bug = {"ML bug":["ML/data science library related (ML imports, error raised by library)"],
                  "python bug":["general code error"]}
# sum4 ---------------------------------------------------------------------------------------------------------------------
label_refined_exp_type = {"variable not found":["variable not found"], # name error
                              "invalid argument":["wrong arguments to API"],
                              "module not found":["module not found"], # name error
                              "attribute error":["attributeerror"],
                              "key error":["keyerror","notfounderror"],
                              "tensor shape mismatch": ["tensor shape mismatch"], # value error
                              "data value violation": ["valueerror - data value violation"], # value error
                              "name error":["function not found ", "class not found","nameerror"], # name error
                              "value error":["cast exception", "valueerror - data range mismatch", "valueerror"], # value error
                              "index error":["indexerror-nd","indexerror-1d"],
                              "OOM":["out of memory (OOM)"],
                              "type error":["typeerror", "typeerror-notcallable", "typeerror-op", "typeerror-notsubscriptable", "typeerror-notiterable", "typeerror-unhashable"],
                              "request error" : ["requesterror"],
                              "unsupported broadcast": ["unsupported broadcast"], # value error
                              "runtime error":["runtimeerror"],
                              "model initialization error": ["initialization error (call mul-times, wrong order)"],
                              "environment error": ["importerror", "environment setup"],
                              "feature name mismatch": ["valueerror - feature name mismatch"], # value error
                         "other":["syntaxerror","indentationerror", "zerodivisionerror", "assertionerror", "systemerror", "executablenotfound", "out of space (disk)", "unknown"],
                         "io error":["filenotfounderror", "unsupported file type (read file)", "file permission", "fileexistserror", "jsondecodeerror", "incompleteparseerror"]}

# label_refined_exp_type_old = {"index":["indexerror-nd","indexerror-1d", "indexerror"],
#                          "name":["module not found", "variable not found", "function not found ", "class not found","nameerror"],
#                          "attribute":["attributeerror"], 
#                           "assertion":["assertionerror"], 
#                           "request" : ["requesterror"], 
#                           "syntax":["syntaxerror","indentationerror"],
#                          "other":["zerodivisionerror","incompleteparseerror","systemerror","systemexit","constraint violation (database)", "executablenotfound", "incompleteparseerror", "illegalmoveerror", "qiskiterror", "nosuchwindowexception"],
#                          "value":["valueerror - data value violation", "valueerror - feature name mismatch", "tensor shape mismatch","valueerror", "valueerror - data range mismatch", "cast exception", "unsupported broadcast"],
#                          "io":["fileexistserror","unsupported file type (read file)", "file permission", "filenotfounderror","jsondecodeerror"],
#                          "unknown":["unknown"],
#                          "API arg":["wrong arguments to API"],
#                          "resource":["out of space (disk)", "out of memory (OOM)"],
#                          "key":["keyerror","notfounderror"],
#                          "runtime":["initialization error (call mul-times, wrong order)","runtimeerror"],
#                          "type":["typeerror", "typeerror-notsubscriptable", "typeerror-op", "typeerror-notiterable", "typeerror-unhashable","typeerror-notcallable"],
#                          "environment": ["environment setup", "importerror"]}

# summarize manual label config (categories) end-------------------------

# summarize manual label abbr setting------------------------------
exp2abbr = {"variable not found":"VNF","invalid argument":"IARG","module not found":"MODULE","attribute error":"ATTR",
            "key error":"KEY","tensor shape mismatch":"TSHAPE","data value violation":"DVIOL", 
            "name error": "NAME", "value error":"VALUE", "index error":"INDEX", "OOM":"OOM", "type error":"TYPE",
            "request error":"RERR","unsupported broadcast":"BCAST","runtime error":"RUNTIME",
            "model initialization error": "INIT","environment error":"ENV","feature name mismatch": "FNAME",
            "other": "OTHER","io error": "IO"}
rc2abbr = {"API misuse":"API","NB specific":"NB","implementation error":"IMPL","environment setting":"ENV","data confusion":"DATA",
           "unknown":"UNK","insufficient resource":"RSC", "ML model confusion": "MODEL", "library cause":"LIB"}
mlpp2abbr = {'environment setup':"ENVS", 'data preparation':"DATAP", 'data visualization':"DATAV",
             'model construction':"MCONS", 'training':"TRAIN", 'evaluation/prediction':"EVAL"}
order_mlpp = ["environment setup", "data preparation", "data visualization", "model construction", "training", "evaluation/prediction"]

# summarize manual label abbr setting end------------------------------

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
