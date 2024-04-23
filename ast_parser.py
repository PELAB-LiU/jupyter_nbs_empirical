import ast
import os
import json
import pickle
import pandas as pd
import re
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

def parse_nbs_to_asts(path_tar, path_ast, parse_func, parser=None):
    res = []
    n_failed_py = 0
    n_failed_ast = 0
    n_total = 0
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                n_total += 1
                f_py = f.replace(".ipynb",".py")
                path_py = f"{path}/{f_py}"
                if os.path.isfile(path_py):
                    exit_code_convert_py = 0
                else:
                    exit_code_convert_py = os.system('jupyter nbconvert --to python {:s}'.format(f"{path}/{f}"))
                if exit_code_convert_py == 0:
                    export_f = f.replace(".ipynb","_ast.pickle")
                    export_path = f"{path_ast}/{export_f}"
                    is_parsed = parse_py_ast(path_py, export_path, parse_func, parser)
                    if not is_parsed:
                        n_failed_ast += 1
                        print("Failed to ast parse", f_py)
                    else:
                        res.append({"fname":f.replace(".py",".ipynb"), "fast":export_f})
                else:
                    n_failed_py += 1
                    print("Failed to convert to python file with exit code", exit_code_convert_py)
    print("total number of nbs processed:", n_total)
    print("number of nbs that failed to convert to py:", n_failed_py)
    print("number of pys that failed to parse to ast:", n_failed_ast)
    return pd.DataFrame(res, columns=['fname', 'fast'])

def parse_pys_to_asts(path_tar, path_ast, parse_func, parser=None):
    res = []
    n_failed_ast = 0
    n_total = 0
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".py"):
                n_total += 1
                path_py = f"{path}/{f}"
                export_f = f.replace(".py","_ast.pickle")
                export_path = f"{path_ast}/{export_f}"
                is_parsed = parse_py_ast(path_py, export_path, parse_func, parser)
                if not is_parsed:
                    n_failed_ast += 1
                    print("Failed to ast parse", f)
                else:
                    res.append({"fname":f.replace(".py",".ipynb"), "fast":export_f})
    print("total number of nbs processed:", n_total)
    print("number of pys that failed to parse to ast:", n_failed_ast)
    return pd.DataFrame(res, columns=['fname', 'fast'])

def parse_py_ast(path_py, export_path, parse_func, parser=None):
    if path_py.endswith(".py"):
        with open(path_py, encoding='utf-8') as fh:
            return parse_func(fh.read(), export_path, parser)
    return False

def parse_code_ast(code, export_path, parser=None):
    try:
        root = ast.parse(code)
    except:
        return False
    
    with open(export_path, 'wb') as f:
        pickle.dump(root, f)
        
    return True

def py23_setup_parser():
    PY_LANGUAGE = Language(tspython.language(), "python")
    parser = Parser()
    parser.set_language(PY_LANGUAGE)
    return parser

def if_error_nodes(root_node):
    for n in root_node.children:
        if n.type == "ERROR" or n.is_missing:
            return True
        if n.has_error:
            # there is an error inside this node
            return True
    return False

# support both python 2 and python 3
def parse_code_ast_py23(code, export_path, parser):
    tree = parser.parse(bytes(code, "utf8",))
    if if_error_nodes(tree.root_node): # if error when parsing
        return False
#     with open(export_path, 'wb') as f: # TypeError: cannot pickle 'tree_sitter.Node' object
#         pickle.dump(tree.root_node, f)
    return True

def check_python_version(nb_meta_data):
    res = "-1"
    if nb_meta_data.get("kernelspec") and nb_meta_data["kernelspec"].get("name"):
        match = re.search('python(\d+)', nb_meta_data["kernelspec"]["name"].lower())
        if match:
            res = match.group(1).strip()
    if nb_meta_data.get("kernelspec") and nb_meta_data["kernelspec"].get("language_info") and nb_meta_data["kernelspec"]["language_info"].get("version"):
        res = nb_meta_data["kernelspec"]["language_info"]["version"].strip()
    if res:
        return int(res.split(".")[0])

def nb_python_version_exact(path_tar):
    total_notebook = 0
    n_pyv = 0
    n_decoding_error = 0
    res = []
    python_version_unknown = -1
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                total_notebook += 1
                try:
                    file = open(f"{path}/{f}", "r", encoding="utf-8")
                except FileNotFoundError:
                    print(f"File {f} not found. Aborting")
                    sys.exit(1)
                except OSError:
                    print(f"OS error occurred trying to open {f}")
                    sys.exit(1)
                except Exception as err:
                    print(f"Unexpected error opening {f} is",repr(err))
                    sys.exit(1) 
                else:
                    try:
                        file_as_json = json.loads(file.read())
                        if not file_as_json.get("metadata"):
                            res_tmp = (f, python_version_unknown)
                            print('No metadata in the notebook', f)
                        else:
                            res_tmp = (f, check_python_version(file_as_json['metadata']))
                        res.append(res_tmp)
                        if res_tmp[1] != python_version_unknown:
                            n_pyv += 1
                    except json.decoder.JSONDecodeError:
                        n_decoding_error += 1
                        print("decoding error: {}".format(f))
    print("Total number of notebooks processed: {}".format(total_notebook))
    print("Total number of notebooks have python version info from metadata: {}".format(n_pyv))
    print("Total number of notebooks that cannot be decoded: {}".format(n_decoding_error))
    return pd.DataFrame(res, columns=['fname', 'python_version'])
    