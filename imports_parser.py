import ast
from collections import namedtuple
import os
from iteration_utilities import unique_everseen
import json
import re
import sys

# This is simply parse and match each line of code
def get_imports_nbs_static(path_tar, get_imports_func = get_imports_line_all):
    res = []
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                try:
                    with open(f"{path}/{f}", "r", encoding="utf-8") as data_file:
                        try:
                            j = json.load(data_file)
                            imports = list()
                            if ("nbformat" in j) and j["nbformat"] >=4:
                                for i,cell in enumerate(j["cells"]):
                                    if cell["cell_type"] == "code":
                                        if isinstance(cell["source"], list):
                                            for line in cell["source"]:
                                                imp = get_imports_func(line)
                                                if len(imp) > 0:
                                                    imports.extend(imp)
                                        else:
                                            for line in cell["source"].split("\n"):
                                                imp = get_imports_func(line)
                                                if len(imp) > 0:
                                                    imports.extend(imp)
                                res.append({"fname":f, "imports":set(imports)})
                            else:
                                print("wrong format of jupyter notebook", f)
                        except json.decoder.JSONDecodeError:
                            print("decoding error: {}".format(f))
                        except Exception as err:
                            print(f"Unexpected error converting to json {f}")
                            
                except FileNotFoundError:
                    print(f"File {f} not found.  Aborting")
                    sys.exit(1)
                except OSError:
                    print(f"OS error occurred trying to open {f}")
                    sys.exit(1)
                except Exception as err:
                    print(f"Unexpected error opening {f}")
                    sys.exit(1)
    return res

#  import xxx
def get_imports_line_outermost(line):
    pattern = re.compile(r"^\s*(?:from|import)\s+(\w+(?:\s*,\s*\w+)*)")
    return re.findall(pattern, line)

# from xxx import xxx as xxx
def get_imports_line_all(line):
    pattern = re.compile(r"(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)(?:[ ]+as[ ]+(\S+))?[ ]*")
    return re.findall(pattern, line)

# The code has to compile by ast parser for this to work
def get_imports_nbs_outermost_ast(path_tar):
    res = []
    n_failed = 0
    n_total = 0
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                n_total += 1
                if os.path.isfile(f"{path}/{f}"):
                    exit_code_convert_py = 0
                else:
                    exit_code_convert_py = os.system('jupyter nbconvert --to python {:s}'.format(f"{path}/{f}"))
                if exit_code_convert_py == 0:
                    f_py = f.replace(".ipynb",".py")
                    path_py = f"{path}/{f_py}"
                    with open(path_py, encoding='utf-8') as fh:
                        imports = get_imports_code_outermost(fh.read())
                        if imports is None:
                            n_failed += 1
                            #print("Failed to ast parse", f_py)
                        else:
                            res.append({"fname":f, "imports":imports})
                else:
                    n_failed += 1
                    print("Failed to convert to python file with exit code", exit_code_convert_py)
    print("total number of nbs processed:", n_total)
    print("number of nbs that failed:", n_failed)
    return res

def get_imports_py_outermost(path_py):
    if path_py.endswith(".py"):
        with open(path_py, encoding='utf-8') as fh:
            res = get_imports_code_outermost(fh.read())
            if res is None:
                # failed to parse ast
                pass
            return res
    return None

def get_imports_py_all(path_py):
    if path_py.endswith(".py"):
        with open(path_py, encoding='utf-8') as fh:
            res = get_imports_code_all(fh.read())
            if res is None:
                # failed to parse ast
                pass
            return res
    return None

# from xxx import xxx as xxx
def get_imports_code_all(code):  
    try:
        root = ast.parse(code)
        res = []
    except:
        return None
        
    for node in ast.walk(root):
        res_import = {"from":[], "import":[], "as":[]}
        if isinstance(node, ast.Import):
            pass
        elif isinstance(node, ast.ImportFrom):  
            res_import["from"] = node.module.split('.')
        else:
            continue

        for n in node.names:
            res_import["import"] = n.name.split('.')
            res_import["as"] = n.asname.split('.') if n.asname!=None else []
            res.append(res_import)
    return list(unique_everseen(res))

# only the module/package names as a list
def get_imports_code_outermost(code):
    try:
        root = ast.parse(code)
        res = []
    except:
        return None
        
    for node in ast.walk(root):
        res_import = {"from":[], "import":[], "as":[]}
        if isinstance(node, ast.Import):
            pass
        elif isinstance(node, ast.ImportFrom):  
            res.append(node.module.split('.')[0])
            continue
        else:
            continue
        for n in node.names:
            res.append(n.name.split('.')[0])
    return set(res)