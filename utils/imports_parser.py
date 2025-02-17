import ast
from collections import namedtuple
import os
from iteration_utilities import unique_everseen
import json
import re
import sys

# This is simply parse and match each line of code
def get_imports_nbs_static(path_tar, get_imports_func):
    res = []
    n_total = 0
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                n_total += 1
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
                                imports = set(imports)
                                # for data_analysis_[additional-manual_labels_libraries]
                                if "keras" in imports:
                                    imports.add("tensorflow/keras")
                                    imports.remove("keras")
                                if "tensorflow" in imports:
                                    imports.add("tensorflow/keras")
                                    imports.remove("tensorflow")
                                # end
                                res.append({"fname":f, "imports":imports})
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
    print("Successfully parsed {0}/{1} notebook files, failed {2} ones.".format(len(res), n_total, n_total-len(res)))
    return res

#  from/import xxx (import xxx) as xxx
def get_imports_line_outermost(line):
    line = line.strip()
    pattern = re.compile(r"^\s*(?:from|import)\s+(\w+(?:\s,\s*\w+)*)")
    pattern_next = re.compile(r"^(\w+(?:\s,\s*\w+)*)")
    imp_res = []
    imps = re.findall(pattern, line) # get the first match
    if len(imps)>0:
        lines = line.split('#')[0].split(';')[0].split(",")
        if len(lines) > 0: # more than 1 imports
            imps_first = re.findall(pattern, lines[0].strip()) # get the first match
            if line.startswith("from"):
                return imps_first
            imp_res.extend(imps_first)
            for i in range(1,len(lines),1):
                #print(lines[i])
                imps_next = re.findall(pattern_next, lines[i].strip())
                #print(imps_next)
                imp_res.extend(imps_next)
    return imp_res

# from xxx import xxx as xxx
def get_imports_line_all(line):
    line = line.strip()
    pattern = re.compile(r"(?m)^(?:from[ ]+(\S+)[ ]+)?import[ ]+(\S+)(?:[ ]+as[ ]+(\S+))?[ ]*")
    pattern_next = re.compile(r"(?m)^(\S+)(?:[ ]+as[ ]+(\S+))?[ ]*")
    imp_res = []
    imps = re.findall(pattern, line) # if it is an import statement
    if len(imps)>0:
        lines = line.split('#')[0].split(';')[0].split(",")
        if len(lines)>0: # more than 1 imports
            imps_first = re.findall(pattern, lines[0].strip()) # get the first match
            imp_res.extend(imps_first)
            imps_next_from=[imps_first[0][0]]
            for i in range(1,len(lines),1):
                #print(lines[i])
                imps_next = re.findall(pattern_next, lines[i].strip())
                #print(imps_next)
                imp_res.extend([tuple(imps_next_from+list(imp_next)) for imp_next in imps_next])
    return imp_res

# imported library name corresponding to alias in the code
# will be import names if no alias has been defined
def get_lib_alias(imps):
    res = []
    for imp in imps:
        res_item = []
        if len(imp[0])>0:
            res_item.append(imp[0].split(".")[0])
        else:
            res_item.append(imp[1].split(".")[0])
        res_item.append(imp[2]) if len(imp[2])>0 else res_item.append(imp[1])
        res.append(res_item)
    return res

# get imports where the imports are actually used in the code
# The code has to compile by ast parser for this to work
def get_imports_nbs_outermost_ast(path_tar, get_imports_func):
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
                    imports = get_imports_func(path_py)
                    if imports is None:
                        n_failed_ast += 1
                        print("Failed to ast parse", f_py)
                    else:
                        res.append({"fname":f, "imports":imports})
                else:
                    n_failed_py += 1
                    print("Failed to convert to python file with exit code", exit_code_convert_py)
    print("total number of nbs processed:", n_total)
    print("number of nbs that failed to convert to py:", n_failed_py)
    print("number of pys that failed to parse to ast:", n_failed_ast)
    return res

def get_imports_pys_outermost_ast(path_tar, get_imports_func):
    res = []
    n_failed_ast = 0
    n_total = 0
    for path, subdirs, files in os.walk(path_tar):
        for f_py in files:
            if f_py.endswith(".py"):
                n_total += 1
                imports = get_imports_func(f"{path}/{f_py}")
                if imports is None:
                    n_failed_ast += 1
                    print("Failed to ast parse", f_py)
                else:
                    res.append({"fname":f_py.replace(".py",".ipynb"), "imports":imports})
    print("total number of pys processed:", n_total)
    print("number of pys that failed to parse to ast:", n_failed_ast)
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