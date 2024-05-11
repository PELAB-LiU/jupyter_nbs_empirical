import sys
import re
import os
import json
import shutil
import pandas as pd
import ast
import importlib
import inspect
import pkgutil
import pickle
import config
import builtins
import matplotlib.pyplot as plt
from pathlib import Path
from exception_id import generate_uuid_for_nb_exception
try:
    from guesslang import Guess
except ImportError:
    pass

def extract_libs_from_exp_mllib(row):
    str_exp_mllib = row["exp_mllib"]
    if (not str_exp_mllib) or pd.isna(str_exp_mllib):
        return None
    try:
        tmp = eval(str_exp_mllib)
        res = set()
        for i in range(len(tmp)):
            res.add(tmp[i]["library"])
        return res
    except:
        print("cannot parse to list, eid:",row["eid"])
        return None

def get_exp_mllib(path_exp_libs, df_err):
    res = {eid:None for eid in df_err.eid}
    with open(path_exp_libs, "r", encoding="utf-8") as output_file:
        data = output_file.readlines()
        
    for line in data:
        j_data = json.loads(line)
        ml_links = j_data["exc_to_lib_links"]
        if len(ml_links) > 0:
            for ml_link in ml_links:
                eid = ml_link["exception_id:"]
                if eid in res:
                    if len(ml_link["libraries"]) > 0:
                        res[eid] = ml_link["libraries"]
#                 else:
#                     print("Error id not exist in error dataset:", Path(j_data["notebook"]).name, eid)
    return res

def parse_traceback(str_traceback):
    """Parses the traceback to remove all ansii escape characters."""
    ansi_escape = re.compile(r'\\x1b\[[0-9;]*m') #re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', str_traceback)

def parse_traceback_2(str_traceback):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    result = ansi_escape.sub('', str_traceback)
    return result

def list_traceback(txt_traceback):
    try:
        tb_list = ast.literal_eval(txt_traceback)
        return tb_list
    except:
        print("exception when listing traceback")
        return None

def print_traceback(txt_traceback):
    tb_list = list_traceback(txt_traceback)
    if tb_list:
        for i in tb_list:
            print(i)

# def get_evalue_ignored_from_traceback(row):
#     txt_traceback = row["traceback"]
#     target_err = str(row["ename"]).strip().lower()
#     list_tbs = list_traceback(txt_traceback)
#     if list_tbs and len(list_tbs) > 0:
#         list_last = list_tbs[-1].split(":")
#         #print(list_last[0].strip().lower())
#         if list_last and list_last[0].strip().lower()==target_err:
#             if len(list_last) <= 1:
#                 return ""
#             res = list_last[1].strip()
#             if len(res)>0:
#                 return res
#     return None

def get_evalue_ignored_from_traceback(row):
    txt_traceback = row["traceback"]
    target_err = str(row["ename"]).strip().lower()
    list_tbs = list_traceback(txt_traceback)
    if list_tbs and len(list_tbs) > 0:
        for j in range(len(list_tbs)-1,-1,-1):
            list_last = list_tbs[j].split("\n")
            for i in range(len(list_last)-1,-1,-1):
                list_last_i = list_last[i].split(":")
                if list_last_i and list_last_i[0].strip().lower()==target_err:
                    if len(list_last_i)>1:
                        res = list_last_i[1].strip()
                        return res
                    else:
                        return ""
    return None

def process_noisy_evalues(row, max_len=400):
    evalue_lines = str(row.evalue).split("\n")
    if len(evalue_lines) > 1: # only handling the noisy ones
        evalue_line = evalue_lines[0]
        # when first line is empty or "in user code:" is in the first line -> from the last line extract evalues starting with ename:
        if len(evalue_line.strip()) <= 0 or ("in user code:" in evalue_line.lower()):
            for i in range(len(evalue_lines)-1, 0, -1):
                evalue_line_last = evalue_lines[i]
                ename = str(row.ename)+":"
                if len(evalue_line_last.strip())>0 and (ename in evalue_line_last.lower()):
                    res = re.sub(ename,"", evalue_line_last, flags=re.IGNORECASE) ## remove error name
                    #print(row.evalue,"\n---->>>",res.strip(),"\n\n")
                    #print("---------------------------------------")
                    return res.strip()[:max_len]
        else:
            #print(row.evalue,"\n---->>>",evalue_line.strip(),"\n\n")
            #print("---------------------------------------")
            return evalue_line.strip()[:max_len] # # take the first line
    return None

def nb_language_exact(path_tar, path_to, lan_list):
    total_notebook = 0
    n_decoding_error = 0
    res = []
    lan_unknown = "unknown"
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                try:
                    file = open(f"{path}/{f}", "r", encoding="utf-8")
                except FileNotFoundError:
                    print(f"File {f} not found.  Aborting")
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
                            res_tmp = (f, lan_unknown)
                            print('No metadata in the notebook', f)
                        else:
                            lan_found = lan_unknown
                            if file_as_json['metadata'].get("kernelspec") and file_as_json['metadata']["kernelspec"].get("language"):
                                lan1 = simple_language_parser(file_as_json['metadata']["kernelspec"]["language"], lan_list)
                                if lan1 in lan_list:
                                    lan_found = lan1
                            if lan_found == lan_unknown and file_as_json['metadata'].get("kernelspec") and file_as_json['metadata']["kernelspec"].get("name"):
                                lan1 = simple_language_parser(file_as_json['metadata']["kernelspec"]["name"], lan_list)
                                if lan1 in lan_list:
                                    lan_found = lan1
                            if lan_found == lan_unknown and file_as_json['metadata'].get("language_info") and file_as_json['metadata']["language_info"].get("name"):
                                lan1 = simple_language_parser(file_as_json['metadata']["language_info"]["name"], lan_list)
                                if lan1 in lan_list:
                                    lan_found = lan1
                            res_tmp = (f, lan_found)
                        res.append(res_tmp)
                        if res_tmp[1] != lan_unknown:
                            total_notebook += 1
                        else:
                            shutil.copyfile(f"{path}/{f}", f"{path_to}/{f}") # copy out for further handling
                            print('Copying notebook..', f)
                    except json.decoder.JSONDecodeError:
                        n_decoding_error += 1
                        print("decoding error: {}".format(f))

    print("Total number of notebooks have language info from metadata: {}".format(total_notebook))
    print("Total number of notebooks that cannot be decoded: {}".format(n_decoding_error))
    return pd.DataFrame(res, columns=['fname', 'language'])

def py_language_detection(path_nolan, conf = 0.5):
    guess = Guess()
    res = []
    for path, subdirs, files in os.walk(path_nolan):
        for f in files:
            if f.endswith(".py"):
                txt_file = open(f"{path}/{f}", "r", encoding="utf-8")
                # Guess the language from code
                language_probs = guess.probabilities(txt_file.read())
                language = "unknown"
                if language_probs[0][1] >= conf:
                    language = language_probs[0][0]
                res.append((f[:-3]+".ipynb",language))
    return pd.DataFrame(res, columns=['fname', 'language'])

def simple_language_parser(lan_tar, lan_list):
    lan_tar = lan_tar.lower()
    if 'python' in lan_tar:
        return 'python'
    if 'julia' in lan_tar:
        return 'julia'
    if 'scala' in lan_tar:
        return 'scala'
    if 'csharp' in lan_tar:
        return 'c#'
    if 'c++' in lan_tar:
        return 'c++'
    if 'sql' in lan_tar:
        return 'sql'
    if 'bash' in lan_tar:
        return 'shell'
    for lan_c in lan_list:
        if lan_c in lan_tar:
            return lan_c
    return lan_tar

def simple_lib_parser(libs_tar):
    """Returns the underlying library of the provided path."""
    if pd.isna(libs_tar):
        return None
    lib_tar = libs_tar.lower().split(",")[-1]
    if 'pandas' in lib_tar:
        return 'pandas'
    if 'torch' in lib_tar:
        return 'torch'
    if ('tensorflow' in lib_tar) or ('keras' in lib_tar):
        return 'tensorflow'
    if 'sklearn' in lib_tar:
        return 'sklearn'
    if 'matplotlib' in lib_tar:
        return 'matplotlib'
    if 'numpy' in lib_tar:
        return 'numpy'
    return lib_tar
    
def lib_alias_isML(lib_alias):
    if lib_alias:
        try:
            lib_alias = eval(lib_alias)
            for imp in lib_alias:
                
                if imp[0] and (simple_lib_parser(imp[0]) in config.top_lib_names):
                    return True
        except:
            return False
    return False

# def extract_lib(txt_traceback):
#     txt_traceback = txt_traceback.replace("\\\\", "/")
#     #print(txt_traceback)
#     pattern = re.compile(r'(\/.*?\/)((?:[^\/]|\\\/)+?)(?:(?<!\\)\s|$)')
#     pattern2 = re.compile(r'(\-packages/+.*?\/)')
#     matches = re.findall(pattern, txt_traceback)
#     #print(matches)
#     libs = []
    
#     for mat in matches:
#         if "-packages/" in mat[0]:
#             matc = re.search(pattern2, mat[0])
#             if matc:
#                 libs.append(matc.group(1)[10:-1])
#             else:
#                 matc_1 = mat[1].split(".py")
#                 if len(matc_1)>0:
#                     libs.append(matc_1[0])
#     libs = set(libs)
#     if len(libs)>0:
#         return ",".join(libs)
#     else:
#         return None
    
# def extract_lib_2(row, df_imports, lib_names, lib_classes_dict):
#     pattern_crash_line = re.compile('(--->\s*\d+\s*(.*))')
#     pattern_obj = re.compile(r"'([^']+)'(?=\s+object)")
#     df_import_tar = df_imports[df_imports.fname==row["fname"]]
#     lib_alias = df_import_tar.lib_alias.iloc[0] if len(df_import_tar)>0 else []
#     for lib_name in lib_names:
#         # rule 2.1
#         if isinstance(row["evalue"],str) and (lib_name in row["evalue"]):
#             return lib_name
#         # rule 4
#         if isinstance(row["evalue"],str):
#             obj_mat = re.findall(pattern_obj, row["evalue"])
#             if len(obj_mat)>0:
#                 for k, v in lib_classes_dict.items():
#                     if obj_mat[0] in v:
#                         return k
#         # get alias for this lib_name
#         if len(lib_alias)<=0:
#             continue
#         alias = []
#         for t in eval(lib_alias):
#             if t[0]==lib_name:
#                 alias.append(t[1])
#         tb_list = list_traceback(row["traceback"]) # util.
#         # rule 2.2
#         if tb_list:
#             for i in range(len(tb_list)-1, -1, -1):
#                 lines = tb_list[i].replace("\\n","\n").split("\n")
#                 for line in lines:
#                     crash_code_mat = re.findall(pattern_crash_line, line)
#                     if len(crash_code_mat)>0 and len(crash_code_mat[0])>1:
#                         crash_code = crash_code_mat[0][1]
#                         #print(crash_code)
#                         if "=" in crash_code:
#                             parts = crash_code.split("=", 1)
#                             if len(parts)>1 and len(alias)>0:
#                                 if any(alia in parts[1].strip() for alia in alias):
#                                     return lib_name 
#     return None

def is_contain_error_output(file_name, file_as_json):
    res = 0
    res_err = []
    for cell_index, cell in enumerate(file_as_json["cells"]):
        if cell["cell_type"] == "code" and "outputs" in cell:
            for output in cell["outputs"]:
                if output["output_type"]=="error":
#                     if output["ename"] in ['FileNotFoundError', 'KeyboardInterrupt']: #ignore error types
#                         continue
#                     print(output["ename"])
                    res = 1
                    exception_id = generate_uuid_for_nb_exception(
                        file_name, cell_index, output["ename"])
                    res_err.append((file_name, exception_id, output["ename"], output["evalue"], output["traceback"]))
    return res, res_err

def filter_notebooks_with_errors(path_tar, is_resave = False, path_des = None):
    total_notebook = 0
#     total_notebook2 = 0
    cell_index = 0
    n_decoding_error = 0
    res_errs = []
    print("\nStarted filtering:")
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                try:
                    file = open(f"{path}/{f}", "r", encoding="utf-8")
                    try:
                        file_as_json = json.loads(file.read())
                        if not file_as_json.get("cells"):
                            raise Exception('No cells property in the notebook, probably a very old version')
                        res, res_err = is_contain_error_output(f, file_as_json)
                        if res == 1:
                            total_notebook += 1
                            if is_resave: shutil.copyfile(f"{path}/{f}", f"{path_des}/{f}")
                        if res > 0:
                            res_errs.extend(res_err)

                    except json.decoder.JSONDecodeError:
                        n_decoding_error += 1
                        print("decoding error: {}".format(f))
                    except Exception as error:
                        n_decoding_error += 1
                except Exception:
                    print("Errors occur when opening file {}".format(f))
                    n_decoding_error += 1
    print("\nTotal number of notebooks containing error: {}".format(total_notebook))
#     print("Total number of notebooks containing ValueError: {}".format(total_notebook2))
    print("Total number of notebooks that cannot be decoded: {}".format(n_decoding_error))
    return pd.DataFrame(res_errs, columns=['fname', 'eid', 'ename', 'evalue', 'traceback'])

def nb_to_py(path_tar):
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                try:
                    nb_file = open(f"{path}/{f}", "r", encoding="utf-8")
                    try:
                        j = json.load(nb_file)
                        on = f.replace('.ipynb',"")+'.py'
                        if ("nbformat" in j) and j["nbformat"] >=4:
                            of = open(f"{path}/pys/{on}", 'w', encoding="utf-8") #output.py
                            for i,cell in enumerate(j["cells"]):
                                if cell["cell_type"] == "code":
                                    for line in cell["source"]:
                                        of.write(line)
                                of.write('\n\n')
                            of.close()
                        elif ("worksheets" in j) and ("cells" in j["worksheets"][0]):
                            of = open(f"{path}/pys/{on}", 'w', encoding="utf-8") #output.py
                            for i,cell in enumerate(j["worksheets"][0]["cells"]):
                                if cell["cell_type"] == "code":
                                    for line in cell["input"]:
                                        of.write(line)
                                of.write('\n\n')
                            of.close()
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
                
                
def export_classes_from_modules(lib_names, export_path='lib_classes.pickle'):
    lib_classes = {} # types/objects
    for lib_name in lib_names:
        package_classes = []
        package = importlib.import_module(lib_name)

        prefix = package.__name__ + "."
        try:
            for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix, onerror=lambda x: None):
                # print(modname) # submodules
                try:
                    module = __import__(modname, fromlist="dummy")
                    package_classes.extend([m[0] for m in inspect.getmembers(module, inspect.isclass)]) # classes within submodules
                except:
                    print("import error for submodule: ", modname)
                    pass
        except:
            package_classes.extend([m[0] for m in inspect.getmembers(package, inspect.isclass)]) # classes within submodules
        lib_classes[lib_name]=package_classes

    with open(export_path, 'wb') as f:
        pickle.dump(lib_classes, f)

    return lib_classes
        
def combine_pickles(list_pickle_paths, export_path):
    res_dict = {}
    for p in list_pickle_paths:
        with open(p, 'rb') as f:
            lib_classes_dict = pickle.load(f)
        res_dict = {**res_dict, **lib_classes_dict}
    with open(export_path, 'wb') as f:
        pickle.dump(res_dict, f)
        
        
# df_err_lib_filtered is expected to have "lib_parsed" column which indicate the utermost crash library
# source is either github or kaggle
def select_builtin_exps(df_err_lib_filtered, source):
    n_selected_exps = 0
    df_err_lib_filtered["lib_parsed_pop"] = df_err_lib_filtered['lib_parsed'].apply(lambda i: i if i in config.top_lib_names else None)
    print("Selected exception types of {} that meet the criterions:\n".format(source))
    for builtin_exp in df_err_lib_filtered.ename.value_counts().index:
        # cutoff 1
        if builtin_exp in config.builtin_exps_excluded:
            continue
        df_err_builtin_exp = df_err_lib_filtered[df_err_lib_filtered["ename"]==builtin_exp]
        # cutoff 2
        if source == "github":
            cutoff_percent = config.err_lib_percent_cutoff_g
            cutoff_count = config.err_lib_count_cutoff_g
        else: # source="kaggle"
            cutoff_percent = config.err_lib_percent_cutoff_k
            cutoff_count = config.err_lib_count_cutoff_k
        libs_n = df_err_builtin_exp.lib_parsed_pop.value_counts()
        lib_percent = len(df_err_builtin_exp[~df_err_builtin_exp["lib_parsed_pop"].isnull()])/len(df_err_builtin_exp)
        if len(df_err_builtin_exp)*lib_percent <= cutoff_count:
            if lib_percent <= cutoff_percent:
                continue
        # select
        n_selected_exps += 1
        n_print = min(3, len(libs_n))
        print("{0}({1}), {2:.2%}({4}) are with the top libraries, top {3}:".format(builtin_exp, len(df_err_builtin_exp),lib_percent, n_print,int(len(df_err_builtin_exp)*lib_percent)))
        for i in range(n_print):
            print("\t{0:<12} {1:>12} samples".format(libs_n.index[i], libs_n.values[i]))
    print("\nIn total, {0} exception types are selected for further analysis".format(n_selected_exps))
    
    
def reload_module(tar_module):
    import importlib
    importlib.reload(tar_module)
#     import tar_module

def get_python_exception_names():
    list_of_exception_names = [
        name for name, value in builtins.__dict__.items() 
        if isinstance(value, type) and issubclass(value, BaseException)
    ]
    list_of_exception_names = [ele for ele in list_of_exception_names if ele not in config.exclude_base_exceptions]
    exception_list = [ele.lower() for ele in list_of_exception_names]
    return exception_list

def visulize_exps(df_err, title="#errors vs. exception types"):
    #plot
    df_err["ename_mapped"].groupby([df_err["ename_mapped"]]).count().sort_values(ascending=0).plot(kind="bar", figsize=(12,4),
                                                                                                   x='ename', y='ecount',
                                                                                                   title=title)
    plt.show()

def visulize_exps_mlnbs(df_mlerr):
    dict_err_MLlib_counts = {}
    dict_err_MLlib_percents = {}
    for builtin_exp in df_mlerr.ename.value_counts().index:
        df_err_builtin_exp = df_mlerr[df_mlerr["ename_mapped"]==builtin_exp]
        #  data prepare
        libs_count = len(df_err_builtin_exp[~df_err_builtin_exp["exp_mllib"].isnull()])
        if libs_count <= 0:
            continue
        lib_percent = libs_count/len(df_err_builtin_exp)
        dict_err_MLlib_counts[builtin_exp]=libs_count
        dict_err_MLlib_percents[builtin_exp]=lib_percent
    #plot
    df_err_MLlib_counts = pd.DataFrame.from_dict(dict_err_MLlib_counts.items())
    df_err_MLlib_counts.columns = ['ename', 'eMLlib_count']
    df_err_MLlib_counts = df_err_MLlib_counts.sort_values("eMLlib_count", ascending=0).reset_index(drop=True)
    df_err_MLlib_counts.plot(title="#MLlib-related errors vs. exception types", 
                             x='ename', y='eMLlib_count',
                             kind="bar", figsize=(12,4))
    plt.show()
    df_err_MLlib_percents = pd.DataFrame.from_dict(dict_err_MLlib_percents.items())
    df_err_MLlib_percents.columns = ['ename', 'eMLlib_percent']
    df_err_MLlib_percents = df_err_MLlib_percents.sort_values("eMLlib_percent", ascending=0).reset_index(drop=True)
    df_err_MLlib_percents.plot(title="%MLlib-related errors vs. exception types", 
                               x='ename', y='eMLlib_percent',
                               kind="bar", figsize=(12,4))
    plt.show()
#     return
#     return df_err_MLlib_counts, df_err_MLlib_percents

########################decrepcated###############################

def visulize_exps_MLlibs(df_err_lib_filtered):
    df_err_lib_filtered["lib_parsed_pop"] = df_err_lib_filtered['lib_parsed'].apply(lambda i: i if i in config.top_lib_names else None)
    dict_err_MLlib_counts = {}
    dict_err_MLlib_percents = {}
    for builtin_exp in df_err_lib_filtered.ename.value_counts().index:
        # cutoff 1
        if builtin_exp in config.builtin_exps_excluded:
            continue
        df_err_builtin_exp = df_err_lib_filtered[df_err_lib_filtered["ename"]==builtin_exp]
        #  data prepare
        libs_count = len(df_err_builtin_exp[~df_err_builtin_exp["lib_parsed_pop"].isnull()])
        if libs_count <= 0:
            continue
        lib_percent = libs_count/len(df_err_builtin_exp)
        dict_err_MLlib_counts[builtin_exp]=libs_count
        dict_err_MLlib_percents[builtin_exp]=lib_percent
    #plot
    df_err_MLlib_counts = pd.DataFrame.from_dict(dict_err_MLlib_counts.items())
    df_err_MLlib_counts.columns = ['ename', 'eMLlib_count']
    df_err_MLlib_counts = df_err_MLlib_counts.sort_values("eMLlib_count", ascending=0).reset_index(drop=True)
    df_err_MLlib_counts.plot(title="#MLlib-related errors vs. exception types", 
                             x='ename', y='eMLlib_count',
                             kind="bar", figsize=(12,4))
    plt.show()
    df_err_MLlib_percents = pd.DataFrame.from_dict(dict_err_MLlib_percents.items())
    df_err_MLlib_percents.columns = ['ename', 'eMLlib_percent']
    df_err_MLlib_percents = df_err_MLlib_percents.sort_values("eMLlib_percent", ascending=0).reset_index(drop=True)
    df_err_MLlib_percents.plot(title="%MLlib-related errors vs. exception types", 
                               x='ename', y='eMLlib_percent',
                               kind="bar", figsize=(12,4))
    plt.show()
    #return
    return df_err_MLlib_counts, df_err_MLlib_percents