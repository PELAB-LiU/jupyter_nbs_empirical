import sys
try:
    import re
    import os
    import json
    import shutil
    import pandas as pd
    import ast
    from guesslang import Guess
except ImportError:
    pass

def parse_traceback(str_traceback):
    ansi_escape = re.compile(r'\\x1b\[[0-9;]*m') #re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', str_traceback)

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

def get_evalue_ignored_from_traceback(row):
    txt_traceback = row["traceback"]
    target_err = str(row["ename"]).strip().lower()
    list_tbs = list_traceback(txt_traceback)
    if list_tbs and len(list_tbs) > 0:
        list_last = list_tbs[-1].split(":")
        #print(list_last[0].strip().lower())
        if list_last and list_last[0].strip().lower()==target_err:
            if len(list_last) <= 1:
                return ""
            res = list_last[1].strip()
            if len(res)>0:
                return res
    return None
            
# def get_evalue_ignored_from_traceback(row, n_cha_cutoff = 150, min_alphanum_rate = 0.5):
#     keyword = str(row['ename'])+':'
#     parts = row['traceback'].rpartition(keyword)
#     if len(parts[2].strip()) > 0:
#         # initial
#         if len(parts[0]) <= 0 or not parts[0][-1] in(['\'', '\"']):
#             rep = ''
#         else:
#             rep = parts[0][-1]
#         value_can = parts[2].replace('\\n','').replace(rep+']','') # remove '] or "]
#         if len(value_can.strip()) > 0:
#             value_res = value_can.partition(rep+', ')[0].strip()
#             if len(value_res) > 10:
#                 alphanum_rate = len([c for c in value_res if c.isalnum()]) / len(value_res)
#                 if alphanum_rate >= min_alphanum_rate:
#                     return value_res[:n_cha_cutoff]
#     return None

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

def extract_lib(txt_traceback):
    txt_traceback = txt_traceback.replace("\\\\", "/")
    #print(txt_traceback)
    pattern = re.compile(r'(\/.*?\/)((?:[^\/]|\\\/)+?)(?:(?<!\\)\s|$)')
    pattern2 = re.compile(r'(\-packages/+.*?\/)')
    matches = re.findall(pattern, txt_traceback)
    #print(matches)
    libs = []
    
    for mat in matches:
        if "-packages/" in mat[0]:
            matc = re.search(pattern2, mat[0])
            if matc:
                libs.append(matc.group(1)[10:-1])
            else:
                matc_1 = mat[1].split(".py")
                if len(matc_1)>0:
                    libs.append(matc_1[0])
    libs = set(libs)
    if len(libs)>0:
        return ",".join(libs)
    else:
        return None

def simple_lib_parser(libs_tar):
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
    
def is_contain_error_output(file_name, file_as_json):
    cells = file_as_json["cells"]
    res = 0
    res_err = []
    for i in range(0, len(cells), 1):
        if cells[i]["cell_type"] == "code" and cells[i]["outputs"]:
            for output in cells[i]["outputs"]:
                if output["output_type"]=="error":
#                     if output["ename"] in ['FileNotFoundError', 'KeyboardInterrupt']: #ignore error types
#                         continue
#                     print(output["ename"])
                    res = 1
                    res_err.append((file_name, output["ename"], output["evalue"], output["traceback"]))
    return res, res_err

def filter_notebooks_with_errors(path_tar, path_des = None, path_des2 = None, is_resave = True):
    total_notebook = 0
#     total_notebook2 = 0
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
    #                         #special attention errors
    #                         for t in res_err:
    #                             if t[0] in ['ValueError']: 
    #                                 total_notebook2 += 1
    #                                 if is_resave: shutil.copyfile(f"{path}/{f}", f"{path_des2}/{f}")
    #                                 break
                        # save err infomation
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
    return pd.DataFrame(res_errs, columns=['fname', 'ename', 'evalue', 'traceback'])

def nb_to_py(path_tar):
    for path, subdirs, files in os.walk(path_tar):
        for f in files:
            if f.endswith(".ipynb"):
                nb_file = open(f"{path}/{f}", "r", encoding="utf-8")
                j = json.load(nb_file)
                on = f.replace('.ipynb',"")+'.py'
                of = open(f"{path}/pys/{on}", 'w', encoding="utf-8") #output.py
                if j["nbformat"] >=4:
                    for i,cell in enumerate(j["cells"]):
                        if cell["cell_type"] == "code":
                            for line in cell["source"]:
                                of.write(line)
                        of.write('\n\n')
                else:
                    for i,cell in enumerate(j["worksheets"][0]["cells"]):
                        if cell["cell_type"] == "code":
                            for line in cell["input"]:
                                of.write(line)
                        of.write('\n\n')
                of.close()