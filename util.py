import re
import os
import json
import shutil
import pandas as pd


def parse_traceback(str_traceback):
    ansi_escape = re.compile(r'\\x1b\[[0-9;]*m') #re.compile(r'(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', str_traceback)

def get_evalue_ignored_from_traceback(row, n_cha_cutoff = 150, min_alphanum_rate = 0.5):
    keyword = str(row['ename'])+':'
    parts = row['traceback'].rpartition(keyword)
    if len(parts[2].strip()) > 0:
        # initial
        if len(parts[0]) <= 0 or not parts[0][-1] in(['\'', '\"']):
            rep = ''
        else:
            rep = parts[0][-1]
        value_can = parts[2].replace('\\n','').replace(rep+']','') # remove '] or "]
        if len(value_can.strip()) > 0:
            value_res = value_can.partition(rep+', ')[0].strip()
            if len(value_res) > 10:
                alphanum_rate = len([c for c in value_res if c.isalnum()]) / len(value_res)
                if alphanum_rate >= min_alphanum_rate:
                    return value_res[:n_cha_cutoff]
    return None

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