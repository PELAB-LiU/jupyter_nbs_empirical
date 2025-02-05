import math
import numpy as np
from scipy.stats import chi2_contingency, barnard_exact
import pickle

# summarize the statistics
def statistics_labels(df_mlerr_label_config, df_mlerr_labels):
    cluster_res = {}
    for label_key in df_mlerr_label_config:
        cluster_res[label_key] = {}

    df_mlerr_labeled_clusters = df_mlerr_labels.cluster_id.unique()
    for cluster_id in df_mlerr_labeled_clusters:
        cluster_errors = df_mlerr_labels[df_mlerr_labels.cluster_id == cluster_id]
        for label_key in df_mlerr_label_config:
            for label_key_option in df_mlerr_label_config[label_key].dropna():
                cluster_label_key_option_size = sum(cluster_errors[label_key]==label_key_option)
                if label_key_option not in cluster_res[label_key]:
                    cluster_res[label_key][label_key_option] = {}
                cluster_res[label_key][label_key_option][cluster_id] = (cluster_label_key_option_size, len(cluster_errors))
    return cluster_res

def save_statistics(save_path, cluster_res):
    with open(save_path, 'wb') as handle:
        pickle.dump(cluster_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_statistics(load_path):
    with open(load_path, 'rb') as handle:
        cluster_res_load = pickle.load(handle)
    return cluster_res_load

def map_element_weights(x):
    if isinstance(x, tuple):
        return x[0]
    else:
        return x

# calculated/summed over all clusters, sorted
def cal_weighted_sum(df_mlerr_label_config, cluster_res):
    cluster_res_weighted_sum = {}
    for label_key in df_mlerr_label_config:
        cluster_res_weight = cluster_res[label_key].map(map_element_weights)
        tmp = cluster_res_weight.iloc[:, 1:].sum().sort_values(ascending=False)
        cluster_res_weighted_sum[label_key] = {k: v for k, v in tmp.items() if k}
    return cluster_res_weighted_sum


def chisquare_test(dict1, dict2, label_key="", min_dp = 5, res_all = False):
    list1, list2 = [],[]
    for k in dict1.keys():
        list1.append(dict1[k])
        list2.append(dict2[k])
    table_label_key_gk = np.array([list1,list2]).T
    table_label_key_gk = np.delete(table_label_key_gk, np.where(table_label_key_gk < min_dp)[0], axis=0)
    print("\n"+label_key)
    print("Removed number of features due to few(<5) data points:",len(dict1)-table_label_key_gk.shape[0],len(dict2)-table_label_key_gk.shape[0])
    chi_res = chi2_contingency(table_label_key_gk)
    pvalue_chisquare = chi_res.pvalue
    print("Chi2ContingencyResult: pvalue is ", pvalue_chisquare, 
          "(Not statistically different)" if pvalue_chisquare > 0.05 else "(Statistically different)")
    if res_all:
        return chi_res
    return pvalue_chisquare


# for 2x2
def barnard_test(dict1, dict2, label_key=""):
    table_label_key_gk = np.array(([v for k, v in dict1.items()],[v for k, v in dict2.items()])).T
    print("\n"+label_key)
#     display(table_label_key_gk)
    pvalue_be = barnard_exact(table_label_key_gk).pvalue
    print("Barnard exact result: pvalue is ", pvalue_be, 
          "(Not statistically different)" if pvalue_be > 0.05 else "(Statistically different)")
    return pvalue_be