import pandas as pd
import utils.config as config
import utils.util as util
import numpy as np

def labeled_data_config_clean(df_mlerr_labels, df_mlerr_label_config, save_config_path = None):
    # labels in the config but are not used in manual labeling
    print("\n++++Labels in the config but are not used in manual labeling++++")
    for label_key in df_mlerr_label_config:
        print(label_key)
        print([i for i in df_mlerr_label_config[label_key].unique() if i not in df_mlerr_labels[label_key].unique()])

        updated_config = [i for i in df_mlerr_label_config[label_key].unique() if i in df_mlerr_labels[label_key].unique()]
        for _ in range(len(df_mlerr_label_config)-len(updated_config)):
            updated_config.append(np.nan)
        for _ in range(len(updated_config)-len(df_mlerr_label_config)):
            df_mlerr_label_config.loc[len(df_mlerr_label_config)] = pd.Series(dtype='float64')

        df_mlerr_label_config[label_key] = updated_config
        
    # labels used in manual labeling but not in config !!should not happen!!
    print("\n++++[Should not happen]Labels used in manual labeling but not in config++++")
    for label_key in df_mlerr_label_config:
        print(label_key)
        print([i for i in df_mlerr_labels[label_key].unique() if i not in df_mlerr_label_config[label_key].unique()])
        
    # need to take care of n/a in df_mlerr_label_config (label_refined_exp_type)
    # it means the same as the enames
    if sum(df_mlerr_labels['label_refined_exp_type'] == 'n/a')>0:
        df_mlerr_labels['label_refined_exp_type'] = np.where(df_mlerr_labels['label_refined_exp_type'] == 'n/a', 
                                                             df_mlerr_labels['ename'], 
                                                             df_mlerr_labels['label_refined_exp_type'])
        real_refined_exp_types = set(df_mlerr_label_config.label_refined_exp_type).union(set(df_mlerr_labels['label_refined_exp_type'].unique()))
        real_refined_exp_types.remove("n/a")
        real_refined_exp_types.remove(np.nan)
        real_refined_exp_types=list(real_refined_exp_types)
        for _ in range(len(df_mlerr_label_config)-len(real_refined_exp_types)):
            real_refined_exp_types.append(np.nan)
        for _ in range(len(real_refined_exp_types)-len(df_mlerr_label_config)):
            df_mlerr_label_config.loc[len(df_mlerr_label_config)] = pd.Series(dtype='float64')

        df_mlerr_label_config['label_refined_exp_type'] = list(real_refined_exp_types)
    
    assert(df_mlerr_labels.label_refined_exp_type.nunique()==df_mlerr_label_config.label_refined_exp_type.nunique())
    if save_config_path:
        df_mlerr_label_config.to_excel(save_config_path, index=False, engine='xlsxwriter')
        
        
def labeled_data_config_sum(df_mlerr_labels, df_mlerr_label_config, save_data_path = None, save_config_path = None):
    # config
    df_mlerr_label_config_sum = df_mlerr_label_config.copy()
    for summed_label_name in config.summed_label_names:
        option_dict = { v: k for k, l in getattr(config, summed_label_name).items() for v in l }
        config_tobe_mapped = df_mlerr_label_config_sum[summed_label_name]
        list_mapped = set(["" if (pd.isnull(item)|(item=="")) else option_dict[item] for item in config_tobe_mapped])
        list_mapped = list(filter(None, list_mapped))
        list_mapped.extend([""]*(len(config_tobe_mapped)-len(list_mapped)))
        df_mlerr_label_config_sum[summed_label_name] = list_mapped
    if save_data_path:
        df_mlerr_label_config_sum.to_excel(save_data_path, index=False, engine='xlsxwriter')

    # data
    df_mlerr_labels_sum = df_mlerr_labels.copy()
    for summed_label_name in config.summed_label_names:
        option_dict = { v: k for k, l in getattr(config, summed_label_name).items() for v in l }
        df_mlerr_labels_sum[summed_label_name] = ["" if (pd.isnull(item)|(item=="")) else option_dict[item] for item in df_mlerr_labels_sum[summed_label_name]]
    if save_config_path:
        df_mlerr_labels_sum.to_excel(save_config_path, index=False, engine='xlsxwriter')