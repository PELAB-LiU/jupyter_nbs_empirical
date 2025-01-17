import pandas as pd
from pathlib import Path
import shutil
import os
import tqdm

def do(base_folder:Path, sheet: Path):
    df = pd.read_csv(sheet, header=0) #, encoding='latin-1'

    git_path = base_folder.joinpath("github notebooks").absolute()
    kaggle_path = base_folder.joinpath('kaggle notebooks').absolute()

    output_path = base_folder.joinpath('clusters').absolute()

    deleted_files = set()

    for idx, row in tqdm.tqdm(df.iterrows()):
        fname = row['fname']
        is_kaggle = row['nb_source'] == 1
        cluster = row['cluster_id']

        real_output_path = output_path.joinpath(str(cluster)).joinpath(fname)
        real_output_dir = real_output_path.parent

        if not os.path.exists(real_output_dir):
            os.makedirs(real_output_dir)
        
        real_input_path = kaggle_path if is_kaggle else git_path
        real_input_path = real_input_path.joinpath(fname)

        deleted_files.add(real_input_path)

        shutil.copy(real_input_path, real_output_path)


    for file in deleted_files:
        os.remove(file)
