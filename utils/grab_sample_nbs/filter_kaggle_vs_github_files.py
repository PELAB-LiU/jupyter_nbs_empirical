import pandas as pd
from pathlib import Path

def do(base_folder: Path, sheet):
    df = pd.read_csv(sheet, header=0) #, encoding='latin-1'

    kaggle_files = df[df['nb_source'] == 1]
    gh_files = df[df['nb_source'] == 2]


    def out(df: pd.DataFrame, out_path: Path):
        fname = df['fname']
        fs = "\n".join(fname)
        with open(out_path, 'w+', encoding='utf-8') as out_file:
            out_file.write(fs)


    out(kaggle_files, base_folder.joinpath("kg_sample_list.txt"))
    out(gh_files, base_folder.joinpath("gh_sample_list.txt"))
