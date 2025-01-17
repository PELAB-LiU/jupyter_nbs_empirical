from pathlib import Path
import tqdm
import shutil
import os
from typing import List

from wmutils.file import iterate_through_files_in_nested_folders, safe_makedirs

def grab(source: Path, nb_list: Path, dest: Path) -> float:
    with open(nb_list, 'r', encoding='utf-8') as nbs_file:
        nbs = set(entry.strip() for entry in nbs_file if entry.strip() != '')
    
    move_count = 0

    for file in tqdm.tqdm(iterate_through_files_in_nested_folders(source, 10_000)):
        file = Path(file)
        if not file.name in nbs:
            continue
        
        dest_file = dest.joinpath(file.name)

        safe_makedirs(dest_file.parent)

        if not os.path.exists(dest_file):
            shutil.copyfile(file, dest_file)

        move_count += 1


    print(f'Moved {move_count}/{len(nbs)} notebooks from "{source}" to "{dest}".')
    
    if len(nbs) > 0:
        ratio = move_count / len(nbs)
        return ratio
    return 1


def take_sample(sample_base_path: Path, kaggle_data_folders: List[Path], gh_data_folders: List[Path]):
    sample_1_gh_nbs = sample_base_path.joinpath(r'gh_sample_list.txt')
    sample_1_kg_nbs = sample_base_path.joinpath(r'kg_sample_list.txt')
    sample_1_gh_dest = sample_base_path.joinpath(r'github notebooks')
    sample_1_kg_dest = sample_base_path.joinpath(r'kaggle notebooks')

    for kg_source_folder in kaggle_data_folders:
        rat = grab(kg_source_folder, sample_1_kg_nbs, sample_1_kg_dest)

    for gh_source_folder in gh_data_folders:
        rat += grab(gh_source_folder, sample_1_gh_nbs, sample_1_gh_dest)

    rat /= 2

    print(f'Found {rat * 100:.2f}% of the NBs in {sample_base_path}.')
