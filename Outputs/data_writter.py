import numpy as np
import h5py
import os, sys


def write_data_to_h5(directory: str, filename: str, ds_label:str, data: np.ndarray):
    file_name = f"{filename}.h5"
    hf = h5py.File(f"{directory}/{file_name}",'a')
    hf.create_dataset(f"{ds_label}", data=data)
    hf.close()
    print(f"Appending: {ds_label} to file: {file_name} ... DONE!", flush=True)

def check_directories(parent_dir:str, source_dir:str, target_dir:str):

    if not os.path.isdir(parent_dir+source_dir):
        print(f'Missing source directory: {parent_dir+source_dir}')
        sys.exit(1)

    if not os.path.isdir(parent_dir+target_dir):
        os.mkdir(parent_dir+target_dir)
        print(f'Missing target directory. Creating: {target_dir}')
