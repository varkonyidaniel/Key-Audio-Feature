import numpy as np
import h5py
import os

def read_h5_file(path:str, data_filename:str, dataset_name:str):
    h5file_data = h5py.File(path + "/" + data_filename, 'r')
    data_set = np.asarray(h5file_data.get(dataset_name))
    return data_set

def get_file_name():
    pass

def read_joblib_file(hive_id: int, num_gen:int, indiv_index:int):

    f"hive_{hive_id}_gen_{num_gen}_indiv_{indiv_index}.joblib"
