import numpy as np
import h5py
import os

def read_h5_file(path:str, data_filename:str, dataset_name:str):
    h5file_data = h5py.File(path + "/" + data_filename, 'r')
    data_set = np.asarray(h5file_data.get(dataset_name))
    return data_set
