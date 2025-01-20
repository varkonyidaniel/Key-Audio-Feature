import numpy as np
from Enum.enum_types import Regression_method as rm
from sklearn.tree import DecisionTreeRegressor
import Outputs.data_reader as dr


# return: 1 dt object, 1 mae result
# indiv_idx:5, reg_method:svm, max_depth:10


# TODO: MEGÍRNI!!!! CREATE és READ is!!!!
def Read_Mapping():

    [0,1,0,0,0,1,0]
    # 0: stft 0
    # 1: stft 1
    # 2: stft 2
    # 3: mfcc 0
    # 4: mfcc 1
    # 5: mfcc 1
    return ['stft 0','stft 1','stft 2', 'mfcc 0', 'mfcc 1', 'mfcc 2']


#TODO: mapping megírása, mapping információk kírásának betétele FE folyamatba!!
def get_individual_data(num_gen:int, individual_idx:int)-> np.ndarray:
    _population = dr.read_h5_file("/DATA/LOG/",f"population_{num_gen}.h5","population")
    _chromosomes = _population[individual_idx]

    # turn chromosomes to concrete numbers a.k.a. MAPPING


def get_label_data(hive_id:int):
    pass






    # file = h5py.File(f"hive_id_{hive_id}_gen_{num_gen}_indiv_{i}.h5", 'r')

#TODO: megírni - Peti!
def eval_individual(num_gen:int, indiv_index:int, reg_method:str, max_depth:int,
                    hive_id:int):

    # regression eredményét a /DATA/LOG könytárba kiírni egy logfile-ba!
    # generation_12_indiv_5.h5
    # file = h5py.File(f"hive_{hive_id}_gen_{num_gen}_indiv_{i}.h5", 'r')


    #dt 1 hive id 1-re
    #dt 2 hive id 2-re
    #dt 3 hive id 3-re


    # def write_data_to_h5(directory: str, filename: str, ds_label:str, data: np.ndarray):
    #     file_name = f"{filename}.h5"
    #     hf = h5py.File(f"{directory}/{file_name}",'a')
    #     hf.create_dataset(f"{ds_label}", data=data)
    #     hf.close()
    #     print(f"Appending: {ds_label} to file: {file_name} ... DONE!", flush=True)

    # output
    # file name = ...
    # ds_label = svr_1p0_linear_metric_name
    # data = result of regression: svr c=1.0, kernel = linear, metric = mae




    # result object serialize - joblib dump
    # read back ser. object get numbers..
    # első létrehozza a  "tmp_gen_*_indiv_*.h5"
    # utolsó átnevezi "gen_*_indiv_*.h5"-ra, key = "lr"
    # !!! párhuzamosságnál nem tudom, hogy melyik az utsó,
    # szekvenciálisan kell a regresszorokat futtatni 1 node-on!!! MEGFONTOLNI!!!!
    # generáció


    # build individual data to X
    X = get_individual_data(num_gen,indiv_index)

    # TODO: innen tovább írni!!!!
    # build timelabels to Y
    y=[]



    indiv = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    reg_m = "dt"
    regressor = DecisionTreeRegressor(max_depth)
    #regressor.

    #dt = decision tree...


    if reg_method == rm.DT:
        return dt, 0.0
    else:
        return None, 1.1

    # generation_12_indiv_5.h5 file-ban menteni az eredményt
    # tartalma: key = svm_ data = 1.24

