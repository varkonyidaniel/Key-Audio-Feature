import numpy as np
from Enum.enum_types import Regression_method as rm
from sklearn.tree import DecisionTreeRegressor
import Outputs.data_reader as dr


# return: 1 dt object, 1 mae result
# indiv_idx:5, reg_method:svm, max_depth:10


# TODO: MEGÍRNI!!!! CREATE és READ is!!!!
def Read_Mapping():
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





# TODO: Megírni!!!!!



    # én írom a regresszorokat

    # regression eredményét a /DATA/LOG könytárba kiírni egy logfile-ba!
    # generation_12_indiv_5.h5
    # file = h5py.File(f"gen_{num_gen}_indiv_{i}.h5", 'r')

def eval_individual(num_gen:int, individual_idx:int, reg_method:str, max_depth:int):

    # result object serialize - joblib dump
    # read back ser. object get numbers..
    # első létrehozza a  "tmp_gen_*_indiv_*.h5"
    # utolsó átnevezi "gen_*_indiv_*.h5"-ra


    # build individual data to X
    X = get_individual_data(num_gen,individual_idx)

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

