import os

import h5py
import numpy as np
from Enum.enum_types import Regression_method as rm
from sklearn.tree import DecisionTreeRegressor
import Outputs.data_reader as dr
import joblib


# return: 1 dt object, 1 mae result
# indiv_idx:5, reg_method:svm, max_depth:10

import random
# TODO: MEGÍRNI!!!! CREATE és READ is!!!!
def get_feature_names_to_use(chromosomes):
    with open("../DATA/feature_list.joblib", 'rb') as f:
        fl = joblib.load(f)
    if chromosomes:
        feature_names=[fl[idx] for idx, v in enumerate(chromosomes) if v]
    else:
        feature_names=random.choices(fl,k=2)


    # 0: stft 0
    # 1: stft 1
    # 2: stft 2
    # 3: mfcc 0
    # 4: mfcc 1
    # 5: mfcc 1
    return feature_names


#TODO: mapping megírása, mapping információk kírásának betétele FE folyamatba!!
def get_individual_data(hive_id:int,num_gen:int, individual_idx:int,dummy_ind:bool=False)-> np.ndarray:

    if dummy_ind:
        feature_names = get_feature_names_to_use(None)
    else:
        _population = dr.read_h5_file("/DATA/LOG/",f"population_{num_gen}.h5","population")
        _chromosomes = _population[individual_idx]
        feature_names = get_feature_names_to_use(_chromosomes)

    print("featurenames",feature_names)

    d=get_data(hive_id,feature_names)

    # turn chromosomes to concrete numbers a.k.a. MAPPING


def get_label_data(hive_id:int):
    pass

"""
def read_data_from_file_name(filename):
    date_and_time=
    pass
"""

preprocessed_data_path="../DATA/FE/"
import datetime
import pandas as pd

data_per_date={}
timestamp_per_date={}

def get_data(hive_id:int, feature_names:list[str]) -> list:
    print('FNS: ', feature_names)
    for file in os.listdir(preprocessed_data_path):
        if (file.startswith( str(hive_id))):
            for feature_name in feature_names:
                fnp=feature_name.split('-')
                fn=fnp[0]
                f=fnp[1]
                if feature_name.replace("-","_") in file:
                    dts=file[3:22]
                    date_and_time=datetime.datetime.strptime(dts, "%Y_%m_%d_%H_%M_%S")


                    print('====== File:',file)
                    print("D:",date_and_time)
                    h5_data=h5py.File(preprocessed_data_path + file)
                    print("K:",h5_data.keys())
                    print("F:",fn)

                    feature_df=pd.DataFrame(np.array(h5_data[fn]))
                    feature_df.index = feature_df.index.map(lambda x: f"{fn}-{f}-{x}")
                    feature_df=feature_df.T
                    if date_and_time not in data_per_date.keys():
                        #TODO ez üresnek tűnik...
                        '''
                        timestamp_file = "_".join([str(hive_id), dts, 'timestamps.h5'])
                        timestamp_data = h5py.File(preprocessed_data_path + timestamp_file)
                        timestamp_df = pd.DataFrame(np.array(timestamp_data))
                        timestamp_df=timestamp_df.T
                        '''
                        diff=datetime.timedelta(minutes=10)/feature_df.shape[0]
                        dates= {"ts": [date_and_time + x*diff for x in range(feature_df.shape[0])]}

                        timestamp_df=pd.DataFrame(dates)
                        data_per_date[date_and_time]=pd.concat([timestamp_df,feature_df],axis=1)
                    else:
                        data_per_date[date_and_time]=pd.concat([data_per_date[date_and_time],feature_df],axis=1)
    data = pd.concat(data_per_date.values())
    data.to_csv('test_concat_data.csv')
                    #df = pd.DataFrame(np.array(h5py.File(preprocessed_data_path+file)['variable_1']))

    # file = h5py.File(f"hive_id_{hive_id}_gen_{num_gen}_indiv_{i}.h5", 'r')

#TODO: megírni - Peti!
def eval_individual(num_gen:int, indiv_index:int, max_depth:int,
                    hive_id:int):
    data=get_individual_data(hive_id,num_gen,indiv_index,dummy_ind=True)

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

    """
    # build individual data to X
    X = get_individual_data(num_gen,indiv_index)
    indexes = [index for index, value in enumerate(X) if value == 1]

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
    """
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    # Add arguments
    parser.add_argument('--num_gen', type=int, help='Generation no',default=1)
    parser.add_argument('--indiv_index', type=int, help='Index of individual',default=0)
    parser.add_argument('--max_depth', type=int, help='max depth of individual',default=4)
    parser.add_argument('--hive_id', type=int, help='hive id',default=26)
    #parser.add_argument('--greet', action='store_true', help='Include a greeting')

    # Parse the arguments
    # TODO reg_metod ??
    args = parser.parse_args()
    eval_individual(args.num_gen,args.indiv_index,  args.max_depth,args.hive_id)