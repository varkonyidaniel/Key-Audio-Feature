import os, argparse

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
def get_feature_group_chromosomes(chromosomes):
    """
    with open("../DATA/feature_list.joblib", 'rb') as f:
        fl = joblib.load(f)
    if chromosomes:
        feature_names=[fl[idx] for idx, v in enumerate(chromosomes) if v]
    else:
        feature_names=random.choices(fl,k=2)
    """
    with open("../DATA/feature_list_with_dim.joblib", 'rb') as f:
        fl = joblib.load(f)
        mapping={name:dim[0] for name,dim in fl.items()}

    if not chromosomes:
        all_features_count=sum(mapping.values())
        chromosomes = np.random.choice([0, 1], size=(all_features_count,), p=[9./10, 1./10])
    start=0
    chromosomes_per_feature_group={}
    for k,v in mapping.items():
        chromosomes_per_feature_group[k]=chromosomes[start:start+v]
        start=start+v

    # 0: stft 0
    # 1: stft 1
    # 2: stft 2
    # 3: mfcc 0
    # 4: mfcc 1
    # 5: mfcc 1
    return chromosomes_per_feature_group

import pandas as pd
#TODO: mapping megírása, mapping információk kírásának betétele FE folyamatba!!
def get_individual_data(hive_id:int,num_gen:int, individual_idx:int,dummy_ind:bool=False)-> pd.DataFrame:

    if dummy_ind:
        per_feature_group_chromosomes = get_feature_group_chromosomes(None)
    else:
        _population = dr.read_h5_file("/DATA/LOG/",f"population_{num_gen}.h5","population")
        _chromosomes = _population[individual_idx]
        per_feature_group_chromosomes = get_feature_group_chromosomes(_chromosomes)

    print("feature names",per_feature_group_chromosomes.keys())

    d=get_data(hive_id,per_feature_group_chromosomes)
    return d

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

data_per_date={}
timestamp_per_date={}

from pandas.api.types import is_complex_dtype

#TODO: detection time beolvasása preproc file szerint
detection_times={22:datetime.datetime(2020,7,30),30:datetime.datetime(2020,7,30),37:datetime.datetime(2020,7,30)}


def get_data(hive_id:int, per_feature_group_chromosomes:dict[str,list[int]]) -> list:
    #print('FNS: ', feature_names)
    for file in os.listdir(preprocessed_data_path):
        if (file.startswith( str(hive_id))):
            for feature_name,chromosomes in per_feature_group_chromosomes.items():
                print(feature_name,'->',chromosomes)
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
                    to_keep_feture_names=[f"{fn}-{f}-{index}" for index, value in enumerate(per_feature_group_chromosomes[feature_name]) if value == 1]
                    feature_df=feature_df.T
                    feature_df=feature_df[to_keep_feture_names]
                    to_convert=[]
                    if f=='dani':
                        print(feature_df.dtypes)
                    for c in feature_df.columns:
                        if is_complex_dtype(feature_df[c]):
                            to_convert.append(c)


                    feature_df[to_convert]=(feature_df[to_convert]**2).astype(float)
                    if date_and_time not in data_per_date.keys():

                        timestamp_file = "_".join([str(hive_id), dts, 'timestamps.h5'])
                        timestamp_data = h5py.File(preprocessed_data_path + timestamp_file)["timelabels"]
                        timestamp_df = pd.DataFrame({'ts':np.array(timestamp_data)})
                        timestamp_df['ts']=pd.to_datetime(timestamp_df['ts'].map(lambda x: x.decode("utf-8")),format="%Y-%m-%d %H-%M-%S-%f")
                        timestamp_df['time_to_detection']=detection_times[hive_id]-timestamp_df['ts']
                        timestamp_df['seconds_to_detection']=timestamp_df['time_to_detection'].dt.total_seconds()
                        timestamp_df.drop(columns=['ts','time_to_detection'],inplace=True)

                        '''
                        diff=datetime.timedelta(minutes=10)/feature_df.shape[0]
                        dates= {"ts": [date_and_time + x*  diff for x in range(feature_df.shape[0])]}

                        timestamp_df=pd.DataFrame(dates)
                        '''
                        data_per_date[date_and_time]=pd.concat([timestamp_df,feature_df],axis=1)
                    else:
                        data_per_date[date_and_time]=pd.concat([data_per_date[date_and_time],feature_df],axis=1)
    data = pd.concat(data_per_date.values()).reset_index(drop=True)
    data.to_csv('../DATA/test_concat_data.csv')
    return data
                    #df = pd.DataFrame(np.array(h5py.File(preprocessed_data_path+file)['variable_1']))

    # file = h5py.File(f"hive_id_{hive_id}_gen_{num_gen}_indiv_{i}.h5", 'r')

#TODO: megírni - Peti!

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


def train_DTR(X_train,y_train):
    param_grid = {
        'max_depth': [10, 20],
    }
    dtree_reg = DecisionTreeRegressor(random_state=42)  # Initialize a decision tree regressor
    grid_search = GridSearchCV(estimator=dtree_reg, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("DTR best params",best_params)
    mse=grid_search.best_score_
    return {"model_class":"DTR","MSE":mse,"params":best_params},model


    #predictions = model.predict(X_test)

def train_SVR(X_train,y_train):
    param_grid = {
        'kernel': ["poly", "rbf"],
        'C':[0.5]
    }
    svr_reg = SVR()  # Initialize a decision tree regressor

    grid_search = GridSearchCV(estimator=svr_reg, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    #model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    mse=grid_search.best_score_
    print('SVR best params:',best_params)
    print('SVR best mse:',mse)
    return {"model_class":"DTR","MSE":mse,"params":best_params}
    #predictions = model.predict(X_test)

from sklearn.model_selection import train_test_split


# gen_{num_gen}_indiv_{i}_hive_{hive}


def eval_individual(num_gen:int, indiv_index:int,hive_id:int):

    if hive_id not in detection_times.keys():
        print(f"No Foul brood in hive {str(hive_id)}")
    data=get_individual_data(hive_id,num_gen,indiv_index,dummy_ind=True)
    pd_y=data['seconds_to_detection']
    pd_X=data.drop(columns=['seconds_to_detection'])

    svr_stats=train_SVR(pd_X,pd_y)
    dtr_stats,dt_model=train_DTR(pd_X,pd_y)
    # TODO: lin reg is

    dir="../DATA/LOG/"
    fn=f"hive_{hive_id}_gen_{num_gen}_indiv_{indiv_index}.joblib"

    # Check if directory exists
    if not os.path.exists(dir):
    # Create directory
        os.makedirs(dir)

    res={'SVR':svr_stats,'DTR':dtr_stats,'DT_model':dt_model}
    joblib.dump(res,f"{dir}/{fn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    # Add arguments
    parser.add_argument('--num_gen', type=int, help='Generation no',default=1)
    parser.add_argument('--indiv_index', type=int, help='Index of individual',default=0)
    parser.add_argument('--hive_id', type=int, help='hive id',default=22)

    # Parse the arguments
    # TODO reg_metod ??
    args = parser.parse_args()
    eval_individual(args.num_gen,args.indiv_index,args.hive_id)
