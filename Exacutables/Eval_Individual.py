#!/usr/bin/env python3

import os, argparse, sys

sys.path.append(os.path.abspath('../Enum'))

import h5py
import numpy as np
from sklearn.metrics import mean_squared_error

from Enum.enum_types import Regression_method as rm, Event_type
from sklearn.tree import DecisionTreeRegressor
import Outputs.data_reader as dr
import joblib
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# return: 1 dt object, 1 mae result
# indiv_idx:5, reg_method:svm, max_depth:10

import random


# TODO: MEGÍRNI!!!! CREATE és READ is!!!!
def get_feature_group_chromosomes(chromosomes, mapping):
    """
    with open("../DATA/feature_list.joblib", 'rb') as f:
        fl = joblib.load(f)
    if chromosomes:
        feature_names=[fl[idx] for idx, v in enumerate(chromosomes) if v]
    else:
        feature_names=random.choices(fl,k=2)
    """

    start = 0
    chromosomes_per_feature_group = {}
    for k, v in mapping.items():
        chromosomes_per_feature_group[k] = chromosomes[start:start + v]
        start = start + v

    # 0: stft 0
    # 1: stft 1
    # 2: stft 2
    # 3: mfcc 0
    # 4: mfcc 1
    # 5: mfcc 1
    return chromosomes_per_feature_group


import pandas as pd


def get_individual_data(tr_hive_ids: list[int],ts_hive_ids: list[int], num_gen: int, individual_idx: int) -> pd.DataFrame:
    def get_detection_time(hive_id):
        with open(f"../DATA/targets_{hive_id}.joblib", 'rb') as f:
            entry = joblib.load(f)
            # print(entry)
            detection_time = entry[str(Event_type.BROOD)]
            detection_time = datetime.datetime(detection_time.year, detection_time.month, detection_time.day)
            return detection_time
    with open("../DATA/feature_list_with_dim.joblib", 'rb') as f:
        fl = joblib.load(f)
        mapping = {name: dim[0] for name, dim in fl.items()}

    def get_per_feature_group_chromosomes_for_hive( mapping):
        #if individual_idx:
        _population = dr.read_h5_file("../DATA/LOG/", f"population_{num_gen}.h5", "population")
        _chromosomes = _population[individual_idx]
        #else:
        #    all_features_count = sum(mapping.values())
        #    _chromosomes = np.random.choice([0, 1], size=(all_features_count,), p=[9. / 10, 1. / 10])
        return get_feature_group_chromosomes(_chromosomes, mapping),_chromosomes

    per_feature_group_chromosomes,_chromosomes=get_per_feature_group_chromosomes_for_hive(mapping)
    # print("feature names",per_feature_group_chromosomes.keys())
    tr_ds_s=[get_data(hive_id, per_feature_group_chromosomes, get_detection_time(hive_id)) for hive_id in tr_hive_ids]
    ts_ds_s=[get_data(hive_id, per_feature_group_chromosomes, get_detection_time(hive_id)) for hive_id in ts_hive_ids]
    tr_d = pd.concat([get_data(hive_id, per_feature_group_chromosomes, get_detection_time(hive_id)).reset_index(drop=True) for hive_id in tr_hive_ids],ignore_index=True)
    ts_d = pd.concat([get_data(hive_id, per_feature_group_chromosomes, get_detection_time(hive_id)).reset_index(drop=True) for hive_id in ts_hive_ids],ignore_index=True)
    return tr_d,ts_d, _chromosomes

    # turn chromosomes to concrete numbers a.k.a. MAPPING


def get_label_data(hive_id: int):
    pass


"""
def read_data_from_file_name(filename):
    date_and_time=
    pass
"""

preprocessed_data_path = "../DATA/FE/"
import datetime

timestamp_per_date = {}

from pandas.api.types import is_complex_dtype

# TODO: detection time beolvasása preproc file szerint
detection_times = {22: datetime.datetime(2020, 7, 30), 30: datetime.datetime(2020, 7, 30),
                   37: datetime.datetime(2020, 7, 30)}


# def get_data(hive_id:int, per_feature_group_chromosomes:dict[str,list[int]],detection_time:datetime.datetime) -> list:
def get_data(hive_id: int, per_feature_group_chromosomes: dict, detection_time: datetime.datetime) -> list:
    # print('FNS: ', feature_names)
    data_per_date = {}

    for file in os.listdir(preprocessed_data_path):
        if (file.startswith(str(hive_id))):
            for feature_name, chromosomes in per_feature_group_chromosomes.items():
                # print(feature_name,'->',chromosomes)
                fnp = feature_name.split('-')
                fn = fnp[0]
                f = fnp[1]
                if feature_name.replace("-", "_") in file:
                    dts = file[3:22]
                    date_and_time = datetime.datetime.strptime(dts, "%Y_%m_%d_%H_%M_%S")

                    # print('====== File:',file)
                    # print("D:",date_and_time)
                    h5_data = h5py.File(preprocessed_data_path + file)
                    # print("K:",h5_data.keys())
                    # print("F:",fn)

                    feature_df = pd.DataFrame(np.array(h5_data[fn]))
                    feature_df.index = feature_df.index.map(lambda x: f"{fn}-{f}-{x}")
                    to_keep_feture_names = [f"{fn}-{f}-{index}" for index, value in
                                            enumerate(per_feature_group_chromosomes[feature_name]) if value == 1]
                    #print(to_keep_feture_names)
                    feature_df = feature_df.T
                    feature_df = feature_df[to_keep_feture_names]
                    to_convert = []
                    # if f=='dani':
                    #    print(feature_df.dtypes)
                    for c in feature_df.columns:
                        if is_complex_dtype(feature_df[c]):
                            to_convert.append(c)

                    feature_df[to_convert] = (feature_df[to_convert] ** 2).astype(float)
                    if date_and_time not in data_per_date.keys():

                        timestamp_file = "_".join([str(hive_id), dts, 'timestamps.h5'])
                        timestamp_data = h5py.File(preprocessed_data_path + timestamp_file)["timelabels"]
                        timestamp_df = pd.DataFrame({'ts': np.array(timestamp_data)})
                        timestamp_df['ts'] = pd.to_datetime(timestamp_df['ts'].map(lambda x: x.decode("utf-8")),
                                                            format="%Y-%m-%d %H-%M-%S-%f")
                        # timestamp_df['time_to_detection']=detection_times[hive_id]-timestamp_df['ts']
                        timestamp_df['time_to_detection'] = detection_time - timestamp_df['ts']
                        timestamp_df['seconds_to_detection'] = timestamp_df['time_to_detection'].dt.total_seconds()
                        timestamp_df.drop(columns=['ts', 'time_to_detection'], inplace=True)

                        '''
                        diff=datetime.timedelta(minutes=10)/feature_df.shape[0]
                        dates= {"ts": [date_and_time + x*  diff for x in range(feature_df.shape[0])]}

                        timestamp_df=pd.DataFrame(dates)
                        '''
                        data_per_date[date_and_time] = pd.concat([timestamp_df, feature_df], axis=1)

                    else:
                        data_per_date[date_and_time] = pd.concat([data_per_date[date_and_time], feature_df], axis=1)
                    #print(hive_id,date_and_time, data_per_date[date_and_time].columns)
    data = pd.concat(data_per_date.values()).reset_index(drop=True)
    data.to_csv('../DATA/test_concat_data.csv')
    return data
    # df = pd.DataFrame(np.array(h5py.File(preprocessed_data_path+file)['variable_1']))

    # file = h5py.File(f"hive_id_{hive_id}_gen_{num_gen}_indiv_{i}.h5", 'r')


# TODO: megírni - Peti!

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

from sklearn import linear_model
from scipy import stats
import numpy as np

# https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
'''
class LinearRegression_Stats(linear_model.LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """
    'fit_intercept=True, copy_X=True, n_jobs=None, positive=False'
    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        if not "copy_X" in kwargs:
            kwargs['copy_X'] = True
        if not "n_jobs" in kwargs:
            kwargs['n_jobs'] = None
        if not "positive" in kwargs:
            kwargs['positive'] = False
        super(LinearRegression, self)\
                .__init__(self, *args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        se = np.array([
            np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
                                                    for i in range(sse.shape[0])
                    ])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self
'''


def train_DTR(X_train, y_train,X_test,y_test):
    param_grid = {
        'max_depth': [10, 20],
    }
    dtree_reg = DecisionTreeRegressor(random_state=42)  # Initialize a decision tree regressor
    grid_search = GridSearchCV(estimator=dtree_reg, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("DTR best params", best_params)
    mse = -grid_search.best_score_
    y_hat=model.predict(X_test)
    test_mse=mean_squared_error(y_test, y_hat)
    print('DTR best mse:', mse)
    print("DTR TEST mse:", test_mse)
    return {"model_class": "DTR", "MSE": mse, "params": best_params, 'importance': model.feature_importances_.tolist(),"test_mse": test_mse}

    # predictions = model.predict(X_test)


def train_SVR(X_train, y_train,X_test,y_test):
    print("same Features",all([x==y for x,y in zip(X_train.columns,X_test.columns)]))
    param_grid = {
        'kernel': ["poly", "rbf"],
        'C': [0.5]
    }
    svr_reg = SVR()  # Initialize a decision tree regressor

    grid_search = GridSearchCV(estimator=svr_reg, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    # model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    mse = -grid_search.best_score_
    y_hat = grid_search.predict(X_test)
    test_mse = mean_squared_error(y_test, y_hat)
    print('SVR best params:', best_params)
    print('SVR best mse:', mse)
    print("SVR TEST mse:", test_mse)

    return {"model_class": "SVR", "MSE": mse, "params": best_params,"test_mse": test_mse}
    # predictions = model.predict(X_test)


from sklearn.model_selection import train_test_split

# gen_{num_gen}_indiv_{i}_hive_{hive}

from scipy import stats


def train_LR(X_train, y_train,X_test,y_test):
    # Calculate p-value using statsmodels
    X_with_const = sm.add_constant(X_train)  # Add a constant term for the intercept
    X_ts_with_const = sm.add_constant(X_test)  # Add a constant term for the intercept

    kf = KFold(n_splits=3, shuffle=True, random_state=1)

    # Initialize a list to store the mean squared errors
    mse_list = []
    p_list = []
    best_model = None
    best_mse = 0

    # Cross-validation loop
    for train_index, test_index in kf.split(X_with_const, y_train):
        train_X, test_X = X_with_const.iloc[train_index], X_with_const.iloc[test_index]
        train_y, test_y = y_train.iloc[train_index], y_train.iloc[test_index]

        # Fit the model
        model = sm.OLS(train_y, train_X).fit()

        # Predict on the test set
        predictions = model.predict(test_X)

        # Calculate and store the mean squared error
        mse = mean_squared_error(test_y, predictions)
        if mse > best_mse:
            best_model_params = model.params
            best_model=model
            best_mse=mse
        p_value = model.pvalues
        p_value.name = len(p_list)
        p_list.append(p_value)
        mse_list.append(mse)

    # Calculate the average MSE across all folds
    average_mse = np.mean(mse_list)
    average_p = pd.concat(p_list, axis=1).T.mean()
    # average_p=map(np.mean, zip(*p_list))
    print(f'Average MSE: {average_mse:.4f}')
    print(f'p-value: {average_p}')
    # p= importance_to_full_list(average_p,chromosome_list=chr)
    print("LR mse:", average_mse)
    print(X_test.head())

    y_hat = best_model.predict(X_ts_with_const)
    test_mse = mean_squared_error(y_test, y_hat)
    print("LR TEST mse:", test_mse)

    # print('SVR best params:', best_params)
    return {"model_class": "LR", "MSE": mse, "params": best_model_params, 'importance': average_p.to_list(),"test_mse": test_mse}


def importance_to_full_list(importance_list, chromosome_list):
    return [importance_list.pop(0) if i == 1 else 0 for i in chromosome_list]


# ezt a 4 paramétert kapod!!!! f"{num_gen} {idx_indiv} {tr_hive_ids} {ts_hive_ids}"
# hive_id-t le kell cserélni a fenti 2-re!!!

def eval_individual(num_gen: int, indiv_index: int, tr_hive_ids: list[int], ts_hive_ids: list[int]):
    print("Eval_Individual.py/eval_individual is running")
    '''
    obsolete
    for hive_id in hive_ids:
        if hive_id not in detection_times.keys():
            print(f"No Foul brood in hive {str(hive_id)}")
            hive_ids.remove(hive_id)
    '''
    tr_data,ts_data, chromosomes = get_individual_data(tr_hive_ids,ts_hive_ids, num_gen, indiv_index)
    tr_pd_y = tr_data['seconds_to_detection']
    tr_pd_X = tr_data.drop(columns=['seconds_to_detection'])
    ts_pd_y = ts_data['seconds_to_detection']
    ts_pd_X = ts_data.drop(columns=['seconds_to_detection'])
    print("NAN in y_train",np.any(np.isnan(tr_pd_y)))
    print("SVR train START")
    print("X_train shape",tr_pd_X.shape)
    print("X_test shape",ts_pd_X.shape)
    svr_stats = train_SVR(tr_pd_X, tr_pd_y,ts_pd_X, ts_pd_y)
    print("SVR train END")

    print("DTR train START")
    dtr_stats = train_DTR(tr_pd_X, tr_pd_y,ts_pd_X, ts_pd_y)
    print("DTR train END")

    print("LR train START")
    lr_stats = train_LR(tr_pd_X, tr_pd_y,ts_pd_X, ts_pd_y)
    print("LR train START")

    dtr_stats['all_data_importance'] = importance_to_full_list(dtr_stats['importance'], chromosome_list=chromosomes)
    lr_stats['all_data_importance'] = importance_to_full_list(lr_stats['importance'], chromosome_list=chromosomes)

    dir = "../DATA/LOG/"
    fn = f"gen_{num_gen}_indiv_{indiv_index}.joblib"

    # Check if directory exists
    if not os.path.exists(dir):
        # Create directory
        os.makedirs(dir)

    res = {'SVR': svr_stats, 'DTR': dtr_stats, 'LR': lr_stats}
    print("result file creation START")
    joblib.dump(res, f"{dir}/{fn}")
    print("result file creation END")


if __name__ == "__main__":
    print(sys.argv)
    num_gen = int(sys.argv[1])
    indiv_idx = int(sys.argv[2])
    tr_hive_ids = [int(ids) for ids in sys.argv[3].replace('[','').replace(']','').split(' ')]
    ts_hive_ids = [int(ids) for ids in sys.argv[4].replace('[','').replace(']','').split(' ')]

    print("Evaluation of individual START")
    eval_individual(num_gen, indiv_idx, tr_hive_ids,ts_hive_ids)
    print("Evaluation of individual END")

# parser = argparse.ArgumentParser(description="A simple example of argparse")
# Add arguments
# parser.add_argument('--num_gen', type=int, help='Generation no',default=1)
# parser.add_argument('--indiv_index', type=int, help='Index of individual',default=None)
# parser.add_argument('--hive_id', type=int, help='hive id',default=22)

# Parse the arguments
# args = parser.parse_args()
# print(args)
#    eval_individual(args.num_gen,args.indiv_index,args.hive_id)
# print("Evaluation of individual ENDS")

