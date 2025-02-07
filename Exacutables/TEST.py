import numpy as np
from Outputs import data_reader as dr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz


if __name__ == "__main__":

    # svr_stats = return {"model_class":"SVR","MSE":mse,"params":best_params}

    # lr_stats = return {"model_class": "LR", "MSE": mse, "params": best_model, 'importance': average_p.to_list()}
    # lr_stats['all_data_importance'] = ....


    # res={'SVR':svr_stats,'DTR':dtr_stats,'LR':lr_stats}


    array = [[0.0,   222.0, 2.0,   3.01,   4.0,  5.01],
             [444.0, 4.0,   8.0,   8.01,   10.0, 10.10],
             [2.0,   5.0,   888.0, 999.0,  1.0,  4.01]]

    print(np.argsort(np.argsort(array, axis=0), axis=0))


    #for i in range(np.shape(z)[0]):
    #    for j in range(np.shape(z)[1]):
    #        print(np.argwhere(array == z[i][j]))
