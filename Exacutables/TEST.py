import numpy as np
from Outputs import data_reader as dr
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz


if __name__ == "__main__":

    size_of_population = 3
    length_of_chromosome = 4
    avg_idx_value = {}
    feature_importance = {}

    # feature_importance[(0, 25, 'LR')] = np.array([0.1,0.2,0.3,0])
    # feature_importance[(0, 26, 'LR')] = np.array([0.2,0.2,0,0.3])
    # feature_importance[(0, 27, 'LR')] = np.array([0.3,0.2,0,0.3])
    # feature_importance[(0, 25, 'DT')] = np.array([0.4,0.2,0.3,0])
    # feature_importance[(0, 26, 'DT')] = np.array([0.5,0,  0.2,0.3])
    # feature_importance[(0, 27, 'DT')] = np.array([0.6,0.2,0,  0.3])
    #
    # feature_importance[(1, 25, 'LR')] = np.array([0.1,0.2,0.3,0])
    # feature_importance[(1, 26, 'LR')] = np.array([0.2,0.2,0.3,0])
    # feature_importance[(1, 27, 'LR')] = np.array([0.3,0.2,0.3,0])
    # feature_importance[(1, 25, 'DT')] = np.array([0.1,0.2,0.3,0])
    # feature_importance[(1, 26, 'DT')] = np.array([0.2,0.2,0.3,0])
    # feature_importance[(1, 27, 'DT')] = np.array([0.3,0.2,0.3,0])

    # i = egyed index, hive = 25,26,27, key = LR/DT
    # self.feature_importance[(i, hive, key)] = _data['all_data_importance']

    # for i in range(size_of_population):                                         # for all individuals
    #     for j in range(length_of_chromosome):                                   # for all genes
    #         avg_idx_value[(i, j, 'DT')] = round(np.average(
    #             [v[j] for k, v in feature_importance.items() if i == k[0] and 'DT' == k[2]]), 2)
    #
    #         avg_idx_value[(i, j, 'LR')] = round(np.average(
    #             [v[j] for k, v in feature_importance.items() if i == k[0] and 'LR' == k[2]]), 2)


    # avg_idx_value[(0, 0, 'DT')] = 0.5
    # avg_idx_value[(0, 1, 'DT')] = 0.4
    # avg_idx_value[(0, 2, 'DT')] = 0.7
    # avg_idx_value[(0, 3, 'DT')] = 0.2
    #
    # avg_idx_value[(1, 0, 'DT')] = 0.1
    # avg_idx_value[(1, 1, 'DT')] = 0.6
    # avg_idx_value[(1, 2, 'DT')] = 0.3
    # avg_idx_value[(1, 3, 'DT')] = 0.8
    #
    # avg_idx_value[(2, 0, 'DT')] = 0.11
    # avg_idx_value[(2, 1, 'DT')] = 0.54
    # avg_idx_value[(2, 2, 'DT')] = 0.33
    # avg_idx_value[(2, 3, 'DT')] = 0.54
    #
    #
    # avg_idx_value[(0, 0, 'LR')] = 0.1
    # avg_idx_value[(0, 1, 'LR')] = 0.3
    # avg_idx_value[(0, 2, 'LR')] = 0.1
    # avg_idx_value[(0, 3, 'LR')] = 0.1
    #
    # avg_idx_value[(1, 0, 'LR')] = 0.2
    # avg_idx_value[(1, 1, 'LR')] = 0.2
    # avg_idx_value[(1, 2, 'LR')] = 0.3
    # avg_idx_value[(1, 3, 'LR')] = 0.3
    #
    # avg_idx_value[(2, 0, 'LR')] = 0.11
    # avg_idx_value[(2, 1, 'LR')] = 0.54
    # avg_idx_value[(2, 2, 'LR')] = 0.33
    # avg_idx_value[(2, 3, 'LR')] = 0.54
    #
    # # feature index = 0, reg = DT
    DT_indiv_feat_order = {}
    LR_indiv_feat_order = {}
    #
    # # i. egyedre
    # for i in range(3):
    #     s = [-v for k, v in avg_idx_value.items() if i == k[0] and 'DT' == k[2]]
    #     DT_indiv_feat_order[(i,'DT')] = np.argsort(np.argsort(s, axis=0), axis=0)
    #
    #     z = [-v for k, v in avg_idx_value.items() if i == k[0] and 'LR' == k[2]]
    #     LR_indiv_feat_order[(i, 'LR')] = np.argsort(np.argsort(z, axis=0), axis=0)
    #
    # print(DT_indiv_feat_order, LR_indiv_feat_order)

    DT_indiv_feat_order[(0, 'DT')] =  [1, 2, 0, 3]
    DT_indiv_feat_order[(1, 'DT')] =  [3, 1, 2, 0]
    DT_indiv_feat_order[(2, 'DT')] =  [3, 0, 2, 1]

    LR_indiv_feat_order[(0, 'LR')] =  [1, 0, 2, 3]
    LR_indiv_feat_order[(1, 'LR')] =  [2, 3, 0, 1]
    LR_indiv_feat_order[(2, 'LR')] =  [3, 0, 2, 1]

    avg_pop_feat = np.zeros((size_of_population,length_of_chromosome))

    for j in range(size_of_population):
        for i in range(length_of_chromosome):
            avg_pop_feat[j,i] = np.average(
            [DT_indiv_feat_order[(j,'DT')][i],
            LR_indiv_feat_order[(j,'LR')][i]]
            )

    print(avg_pop_feat)
    avg_pop = np.zeros(length_of_chromosome)

    for j in range(length_of_chromosome):
        avg_pop = np.average(avg_pop_feat, axis=0)

    print(np.argsort(np.argsort(avg_pop, axis=0), axis=0))
    print(avg_pop)



    #for i in range(size_of_population):  # for all individuals
    #    for j in range(length_of_chromosome):  # for all genes
    #        np.argsort(np.argsort(array, axis=0), axis=0))






    #array = [[0.0,   222.0, 2.0,   3.01,   4.0,  5.01],
    #         [444.0, 4.0,   8.0,   8.01,   10.0, 10.10],
    #         [2.0,   5.0,   888.0, 999.0,  1.0,  4.01]]

    #print(np.argsort(np.argsort(array, axis=0), axis=0))


    #for i in range(np.shape(z)[0]):
    #    for j in range(np.shape(z)[1]):
    #        print(np.argwhere(array == z[i][j]))
