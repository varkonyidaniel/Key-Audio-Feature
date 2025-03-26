import numpy as np

if __name__ == "__main__":


    if __name__ == "__main__":

        size_of_population = 2
        length_of_chromosome = 4
        avg_idx_value = {}
        feature_importance = {}

        feature_importance[(0, 25, 'LR')] = np.array([0.1,0.2,0.3,0])
        feature_importance[(0, 26, 'LR')] = np.array([0.2,0.2,0,0.3])
        feature_importance[(0, 27, 'LR')] = np.array([0.3,0.2,0,0.3])
        feature_importance[(0, 25, 'DT')] = np.array([0.4,0.2,0.3,0])
        feature_importance[(0, 26, 'DT')] = np.array([0.5,0,  0.2,0.3])
        feature_importance[(0, 27, 'DT')] = np.array([0.6,0.2,0,  0.3])

        feature_importance[(1, 25, 'LR')] = np.array([0.1,0.2,0.3,0])
        feature_importance[(1, 26, 'LR')] = np.array([0.2,0.2,0.3,0])
        feature_importance[(1, 27, 'LR')] = np.array([0.3,0.2,0.3,0])
        feature_importance[(1, 25, 'DT')] = np.array([0.1,0.2,0.3,0])
        feature_importance[(1, 26, 'DT')] = np.array([0.2,0.2,0.3,0])
        feature_importance[(1, 27, 'DT')] = np.array([0.3,0.2,0.3,0])



        for i in range(size_of_population):                                         # for all individuals
            for j in range(length_of_chromosome):                                   # for all genes
                avg_idx_value[(i, j, 'DT')] = round(np.average(
                    [v[j] for k, v in feature_importance.items() if i == k[0] and 'DT' == k[2]]), 2)

                avg_idx_value[(i, j, 'LR')] = round(np.average(
                    [v[j] for k, v in feature_importance.items() if i == k[0] and 'LR' == k[2]]), 2)

        #print(avg_idx_value)

        # feature index = 0, reg = DT
        DT_indiv_feat_order = {}
        LR_indiv_feat_order = {}

        for i in range(size_of_population):
            s = [-v for k, v in avg_idx_value.items() if i == k[0] and 'DT' == k[2]]
            DT_indiv_feat_order[(i,'DT')] = np.argsort(np.argsort(s, axis=0), axis=0)

            z = [-v for k, v in avg_idx_value.items() if i == k[0] and 'LR' == k[2]]
            LR_indiv_feat_order[(i, 'LR')] = np.argsort(np.argsort(z, axis=0), axis=0)

        #print(DT_indiv_feat_order, LR_indiv_feat_order)

        avg_pop_feat = np.zeros((size_of_population, length_of_chromosome))

        for j in range(size_of_population):
            for i in range(length_of_chromosome):
                avg_pop_feat[j, i] = np.average(
                    [DT_indiv_feat_order[(j, 'DT')][i],
                     LR_indiv_feat_order[(j, 'LR')][i]]
                )

        #print(avg_pop_feat)

        avg_pop = np.zeros(length_of_chromosome)

        for j in range(length_of_chromosome):
            avg_pop = np.average(avg_pop_feat, axis=0)

        #populáció szinten a feature-ök fontossági sorrendje csökkenő rendben:
        # sorrendben az indexek kellenek a return értékbe (0,2,1,3)

        print(avg_pop)  # feature-ök relatív fontossági sorrendje csökkenő rendben, innen kellenek az indexek

        res = np.argsort(avg_pop)[:2]
        print(res)

