import sys,os
sys.path.append(os.path.abspath('../Genetic_Algorithm'))
sys.path.append(os.path.abspath('../Outputs'))

from Genetic_Algorithm import GA as ga
import numpy as np
import Outputs.data_writter as dw
import os
#import argparse

def check_input_params(argv:list):
    if len(argv) != 14:
        print(f"wrong number of params {argv}")
        exit(3)
    else:
        try:
            S_of_Pop = int(argv[1]) # páros legyen!!
            L_of_Chr = int(argv[2])
            E_Stp_Max = int(argv[3])
            max_gen = int(argv[4])
            n_elites = int(argv[5])
            mut_prob = float(argv[6])
            fit_limit = int(argv[7])
            num_imp_feat = int(argv[8])
            tournament_k = int(argv[9])
            tr_hive_ids = np.fromstring(argv[10],sep=',', dtype=int)
            ts_hive_ids = np.fromstring(argv[11],sep=',', dtype=int)
            working_node_ids = np.fromstring(argv[12],sep=',', dtype=int)
            wait_sec = int(argv[13])

            return S_of_Pop, L_of_Chr, E_Stp_Max, max_gen, n_elites, mut_prob, \
                   fit_limit, num_imp_feat, tournament_k, tr_hive_ids, ts_hive_ids, \
                   working_node_ids,wait_sec
        except Exception as ex:
            print(ex.with_traceback())


#TODO: fitness limitet úgy beállítani, hogy 1 napnál ne legyen nagyobb az átlagos tévedés!!


if __name__ == "__main__":

    source_dir = "/DATA/FE"
    target_dir = "/DATA/LOG"
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    n_generation = 0

    #hive_id = 26
    #select_k = 20
    #hive_ids = [20,30,37]
    #
    # if local_test:
    #     hive_id = 22
    #     hive_ids = [22]
    #     max_generation=2
    # parser = argparse.ArgumentParser(description="A simple example of argparse")
    # # Add arguments
    # parser.add_argument('--Size_of_Population', type=int, help='Generation no',default=10)
    # parser.add_argument('--Lengt_of_Chromosome', type=int, help='Generation no',default=10)
    # parser.add_argument('--Early_Stopping_Max', type=int, help='Generation no',default=5)
    # parser.add_argument('--max_generation', type=int, help='hive id',default=15)
    # parser.add_argument('--n_elites', type=int, help='hive id',default=3)
    # #parser.add_argument('--n_generation', type=int, help='hive id',default=0)
    # parser.add_argument('--mutation_probability', type=float, help='hive id',default=0.4)
    # parser.add_argument('--fitness_limit', type=int, help='hive id',default=11)
    # parser.add_argument('--num_important_features', type=int, help='hive id',default=5)
    # parser.add_argument('--tournament_k', type=int, help='hive id',default=20)
    #
    # # Parse the arguments
    # args = parser.parse_args()

    Size_of_Population, Lengt_of_Chromosome, Early_Stopping_Max, \
    max_generation, n_elites, mutation_prob, fitness_trsh, \
    num_important_features, tourn_sel_k,tr_hive_ids, ts_hive_ids, \
    working_node_ids,wait_sec = check_input_params(sys.argv)


    #hack

    local_test= ts_hive_ids[0] in tr_hive_ids
    if local_test:
        Size_of_Population=4
        n_elites=2
        Lengt_of_Chromosome=5
        tourn_sel_k=4

    '''
    if local_test:
        
        import joblib
        with open("../DATA/feature_list_with_dim.joblib", 'rb') as f:
            fl = joblib.load(f)
            Lengt_of_Chromosome = sum([dim[0] for dim in fl.values()])
    '''


    print("Checking directories... START")
    dw.check_directories(parent_dir,source_dir, target_dir)
    print("Checking directories... END")

    print("GENETIC ALGORITHM... START")

    print("    Creation of GA object... START")
    ga = ga.GeneticAlgorithm(Size_of_Population, Lengt_of_Chromosome, Early_Stopping_Max,local_test=local_test)
    print("    Creation of GA object... END")

    print(f"    Prosessing of 0th generation ... START")
    print(f"        Creation + save of initial population to GA object... START")
    initial_population = ga.init_population()
    print("        Creation + save of initial population to GA object... END")

    # write out all individuals of population
    #filename=f"{hive_id}_population_0",
    dw.re_write_data_to_h5(directory=parent_dir + target_dir, filename=f"population_{n_generation}",
                           ds_label="population", data=initial_population)

    # evaluate all individuals, count fitness values belonging to all individuals
    print("        Evaluation of population 0... START")

    # num_gen = 0, 1st generation
    ga.eval_population(n_generation,tr_hive_ids, ts_hive_ids, working_node_ids, wait_sec)
    print("        Evaluation of population 0... END")

    # write out all fitness values belonging to the population
    dw.write_data_to_h5(directory=parent_dir + target_dir,filename=f"population_{n_generation}",
                        ds_label="fitness_values", data=ga.get_all_fitness_values())

    while(n_generation < max_generation):

        print(f"        Get most important features of{n_generation}th generation... START")
        idx_imp_features = ga.get_most_important_features(num_important_features)
        print(f"        Get most important features of{n_generation}th generation... END")

        print(f"    Prosessing of {n_generation}th generation ... END")

        # generation index increase by 1
        n_generation = n_generation + 1

        print(f"    Prosessing of {n_generation}th generation ... START")

        # generate next population , overwrites old population with new in GA object
        print(f"        Creation + save of {n_generation}th generation... START")
        pop_n = ga.gen_next_generation (n_elites,mutation_prob,idx_imp_features, tourn_sel_k)
        print(f"        Creation + save of {n_generation}th generation... END")

        # write outactual (n+1)th  generation into file.
        dw.write_data_to_h5(directory=parent_dir + target_dir, filename=f"population_{n_generation}",
                            ds_label="population", data=pop_n)

        print(f"        Evaluation of population {n_generation}... START")
        ga.eval_population(n_generation,tr_hive_ids, ts_hive_ids,working_node_ids,wait_sec)
        print(f"        Evaluation of population {n_generation}... END")

        # evaluate all individuals, count fitness values belonging to all individuals
        dw.write_data_to_h5(directory=parent_dir + target_dir,
                            filename=f"population_{n_generation}",
                            ds_label="fitness_values", data=ga.get_all_fitness_values())

        # Checking BREAK conditions 1 by 1
        _val = ga.get_best_fitness_value()

        print(f"        Checking fitness limit: current value: {-_val}, limit:{-fitness_trsh} START")
        if -_val >= -fitness_trsh:
            print(f"        Limit raching: TRUE")
            break
        print(f"        Limit raching: FALSE")
        print(f"        Checking fitness limit: current value: {-_val}, limit:{-fitness_trsh} END")

        print(f"        Early stopping check... START")
        # if best score was not changing for multiple rounds
        if ga.early_stopping_check():
            print(f"        Early stopping! Threshold reached!")
            break
        print(f"        Early stopping check... END")

    print("GENETIC ALGORITHM... ENDS")
    ga.print_lr()
