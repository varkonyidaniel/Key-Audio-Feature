from Genetic_Algorithm import GA as ga
import numpy as np
import Outputs.data_writter as dw
import Outputs.data_reader as dr
import os

# TODO: paraméterek külső beállítását, megírni!!!
if __name__ == "__main__":

    Size_of_Population = 10      #páros kell, hogy legyen a szülők generálása miatt!!
    Lengt_of_Chromosome = 10    # = number of features et all.
    Early_Stopping_Max = 5
    max_generation = 15
    n_elites = 3
    n_generation= 0
    mutation_probability = 0.4
    fitness_limit = 11
    num_important_features = 5
    source_dir = "/DATA/FE"
    target_dir = "/DATA/LOG"
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    hive_id = 26
    select_k = 20

    print("Creation of GA object... START")
    ga = ga.GeneticAlgorithm(Size_of_Population, Lengt_of_Chromosome, Early_Stopping_Max)
    print("Creation of GA object... END")

    print("Creation + save of initial population... START")
    initial_population = ga.init_population()
    print("Creation + save of initial population... END")

    print("Checking directories... START")
    dw.check_directories(parent_dir,source_dir, target_dir)
    print("Checking directories... END")

    dw.write_data_to_h5(directory=parent_dir + target_dir,filename=f"{hive_id}_population_0",
                        ds_label="population", data=initial_population)

    print("Evaluation of population 0... START")
    ga.eval_population(0)                       # num_gen = 0, 1st generation
    print("Evaluation of population 0... END")

    dw.write_data_to_h5(directory=parent_dir + target_dir,filename=f"{hive_id}_population_0",
                        ds_label="fitness_values", data=ga.get_all_fitness_values())

    n_generation = n_generation + 1

    while(n_generation < max_generation):

        print(f"Processing {n_generation}th generation START")
        # selection / crossover / mutation
        print(f"Creation + save of {n_generation}th generation... START")

        idx_imp_features = ga.get_most_important_features(num_important_features, n_generation)
        pop_n = ga.gen_next_generation (n_elites,mutation_probability,idx_imp_features, select_k)
        print(f"Creation + save of {n_generation}th generation... END")

        dw.write_data_to_h5(directory=parent_dir + target_dir,
                            filename=f"{hive_id}_population_{n_generation}",
                            ds_label="population", data=pop_n)

        print(f"Evaluation of population {n_generation}... START")
        ga.eval_population(n_generation)
        print(f"Evaluation of population {n_generation}... END")

        dw.write_data_to_h5(directory=parent_dir + target_dir,
                            filename=f"{hive_id}_population_{n_generation}",
                            ds_label="fitness_values", data=ga.get_all_fitness_values())

        # Checking BREAK conditions 1 by 1
        _val = ga.get_best_fitness_value()

        print(f"Checking fitness limit: current value: {_val}, limit:{fitness_limit}")
        if _val >= fitness_limit:
            print(f"Limit raching: TRUE")
            break
        print(f"Limit raching: FALSE")

        print(f"Early stopping check... START")
        # if best score was not changing for multiple rounds
        if ga.early_stopping_check():
            print(f"Early stopping! Threshold reached!")
            break
        print(f"Early stopping check... END")
        print(f"Processing {n_generation}th generation END")

        n_generation = n_generation + 1
