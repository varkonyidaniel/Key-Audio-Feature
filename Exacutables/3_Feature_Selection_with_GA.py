from Genetic_Algorithm import GA as ga
import numpy as np
import Outputs.data_writter as dw
import Outputs.data_reader as dr
import os
import argparse
# TODO: paraméterek külső beállítását, megírni!!!
if __name__ == "__main__":



    #Size_of_Population = 10      #páros kell, hogy legyen a szülők generálása miatt!!
    #Lengt_of_Chromosome = 10    # = number of features et all.
    #Early_Stopping_Max = 5
    #max_generation = 15
    #n_elites = 3
    #n_generation= 0
    #mutation_probability = 0.4
    #fitness_limit = 11
    #num_important_features = 5
    source_dir = "/DATA/FE"
    target_dir = "/DATA/LOG"
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    hive_id = 26
    #select_k = 20
    hive_ids = [25,26,27]
    local_test=True
    if local_test:
        hive_id = 22
        hive_ids = [22]
        max_generation=2
    parser = argparse.ArgumentParser(description="A simple example of argparse")
    # Add arguments
    parser.add_argument('--Size_of_Population', type=int, help='Generation no',default=10)
    parser.add_argument('--Lengt_of_Chromosome', type=int, help='Generation no',default=10)
    parser.add_argument('--Early_Stopping_Max', type=int, help='Generation no',default=5)
    parser.add_argument('--max_generation', type=int, help='hive id',default=15)
    parser.add_argument('--n_elites', type=int, help='hive id',default=3)
    parser.add_argument('--n_generation', type=int, help='hive id',default=0)
    parser.add_argument('--mutation_probability', type=float, help='hive id',default=0.4)
    parser.add_argument('--fitness_limit', type=int, help='hive id',default=11)
    parser.add_argument('--num_important_features', type=int, help='hive id',default=5)
    parser.add_argument('--select_k', type=int, help='hive id',default=20)
    parser.add_argument('--n_generation', type=int, help='hive id',default=0)

    # Parse the arguments
    args = parser.parse_args()

    print("Checking directories... START")
    dw.check_directories(parent_dir,source_dir, target_dir)
    print("Checking directories... END")


    print("Creation of GA object... START")
    ga = ga.GeneticAlgorithm(args.Size_of_Population, args.Lengt_of_Chromosome, args.Early_Stopping_Max,local_test=local_test)
    print("Creation of GA object... END")

    print("GENETIC ALGORITHM... START")

    print("Creation + save of initial population to GA object... START")
    initial_population = ga.init_population()
    print("Creation + save of initial population to GA object... END")

    # write out all individuals of population
    #TODO
    dw.re_write_data_to_h5(directory=parent_dir + target_dir,filename=f"{hive_id}_population_0",
                        ds_label="population", data=initial_population)

    # Eval all individuals of firts generation.
    # Save eval results and DT objects to GA object.

    # write out all fitness values belonging to all individuals
    print("Evaluation of population 0... START")
    ga.eval_population(0,hive_ids)                       # num_gen = 0, 1st generation
    print("Evaluation of population 0... END")


    # write out all individuals of population
    dw.write_data_to_h5(directory=parent_dir + target_dir,filename=f"{hive_id}_population_0",
                        ds_label="fitness_values", data=ga.get_all_fitness_values())

    n_generation = args.n_generation + 1

    while(n_generation < max_generation):

        print(f"Processing {n_generation}th generation START")

        print(f"Creation + save of {n_generation}th generation... START")

        # select n most important feature by DT objects of all individuals
        idx_imp_features = ga.get_most_important_features(args.num_important_features,hive_ids)

        # generate next population using selection, crossover, mutation
        # overwrites old population with new
        pop_n = ga.gen_next_generation (args.n_elites,args.mutation_probability,idx_imp_features, args.select_k)


        print(f"Creation + save of {n_generation}th generation... END")

        dw.write_data_to_h5(directory=parent_dir + target_dir,
                            filename=f"{hive_id}_population_{n_generation}",
                            ds_label="population", data=pop_n)

        print(f"Evaluation of population {n_generation}... START")
        ga.eval_population(n_generation,hive_ids)
        print(f"Evaluation of population {n_generation}... END")

        dw.write_data_to_h5(directory=parent_dir + target_dir,
                            filename=f"{hive_id}_population_{n_generation}",
                            ds_label="fitness_values", data=ga.get_all_fitness_values())

        # Checking BREAK conditions 1 by 1
        _val = ga.get_best_fitness_value()

        print(f"Checking fitness limit: current value: {_val}, limit:{args.fitness_limit}")
        if _val >= args.fitness_limit:
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

    print("GENETIC ALGORITHM... ENDS")
