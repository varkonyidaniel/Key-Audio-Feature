import random, subprocess, os, sys, fnmatch
import numpy as np
from Enum.enum_types import Regression_method as rm
from operator import itemgetter
from multiprocessing import Process,Manager
from time import sleep
import h5py
import joblib


# https://www.datacamp.com/tutorial/genetic-algorithm-python


#python-bólindítani a slurm process-eket, hogy a DT tanítások párhuzamosan tudjanak menni!
# a jó featur-ökre indítani több regressor-t is!
# 3 brood-os pozitív, 3 nem brood-os negatív kaptárt!
# DT helyett más regress-orokat!
# Több regressor eredményének összehasonlítása Pareto Front - segítségével.
# https://en.wikipedia.org/wiki/Pareto_front


class GeneticAlgorithm:

    def __init__(self, size_of_population:int, length_of_chromosome:int, early_stopping_max_iter:int):

        # number of population
        self.size_of_population = size_of_population

        # number of features at all
        self.length_of_chromosome = length_of_chromosome

        self.feature_index_list = list(range(length_of_chromosome))
        self.fitness_values = np.zeros(size_of_population, dtype=float)
        self.current_best_fitness = 0
        self.early_stopping_last_value = 0
        self.early_stopping_count = 0

        # max number of iterations when the best fitness score not changing (burnout threshold)
        self.early_stopping_max_iter = early_stopping_max_iter

        # object to store decision tree object, to determine feature importance.

        self.feature_importance = {}
        self.results = {}

        #self.DTs = {}
        #self.max_reg = np.array([])
        # individual is a list of numbers a.k.a: index of not zero genes a.k.a:
        # the selected features for furter steps

    # generate individual - private function
    def __generate_individual(self) -> np.ndarray:
        # generate "len_of_chr" number of values from [0,1]
        genes = np.random.choice(2, self.length_of_chromosome)
        return genes

    # generate population
    def init_population(self) -> np.ndarray:
        pop = np.array([])
        for i in range(self.size_of_population):
            if i == 0:
                pop = np.hstack((pop,self.__generate_individual()))
            else:
                pop = np.vstack((pop, self.__generate_individual()))
        self.population = pop
        return pop

    def set_population(self,pop) -> None:
        self.population = pop

    def get_population(self) -> np.ndarray:
        return self.population

    def get_individual(self,index:int) -> np.ndarray:
        return self.population[index]

    def get_all_fitness_values(self) -> np.ndarray:
        return self.fitness_values

    def set_all_fitness_values(self, values:np.ndarray) -> None:
         self.fitness_values = values

    def get_fitness_value(self,index:int) -> float:
         return self.fitness_values[index]

    # https://algorithmafternoon.com/books/genetic_algorithm/chapter04/
    def selection(self,k) -> np.ndarray:
        _candidates_idxs = random.sample(range(self.size_of_population), k)
        parent_idxs = sorted(_candidates_idxs, key=lambda i: -self.fitness_values[i])[:-2] # sorted = növekvő sorrend
        return itemgetter(*parent_idxs)(self.population)

    # 1 point crossover, 2 child
    def crossover(self, parents:np.ndarray) -> np.ndarray:
        split_idx = random.randint(0, self.length_of_chromosome)

        ch1 = np.concatenate((parents[0][:split_idx],
                              parents[1][split_idx:]))
        ch2 = np.concatenate((parents[0][split_idx:],
                              parents[1][:split_idx]))
        children = np.vstack((ch1,ch2))
        return children

    # mutation for all genes included heuristic mutation
    def mutation(self, indiv, probability: float, idx_imp_features: np.ndarray) -> np.ndarray:
        for i in range(self.length_of_chromosome):
            if i in idx_imp_features:
                indiv[i] = 1
            else:
                # number between 0.0 and 1.0, randomly mutate
                if random.random() <= probability:
                    indiv[i] = abs(indiv[i] - 1)
        return indiv

    def get_best_fitness_value(self) -> float:
        # indexes of individuals with largest fitness values, n pieces
        best_fitness = max(self.fitness_values)
        return best_fitness

    # best score not changing, early stop after some rounds
    def early_stopping_check(self):
        # no change in the best score
        if self.current_best_fitness == self.early_stopping_last_value:
            # increase the number of no changes in the best score
            self.early_stopping_count += 1
            # reached the maximum number, ...
            if self.early_stopping_count == self.early_stopping_max_iter:
                self.early_stopping_count = 0
                return True
        else:  # if not equal, then greater
            self.current_best_fitness = self.early_stopping_last_value
            early_stopping_count = 0
        return False

    def gen_next_generation(self, n_elites: int, mutation_prob: float,
                                idx_imp_features: np.ndarray,
                            selection_k:int) -> np.ndarray:

        shape = self.population.shape
        next_generation = np.zeros(shape)
        next_gens_fitness = np.zeros((self.size_of_population))

        # index of individuals will go to next generation without any change
        # fitness of elites no need to recalculate
        elite_idxs = self.get_n_best_individuals(n_elites)
        next_generation[elite_idxs] = self.population[elite_idxs]
        next_gens_fitness[elite_idxs] = self.fitness_values[elite_idxs]

        for i in range(int(self.size_of_population / 2)):
            parents = self.selection(selection_k)  # select 2 parents
            children = self.crossover(parents)
            if 2 * i not in elite_idxs:
                children[0] = self.mutation(children[0], mutation_prob, idx_imp_features)
                next_generation[2 * i] = children[0]

            if 2 * i + 1 not in elite_idxs:
                children[1] = self.mutation(children[1], mutation_prob, idx_imp_features)
                next_generation[2 * i + 1] = children[1]

        self.population=next_generation
        return next_generation

    # return index list
    def get_n_best_individuals(self, n:int) -> np.ndarray:
        # indexes of individuals with largest fitness values, n pieces
        top_indeces = np.argpartition(self.fitness_values, -n)[-n:]
        return top_indeces



# TESZTELNI

    # TODO: Peti - teszt szekvenciális eval individula hívás 3-szor egymás után a práhuzamosság helyett+

    #TODO: bemenő paramétereket megírni hozzá!!!
    def eval_population(self, num_gen:int, hive_ids:np.ndarray):

        node_cnt = 10
        img_f_name = "bee_project_2.simg"
        img_f_path = "/singularity/09-daniel-bee_project"
        path_run_job_sh = "/.../runjob.sh"
        log_dir = "DATA/LOG"
        node_idx = 0  # 1,2,3,6-nál hogyan állítjuk be???
        metric = "MSE"
        log_pattern = "gen_*_indiv_*_hive_*.h5"
        active_processes = []

        for hive in hive_ids:
            for idx_indiv in range(self.size_of_population):

                p = Process(target=self.start_slurm_proc(),
                            args=(self,num_gen,idx_indiv,node_idx,
                                  img_f_path,img_f_name,path_run_job_sh,hive))

                p.start()
                active_processes.append(p)
            for proc in active_processes:
                proc.join()

                # Execution waits here until last's start

        # Start slurm process --> runjob.sh --> Exacutables/Eval_Individual.py 4 params

        while(self.size_of_population* len(hive_ids) >self.all_slurm_jobs_finished(log_dir,log_pattern)):
            sleep(600)

        for hive in hive_ids:
            for i in range(self.size_of_population):                                # far all individual
                #file = h5py.File(f"gen_{num_gen}_indiv_{i}_hive_{hive}.h5", 'r')    # open file
                file = joblib.load(f"hive_{hive}_gen_{num_gen}_indiv_{i}.joblib")

                for key in file.keys():
                    _data = file.get(key)   # lr_stats / dtr_stats / svr_stats
                    self.results[(i, hive, key)] = _data['"MSE"']
                    if not key == 'SVR':
                        self.feature_importance[(i, hive, key)] = _data['importance']



                for i in range(self.size_of_population):
                    self.fitness_values[i] = \
                        np.median([
                            np.average([v for k, v in self.results.items() if i == k[0] and 'SVR' == k[2]]),
                            np.average([v for k, v in self.results.items() if i == k[0] and 'LR' == k[2]]),
                            np.average([v for k, v in self.results.items() if i == k[0] and 'DTR' == k[2]])
                                  ])

                #for key in file.keys():                                     # for all keys
                #   if key == "DT_model":                                   # if key is dt
                #       self.DTs[(i,hive)] = file.get(key)                  # save dt object
                #   else:                                                   # else: key --> result
                #       _res = file.get(key)[metric]
                #       if _res > self.fitness_values[i]:                   # save max result
                #           self.fitness_values[i] = _res                   # save max fitness
                #           self.max_reg[i] = key                           # save best regressor


    def all_slurm_jobs_finished(self,log_dir:str, pattern:str):
        par_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        return len(fnmatch.filter(os.listdir(f"{par_dir}/{log_dir}"), f"{pattern}"))

    def start_slurm_proc(self,num_gen: int, idx_indiv: int, node_id: int,
                         img_f_path: str, img_f_name: str, path_run_job_sh: str,
                         hive_id: int):

        if not os.path.isdir("./slurm_logs"):
            os.mkdir("./slurm_logs")
            print(f'Missing source directory: ./slurm_logs. Created!')
            sys.exit(1)

        obj_name = f"node_{node_id}_generation_{num_gen}_individual_{idx_indiv}_hive_{hive_id}"

        runjob_sh_params = f"{num_gen} {idx_indiv} {hive_id}"
        cmd_text = ["sbatch "]
        cmd_args = [f"--output=./slurm_logs/log_job_${obj_name}.out",
                    f"--job-name=job_{obj_name}",
                    f"--nodelist=node{node_id}",
                    f"--wrap 'singularity exec {img_f_path}/{img_f_name}'"
                    f"/{path_run_job_sh} {runjob_sh_params}"]

        subprocess.run(cmd_text + cmd_args)

    def create_level_info(self,children_left: np.ndarray, children_right:
                            np.ndarray) -> np.ndarray:

        level_info = -1 * np.ones((len(children_right)))
        proc = np.array([0])
        next_proc = np.array([])
        level = 0

        while (len(proc) > 0):

            act_ = int(proc[0])

            level_info[act_] = level
            next_proc = np.append(next_proc, children_left[act_])
            next_proc = np.append(next_proc, children_right[act_])
            proc = np.delete(proc, 0)
            if len(proc) == 0:
                level = level + 1
                proc = next_proc
                next_proc = np.array([])
                if proc.__contains__(-1):
                    proc = np.delete(proc, np.where(proc == -1))

        return level_info


    def tr_fet_values_to_chr_values(self,dt_fet_values: np.ndarray, idx: int) -> np.ndarray:
        indiv = self.population[idx]
        chr_fet_values = np.zeros(self.length_of_chromosome)
        x_ = []
        for i in range(self.length_of_chromosome):
            if indiv[i] == 1:
                x_.append(i)

        for idx in range(self.length_of_chromosome):
            chr_fet_values[x_[idx]] = dt_fet_values[idx]

        return chr_fet_values

    # TODO: Dani csinálja
    # TODO: weight-ek kiszámolásának módját átnézni.
    # TODO: Ne legyen az összeg mindíg 1? Check Tomival?
    # https://www.codecademy.com/article/fe-feature-importance-final
    # GINI IMPIRITY

    def calc_feature_values(self,tree_level_info: np.ndarray, features: np.ndarray) -> np.ndarray:

        len_ = len(tree_level_info)  # node number in the tree
        max_level = np.max(tree_level_info) + 1
        fet_values = np.zeros((len_))

        for i in range(len_):
            if features[i] != -2:
                weight_ = (max_level - tree_level_info[i]) / max_level  # 1/7
                fet_values[features[i]] = fet_values[features[i]] + weight_

        return fet_values



    # GINI INDEX
    def get_most_important_features(self, num_features: int, hive_ids:np.ndarray):

        # svr_stats = return {"model_class":"SVR","MSE":mse,"params":best_params}

        # lr_stats = return {"model_class": "LR", "MSE": mse, "params": best_model, 'importance': average_p.to_list()}
        # lr_stats['all_data_importance'] = ....

        # dtr_stats = return {"model_class":"DTR","MSE":mse,"params":best_params,'importance':model.feature_importances_.tolist()}
        # dtr_stats['all_data_importance'] = ...

        # res={'SVR':svr_stats,'DTR':dtr_stats,'LR':lr_stats}


        # egy egyedre
        individual_index = 1
        avg_gini = np.zeros(self.length_of_chromosome)
        avg_lr = np.zeros(self.length_of_chromosome)
        for i in range(self.length_of_chromosome): # for all genes
            avg_gini[i] = \
                np.average([v for k, v in self.feature_importance.items() if i == k[0] and 'DT' == k[2]])



        #feature_values_by_indiv = {}

        #for idx in range(self.size_of_population):         # for all individual
        #    for hive in hive_ids:



                #dt_tree_level_info = self.create_level_info(self.DTs[(idx,hive)].tree_.children_left,
                #                                         self.DTs[(idx,hive)].tree_.children_right)

                #dt_feature_value = self.calc_feature_values(dt_tree_level_info,
                #                                            self.DTs[(idx,hive)].tree_.feature)



                #feature_values_by_indiv[(idx,hive)] = self.tr_fet_values_to_chr_values(dt_feature_value,idx)

                #order_features = self.calc_order(feature_values_by_indiv)

        return order_features[:num_features]



    # MEGÍRNI



    # TODO: DANI csinálja - UTOLSÓ!!!
    def calc_order(self,value_matrix:dict)-> np.ndarray:

        # in nth generation - n th population
        # feature_values[(individual id,hive id)]

        # feature_values(1,26) - [0,0,345,3,4,0,0,0] - chromosome hosszú
        # feature_values(1,27) -
        # feature_values(1,28) -

        # feature_values(2,26) -
        # feature_values(2,27) -
        # feature_values(2,28) -

        # .
        # .
        # .

        # feature_values(500,26) -
        # feature_values(500,27) -
        # feature_values(500,28) -
        pass
