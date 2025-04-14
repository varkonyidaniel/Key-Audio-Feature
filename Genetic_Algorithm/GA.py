import random, subprocess, os, sys, fnmatch
import numpy as np
from operator import itemgetter
from time import sleep
import joblib
from multiprocessing import Process


# https://www.datacamp.com/tutorial/genetic-algorithm-python


class GeneticAlgorithm:

    def __init__(self, size_of_population: int, length_of_chromosome: int, early_stopping_max_iter: int,
                 local_test: bool = False):

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
        self.test_results = {}

        # self.DTs = {}
        # self.max_reg = np.array([])
        # individual is a list of numbers a.k.a: index of not zero genes a.k.a:
        # the selected features for furter steps
        self.local_test = local_test
        """
        if self.local_test:
            self.size_of_population = 2
            self.length_of_chromosome = 5
        """

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
                pop = np.hstack((pop, self.__generate_individual()))
            else:
                pop = np.vstack((pop, self.__generate_individual()))
        self.population = pop
        return pop

    def set_population(self, pop) -> None:
        self.population = pop

    def get_population(self) -> np.ndarray:
        return self.population

    def get_individual(self, index: int) -> np.ndarray:
        return self.population[index]

    def get_all_fitness_values(self) -> np.ndarray:
        return self.fitness_values

    def set_all_fitness_values(self, values: np.ndarray) -> None:
        self.fitness_values = values

    def get_fitness_value(self, index: int) -> float:
        return self.fitness_values[index]

    def get_best_fitness_value(self) -> float:
        # indexes of individuals with largest fitness values, n pieces
        best_fitness = max(self.fitness_values)
        return best_fitness

    # https://algorithmafternoon.com/books/genetic_algorithm/chapter04/
    def selection(self, tournament_k) -> np.ndarray:
        _candidates_idxs = random.sample(range(self.size_of_population), tournament_k)
        parent_idxs = sorted(_candidates_idxs, key=lambda i: -self.fitness_values[i])[:-2]  # sorted = növekvő sorrend
        return itemgetter(*parent_idxs)(self.population)

    # 1 point crossover, 2 child
    def crossover(self, parents: np.ndarray) -> np.ndarray:
        split_idx = random.randint(0, self.length_of_chromosome)

        ch1 = np.concatenate((parents[0][:split_idx],
                              parents[1][split_idx:]))
        ch2 = np.concatenate((parents[0][split_idx:],
                              parents[1][:split_idx]))
        children = np.vstack((ch1, ch2))
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

    # best score not changing, early stop after some rounds
    def early_stopping_check(self):
        # no change in the best score
        if self.current_best_fitness == self.early_stopping_last_value:
            # increase the number of no changes in the best score
            self.early_stopping_count += 1
            # reached the maximum number, ...
            if self.early_stopping_count == self.early_stopping_max_iter:
                self.early_stopping_count = 0
                self.current_best_fitness = 0
                return True
        else:  # if not equal, then greater
            self.current_best_fitness = self.early_stopping_last_value
            self.early_stopping_count = 0
        return False

    def gen_next_generation(self, n_elites: int, mutation_prob: float,
                            idx_imp_features: np.ndarray,
                            selection_k: int) -> np.ndarray:

        shape = self.population.shape
        next_generation = np.zeros(shape)
        next_gens_fitness = np.zeros((self.size_of_population))

        # keep elite individuals and fitness values
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

        self.population = next_generation
        return next_generation
    # return index list
    def get_n_best_individuals(self, n: int) -> np.ndarray:
        # indexes of individuals with largest fitness values, n pieces
        top_indeces = np.argpartition(self.fitness_values, -n)[-n:]
        return top_indeces

    # number of logfiles according to a give pattern
    def n_finished_slurm_jobs(self, log_dir: str, file_name_pattern: str):
        return len(fnmatch.filter(os.listdir(f"{log_dir}"), f"{file_name_pattern}"))

    def start_slurm_proc(self, num_gen: int, idx_indiv: int, node_id: int,
                         img_f_path: str, img_f_name: str, path_eval_indiv: str,
                         tr_hive_ids: np.ndarray, ts_hive_ids:np.ndarray):

        if not os.path.isdir("./slurm_logs"):
            os.mkdir("./slurm_logs")
            print(f'Missing source directory: ./slurm_logs. Created!')
            sys.exit(1)

        obj_name = f"node_{node_id}_gen_{num_gen}_indiv_{idx_indiv}_"\
                   f"tr_{str(tr_hive_ids).replace('[', '').replace(']', '').replace(' ', '_')}_"\
                   f"ts_{str(ts_hive_ids).replace('[', '').replace(']', '').replace(' ', '_')}"

        eval_indiv_params = f"{num_gen} {idx_indiv} "\
                            f"{str(tr_hive_ids).replace('[', '').replace(']', '').replace(' ',',')} "\
                            f"{str(ts_hive_ids).replace('[', '').replace(']', '').replace(' ',',')}"

        cmd_ = f"/usr/bin/sbatch " \
               f"--output=./slurm_logs/log_job_{obj_name}.out " \
               f"--job-name=job_{obj_name} " \
               f"--nodelist=node{node_id} " \
               f"--wrap 'singularity exec {img_f_path}/{img_f_name} " \
               f"{path_eval_indiv} {eval_indiv_params}'"

        print(cmd_)

        subprocess.run(cmd_, shell=True)
        print(f"start_slurm_proc fc: {idx_indiv}th slurm process ... STARR")


    def get_most_important_features(self, num_features: int):

        # in eval_population function
        # self.feature_importance[(i, key)] = _data['all_data_importance']
        # key = LR / DTR / SVR
        # i = individual index

        f_rank_by_indiv_DT = {}
        f_rank_by_indiv_LR = {}

        for i in range(self.size_of_population):
            f_rank_by_indiv_DT[(i, 'DT')] = np.argsort(np.argsort(self.feature_importance[(i, 'DT')]))
            f_rank_by_indiv_LR[(i, 'LR')] = np.argsort(np.argsort(self.feature_importance[(i, 'LR')]))

        f_avg_order = [np.mean(k) for k in zip(
            np.array(list(f_rank_by_indiv_DT.values())).mean(axis=0),
            np.array(list(f_rank_by_indiv_LR.values())).mean(axis=0))]

        return np.argsort(f_avg_order)[:num_features]


    def eval_population(self, num_gen: int, tr_hive_ids: np.ndarray, ts_hive_ids:np.ndarray,
                        working_node_ids:np.ndarray, wait_sec:int):

        SICMD = "singularity exec --bind /$HOME/Key-Audio-Feature:/mnt /singularity/21_Peter/kaf.simg"
        SCRIPT = "conda run -n KAF python3 Executables/Eval_Individual.py"
        img_f_name = "key_audio.simg"  # "bee_project_2.simg"
        img_f_path = "/singularity/09-daniel-bee_project"
        path_eval_indiv = "/home/daniel.varkonyi/KEY_AUDIO/Executables/Eval_Individual.py"
        log_dir = "../DATA/LOG"
        log_pattern = f"gen_*_indiv_*.joblib"
        active_processes = []

        # indiv idx: 1, ... , 500

        for idx_indiv in range(self.size_of_population):
            if self.local_test:
                #runjob_sh_params = ["--num_gen", str(num_gen), "--indiv_index", str(idx_indiv), "--hive_id",
                #                    str(hive)]
                runjob_sh_params = [ str(num_gen),  str(idx_indiv), str(tr_hive_ids), str(ts_hive_ids)]
                # current_directory = os.getcwd()

                # Print the current working directory
                # print("Current Working Directory:", current_directory)
                # TODO WINDOWS:
                subprocess.Popen(
                    ["../venv/Scripts/python.exe", "../Exacutables/Eval_Individual.py"] + runjob_sh_params)
                # TODO UNIX (?):
                # subprocess.Popen(["../venv/bin/python", "../Exacutables/Eval_Individual.py"]+runjob_sh_params)
                print("-->", idx_indiv, "started...")
                waitting_term = 10

            else:

                cnt_work_nodes = len(working_node_ids)
                node_idx = working_node_ids[idx_indiv%cnt_work_nodes]

                # start multiple Eval_Indivual
                print("else case: multi slurm process start")
                p = Process(target=self.start_slurm_proc,
                            args=(num_gen, idx_indiv, node_idx,
                                  img_f_path, img_f_name, path_eval_indiv,
                                  tr_hive_ids,ts_hive_ids,))
                p.start()
                active_processes.append(p)
                for proc in active_processes:
                    proc.join()
        # Execution waits here until last's start

        # Start slurm process --> runjob.sh --> Exacutables/Eval_Individual.py

        n_fin_jobs = self.n_finished_slurm_jobs(log_dir, log_pattern)
        while (self.size_of_population * (num_gen+1) > n_fin_jobs):     # num_gen 0-tól indul ezért kell a +1
            print(f"main process: num of finished jobs: {n_fin_jobs}, sleep: {wait_sec}")
            sleep(wait_sec)

        print("main process:all subjobs finished, progress continue!")

        # Read all data from files created by Eval_Individual.pys

        for i in range(self.size_of_population):                                # far all individual
            file = joblib.load(f"../DATA/LOG/gen_{num_gen}_indiv_{i}.joblib")

            for key in file.keys():   # SVR / DTR / LR : svr_stats / dtr_stats / lr_stats
                _data = file.get(key)  # _data = lr_stats / dtr_stats / svr_stats
                self.results[(i, key)] = _data['MSE']
                self.test_results[(i, key)] = _data['test_mse']
                if not key == 'SVR':
                    self.feature_importance[(i, key)] = _data['all_data_importance']

            for i in range(self.size_of_population):
                self.fitness_values[i] = \
                    np.median([
                        self.results.get((i, 'SVR')),
                        self.results.get((i, 'LR')),
                        self.results.get((i, 'DTR'))
                    ])
