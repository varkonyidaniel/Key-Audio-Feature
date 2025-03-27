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

        # self.DTs = {}
        # self.max_reg = np.array([])
        # individual is a list of numbers a.k.a: index of not zero genes a.k.a:
        # the selected features for furter steps
        self.local_test = local_test
        if self.local_test:
            self.size_of_population = 2
            self.length_of_chromosome = 5

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

    #-----------------

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

    #-----------------

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

    #-----------------

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
                   f"tr_{tr_hive_ids.replace(',','_')}_ts_{ts_hive_ids.replace(',','_')}"

        eval_indiv_params = f"{num_gen} {idx_indiv} {tr_hive_ids} {ts_hive_ids}"

        cmd_ = f"/usr/bin/sbatch " \
               f"--output=./slurm_logs/log_job_{obj_name}.out " \
               f"--job-name=job_{obj_name} " \
               f"--nodelist=node{node_id} " \
               f"--wrap 'singularity exec {img_f_path}/{img_f_name} " \
               f"{path_eval_indiv} {eval_indiv_params}'"

        print(cmd_)

        subprocess.run(cmd_, shell=True)
        print(f"start_slurm_proc fc: {idx_indiv}th slurm process ... STARR")








    # ------------------    EDDIG ÁTNÉZVE 2025.03.27.   ------------------











    #TODO: új adatminőség függvényében ezt át kell nézni! Lehet, hogy sokkal egyszerűbb lesz.

    def get_most_important_features(self, num_features: int):

        avg_idx_value = {}
        # i = egyed index, hive = 25,26,27, key = LR/DT
        # self.feature_importance[(i, hive, key)] = _data['all_data_importance']

        # Minden egyedre és minden kromoszómára külön vesszük a 3 DT/LR forrás alapján a fontosságok átlagát
        #         feature_importance[(0, 25, 'LR')] = np.array([0.1,0.2,0.3,0])
        #         feature_importance[(0, 26, 'LR')] = np.array([0.2,0.2,0,0.3])
        #         feature_importance[(0, 27, 'LR')] = np.array([0.3,0.2,0,0.3]) -->

        #       (0, 0, 'LR'): 0.2; (0, 1, 'LR'): 0.2; (0, 2, 'LR'): 0.1; (0, 3, 'LR'): 0.2

        for i in range(self.size_of_population):  # for all individuals
            for j in range(self.length_of_chromosome):  # for all genes
                avg_idx_value[(i, j, 'DT')] = round(np.average(
                    [v[j] for k, v in self.feature_importance.items() if i == k[0] and 'DT' == k[2]]), 2)

                avg_idx_value[(i, j, 'LR')] = round(np.average(
                    [v[j] for k, v in self.feature_importance.items() if i == k[0] and 'LR' == k[2]]), 2)

        # minden egyedre vesszük a feature-ök sorrendjét külön DT/LR szerint a korábban kisz. átlagok alapján
        # (0, 0, 'LR'): 0.2; (0, 1, 'LR'): 0.2; (0, 2, 'LR'): 0.1; (0, 3, 'LR'): 0.2 -->
        #  (0, 'LR'): array([0, 1, 3, 2]) FONTOS!!! CSökkenő sorrend van itt (NAGY -> KICSI)!

        DT_indiv_feat_order = {}
        LR_indiv_feat_order = {}

        for i in range(self.size_of_population):
            s = [-v for k, v in avg_idx_value.items() if i == k[0] and 'DT' == k[2]]
            DT_indiv_feat_order[(i, 'DT')] = np.argsort(np.argsort(s, axis=0), axis=0)

            z = [-v for k, v in avg_idx_value.items() if i == k[0] and 'LR' == k[2]]
            LR_indiv_feat_order[(i, 'LR')] = np.argsort(np.argsort(z, axis=0), axis=0)

        # minden egyedre vesszük a DT és az LR alapján a sorrendekből vett egyedszintű átlagos feature sorrendet
        # csökkenő sorrendek átlaga, átlagos csökkenő sorrend of features to each egyed
        # {(0, 'DT'): array([0, 3, 2, 1]);  (1, 'DT'): array([1, 2, 0, 3])}
        # {(0, 'LR'): array([0, 1, 3, 2]);  (1, 'LR'): array([1, 2, 0, 3])} -->
        # [                 [0. 2. 2.5 1.5]                  [1. 2. 0. 3. ]]

        avg_pop_feat = np.zeros((self.size_of_population, self.length_of_chromosome))

        for j in range(self.size_of_population):
            for i in range(self.length_of_chromosome):
                avg_pop_feat[j, i] = np.average(
                    [DT_indiv_feat_order[(j, 'DT')][i],
                     LR_indiv_feat_order[(j, 'LR')][i]]
                )

        # az egyedek feature sorrendje alapján képezzük a populáció szintű feature sorrendet
        # az egyedek sorrendjeinek átlaga alapján
        # átlagos csökkenő sorrend of features to population

        #  [0. 2. 2.5 1.5] [1. 2. 0. 3. ] --> [0.5  2.   1.25 2.25] <-- [0 2 1 3]

        avg_pop = np.zeros(self.length_of_chromosome)

        for j in range(self.length_of_chromosome):
            avg_pop = -np.average(avg_pop_feat, axis=0)

        # [0.5  2.   1.25 2.25] < -- [0 2 1 3]
        # a feature-ök sorrendje egy csökkenő rendszerben populáció szinten
        # res-be a 0. sorszámú index kell (0), majd az 1. sorszámú (2), ...

        # feature_values_by_indiv = {}

        # for idx in range(self.size_of_population):         # for all individual
        #    for hive in hive_ids:

        # dt_tree_level_info = self.create_level_info(self.DTs[(idx,hive)].tree_.children_left,
        #                                         self.DTs[(idx,hive)].tree_.children_right)

        # dt_feature_value = self.calc_feature_values(dt_tree_level_info,
        #                                            self.DTs[(idx,hive)].tree_.feature)

        # feature_values_by_indiv[(idx,hive)] = self.tr_fet_values_to_chr_values(dt_feature_value,idx)

        # order_features = self.calc_order(feature_values_by_indiv)

        return np.argsort(avg_pop)[:num_features]


    #TODO: bemenő paramétereket megírni hozzá!!!
    #TODO: tr_hive_id és ts_hive_id felhasználás megírása
    def eval_population(self, num_gen: int, tr_hive_ids: np.ndarray, ts_hive_ids:np.ndarray):



        SICMD = "singularity exec --bind /$HOME/Key-Audio-Feature:/mnt /singularity/21_Peter/kaf.simg"
        SCRIPT = "conda run -n KAF python3 Executables/Eval_Individual.py"
        node_cnt = 10
        img_f_name = "key_audio.simg"  # "bee_project_2.simg"
        img_f_path = "/singularity/09-daniel-bee_project"
        path_eval_indiv = "/home/daniel.varkonyi/KEY_AUDIO/Executables/Eval_Individual.py"
        log_dir = "../DATA/LOG"
        node_idx = 2  # 1,2,3,6-nál hogyan állítjuk be???
        metric = "MSE"
        log_pattern = f"hive_*_gen_*_indiv_*.joblib"
        active_processes = []
        term = 120
        # nth generation processing

        # hive_id: 25,26,27
        # indiv idx: 1, ... , 500

        # for hive in hive_ids:
        print(f"hive ids: {hive_ids}")
        for hive in [26]:
            for idx_indiv in range(2):
                # for idx_indiv in range(self.size_of_population):
                if self.local_test:
                    runjob_sh_params = ["--num_gen", str(num_gen), "--indiv_index", str(idx_indiv), "--hive_id",
                                        str(hive)]
                    # current_directory = os.getcwd()

                    # Print the current working directory
                    # print("Current Working Directory:", current_directory)
                    # TODO WINDOWS:
                    subprocess.Popen(
                        ["../venv/Scripts/python.exe", "../Exacutables/Eval_Individual.py"] + runjob_sh_params)
                    # TODO UNIX (?):
                    # subprocess.Popen(["../venv/bin/python", "../Exacutables/Eval_Individual.py"]+runjob_sh_params)
                    print("-->", hive, idx_indiv, "started...")

                else:
                    print("else case: multi slurm process start")
                    p = Process(target=self.start_slurm_proc,
                                args=(num_gen, idx_indiv, node_idx,
                                      img_f_path, img_f_name, path_eval_indiv,
                                      tr_hive_ids,ts_hive_ids,))

                    p.start()
                    active_processes.append(p)
                    for proc in active_processes:
                        proc.join()

                    # job_file_name=f"run_scripts/run_ei_{num_gen}_{idx_indiv}.sh"
                    # runjob_sh_params = ["--num_gen", str(num_gen), "--indiv_index", str(idx_indiv), "--hive_id",
                    #                     str(hive)]
                    #
                    # content=f"""#!/bin/bash
                    # {SICMD} {SCRIPT} {' '.join(runjob_sh_params)}
                    # """
                    # # Open a file in write mode
                    # with open(job_file_name, "w") as file:
                    #     # Use print to write the string to the file
                    #     print(content, file=file)
                    # # current_directory = os.getcwd()
                    #
                    # # Print the current working directory
                    # # print("Current Working Directory:", current_directory)
                    # subprocess.Popen(
                    #     f"sbatch --nodelist=node4 -vv {job_file_name}")
                    #
                    # '''
                    # p = Process(target=self.start_slurm_proc(),
                    #             args=(self,num_gen,idx_indiv,node_idx,
                    #                   img_f_path,img_f_name,path_run_job_sh,hive))
                    #
                    # p.start()
                    # active_processes.append(p)
                    # for proc in active_processes:
                    #     proc.join()
                    # '''
                # Execution waits here until last's start


        # Start slurm process --> runjob.sh --> Exacutables/Eval_Individual.py 4 params
        # TODO: Peti
        # TODO: teszt szekvenciális eval individula hívás 3-szor egymás után a práhuzamosság helyett+
        # TODO: akkor jó az ellenőrzés, ha a futás legvégén generálódik le a logfile,
        # TODO: amiket a self.n_finished_slurm_jobs fv megszámo!!!!


        if self.local_test:
            term = 10
        while (self.size_of_population * len(hive_ids) > self.n_finished_slurm_jobs(log_dir, log_pattern)):
            print(f"main process: num of finished jobs: {self.n_finished_slurm_jobs(log_dir, log_pattern)}")
            print("main process: wait for subprocess to be completted...")
            sleep(term)

        print("main process:all subjobs finished, progress continue!")
        # Read all data from files created by Eval_Individual.pys
        for hive in hive_ids:
            for i in range(self.size_of_population):  # far all individual
                file = joblib.load(f"../DATA/LOG/hive_{hive}_gen_{num_gen}_indiv_{i}.joblib")

                for key in file.keys():  # LR / DTR / SVR
                    _data = file.get(key)  # _data = lr_stats / dtr_stats / svr_stats

                    # save MSEs to GA object
                    self.results[(i, hive, key)] = _data['"MSE"']

                    # Save feature importance to GA object
                    if not key == 'SVR':
                        self.feature_importance[(i, hive, key)] = _data['all_data_importance']

                # TESTED 2025.02.07. - VD
                # TODO: fitness értéknek növekedőnek kelle lennie, ez akkor jó ha csökken,
                #  ellentmondást feloldani 1/x-szel , vaahogy?
                # TODO: jó helyen van ez?egyek kiljebb?
                for i in range(self.size_of_population):
                    self.fitness_values[i] = \
                        np.median([
                            np.average([v for k, v in self.results.items() if i == k[0] and 'SVR' == k[2]]),
                            np.average([v for k, v in self.results.items() if i == k[0] and 'LR' == k[2]]),
                            np.average([v for k, v in self.results.items() if i == k[0] and 'DTR' == k[2]])
                        ])

                # for key in file.keys():                                     # for all keys
                #   if key == "DT_model":                                   # if key is dt
                #       self.DTs[(i,hive)] = file.get(key)                  # save dt object
                #   else:                                                   # else: key --> result
                #       _res = file.get(key)[metric]
                #       if _res > self.fitness_values[i]:                   # save max result
                #           self.fitness_values[i] = _res                   # save max fitness
                #           self.max_reg[i] = key                           # save best regressor
