import numpy as np
import torch
import copy
from typing import Any, Tuple
import time
from .learnable_optimizer import Learnable_Optimizer

def DE_mutation(populations):
    """
    # Introduction
    Performs the mutation operation for Differential Evolution (DE) on a population of candidate solutions.
    # Args:
    - populations (np.ndarray): A 2D numpy array of shape (population_cnt, dim), where each row represents an individual in the population.
    # Returns:
    - np.ndarray: A 2D numpy array of shape (population_cnt, dim) containing the mutated individuals (mutants).
    # Notes:
    - The mutation is performed by selecting three distinct individuals (other than the current one) and combining them according to the DE/rand/1 scheme.
    - The resulting mutant vectors are clipped to the range [0, 1].
    """
    
    # input: pupulations [population_cnt, dim]
    # output: mutants [population_cnt, dim]
    F = 0.5
    pop_cnt, dim = populations.shape
    mutants = copy.deepcopy(populations)
    for j in range(pop_cnt):
        r1 = np.random.randint(low=0, high=pop_cnt)
        r2 = np.random.randint(low=0, high=pop_cnt)
        r3 = np.random.randint(low=0, high=pop_cnt)
        while r1 == j:
            r1 = np.random.randint(low=0, high=pop_cnt)
        while r2 == r1 or r2 == j:
            r2 = np.random.randint(low=0, high=pop_cnt)
        while r3 == r2 or r3 == r1 or r3 == j:
            r3 = np.random.randint(low=0, high=pop_cnt)

        x1 = populations[r1]
        x2 = populations[r2]
        x3 = populations[r3]
        mutant = x1 + F * (x2 - x3)
        mutant = np.clip(mutant, a_min=0, a_max=1)
        mutants[j] = mutant

    return mutants

def DE_crossover(mutants, populations):
    """
    # Introduction
    Performs the crossover operation in Differential Evolution (DE) by combining mutant and population vectors to produce trial vectors.
    # Args:
    - mutants (np.ndarray): Array of mutant vectors with shape (population_cnt, dim).
    - populations (np.ndarray): Array of current population vectors with shape (population_cnt, dim).
    # Returns:
    - np.ndarray: Array of trial vectors after crossover, with the same shape as `mutants` and `populations`.
    # Raises:
    - ValueError: If the input arrays do not have the expected shape or are incompatible for crossover.
    """
    
    CR = 0.7
    U = copy.deepcopy(mutants)
    try:
        population_cnt, dim = mutants.shape
    except ValueError as e:
        print("ValueError occurred:", e)
        print('mutant_shape',mutants.shape)

    for j in range(population_cnt):
        rand_pos = np.random.randint(low=0, high=dim)
        for k in range(dim):
            mutant = mutants[j]
            rand = np.random.rand()
            if rand <= CR or k == rand_pos:
                U[j][k] = mutant[k]

            if rand > CR and k != rand_pos:
                U[j][k] = populations[j][k]
    return U

def DE_rand_1(populations):
    """
    # Introduction
    Applies the DE/rand/1 strategy from Differential Evolution to a population, generating new candidate solutions (offsprings) through mutation and crossover operations.
    # Args:
    - populations (np.ndarray): The current population of candidate solutions, typically represented as a 2D NumPy array where each row is an individual.
    # Returns:
    - np.ndarray: The new population (offsprings) generated after mutation and crossover.
    # Raises:
    - None
    """
    
    mutants = DE_mutation(populations)
    DE_offsprings = DE_crossover(mutants, populations)
    return DE_offsprings


def mixed_DE(populations, source_pupulations, KT_index, action_2, action_3):
    """
    # Introduction
    Performs a mixed Differential Evolution (DE) mutation and crossover operation on given populations, generating a new set of candidate solutions (mutants) based on the provided actions and source populations.
    # Args:
    - populations (np.ndarray): Array of target populations, where each row represents an individual solution.
    - source_pupulations (np.ndarray): Array of source populations used for mutation, with the same shape as populations.
    - KT_index (int): Index specifying which population in `populations` is the target for mutation and crossover.
    - action_2 (float): Mixing coefficient controlling the contribution of target and source populations in mutation.
    - action_3 (float): Mixing coefficient controlling the contribution of different mutation strategies.
    # Returns:
    - np.ndarray: Array of new candidate solutions (mutants) generated after mutation and crossover.
    # Raises:
    - ValueError: If the number of available individuals in `source_pupulations` is less than 6, as unique selection is required.
    """
    
    population_target = populations[KT_index]
    pop_cnt, dim = source_pupulations.shape
    mutants = []
    F = 0.5
    for i in range(population_target.shape[0]):
        r1, r2, r3, r4, r5, r6 = np.random.choice(np.arange(pop_cnt),size=6, replace=False)
        X_r1 = populations[r1]
        X_r2 = source_pupulations[r2]
        X_r3 = populations[r3]
        X_r4 = populations[r4]
        X_r5 = source_pupulations[r5]
        X_r6 = source_pupulations[r6]

        mutant = (1 - action_2) * X_r1 + action_2 * X_r2 + F * (1 - action_3) * (X_r3 - X_r4) + F * action_3 * (
                    X_r5 - X_r6)

        mutants.append(mutant)

    mutants = np.array(mutants)
    U = DE_crossover(mutants, population_target)

    return U

class L2T_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    L2T_Optimizer is a learnable optimizer designed for multi-task optimization problems. It leverages evolutionary strategies and knowledge transfer mechanisms to optimize multiple tasks simultaneously. The optimizer maintains separate populations for each task and applies both standard and knowledge transfer-based differential evolution operations to generate new candidate solutions. It tracks various statistics such as stagnation, improvement flags, and rewards to guide the optimization process.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with configuration parameters, sets up task-specific attributes, and allocates memory for tracking optimization progress and statistics.
        # Args:
        - config (object): Config object containing problem settings.
            - Attributes needed for the L2T_Optimizer are the following:
                - train_problem (str): The training problem to be used.
                - test_problem (str): The testing problem to be used.
                - dim (int): Dimensionality of the optimization problem.
                - log_interval (int): Interval for logging progress.
                - n_logpoint (int): Number of log points to record.
                - full_meta_data (bool): Flag indicating whether to use full meta data.
                - device (str): Device to use for computations (e.g., "cpu", "cuda").
        # Built-in Attributes:
        - __config (object): Configuration object containing algorithm parameters.
        - task_cnt (int): Number of tasks to be optimized.Decided based on the problem type.
        - dim (int): Dimensionality of the optimization problem.Default is 50.
        - generation (int): Current generation count.Default is 0.
        - pop_cnt (int): Population size for each task.Default is 50.
        - total_generation (int): Total number of generations for the optimization process.
        - flag_improved (np.ndarray): Array to track improvement flags for each task.
        - stagnation (np.ndarray): Array to track stagnation counts for each task.
        - old_action_1 (np.ndarray): Array to store the last action taken for each task.
        - old_action_2 (np.ndarray): Array to store the last action taken for each task.
        - old_action_3 (np.ndarray): Array to store the last action taken for each task.
        - N_kt (np.ndarray): Array to track the number of knowledge transfer operations for each task.Decided based on the problem type.
        - Q_kt (np.ndarray): Array to track the quality of knowledge transfer for each task.
        - gbest (np.ndarray): Array to store the best fitness values for each task.
        - task (Any): Placeholder for the current task being optimized.Default is None.
        - offsprings (np.ndarray): Array to store the generated offsprings for each task.
        - noKT_offsprings (np.ndarray): Array to store the generated offsprings without knowledge transfer for each task.
        - KT_offsprings (list): List to store the generated offsprings with knowledge transfer for each task.
        - KT_index (list): List to store the indices of individuals selected for knowledge transfer for each task.
        - parent_population (np.ndarray): Array to store the current population for each task.
        - reward (list): List to store the rewards for each task.
        - total_reward (float): Total accumulated reward across all tasks.
        - begin_best (list): List to store the best fitness values at the beginning of the optimization for each task.
        - last_gen_best (list): List to store the best fitness values from the last generation for each task.
        - this_gen_best (list): List to store the best fitness values from the current generation for each task.
        - optimal_value (list): List to store the optimal values for each task.
        - fes (int): Counter for the number of function evaluations.Default is None.
        - cost (list): List to store the best cost values during optimization.Default is None.
        - log_index (int): Index for logging progress.Default is None.
        # Returns:
        - None
        # Raises:
        - None
        """
        
        super().__init__(config)
        self.__config = config

        if config.train_problem == 'wcci2020':
            self.task_cnt = 50
        if config.train_problem == 'cec2017mto':
            self.task_cnt = 2
        if config.test_problem == 'wcci2020':
            self.task_cnt = 50
        if config.test_problem == 'cec2017mto':
            self.task_cnt = 2
        if config.train_problem == 'augmented-wcci2020':
            self.task_cnt = 10
        if config.test_problem == 'augmented-wcci2020':
            self.task_cnt = 10
        self.dim = config.dim = 50
        self.generation = 0
        self.pop_cnt = 50
        self.total_generation = 250

        self.flag_improved = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.stagnation = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32) 
        self.old_action_1 = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.old_action_2 = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.old_action_3 = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.N_kt = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)
        self.Q_kt = np.array([0 for _ in range(self.task_cnt)], dtype=np.float32)

        self.gbest = np.array([1e+32 for _ in range(self.task_cnt)],dtype=np.float32)
        self.task = None
        self.offsprings = np.array([[np.random.rand(self.dim) for i in range(self.pop_cnt)] for _ in range(self.task_cnt)])
        self.noKT_offsprings = np.array([[np.random.rand(self.dim) for i in range(self.pop_cnt)] for _ in range(self.task_cnt)])
        self.KT_offsprings = [None for _ in range(self.task_cnt)]
        self.KT_index = [None for _ in range(self.task_cnt)]
        self.parent_population = None
        self.reward = [0 for _ in range(self.task_cnt)]
        self.total_reward = 0

        self.begin_best = [1e+32 for _ in range(self.task_cnt)]
        self.last_gen_best = [1e+32 for _ in range(self.task_cnt)]
        self.this_gen_best = [1e+32 for _ in range(self.task_cnt)]
        self.optimal_value = [1e+32 for _ in range(self.task_cnt)]

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def get_state(self):
        """
        # Introduction
        Computes and returns the current state representation of the optimizer, aggregating various statistics and features for each task.
        # Args:
        None
        # Returns:
        - np.ndarray: A 1D array of type float32 containing the concatenated state features for all tasks, including normalized generation count, stagnation, improvement flags, Q-values, population statistics, and previous actions.
        # Raises:
        None
        """
        
        state_o = self.generation / self.total_generation
        states = np.array([state_o], dtype=np.float32)

        for i in range(self.task_cnt):
            states_t = []
            state_1 = self.stagnation[i] / self.total_generation
            states_t.append(state_1)
            state_2 = self.flag_improved[i]
            states_t.append(state_2)
            if self.N_kt[i] == 0:
                state_3 = 0
            else:
                state_3 = self.Q_kt[i]

            states_t.append(state_3)
            state_4 = np.mean(np.std(self.parent_population[i], axis=-1))
            states_t.append(state_4)
            state_5 = self.old_action_1[i]
            state_6 = self.old_action_2[i]
            state_7 = self.old_action_3[i]

            states_t.append(state_5)
            states_t.append(state_6)
            states_t.append(state_7)

            states_t = np.array(states_t, dtype=np.float32)
            states = np.concatenate((states, states_t),axis=-1)
        self.generation += 1

        return states

    def init_population(self, tasks):
        """
        # Introduction
        Initializes the population for a multi-task optimization process, evaluates their fitness, and prepares meta-data if required.
        # Args:
        - tasks (list): A list of task objects, each providing an `eval` method to compute the fitness of a population.
        # Returns:
        - state (Any): The current state of the optimizer after population initialization, as returned by `self.get_state()`.
        # Side Effects:
        - Initializes and updates several instance attributes including `self.fes`, `self.task`, `self.parent_population`, `self.log_index`, `self.gbest`, `self.cost`, `self.meta_X`, and `self.meta_Cost`.
        - Evaluates the fitness of the initial population for each task and stores the results.
        - Optionally stores meta-data if `self.__config.full_meta_data` is set to True.
        """

        self.fes = 0
        self.task = tasks.tasks
        self.parent_population = np.array([[self.rng.rand(self.dim) for i in range(self.pop_cnt)] for _ in range(self.task_cnt)])
        self.log_index = 1
        

        parent_fitnesses_list = []
        for i in range(self.task_cnt):
            fitnesses = self.task[i].eval(self.parent_population[i])
            self.gbest[i] = np.min(fitnesses, axis=-1)
            parent_fitnesses_list.append(fitnesses)

            self.begin_best[i] = self.gbest[i]
            self.last_gen_best[i] = self.gbest[i]
            self.this_gen_best[i] = self.gbest[i]
            self.optimal_value[i] = self.task[i].optimum

        parent_fitnesses_np = np.array(parent_fitnesses_list, dtype=np.float32)
        
        self.cost = [copy.deepcopy(self.gbest)]
        state = self.get_state()


        if self.__config.full_meta_data:
            self.meta_X = [self.parent_population.copy()]
            self.meta_Cost = [parent_fitnesses_np.copy()]
            
        return state

    def self_update(self):
        """
        # Introduction
        Updates the `noKT_offsprings` attribute for each task by generating new offsprings using the `DE_rand_1` differential evolution strategy.
        # Args:
        None
        # Returns:
        None
        # Raises:
        - IndexError: If `self.parent_population` or `self.noKT_offsprings` do not have sufficient elements for the range of `self.task_cnt`.
        """
        
        for i in range(self.task_cnt):
            self.noKT_offsprings[i] = DE_rand_1(self.parent_population[i])
    

    def transfer(self,actions):
        """
        # Introduction
        Transfers knowledge between tasks by generating offsprings using actions and a randomly selected source population. This method applies a mixed differential evolution (DE) strategy to a subset of individuals in each task's population, based on the provided actions.
        # Args:
        - actions (list or array-like): A sequence of three action values [action_1, action_2, action_3] used to control the transfer process and DE parameters.
        # Notes:
        - The method ensures that the source population for transfer is different from the target task.
        - At least one individual is always selected for transfer per task.
        - Uses `mixed_DE` for generating transferred offsprings and `copy.deepcopy` to preserve non-transferred offsprings.
        """
        
        for i in range(self.task_cnt):
            action_1 = actions[0]
            action_2 = actions[1]
            action_3 = actions[2]

            rand_source_index = self.rng.randint(low=0,high=self.task_cnt)
            while rand_source_index == i:
                rand_source_index = self.rng.randint(low=0, high=self.task_cnt)

            source_population = self.parent_population[rand_source_index]

            self.N_kt[i] = 0.5 * action_1
            self.KT_count = int(np.ceil(self.N_kt[i] * self.pop_cnt))
            if self.KT_count == 0:
                self.KT_count = 1
            self.KT_index[i] = self.rng.choice(np.arange(self.pop_cnt), size=self.KT_count, replace=False)

            self.KT_offsprings[i] = mixed_DE(self.parent_population[i], source_population, self.KT_index[i], action_2, action_3)
            self.offsprings[i] = copy.deepcopy(self.noKT_offsprings[i])
            for j in range(self.KT_count):
                self.offsprings[i][self.KT_index[i][j]] = self.KT_offsprings[i][j]

            self.old_action_1[i] = action_1
            self.old_action_2[i] = action_2
            self.old_action_3[i] = action_3

    def seletion(self):
        """
        # Introduction
        Performs the selection operation in an evolutionary optimization process, updating the parent population based on the fitness of offspring and parent individuals. It also updates rewards, quality metrics, and tracks improvements or stagnation for each task.
        # Args:
        None
        # Returns:
        - np.ndarray: A 1D array of type float32 containing the concatenated state features for all tasks, including normalized generation count, stagnation, improvement flags, Q-values, population statistics, and previous actions.
        # Side Effects:
        - Updates `self.parent_population`, `self.reward`, `self.Q_kt`, `self.gbest`, `self.flag_improved`, `self.stagnation`, and meta-data lists (`self.meta_X`, `self.meta_Cost`) as part of the selection process.
        # Notes:
        - Assumes that `self.task`, `self.parent_population`, `self.offsprings`, `self.KT_index`, `self.KT_count`, `self.gbest`, `self.flag_improved`, `self.stagnation`, and `self.__config.full_meta_data` are properly initialized and maintained elsewhere in the class.
        - Uses deep copies to avoid unintended side effects when updating populations.
        """
        
        parent_finesses_list = []
        for i in range(self.task_cnt):
            self.last_gen_best[i] = self.this_gen_best[i]
            ps = self.parent_population[i].shape[0]
            self.fes += ps
            parent_population_fitness = self.task[i].eval(self.parent_population[i])
            offsprings_population_fitness = self.task[i].eval(self.offsprings[i])

            next_population = copy.deepcopy(self.parent_population)
           
            S_update = 0
            S_KT = 0
            for j in range(self.pop_cnt):
                if offsprings_population_fitness[j] <= parent_population_fitness[j]:
                    if j not in self.KT_index[i]:
                        S_update += 1
                    else:
                        S_KT += 1

                    next_population[i][j] = self.offsprings[i][j]
                else:
                    next_population[i][j] = self.parent_population[i][j]

            reward_kt = (float)(S_update-S_KT) / self.pop_cnt
            self.Q_kt[i] = float(S_KT) / self.KT_count

            flag = 0
            fitnesses = self.task[i].eval(next_population[i])
            parent_finesses_list.append(fitnesses)
            best_fitness = np.min(fitnesses,axis=-1)
            if(best_fitness < self.gbest[i]):
                self.gbest[i] = best_fitness
                flag = 1

            if(flag):
                self.flag_improved[i] = 1
            else:
                self.flag_improved[i] = 0
                self.stagnation[i] += 1

            self.this_gen_best[i] = self.gbest[i]
            self.parent_population[i] = next_population[i]

            reward_indicators = 1 if (self.this_gen_best[i] - self.optimal_value[i])<1e-8 else 0
            reward_converge = (self.this_gen_best[i] - self.optimal_value[i]) / (self.begin_best[i] - self.optimal_value[i])
            self.reward[i] = 1*reward_converge + 10*reward_kt + 250*reward_indicators

        parent_finesses_np = np.array(parent_finesses_list, dtype=np.float32)
        if self.__config.full_meta_data:
            self.meta_X.append(self.parent_population.copy())
            self.meta_Cost.append(parent_finesses_np.copy())
        return self.get_state()

    def update(self, actions, tasks):
        """
        # Introduction
        Updates the optimizer's state based on the provided actions and tasks, manages reward accumulation, logging, and determines if the optimization process has ended.
        # Args:
        - actions (Any): Actions to be applied in the current update step.
        - tasks (Any): Tasks relevant to the current optimization step.
        # Returns:
        - next_state (Any): The next state after applying the actions.
        - total_reward (float): The accumulated reward after the update.
        - is_end (bool): Flag indicating whether the optimization process has ended.
        - info (dict): Additional information (currently empty).
        # Raises:
        - None
        """
        
        self.self_update()
        self.transfer(actions)
        next_state = self.seletion()

        for _ in range(self.task_cnt):
            self.total_reward += self.reward[_]

        is_end = False
        if self.generation > self.total_generation:
            is_end = True
        
        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(copy.deepcopy(self.gbest))

        if is_end:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = copy.deepcopy(self.gbest)
            else: 
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(copy.deepcopy(self.gbest))
        
        if is_end:
            tasks.update_T1()
        
        info = {}
        return next_state, self.total_reward, is_end, info
