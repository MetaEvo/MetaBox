import numpy as np
import warnings
import copy
from .learnable_optimizer import Learnable_Optimizer
import time
import scipy.stats as stats


class RLDAS_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    RLDAS is a deep reinforcement learning-based dynamic algorithm selection framework.
    #Original paper
    "[**Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution**](https://ieeexplore.ieee.org/abstract/document/10496708/)." IEEE Transactions on Systems, Man, and Cybernetics: Systems (2024).
    # Official Implementation
    [RL-DAS](https://github.com/GMC-DRL/RL-DAS)
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration, setting up key parameters such as maximum function evaluations, logging intervals, and problem-specific periods.
        # Args:
        - config (object): Config object containing optimizer settings.
            - The Attributes needed for the RLDAS_Optimizer:
                - maxFEs (int): Maximum number of function evaluations allowed.
                - train_problem (str): The training problem name.
                - test_problem (str): The testing problem name.
                - log_interval (int): Interval for logging, set in the config.
                - full_meta_data (bool): Flag to indicate if full meta data should be collected.
                - n_logpoint (int): Number of log points to collect.
        # Built-in Attribute:
        - MaxFEs (int): Maximum number of function evaluations allowed.
        - period (int): Evaluation period, set based on the problem type.
        - max_step (int): Maximum number of steps, calculated as MaxFEs divided by period.
        - sample_times (int): Number of times to sample observations.
        - n_dim_obs (int): Number of dimensions in the observation space.
        - final_obs (Any): Stores the final observation (initialized as None).
        - terminal_error (float): Threshold for terminal error.
        - __config (object): Stores the configuration object.
        - FEs (Any): Tracks function evaluations (initialized as None).
        - cost (Any): Tracks cost values (initialized as None).
        - log_index (Any): Tracks logging index (initialized as None).
        - log_interval (int): Interval for logging, taken from config.
        # Returns:
        - None
        # Raises:
        - AttributeError: If required attributes are missing from the config object.
        """
        
        super().__init__(config)
        self.MaxFEs = config.maxFEs
        # self.period = 2500

        if 'protein' in config.train_problem or 'protein' in config.test_problem:
            self.period = 100
        else:
            self.period = 2500
        self.max_step = self.MaxFEs // self.period
        self.sample_times = 2
        self.n_dim_obs = 6

        self.final_obs = None
        self.terminal_error = 1e-8

        self.__config = config

        self.FEs = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        """
        Returns a string representation of the RLDAS_Optimizer class.
        # Returns:
            str: The name of the optimizer, "RLDAS_Optimizer".
        """
        
        return "RLDAS_Optimizer"

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and associated optimizers for the given optimization problem. Sets up tracking for best and worst solutions, initializes the population, and prepares metadata if required.
        # Args:
        - problem (object): The optimization problem object.
        # Built-in Attribute:
        - `self.dim` (int): Dimensionality of the problem.
        - `self.problem` (object): Reference to the optimization problem.
        - `self.optimizers` (list): List of optimizer instances used for the population.Default is ['NL_SHADE_RSP', 'MadDE', 'JDE21'].
        - `self.best_history` (list): Tracks the best solution history for each optimizer.
        - `self.worst_history` (list): Tracks the worst solution history for each optimizer.
        - `self.population` (Population): The initialized population object.
        - `self.cost_scale_factor` (float): The best cost found in the initial population.
        - `self.FEs` (int): Number of function evaluations used so far.
        - `self.done` (bool): Flag indicating if the optimization is complete.
        - `self.cost` (list): List of best costs found so far.
        - `self.log_index` (int): Index for logging progress.
        - `self.meta_X` (list, optional): Metadata for population groups (if enabled).
        - `self.meta_Cost` (list, optional): Metadata for population costs (if enabled).
        # Returns:
        - observation (object): The result of the `observe` method, representing the current state or observation of the problem.
        # Raises:
        - AttributeError: If the `problem` object does not have the required attributes.
        - Exception: Propagates any exceptions raised during population or optimizer initialization.
        """
        
        self.dim = problem.dim
        self.problem = problem

        optimizers = ['NL_SHADE_RSP', 'MadDE', 'JDE21']
        self.optimizers = []
        for optimizer in optimizers:
            self.optimizers.append(eval(optimizer)(self.dim, self.rng))
        self.best_history = [[] for _ in range(len(optimizers))]
        self.worst_history = [[] for _ in range(len(optimizers))]

        self.population = Population(self.dim, self.rng)
        self.population.initialize_costs(self.problem)
        self.cost_scale_factor = self.population.gbest
        self.FEs = self.population.NP
        self.done = False
        self.cost = [self.population.gbest]
        self.log_index = 1

        if self.__config.full_meta_data:
            self.meta_X = [self.population.group]
            self.meta_Cost = [self.population.cost]

        return self.observe(problem)

    def local_sample(self):
        """
        # Introduction
        Generates multiple local samples using different optimizers, collects their costs, and returns the samples and their associated costs as NumPy arrays. Ensures all cost arrays are truncated to the minimum length among them for consistency.
        # Args:
        None
        # Returns:
        - samples: Array of sampled populations generated by the optimizers.
        - costs: Array of cost values associated with each sample, truncated to the minimum length.
        # Raises:
        None
        # Notes:
        - Updates the number of function evaluations (`self.FEs`) and sets `self.done` to True if the maximum number of evaluations (`self.MaxFEs`) is reached.
        - Assumes that each optimizer's `step` method returns a sample with a `cost` attribute.
        """
        
        samples = []
        costs = []
        min_len = 1e9
        sample_size = self.population.NP
        for i in range(self.sample_times):
            sample, _ = self.optimizers[self.rng.randint(len(self.optimizers))].step(copy.deepcopy(self.population),
                                                                                     self.problem,
                                                                                     self.FEs,
                                                                                     self.FEs + sample_size,
                                                                                     self.MaxFEs)
            samples.append(sample)
            cost = sample.cost
            costs.append(cost)
            min_len = min(min_len, cost.shape[0])
        self.FEs += sample_size * self.sample_times
        if self.FEs >= self.MaxFEs:
            self.done = True
        for i in range(self.sample_times):
            costs[i] = costs[i][:min_len]
        return np.array(samples), np.array(costs)

    # observed env state
    def observe(self, problem):
        """
        # Introduction
        Observes the current state of the optimization process by collecting features, best and worst moves from optimizer histories, and returns a structured array representing these observations.
        # Args:
        - problem: The optimization problem instance being solved.
        # Returns:
        - np.ndarray: An array (dtype=object) containing the extracted feature and the mean best/worst moves for each optimizer.
        # Notes:
        - The first element of the returned array is the feature vector extracted from the current population and sample costs.
        - Subsequent elements represent the mean of best and worst moves for each optimizer, if available.
        """

        samples, sample_costs = self.local_sample()
        feature = self.population.get_feature(self.problem,
                                              sample_costs,
                                              self.cost_scale_factor,
                                              self.FEs / self.MaxFEs)

        # =======================================================================
        best_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        worst_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        move = np.zeros((len(self.optimizers) * 2, self.dim)).tolist()
        for i in range(len(self.optimizers)):
            if len(self.best_history[i]) > 0:
                move[i * 2] = np.mean(self.best_history[i], 0).tolist()
                move[i * 2 + 1] = np.mean(self.worst_history[i], 0).tolist()
                best_move[i] = np.mean(self.best_history[i], 0).tolist()
                worst_move[i] = np.mean(self.worst_history[i], 0).tolist()
        move.insert(0, feature)
        return np.array(move, dtype = object)

    def update(self, action, problem):
        """
        # Introduction
        Executes an optimization step using the selected optimizer, updates the population, tracks progress, and computes the reward for reinforcement learning-based dynamic algorithm selection.
        # Args:
        - action (int): The index of the optimizer to use for this update step.
        - problem (object): The optimization problem instance, which should provide an `optimum` attribute and be compatible with the optimizer's `step` method.
        # Returns:
        - observe (Any): The observation/state after the update, as returned by `self.observe(problem)`.
        - reward (float): The reward computed based on the improvement in the global best cost.
        - done (bool): Whether the optimization process has reached a terminal state.
        - info (dict): An empty dictionary for compatibility with RL environments.
        # Notes:
        - Updates internal histories for best and worst solutions.
        - Handles logging and meta-data collection if enabled in configuration.
        - Suppresses warnings during execution.
        """
        
        warnings.filterwarnings("ignore")
        act = action

        last_cost = self.population.gbest
        pre_best = self.population.gbest_solution
        pre_worst = self.population.group[np.argmax(self.population.cost)]
        period = self.period
        end = self.FEs + self.period
        while self.FEs < end and self.FEs < self.MaxFEs and self.population.gbest > self.terminal_error:
            optimizer = self.optimizers[act]
            FEs_end = self.FEs + period

            self.population, self.FEs = optimizer.step(self.population,
                                                       problem,
                                                       self.FEs,
                                                       FEs_end,
                                                       self.MaxFEs,
                                                       )
        end = time.time()
        pos_best = self.population.gbest_solution
        pos_worst = self.population.group[np.argmax(self.population.cost)]
        self.best_history[act].append((pos_best - pre_best) / 200)
        self.worst_history[act].append((pos_worst - pre_worst) / 200)
        if problem.optimum is None:
            self.done = self.FEs >= self.MaxFEs
        else:
            self.done = self.FEs >= self.MaxFEs 
        # self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)
        reward = max((last_cost - self.population.gbest) / self.cost_scale_factor, 0)

        observe = self.observe(problem)
        if self.FEs >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.population.gbest)

        if self.__config.full_meta_data:
            self.meta_X.append(self.population.group.copy())
            self.meta_Cost.append(self.population.cost.copy())

        if self.done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.population.gbest
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.population.gbest)
        return observe, reward, self.done, {}  # next state, reward, is done


class Population:
    def __init__(self, dim, rng):
        """
        # Introduction
        Initializes the RL-Differential Adaptive Search (RLDAS) optimizer with specified population and algorithm parameters.
        # Args:
        - dim (int): The dimensionality of each individual in the population.
        - rng (np.random.Generator): A NumPy random number generator instance for stochastic operations.
        # Built-in Attributes:
        - `self.Nmax` (int): The upper bound of the population size.Default is 170.
        - `self.Nmin` (int): The lower bound of the population size.Default is 30.
        - `self.NP` (int): The current population size.
        - `self.NA` (int): The size of the archive (collection of replaced individuals).
        - `self.dim` (int): The dimension of individuals.
        - `self.cost` (np.ndarray): The cost values of individuals in the population.
        - `self.cbest` (float): The best cost in the current population.Default is 1e15.
        - `self.cbest_id` (int): The index of the individual with the best cost.Default is -1.
        - `self.gbest` (float): The global best cost found so far.Default is 1e15.
        - `self.gbest_solution` (np.ndarray): The individual with the global best cost.Default is a zero vector of length `dim`.
        - `self.Xmin` (np.ndarray): The lower bounds for individual values.Default is np.ones(dim) * -5.
        - `self.Xmax` (np.ndarray): The upper bounds for individual values.Default is np.ones(dim) * 5.
        - `self.rng` (np.random.Generator): The random number generator used for stochastic operations.
        - `self.group` (np.ndarray): The current population of individuals.
        - `self.archive` (np.ndarray): The archive of replaced individuals.
        - `self.MF` (np.ndarray): The set of step lengths for Differential Evolution (DE).Default is np.ones(dim * 20) * 0.2.
        - `self.MCr` (np.ndarray): The set of crossover rates for DE.Default is np.ones(dim * 20) * 0.2.
        - `self.k` (int): The index for updating elements in MF and MCr.Default is 0.
        - `self.F` (np.ndarray): The set of successful step lengths.Default is np.ones(self.NP) * 0.5.
        - `self.Cr` (np.ndarray): The set of successful crossover rates.Default is np.ones(self.NP) * 0.9.
        # Raises:
        - None
        """
        
        self.Nmax = 170  # the upperbound of population size
        self.Nmin = 30  # the lowerbound of population size
        self.NP = self.Nmax  # the population size
        self.NA = int(self.NP * 2.1)  # the size of archive(collection of replaced individuals)
        self.dim = dim  # the dimension of individuals
        self.cost = np.zeros(self.NP)  # the cost of individuals
        self.cbest = 1e15  # the best cost in current population, initialize as 1e15
        self.cbest_id = -1  # the index of individual with the best cost
        self.gbest = 1e15  # the global best cost
        self.gbest_solution = np.zeros(dim)  # the individual with global best cost
        self.Xmin = np.ones(dim) * -5  # the upperbound of individual value
        self.Xmax = np.ones(dim) * 5  # the lowerbound of individual value

        self.rng = rng

        self.group = self.initialize_group()  # the population
        self.archive = np.array([])  # the archive(collection of replaced individuals)
        self.MF = np.ones(dim * 20) * 0.2  # the set of step length of DE
        self.MCr = np.ones(dim * 20) * 0.2  # the set of crossover rate of DE
        self.k = 0  # the index of updating element in MF and MCr
        self.F = np.ones(self.NP) * 0.5  # the set of successful step length
        self.Cr = np.ones(self.NP) * 0.9  # the set of successful crossover rate

    # generate an initialized population with size(default self population size)
    def initialize_group(self, size = -1):
        """
        # Introduction
        Initializes a group of individuals (solutions) within the defined search space boundaries.
        # Args:
        - size (int, optional): The number of individuals to initialize. If negative, defaults to `self.NP`.
        # Built-in Attribute:
        - `self.NP` (int): Default population size if `size` is not specified.
        - `self.dim` (int): Dimensionality of each individual.
        - `self.Xmin` (float or np.ndarray): Lower bound(s) of the search space.
        - `self.Xmax` (float or np.ndarray): Upper bound(s) of the search space.
        - `self.rng` (np.random.Generator): Random number generator instance.
        # Returns:
        - np.ndarray: An array of shape `(size, self.dim)` containing randomly initialized individuals within `[self.Xmin, self.Xmax]`.
        # Raises:
        - ValueError: If `size` is zero or negative after adjustment.
        """
        
        if size < 0:
            size = self.NP
        return self.rng.random((size, self.dim)) * (self.Xmax - self.Xmin) + self.Xmin

    # initialize cost
    def initialize_costs(self, problem):
        """
        # Introduction
        Initializes the cost values for the optimizer based on the provided problem instance. 
        Computes the cost of the current group of solutions, updates the global and current best costs, 
        and stores the best solution found so far.
        # Args:
        - problem (object): An object representing the optimization problem. It must have an `optimum` attribute 
          and an `eval` method that evaluates the cost of a group of solutions.
        # Built-in Attribute:
        - `self.cost` (np.ndarray or float): The computed cost(s) for the current group.
        - `self.gbest` (float): The global best cost found so far.
        - `self.cbest` (float): The current best cost in the group.
        - `self.cbest_id` (int): The index of the current best solution in the group.
        - `self.gbest_solution` (np.ndarray): The solution vector corresponding to the global best cost.
        # Returns:
        - None
        # Raises:
        - AttributeError: If `problem` does not have the required attributes or methods.
        """
        
        if problem.optimum is not None:
            self.cost = problem.eval(self.group) - problem.optimum
        else:
            self.cost = problem.eval(self.group)
        self.gbest = self.cbest = np.min(self.cost)
        self.cbest_id = np.argmin(self.cost)
        self.gbest_solution = self.group[self.cbest_id]

    def clear_context(self):
        """
        # Introduction
        Resets the optimizer's internal context and parameters to their initial states.
        # Built-in Attribute:
        - self.archive (np.ndarray): Archive for storing replaced individuals, reset to an empty array.
        - self.MF (np.ndarray): Step length set for Differential Evolution (DE), reset to an array of 0.2.
        - self.MCr (np.ndarray): Crossover rate set for DE, reset to an array of 0.2.
        - self.k (int): Index for updating elements in MF and MCr, reset to 0.
        - self.F (np.ndarray): Set of successful step lengths, reset to an array of 0.5.
        - self.Cr (np.ndarray): Set of successful crossover rates, reset to an array of 0.9.
        # Returns:
        - None
        """
        
        self.archive = np.array([])  # the archive(collection of replaced individuals)
        self.MF = np.ones(self.dim * 20) * 0.2  # the set of step length of DE
        self.MCr = np.ones(self.dim * 20) * 0.2  # the set of crossover rate of DE
        self.k = 0  # the index of updating element in MF and MCr
        self.F = np.ones(self.NP) * 0.5  # the set of successful step length
        self.Cr = np.ones(self.NP) * 0.9  # the set of successful crossover rate

    # sort former 'size' population in respect to cost
    def sort(self, size, reverse = False):
        """
        # Introduction
        Sorts the population based on their cost values up to a specified size, optionally in reverse order, and updates related attributes accordingly.
        # Args:
        - size (int): The number of individuals to sort based on their cost.
        - reverse (bool, optional): If True, sorts in descending order of cost. Defaults to False (ascending order).
        # Updates:
        - self.cost: Reordered array of costs after sorting.
        - self.cbest: The minimum cost value after sorting.
        - self.cbest_id: The index of the minimum cost value after sorting.
        - self.group: Reordered group attribute to match the sorted order.
        - self.F: Reordered F attribute to match the sorted order.
        - self.Cr: Reordered Cr attribute to match the sorted order.
        # Returns:
        - None
        """
        
        # new index after sorting
        r = -1 if reverse else 1
        ind = np.concatenate((np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:]))
        self.cost = self.cost[ind]
        self.cbest = np.min(self.cost)
        self.cbest_id = np.argmin(self.cost)
        self.group = self.group[ind]
        self.F = self.F[ind]
        self.Cr = self.Cr[ind]

    # calculate new population size with non-linear population size reduction
    def cal_NP_next_gen(self, FEs, MaxFEs):
        """
        # Introduction
        Calculates the population size (NP) for the next generation in an evolutionary algorithm based on the current number of function evaluations.
        # Args:
        - FEs (int or float): The current number of function evaluations performed.
        - MaxFEs (int or float): The maximum number of function evaluations allowed.
        # Built-in Attribute:
        - self.Nmax (int or float): The maximum population size.
        - self.Nmin (int or float): The minimum population size.
        # Returns:
        - float: The computed population size for the next generation.
        # Raises:
        - AttributeError: If `self.Nmax` or `self.Nmin` are not defined in the class.
        """
        NP = np.round(self.Nmax + (self.Nmin - self.Nmax) * np.power(FEs / MaxFEs, 1 - FEs / MaxFEs))
        return NP

    # slice the population and its cost, crossover rate, etc
    def slice(self, size):
        """
        # Introduction
        Reduces the population size and associated attributes of the optimizer to the specified size.
        # Args:
        - size (int): The new population size to slice to.
        # Modifies:
        - self.NP: Updates to the new population size.
        - self.group: Truncates to the first `size` individuals.
        - self.cost: Truncates to the first `size` cost values.
        - self.F: Truncates to the first `size` F values.
        - self.Cr: Truncates to the first `size` Cr values.
        - self.cbest_id: Updates if the current best index is out of bounds.
        - self.cbest: Updates if the current best index is out of bounds.
        # Raises:
        - None
        """
        self.NP = size
        self.group = self.group[:size]
        self.cost = self.cost[:size]
        self.F = self.F[:size]
        self.Cr = self.Cr[:size]
        if self.cbest_id >= size:
            self.cbest_id = np.argmin(self.cost)
            self.cbest = np.min(self.cost)

    # reduce population in JDE way
    def reduction(self, bNP):
        """
        # Introduction
        Reduces the population size and associated attributes by removing the middle portion of the arrays, keeping only the first half and the last segment.
        # Args:
        - bNP (int): The current population size before reduction.
        # Built-in Attribute:
        - self.group (np.ndarray): The population group array to be reduced.
        - self.F (np.ndarray): The scaling factor array to be reduced.
        - self.Cr (np.ndarray): The crossover rate array to be reduced.
        - self.cost (np.ndarray): The cost array to be reduced.
        - self.NP (int): The updated population size after reduction.
        # Returns:
        - None
        # Raises:
        - None
        """

        self.group = np.concatenate((self.group[:bNP // 2], self.group[bNP:]), 0)
        self.F = np.concatenate((self.F[:bNP // 2], self.F[bNP:]), 0)
        self.Cr = np.concatenate((self.Cr[:bNP // 2], self.Cr[bNP:]), 0)
        self.cost = np.concatenate((self.cost[:bNP // 2], self.cost[bNP:]), 0)
        self.NP = bNP // 2 + 10

    # calculate wL mean
    def mean_wL(self, df, s):
        """
        # Introduction
        Computes the weighted mean of the squared values in `s` using the normalized weights from `df`.
        If the weighted sum of `s` is very small (less than or equal to 1e-6), returns 0.5 as a fallback.
        # Args:
        - df (np.ndarray or pd.Series): Array or series of values to be used as weights.
        - s (np.ndarray or pd.Series): Array or series of values whose weighted mean of squares is to be computed.
        # Returns:
        - float: The weighted mean of the squared values in `s`, or 0.5 if the weighted sum is too small.
        # Notes:
        - The weights are normalized so that their sum is 1.
        - If the denominator in the weighted mean calculation is very small, a default value of 0.5 is returned to avoid instability.
        """
        
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # randomly choose step length nad crossover rate from MF and MCr
    def choose_F_Cr(self):
        """
        # Introduction
        Generates crossover rates (Cr) and scaling factors (F) for a population in a differential evolution optimizer, using normal and Cauchy distributions respectively.
        # Args:
        None
        # Returns:
        - C_r (np.ndarray): Array of crossover rates for the population, clipped to [0, 1].
        - F (np.ndarray): Array of scaling factors for the population, with negative values reflected and clipped to a maximum of 1.
        # Notes:
        - Crossover rates are sampled from a normal distribution centered at memory values `MCr`.
        - Scaling factors are sampled from a Cauchy distribution centered at memory values `MF`, with negative values reflected to ensure non-negativity.
        """
        
        # generate Cr can be done simutaneously
        gs = self.NP
        ind_r = self.rng.randint(0, self.MF.shape[0], size = gs)  # index
        C_r = np.minimum(1, np.maximum(0, self.rng.normal(loc = self.MCr[ind_r], scale = 0.1, size = gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self.MF[ind_r]
        F = stats.cauchy.rvs(loc = cauchy_locs, scale = 0.1, size = gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        # F = []
        # for i in range(gs):
        #     while True:
        #         f = stats.cauchy.rvs(loc=cauchy_locs[i], scale=0.1)
        #         if f >= 0:
        #             F.append(f)
        #             break
        return C_r, np.minimum(1, F)

    # update MF and MCr, join new value into the set if there are some successful changes or set it to initial value
    def update_M_F_Cr(self, SF, SCr, df):
        """
        # Introduction
        Updates the memory arrays `MF` and `MCr` with new mean values based on successful F and Cr values, or resets them if no successful values are present.
        # Args:
        - SF (np.ndarray): Array of successful F values from the current generation.
        - SCr (np.ndarray): Array of successful Cr values from the current generation.
        - df (np.ndarray): Array of weights or fitness differences associated with the successful individuals.
        # Built-in Attribute:
        - self.MF (np.ndarray): Memory array for F values.
        - self.MCr (np.ndarray): Memory array for Cr values.
        - self.k (int): Current index for updating memory arrays.
        # Returns:
        - None
        # Notes:
        - If there are no successful individuals (`SF.shape[0] == 0`), the memory arrays are reset to 0.5 at the current index.
        """
        
        if SF.shape[0] > 0:
            mean_wL = self.mean_wL(df, SF)
            self.MF[self.k] = mean_wL
            mean_wL = self.mean_wL(df, SCr)
            self.MCr[self.k] = mean_wL
            self.k = (self.k + 1) % self.MF.shape[0]
        else:
            self.MF[self.k] = 0.5
            self.MCr[self.k] = 0.5

    # non-linearly reduce population size and update it into new population
    def NLPSR(self, FEs, MaxFEs):
        """
        # Introduction
        Adjusts the population size and archive size for the next generation in the optimization process based on the current number of function evaluations.
        # Args:
        - FEs (int): The current number of function evaluations.
        - MaxFEs (int): The maximum allowed number of function evaluations.
        # Built-in Attribute:
        - self.NP (int): Current population size.
        - self.Nmin (int): Minimum allowed population size.
        - self.archive (np.ndarray): Archive of solutions.
        - self.NA (int): Current archive size.
        # Returns:
        - None
        # Raises:
        - None
        """
        self.sort(self.NP)
        N = self.cal_NP_next_gen(FEs, MaxFEs)
        A = int(max(N * 2.1, self.Nmin))
        N = int(N)
        if N < self.NP:
            self.slice(N)
        if A < self.archive.shape[0]:
            self.NA = A
            self.archive = self.archive[:A]

    # update archive, join new individual
    def update_archive(self, old_id):
        """
        # Introduction
        Updates the archive of solutions by either appending a new solution or replacing an existing one.
        # Args:
        - old_id (int): The index of the solution in the current group to be added to the archive.
        # Built-in Attribute:
        - self.archive (np.ndarray): The current archive of solutions.
        - self.NA (int): The maximum allowed size of the archive.
        - self.group (np.ndarray): The current group of solutions.
        - self.dim (int): The dimensionality of each solution.
        - self.rng (np.random.Generator): Random number generator for selecting replacement index.
        # Returns:
        - None
        # Raises:
        - None
        """
        if self.archive.shape[0] < self.NA:
            self.archive = np.append(self.archive, self.group[old_id]).reshape(-1, self.dim)
        else:
            self.archive[self.rng.randint(self.archive.shape[0])] = self.group[old_id]

    # collect all the features of the group  dim = 6
    def get_feature(self,
                    problem,  # the optimizing problem
                    sample_costs,
                    cost_scale_factor,  # a scale factor to normalize costs
                    progress  # the current progress of evaluations
                    ):
        """
        # Introduction
        Extracts a set of numerical features from the current optimization state, including normalized global best cost, fitness distance correlation, population dispersion, and other metrics relevant to the optimization process.
        # Args:
        - problem: The optimization problem instance, expected to provide a `func` method for evaluating samples.
        - sample_costs (array-like): Costs associated with a set of sample solutions.
        - cost_scale_factor (float): A scaling factor used to normalize cost values.
        - progress (float): The current progress of evaluations, typically between 0 and 1.
        # Returns:
        - list: A list of 9 computed features representing the current state of the optimizer.
        # Raises:
        - Any exception raised by the underlying feature calculation functions or the problem's `func` method.
        """
        gbc = self.gbest / cost_scale_factor
        fdc = cal_fdc(self.group / 100, self.cost / cost_scale_factor)
        random_walk_samples = rw_sampling(self.group, self.rng)
        walk_costs = problem.func(random_walk_samples)
        rf = cal_rf(walk_costs)
        acf = cal_acf(walk_costs)
        nopt = cal_nopt(random_walk_samples, walk_costs)
        # disp, disp_ratio, evp, nsc, anr, ni, nw, adf
        disp, disp_ratio = dispersion(self.group, self.cost)
        evp = population_evolvability(self.cost, sample_costs)
        nsc = negative_slope_coefficient(self.cost, sample_costs[0])
        anr = average_neutral_ratio(self.cost, sample_costs)
        ni, nw = non_improvable_worsenable(self.cost, sample_costs)
        adf = average_delta_fitness(self.cost, sample_costs)
        # return [gbc, fdc, rf, acf, nopt, disp, disp_ratio, evp, nsc, anr, ni, nw, adf, progress]  # 14
        return [gbc, fdc, disp, disp_ratio, nsc, anr, ni, nw, progress]  # 9


class NL_SHADE_RSP:
    def __init__(self, dim, rng, error = 1e-8):
        """
        # Introduction
        Initializes the optimizer with problem dimension, random number generator, and error tolerance.
        # Args:
        - dim (int): The dimensionality of the optimization problem.
        - rng (np.random.Generator): Random number generator for stochastic operations.
        - error (float, optional): Error tolerance for convergence (default: 1e-8).
        # Built-in Attribute:
        - pb (float): Rate of best individuals in mutation (default: 0.4).
        - pa (float): Rate of selecting individual from archive (default: 0.5).
        - dim (int): Dimension of the problem.
        - error (float): Error tolerance for convergence.
        - rng (np.random.Generator): Random number generator instance.
        # Returns:
        - None
        """
        
        self.pb = 0.4  # rate of best individuals in mutation
        self.pa = 0.5  # rate of selecting individual from archive
        self.dim = dim  # dimension of problem
        self.error = error
        self.rng = rng

    def evaluate(self, problem, u):
        """
        # Introduction
        Evaluates the cost of a solution `u` for a given optimization problem, optionally normalizing by the known optimum.
        # Args:
        - problem: An object representing the optimization problem, expected to have `optimum` and `eval` attributes.
        - u: The candidate solution(s) to be evaluated.
        # Returns:
        - cost: The evaluated cost(s) for the solution(s) `u`. If the problem's optimum is known, returns the difference between the evaluated value and the optimum, setting values below the error threshold to zero.
        # Notes:
        - If `problem.optimum` is not `None`, the cost is normalized by subtracting the optimum and thresholded by `self.error`.
        - Assumes `problem.eval(u)` returns a numeric value or array compatible with the operations performed.
        """
        
        if problem.optimum is not None:
            cost = problem.eval(u) - problem.optimum
            cost[cost < self.error] = 0.0
        else:
            cost = problem.eval(u)
        return cost

    # Binomial crossover
    def Binomial(self, x, v, cr):
        dim = len(x)
        jrand = self.rng.randint(dim)
        u = np.where(self.rng.random(dim) < cr, v, x)
        u[jrand] = v[jrand]
        return u

    # Binomial crossover
    def Binomial_(self, x, v, cr):
        NP, dim = x.shape
        jrand = self.rng.randint(dim, size = NP)
        u = np.where(self.rng.rand(NP, dim) < cr.repeat(dim).reshape(NP, dim), v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    # Exponential crossover
    def Exponential(self, x, v, cr):
        dim = len(x)
        u = x.copy()
        L = self.rng.randint(dim)
        for i in range(L, dim):
            if self.rng.random() < cr:
                u[i] = v[i]
            else:
                break
        return u

    # Exponential crossover
    def Exponential_(self, x, v, cr):
        NP, dim = x.shape
        u = x.copy()
        L = self.rng.randint(dim, size = NP).repeat(dim).reshape(NP, dim)
        L = L <= np.arange(dim)
        rvs = self.rng.rand(NP, dim)
        L = np.where(rvs > cr.repeat(dim).reshape(NP, dim), L, 0)
        u = u * (1 - L) + v * L
        return u

    # update pa according to cost changes
    def update_Pa(self, fa, fp, na, NP):
        if na == 0 or fa == 0:
            self.pa = 0.5
            return
        self.pa = (fa / (na + 1e-15)) / ((fa / (na + 1e-15)) + (fp / (NP - na + 1e-15)))
        self.pa = np.minimum(0.9, np.maximum(self.pa, 0.1))

    # step method for ensemble, optimize population for a few times
    def step(self,
             population,  # an initialized or half optimized population, the method will optimize it
             problem,  # the problem instance
             FEs,  # used number of evaluations, also the starting of current step
             FEs_end,  # the ending evaluation number of step, step stop while reaching this limitation
             # i.e. user wants to run a step with 1000 evaluations period, it should be FEs + 1000
             MaxFEs,  # the max number of evaluations
             ):
        # initialize population and archive
        NP, dim = population.NP, population.dim
        NA = int(NP * 2.1)
        if NA < population.archive.shape[0]:
            population.archive = population.archive[:NA]
        self.pa = 0.5
        population.sort(population.NP)
        # start optimization loop
        while FEs < FEs_end and FEs < MaxFEs:
            t1 = time.time()
            # select crossover rate and step length
            Cr, F = population.choose_F_Cr()
            Cr = np.sort(Cr)
            # initialize some record values
            fa = 0  # sum of cost improvement using archive
            fp = 0  # sum of cost improvement without archive
            ap = np.zeros(NP, bool)  # record of whether a individual update with archive
            df = np.array([])  # record of cost improvement of each individual
            pr = np.exp(-(np.arange(NP) + 1) / NP)  # calculate the rate of individuals at different positions being selected in others' mutation
            pr /= np.sum(pr)
            na = 0  # the number of archive usage
            SF = np.array([])  # the set records successful step length
            SCr = np.array([])  # the set records successful crossover rate
            u = np.zeros((NP, dim))  # trail vectors
            # randomly select a crossover method for the population
            CrossExponential = self.rng.random() < 0.5
            t2 = time.time()
            pb_upper = int(np.maximum(2, NP * self.pb))  # the range of pbest selection
            pbs = self.rng.randint(pb_upper, size = NP)  # select pbest for all individual
            count = 0
            duplicate = np.where(pbs == np.arange(NP))[0]
            while duplicate.shape[0] > 0 and count < 1:
                pbs[duplicate] = self.rng.randint(NP, size = duplicate.shape[0])
                duplicate = np.where(pbs == np.arange(NP))[0]
                count += 1
            xpb = population.group[pbs]
            t3 = time.time()
            r1 = self.rng.randint(NP, size = NP)
            count = 0
            duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = self.rng.randint(NP, size = duplicate.shape[0])
                duplicate = np.where((r1 == np.arange(NP)) + (r1 == pbs))[0]
                count += 1
            x1 = population.group[r1]
            t4 = time.time()
            rvs = self.rng.rand(NP)
            r2_pop = np.where(rvs >= self.pa)[0]  # the indices of mutation with population
            r2_arc = np.where(rvs < self.pa)[0]  # the indices of mutation with archive
            use_arc = np.zeros(NP, dtype = bool)  # a record for archive usage, used in parameter updating
            use_arc[r2_arc] = 1
            if population.archive.shape[0] < 25:  # if the archive is empty, indices above are canceled
                r2_pop = np.arange(NP)
                r2_arc = np.array([], dtype = np.int32)
            r2 = self.rng.choice(np.arange(NP), size = r2_pop.shape[0], p = pr)
            count = 0
            duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r2[duplicate] = self.rng.choice(np.arange(NP), size = duplicate.shape[0], p = pr)
                duplicate = np.where((r2 == r2_pop) + (r2 == pbs[r2_pop]) + (r2 == r1[r2_pop]))[0]
                count += 1
            x2 = np.zeros((NP, self.dim))
            t5 = time.time()
            # scatter indiv from population and archive into x2
            if r2_pop.shape[0] > 0:
                x2[r2_pop] = population.group[r2]
            if r2_arc.shape[0] > 0:
                x2[r2_arc] = population.archive[self.rng.randint(np.minimum(population.archive.shape[0], NA), size = r2_arc.shape[0])]
            Fs = F.repeat(self.dim).reshape(NP, self.dim)  # adjust shape for batch processing
            vs = population.group + Fs * (xpb - population.group) + Fs * (x1 - x2)
            # crossover rate for Binomial crossover has a different way for calculating
            Crb = np.zeros(NP)
            tmp_id = np.where(np.arange(NP) + FEs < 0.5 * MaxFEs)[0]
            Crb[tmp_id] = 2 * ((FEs + tmp_id) / MaxFEs - 0.5)
            if CrossExponential:
                Cr = Crb
                us = self.Binomial_(population.group, vs, Cr)
            else:
                us = self.Exponential_(population.group, vs, Cr)
            # reinitialize values exceed valid range
            us = us * ((-100 <= us) * (us <= 100)) + ((us > 100) + (us < -100)) * (self.rng.rand(NP, dim) * 200 - 100)
            t6 = time.time()
            cost = self.evaluate(problem, us)
            optim = np.where(cost < population.cost)[0]  # the indices of indiv whose costs are better than parent
            for i in range(optim.shape[0]):
                population.update_archive(i)
            population.F[optim] = F[optim]
            population.Cr[optim] = Cr[optim]
            SF = F[optim]
            SCr = Cr[optim]
            df = (population.cost[optim] - cost[optim]) / (population.cost[optim] + 1e-9)
            arc_usage = use_arc[optim]
            fp = np.sum(df[arc_usage])
            fa = np.sum(df[np.array(1 - arc_usage, dtype = bool)])
            na = np.sum(arc_usage)
            population.group[optim] = us[optim]
            population.cost[optim] = cost[optim]
            t7 = time.time()

            if np.min(cost) < population.gbest:
                population.gbest = np.min(cost)
                population.gbest_solution = population.group[np.argmin(cost)]

            FEs += NP
            # adaptively adjust parameters
            self.pb = 0.4 - 0.2 * (FEs / MaxFEs)
            population.NLPSR(FEs, MaxFEs)
            population.update_M_F_Cr(SF, SCr, df)
            self.update_Pa(fa, fp, na, NP)
            NP = population.NP
            NA = population.NA
            if np.min(cost) < self.error:
                return population, min(FEs, MaxFEs)

        return population, min(FEs, MaxFEs)


class JDE21:
    def __init__(self, dim, rng, error = 1e-8):
        self.dim = dim  # problem dimension
        self.sNP = 10  # size of small population
        self.bNP = 160  # size of big population
        # meaning of following parameters reference from the JDE21 paper
        self.tao1 = 0.1
        self.tao2 = 0.1
        self.Finit = 0.5
        self.CRinit = 0.9
        self.Fl_b = 0.1
        self.Fl_s = 0.17
        self.Fu = 1.1
        self.CRl_b = 0.0
        self.CRl_s = 0.1
        self.CRu_b = 1.1
        self.CRu_s = 0.8
        self.eps = 1e-12
        self.MyEps = 0.25
        # record number of operation called
        self.nReset = 0
        self.sReset = 0
        self.cCopy = 0
        self.terminateErrorValue = error
        self.rng = rng

    # check whether the optimization stuck(global best doesn't improve for a while)
    def prevecEnakih(self, cost, best):
        eqs = len(cost[np.fabs(cost - best) < self.eps])
        return eqs > 2 and eqs > len(cost) * self.MyEps

    # crowding operation describe in JDE21
    def crowding(self, group, v):
        dist = np.sum((group - v) ** 2, -1)
        return np.argmin(dist)

    def crowding_(self, group, vs):
        NP, dim = vs.shape
        dist = np.sum(((group * np.ones((NP, NP, dim))).transpose(1, 0, 2) - vs) ** 2, -1).transpose()
        return np.argmin(dist, -1)

    def evaluate(self, Xs, problem):
        if problem.optimum is not None:
            cost = problem.eval(Xs) - problem.optimum
        else:
            cost = problem.eval(Xs)
        cost[cost < self.terminateErrorValue] = 0.0
        return cost

    def step(self,
             population,  # an initialized or half optimized population, the method will optimize it
             problem,  # the problem instance
             FEs,  # used number of evaluations, also the starting of current step
             FEs_end,  # the ending evaluation number of step, step stop while reaching this limitation
             MaxFEs,  # the max number of evaluations
             ):
        # initialize population
        NP = population.NP
        dim = population.dim
        sNP = self.sNP
        bNP = NP - sNP
        age = 0

        def mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big):
            if big:
                xNP = bNP
                randF = self.rng.rand(xNP) * self.Fu + self.Fl_b
                randCr = self.rng.rand(xNP) * self.CRu_b + self.CRl_b
                pF = population.F[:xNP]
                pCr = population.Cr[:xNP]
            else:
                xNP = sNP
                randF = self.rng.rand(xNP) * self.Fu + self.Fl_s
                randCr = self.rng.rand(xNP) * self.CRu_b + self.CRl_s
                pF = population.F[-sNP:]
                pCr = population.Cr[-sNP:]

            rvs = self.rng.rand(xNP)
            F = np.where(rvs < self.tao1, randF, pF)
            rvs = self.rng.rand(xNP)
            Cr = np.where(rvs < self.tao2, randCr, pCr)
            Fs = F.repeat(dim).reshape(xNP, dim)
            Crs = Cr.repeat(dim).reshape(xNP, dim)
            v = population.group[r1] + Fs * (population.group[r2] - population.group[r3])
            v = np.clip(v, population.Xmin, population.Xmax)
            jrand = self.rng.randint(dim, size = xNP)
            u = np.where(self.rng.rand(xNP, dim) < Crs, v, (population.group[:bNP] if big else population.group[bNP:]))
            u[np.arange(xNP), jrand] = v[np.arange(xNP), jrand]
            cost = self.evaluate(u, problem)
            if big:
                crowding_ids = self.crowding_(population.group[:xNP], u)
            else:
                crowding_ids = np.arange(xNP) + bNP
            age += xNP
            for i in range(xNP):
                id = crowding_ids[i]
                if cost[i] < population.cost[id]:
                    # update and record
                    population.update_archive(id)
                    population.group[id] = u[i]
                    population.cost[id] = cost[i]
                    population.F[id] = F[i]
                    population.Cr[id] = Cr[i]
                    SF = np.append(SF, F[i])
                    SCr = np.append(SCr, Cr[i])
                    d = (population.cost[i] - cost[i]) / (population.cost[i] + 1e-9)
                    df = np.append(df, d)
                    if cost[i] < population.cbest:
                        age = 0
                        population.cbest_id = id
                        population.cbest = cost[i]
                        if cost[i] < population.gbest:
                            population.gbest = cost[i]
                            population.gbest_solution = u[i]

            return SF, SCr, df, age

        population.sort(NP, True)
        # check record point lest missing it
        while FEs < FEs_end:
            # initialize temp records
            v = np.zeros((NP, dim))
            F = self.rng.random(NP)
            Cr = self.rng.random(NP)
            # small population evaluates same times as big one thus the total evaluations for a loop is doubled big one
            N = bNP * 2
            I = -1
            df = np.array([])
            SF = np.array([])
            SCr = np.array([])
            if self.prevecEnakih(population.cost[:bNP], population.gbest) or age > MaxFEs / 10:
                self.nReset += 1
                population.group[:bNP] = population.initialize_group(bNP)
                population.F[:bNP] = self.Finit
                population.Cr[:bNP] = self.CRinit
                population.cost[:bNP] = 1e15
                age = 0
                population.cbest = np.min(population.cost)
                population.cbest_id = np.argmin(population.cost)

            if FEs < MaxFEs / 3:
                mig = 1
            elif FEs < 2 * MaxFEs / 3:
                mig = 2
            else:
                mig = 3

            r1 = self.rng.randint(bNP, size = bNP)
            count = 0
            duplicate = np.where((r1 == np.arange(bNP)) * (r1 == population.cbest_id))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = self.rng.randint(bNP, size = duplicate.shape[0])
                duplicate = np.where((r1 == np.arange(bNP)) * (r1 == population.cbest_id))[0]
                count += 1

            r2 = self.rng.randint(bNP + mig, size = bNP)
            count = 0
            duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r2[duplicate] = self.rng.randint(bNP + mig, size = duplicate.shape[0])
                duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
                count += 1

            r3 = self.rng.randint(bNP + mig, size = bNP)
            count = 0
            duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r3[duplicate] = self.rng.randint(bNP + mig, size = duplicate.shape[0])
                duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[0]
                count += 1

            SF, SCr, df, age = mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big = True)
            FEs += bNP

            if np.min(population.cost) < self.terminateErrorValue:
                return population, min(FEs, MaxFEs)
            if FEs >= FEs_end or FEs >= MaxFEs:
                return population, min(FEs, MaxFEs)

            if population.cbest_id >= bNP and self.prevecEnakih(population.cost[bNP:], population.cbest):
                self.sReset += 1
                cbest = population.cbest
                cbest_id = population.cbest_id
                tmp = population.group[cbest_id]
                population.group[bNP:] = population.initialize_group(sNP)
                population.F[bNP:] = self.Finit
                population.Cr[bNP:] = self.CRinit
                population.cost[bNP:] = 1e15
                population.cbest = cbest
                population.cbest_id = cbest_id
                population.group[cbest_id] = tmp
                population.cost[cbest_id] = cbest

            if population.cbest_id < bNP:
                self.cCopy += 1
                population.cost[bNP] = population.cbest
                population.group[bNP] = population.group[population.cbest_id]
                population.cbest_id = bNP

            for i in range(bNP // sNP):

                r1 = self.rng.randint(sNP, size = sNP) + bNP
                count = 0
                duplicate = np.where(r1 == (np.arange(sNP) + bNP))[0]
                while duplicate.shape[0] > 0 and count < 25:
                    r1[duplicate] = self.rng.randint(sNP, size = duplicate.shape[0]) + bNP
                    duplicate = np.where(r1 == (np.arange(sNP) + bNP))[0]
                    count += 1

                r2 = self.rng.randint(sNP, size = sNP) + bNP
                count = 0
                duplicate = np.where((r2 == (np.arange(sNP) + bNP)) + (r2 == r1))[0]
                while duplicate.shape[0] > 0 and count < 25:
                    r2[duplicate] = self.rng.randint(sNP, size = duplicate.shape[0]) + bNP
                    duplicate = np.where((r2 == (np.arange(sNP) + bNP)) + (r2 == r1))[0]
                    count += 1

                r3 = self.rng.randint(sNP, size = sNP) + bNP
                count = 0
                duplicate = np.where((r3 == (np.arange(sNP) + bNP)) + (r3 == r1) + (r3 == r2))[0]
                while duplicate.shape[0] > 0 and count < 25:
                    r3[duplicate] = self.rng.randint(sNP, size = duplicate.shape[0]) + bNP
                    duplicate = np.where((r3 == (np.arange(sNP) + bNP)) + (r3 == r1) + (r3 == r2))[0]
                    count += 1

                SF, SCr, df, age = mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big = False)
                FEs += sNP

                if np.min(population.cost) < self.terminateErrorValue:
                    return population, min(FEs, MaxFEs)
                if FEs >= FEs_end or FEs >= MaxFEs:
                    return population, min(FEs, MaxFEs)

            # update and record information for NL-SHADE-RSP and reduce population

            population.update_M_F_Cr(SF, SCr, df)
            NP = int(population.cal_NP_next_gen(FEs, MaxFEs))
            population.NP = NP
            # population.sort(NP, True)
            population.group = population.group[-NP:]
            population.cost = population.cost[-NP:]
            population.F = population.F[-NP:]
            population.Cr = population.Cr[-NP:]
            population.cbest_id = np.argmin(population.cost)
            population.cbest = np.min(population.cost)
            bNP = NP - sNP

        return population, min(FEs, MaxFEs)

    # a testing method which runs a complete optimization on a population and show its performance, similar to step

class MadDE:
    def __init__(self, dim, rng, error = 1e-8):
        self.dim = dim
        self.p = 0.18
        self.PqBX = 0.01
        self.F0 = 0.2
        self.Cr0 = 0.2
        self.pm = np.ones(3) / 3
        self.error = error
        self.rng = rng

    def ctb_w_arc(self, group, best, archive, Fs):
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]

        count = 0
        rb = self.rng.randint(NB, size = NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = self.rng.randint(NB, size = duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = self.rng.randint(NP, size = NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = self.rng.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = self.rng.randint(NP + NA, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = self.rng.randint(NP + NA, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (xb - group) + Fs * (x1 - x2)

        return v

    def ctr_w_arc(self, group, archive, Fs):
        NP, dim = group.shape
        NA = archive.shape[0]

        count = 0
        r1 = self.rng.randint(NP, size = NP)
        duplicate = np.where((r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = self.rng.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = self.rng.randint(NP + NA, size = NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = self.rng.randint(NP + NA, size = duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (x1 - x2)

        return v

    def weighted_rtb(self, group, best, Fs, Fas):
        NP, dim = group.shape
        NB = best.shape[0]

        count = 0
        rb = self.rng.randint(NB, size = NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = self.rng.randint(NB, size = duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = self.rng.randint(NP, size = NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = self.rng.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = self.rng.randint(NP, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = self.rng.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]
        v = Fs * x1 + Fs * Fas * (xb - x2)

        return v

    def binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = self.rng.randint(dim, size = NP)
        u = np.where(self.rng.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def step(self, population, problem, FEs, FEs_end, MaxFEs):
        population.sort(population.NP)
        while FEs < FEs_end and FEs < MaxFEs:
            NP, dim = population.NP, population.dim
            q = 2 * self.p - self.p * FEs / MaxFEs
            Fa = 0.5 + 0.5 * FEs / MaxFEs
            Cr, F = population.choose_F_Cr()
            mu = self.rng.choice(3, size = NP, p = self.pm)
            p1 = population.group[mu == 0]
            p2 = population.group[mu == 1]
            p3 = population.group[mu == 2]
            pbest = population.group[:max(int(self.p * NP), 2)]
            qbest = population.group[:max(int(q * NP), 2)]
            Fs = F.repeat(dim).reshape(NP, dim)
            v1 = self.ctb_w_arc(p1, pbest, population.archive, Fs[mu == 0])
            v2 = self.ctr_w_arc(p2, population.archive, Fs[mu == 1])
            v3 = self.weighted_rtb(p3, qbest, Fs[mu == 2], Fa)
            v = np.zeros((NP, dim))
            v[mu == 0] = v1
            v[mu == 1] = v2
            v[mu == 2] = v3
            v[v < -100] = (population.group[v < -100] - 100) / 2
            v[v > 100] = (population.group[v > 100] + 100) / 2
            rvs = self.rng.rand(NP)
            Crs = Cr.repeat(dim).reshape(NP, dim)
            u = np.zeros((NP, dim))
            if np.sum(rvs <= self.PqBX) > 0:
                qu = v[rvs <= self.PqBX]
                if population.archive.shape[0] > 0:
                    qbest = np.concatenate((population.group, population.archive), 0)[:max(int(q * (NP + population.archive.shape[0])), 2)]
                cross_qbest = qbest[self.rng.randint(qbest.shape[0], size = qu.shape[0])]
                qu = self.binomial(cross_qbest, qu, Crs[rvs <= self.PqBX])
                u[rvs <= self.PqBX] = qu
            bu = v[rvs > self.PqBX]
            bu = self.binomial(population.group[rvs > self.PqBX], bu, Crs[rvs > self.PqBX])
            u[rvs > self.PqBX] = bu
            if problem.optimum is not None:
                ncost = problem.eval(u) - problem.optimum
            else:
                ncost = problem.eval(u)
            FEs += NP
            optim = np.where(ncost < population.cost)[0]
            for i in optim:
                population.update_archive(i)
            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, population.cost - ncost)
            population.update_M_F_Cr(SF, SCr, df[optim])
            count_S = np.zeros(3)
            for i in range(3):
                count_S[i] = np.mean(df[mu == i] / population.cost[mu == i])
            if np.sum(count_S) > 0:
                self.pm = np.maximum(0.1, np.minimum(0.9, count_S / np.sum(count_S)))
                self.pm /= np.sum(self.pm)
            else:
                self.pm = np.ones(3) / 3

            population.group[optim] = u[optim]
            population.cost = np.minimum(population.cost, ncost)
            population.NLPSR(FEs, MaxFEs)
            if np.min(population.cost) < population.gbest:
                population.gbest = np.min(population.cost)
                population.gbest_solution = population.group[np.argmin(population.cost)]

            if np.min(population.cost) < self.error:
                return population, min(FEs, MaxFEs)
        return population, min(FEs, MaxFEs)


class Info:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get(self):
        return self.kwargs

    def add(self, key, value):
        self.kwargs[key] = value


# random walk sampler
def rw_sampling(group, rng):
    gs, dim = group.shape
    Pmax = np.max(group, axis = 0)
    Pmin = np.min(group, axis = 0)
    size = Pmax - Pmin
    walk = []
    walk.append(rng.rand(dim))
    for i in range(1, gs):
        step = rng.rand(dim) * size
        tmp_res = walk[i - 1] + step
        walk.append(tmp_res - np.floor(tmp_res))
    return np.array(walk)


# compare diff between 2 characters
def compare_diff(diff, epsilon):
    S_epsilon = []
    label_counts = np.zeros(6) + 1
    for i in range(len(diff)):
        if diff[i] < -1 * epsilon:
            S_epsilon.append(-1)
        elif diff[i] > epsilon:
            S_epsilon.append(1)
        else:
            S_epsilon.append(0)
    for i in range(len(S_epsilon) - 1):
        if S_epsilon[i] == -1 and S_epsilon[i + 1] == 0:
            label_counts[0] += 1
        if S_epsilon[i] == -1 and S_epsilon[i + 1] == 1:
            label_counts[1] += 1
        if S_epsilon[i] == 1 and S_epsilon[i + 1] == 0:
            label_counts[2] += 1
        if S_epsilon[i] == 1 and S_epsilon[i + 1] == -1:
            label_counts[3] += 1
        if S_epsilon[i] == 0 and S_epsilon[i + 1] == -1:
            label_counts[4] += 1
        if S_epsilon[i] == 0 and S_epsilon[i + 1] == 1:
            label_counts[5] += 1
    probs = label_counts / np.sum(label_counts)
    entropy = -1 * np.sum(probs * (np.log(probs) / np.log(6)))
    return entropy


# calculate FDC of group
def cal_fdc(group, costs):
    opt_x = sorted(zip(group, costs), key = lambda x: x[1])[0][0]
    ds = np.sum((group - opt_x) ** 2, axis = 1)
    fs = 1 / (costs + (1e-8))
    C_fd = ((fs - fs.mean()) * (ds - ds.mean())).mean()
    delta_f = ((fs - fs.mean()) ** 2).mean()
    delta_d = ((ds - ds.mean()) ** 2).mean()
    return C_fd / ((delta_d * delta_f) + (1e-8))


# calculate RIE(ruggedness of information entropy) of group
def cal_rf(costs):
    diff = costs[1:] - costs[:len(costs) - 1]
    epsilon_max = np.max(diff)
    entropy_list = []
    factor = 128
    while factor >= 1:
        entropy_list.append(compare_diff(diff, epsilon_max / factor))
        factor /= 2
    entropy_list.append(compare_diff(diff, 0))
    return np.max(entropy_list)


# calculate Auto-correlation of group fitness
def cal_acf(costs):
    temp = costs[:-1]
    temp_shift = costs[1:]
    fmean = np.mean(costs)
    cov = np.sum((temp - fmean) * (temp_shift - fmean))
    v = np.sum((costs - fmean) ** 2)
    return cov / (v + (1e-8))


# calculate local fitness landscape metric
def cal_nopt(group, costs):
    opt_x = sorted(zip(group, costs), key = lambda x: x[1])[0][0]
    ds = np.sum((group - opt_x) ** 2, axis = 1)
    costs_sorted, _ = zip(*sorted(zip(costs, ds), key = lambda x: x[1]))
    counts = 0
    for i in range(len(costs) - 1):
        if costs_sorted[i + 1] <= costs_sorted[i]:
            counts += 1
    return counts / len(costs)


# dispersion metric and ratio
def dispersion(group, costs):  # [a] + [f]
    gs, dim = group.shape
    group_sorted = group[np.argsort(costs)]
    group_sorted = (group_sorted / 200) + 0.5
    diam = np.sqrt(dim)
    # calculate max and avg distances
    max_dis = 0
    disp = 0
    for i in range(1, gs):
        shift_group = np.concatenate((group_sorted[i:], group_sorted[:i]), 0)
        distances = np.sqrt(np.sum((group_sorted - shift_group) ** 2, -1))
        disp += np.sum(distances)
        max_dis = np.maximum(max_dis, np.max(distances))
    disp /= gs ** 2
    # calculate avg distance of 10% individuals
    disp10 = 0
    gs10 = gs * 10 // 100
    group_sorted = group_sorted[:gs10]
    for i in range(1, gs10):
        shift_group = np.concatenate((group_sorted[i:], group_sorted[:i]), 0)
        disp10 += np.sum(np.sqrt(np.sum((group_sorted - shift_group) ** 2, -1)))
    disp10 /= gs10 ** 2
    return disp10 - disp, max_dis / diam


def population_evolvability(group_cost, sample_costs):  # [i]
    fbs = np.min(sample_costs, -1)
    n_plus = np.sum(fbs < np.min(group_cost))
    gs = group_cost.shape[0]
    if n_plus == 0:
        return 0
    evp = np.sum(np.fabs(fbs - np.min(group_cost)) / gs / (np.std(group_cost) + 1e-8)) / sample_costs.shape[0]
    return evp


def negative_slope_coefficient(group_cost, sample_cost):  # [j]
    gs = sample_cost.shape[0]
    m = 10
    gs -= gs % m  # to be divisible
    if gs < m:  # not enough costs for m dividing
        return 0
    sorted_cost = np.array(sorted(list(zip(group_cost[:gs], sample_cost[:gs]))))
    sorted_group = sorted_cost[:, 0].reshape(m, -1)
    sorted_sample = sorted_cost[:, 1].reshape(m, -1)
    Ms = np.mean(sorted_group, -1)
    Ns = np.mean(sorted_sample, -1)
    nsc = np.minimum((Ns[1:] - Ns[:-1]) / (Ms[1:] - Ms[:-1] + 1e-8), 0)
    return np.sum(nsc)


def average_neutral_ratio(group_cost, sample_costs, eps = 1):
    gs = sample_costs.shape[1]
    dcost = np.fabs(sample_costs - group_cost[:gs])
    return np.mean(np.sum(dcost < eps, 0) / sample_costs.shape[0])


def non_improvable_worsenable(group_cost, sample_costs):
    gs = sample_costs.shape[1]
    NI = 1 - np.count_nonzero(np.sum(group_cost[:gs] > sample_costs, -1)) / sample_costs.shape[0]
    NW = 1 - np.count_nonzero(np.sum(group_cost[:gs] < sample_costs, -1)) / sample_costs.shape[0]
    return NI, NW


def average_delta_fitness(group_cost, sample_costs):
    gs = sample_costs.shape[1]
    return np.sum(sample_costs - group_cost[:gs]) / sample_costs.shape[0] / gs / np.max(group_cost[:gs])


# Online score judge, get performance sequences from running algorithms and calculate scores
def score_judge(results):
    alg_num = len(results)
    n = 30
    score = np.zeros(alg_num)
    for problem in list(results[0].keys()):
        for config in list(results[0][problem].keys()):
            Fevs = np.array([])
            FEs = np.array([])
            for alg in range(alg_num):
                Fevs = np.append(Fevs, results[alg][problem][config]['Fevs'][:, -1])
                FEs = np.append(FEs, results[alg][problem][config]['success_fes'])
            nm = n * alg_num
            order = sorted(list(zip(FEs, Fevs, np.arange(nm))))
            for i in range(nm):
                score[order[i][2] // n] += nm - i
            score -= n * (n + 1) / 2
    return score


# score judge, get performance sequences from result files
def score_judge_from_file(result_paths, num_problem):
    alg_num = len(result_paths)
    n = 30
    nm = n * alg_num
    score = np.zeros(alg_num)
    fpts = []
    for i in range(alg_num):
        fpts.append(open(result_paths[i], 'r'))
    for p in range(num_problem):
        Fevs = np.array([])
        FEs = np.array([])
        for alg in range(alg_num):
            fpt = fpts[alg]
            text = fpt.readline()
            while text != 'Function error values:\n':
                text = fpt.readline()
            for i in range(n):
                text = fpt.readline().split()
                success_fes = float(text[-1])
                error_value = float(text[-2])
                Fevs = np.append(Fevs, error_value)
                FEs = np.append(FEs, success_fes)
        order = sorted(list(zip(FEs, Fevs, np.arange(nm))))
        for i in range(nm):
            score[order[i][2] // n] += nm - i
        score -= n * (n + 1) / 2
    return score
