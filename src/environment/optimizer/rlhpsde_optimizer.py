import numpy as np
import scipy.stats as stats
from .learnable_optimizer import Learnable_Optimizer
from typing import Union, Iterable


class Population:
    def __init__(self, config, rng, dim):
        self.Nmax = config.NP_max                      # the upperbound of population size
        self.Nmin = config.NP_min                      # the lowerbound of population size
        self.NP = self.Nmax                            # the population size
        self.dim = dim                                 # the dimension of individuals
        self.group = None                              # the population
        self.cost = None                               # the cost of individuals
        self.gbest = None                              # the global best cost
        self.gbest_solution = None                     # the individual with global best cost
        self.F = config.F
        self.Cr = config.Cr
        self.MF = np.ones(int(self.dim * config.Hm)) * self.F       # the set of step length of DE
        self.MCr = np.ones(int(self.dim * config.Hm)) * self.Cr     # the set of crossover rate of DE
        self.k = 0                                     # the index of updating element in MF and MCr

        self.init_best = None
        
        self.rng = rng # random state

    # generate an initialized population with size(default self population size)
    def initialize_group(self, lb, ub, size=-1):
        if size < 0:
            size = self.NP
        self.group = self.rng.rand(size, self.dim) * (ub - lb) + lb

    # initialize cost
    def initialize_costs(self, problem):
        if problem.optimum is None:
            self.cost = problem.eval(self.group)
        else:
            self.cost = problem.eval(self.group) - problem.optimum
        self.gbest = np.min(self.cost)
        self.gbest_solution = self.group[np.argmin(self.cost)]
        self.init_best = np.min(self.cost)

    # sort former 'size' population in respect to cost
    def sort(self, size, reverse=False):
        # new index after sorting
        r = -1 if reverse else 1
        ind = np.concatenate((np.argsort(r * self.cost[:size]), np.arange(self.NP)[size:]))
        self.group = self.group[ind]
        self.cost = self.cost[ind]
        self.gbest = np.min(self.cost)
        self.gbest_solution = self.group[np.argmin(self.cost)]

    # randomly choose step length nad crossover rate from MF and MCr
    def choose_F_Cr(self, F_dist):
        # generate Cr
        gs = self.NP
        ind_r = self.rng.randint(0, self.MF.shape[0], size=gs)  # index
        Cr = np.minimum(1, np.maximum(0, self.rng.normal(loc=self.MCr[ind_r], scale=0.1, size=gs)))  # 0~1
        # generate F
        locs = self.MF[ind_r]
        F = None
        if F_dist == 'cauchy':
            F = stats.cauchy.rvs(loc=locs, scale=0.1, size=gs)
        elif F_dist == 'levy':
            F = stats.levy.rvs(loc=locs, scale=0.1, size=gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * locs[err] - F[err]
        F = np.minimum(1, F)
        return Cr, F

    # calculate wL mean
    def mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # update MF and MCr, join new value into the set if there are some successful changes or set it to initial value
    def update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_wL = self.mean_wL(df, SF)
            self.MF[self.k] = mean_wL
            mean_wL = self.mean_wL(df, SCr)
            self.MCr[self.k] = mean_wL
            self.k = (self.k + 1) % self.MF.shape[0]
        else:
            self.MF[self.k] = 0.5
            self.MCr[self.k] = 0.5

    # linearly population size reduction
    def LPSR(self, fes, maxFEs):
        self.sort(self.NP)
        N = max(int(self.Nmax + np.round(self.Nmin - self.Nmax) * fes / maxFEs), 1)
        if N < self.NP:
            self.NP = N
            self.group = self.group[:N]
            self.cost = self.cost[:N]
            self.gbest = np.min(self.cost)
            self.gbest_solution = self.group[np.argmin(self.cost)]


class RLHPSDE_Optimizer(Learnable_Optimizer):
    """
    # RLHPSDE_Optimizer
    A reinforcement learning-based hyper-parameter self-adaptive differential evolution optimizer.  
    This optimizer dynamically adapts its mutation and crossover strategies using reinforcement learning, and employs random walk-based landscape analysis to guide its search process.
    # Introduction
    The `RLHPSDE_Optimizer` class extends `Learnable_Optimizer` and implements a self-adaptive differential evolution algorithm enhanced with reinforcement learning.  
    It utilizes random walk sampling and landscape analysis (fitness distance correlation and information entropy ruggedness) to determine the current state of the optimization landscape, which is then used to select appropriate mutation strategies.  
    The optimizer maintains a population of candidate solutions and iteratively updates them to minimize a given objective function.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration, setting up algorithm-specific parameters and internal state variables.
        # Args:
        - config (object): Configuration object containing optimizer parameters such as mutation factor, crossover rate, minimum population size, memory factor, random walk steps, step size, maximum function evaluations, and logging interval.
            - The Attributes needed for the RLHPSDE_Optimizer:
                    
        # Built-in Attribute:
        - self.__config: Stores the configuration object.
        - self.__population: Placeholder for the population, initialized as None.
        - self.__rw_steps: Number of random walk steps, taken from config.
        - self.__step_size: Step size for the optimizer, taken from config.
        - self.__maxFEs: Maximum number of function evaluations, taken from config.
        - self.fes: Counter for function evaluations, initialized as None.
        - self.cost: Placeholder for the cost value, initialized as None.
        - self.log_index: Index for logging, initialized as None.
        - self.log_interval: Interval for logging, taken from config.
        # Raises:
        - AttributeError: If required attributes are missing from the config object.
        """
        
        super().__init__(config)
        config.F = 0.5
        config.Cr = 0.5
        config.NP_min = 4
        config.Hm = 0.5
        config.rw_steps = 200
        config.step_size = 10
        self.__config = config

        self.__population = None
        self.__rw_steps = config.rw_steps
        self.__step_size = config.step_size
        self.__maxFEs = config.maxFEs
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        """
        Returns a string representation of the RLHPSDE optimizer.
        # Returns:
            str: The name of the optimizer, "RLHPSDE".
        """
        
        return "RLHPSDE"

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimization process, sets up costs, sorts individuals, and prepares logging and metadata as required.
        # Args:
        - problem (object): An object representing the optimization problem, expected to have attributes `lb` (lower bounds), `ub` (upper bounds), and methods or properties for cost evaluation.
        # Returns:
        - object: The current state of the optimizer after population initialization, as returned by `self.__get_state(problem)`.
        # Side Effects:
        - Initializes and modifies internal attributes such as `self.__population`, `self.fes`, `self.log_index`, `self.meta_X`, `self.meta_Cost`, and `self.cost`.
        """
        self.__dim = problem.dim
        self.__config.NP_max = 18 * problem.dim
        self.__population = Population(self.__config, self.rng, problem.dim)
        self.__population.initialize_group(lb=problem.lb, ub=problem.ub)
        self.__population.initialize_costs(problem)
        self.__population.sort(self.__population.NP)
        self.fes = self.__population.NP
        self.log_index = 1

        if self.__config.full_meta_data:
            self.meta_X = [self.__population.group.copy()]
            self.meta_Cost = [self.__population.cost.copy()]
        self.cost = [self.__population.gbest]


        
        return self.__get_state(problem)

    def __simple_random_walk(self, lb, ub):
        """
        # Introduction
        Generates a sequence of samples using a simple random walk within specified lower and upper bounds.
        # Args:
        - lb (np.ndarray): Lower bounds for each dimension of the random walk (shape: [dim,]).
        - ub (np.ndarray): Upper bounds for each dimension of the random walk (shape: [dim,]).
        # Built-in Attribute:
        - self.__rw_steps (int): Number of random walk steps to perform.
        - self.__dim (int): Dimensionality of the search space.
        - self.__step_size (float): Maximum step size for each random walk move.
        - self.rng (np.random.Generator): Random number generator used for sampling.
        # Returns:
        - np.ndarray: Array of shape (self.__rw_steps + 1, self.__dim) containing the random walk samples.
        # Raises:
        - None
        """
        samples = np.zeros((self.__rw_steps + 1, self.__dim))
        samples[0] = lb + self.rng.random(self.__dim) * (ub - lb)
        for step in range(1, self.__rw_steps + 1):
            samples[step] = samples[step - 1] + self.rng.uniform(low=-self.__step_size,
                                                                  high=self.__step_size,
                                                                  size=self.__dim)
            while True:
                outter_index = np.where(np.any([samples[step] > ub, samples[step] < lb], axis=0))[0]
                if outter_index.shape[0] > 0:
                    samples[step][outter_index] = samples[step - 1][outter_index] + self.rng.uniform(low=-self.__step_size,
                                                                                                      high=self.__step_size,
                                                                                                      size=outter_index.shape[0])
                else:
                    break
        return samples

    def __progressive_random_walk(self, lb, ub):
        """
        # Introduction
        Generates a sequence of samples using a progressive random walk within specified lower and upper bounds. The walk starts from a randomly initialized point and iteratively updates the position, reflecting off the boundaries when exceeded.
        # Args:
        - lb (float or np.ndarray): The lower bound(s) for each dimension of the random walk.
        - ub (float or np.ndarray): The upper bound(s) for each dimension of the random walk.
        # Returns:
        - np.ndarray: An array of shape (self.__rw_steps + 1, self.__dim) containing the sequence of sampled points during the random walk.
        # Raises:
        - None
        """
        samples = np.zeros((self.__rw_steps + 1, self.__dim))
        startingZone = self.rng.rand(self.__dim)
        startingZone[startingZone < 0.5] = -1
        startingZone[startingZone >= 0.5] = 1
        r = self.rng.rand(self.__dim) * (ub - lb) / 2
        samples[0] = (ub + lb) / 2 + startingZone * r
        rD = self.rng.choice(self.__dim, 1)
        if startingZone[rD] == -1:
            lbj = lb if np.isscalar(lb) else lb[rD]
            samples[0][rD] = lbj
        else:
            ubj = ub if np.isscalar(ub) else ub[rD]
            samples[0][rD] = ubj
        for step in range(1, self.__rw_steps + 1):
            samples[step] = samples[step - 1] + self.rng.rand(self.__dim) * (-self.__step_size) * startingZone
            cro_ub = samples[step] > ub
            cro_lb = samples[step] < lb
            samples[step][cro_ub] = (2 * ub - samples[step])[cro_ub]
            samples[step][cro_lb] = (2 * lb - samples[step])[cro_lb]
            startingZone[np.any([cro_ub, cro_lb], axis=0)] *= -1
        return samples

    # Dynamic Fitness Distance Correlation
    def __DFDC(self, sample, cost):
        """
        # Introduction
        Calculate the Dynamic Fitness Distance Correlation.
        # Args:
        - sample (np.ndarray): Array of candidate solutions, where the first element is excluded from analysis.
        - cost (np.ndarray): Array of cost values corresponding to each sample, with the first element excluded from analysis.
        # Returns:
        - bool: `True` if the sample is classified as "easy" (correlation coefficient between 0.15 and 1), `False` if "difficult" (between -1 and 0.15).
        # Raises:
        - ValueError: If the computed correlation coefficient is outside the expected range [-1, 1] or if standard deviations are zero.
        """
        
        sample = sample[1:]
        cost = cost[1:]
        gbest_solution = self.__population.gbest_solution
        dist = np.linalg.norm(sample - gbest_solution, ord=2, axis=-1)
        r = np.mean((cost - cost.mean()) * (dist - dist.mean())) / (cost.std() * dist.std() + 1e-32)
        if 0.15 < r <= 1:
            return True  # easy
        elif -1 <= r < 0.15:
            return False  # difficult
        else:
            raise ValueError(f"DFDC error: {r}, {cost.std()}, {dist.std()}")

    # Dynamic Ruggedness of Information Entropy
    def __DRIE(self, cost):
        """
        # Introduction
        Determines the difficulty of a cost sequence using the DRIE (Difficulty Rating Index Estimator) method based on entropy of symbol transitions.
        # Args:
        - cost (np.ndarray): 1D array of cost values representing a sequence of steps.
        # Built-in Attribute:
        - self.__rw_steps (int): Number of random walk steps used for windowing the cost sequence.
        # Returns:
        - bool: 
            - True if the sequence is classified as "easy" (0.5 <= r <= 1).
            - False if the sequence is classified as "difficult" (0 <= r < 0.5).
        # Raises:
        - ValueError: If the computed DRIE value `r` falls outside the expected range [0, 1].
        """
        diff = cost[1:] - cost[:self.__rw_steps]
        e_star = np.max(np.abs(diff))
        r = None
        for i, scale in enumerate([0, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]):
            symbol = (diff < (-scale * e_star)) * (-1) + ((scale * e_star) < diff) * 1
            prob = np.zeros(6)
            for j in range(self.__rw_steps - 1):
                if symbol[j] == -1 and symbol[j + 1] == 0:
                    prob[0] += 1
                elif symbol[j] == -1 and symbol[j + 1] == 1:
                    prob[1] += 1
                elif symbol[j] == 0 and symbol[j + 1] == -1:
                    prob[2] += 1
                elif symbol[j] == 0 and symbol[j + 1] == 1:
                    prob[3] += 1
                elif symbol[j] == 1 and symbol[j + 1] == -1:
                    prob[4] += 1
                elif symbol[j] == 1 and symbol[j + 1] == 0:
                    prob[5] += 1
            prob /= self.__rw_steps
            prob[prob < 1e-15] = 1e-15
            if i == 0:
                r = -np.sum(prob * np.log(prob) / np.log(6))
            else:
                r = max(r, -np.sum(prob * np.log(prob) / np.log(6)))
        if 0.5 <= r <= 1:
            return True  # easy
        elif 0 <= r < 0.5:
            return False  # difficult
        else:
            raise ValueError(f"DRIE error: {r}")

    def __get_state(self, problem):
        """
        # Introduction
        Generates the current state representation for the optimizer by performing a random walk within the problem's bounds, evaluating the sampled solutions, and combining feature extraction methods.
        # Args:
        - problem (object): An optimization problem instance that must have attributes `lb` (lower bounds), `ub` (upper bounds), and `optimum` (optional), as well as an `eval` method for evaluating solutions.
        # Returns:
        - np.ndarray or float: The computed state representation, which is a combination of features extracted from the sampled solutions and their costs.
        # Notes:
        - Increments the function evaluation counter (`self.fes`) by the number of samples evaluated.
        - Uses either a simple or progressive random walk to generate samples.
        - Applies feature extraction methods `__DFDC` and `__DRIE` to the sampled data.
        """
        # random walk
        # sample = self.simple_random_walk(lb=problem.lb, ub=problem.ub)
        sample = self.__progressive_random_walk(lb=problem.lb, ub=problem.ub)
        if problem.optimum is None:
            sample_cost = problem.eval(sample)
        else:
            sample_cost = problem.eval(sample) - problem.optimum
        self.fes += sample.shape[0]
        # get state
        return self.__DFDC(sample, sample_cost) * 1 + self.__DRIE(sample_cost) * 2

    def update(self, action, problem):
        """
        # Introduction
        Updates the optimizer's population based on the selected action and the given problem instance. This method performs mutation, crossover, selection, and updates the best solution found so far. It also manages logging, reward calculation, and meta-data collection for reinforcement learning-based hyper-parameter search.
        # Args:
        - action (int): The action index specifying which mutation and crossover strategy to use.
        - problem (object): The problem instance providing evaluation, lower/upper bounds, and optimum value.
        # Returns:
        - state (object): The updated state representation for the RL agent.
        - reward (float): The reward signal based on the proportion of improved solutions.
        - done (bool): Whether the optimization process has reached its termination condition.
        - info (dict): Additional information (currently empty).
        # Raises:
        - ValueError: If the provided `action` is not a valid index for the available strategies.
        """
        
        population = self.__population
        NP, dim = population.NP, population.dim
        # Mu
        if action == 0:
            Cr, F = population.choose_F_Cr("cauchy")
            v = cur_to_rand_1(population.group, F, rng=self.rng)
        elif action == 1:
            Cr, F = population.choose_F_Cr("cauchy")
            v = cur_to_best_1(population.group, population.gbest_solution, F, rng=self.rng)
        elif action == 2:
            Cr, F = population.choose_F_Cr("levy")
            v = cur_to_rand_1(population.group, F, rng=self.rng)
        elif action == 3:
            Cr, F = population.choose_F_Cr("levy")
            v = cur_to_best_1(population.group, population.gbest_solution, F, rng=self.rng)
        else:
            raise ValueError(f'action error: {action}')
        # BC
        v = clipping(v, problem.lb, problem.ub)
        # Cr
        u = binomial(population.group, v, Cr,self.rng)
        # Selection
        if problem.optimum is None:
            ncost = problem.eval(u)
        else:
            ncost = problem.eval(u) - problem.optimum
        self.fes += NP
        optim = np.where(ncost < population.cost)[0]
        SF = F[optim]
        SCr = Cr[optim]
        df = np.maximum(0, population.cost - ncost)[optim]
        population.update_M_F_Cr(SF, SCr, df)

        population.group[optim] = u[optim]
        population.cost = np.minimum(population.cost, ncost)
        # update gbest
        if population.cost.min() < population.gbest:
            population.gbest = population.cost.min()
            population.gbest_solution = population.group[np.argmin(population.cost)]
        population.LPSR(fes=self.fes, maxFEs=self.__maxFEs)  # be sorted at the same time

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(population.gbest)

        reward = optim.shape[0] / NP
        if problem.optimum is None:
            done = self.fes >= self.__maxFEs
        else:
            done = self.fes >= self.__maxFEs 

        if self.__config.full_meta_data:
            self.meta_X.append(population.group.copy())
            self.meta_Cost.append(population.cost.copy())

        if done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = population.gbest
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(population.gbest)
                
        info = {}        
        
        return self.__get_state(problem), reward, done , info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)

def binomial(x: np.ndarray, v: np.ndarray, Cr: Union[np.ndarray, float], rng) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(1, -1)
        v = v.reshape(1, -1)
    NP, dim = x.shape
    jrand = rng.randint(dim, size = NP)
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    u = np.where(rng.rand(NP, dim) < Cr, v, x)
    u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
    if u.shape[0] == 1:
        u = u.squeeze(axis = 0)
    return u

def generate_random_int(NP: int, cols: int, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a matrix of random integers for use in mutation operations, ensuring that each row contains unique values and no value matches its row index.
    # Args:
    - NP (int): Population size, determines the number of rows and the range of random integers [0, NP-1].
    - cols (int): Number of random integers to generate for each individual (number of columns).
    - rng (np.random.RandomState, optional): Random number generator instance. If None, a default RNG should be used.
    # Returns:
    - np.ndarray: A (NP, cols) shaped matrix of random integers, where each row contains unique values and no value equals its row index.
    # Raises:
    - ValueError: If NP or cols is not a positive integer.
    """
    r = rng.randint(low = 0, high = NP, size = (NP, cols))
    # validity checking and modification for r
    for col in range(0, cols):
        while True:
            is_repeated = [np.equal(r[:, col], r[:, i]) for i in range(col)]
            is_repeated.append(np.equal(r[:, col], np.arange(NP)))
            repeated_index = np.nonzero(np.any(np.stack(is_repeated), axis = 0))[0]
            repeated_sum = repeated_index.size
            if repeated_sum != 0:
                r[repeated_index[:], col] = rng.randint(low = 0, high = NP, size = repeated_sum)
            else:
                break
    return r

def cur_to_best_1(x: np.ndarray, best: np.ndarray, F: Union[np.ndarray, float], rng: np.random.RandomState = None) -> np.ndarray:
    """
    :param x: The 2-D population matrix of shape [NP, dim].
    :param best: An array of the best individual of shape [dim].
    :param F: The mutation factor, which could be a float or a 1-D array of shape[NP].
    """
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 2, rng = rng)
    return x + F * (best - x + x[r[:, 0]] - x[r[:, 1]])

def cur_to_rand_1(x: np.ndarray, F: Union[np.ndarray, float], rng: np.random.RandomState = None) -> np.ndarray:
    if isinstance(F, np.ndarray) and F.ndim == 1:
        F = F.reshape(-1, 1)
    r = generate_random_int(x.shape[0], 3, rng = rng)
    return x + F * (x[r[:, 0]] - x + x[r[:, 1]] - x[r[:, 2]])
