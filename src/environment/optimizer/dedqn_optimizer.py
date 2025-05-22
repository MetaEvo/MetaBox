from .learnable_optimizer import Learnable_Optimizer
import numpy as np
from typing import Union, Iterable

def cal_fdc(sample, fitness):
    """
    # Introduction
    Calculates the fitness distance correlation (FDC) for a given set of samples and their corresponding fitness values. FDC is a metric used to assess the relationship between the fitness of solutions and their distance to the best solution in the sample set.
    # Args:
    - sample (np.ndarray): An array of candidate solutions, where each row represents a solution.
    - fitness (np.ndarray): A 1D array of fitness values corresponding to each solution in `sample`.
    # Returns:
    - float: The computed fitness distance correlation coefficient.
    # Notes:
    - The function normalizes the correlation by the product of the variances of distance and fitness, with a small epsilon added to avoid division by zero.
    """
    
    best = np.argmin(fitness)
    distance = np.linalg.norm(sample-sample[best], axis=-1)
    cfd = np.mean((fitness - np.mean(fitness)) * (distance - np.mean(distance)))
    return cfd / (np.var(distance)*np.var(fitness) + 1e-6)


def cal_rie(fitness):
    """
    # Introduction
    Calculates the ruggedness of information entropy (RIE) of a given fitness sequence, which quantifies the complexity or unpredictability of changes in the sequence at multiple scales.
    # Args:
    - fitness (list or np.ndarray): A sequence of numerical fitness values representing the progression of a process or optimization.
    # Returns:
    - float: The maximum entropy value computed across different epsilon scales, representing the RIE of the input sequence.
    # Notes:
    - The function uses a multi-scale approach by varying the epsilon threshold to detect significant changes in the fitness sequence.
    - Entropy is normalized by the logarithm of the number of possible state transitions (6 in this case).
    - Zero frequencies are replaced with the length of the fitness sequence to avoid log(0) issues.
    """
    
    epsilon_star = 0
    for i in range(1,len(fitness)):
        if (fitness[i] - fitness[i-1]) > epsilon_star:
            epsilon_star = fitness[i] - fitness[i-1]
    # cal rie
    hs = []
    for k in range(9):
        epsilon = 0
        if k < 8:
            epsilon = epsilon_star / (2 ** k)
        s = []
        for i in range(len(fitness) - 1):
            if (fitness[i+1] - fitness[i]) < -epsilon:
                s.append(-1)
            elif (fitness[i+1] - fitness[i]) > epsilon:
                s.append(1)
            else:
                s.append(0)
        freq = np.zeros(6)
        for i in range(len(fitness) - 2):
            if s[i] == -1 and s[i+1] == 0:
                freq[0] += 1
            elif s[i] == -1 and s[i+1] == 1:
                freq[1] += 1
            elif s[i] == 0 and s[i+1] == 1:
                freq[2] += 1
            elif s[i] == 0 and s[i+1] == -1:
                freq[3] += 1
            elif s[i] == 1 and s[i+1] == -1:
                freq[4] += 1
            else:
                freq[5] += 1
        freq[freq == 0] = len(fitness)
        freq /= len(fitness)
        entropy = -np.sum(freq * np.log(freq) / np.log(6))
        hs.append(entropy)
    return max(hs)


def cal_acf(fitness):
    """
    # Introduction
    Calculates the autocorrelation coefficient (ACF) of a given fitness sequence.
    # Args:
    - fitness (np.ndarray or list of float): Sequence of fitness values for which the autocorrelation coefficient is to be computed.
    # Returns:
    - float: The computed autocorrelation coefficient of the input fitness sequence.
    # Notes:
    - A small constant (1e-6) is added to the denominator to prevent division by zero.
    """

    
    avg_f = np.mean(fitness)
    a = np.sum((fitness - avg_f) ** 2) + 1e-6
    acf = 0
    for i in  range(len(fitness) - 1):
        acf += (fitness[i] - avg_f) * (fitness[i + 1] - avg_f)
    acf /= a
    return acf


def cal_nop(sample, fitness):
    """
    # Introduction
    Calculates the number of optimal values (NOP) metric for a set of samples and their corresponding fitness values. The NOP metric quantifies how often the order of fitness values is preserved when samples are sorted by their distance to the best sample.
    # Args:
    - sample (np.ndarray): An array of sample points, where each row corresponds to a sample.
    - fitness (np.ndarray): A 1D array of fitness values corresponding to each sample.
    # Returns:
    - float: The normalized count of order-preserving pairs, representing the proportion of times a better fitness value follows a worse one when samples are sorted by distance to the best sample.
    # Raises:
    - None
    """

    
    best = np.argmin(fitness)
    distance = np.linalg.norm(sample - sample[best], axis=-1)
    data = np.stack([fitness, distance], axis=0)
    data = data.T
    data = data[np.argsort(data[:, 1]), :]
    fitness_sorted = data[:,0]
    r = 0
    for i in range(len(fitness) - 1):
        if fitness_sorted[i+1] < fitness_sorted[i]:
            r += 1
    return r / len(fitness)


def random_walk_sampling(population, dim, steps, rng):
    """
    # Introduction
    Generates a sequence of random walk samples within the bounds of a given population in a multi-dimensional space.
    # Args:
    - population (np.ndarray): The current population of solutions, shape (n_individuals, dim).
    - dim (int): The dimensionality of the search space.
    - steps (int): The number of steps (samples) to generate in the random walk.
    - rng (np.random.Generator): A NumPy random number generator instance for reproducibility.
    # Returns:
    - np.ndarray: An array of shape (steps, dim) containing the random walk samples, scaled to the range defined by the population.
    """
    
    pmin = np.min(population, axis=0)
    pmax = np.max(population, axis=0)
    walks = []
    start_point = rng.rand(dim)
    walks.append(start_point.tolist())
    for _ in range(steps - 1):
        move = rng.rand(dim)
        start_point = (start_point + move) % 1
        walks.append(start_point.tolist())
    return pmin + (pmax - pmin) * np.array(walks)


def cal_reward(survival, pointer):
    """
    # Introduction
    Calculates a custom reward based on the survival status of elements and a specified pointer index.
    # Args:
    - survival (list): A sequence where each element represents the survival status (typically 1 or another integer) of an entity.
    - pointer (int): The index in the survival list that is treated specially in the reward calculation.
    # Returns:
    - float: The computed reward value, normalized by the length of the survival list.
    # Notes:
    - If the element at the pointer index has a survival value of 1, the reward is incremented by 1.
    - For all other indices, the reward is incremented by the reciprocal of their survival value.
    - The final reward is averaged over the total number of elements in the survival list.
    """
    
    reward = 0
    for i in range(len(survival)):
        if i == pointer:
            if survival[i] == 1:
                reward += 1
        else:
            reward += 1/survival[i]
    return reward / len(survival)


class DEDQN_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    DEDQN is a mixed mutation strategy Differential Evolution (DE) algorithm based on deep Q-network (DQN), in which a deep reinforcement learning approach realizes the adaptive selection of mutation strategy in the evolution process.
    # Original paper
    "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing (2021).
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration, setting up key parameters and internal state variables for the optimization process.
        # Args:
        - config (object): Config object containing parameters for the optimizer。
            - Attributes needed for the DEDQN_Optimizer are the following:
                - log_interval (int): Interval at which logs are recorded.Default is config.maxFEs/config.n_logpoint.
                - n_logpoint (int): Number of log points to record.Default is 50.
                - full_meta_data (bool): Flag indicating whether to use full meta data.Default is False.
                - maxFEs (int): Maximum number of function evaluations.
                
        # Built-in Attribute:
        - self.__config: Stores the configuration object.
        - self.__NP: Population size for the optimizer. Default is 100.
        - self.__F: Mutation factor. Default is 0.5.
        - self.__Cr: Crossover rate. Default is 0.5.
        - self.__maxFEs: Maximum number of function evaluations.
        - self.__rwsteps: Number of random walk steps. Default is 100.
        - self.__solution_pointer: Index indicating which solution receives the action. Default is 0.
        - self.__population: Stores the current population of solutions. Default is None.
        - self.__cost: Stores the cost values for the population. Default is None.
        - self.__gbest: Stores the global best solution found. Default is None.
        - self.__gbest_cost: Stores the cost of the global best solution. Default is None.
        - self.__state: Stores the current state of the optimizer. Default is None.
        - self.__survival: Stores survival information for the population. Default is None.
        - self.fes: Tracks the number of function evaluations performed.Default is None.
        - self.cost: Tracks the cost values.Default is None.
        - self.log_index: Tracks the logging index.Default is None.
        - self.log_interval: Interval for logging progress.
        # Returns:
        - None
        """
        
        super().__init__(config)
        config.NP = 100
        config.F = 0.5
        config.Cr = 0.5
        config.rwsteps = config.NP
        self.__config = config

        self.__NP = config.NP
        self.__F = config.F
        self.__Cr = config.Cr
        self.__maxFEs = config.maxFEs
        self.__rwsteps = config.rwsteps
        self.__solution_pointer = 0 #indicate which solution receive the action
        self.__population = None
        self.__cost = None
        self.__gbest = None
        self.__gbest_cost = None
        self.__state = None
        self.__survival = None
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __cal_feature(self, problem):
        """
        # Introduction
        Calculates a set of feature descriptors for the current population in the optimization process, including FDC, RIE, ACF, and NOP, based on random walk sampling and cost evaluation.
        # Args:
        - problem (object): The optimization problem instance, which must provide an `eval` method for evaluating solutions and an `optimum` attribute for the known optimum value (if available).
        # Returns:
        - np.ndarray: A NumPy array containing the computed feature values [fdc, rie, acf, nop].
        # Raises:
        - AttributeError: If the `problem` object does not have the required `eval` method or `optimum` attribute.
        """
        
        samples = random_walk_sampling(self.__population, self.__dim, self.__rwsteps, self.rng)
        if problem.optimum is None:
            samples_cost = problem.eval(self.__population)
        else:
            samples_cost = problem.eval(self.__population) - problem.optimum
        # calculate fdc
        fdc = cal_fdc(samples, samples_cost)
        rie = cal_rie(samples_cost)
        acf = cal_acf(samples_cost)
        nop = cal_nop(samples, samples_cost)
        self.fes += self.__rwsteps
        return np.array([fdc, rie, acf, nop])

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimizer based on the provided problem definition. Sets up the initial population, evaluates their costs, determines the global best solution, and prepares meta-data if required.
        # Args:
        - problem (object):problem object, which have attributes `dim`, `lb`, `ub`, `optimum`, and a method `eval()` for evaluating solutions.
        # Built-in Attribute:
        - self.__dim (int): Dimensionality of the problem.
        - self.__population (np.ndarray): The initialized population within the problem bounds.
        - self.__survival (np.ndarray): Survival status of each individual in the population.
        - self.__cost (np.ndarray): Cost values of the population.
        - self.__gbest (np.ndarray): The best solution found so far.
        - self.__gbest_cost (float): The cost of the best solution.
        - self.fes (int): Function evaluation count.
        - self.log_index (int): Logging index for tracking progress. Default is 1.
        - self.cost (list): History of global best costs.
        - self.__state (np.ndarray): Feature vector representing the current state.
        - self.meta_X (list, optional): History of populations (if full meta-data is enabled).
        - self.meta_Cost (list, optional): History of costs (if full meta-data is enabled).
        # Returns:
        - np.ndarray: The feature vector representing the current state of the optimizer.
        # Raises:
        - None explicitly, but may raise exceptions from problem.eval() or if problem attributes are missing.
        """
        
        self.__dim = problem.dim
        self.__population = self.rng.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb  # [lb, ub]
        self.__survival = np.ones(self.__population.shape[0])
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__gbest = self.__population[self.__cost.argmin()]
        self.__gbest_cost = self.__cost.min()
        self.fes = self.__NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]
        self.__state = self.__cal_feature(problem)

        if self.__config.full_meta_data:
            self.meta_X = [self.__population.copy()]
            self.meta_Cost = [self.__cost.copy()]

        return self.__state

    def update(self, action, problem):
        """
        # Introduction
        Updates the current solution in the population using a specified mutation and crossover strategy, evaluates the new solution, updates the best solution found, and manages logging and meta-data for the optimization process.
        # Args:
        - action (int): The index of the mutation strategy to use (0: rand/1, 1: current-to-rand/1, 2: best/2).
        - problem (object): The optimization problem instance, which must provide lower and upper bounds (`lb`, `ub`), an evaluation function (`eval`), and optionally an optimum value (`optimum`).
        # Returns:
        - state (np.ndarray): The current feature state of the optimizer.
        - reward (float): The reward calculated for the current update step.
        - is_done (bool): Whether the optimization process has reached its termination condition.
        - info (dict): Additional information (currently empty).
        # Raises:
        - ValueError: If the provided `action` is not one of the supported mutation strategies (0, 1, or 2).
        """
        
        # mutate first
        if action == 0:
            u = rand_1_single(self.__population, self.__F, self.__solution_pointer, rng=self.rng)
        elif action == 1:
            u = cur_to_rand_1_single(self.__population, self.__F, self.__solution_pointer, rng=self.rng)
        elif action == 2:
            u = best_2_single(self.__population, self.__gbest, self.__F, self.__solution_pointer, rng=self.rng)
        else:
            raise ValueError(f'action error: {action}')
        # BC
        u = clipping(u, problem.lb, problem.ub)
        # then crossover
        u = binomial(self.__population[self.__solution_pointer], u, self.__Cr, self.rng)
        # select from u and x
        if problem.optimum is None:
            u_cost = problem.eval(u)
        else:
            u_cost = problem.eval(u) - problem.optimum
        self.fes += self.__NP
        if u_cost <= self.__cost[self.__solution_pointer]:
            self.__population[self.__solution_pointer] = u
            self.__cost[self.__solution_pointer] = u_cost
            self.__survival[self.__solution_pointer] = 1
            if u_cost < self.__gbest_cost:
                self.__gbest = u
                self.__gbest_cost = u_cost
        else:
            self.__survival[self.__solution_pointer] += 1
        self.__state = self.__cal_feature(problem)

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__gbest_cost)

        reward = cal_reward(self.__survival, self.__solution_pointer)

        
        if problem.optimum is None:
            is_done = self.fes >= self.__maxFEs
        else:
            is_done = self.fes >= self.__maxFEs

        if self.__config.full_meta_data:
            self.meta_X.append(self.__population.copy())
            self.meta_Cost.append(self.__cost.copy())

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__gbest_cost
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__gbest_cost)
        self.__solution_pointer = (self.__solution_pointer + 1) % self.__population.shape[0]
        info = {}
        return self.__state, reward, is_done , info

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
    jrand = rng.randint(dim, size=NP)
    if isinstance(Cr, np.ndarray) and Cr.ndim == 1:
        Cr = Cr.reshape(-1, 1)
    u = np.where(rng.rand(NP, dim) < Cr, v, x)
    u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
    if u.shape[0] == 1:
        u = u.squeeze(axis=0)
    return u

def generate_random_int_single(NP: int, cols: int, pointer: int, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a random array of integers within a specified range, ensuring that a given pointer value is not included in the result.
    # Args:
    - NP (int): The upper bound (exclusive) for the random integers.
    - cols (int): The number of random integers to generate.
    - pointer (int): The integer value that must not appear in the generated array.
    - rng (np.random.RandomState, optional): A random number generator instance. Defaults to None.
    # Returns:
    - np.ndarray: An array of randomly generated integers of length `cols`, excluding the `pointer` value.
    """
    r = rng.randint(low=0, high=NP, size=cols)
    while pointer in r:
        r = rng.randint(low=0, high=NP, size=cols)
    return r

def rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a new candidate vector using the DE/rand/1 mutation strategy for Differential Evolution (DE) optimization algorithms.
    # Args:
    - x (np.ndarray): Population array of candidate solutions, where each row represents an individual.
    - F (float): Differential weight, a scaling factor for the mutation.
    - pointer (int): Index of the current target vector in the population.
    - r (np.ndarray, optional): Array of three unique indices for mutation. If None, random indices are generated.
    - rng (np.random.RandomState, optional): Random number generator for reproducibility.
    # Returns:
    - np.ndarray: The mutated vector generated by the DE/rand/1 strategy.
    # Raises:
    - ValueError: If the generated or provided indices in `r` are not unique or include the `pointer` index.
    """
    
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer,rng=rng)
    return x[r[0]] + F * (x[r[1]] - x[r[2]])

def best_2_single(x: np.ndarray, best: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a new candidate solution vector using the "best/2" differential evolution strategy. This method perturbs the current best solution by adding the weighted difference of two pairs of randomly selected individuals from the population.
    # Args:
    - x (np.ndarray): The population array of candidate solutions.
    - best (np.ndarray): The current best solution vector.
    - F (float): The differential weight (scaling factor) applied to the difference vectors.
    - pointer (int): The index of the target individual in the population.
    - r (np.ndarray, optional): An array of 4 unique random indices for selecting individuals from the population. If None, random indices are generated.
    - rng (np.random.RandomState, optional): Random number generator for reproducibility. If None, the default NumPy RNG is used.
    # Returns:
    - np.ndarray: The newly generated candidate solution vector.
    # Raises:
    - ValueError: If the population size is less than 4 or if invalid indices are provided.
    """
    
    if r is None:
        r = generate_random_int_single(x.shape[0], 4, pointer, rng=rng)
    return best + F * (x[r[0]] - x[r[1]] + x[r[2]] - x[r[3]])

def cur_to_rand_1_single(x: np.ndarray, F: float, pointer: int, r: np.ndarray = None, rng: np.random.RandomState = None) -> np.ndarray:
    """
    # Introduction
    Generates a new candidate vector using the "current-to-rand/1" mutation strategy, commonly used in Differential Evolution (DE) algorithms.
    # Args:
    - x (np.ndarray): Population array of candidate solutions, where each row represents an individual.
    - F (float): Differential weight, a scaling factor for the mutation.
    - pointer (int): Index of the current target vector in the population.
    - r (np.ndarray, optional): Array of three unique random indices for mutation. If None, random indices are generated.
    - rng (np.random.RandomState, optional): Random number generator for reproducibility. If None, the global numpy RNG is used.
    # Returns:
    - np.ndarray: The mutated candidate vector generated by the current-to-rand/1 strategy.
    # Raises:
    - ValueError: If the generated or provided indices in `r` are not unique or include `pointer`.
    """
    
    if r is None:
        r = generate_random_int_single(x.shape[0], 3, pointer, rng=rng)
    return x[pointer] + F * (x[r[0]] - x[pointer] + x[r[1]] - x[r[2]])
