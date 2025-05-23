from .learnable_optimizer import Learnable_Optimizer
import numpy as np
from typing import Union, Iterable

# a function for optimizer to calculate reward
def cal_reward(f_old, f_new, d_old, d_new):
    if f_new < f_old and d_new > d_old:
        return 2
    if f_new < f_old and d_new <= d_old:
        return 1
    if f_new >= f_old and d_new > d_old:
        return 0
    if f_new >= f_old and d_new <= d_old:
        return -2


class QLPSO_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    QLPSO is a problem-free PSO which integrates a reinforcement learning method.
    # Original paper
    "[**A reinforcement learning-based communication topology in particle swarm optimization**](https://link.springer.com/article/10.1007/s00521-019-04527-9)." Neural Computing and Applications (2020).
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the QLPSO optimizer with the provided configuration, setting up hyperparameters and internal state variables required for optimization.
        # Args:
        - config (object): Config object containing optimizer parameters.
            - The Attributes needed for QLPSO are the following:
                - maxFEs: Maximum function evaluations allowed.
                - n_logpoint: Number of log points for cost history.
                - log_interval: Interval for logging progress
                - full_meta_data: Flag to enable/disable full metadata logging
        # Built-in Attributes:
        - `self.__config`: Configuration object containing optimizer parameters.
        - `self.__C`: Cognitive coefficient for PSO. Default is 1.49618.
        - `self.__W`: Inertia weight for PSO. Default is 0.729844.
        - `self.__NP`: Population size for PSO. Default is 30.
        - `self.solution_pointer`: Pointer to the current solution in the population.
        - `self.__population`: Current population of solutions.
        - `self.__pbest`: Personal best positions of the population.
        - `self.__velocity`: Current velocities of the population.
        - `self.__cost`: Current costs of the population.
        - `self.__gbest_cost`: Global best cost found so far.
        - `self.__diversity`: Diversity of the population.
        - `self.__state`: State of each individual in the population.
        - `self.fes`: Function evaluation count.
        - `self.cost`: List of costs that need to be maintained by every backbone optimizer.
        - `self.log_index`: Logging index for tracking progress.
        
        
        
        """
        
        super().__init__(config)
        # define hyperparameters that backbone optimizer needs
        config.NP = 30
        config.C = 1.49618
        config.W = 0.729844
        self.__config = config

        self.__C = config.C
        self.__W = config.W
        self.__NP = config.NP
        self.__maxFEs = config.maxFEs
        self.__solution_pointer = 0  # indicate which solution receive the action
        self.__population = None
        self.__pbest = None
        self.__velocity = None
        self.__cost = None
        self.__gbest_cost = None
        self.__diversity = None
        self.__state = None
        self.fes = None
        self.cost = None  # a list of costs that need to be maintained by EVERY backbone optimizers
        self.log_index = None
        self.log_interval = config.log_interval

    def __cal_diversity(self):
        """
        # Introduction
        Calculates the diversity of the current population in the optimizer.
        The diversity is computed as the mean Euclidean distance of each individual in the population from the population mean.
        # Returns:
        - float: The average Euclidean distance representing the diversity of the population.
        """
        
        return np.mean(np.sqrt(np.sum(np.square(self.__population - np.mean(self.__population,0)),1)))

    def __cal_velocity(self, action):
        """
        # Introduction
        Calculates the updated velocity vector for a particle in the QLPSO (Quantum-behaved Learning Particle Swarm Optimization) algorithm based on the selected neighborhood size determined by the `action` parameter.
        # Args:
        - action (int): Determines the neighborhood size for local best selection. Possible values are:
            - 0: Neighborhood size 4
            - 1: Neighborhood size 8
            - 2: Neighborhood size 16
            - 3: Neighborhood size 30
        # Returns:
        - np.ndarray: The updated velocity vector for the current particle.
        # Notes:
        - The method uses the current particle's position, velocity, personal best, and the best position found in the selected neighborhood to compute the new velocity.
        - The random number generator (`self.rng.rand()`) is used for stochastic updates.
        """
        
        i = self.__solution_pointer
        x = self.__population[i]
        v = self.__velocity[i]
        k = 0
        # calculate neighbour indexes
        if action == 0:
            k=4
        if action == 1:
            k=8
        if action == 2:
            k=16
        if action == 3:
            k=30

        nbest = None
        nbest_cost = np.inf
        for j in range(-k//2,k//2+1):
            if self.__cost[(i+j) % self.__population.shape[0]] < nbest_cost:
                nbest_cost = self.__cost[(i+j) % self.__population.shape[0]]
                nbest = self.__population[(i+j) % self.__population.shape[0]]
        return self.__W * v \
               + self.__C * self.rng.rand() * (nbest - x) \
               + self.__C * self.rng.rand() * (self.__pbest[i] - x)

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and related attributes for the QLPSO optimizer based on the provided optimization problem.
        # Args:
        - problem: An object representing the optimization problem, which must have attributes `ub` (upper bounds), `lb` (lower bounds), `optimum` (optional known optimum), and an `eval` method for evaluating solutions.
        # Built-in Attributes:
        - `self.__dim`: Dimension of the problem.
        - `self.__population`: Current population of solutions.
        - `self.__pbest`: Personal best positions of the population.
        - `self.__velocity`: Current velocities of the population.
        - `self.__cost`: Current costs of the population.
        - `self.__gbest_cost`: Global best cost found so far.
        - `self.__diversity`: Diversity of the population.
        - `self.__state`: State of each individual in the population.
        - `self.fes`: Function evaluation count.
        - `self.cost`: List of costs that need to be maintained by every backbone optimizer.
        - `self.log_index`: Logging index for tracking progress.Default is 1.
        - `self.cost`: List of costs that need to be maintained by every backbone optimizer.
        - `self.__state`: State of each individual in the population.
        - `self.meta_X`: List to store population positions for metadata logging.
        - `self.meta_Cost`: List to store population costs for metadata logging.
        - `self.meta_tmp_x`: Temporary list to store population positions for metadata logging.
        - `self.meta_tmp_cost`: Temporary list to store population costs for metadata logging.
        
        # Returns:
        - int: The state value of the solution pointer after initialization.
        
        # Notes:
        - If the problem's optimum is provided, the cost is offset by this value.
        - If `full_meta_data` is enabled in the configuration, additional metadata is stored for analysis.
        """
        self.__dim = problem.dim
        self.__population = self.rng.rand(self.__NP, self.__dim) * (problem.ub - problem.lb) + problem.lb  # [lb, ub]
        self.__pbest = self.__population.copy()
        self.__velocity = np.zeros(shape=(self.__NP, self.__dim))
        self.__diversity = self.__cal_diversity()
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__gbest_cost = self.__cost.min().copy()
        self.fes = self.__NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]
        self.__state = self.rng.randint(low=0, high=4, size=self.__NP)
        if self.__config.full_meta_data:
            self.meta_X = [self.__population.copy()]
            self.meta_Cost = [self.__cost.copy()]
            self.meta_tmp_x = []
            self.meta_tmp_cost = []

        return self.__state[self.__solution_pointer]

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the optimizer by applying the given action to the current solution, evaluating the new solution, updating rewards, and managing logging and metadata.
        # Args:
        - action (Any): The action to be applied to the current solution, typically representing a velocity or direction in the search space.
        - problem (object): The optimization problem instance, which must provide `lb`, `ub`, `eval()`, and `optimum` attributes/methods.
        # Returns:
        - state (Any): The updated state after applying the action.
        - reward (float): The calculated reward based on the change in cost and diversity.
        - is_done (bool): Whether the optimization episode should be terminated.
        - info (dict): Additional information (currently empty).
        # Notes:
        - The method updates internal population, velocity, cost, diversity, and logging information.
        - Handles boundary control and personal/global best updates.
        - Supports optional metadata logging if configured.
        - Ensures the cost log is filled up to the required number of log points at the end of an episode.
        """
        
        self.__velocity[self.__solution_pointer] = self.__cal_velocity(action)
        self.__population[self.__solution_pointer] += self.__velocity[self.__solution_pointer]
        # Boundary control
        self.__population[self.__solution_pointer] = clipping(self.__population[self.__solution_pointer], problem.lb, problem.ub)
        # calculate reward's data
        f_old = self.__cost[self.__solution_pointer]
        if problem.optimum is None:
            f_new = problem.eval(self.__population[self.__solution_pointer])
        else:
            f_new = problem.eval(self.__population[self.__solution_pointer]) - problem.optimum
        self.fes += 1
        d_old = self.__diversity
        d_new = self.__cal_diversity()
        reward = cal_reward(f_old,f_new,d_old,d_new)
        # update population information
        self.__cost[self.__solution_pointer] = f_new
        self.__diversity = d_new
        self.__gbest_cost = min(self.__gbest_cost, self.__cost.min().copy())
        if f_new < f_old:
            self.__pbest[self.__solution_pointer] = self.__population[self.__solution_pointer] #record pbest position
        self.__state[self.__solution_pointer] = action

        if self.__config.full_meta_data:
            self.meta_tmp_x.append(self.__population[self.__solution_pointer].copy())
            self.meta_tmp_cost.append(self.__cost[self.__solution_pointer].copy())

            # 在某一轮迭代结束后（例如在 for j in range(NP) 之后）
            if len(self.meta_tmp_cost) == self.__NP:  # 或 len(self.meta_tmp_x) == NP
                self.meta_X.append(np.array(self.meta_tmp_x))
                self.meta_Cost.append(np.array(self.meta_tmp_cost))

                self.meta_tmp_x.clear()
                self.meta_tmp_cost.clear()

        self.__solution_pointer = (self.__solution_pointer + 1) % self.__NP

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__gbest_cost)
        # if an episode should be terminated
        if problem.optimum is None:
            is_done = self.fes >= self.__maxFEs
        else:
            is_done = self.fes >= self.__maxFEs

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__gbest_cost
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__gbest_cost)
                
        info = {}
        return self.__state[self.__solution_pointer], reward, is_done , info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)
