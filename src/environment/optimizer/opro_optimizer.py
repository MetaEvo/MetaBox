import numpy as np
import torch
from .learnable_optimizer import Learnable_Optimizer

def scale(x,lb,ub):
    """
    Scales the input tensor `x` to a specified range [`lb`, `ub`] using the sigmoid function.
    # Introduction
    Applies the sigmoid activation to `x`, then linearly scales the result to the interval [`lb`, `ub`].
    # Args:
    - x (torch.Tensor): The input tensor to be scaled.
    - lb (float or torch.Tensor): The lower bound of the target range.
    - ub (float or torch.Tensor): The upper bound of the target range.
    # Returns:
    - torch.Tensor: The scaled tensor with values in the range [`lb`, `ub`].
    """
    
    x=torch.sigmoid(x)
    x=lb+(ub-lb)*x
    return x

def np_scale(x,lb,ub):
    """
    # Introduction
    Scales the input value(s) `x` to a specified range [`lb`, `ub`] using a sigmoid transformation.
    # Args:
    - x (float or np.ndarray): Input value or array of values to be scaled.
    - lb (float): Lower bound of the target range.
    - ub (float): Upper bound of the target range.
    # Returns:
    - float or np.ndarray: Scaled value(s) in the range [`lb`, `ub`].
    # Notes:
    - The function first applies the sigmoid function to `x`, mapping it to (0, 1), and then linearly scales it to the range [`lb`, `ub`].
    """
    
    x=1/(1 + np.exp(-x))
    x=lb+(ub-lb)*x
    return x

class OPRO_Optimizer(Learnable_Optimizer):
    def __init__(self, config):
        """
        Initializes the optimizer with the given configuration.
        # Args:
        - config (dict): Configuration parameters for the optimizer.
            - The attributes needed for the OPRO are the following:
                - pop_size (int): Population size for the optimizer. Default is 10.
                - full_meta_data (bool): Flag indicating whether to use full meta data. Default is False.
                - log_interval (int): Interval at which logs are recorded. Default is config.maxFEs/config.n_logpoint.
                - n_logpoint (int): Number of log points to record. Default is 50.
                - __FEs (int): Counter for the number of function evaluations. Default is 0.
        # Built-in Attributes:
        - config (dict): Stores the configuration parameters.
        - pop_size (int): The population size, default is 10.
        - fes (Any): Function evaluation state, initialized as None.
        - best (Any): Stores the best solution found, initialized as None.
        - old_value_pairs (Any): Stores previous value pairs, initialized as None.
        - cost (list): List to store cost values, initialized as an empty list.
        """
        
        super().__init__(config)
        self.config = config
        self.pop_size = 10
        self.fes = None
        self.best = None
        self.old_value_pairs = None
        self.cost = []
    def __str__(self):
        """
        Returns a string representation of the OPRO_Optimizer object.
        # Returns:
            str: The name of the optimizer, "OPRO_Optimizer".
        """
        
        return "OPRO_Optimizer"

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimization problem, evaluates their fitness, and stores relevant metadata.
        # Args:
        - problem (object): An optimization problem instance that provides lower and upper bounds (`lb`, `ub`), dimensionality (`dim`), and an evaluation function (`func`). The problem should also have a `reset()` method.
        # Built-in Attribute:
        - self.population (np.ndarray): The initialized population of candidate solutions.
        - self.old_value_pairs (list): List of tuples containing each individual and its corresponding fitness value.
        - self.fes (int): Counter for function evaluations, reset to 0.
        - self.best (float): The best fitness value found in the initial population.
        - self.cost (list): List tracking the best cost found at each step.
        - self.meta_X (list, optional): Stores population snapshots if `full_meta_data` is enabled in config.
        - self.meta_Cost (list, optional): Stores cost snapshots if `full_meta_data` is enabled in config.
        # Returns:
        - list: A list of tuples, each containing a population member and its corresponding fitness value.
        # Raises:
        - AttributeError: If the `problem` object does not have required attributes or methods.
        """
        
        problem.reset()
        self.old_value_pairs = []
        self.fes=0
        self.best=None
    #     init population here and return the best popsize pop when step
        self.population = self.rng.random.uniform(
            low=problem.lb, high=problem.ub, size=(self.pop_size, problem.dim)
        )
        y = problem.func(self.population)
        # find the best theta and y
        best_theta = self.population[np.argmin(y)]
        self.best = np.min(y)
        self.cost.append(self.best)
        for i in range(self.pop_size):
            self.old_value_pairs.append((self.population[i], y[i]))

        if self.config.full_meta_data:
            self.meta_X = [self.population.copy()]
            self.meta_Cost = [y.copy()]

        return self.old_value_pairs

    def get_old_value_pairs(self):
        return self.old_value_pairs
    def update(self,action,problem):
        """
        # Introduction
        Updates the optimizer's state with new candidate solutions (thetas) by evaluating them on the given problem, updating the best found solution, and maintaining historical data for meta-learning.
        # Args:
        - action (list or np.ndarray): A list or array of new candidate solutions (thetas) to evaluate.
        - problem (object): An object representing the optimization problem, which must implement an `eval` method to evaluate candidate solutions.
        # Built-in Attribute:
        - self.best (float): The best objective value found so far.
        - self.old_value_pairs (list): List of tuples containing previous thetas and their evaluated values.
        - self.config.full_meta_data (bool): Flag indicating whether to store full meta-data for each generation.
        - self.meta_Cost (list): List of arrays containing costs for meta-learning.
        - self.meta_X (list): List of arrays containing thetas for meta-learning.
        - self.cost (list): List of best costs found at each update.
        - self.fes (int): Counter for the number of function evaluations.
        # Returns:
        - self.old_value_pairs (list): Updated list of (theta, value) pairs.
        - int: Always 0 (placeholder for compatibility).
        - bool: Always False (placeholder for compatibility).
        - info (dict): Additional information (currently empty).
        # Raises:
        - None explicitly, but may raise exceptions if `problem.eval` fails or if input shapes are incompatible.
        """
        
        new_thetas = action

        if len(new_thetas) > 0:
            new_thetas = np.stack(new_thetas)
            # evaluate the new theta
            new_y = problem.eval(new_thetas)
            # update the best theta and y
            cur_best_theta = new_thetas[np.argmin(new_y)]
            cur_best_y = np.min(new_y)
            if cur_best_y < self.best or self.best is None:
                best_theta = cur_best_theta
                self.best = cur_best_y
            # add to old_value_pairs_set
            for i in range(len(new_thetas)):
                self.old_value_pairs.append((new_thetas[i], new_y[i]))

            if self.config.full_meta_data:
                gen_meta_cost = []
                gen_meta_X = []
                for i in range(len(new_thetas)):
                    gen_meta_cost.append(new_y[i])
                    gen_meta_X.append(new_thetas[i])
                self.meta_Cost.append(np.array(gen_meta_cost).copy())
                self.meta_X.append(np.array(gen_meta_X).copy())

        self.cost.append(self.best)
        self.fes += len(new_thetas)

        info = {}

        return self.old_value_pairs, 0, False, info