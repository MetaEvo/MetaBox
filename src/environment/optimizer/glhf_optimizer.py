from typing import Any

from .learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np

# torch
class GLHF_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    GLHF: General Learned Evolutionary Algorithm Via Hyper Functions
    # Original paper
    "[**GLHF: General Learned Evolutionary Algorithm Via Hyper Functions**](https://arxiv.org/abs/2405.03728)." arXiv preprint arXiv:2405.03728 (2024).
    # Official Implementation
    [GLHF](https://github.com/ninja-wm/POM/)
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration and sets up key attributes for the optimization process.
        # Args:
        - config (object): Config object containing parameters for the optimizer.
            - The Attributes needed for the GLHF_Optimizer are the following:
                - maxFEs (int): Maximum number of function evaluations.
                - log_interval (int): Interval for logging progress.
                - full_meta_data (bool): Flag for whether to store full meta data.
                - device (str): Device to use for computations (e.g., "cpu", "cuda").
                - n_logpoint (int): Number of log points for logging.
                
        # Built-in Attributes:
        - config (object): Configuration object containing algorithm parameters.
        - NP (int): Population size, set to 100 by default.
        - MaxFEs (int): Maximum number of function evaluations.
        - fes (int): Counter for the number of function evaluations.
        - cost (list): List to store the best cost values during optimization.
        - log_index (int): Index for logging progress.
        - log_interval (int): Interval for logging progress.
        
        # Raises:
        - None
        """
        
        super().__init__(config)
        self.config = config

        self.NP = 100

        self.MaxFEs = config.maxFEs

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the GLHF_Optimizer object.
        # Returns:
        - str: The name of the optimizer, "GLHF_Optimizer".
        """
        
        return "GLHF_Optimizer"

    def get_costs(self, position, problem):
        """
        # Introduction
        Calculates the cost of a given position for a specified optimization problem. If the problem has a known optimum, the cost is computed as the difference between the evaluated position and the optimum; otherwise, it returns the evaluated value directly.
        # Args:
        - position (Any): The candidate solution or position to be evaluated.
        - problem (object): The optimization problem object, which has an `eval` method and an `optimum` attribute.
        # Returns:
        - float: The computed cost for the given position.
        # Raises:
        - AttributeError: If the `problem` object does not have the required `eval` method or `optimum` attribute.
        """
        
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum

        if isinstance(cost, np.ndarray):
            cost = torch.Tensor(cost)
        return cost

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimizer based on the problem's dimensionality and bounds, sets up random number generators according to the device configuration, evaluates the initial costs, and prepares metadata for tracking optimization progress.
        # Args:
        - problem (object): The optimization problem object, which has attributes `dim` (int), `ub` (upper bound, tensor or array), and `lb` (lower bound, tensor or array).
        # Returns:
        - torch.Tensor: A tensor where the first column is the cost (unsqueezed to shape [N, 1]) and the remaining columns are the population data, concatenated along dimension 1.as returned by `self.get_state()`.
        # Notes:
        - Updates internal attributes such as `self.population`, `self.c_cost`, `self.gbest_val`, `self.init_gbest`, `self.cost`, and optionally metadata attributes if `self.config.full_meta_data` is True.
        - Increments the function evaluation counter (`self.fes`) by the population size (`self.NP`).
        """
        
        dim = problem.dim
        self.rng_torch = self.rng_cpu
        if self.config.device != "cpu":
            self.rng_torch = self.rng_gpu

        self.fes = 0
        self.population = (problem.ub - problem.lb) * torch.rand((self.NP, dim), generator = self.rng_torch, device = self.config.device, dtype = torch.float64) + problem.lb
        self.c_cost = self.get_costs(position = self.population, problem = problem)

        self.fes += self.NP

        self.gbest_val = torch.min(self.c_cost).detach().cpu().numpy()

        self.init_gbest = torch.min(self.c_cost).detach().cpu()

        self.cost = [self.gbest_val]
        self.log_index = 1

        if self.config.full_meta_data:
            self.meta_X = [self.population.detach().cpu().numpy()]
            self.meta_Cost = [self.c_cost.detach().cpu().numpy()]

        return self.get_state()
    def get_state(self):
        """
        # Introduction
        Returns the current state of the optimizer by concatenating the cost and population tensors.
        # Returns:
        - torch.Tensor: A tensor where the first column is the cost (unsqueezed to shape [N, 1]) and the remaining columns are the population data, concatenated along dimension 1.
        """
        
        X = self.population
        Y = self.c_cost.unsqueeze(1)
        return torch.cat([Y, X], dim = 1)

    def update(self, action, problem):
        """
        # Introduction
        Updates the optimizer's population and cost values based on the provided action (typically a policy network) and the given problem instance. Calculates the reward, checks for termination, logs progress, and returns the next state and relevant information.
        # Args:
        - action (Callable): A function or policy network that takes the current population state and returns a new population.
        - problem (object): The problem instance containing the objective function and (optionally) the optimum value.
        # Returns:
        - next_state (torch.Tensor): A tensor where the first column is the cost (unsqueezed to shape [N, 1]) and the remaining columns are the population data, concatenated along dimension 1.as returned by `self.get_state()`.
        - reward (float): The normalized improvement in the global best cost.
        - is_end (bool): Flag indicating whether the optimization process has reached its end condition.
        - info (dict): Additional information (currently empty, but can be extended).
        # Notes:
        - Updates internal logging and meta-data if configured.
        - Handles both cases where the problem's optimum is known or unknown.
        """
        
        # 这里的action 是policy 网络
        pre_gbest = torch.min(self.c_cost.detach()).detach().cpu()
        batch_pop = self.get_state()[None, :].clone().detach()

        new_population = action(batch_pop)[0]
        new_cost = self.get_costs(position = new_population, problem = problem)

        old_population = self.population.clone().detach()
        old_c_cost = self.c_cost.clone().detach()
        optim = new_cost.detach() < old_c_cost.detach()

        old_population[optim] = new_population[optim]
        old_c_cost[optim] = new_cost[optim]

        self.population = old_population
        self.c_cost = old_c_cost
        self.fes += self.NP

        # self.population = new_population
        # self.c_cost = new_cost

        new_gbest_val = torch.min(self.c_cost).detach().cpu()

        reward = (pre_gbest - new_gbest_val) / self.init_gbest

        new_gbest_val = new_gbest_val.numpy()

        self.gbest_val = np.minimum(self.gbest_val, new_gbest_val)

        if problem.optimum is None:
            is_end = self.fes >= self.MaxFEs
        else:
            is_end = self.fes >= self.MaxFEs

        if self.config.full_meta_data:
            self.meta_X.append(self.population.detach().cpu().numpy())
            self.meta_Cost.append(self.c_cost.detach().cpu().numpy())

        next_state = self.get_state()

        if self.fes >= self.log_interval * self.log_index:
            self.log_index += 1
            self.cost.append(self.gbest_val)

        if is_end:
            if len(self.cost) >= self.config.n_logpoint + 1:
                self.cost[-1] = self.gbest_val
            else:
                while len(self.cost) < self.config.n_logpoint + 1:
                    self.cost.append(self.gbest_val)

        info = {}

        return next_state, reward, is_end, info
