from typing import Any

from .learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np

# torch
class B2OPT_Optimizer(Learnable_Optimizer):
    """
    # Introduction
      B2Opt: Learning to Optimize Black-box Optimization with Little Budget.
    # Original paper
    "[**B2Opt: Learning to Optimize Black-box Optimization with Little Budget**](https://arxiv.org/abs/2304.11787)". arXiv preprint arXiv:2304.11787, (2023).
    # Official Implementation
    [B2Opt](https://github.com/ninja-wm/B2Opt)
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration, setting up population size, evaluation limits, logging parameters, and internal state variables.
        # Args:
        - config (object): Configuration object.
            - The Attributes needed for the B2OPT are the following:
                - maxFEs (int): Maximum number of function evaluations.
                - log_interval (int): Interval at which logs are recorded.Default is config.maxFEs/config.n_logpoint.
                - n_logpoint (int): Number of log points to record.Default is 50.
                - full_meta_data (bool): Flag indicating whether to use full meta data.Default is False.
                
        # Built-in Attribute:
        - NP (int): Population size, set to 100.
        - ems (int): Number of epochs or main steps, computed based on `MaxFEs` and `NP`.
        - fes (Any): Placeholder for the current number of function evaluations,which will be initialized in the `init_population` method.
        - cost (Any): Placeholder for the current cost or fitness value, which will be initialized in the `init_population` method.
        - log_index (Any): Placeholder for the logging index, which will be initialized in the `init_population` method.
        - log_interval (Any): Logging interval, taken from `config.log_interval`.Default is `config.maxFEs/config.n_logpoint`.
        - ems_index (int): Index for the current epoch or main step. Default is 0.
        # Returns:
        - None
        """
        
        super().__init__(config)
        self.config = config

        self.NP = 100

        self.MaxFEs = config.maxFEs
        self.ems = (self.MaxFEs + self.NP - 1) // self.NP - 1

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

        self.ems_index = 0

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the B2OPT optimizer instance.
        # Returns:
        - str: The name of the optimizer, "B2OPT_Optimizer".
        """
        
        return "B2OPT_Optimizer"

    def get_costs(self, position, problem):
        """
        # Introduction
        Computes the cost of a given position for a specified optimization problem, optionally normalizing by the known optimum.
        # Args:
        - position (Any): The candidate solution whose cost is to be evaluated.
        - problem (object): The optimization problem instance, expected to have `eval` and `optimum` attributes.
        # Returns:
        - torch.Tensor or float: The computed cost, converted to a torch.Tensor if originally a numpy array.
        # Raises:
        - AttributeError: If `problem` does not have the required `eval` method or `optimum` attribute.
        """
        
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum

        if isinstance(cost, np.ndarray):
            cost = torch.Tensor(cost)

        return cost

    def __sort(self):
        """
        # Introduction
        Sorts the population and corresponding cost values in ascending order based on the cost.
        # Built-in Attribute:
        - self.population (torch.Tensor): The current population of solutions.
        - self.c_cost (torch.Tensor): The cost values associated with each member of the population.
        # Returns:
        None. Updates `self.population` and `self.c_cost` in-place to reflect the sorted order.
        """
        
        _, index = torch.sort(self.c_cost)
        self.population = self.population[index]
        self.c_cost = self.c_cost[index]

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimizer based on the given problem's bounds and dimensionality. Sets up random number generators, evaluates initial costs, and prepares metadata for tracking optimization progress.
        # Args:
        - problem (object): An object representing the optimization problem.
            - `dim` (int): Dimensionality of the problem.
            - `lb` (torch.Tensor): Lower bounds for the variables.
            - `ub` (torch.Tensor): Upper bounds for the variables.
            - `eval(x)` (callable): Function to evaluate the objective at point `x`.
            - `optimum` (float or None): Known optimum value of the problem (if available).
        # Built-in Attribute:
        - self.population (torch.Tensor): The initialized population of candidate solutions.
        - self.c_cost (torch.Tensor): The costs of the initial population.
        - self.gbest_val (float): The best cost found in the initial population.
        - self.init_gbest (torch.Tensor): The best cost tensor in the initial population.
        - self.cost (list): List tracking the best cost at each iteration.
        - self.meta_X (list, optional): List of population states for metadata (if enabled).
        - self.meta_Cost (list, optional): List of cost states for metadata (if enabled).
        # Returns:
        - dict: The current state of the optimizer after population initialization,using get_state() method.
        # Raises:
        - None
        """
        dim = problem.dim
        self.rng_torch = self.rng_cpu
        if self.config.device != "cpu":
            self.rng_torch = self.rng_gpu

        self.fes = 0
        self.population = (problem.ub - problem.lb) * torch.rand((self.NP, dim), generator = self.rng_torch, device = self.config.device, dtype = torch.float64) + problem.lb
        self.c_cost = self.get_costs(position = self.population, problem = problem)

        self.fes += self.NP

        self.ems_index = 0 # opt ob pointer

        self.gbest_val = torch.min(self.c_cost).detach().cpu().numpy()

        self.init_gbest = torch.min(self.c_cost).detach().cpu()

        self.cost = [self.gbest_val]
        self.log_index = 1

        self.__sort()

        if self.config.full_meta_data:
            self.meta_X = [self.population.detach().cpu().numpy()]
            self.meta_Cost = [self.c_cost.detach().cpu().numpy()]

        return self.get_state()
    def get_state(self):
        """
        # Introduction
        Retrieves the current state of the optimizer, represented by the cost value.
        # Returns:
        - Any: The current cost value stored in `self.c_cost`.
        """
        
        Y = self.c_cost
        return Y

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the optimizer based on the given action and problem, and calculates the reward, next state, and termination condition.
        # Args:
        - action (callable): A policy network function that takes the current population, costs, and EMS index as input and returns updated positions.
        - problem (object): The optimization problem instance containing problem-specific details.
        # Returns:
        - next_state (torch.Tensor): The updated state of the optimizer.
        - reward (float): The reward calculated based on the improvement in the global best value.
        - is_end (bool): A flag indicating whether the optimization process has reached its termination condition.
        - info (dict): Additional information (currently empty).
        # Notes:
        - The method updates the population and costs based on the optimization process.
        - It calculates the reward as the relative improvement in the global best value compared to the initial best value.
        - The termination condition is determined by the maximum number of function evaluations (`MaxFEs`).
        - If `full_meta_data` is enabled in the configuration, the population and costs are logged for each step.
        - The global best value (`gbest_val`) is updated and logged at specified intervals.
        """

        # 这里的action 是policy 网络
        pre_gbest = torch.min(self.c_cost.detach().cpu())


        v = action(self.population[None, :].clone().detach(), self.c_cost[None, :].clone().detach(), self.ems_index)[0] # off
        self.ems_index += 1

        new_cost = self.get_costs(position = v, problem = problem)
        self.fes += self.NP

        old_population = self.population.clone().detach()
        old_c_cost = self.c_cost.clone().detach()
        optim = new_cost.detach() < old_c_cost

        old_population[optim] = v[optim]
        old_c_cost[optim] = new_cost[optim]
        self.population = old_population
        self.c_cost = old_c_cost

        new_gbest_val = torch.min(self.c_cost).detach().cpu()

        reward = (pre_gbest - new_gbest_val) / self.init_gbest

        new_gbest_val = new_gbest_val.numpy()

        self.gbest_val = np.minimum(self.gbest_val, new_gbest_val)

        if problem.optimum is None:
            is_end = self.fes >= self.MaxFEs
        else:
            is_end = self.fes >= self.MaxFEs

        self.__sort()

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


