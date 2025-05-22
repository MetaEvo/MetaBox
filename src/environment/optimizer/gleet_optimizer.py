from .learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np


class GLEET_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    GLEET is a **G**eneralizable **L**earning-based **E**xploration-**E**xploitation **T**radeoff framework, which could explicitly control the exploration-exploitation tradeoff hyper-parameters of a given EC algorithm to solve a class of problems via reinforcement learning. 
    # Original paper
    "[**Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning**](https://dl.acm.org/doi/abs/10.1145/3638529.3653996)." Proceedings of the Genetic and Evolutionary Computation Conference (2024).
    # Official Implementation
    [GLEET](https://github.com/GMC-DRL/GLEET)
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration and sets up internal parameters for optimization.
        # Args:
        - config (object): Config object containing optimizer settings.
            - The Attributes needed for the GLEET_Optimizer are the following:
                - log_interval (int): Interval at which logs are recorded.Default is config.maxFEs/config.n_logpoint.
                - n_logpoint (int): Number of log points to record.Default is 50.
                - full_meta_data (bool): Flag indicating whether to use full meta data.Default is False.
                - maxFEs (int): Maximum number of function evaluations.
                - __FEs (int): Counter for the number of function evaluations.Default is 0.
                - __config (object): Stores the config object from src/config.py.
                - PS (int): Population size.Default is 100.
        # Built-in Attribute:
        - self.__config (object): Stores the configuration object.
        - self.w_decay (bool): Flag to determine weight decay usage.Default is True.
        - self.w (float): Inertia weight, set based on `w_decay`.Default is 0.9 if `w_decay` is True, otherwise 0.729.
        - self.c (float): Acceleration coefficient.Default is 4.1.
        - self.reward_scale (int): Scaling factor for rewards.Default is 100.
        - self.ps (int): Population size or related parameter.Default is 100.
        - self.no_improve (int): Counter for iterations without improvement.Default is 0.
        - self.boarder_method (str): Method for handling boundaries.Default is 'clipping'.
        - self.reward_func (str): Reward function type.Default is 'direct'.
        - self.fes (Any): Tracks function evaluations (initialized as None).Default is None.
        - self.cost (Any): Tracks cost (initialized as None).Default is None.
        - self.log_index (Any): Logging index (initialized as None).Default is None.
        - self.log_interval (int): Interval for logging progress.
        # Returns:
        - None
        """
        
        super().__init__(config)
        self.__config = config

        self.w_decay = True
        if self.w_decay:
            self.w = 0.9
        else:
            self.w = 0.729
        self.c = 4.1

        self.reward_scale = 100

        self.ps = 100

        self.no_improve = 0

        self.max_fes = config.maxFEs

        self.boarder_method = 'clipping'
        self.reward_func = 'direct'

        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the GLEET optimizer instance.
        # Returns:
        - str: The name of the optimizer, "GLEET_Optimizer".
        """
        
        return "GLEET_Optimizer"

    # initialize GPSO environment
    def initialize_particles(self, problem):
        """
        # Introduction
        Initializes the particles for a particle swarm optimization (PSO) algorithm by generating random positions and velocities, evaluating initial costs, and setting up personal and global bests.
        # Args:
        - problem (object): The problem object, which has attributes `lb` (lower bounds), `ub` (upper bounds), and be compatible with the `get_costs` method.
        # Returns:
        - None: This method updates the internal state of the optimizer by initializing the `particles` attribute with positions, velocities, costs, and bests.
        # Notes:
        - The method uses the optimizer's random number generator (`self.rng`) and assumes the existence of attributes such as `ps` (particle size), `dim` (problem dimensionality), and `max_velocity`.
        - The `particles` dictionary stores all relevant information for each particle, including current and best positions, costs, velocities, and the global best.
        """
        
        # randomly generate the position and velocity
        self.dim = problem.dim
        rand_pos = self.rng.uniform(low = problem.lb, high = problem.ub, size = (self.ps, self.dim))
        rand_vel = self.rng.uniform(low = -self.max_velocity, high = self.max_velocity, size = (self.ps, self.dim))

        # get the initial cost
        c_cost = self.get_costs(rand_pos, problem)  # ps

        # find out the gbest_val
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]

        # record
        self.max_cost = np.min(c_cost)
        # store all the information of the paraticles
        self.particles = {'current_position': rand_pos.copy(),  # ps, dim
                          'c_cost': c_cost.copy(),  # ps
                          'pbest_position': rand_pos.copy(),  # ps, dim
                          'pbest': c_cost.copy(),  # ps
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'velocity': rand_vel.copy(),  # ps,dim
                          'gbest_index': gbest_index  # 1
                          }

    def get_cat_xy(self):
        """
        # Introduction
        Concatenates the current, personal best, and global best positions and their corresponding cost/fitness values for all particles in the optimizer.
        # Returns:
        - np.ndarray: A concatenated NumPy array containing the current positions and costs, personal best positions and values, and global best positions and values for all particles.
        # Notes:
        - The method assumes that the `self.particles` dictionary contains the keys: 'current_position', 'c_cost', 'pbest_position', 'pbest', 'gbest_position', and 'gbest_val'.
        - The concatenation is performed along the last axis for position-value pairs and along the first axis to combine all groups.
        """
        
        cur_x = self.particles['current_position']
        cur_y = self.particles['c_cost']
        cur_xy = np.concatenate((cur_x, cur_y), axis = -1)
        pbest_x = self.particles['pbest_position']
        pbest_y = self.particles['pbest']
        pbest_xy = np.concatenate((pbest_x, pbest_y), axis = -1)
        gbest_x = self.particles['gbest_position']
        gbest_y = self.particles['gbest_val']
        gbest_xy = np.concatenate((gbest_x, gbest_y), axis = -1)

        return np.concatenate((cur_xy, pbest_xy, gbest_xy), axis = 0)

    # the interface for environment reseting
    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and related state variables for the optimizer, preparing it for a new optimization run.
        # Args:
        - problem (object): An object representing the optimization problem, expected to have attributes `ub` (upper bounds) and `lb` (lower bounds) for the search space.
        # Built-in Attribute:
        - self.fes (int): Function evaluation steps, initialized to 0.
        - self.per_no_improve (np.ndarray): Array to track the number of iterations without improvement for each particle, initialized to zeros.
        - self.max_velocity (np.ndarray): Maximum velocity for each particle, calculated based on the problem's bounds.
        - self.max_dist (float): Maximum distance in the search space, calculated based on the problem's bounds.
        - self.no_improve (int): Counter for the number of iterations without improvement, initialized to 0.
        - self.log_index (int): Index for logging progress, initialized to 1.
        - self.cost (list): List to store the best cost found at each logging interval, initialized with the global best value.
        - self.pbest_feature (np.ndarray): Array to store the personal best features of the particles.
        - self.gbest_feature (np.ndarray): Array to store the global best features of the particles.
        - self.meta_X (list): List to store the positions of the particles for meta-data logging, if configured.
        - self.meta_Cost (list): List to store the costs of the particles for meta-data logging, if configured.
        # Returns:
        - np.ndarray: The concatenated state of the population, including both the population state and additional features, with shape (ps, 27).
        # Notes:
        - Resets various counters and state variables to their initial values.
        - Initializes particle positions and velocities.
        - Optionally stores meta-data if configured.
        - Prepares features for exploration and exploitation tracking.
        """
        
        self.fes = 0
        self.per_no_improve = np.zeros((self.ps,))
        self.max_velocity = 0.1 * (problem.ub - problem.lb)
        # set the hyperparameters back to init value if needed
        if self.w_decay:
            self.w = 0.9

        self.max_dist = np.sqrt(np.sum((problem.ub - problem.lb) ** 2))

        self.no_improve -= self.no_improve
        self.fes -= self.fes
        self.per_no_improve -= self.per_no_improve

        # initialize the population
        self.initialize_particles(problem)

        self.log_index = 1
        self.cost = [self.particles['gbest_val']]

        # get state

        # get the population state
        state = self.observe()  # ps, 9

        # get the exploration state
        self.pbest_feature = state.copy()  # ps, 9

        # get the explotation state
        self.gbest_feature = state[self.particles['gbest_index']]  # 9

        # get and return the total state (population state, exploration state, exploitation state)
        gp_cat = self.gp_cat()  # ps, 18

        if self.__config.full_meta_data:
            self.meta_X = [self.particles['current_position'].copy()]
            self.meta_Cost = [self.particles['c_cost'].copy()]

        return np.concatenate((state, gp_cat), axis = -1)  # ps, 9+18

    # calculate costs of solutions
    def get_costs(self, position, problem):
        """
        # Introduction
        Calculates the cost(s) for a given position or set of positions in the search space, updating the function evaluation count.
        # Args:
        - position (np.ndarray): The position(s) in the search space for which the cost is to be evaluated. Shape is typically (n_samples, n_dimensions).
        - problem (object): The optimization problem instance, which must provide an `eval` method and an optional `optimum` attribute.
        # Returns:
        - np.ndarray or float: The evaluated cost(s) for the given position(s). If `problem.optimum` is defined, returns the difference between the evaluated value and the optimum.
        # Notes:
        - Increments the `fes` (function evaluation steps) counter by the number of positions evaluated.
        """
        
        ps = position.shape[0]
        self.fes += ps
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    # feature encoding
    def observe(self):
        """
        # Introduction
        Computes and returns a set of normalized features representing the current state of the particle swarm optimizer. These features are used for monitoring or as input to learning-based optimization strategies.
        # Returns:
        - np.ndarray: A 2D array of shape (ps, 9), where each row contains the following normalized features for each particle:
            - fea0: Current cost normalized by maximum cost.
            - fea1: Difference between current cost and global best value, normalized by maximum cost.
            - fea2: Difference between current cost and personal best, normalized by maximum cost.
            - fea3: Remaining function evaluations normalized by maximum evaluations.
            - fea4: Number of iterations without improvement for each particle, normalized by maximum steps.
            - fea5: Number of iterations without improvement for the whole swarm, normalized by maximum steps.
            - fea6: Euclidean distance between current position and global best position, normalized by maximum distance.
            - fea7: Euclidean distance between current position and personal best position, normalized by maximum distance.
            - fea8: Cosine similarity between the vectors from current to personal best and from current to global best.
        # Notes:
        - Handles division by zero and NaN values in cosine similarity calculation.
        - Assumes all required attributes (such as `self.particles`, `self.max_cost`, etc.) are properly initialized.
        """
        
        max_step = self.max_fes // self.ps
        # cost cur
        fea0 = self.particles['c_cost'] / self.max_cost
        # cost cur_gbest
        fea1 = (self.particles['c_cost'] - self.particles['gbest_val']) / self.max_cost  # ps
        # cost cur_pbest
        fea2 = (self.particles['c_cost'] - self.particles['pbest']) / self.max_cost
        # fes cur_fes
        fea3 = np.full(shape = (self.ps), fill_value = (self.max_fes - self.fes) / self.max_fes)
        # no_improve  per
        fea4 = self.per_no_improve / max_step
        # no_improve  whole
        fea5 = np.full(shape = (self.ps), fill_value = self.no_improve / max_step)
        # distance between cur and gbest
        fea6 = np.sqrt(np.sum((self.particles['current_position'] - np.expand_dims(self.particles['gbest_position'], axis = 0)) ** 2, axis = -1)) / self.max_dist
        # distance between cur and pbest
        fea7 = np.sqrt(np.sum((self.particles['current_position'] - self.particles['pbest_position']) ** 2, axis = -1)) / self.max_dist

        # cos angle
        pbest_cur_vec = self.particles['pbest_position'] - self.particles['current_position']
        gbest_cur_vec = np.expand_dims(self.particles['gbest_position'], axis = 0) - self.particles['current_position']
        fea8 = np.sum(pbest_cur_vec * gbest_cur_vec, axis = -1) / ((np.sqrt(np.sum(pbest_cur_vec ** 2, axis = -1)) * np.sqrt(np.sum(gbest_cur_vec ** 2, axis = -1))) + 1e-5)
        fea8 = np.where(np.isnan(fea8), np.zeros_like(fea8), fea8)

        return np.concatenate((fea0[:, None], fea1[:, None], fea2[:, None], fea3[:, None], fea4[:, None], fea5[:, None], fea6[:, None], fea7[:, None], fea8[:, None]), axis = -1)

    def gp_cat(self):
        """
        # Introduction
        Concatenates the personal best features and the repeated global best feature for all particles.
        # Returns:
        - np.ndarray: A concatenated array of shape (ps, 18), where `ps` is the number of particles. The array consists of each particle's personal best features and the global best feature repeated for each particle.
        # Notes:
        - Assumes `self.pbest_feature` is an array of shape (ps, n_features).
        - Assumes `self.gbest_feature` is an array of shape (n_features,).
        - The concatenation is performed along the last axis.
        """
        
        return np.concatenate((self.pbest_feature, self.gbest_feature[None, :].repeat(self.ps, axis = 0)), axis = -1)  # ps, 18

    # direct reward function
    def cal_reward_direct(self, new_gbest, pre_gbest):
        """
        # Introduction
        Calculates the direct reward based on the improvement of the global best cost in an optimization process.
        # Args:
        - new_gbest (float or np.ndarray): The new global best cost(s) after an optimization step.
        - pre_gbest (float or np.ndarray): The previous global best cost(s) before the optimization step.
        # Returns:
        - float or np.ndarray: The normalized bonus reward(s) computed as the improvement in global best cost divided by `self.max_cost`.
        # Raises:
        - AssertionError: If any computed reward is less than 0, indicating that the new global best is not better than the previous one.
        """
        
        bonus_reward = (pre_gbest - new_gbest) / self.max_cost
        assert np.min(bonus_reward) >= 0, 'reward should be bigger than 0!'
        return bonus_reward

    # 1 -1 reward function
    def cal_reward_11(self, new_gbest, pre_gbest):
        """
        # Introduction
        Calculates a reward based on the comparison between the new global best value and the previous global best value.
        # Args:
        - new_gbest (float): The new global best value obtained.
        - pre_gbest (float): The previous global best value.
        # Returns:
        - int: Returns 1 if the new global best is better (i.e., less than) the previous global best, otherwise returns -1.
        """
        
        if new_gbest < pre_gbest:
            reward = 1
        else:
            reward = -1
        return reward

    # relative reward function
    def cal_reward_relative(self, new_gbest, pre_gbest):
        """
        # Introduction
        Calculates the relative reward based on the change in global best values.
        # Args:
        - new_gbest (float): The new global best value after an optimization step.
        - pre_gbest (float): The previous global best value before the optimization step.
        # Returns:
        - float: The relative improvement in the global best value, computed as (pre_gbest - new_gbest) / pre_gbest.
        # Raises:
        - ZeroDivisionError: If `pre_gbest` is zero, as division by zero is not allowed.
        """
        
        return (pre_gbest - new_gbest) / pre_gbest

    # triangle reward function
    def cal_reward_triangle(self, new_gbest, pre_gbest):
        """
        # Introduction
        Calculates the reward based on the improvement of the global best cost (gbest) using a triangular reward function.
        # Args:
        - new_gbest (float): The new global best cost after an optimization step.
        - pre_gbest (float): The previous global best cost before the optimization step.
        # Returns:
        - float: The calculated reward, which is non-negative and reflects the improvement in gbest.
        # Raises:
        - AssertionError: If the computed reward is negative, indicating an unexpected calculation error.
        """
        
        reward = 0
        if new_gbest < pre_gbest:
            p_t = (self.max_cost - pre_gbest) / self.max_cost
            p_t_new = (self.max_cost - new_gbest) / self.max_cost
            reward = 0.5 * (p_t_new ** 2 - p_t ** 2)
        else:
            reward = 0
        # assert reward >= 0, 'reward should be bigger than 0!'
        return reward

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the particle swarm optimizer (PSO) for one iteration based on the given action and problem definition. This includes updating particle velocities and positions, handling boundary conditions, evaluating costs, updating personal and global bests, managing stagnation counters, calculating rewards, and preparing the next state for further optimization or reinforcement learning.
        # Args:
        - action (np.ndarray): The action(s) to be applied to the particles, typically representing control parameters or decisions for the optimizer.
        - problem (object): The optimization problem instance, which must provide lower and upper bounds (`lb`, `ub`), and optionally an `optimum` attribute.
        # Returns:
        - next_state (np.ndarray): The updated state representation of the particle population after the current iteration.
        - reward (float): The reward signal calculated based on the improvement in global best value.
        - is_end (bool): Flag indicating whether the optimization process has reached its termination condition.
        - info (dict): Additional information (currently empty, but can be extended for logging or debugging).
        # Raises:
        - None explicitly, but may raise exceptions if input shapes are inconsistent or if required attributes are missing from `problem`.
        """
        
        is_end = False

        # record the gbest_val in the begining
        pre_gbest = self.particles['gbest_val']

        # linearly decreasing the coefficient of inertia w
        if self.w_decay:
            self.w -= 0.5 / (self.max_fes / self.ps)

        # generate two set of random val for pso velocity update
        rand1 = self.rng.rand(self.ps, 1)
        rand2 = self.rng.rand(self.ps, 1)

        action = action[:, None]

        # update velocity
        new_velocity = self.w * self.particles['velocity'] + self.c * action * rand1 * (self.particles['pbest_position'] - self.particles['current_position']) + \
                       self.c * (1 - action) * rand2 * (self.particles['gbest_position'][None, :] - self.particles['current_position'])

        # clip the velocity if exceeding the boarder
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        # update position according the boarding method
        if self.boarder_method == "clipping":
            raw_position = self.particles['current_position'] + new_velocity
            new_position = np.clip(raw_position, problem.lb, problem.ub)
        elif self.boarder_method == "random":
            raw_position = self.particles['current_position'] + new_velocity
            filter = raw_position.abs() > problem.ub
            new_position = np.where(filter, self.rng.uniform(low = problem.lb, high = problem.ub, size = (self.ps, self.dim)), raw_position)
        elif self.boarder_method == "periodic":
            raw_position = self.particles['current_position'] + new_velocity
            new_position = problem.lb + ((raw_position - problem.ub) % (2. * problem.ub))
        elif self.boarder_method == "reflect":
            raw_position = self.particles['current_position'] + new_velocity
            filter_low = raw_position < problem.lb
            filter_high = raw_position > problem.ub
            new_position = np.where(filter_low, problem.lb + (problem.lb - raw_position), raw_position)
            new_position = np.where(filter_high, problem.ub - (new_position - problem.ub), new_position)

        # calculate the new costs
        new_cost = self.get_costs(new_position, problem)

        # update particles
        filters = new_cost < self.particles['pbest']

        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)
        filters_best_val = new_cbest_val < self.particles['gbest_val']

        new_particles = {'current_position': new_position,
                         'c_cost': new_cost,
                         'pbest_position': np.where(np.expand_dims(filters, axis = -1),
                                                    new_position,
                                                    self.particles['pbest_position']),
                         'pbest': np.where(filters,
                                           new_cost,
                                           self.particles['pbest']),
                         'velocity': new_velocity,
                         'gbest_val': new_cbest_val if filters_best_val else self.particles['gbest_val'],
                         'gbest_position': np.where(np.expand_dims(filters_best_val, axis = -1),
                                                    new_position[new_cbest_index],
                                                    self.particles['gbest_position']),
                         'gbest_index': np.where(filters_best_val, new_cbest_index, self.particles['gbest_index'])
                         }

        # update the stagnation steps for the whole population
        if new_particles['gbest_val'] < self.particles['gbest_val']:
            self.no_improve = 0
        else:
            self.no_improve += 1

        # update the stagnation steps for singal particle in the population
        filter_per_patience = new_particles['c_cost'] < self.particles['c_cost']
        self.per_no_improve += 1
        tmp = np.where(filter_per_patience, self.per_no_improve, np.zeros_like(self.per_no_improve))
        self.per_no_improve -= tmp

        # update the population
        self.particles = new_particles

        if self.__config.full_meta_data:
            self.meta_X.append(self.particles['current_position'].copy())
            self.meta_Cost.append(self.particles['c_cost'].copy())

        # see if the end condition is satisfied
        if problem.optimum is None:
            is_end = self.fes >= self.max_fes
        else:
            is_end = self.fes >= self.max_fes

        # cal the reward
        if self.reward_func == '11':
            reward = self.cal_reward_11(self.particles['gbest_val'], pre_gbest)
        elif self.reward_func == 'direct':
            reward = self.cal_reward_direct(self.particles['gbest_val'], pre_gbest)
        elif self.reward_func == 'relative':
            reward = self.cal_reward_relative(self.particles['gbest_val'], pre_gbest)
        elif self.reward_func == 'triangle':
            reward = self.cal_reward_triangle(self.particles['gbest_val'], pre_gbest)
        reward *= self.reward_scale

        # get the population next_state
        next_state = self.observe()  # ps, 9

        # update exploration state
        self.pbest_feature = np.where(self.per_no_improve[:, None] == 0, next_state, self.pbest_feature)
        # update exploitation state
        if self.no_improve == 0:
            self.gbest_feature = next_state[self.particles['gbest_index']]
        next_gpcat = self.gp_cat()
        next_state = np.concatenate((next_state, next_gpcat), axis = -1)

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.particles['gbest_val'])

        if is_end:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.particles['gbest_val']
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.particles['gbest_val'])

        info = {}
        return next_state, reward, is_end, info
