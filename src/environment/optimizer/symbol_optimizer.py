import numpy as np
import re
import copy
from scipy.spatial.distance import cdist
import torch
from .learnable_optimizer import Learnable_Optimizer
from typing import List
import math
from sympy import lambdify
import scipy.stats as stats

# preprocess the randx operand
class Myreplace(object):
    def __init__(self):
        self.count = 0

    def replace(self, match):
        """
        # Introduction
        Replaces matched patterns in a string with a modified pattern based on the number of replacements performed.
        # Args:
        - match (re.Match): The match object corresponding to the current regex match.
        # Built-in Attribute:
        - self.count (int): Tracks the number of replacements performed.
        - self.pattern (str): The pattern string to use for replacement.
        # Returns:
        - str: The replacement string. If this is the first match, returns the original matched string; otherwise, returns the pattern with a numeric suffix.
        # Raises:
        - AttributeError: If `self.count` or `self.pattern` are not defined in the instance.
        """
        
        self.count += 1
        if self.count > 1:
            return self.pattern + str(self.count - 1)
        else:
            return match.group()

    def process_string(self, string, pattern):
        """
        # Introduction
        Processes the input string by replacing all occurrences that match the given pattern using a custom replacement method.
        # Args:
        - string (str): The input string to be processed.
        - pattern (str or Pattern): The regular expression pattern to search for in the string.
        # Built-in Attribute:
        - self.pattern: Stores the current pattern being used for replacement.
        - self.count: Tracks the number of replacements made.Default is 0.
        # Returns:
        - str: The processed string with all pattern matches replaced.
        # Raises:
        - re.error: If the provided pattern is not a valid regular expression.
        """
        self.pattern = pattern
        self.count = 0
        new_string = re.sub(pattern, self.replace, string)
        return new_string


def symbol_config(config):
    """
    # Introduction
    Configures the parameters for the symbolic optimizer by setting various attributes on the provided configuration object.
    # Args:
    - config (object): A configuration object whose attributes will be set to define the optimizer's behavior.
        - The attributes needed for the Symbol include:
            - init_pop (str): Method for initializing the population (default: 'random').
            - teacher (str): The teacher algorithm to use (default: 'MadDE').
            - population_size (int): Number of individuals in the population (default: 100).
            - boarder_method (str): Method for handling boundary conditions (default: 'clipping').
            - skip_step (int): Number of steps to skip during training (default: 5).
            - test_skip_step (int): Number of steps to skip during testing (default: 5).
            - max_c (float): Maximum value for coefficient c (default: 1.0).
            - min_c (float): Minimum value for coefficient c (default: -1.0).
            - c_interval (float): Interval for coefficient c (default: 0.4).
            - max_layer (int): Maximum number of layers allowed (default: 6).
            - value_dim (int): Dimensionality of the value (default: 1).
            - hidden_dim (int): Dimensionality of the hidden layer (default: 16).
            - num_layers (int): Number of layers in the model (default: 1).
            - lr (float): Learning rate for the optimizer (default: 1e-3).
            - lr_critic (float): Learning rate for the critic (default: 1e-3).
            
    # Built-in Attributes:
    - init_pop (str): Method for initializing the population (default: 'random').
    - teacher (str): The teacher algorithm to use (default: 'MadDE').
    - population_size (int): Number of individuals in the population (default: 100).
    - boarder_method (str): Method for handling boundary conditions (default: 'clipping').
    - skip_step (int): Number of steps to skip during training (default: 5).
    - test_skip_step (int): Number of steps to skip during testing (default: 5).
    - max_c (float): Maximum value for coefficient c (default: 1.0).
    - min_c (float): Minimum value for coefficient c (default: -1.0).
    - c_interval (float): Interval for coefficient c (default: 0.4).
    - max_layer (int): Maximum number of layers allowed (default: 6).
    - value_dim (int): Dimensionality of the value (default: 1).
    - hidden_dim (int): Dimensionality of the hidden layer (default: 16).
    - num_layers (int): Number of layers in the model (default: 1).
    - lr (float): Learning rate for the optimizer (default: 1e-3).
    - lr_critic (float): Learning rate for the critic (default: 1e-3).
    # Returns:
    - None
    """
    
    config.init_pop = 'random'
    config.teacher = 'MadDE'
    config.population_size = 100
    config.boarder_method = 'clipping'
    config.skip_step = 5
    config.test_skip_step = 5
    config.max_c = 1.
    config.min_c = -1.
    config.c_interval = 0.4
    config.max_layer = 6
    config.value_dim = 1
    config.hidden_dim = 16
    config.num_layers = 1
    config.lr = 1e-3
    config.lr_critic = 1e-3


class SYMBOL_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    SurrRLDE is a novel MetaBBO framework which combines surrogate learning process and reinforcement learning-aided Differential Evolution (DE) algorithm.
    # Original paper
    "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
    # Official Implementation
    [SurrRLDE](https://github.com/GMC-DRL/Surr-RLDE)
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the SymbolOptimizer with the provided configuration, setting up tokenizer, replacement strategy, and various optimization parameters.
        # Args:
        - config (object): Configuration object.
            - The attributes needed for the Symbol include:
                - init_pop (str): Method for initializing the population (default: 'random').
                - teacher (str): The teacher algorithm to use (default: 'MadDE').
                - population_size (int): Number of individuals in the population (default: 100).
                - boarder_method (str): Method for handling boundary conditions (default: 'clipping').
                - skip_step (int): Number of steps to skip during training (default: 5).
                - test_skip_step (int): Number of steps to skip during testing (default: 5).
                
        # Built-in Attribute:
        - self.tokenizer (MyTokenizer): Tokenizer instance for processing symbols.
        - self.__config (object): Stores the configuration object.
        - self.NP (int): Number of particles or population size, set to 100.
        - self.no_improve (int): Counter for iterations without improvement.
        - self.per_no_improve (np.ndarray): Array tracking no-improvement status per particle.
        - self.evaling (bool): Flag indicating if evaluation is in progress.
        - self.max_fes (int): Maximum number of function evaluations.
        - self.boarder_method (str): Method for handling boundaries, default is 'periodic'.
        - self.replace (Myreplace): Replacement strategy instance.
        - self.log_interval (int): Interval for logging progress.
        - self.teacher_optimizer (object or None): Optional teacher optimizer for advanced strategies.
        - self.is_train (bool): Indicates if the optimizer is in training mode.
        # Raises:
        - None
        """
        
        super().__init__(config)

        # !add
        symbol_config(config)

        self.tokenizer = MyTokenizer()
        self.__config = config

        self.NP = 100

        self.no_improve = 0
        self.per_no_improve = np.zeros((self.NP,))

        self.evaling = False
        self.max_fes = config.maxFEs

        self.boarder_method = 'periodic'

        self.replace = Myreplace()

        self.log_interval = config.log_interval

        self.teacher_optimizer = None

        # ! change when using symbol_optimizer
        self.is_train = True
        self.log_interval = config.log_interval

    def __str__(self):
        """
        Returns a string representation of the SYMBOL_Optimizer object.
        # Returns:
            str: The string "SYMBOL_Optimizer" representing the object.
        """
        
        return "SYMBOL_Optimizer"

    # the interface for environment reseting
    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimization process, optionally using a teacher optimizer for population initialization if in training mode.
        # Args:
        - problem (Problem): An instance of the optimization problem, containing bounds and other problem-specific information.
        # Returns:
        - Any: The observed state after population initialization, as returned by `self.observe()`.
        # Raises:
        - None directly, but may raise exceptions from called methods such as `eval`, `copy.deepcopy`, or population initialization routines.
        # Notes:
        - If `self.is_train` is True, the population is initialized using a teacher optimizer and a custom initialization method.
        - If `self.is_train` is False, the population is initialized normally and reset.
        - Logs and meta-data are initialized if configured.
        """
        
        # self.NP=self.__Nmax
        self.max_x = problem.ub
        self.min_x = problem.lb
        self.problem = problem
        self.dim = problem.dim
        self.__config.dim = problem.dim
        if self.teacher_optimizer is None:
             self.teacher_optimizer = eval(self.__config.teacher)(self.__config, self.rng)
        if self.is_train:
            tea_pop = self.teacher_optimizer.init_population(copy.deepcopy(problem))
            self.population = Population(self.dim, self.NP, self.min_x, self.max_x, self.max_fes, copy.deepcopy(problem), self.rng)
            get_init_pop(tea_pop = tea_pop, stu_pop = self.population, method = self.__config.init_pop, rng = self.rng)
        else:
            self.population = Population(self.dim, self.NP, self.min_x, self.max_x, self.max_fes, problem, self.rng)
            self.population.reset()

        self.log_index = 1
        self.cost = [self.population.gbest_cost]

        if self.__config.full_meta_data:
            self.meta_X = [self.population.current_position.copy()]
            self.meta_Cost = [self.population.c_cost.copy()]

        # return state
        return self.observe()

    def eval(self):
        self.evaling = True

    def train(self):
        # set_seed()
        self.evaling = False

    # feature encoding
    def observe(self):
        return self.population.feature_encoding()

    # input the self.population and expr function, return the population after applying expr function
    def update(self, action, problem):
        """
        # Introduction
        Updates the optimizer's population based on the provided action and problem, applying the specified update expression, handling boundary conditions, and calculating rewards. This method is central to the optimization process, performing one or more update steps and logging progress.
        # Args:
        - action (dict): A dictionary containing the update expression (`'expr'`) and the number of steps to skip (`'skip_step'`).
        - problem: The optimization problem instance, which may provide information such as the optimum value.
        # Returns:
        - observation: The current observation of the optimizer's state.
        - reward (float): The reward calculated for this update step.
        - is_end (bool): Whether the optimization process has reached its end condition.
        - info (dict): Additional information (currently an empty dictionary).
        # Raises:
        - AssertionError: If the number of 'randx' replacements does not match the expected count.
        - AssertionError: If the shapes of the update variables do not match.
        - AssertionError: If an unsupported boundary method is specified.
        """

        expr = action['expr']
        skip_step = action['skip_step']
        # debug
        # print(f"x + {expr}")
        # record the previous gbest
        self.population.pre_gbest = self.population.gbest_cost

        cnt_randx = expr.count('randx')
        pattern = 'randx'
        expr = self.replace.process_string(expr, pattern)
        count = self.replace.count

        assert count == cnt_randx, 'randx count is wrong!!'
        variables = copy.deepcopy(self.tokenizer.variables)
        for i in range(1, count):
            variables.append(f'randx{i}')
        update_function = expr_to_func(expr, variables)

        for sub_step in range(skip_step):
            x = self.population.current_position

            gb = self.population.gbest_position[None, :].repeat(self.NP, 0)
            gw = self.population.gworst_position[None, :].repeat(self.NP, 0)

            dx = self.population.delta_x
            randx = x[self.rng.randint(self.NP, size = self.NP)]

            pbest = self.population.pbest_position

            names = locals()
            inputs = [x, gb, gw, dx, randx, pbest]
            for i in range(1, count):
                names[f'randx{i}'] = x[self.rng.randint(self.NP, size = self.NP)]
                inputs.append(eval(f'randx{i}'))

            assert x.shape == gb.shape == gw.shape == dx.shape == randx.shape, 'not same shape'

            next_position = x + update_function(*inputs)

            # boarder clip or what
            if self.boarder_method == "clipping":
                next_position = np.clip(next_position, self.min_x, self.max_x)
            elif self.boarder_method == "periodic":
                next_position = self.min_x + (next_position - self.max_x) % (self.max_x - self.min_x)
            else:
                raise AssertionError('this board method is not surported!')

            # update population
            self.population.update(next_position, filter_survive = False)

        if self.population.cur_fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.population.gbest_cost)
        reward = 0
        if self.is_train:
            tea_pop, _, _, _ = self.teacher_optimizer.update({'skip_step': skip_step})
            # cal reward
            reward = self.cal_reward(tea_pop, max_step = self.max_fes / self.NP / skip_step)
        else:
            reward = (self.population.pre_gbest - self.population.gbest_cost) / (self.population.init_cost - 0)

        if self.__config.full_meta_data:
            self.meta_X.append(self.population.current_position.copy())
            self.meta_Cost.append(self.population.c_cost.copy())

        is_end = False
        # see if the end condition is satisfied
        if problem.optimum is None:
            is_end = self.population.cur_fes >= self.max_fes
        else:
            is_end = self.population.cur_fes >= self.max_fes

        if is_end:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.population.gbest_cost
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.population.gbest_cost)
            # print(f'problem: {self.problem.__str__()}')

        return self.observe(), reward, is_end, {}

    def cal_reward(self, tea_pop, max_step):
        """
        # Introduction
        Calculates the reward value based on the imitation distance and the improvement in global best cost for the current population.
        # Args:
        - tea_pop (Population): The target or teacher population used for imitation comparison.
        - max_step (int): The maximum number of steps or iterations allowed in the optimization process.
        # Returns:
        - float: The computed reward, which is the sum of the imitation reward and the base reward.
        # Raises:
        - ZeroDivisionError: If `self.population.init_cost` is zero, as division by zero is not allowed.
        """
        dist = cal_gap_nearest(self.population, tea_pop)

        imitation_r = -dist / max_step

        # base_reward = -(self.population.gbest_cost- self.problem.optimum) / (self.population.init_cost-self.problem.optimum) / max_step
        base_reward = (self.population.pre_gbest - self.population.gbest_cost) / (self.population.init_cost - 0)

        return imitation_r + base_reward
        # return base_reward


'''forming init pop'''


def get_init_pop(tea_pop, stu_pop, method, rng):
    """
    # Introduction
    Initializes the population of a student population (`stu_pop`) based on a teacher population (`tea_pop`) using a specified initialization method.
    # Args:
    - tea_pop: An object representing the teacher population, expected to have attributes `c_cost`, `current_position`, and `pop_size`.
    - stu_pop: An object representing the student population, expected to have attributes `pop_size` and a `reset` method.
    - method (str): The initialization method to use. Supported values are `'best'`, `'harf'`, `'random'`, and `'uniform'`.
    - rng: A random number generator object with a `randint` method.
    # Returns:
    - None: The function modifies `stu_pop` in place by resetting its population.
    # Raises:
    - ValueError: If the specified `method` is not supported.
    """
    if method == 'best':
        sort_index = np.argsort(tea_pop.c_cost)
        init_pos = tea_pop.current_position[sort_index[:stu_pop.pop_size]]
        stu_pop.reset(init_pop = init_pos)
    elif method == 'harf':
        sort_index = np.argsort(tea_pop.c_cost)
        init_pos = np.concatenate((tea_pop.current_position[sort_index[:int(stu_pop.pop_size * 0.5)]], tea_pop.current_position[sort_index[:stu_pop.pop_size - int(stu_pop.pop_size * 0.5)]]), axis = 0)
        stu_pop.reset(init_pop = init_pos)
    elif method == 'random':
        rand_index = rng.randint(0, tea_pop.pop_size, size = (stu_pop.pop_size,))
        init_pos = tea_pop.current_position[rand_index]
        stu_pop.reset(init_pop = init_pos)
    elif method == 'uniform':
        sort_index = np.argsort(tea_pop.c_cost)
        init_pos = tea_pop.current_position[sort_index[::tea_pop.pop_size // stu_pop.pop_size]]
        stu_pop.reset(init_pop = init_pos)
    else:
        raise ValueError('init pop method is currently not supported!!')


def cal_gap_nearest(stu_pop, tea_pop):
    """
    # Introduction
    Calculates the normalized maximum minimum Euclidean distance ("gap") between each student position and its nearest teacher position in a normalized search space.
    # Args:
    - stu_pop: An object representing the student population, expected to have attributes `current_position` (numpy.ndarray of shape [n_students, dim]) and `max_x` (float or array-like for normalization).
    - tea_pop: An object representing the teacher population, expected to have attribute `current_position` (numpy.ndarray of shape [n_teachers, dim]).
    # Returns:
    - float: The normalized gap, defined as the maximum of the minimum distances from each student to the nearest teacher, divided by the maximum possible distance in the normalized space.
    # Raises:
    - AttributeError: If `stu_pop` or `tea_pop` do not have the required attributes.
    - ValueError: If the shapes of `current_position` arrays are incompatible.
    """
    max_x = stu_pop.max_x

    stu_position = stu_pop.current_position
    tea_position = tea_pop.current_position

    norm_p1 = stu_position / max_x
    norm_p1 = norm_p1[None, :, :]
    norm_p2 = tea_position / max_x
    norm_p2 = norm_p2[:, None, :]
    dist = np.sqrt(np.sum((norm_p2 - norm_p1) ** 2, -1))
    min_dist = np.min(dist, -1)

    gap = np.max(min_dist)
    dim = stu_position.shape[1]
    max_dist = 2 * np.sqrt(dim)
    return gap / max_dist


def dist(x, y):
    """
    Calculates the Euclidean distance between two arrays.
    # Args:
    - x (np.ndarray): The first input array.
    - y (np.ndarray): The second input array.
    # Returns:
    - np.ndarray or float: The Euclidean distance(s) between `x` and `y`.
    # Raises:
    - ValueError: If `x` and `y` have incompatible shapes for broadcasting.
    """
    return np.sqrt(np.sum((x - y) ** 2, axis = -1))


class MadDE():
    def __init__(self, config, rng):

        self.__dim = config.dim
        self.__p = 0.18
        self.__PqBX = 0.01
        self.__F0 = 0.2
        self.__Cr0 = 0.2
        self.__pm = np.ones(3) / 3
        self.__npm = 2
        self.__hm = 10
        self.__Nmin = 4
        self.__Nmax = self.__npm * self.__dim * self.__dim
        # self.__Nmax = 200
        self.__H = self.__hm * self.__dim
        # self.__H=100
        self.__FEs = 0
        self.__MaxFEs = config.maxFEs

        self.config = config
        self.rng = rng

    def __ctb_w_arc(self, group, best, archive, Fs):
        NP, dim = group.shape
        NB = best.shape[0]
        NA = archive.shape[0]

        count = 0
        rb = np.random.randint(NB, size = NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size = duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size = NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size = duplicate.shape[0])
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

    def __ctr_w_arc(self, group, archive, Fs):
        NP, dim = group.shape
        NA = archive.shape[0]

        count = 0
        r1 = np.random.randint(NP, size = NP)
        duplicate = np.where((r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size = NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size = duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        v = group + Fs * (x1 - x2)

        return v

    def __weighted_rtb(self, group, best, Fs, Fas):
        NP, dim = group.shape
        NB = best.shape[0]

        count = 0
        rb = np.random.randint(NB, size = NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size = duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size = NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]
        v = Fs * x1 + Fs * Fas * (xb - x2)

        return v

    def __binomial(self, x, v, Crs):
        NP, dim = x.shape
        jrand = np.random.randint(dim, size = NP)
        u = np.where(np.random.rand(NP, dim) < Crs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

    def __sort(self):
        # new index after sorting
        ind = np.argsort(self.population.c_cost)

        self.population.reset_order(ind)

    def __update_archive(self, old_id):
        if self.__archive.shape[0] < self.__NA:
            self.__archive = np.append(self.__archive, self.population.current_position[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[np.random.randint(self.__archive.shape[0])] = self.population.current_position[old_id]

    def __mean_wL(self, df, s):
        w = df / np.sum(df)
        if np.sum(w * s) > 0.000001:
            return np.sum(w * (s ** 2)) / np.sum(w * s)
        else:
            return 0.5

    # randomly choose step length nad crossover rate from MF and MCr
    def __choose_F_Cr(self):
        # generate Cr can be done simutaneously
        gs = self.__NP
        ind_r = np.random.randint(0, self.__MF.shape[0], size = gs)  # index
        C_r = np.minimum(1, np.maximum(0, np.random.normal(loc = self.__MCr[ind_r], scale = 0.1, size = gs)))
        # as for F, need to generate 1 by 1
        cauchy_locs = self.__MF[ind_r]
        F = stats.cauchy.rvs(loc = cauchy_locs, scale = 0.1, size = gs)
        err = np.where(F < 0)[0]
        F[err] = 2 * cauchy_locs[err] - F[err]
        return C_r, np.minimum(1, F)

    # update MF and MCr, join new value into the set if there are some successful changes or set it to initial value
    def __update_M_F_Cr(self, SF, SCr, df):
        if SF.shape[0] > 0:
            mean_wL = self.__mean_wL(df, SF)
            self.__MF[self.__k] = mean_wL
            mean_wL = self.__mean_wL(df, SCr)
            self.__MCr[self.__k] = mean_wL
            self.__k = (self.__k + 1) % self.__MF.shape[0]
        else:
            self.__MF[self.__k] = 0.5
            self.__MCr[self.__k] = 0.5

    def init_population(self, problem):

        self.problem = problem
        self.min_x = problem.lb
        self.max_x = problem.ub

        self.__NP = self.__Nmax
        self.__NA = int(2.3 * self.__NP)

        self.population = MadDE_Population(self.__dim, self.__NP, self.min_x, self.max_x, self.__MaxFEs, problem, self.rng)
        self.population.reset()
        self.__FEs = self.__NP
        self.__archive = np.array([])
        self.__MF = np.ones(self.__H) * self.__F0
        self.__MCr = np.ones(self.__H) * self.__Cr0
        self.__k = 0
        # self.gbest = np.min(self.population.c_cost)
        return self.population

    def update(self, action):

        if action.get('skip_step') is not None:
            skip_step = action['skip_step']
        elif action.get('fes') is not None:
            step_fes = action['fes']
            next_fes = self.population.cur_fes + step_fes
        else:
            assert True, 'action error!!'

        step_end = self.population.cur_fes >= self.__MaxFEs
        if action.get('fes') is not None:
            step_end = (step_end or (self.population.cur_fes >= next_fes))
        step = 0

        while not step_end:
            self.__sort()
            NP, dim = self.__NP, self.__dim
            q = 2 * self.__p - self.__p * self.__FEs / self.__MaxFEs
            Fa = 0.5 + 0.5 * self.__FEs / self.__MaxFEs
            Cr, F = self.__choose_F_Cr()
            mu = np.random.choice(3, size = NP, p = self.__pm)
            p1 = self.population.current_position[mu == 0]
            p2 = self.population.current_position[mu == 1]
            p3 = self.population.current_position[mu == 2]
            pbest = self.population.current_position[:max(int(self.__p * NP), 2)]
            qbest = self.population.current_position[:max(int(q * NP), 2)]
            Fs = F.repeat(dim).reshape(NP, dim)
            v1 = self.__ctb_w_arc(p1, pbest, self.__archive, Fs[mu == 0])
            v2 = self.__ctr_w_arc(p2, self.__archive, Fs[mu == 1])
            v3 = self.__weighted_rtb(p3, qbest, Fs[mu == 2], Fa)
            v = np.zeros((NP, dim))
            v[mu == 0] = v1
            v[mu == 1] = v2
            v[mu == 2] = v3

            v = np.where(v < self.min_x, (v + self.min_x) / 2, v)
            v = np.where(v > self.min_x, (v + self.max_x) / 2, v)
            rvs = np.random.rand(NP)
            Crs = Cr.repeat(dim).reshape(NP, dim)
            u = np.zeros((NP, dim))
            if np.sum(rvs <= self.__PqBX) > 0:
                qu = v[rvs <= self.__PqBX]
                if self.__archive.shape[0] > 0:
                    qbest = np.concatenate((self.population.current_position, self.__archive), 0)[
                            :max(int(q * (NP + self.__archive.shape[0])), 2)]
                cross_qbest = qbest[np.random.randint(qbest.shape[0], size = qu.shape[0])]
                qu = self.__binomial(cross_qbest, qu, Crs[rvs <= self.__PqBX])
                u[rvs <= self.__PqBX] = qu
            bu = v[rvs > self.__PqBX]
            bu = self.__binomial(self.population.current_position[rvs > self.__PqBX], bu, Crs[rvs > self.__PqBX])
            u[rvs > self.__PqBX] = bu

            ncost = self.population.get_costs(u)
            self.__FEs += NP

            optim = np.where(ncost < self.population.c_cost)[0]
            for i in optim:
                self.__update_archive(i)
            SF = F[optim]
            SCr = Cr[optim]
            df = np.maximum(0, self.population.c_cost - ncost)
            self.__update_M_F_Cr(SF, SCr, df[optim])
            count_S = np.zeros(3)
            for i in range(3):
                count_S[i] = np.mean(df[mu == i] / self.population.c_cost[mu == i])
            if np.sum(count_S) > 0:
                self.__pm = np.maximum(0.1, np.minimum(0.9, count_S / np.sum(count_S)))
                self.__pm /= np.sum(self.__pm)
            else:
                self.__pm = np.ones(3) / 3

            self.population.update(u, ncost, filter_survive = True)

            self.__NP = int(np.round(self.__Nmax + (self.__Nmin - self.__Nmax) * self.__FEs / self.__MaxFEs))
            self.__NP = max(self.__NP, self.__Nmin)
            self.__NA = int(2.3 * self.__NP)
            self.__sort()

            self.population.reset_popsize(self.__NP)

            self.__archive = self.__archive[:self.__NA]

            step += 1
            if action.get('fes') is not None:
                if self.population.cur_fes >= next_fes or self.population.cur_fes >= self.__MaxFEs:
                    step_end = True
                    break
            elif action.get('skip_step') is not None:
                if step >= skip_step:
                    step_end = True
                    break

        return self.population, 0, self.__FEs >= self.__MaxFEs, {}


class Population(object):
    def __init__(self, dim, pop_size, min_x, max_x, max_fes, problem, rng):
        self.dim = dim
        self.pop_size = pop_size
        self.min_x = min_x
        self.max_x = max_x
        self.max_fes = max_fes
        self.problem = problem
        self.max_dist = np.sqrt(np.sum((problem.ub - problem.lb) ** 2))
        self.cur_fes = 0
        self.rng = rng

    # calculate costs of solutions
    def get_costs(self, position):
        ps = position.shape[0]
        self.cur_fes += ps
        if self.problem.optimum is None:
            cost = self.problem.eval(position)
        else:
            cost = self.problem.eval(position) - self.problem.optimum

        return cost

    def reset(self, init_pop = None, init_y = None, need_his = True):
        # init fes and stag_count
        if init_y is not None:
            self.cur_fes += init_y.shape[0]
        else:
            self.cur_fes = 0
        self.stag_count = 0

        # init population
        if init_pop is None:
            # randomly generate the position and velocity
            rand_pos = self.rng.uniform(low = -self.max_x, high = self.max_x, size = (self.pop_size, self.dim))
        else:
            rand_pos = init_pop

        self.current_position = rand_pos.copy()
        self.dx = np.zeros_like(rand_pos)
        self.delta_x = np.zeros_like(rand_pos)

        # get the initial cost
        if init_y is None:
            self.c_cost = self.get_costs(self.current_position)  # ps
        else:
            self.c_cost = init_y

        # init pbest related
        self.pbest_position = rand_pos.copy()
        self.pbest_cost = self.c_cost.copy()

        # find out the gbest_val
        self.gbest_cost = np.min(self.c_cost)
        gbest_index = np.argmin(self.c_cost)
        self.gbest_position = rand_pos[gbest_index]

        # init cbest related
        self.cbest_cost = self.gbest_cost
        self.cbest_position = self.gbest_position
        self.cbest_index = gbest_index

        # init gworst related
        self.gworst_cost = np.max(self.c_cost)
        gworst_index = np.argmax(self.c_cost)
        self.gworst_position = rand_pos[gworst_index]

        # record
        self.init_cost = np.min(self.c_cost)
        self.pre_position = self.current_position
        self.pre_cost = self.c_cost
        self.pre_gbest = self.gbest_cost

    def update(self, next_position, filter_survive = False):
        self.pre_cost = self.c_cost
        self.pre_position = copy.deepcopy(self.current_position)
        # self.pre_gbest=self.gbest_cost

        self.before_select_pos = next_position

        new_cost = self.get_costs(next_position)

        if filter_survive:
            surv_filter = new_cost <= self.c_cost
            next_position = np.where(surv_filter[:, None], next_position, self.current_position)
            new_cost = np.where(surv_filter, new_cost, self.c_cost)

        filters = new_cost < self.pbest_cost

        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)

        self.current_position = next_position
        self.c_cost = new_cost
        self.pbest_position = np.where(np.expand_dims(filters, axis = -1),
                                       next_position,
                                       self.pbest_position)
        self.pbest_cost = np.where(filters,
                                   new_cost,
                                   self.pbest_cost)
        if new_cbest_val < self.gbest_cost:
            self.gbest_cost = new_cbest_val
            self.gbest_position = self.current_position[new_cbest_index]
            # gbest_index=new_cbest_index
            self.stag_count = 0
        else:
            self.stag_count += 1

        self.cbest_cost = new_cbest_val
        self.cbest_position = next_position[new_cbest_index]
        self.cbest_index = new_cbest_index

        new_cworst_val = np.max(new_cost)
        if new_cworst_val > self.gworst_cost:
            self.gworst_cost = new_cworst_val
            gworst_index = np.argmax(new_cost)
            self.gworst_position = next_position[gworst_index]

        self.dx = (self.c_cost - self.pre_cost)[:, None] / (self.current_position - self.pre_position + 1e-5)
        self.dx = np.where(np.isnan(self.dx), np.zeros_like(self.current_position), self.dx)

        self.delta_x = self.current_position - self.pre_position

    def update_cmaes(self, next_position, next_y):
        self.pre_cost = self.c_cost
        self.pre_position = self.current_position
        # self.pre_gbest=self.c_cost

        new_cost = next_y

        # update particles
        filters = new_cost < self.pbest_cost

        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)

        self.current_position = next_position
        self.c_cost = new_cost
        self.pbest_position = np.where(np.expand_dims(filters, axis = -1),
                                       next_position,
                                       self.pbest_position)
        self.pbest_cost = np.where(filters,
                                   new_cost,
                                   self.pbest_cost)
        if new_cbest_val < self.gbest_cost:
            self.gbest_cost = new_cbest_val
            self.gbest_position = self.current_position[new_cbest_index]
            self.stag_count = 0
        else:
            self.stag_count += 1

        self.cbest_cost = new_cbest_val
        self.cbest_position = next_position[new_cbest_index]
        self.cbest_index = new_cbest_index

        new_cworst_val = np.max(new_cost)
        if new_cworst_val > self.gworst_cost:
            self.gworst_cost = new_cworst_val
            gworst_index = np.argmax(new_cost)
            self.gworst_position = next_position[gworst_index]

    def feature_encoding(self):
        assert self.gbest_cost != self.gworst_cost, f'gbest == gworst!!,{self.gbest_cost}'
        fea_1 = (self.c_cost - self.gbest_cost) / (self.gworst_cost - self.gbest_cost + 1e-8)
        fea_1 = np.mean(fea_1)

        fea_2 = calculate_mean_distance(self.current_position) / self.max_dist

        fit = np.zeros_like(self.c_cost)
        fit[:self.pop_size // 2] = self.gworst_cost
        fit[self.pop_size // 2:] = self.gbest_cost
        maxstd = np.std(fit)
        fea_3 = np.std(self.c_cost) / (maxstd + 1e-8)

        fea_4 = (self.max_fes - self.cur_fes) / self.max_fes

        fea_5 = self.stag_count / (self.max_fes // self.pop_size)

        fea_6 = dist(self.current_position, self.cbest_position[None, :]) / self.max_dist
        fea_6 = np.mean(fea_6)

        fea_7 = (self.c_cost - self.cbest_cost) / (self.gworst_cost - self.gbest_cost + 1e-8)
        fea_7 = np.mean(fea_7)

        fea_8 = dist(self.current_position, self.gbest_position[None, :]) / self.max_dist
        fea_8 = np.mean(fea_8)

        fea_9 = 0
        if self.gbest_cost < self.pre_gbest:
            fea_9 = 1

        feature = np.array([fea_1, fea_2, fea_3, fea_4, fea_5, fea_6, fea_7, fea_8, fea_9])

        assert not np.any(np.isnan(feature)), f'feature has nan!!,{feature}'
        return feature


class MadDE_Population(Population):
    def __init__(self, dim, pop_size, min_x, max_x, max_fes, problem, rng):
        super().__init__(dim, pop_size, min_x, max_x, max_fes, problem, rng)

    def reset(self):
        self.index = np.arange(self.pop_size)
        return super().reset()

    def update(self, next_position, new_cost, filter_survive = False):
        self.pre_cost = self.c_cost
        self.pre_position = self.current_position
        self.pre_gbest = self.gbest_cost

        self.before_select_pos = next_position

        # new_cost=self.get_costs(next_position)
        if filter_survive:
            surv_filter = new_cost <= self.c_cost
            next_position = np.where(surv_filter[:, None], next_position, self.current_position)
            new_cost = np.where(surv_filter, new_cost, self.c_cost)

        # update particles
        filters = new_cost < self.pbest_cost

        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)

        self.current_position = next_position
        self.c_cost = new_cost
        self.pbest_position = np.where(np.expand_dims(filters, axis = -1),
                                       next_position,
                                       self.pbest_position)
        self.pbest_cost = np.where(filters,
                                   new_cost,
                                   self.pbest_cost)
        if new_cbest_val < self.gbest_cost:
            self.gbest_cost = new_cbest_val
            self.gbest_position = self.current_position[new_cbest_index]
            # self.gbest_index=new_cbest_index
            self.stag_count = 0
        else:
            self.stag_count += 1

        self.cbest_cost = new_cbest_val
        self.cbest_position = next_position[new_cbest_index]

        new_cworst_val = np.max(new_cost)
        if new_cworst_val > self.gworst_cost:
            self.gworst_cost = new_cworst_val
            gworst_index = np.argmax(new_cost)
            self.gworst_position = next_position[gworst_index]

        # deprecated
        self.dx = (self.c_cost - self.pre_cost)[:, None] / (self.current_position - self.pre_position + 1e-5)
        self.dx = np.where(np.isnan(self.dx), np.zeros_like(self.current_position), self.dx)

    def update2(self, next_position, new_cost):
        self.pre_cost = self.c_cost
        self.pre_position = self.current_position
        self.pre_gbest = self.gbest_cost

        self.before_select_pos = next_position

        filters = new_cost < self.pbest_cost
        new_cbest_val = np.min(new_cost)
        new_cbest_index = np.argmin(new_cost)

        self.current_position = next_position
        self.c_cost = new_cost
        self.pbest_position = np.where(np.expand_dims(filters, axis = -1),
                                       next_position,
                                       self.pbest_position)
        self.pbest_cost = np.where(filters,
                                   new_cost,
                                   self.pbest_cost)
        if new_cbest_val < self.gbest_cost:
            self.gbest_cost = new_cbest_val
            self.gbest_position = self.current_position[new_cbest_index]
            # self.gbest_index=new_cbest_index
            self.stag_count = 0
        else:
            self.stag_count += 1

        self.cbest_cost = new_cbest_val
        self.cbest_position = next_position[new_cbest_index]

        new_cworst_val = np.max(new_cost)
        if new_cworst_val > self.gworst_cost:
            self.gworst_cost = new_cworst_val
            gworst_index = np.argmax(new_cost)
            self.gworst_position = next_position[gworst_index]

        self.dx = (self.c_cost - self.pre_cost)[:, None] / (self.current_position - self.pre_position + 1e-5)
        self.dx = np.where(np.isnan(self.dx), np.zeros_like(self.current_position), self.dx)

    def reset_order(self, index):
        self.current_position = self.current_position[index]
        self.c_cost = self.c_cost[index]
        self.dx = self.dx[index]
        self.pbest_cost = self.pbest_cost[index]
        self.pbest_position = self.pbest_position[index]
        self.index = self.index[index]

    def reset_popsize(self, NP):
        self.current_position = self.current_position[:NP]
        self.c_cost = self.c_cost[:NP]
        self.pbest_cost = self.pbest_cost[:NP]
        self.pbest_position = self.pbest_position[:NP]
        self.index = self.index[:NP]
        self.pop_size = NP


class Tokenizer:
    SPECIAL_SYMBOLS = {}

    SPECIAL_FLOAT_SYMBOLS = {}

    SPECIAL_OPERATORS = {}

    SPECIAL_INTEGERS = {}

    def __init__(self):
        self.start = "<START>"
        self.start_id = 1
        self.end = "<END>"
        self.end_id = 2
        self.pad = "<PAD>"
        self.pad_id = 0
        self.vocab = [self.pad, self.start, self.end]

    def encode(self, expr):
        raise NotImplementedError()

    def decode(self, expr):
        raise NotImplementedError()

    def is_unary(self, token):
        raise NotImplementedError()

    def is_binary(self, token):
        raise NotImplementedError()

    def is_leaf(self, token):
        raise NotImplementedError()

    def get_constant_ids(self):
        pass


class MyTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.variables = [
            "x",
            "gb",
            "gw",
            "dx",
            "randx",
            "pb"
        ]
        self.binary_ops = ["+", "*"]
        self.unary_ops = [
            # "sin",
            # "cos",
            "-",
            # "sign"
        ]
        self.constants = [f"C{i}" for i in range(-1, 1)]
        self.leafs = self.constants + self.variables
        self.vocab = list(self.binary_ops) + list(self.unary_ops) + self.leafs
        self.lookup_table = dict(zip(self.vocab, range(len(self.vocab))))
        self.leaf_index = np.arange(len(self.vocab))[len(self.vocab) - len(self.leafs):]
        self.operator_index = np.arange(len(self.vocab) - len(self.leafs))
        self.binary_index = np.arange(len(self.binary_ops))
        self.unary_index = np.arange(len(self.unary_ops)) + len(self.binary_ops)
        self.vocab_size = len(self.vocab)
        self.constants_index = self.leaf_index[:len(self.constants)]
        self.non_const_index = list(set(range(self.vocab_size)) - set(self.constants_index))
        self.var_index = self.leaf_index[len(self.constants):]

    def decode(self, expr):
        return self.vocab[expr]

    def encode(self, expr):
        return self.lookup_table[expr]

    def is_consts(self, id):
        if torch.is_tensor(id):
            id = id.cpu()
        return np.isin(id, self.constants_index)
        # return id in self.constants_index

    def is_binary(self, token):
        return token in self.binary_ops

    def is_unary(self, token):
        return token in self.unary_ops

    def is_leaf(self, token):
        return token in self.leafs

    def is_var(self, token):
        return token in self.variables


# expression related function


def get_mask(pre_seq, tokenizer, position, max_layer):
    if len(pre_seq.shape) == 1:
        pre_seq = [pre_seq]
    bs, _ = pre_seq.size()
    old_device = pre_seq.device
    pre_seq = pre_seq.cpu().numpy()
    position = position.cpu().numpy()
    masks = []
    for sub_seq, pos in zip(pre_seq, position):
        # if position==-1: set mask all to be zero
        if pos == -1:
            mask = np.zeros(tokenizer.vocab_size)
            masks.append(mask)
            continue
        # init mask
        mask = np.ones(tokenizer.vocab_size)
        # rule: token in the root should not be operands
        if pos == 0:
            mask[tokenizer.leaf_index] = 0
            # mask[tokenizer.encode('sign')]=0
            # mask[tokenizer.encode('sin')]=0
            # mask[tokenizer.encode('cos')]=0

            # mask[tokenizer.encode('*')]=0
            mask[tokenizer.encode('-')] = 0
        else:
            # rule: Avoid invalid operations of + -
            father_token = tokenizer.decode(sub_seq[(pos - 1) // 2])

            if (tokenizer.is_binary(father_token) and pos % 2 == 0) or tokenizer.is_unary(father_token):
                neg_ancestor, target_vocab = find_prefix_of_token_ancestor(tokenizer, sub_seq, pos, '-')
                # rule: direct child of - should not be - or +
                if neg_ancestor == (pos - 1) // 2:
                    mask[tokenizer.encode('+')] = 0
                    mask[tokenizer.encode('-')] = 0
                    # rule: direct child of - located in root should not be x
                    if neg_ancestor == 0:
                        mask[tokenizer.encode('x')] = 0

                if target_vocab is not None:
                    pre_vocab = along_continuous_plus(tokenizer, sub_seq, neg_ancestor)

                    if pre_vocab is not None:
                        mask_index = test_pre(target_vocab[1:], pre_vocab, tokenizer)
                        mask[mask_index] = 0

            if father_token == '+' or (tokenizer.is_binary(father_token) and pos % 2 == 0) or tokenizer.is_unary(father_token):
                plus_ancestor, target_vocab = find_prefix_of_token_ancestor(tokenizer, sub_seq, pos, '+')
                # print(f'plus_ancestor:{plus_ancestor}')
                if target_vocab is not None:
                    visited = np.zeros_like(sub_seq)
                    if father_token == '+' and left_or_right(pos, plus_ancestor) == 'l':
                        visited[2 * plus_ancestor + 1] = 1
                        target_vocab = get_prefix(sub_seq, 2 * plus_ancestor + 1)
                    else:
                        visited[2 * plus_ancestor + 2] = 1
                        target_vocab = get_prefix(sub_seq, 2 * plus_ancestor + 2)

                    sub_root_list = get_along_continuous_plus_with_minus(tokenizer, sub_seq, plus_ancestor, visited)

                    pre_vocab = [get_prefix(sub_seq, sub_root) for sub_root in sub_root_list]
                    if pre_vocab is not None:
                        mask_index = test_pre(target_vocab, pre_vocab, tokenizer)
                        mask[mask_index] = 0
            # rule: pure calculation between constant values is not allowed
            if have_continous_const(sub_seq, pos, tokenizer):
                mask[tokenizer.constants_index] = 0

            # rule: [sin cos sign] cannot directly nest with each other (if they are in the basis symbol set)
            # if father_token in ['sin','cos']:
            #     mask[tokenizer.encode('sign')]=0
            #     mask[tokenizer.encode('sin')]=0
            #     # mask[tokenizer.encode('cos')]=0
            # if father_token == 'sign':
            #     mask[tokenizer.encode('sign')]=0

            # rule: the direct children of + should not be constant values
            if father_token == '+' or father_token == '-':
                mask[tokenizer.constants_index] = 0

            if father_token == '+':
                # children of sign should not be sign (if sign is in the basis symbol set)
                # mask[tokenizer.encode('sign')]=0

                # rule: x+x, gbest+gbest ... is not allowed
                if pos % 2 == 0:
                    left_token = tokenizer.decode(sub_seq[pos - 1])
                    if tokenizer.is_leaf(left_token) and left_token != 'randx':
                        mask[sub_seq[pos - 1]] = 0

            # rule: children of * should not be the same
            if father_token == '*':
                mask[tokenizer.encode('*')] = 0
                mask[tokenizer.encode('-')] = 0
                if pos % 2 == 0:
                    left_id = sub_seq[pos - 1]
                    if not tokenizer.is_consts(left_id):
                        mask[tokenizer.non_const_index] = 0
                    else:
                        mask[tokenizer.constants_index] = 0

            # ! optional: set the minimum layer of the equation tree (you can uncomment the following code if needed)
            if which_layer(position = pos) <= 2:
                if father_token == '*':
                    mask[tokenizer.var_index] = 0
                elif (tokenizer.is_binary(father_token) and pos % 2 == 0 and tokenizer.is_leaf(tokenizer.decode(sub_seq[pos - 1]))) or tokenizer.is_unary(father_token):
                    mask[tokenizer.leaf_index] = 0

            # rule: the leaves should not be operators
            if pos >= int(2 ** (max_layer - 1) - 1):
                mask[tokenizer.operator_index] = 0
        # if np.all(mask<=0.2):
        #     # mask[tokenizer.leaf_index]=1
        #     print(f'mask:{mask}, pos:{pos}, seq:{sub_seq}')
        masks.append(mask)

    return torch.FloatTensor(masks).to(old_device)


def which_layer(position):
    level = math.floor(math.log2(position + 1))
    return level + 1


def left_or_right(position, root):
    tmp = position
    while tmp != root:
        position = (position - 1) // 2
        if position == root:
            if 2 * root + 1 == tmp:
                return 'l'
            else:
                return 'r'
        tmp = position


def have_continous_const(seq, position, tokenizer):
    father_index = (position - 1) // 2
    father_token = tokenizer.decode(seq[father_index])
    if tokenizer.is_unary(father_token):
        return True
    if tokenizer.is_binary(father_token):
        if position == father_index * 2 + 1:
            return False
        elif tokenizer.is_consts(seq[father_index * 2 + 1]):
            return True


def continus_mul_c(seq, position, tokenizer):
    list = []
    sub_root = (position - 1) // 2
    if tokenizer.decode(seq[sub_root]) == '*':
        visited = np.zeros_like(seq)
        visited[position] = 1

        return get_along_continuous_mul(tokenizer, seq, sub_root, visited)
    else:
        return False


def get_along_continuous_mul(tokenizer, seq, begin, visited):
    # list.append(begin)
    visited[begin] = 1

    if begin != 0 and visited[(begin - 1) // 2] != 1:
        father_token = tokenizer.decode(seq[(begin - 1) // 2])
        if father_token == '*':
            if get_along_continuous_mul(tokenizer, seq, (begin - 1) // 2, visited):
                return True

    if visited[begin * 2 + 1] == 0 and seq[begin * 2 + 1] != -1:
        left_child_token = tokenizer.decode(seq[begin * 2 + 1])
        if left_child_token == '*':
            if get_along_continuous_mul(tokenizer, seq, begin * 2 + 1, visited):
                return True
        elif left_child_token[0] == 'C':
            return True

    if visited[begin * 2 + 2] == 0 and seq[begin * 2 + 2] != -1:
        right_child_token = tokenizer.decode(seq[begin * 2 + 2])
        if right_child_token == '*':
            if get_along_continuous_mul(tokenizer, seq, begin * 2 + 2, visited):
                return True
        elif right_child_token[0] == 'C':
            return True

    return False


def test_pre(target_vocab, pre_vocab, tokenizer):
    target_len = len(target_vocab)
    mask_index = []
    for pre_prefix in pre_vocab:
        if len(pre_prefix) == target_len + 1 and np.all(pre_prefix[:-1] == target_vocab):
            last_token = tokenizer.decode(pre_prefix[-1])
            if last_token != 'randx' and last_token[0] != 'C':
                mask_index.append(pre_prefix[-1])

    return mask_index


def get_along_continuous_plus_with_minus(tokenizer, seq, begin, visited):
    list = []

    # list.append(begin)
    visited[begin] = 1

    if begin != 0 and visited[(begin - 1) // 2] == 0:
        father_token = tokenizer.decode(seq[(begin - 1) // 2])
        if father_token == '+':
            l = get_along_continuous_plus_with_minus(tokenizer, seq, (begin - 1) // 2, visited)
            list.extend(l)

    if visited[begin * 2 + 1] == 0 and seq[begin * 2 + 1] != -1:
        left_child_token = tokenizer.decode(seq[begin * 2 + 1])
        if left_child_token == '+':
            l = get_along_continuous_plus_with_minus(tokenizer, seq, begin * 2 + 1, visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2 * (begin * 2 + 1) + 1)

    if visited[begin * 2 + 2] == 0 and seq[begin * 2 + 2] != -1:
        right_child_token = tokenizer.decode(seq[begin * 2 + 2])
        if right_child_token == '+':
            l = get_along_continuous_plus_with_minus(tokenizer, seq, begin * 2 + 2, visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2 * (begin * 2 + 2) + 1)

    return list


def get_along_continuous_plus(tokenizer, seq, begin, visited):
    list = []
    # list.append(begin)
    along_root = False
    visited[begin] = 1
    if begin == 0 and seq[begin] == tokenizer.encode('+'):
        along_root = True

    if begin != 0 and visited[(begin - 1) // 2] == 0:
        father_token = tokenizer.decode(seq[(begin - 1) // 2])
        if father_token == '+':
            l, flag = get_along_continuous_plus(tokenizer, seq, (begin - 1) // 2, visited)
            list.extend(l)
            if flag:
                along_root = True

    if visited[begin * 2 + 1] == 0 and seq[begin * 2 + 1] != -1:
        left_child_token = tokenizer.decode(seq[begin * 2 + 1])
        if left_child_token == '+':
            l, flag = get_along_continuous_plus(tokenizer, seq, begin * 2 + 1, visited)
            list.extend(l)
            if flag:
                along_root = True
        else:
            list.append(begin * 2 + 1)

    if visited[begin * 2 + 2] == 0 and seq[begin * 2 + 2] != -1:
        right_child_token = tokenizer.decode(seq[begin * 2 + 2])
        if right_child_token == '+':
            l, flag = get_along_continuous_plus(tokenizer, seq, begin * 2 + 2, visited)
            list.extend(l)
            if flag:
                along_root = True
        else:
            list.append(begin * 2 + 2)

    return list, along_root


def along_continuous_plus(tokenizer, seq, neg_ancestor):
    list = []
    sub_root = (neg_ancestor - 1) // 2
    if tokenizer.decode(seq[sub_root]) == '+':
        visited = np.zeros_like(seq)
        visited[neg_ancestor] = 1
        continuous_plus_token_list, along_root = get_along_continuous_plus(tokenizer, seq, sub_root, visited)

        pre_vocab = [get_prefix(seq, sub_root) for sub_root in continuous_plus_token_list]

        if along_root:
            pre_vocab.append([tokenizer.encode('x')])
        return pre_vocab
    else:
        return None


def find_prefix_of_token_ancestor(tokenizer, seq, position, token):
    while True:
        father_index = (position - 1) // 2
        father_token = tokenizer.decode(seq[father_index])
        if father_token != token:
            position = father_index
            if position == 0:
                break
        else:
            return father_index, get_prefix(seq, father_index)
    return -1, None


def get_prefix(seq, sub_root):
    if sub_root >= len(seq) or seq[sub_root] == -1:
        return []
    list = []
    list.append(seq[sub_root])
    list.extend(get_prefix(seq, 2 * sub_root + 1))
    list.extend(get_prefix(seq, 2 * sub_root + 2))
    return list


def get_prefix_with_consts(seq, consts, sub_root):
    if sub_root >= len(seq) or seq[sub_root] == -1:
        return [], []
    list_expr = []
    list_c = []
    list_expr.append(seq[sub_root])
    list_c.append(consts[sub_root])
    left_output = get_prefix_with_consts(seq, consts, 2 * sub_root + 1)
    list_expr.extend(left_output[0])
    list_c.extend(left_output[1])
    right_output = get_prefix_with_consts(seq, consts, 2 * sub_root + 2)

    list_expr.extend(right_output[0])
    list_c.extend(right_output[1])
    return list_expr, list_c


def get_next_position(seq, choice, position, tokenizer):
    old_device = position.device
    position = position.cpu().numpy()
    choice = choice.cpu().numpy()
    seq = seq.cpu().numpy()
    next_position = []
    for i in range(len(position)):
        c = choice[i]
        pos = position[i]
        sub_seq = seq[i]
        if c in tokenizer.operator_index:
            next_position.append(2 * pos + 1)
        else:
            append_index = -1
            while True:
                father_index = (pos - 1) // 2
                if father_index < 0:
                    break
                if sub_seq[father_index] in tokenizer.binary_index and sub_seq[2 * father_index + 2] == -1:
                    append_index = father_index * 2 + 2
                    break
                pos = father_index
            next_position.append(append_index)

    return torch.tensor(next_position, dtype = torch.long).to(old_device)


#
def get_str_prefix(seq, const_vals, tokenizer):
    str_expr = []
    c = []
    for i, token_id in enumerate(seq):
        if token_id != -1:
            str_expr.append(tokenizer.decode(token_id))
            c.append(const_vals[i])
    return str_expr, c


def prefix_to_infix(
        expr, constants, tokenizer: Tokenizer
):
    stack = []
    for i, symbol in reversed(list(enumerate(expr))):
        if tokenizer.is_binary(symbol):
            if len(stack) < 2:
                return False, None
            tmp_str = "(" + stack.pop() + symbol + stack.pop() + ")"
            stack.append(tmp_str)
        elif tokenizer.is_unary(symbol) or symbol == "abs":
            if len(stack) < 1:
                return False, None
            if symbol in tokenizer.SPECIAL_SYMBOLS:
                stack.append(tokenizer.SPECIAL_SYMBOLS[symbol].format(stack.pop()))
            else:
                stack.append(symbol + "(" + stack.pop() + ")")
        elif tokenizer.is_leaf(symbol):
            if symbol == "C":
                stack.append(str(constants[i]))
            elif "C" in symbol:
                exponent = int(symbol[1:])
                stack.append(str(constants[i] * 10 ** exponent))
            else:
                stack.append(symbol)

    if len(stack) != 1:
        return False, None

    return True, stack.pop()


def expr_to_func(sympy_expr, variables: List[str]):
    return lambdify(
        variables,
        sympy_expr,
        modules = ["numpy"],
    )


def calculate_mean_distance(population):
    distances = cdist(population, population, metric = 'euclidean')

    np.fill_diagonal(distances, 0)

    mean_distance = np.mean(distances)

    return mean_distance
