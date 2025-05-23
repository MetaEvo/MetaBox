import numpy as np
from .learnable_optimizer import Learnable_Optimizer
from typing import Union, Iterable

class RLPSO_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    RLPSO develops a reinforcement learning strategy to enhance PSO in convergence by replacing the uniformly distributed random number in the updating function with a random number generated from a selected normal distribution.
    # Original paper
    "[**Employing reinforcement learning to enhance particle swarm optimization methods**](https://www.tandfonline.com/doi/abs/10.1080/0305215X.2020.1867120)." Engineering Optimization (2022).Intelligence. (2021).
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the RLPSO optimizer with the provided configuration, setting up key hyperparameters and internal state.
        # Args:
        - config (object): Configuration object containing optimizer parameters such as inertia weight decay, acceleration coefficient, population size, maximum function evaluations, and logging interval.
        # Built-in Attribute:
        - self.__config (object): Stores the configuration object.
        - self.__w_decay (bool): Indicates whether inertia weight decay is enabled.Default is True.
        - self.__w (float): Inertia weight value, initialized based on `w_decay`.Default is 0.9 if `w_decay` is True, otherwise 0.729.
        - self.__c (float): Acceleration coefficient. Default is 2.05.
        - self.__NP (int): Population size.Default is 100.
        - self.__max_fes (int): Maximum number of function evaluations.
        - self.fes (Any): Tracks the number of function evaluations (initialized as None).
        - self.cost (Any): Tracks the cost or fitness value (initialized as None).
        - self.log_index (Any): Tracks the logging index (initialized as None).
        - self.log_interval (int): Interval for logging progress.
        # Returns:
        - None
        """
        
        super().__init__(config)
        
        config.w_decay = True
        config.c = 2.05
        config.NP = 100
        self.__config = config

        self.__w_decay = config.w_decay
        if self.__w_decay:
            self.__w = 0.9
        else:
            self.__w = 0.729
        self.__c = config.c
        
        self.__NP = config.NP

        self.__max_fes = config.maxFEs
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        """
        Returns a string representation of the RLPSO_Optimizer instance.
        # Returns:
        - str: The name of the optimizer, "RLPSO_Optimizer".
        """
        
        return "RLPSO_Optimizer"

    # initialize PSO environment
    def init_population(self, problem):
        """
        # Introduction
        Initializes the particle population for the RLPSO (Reinforcement Learning Particle Swarm Optimization) algorithm, setting up positions, velocities, and tracking variables for optimization.
        # Args:
        - problem (object): The optimization problem object, which provides lower and upper bounds (`lb`, `ub`) for the search space.
        # Built-in Attribute:
        - self.__dim (int): The dimensionality of the problem, set to `problem.dim`.
        - self.__particles (dict): A dictionary to hold particle attributes such as current position, cost, personal best position, global best position, and velocity.
        - self.__max_velocity (float): The maximum velocity for particles, calculated based on the problem's bounds.
        - self.fes (int): The number of function evaluations, initialized to 0.
        - self.__max_cost (float): The maximum cost value among the particles, initialized based on the initial costs.
        - self.__cur_index (int): The current index of the particle being updated, initialized to 0.
        - self.log_index (int): The index for logging progress, initialized to 1.
        - self.cost (list): A list to store the global best cost values at specified intervals.
        - self.meta_X (list): A list to store the positions of particles for full meta-data logging.
        - self.meta_Cost (list): A list to store the costs of particles for full meta-data logging.
        - self.meta_tmp_x (list): A temporary list to store positions during a single iteration for full meta-data logging.
        - self.meta_tmp_cost (list): A temporary list to store costs during a single iteration for full meta-data logging.
        # Returns:
        - object: The initial state of the optimizer, as returned by `self.__get_state(self.__cur_index)`.
        # Notes:
        - Assumes that `self.rng` is a random number generator and that `self.__get_costs` and `self.__get_state` are defined elsewhere in the class.
        - The method is intended to be called at the start of the optimization process.
        """
        self.__dim = problem.dim
        rand_pos = self.rng.uniform(low=problem.lb, high=problem.ub, size=(self.__NP, self.__dim))
        self.fes = 0
        self.__max_velocity=0.1*(problem.ub-problem.lb)
        rand_vel = self.rng.uniform(low=-self.__max_velocity, high=self.__max_velocity, size=(self.__NP, self.__dim))

        c_cost = self.__get_costs(problem,rand_pos) # ps

        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]
        self.__max_cost = np.max(c_cost)

        
        self.__particles={'current_position': rand_pos.copy(),  # ?ps, dim
                          'c_cost': c_cost.copy(),  # ?ps
                          'pbest_position': rand_pos.copy(),  # ps, dim
                          'pbest': c_cost.copy(),  # ?ps
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'velocity': rand_vel.copy(),  # ps,dim
                          'gbest_index': gbest_index  # 1
                          }
        if self.__w_decay:
            self.__w = 0.9
            
        self.__cur_index = 0
        self.log_index = 1
        self.cost = [self.__particles['gbest_val']]

        if self.__config.full_meta_data:
            self.meta_X = [rand_pos.copy()]
            self.meta_Cost = [c_cost.copy()]
            self.meta_tmp_x = []
            self.meta_tmp_cost = []

        return self.__get_state(self.__cur_index)

    def __get_state(self, index):
        """
        # Introduction
        Retrieves the current state representation for a given particle by concatenating the global best position and the current position of the specified particle.
        # Args:
        - index (int): The index of the particle whose state is to be retrieved.
        # Returns:
        - np.ndarray: A 1D array representing the concatenated state of the global best position and the current position of the specified particle.
        # Raises:
        - IndexError: If `index` is out of bounds for the current positions array.
        """
        return np.concatenate((self.__particles['gbest_position'], self.__particles['current_position'][index]), axis=-1)

    # calculate costs of solutions
    def __get_costs(self, problem, position):
        """
        # Introduction
        Computes the cost(s) of a given position or set of positions for a specified optimization problem, updating the function evaluation count.
        # Args:
        - problem: An object representing the optimization problem, expected to have `eval(position)` and `optimum` attributes.
        - position (np.ndarray): The position(s) in the search space for which the cost is to be evaluated. Can be a 1D or 2D numpy array.
        # Built-in Attribute:
        - self.fes (int): Tracks the number of function evaluations performed.
        # Returns:
        - float or np.ndarray: The computed cost(s) for the given position(s). If `problem.optimum` is set, returns the difference between the evaluated value and the optimum.
        # Raises:
        - AttributeError: If `problem` does not have the required `eval` method or `optimum` attribute.
        """
        
        if len(position.shape) == 2:
            self.fes += position.shape[0]
        else:
            self.fes += 1
        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the RL-PSO (Reinforcement Learning Particle Swarm Optimization) optimizer for a single particle based on the provided action and problem definition. This includes updating velocity, position, personal best, and global best, as well as calculating rewards and logging progress.
        # Args:
        - action (np.ndarray or float): The action to be taken, typically representing a random factor for velocity update.
        - problem (object): The optimization problem instance, which must provide lower and upper bounds (`lb`, `ub`), and optionally an optimum value.
        # Returns:
        - state (np.ndarray): The updated state representation for the next step.
        - reward (float): The reward signal computed from the improvement in cost.
        - is_done (bool): Whether the optimization process has reached its termination condition.
        - info (dict): Additional information (currently empty, reserved for future use).
        # Notes:
        - The method linearly decreases the inertia coefficient if enabled.
        - Velocity and position are updated according to the PSO update rules, with velocity clipping and position boundary handling.
        - Updates personal and global bests if improvements are found.
        - Logs global best values at specified intervals.
        - Handles full meta-data logging if configured.
        - Ensures the cost log is filled up to the required number of log points upon completion.
        """
        
        is_done = False

        # record the gbest_val in the begining
        self.__pre_gbest = self.__particles['gbest_val']

        # linearly decreasing the coefficient of inertia w
        if self.__w_decay:
            self.__w -= 0.5 / (self.__max_fes / self.__NP)

        # generate two set of random val for pso velocity update
        rand1 = self.rng.rand()
        rand2 = np.squeeze(action)
       
        j = self.__cur_index
        v = self.__particles['velocity'][j]
        x = self.__particles['current_position'][j]
        pbest_pos = self.__particles['pbest_position'][j]
        gbest_pos = self.__particles['gbest_position']
        pre_cost = self.__particles['c_cost'][j]

        # update velocity
        new_velocity = self.__w*v+self.__c*rand1*(pbest_pos-x)+self.__c*rand2*(gbest_pos-x)

        # clip the velocity if exceeding the boarder
        new_velocity = np.clip(new_velocity, -self.__max_velocity, self.__max_velocity)
        
        # update position
        new_x = x+new_velocity

        # print("velocity.shape = ",new_velocity.shape)
        new_x = clipping(new_x, problem.lb, problem.ub)

        # update population
        self.__particles['current_position'][j] = new_x
        self.__particles['velocity'][j] = new_velocity

        # calculate the new costs
        new_cost = self.__get_costs(problem, new_x)
        self.__particles['c_cost'][j] = new_cost
        
        # update pbest
        if new_cost < self.__particles['pbest'][j]:
            self.__particles['pbest'][j] = new_cost
            self.__particles['pbest_position'][j] = new_x
        # update gbest
        if new_cost < self.__particles['gbest_val']:
            self.__particles['gbest_val'] = new_cost
            self.__particles['gbest_position'] = new_x
            self.__particles['gbest_index'] = j

        # see if the end condition is satisfied
        if problem.optimum is None:
            is_done = self.fes >= self.__max_fes
        else:
            is_done = self.fes >= self.__max_fes

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.__particles['gbest_val'])

        reward = (pre_cost-new_cost)/(self.__max_cost-self.__particles['gbest_val'])

        if self.__config.full_meta_data:
            self.meta_tmp_x.append(self.__particles['current_position'][j].copy())
            self.meta_tmp_cost.append(self.__particles['c_cost'][j].copy())

            # 在某一轮迭代结束后（例如在 for j in range(NP) 之后）
            if len(self.meta_tmp_cost) == self.__NP:  # 或 len(self.meta_tmp_x) == NP
                self.meta_X.append(np.array(self.meta_tmp_x))
                self.meta_Cost.append(np.array(self.meta_tmp_cost))

                self.meta_tmp_x.clear()
                self.meta_tmp_cost.clear()


        self.__cur_index = (self.__cur_index+1) % self.__NP

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.__particles['gbest_val']
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.__particles['gbest_val'])
                
        info = {}
        
        return self.__get_state(self.__cur_index), reward, is_done, info

def clipping(x: Union[np.ndarray, Iterable],
             lb: Union[np.ndarray, Iterable, int, float, None],
             ub: Union[np.ndarray, Iterable, int, float, None]
             ) -> np.ndarray:
    return np.clip(x, lb, ub)
