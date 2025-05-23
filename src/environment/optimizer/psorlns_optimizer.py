from .learnable_optimizer import Learnable_Optimizer
import torch
import numpy as np
from scipy.spatial import distance

class PSORLNS_Optimizer(Learnable_Optimizer):
    """
    # PSORLNS_Optimizer
    An implementation of a learnable optimizer that combines Particle Swarm Optimization (PSO) with Reinforcement Learning-based Neighborhood Search (RLNS). This optimizer maintains a population of particles and updates their positions and velocities based on both PSO dynamics and adaptive neighborhood strategies, aiming to solve optimization problems efficiently.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the PSORLNS optimizer with the provided configuration and sets default parameters for the optimization process.
        # Args:
        - config (object): Config object containing parameters for the optimizer.
            - The Attributes needed for the DEDDQN_Optimizer are the following:
                - n_logpoint (int): Number of log points for the optimizer.
                - full_meta_data (bool): Flag to indicate if full meta-data should be collected.
        # Attributes:
        - w (float): Inertia weight for the optimizer.Default is 1.
        - c1 (float): Cognitive coefficient.Default is 1.49445.
        - c2 (float): Social coefficient.Default is 1.49445.
        - ps (int): Population size.Default is 100.
        - TT2 (float): Threshold for selecting worse or better solutions.Default is 0.8.
        - neighbor_num (list of int): List of neighbor counts for local search.Default is [5, 10, 20, 30, 40].
        - fes (Any): Function evaluation counter (initialized as None).
        - cost (Any): Cost value (initialized as None).
        - pr (Any): Probability rate (initialized as None).
        - sr (Any): Success rate (initialized as None).
        - log_index (Any): Logging index (initialized as None).
        - log_interval (Any): Logging interval (initialized as None).
        - archive (Any): Archive for storing solutions (initialized as None).
        - archive_val (Any): Archive for storing solution values (initialized as None).
        """
        super().__init__(config)
        self.__config = config

        self.w = 1
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.ps = 100
        
        self.TT2 = 0.8 # 选择worse better的随机数阈值
        self.neighbor_num = [5,10,20,30,40]
        

        self.fes = None
        self.cost = None
        self.pr = None
        self.sr = None
        self.log_index = None
        self.log_interval = None
        self.archive = None
        self.archive_val = None

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the PSORLNS_Optimizer instance.
        # Returns:
        - str: The name of the optimizer, "PSORLNS_Optimizer".
        """
        
        return "PSORLNS_Optimizer"

    def cal_pr_sr(self, problem):
        """
        # Introduction
        Calculates the Peak Ratio (PR) and Success Rate (SR) at multiple accuracy levels for the current solution archive with respect to the given optimization problem.
        # Args:
        - problem: The optimization problem object, which provides the attributes `nopt` (number of global optima) and the method `how_many_goptima` for evaluating solutions.
        # Returns:
        - raw_PR (np.ndarray): Array of shape (5,) containing the peak ratio at each accuracy level.
        - raw_SR (np.ndarray): Array of shape (5,) containing the success rate at each accuracy level.
        # Notes:
        - The method uses five predefined accuracy thresholds: 1e-1, 1e-2, 1e-3, 1e-4, and 1e-5.
        - The solution archive (`self.archive`) and their corresponding values (`self.archive_val`) are used to determine which solutions meet each accuracy level.
        - The peak ratio is the fraction of global optima found at each accuracy level.
        - The success rate is set to 1 if all global optima are found at a given accuracy level, otherwise 0.
        """
        
        raw_PR = np.zeros(5)
        raw_SR = np.zeros(5)
        solu = self.archive.copy()
        solu_val = self.archive_val.copy()
        # assert (self.get_costs(solu, problem) == solu_val).all()
        accuracy = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        total_pkn = problem.nopt
        for acc_level in range(5):
            sub_sol = solu[solu_val < accuracy[acc_level]].copy()
            nfp, _ = problem.how_many_goptima(sub_sol, accuracy[acc_level])
            raw_PR[acc_level] = nfp / total_pkn
            if nfp >= total_pkn:
                raw_SR[acc_level] = 1
        return raw_PR, raw_SR

    # initialize GPSO environment
    def initialize_particles(self, problem):
        """
        # Introduction
        Initializes the particles for the PSO (Particle Swarm Optimization) algorithm by generating random positions and velocities within the problem's bounds, and computes initial costs and best values.
        # Args:
        - problem: The optimization problem object, which has attributes `lb` (lower bounds), `ub` (upper bounds), and be compatible with the `get_costs` method.
        # Returns:
        - None: Updates the `self.particles` attribute with initialized positions, velocities, costs, and best values.
        # Side Effects:
        - Sets `self.particles` to a dictionary containing:
            - 'current_position': Initial positions of all particles.
            - 'c_cost': Initial costs of all particles.
            - 'pbest_position': Initial personal best positions.
            - 'pbest': Initial personal best costs.
            - 'gbest_position': Initial global best position.
            - 'gbest_val': Initial global best cost.
            - 'velocity': Initial velocities of all particles.
            - 'pop_dist': Pairwise distance matrix of all particles.
        """
        
        # randomly generate the position and velocity
        rand_pos = self.rng.uniform(low = problem.lb, high = problem.ub, size = (self.ps, self.dim))
        rand_vel = self.rng.uniform(low = -self.max_velocity, high = self.max_velocity, size = (self.ps, self.dim))

        # get the initial cost
        c_cost = self.get_costs(rand_pos, problem)  # ps

        # find out the gbest_val
        gbest_val = np.min(c_cost)
        gbest_index = np.argmin(c_cost)
        gbest_position = rand_pos[gbest_index]

        pop_dist = distance.cdist(rand_pos, rand_pos)

        # store all the information of the paraticles
        self.particles = {'current_position': rand_pos.copy(),  # ps, dim
                          'c_cost': c_cost.copy(),  # ps
                          'pbest_position': rand_pos.copy(),  # ps, dim
                          'pbest': c_cost.copy(),  # ps
                          'gbest_position': gbest_position.copy(),  # dim
                          'gbest_val': gbest_val,  # 1
                          'velocity': rand_vel.copy(),  # ps,dim
                          'pop_dist': pop_dist.copy()
                          }

    # the interface for environment reseting
    def init_population(self, problem):
        """
        # Introduction
        Initializes the population of particles for the PSO-RLNS optimizer, sets up problem-specific parameters, and prepares logging and meta-data structures.
        # Args:
        - problem (object): The optimization problem, which has attributes such as `maxfes`, `dim`, `ub`, and `lb`.
        # Built-in Attributes:
        - `self.max_fes`: Maximum function evaluations allowed for the problem.
        - `self.log_interval`: Interval for logging progress.
        - `self.dim`: Dimensionality of the problem.
        - `self.fes`: Function evaluation counter. Default is 0.
        - `self.max_velocity`: Maximum velocity for particles, calculated based on the problem's bounds.
        - `self.w`: Inertia weight for the optimizer, default is 1.
        - `self.archive`: Archive for storing solutions, initialized as an empty array.
        - `self.archive_val`: Archive for storing solution values, initialized as an empty array.
        - `self.max_dist`: Maximum distance between particles, calculated based on the problem's bounds.
        - `self.eps`: Threshold for neighborhood size, default is 0.1.
        - `self.log_index`: Index for logging progress, initialized to 1.
        - `self.cost`: List to store the best cost values during optimization.
        - `self.pr`: List to store the peak ratio values during optimization.
        - `self.sr`: List to store the success rate values during optimization.
        - `self.meta_X`: List to store the positions of particles for meta-data collection (if enabled).
        - `self.meta_Cost`: List to store the costs of particles for meta-data collection (if enabled).
        - `self.meta_Pr`: List to store the peak ratio values for meta-data collection (if enabled).
        - `self.meta_Sr`: List to store the success rate values for meta-data collection (if enabled).
        # Returns:
        - np.ndarray: The initial observed state of the population, as returned by the `observe()` method.
        # Notes:
        - Resets or initializes several optimizer attributes, including function evaluation counters, velocity limits, and logging intervals.
        - Initializes the particle population and their associated costs and statistics.
        - Optionally stores meta-data if configured to do so.
        """
        
        self.max_fes = problem.maxfes
        self.log_interval =(self.max_fes // self.__config.n_logpoint)
        self.dim = problem.dim
        self.fes = 0
        self.max_velocity = 0.1 * (problem.ub - problem.lb)
        # set the hyperparameters back to init value if needed
        self.w = 1
        self.archive = np.zeros((0, self.dim))
        self.archive_val = np.array([])

        self.max_dist = np.sqrt(np.sum((problem.ub - problem.lb) ** 2))
        self.eps =  0.1 # todo: 邻域个数判断阈值
        # initialize the population
        self.initialize_particles(problem)
        self.archive = np.vstack((self.archive, self.particles['current_position'][self.particles['c_cost'] < 1e-1].copy()))
        self.archive_val = np.hstack((self.archive_val, self.particles['c_cost'][self.particles['c_cost']<1e-1].copy()))

        self.log_index = 1
        self.cost = [self.particles['gbest_val']]
        raw_pr, raw_sr = self.cal_pr_sr(problem)
        self.pr = [raw_pr.copy()]
        self.sr = [raw_sr.copy()]

        # get state

        # get the population state
        state = self.observe()  # ps, 9


        if self.__config.full_meta_data:
            self.meta_X = [self.particles['current_position'].copy()]
            self.meta_Cost = [self.particles['c_cost'].copy()]
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.meta_Pr = [raw_pr.copy()]
            self.meta_Sr = [raw_sr.copy()]

        return state

    # calculate costs of solutions
    def get_costs(self, position, problem):
        """
        # Introduction
        Calculates the cost values for a given set of positions by evaluating them on the provided problem and adjusting by the problem's optimum value. Also updates the function evaluation count.
        # Args:
        - position (np.ndarray): An array of candidate solutions to be evaluated.
        - problem (object): An optimization problem instance with `eval` and `optimum` attributes.
        # Built-in Attribute:
        - self.fes (int): Counter for the number of function evaluations, incremented by the batch size.
        # Returns:
        - np.ndarray: The computed cost values for each position, adjusted by the problem's optimum.
        # Raises:
        - AttributeError: If `problem` does not have `eval` or `optimum` attributes.
        """
        
        ps = position.shape[0]
        self.fes += ps
        cost = problem.eval(position) - problem.optimum
        return cost

    # feature encoding
    def observe(self):
        """
        # Introduction
        Computes a normalized state representation based on the neighborhood structure of particles in the optimizer.
        # Args:
        None
        # Returns:
        - np.ndarray: A (ps, 1) array where each entry represents the normalized count of neighbors for each particle.
        # Raises:
        None
        """
        
        state = np.zeros((self.ps, 1))
        pop_dist = self.particles['pop_dist'].copy()
        pop_dist[range(self.ps), range(self.ps)] = np.inf
        neighbor_matrix = np.zeros((self.ps, self.ps))
        neighbor_matrix[pop_dist < self.eps] = 1
        sum_neighbors = np.sum(neighbor_matrix, axis = -1, keepdims=True)
        state = sum_neighbors / 100
        
        return state

    # direct reward function
    def cal_reward(self, current_cost, parent_cost):
        """
        # Introduction
        Calculates the reward for each solution in the population based on the comparison between current and parent costs.
        # Args:
        - current_cost (np.ndarray): An array of current costs for each solution in the population.
        - parent_cost (np.ndarray): An array of parent costs for each solution in the population.
        # Returns:
        - np.ndarray: An array of rewards where each element is 1 if the current cost is less than the parent cost, -1 if greater, and 0 otherwise.
        # Notes:
        - The length of `current_cost` and `parent_cost` should match the population size (`self.ps`).
        """
        reward = np.zeros(self.ps)
        reward[current_cost < parent_cost]  = 1
        reward[current_cost > parent_cost] = -1
        
        return reward

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the PSO-RLNS optimizer for one iteration, applying particle swarm optimization and local neighborhood search strategies based on the provided actions and problem definition.
        # Args:
        - action (np.ndarray): An array of actions (typically integer indices) specifying the neighborhood size or strategy for each particle.
        - problem (object): An object representing the optimization problem, which must provide lower and upper bounds (`lb`, `ub`) and a cost evaluation method.
        # Returns:
        - next_state (np.ndarray): The observed state of the population after the update (shape: [ps, 9]).
        - reward (np.ndarray): The calculated reward for the current step, based on the improvement in cost.
        - is_end (np.ndarray): A boolean array indicating whether the termination condition has been met for each particle.
        - info (dict): Additional information (currently empty, reserved for future use).
        # Notes:
        - The method updates particle positions and velocities using a combination of PSO and RLNS-inspired rules.
        - It maintains and updates the best-known positions and costs for each particle and the global best.
        - Logging and meta-data collection are performed if enabled in the configuration.
        - The function assumes that the optimizer's state (e.g., `self.particles`, `self.fes`, etc.) is managed externally and updated in-place.
        """
        
        is_end = [False] * self.ps

        # record the gbest_val in the begining
        parent_cost = self.particles['c_cost'].copy()


        # generate two set of random val for pso velocity update
        new_position = np.zeros((self.ps, self.dim))
        new_velocity = np.zeros((self.ps, self.dim))
        
        pop_dist = self.particles['pop_dist'].copy()
        pop_dist[range(self.ps), range(self.ps)] = np.inf
        rank_dist = np.argsort(pop_dist, axis = -1)
        c_pos = self.particles['current_position'].copy()
        for i in range(self.ps):
            neighbors = rank_dist[i][:self.neighbor_num[action[i]]]
            # neighbors = np.append(neighbors, i)
            neighbors = neighbors[np.argsort(self.particles['c_cost'][neighbors])[::-1]]
            k = len(neighbors)
            
            rand1 = self.rng.rand()
            rand2 = self.rng.rand()
            if self.rng.rand() <= self.TT2:
                worse = neighbors[0]
            else:
                worse = neighbors[self.rng.randint(0, 0.05*k + 1)]

            if self.rng.rand() <= self.TT2:
                better = neighbors[-1]
            else:
                better = neighbors[self.rng.randint(0.95*k - 1, k)]
            if self.rng.rand() < 0.5:
                # update velocity
                new_velocity[i] = self.w * self.particles['velocity'][i] + self.c1 * rand1 * (self.particles['pbest_position'][i] - self.particles['current_position'][i]) + \
                            self.c2 * rand2 * (self.particles['current_position'][better] - self.particles['current_position'][i])
                # clip the velocity if exceeding the boarder
                new_velocity[i] = np.clip(new_velocity[i], -self.max_velocity, self.max_velocity)
                # update position according the boarding method
                raw_position = self.particles['current_position'][i] + new_velocity[i]
                new_position[i] = np.clip(raw_position, problem.lb, problem.ub)
            else:
                raw_position = c_pos[i] + self.rng.rand() * (c_pos[i] - c_pos[worse]) + self.rng.rand() * (c_pos[better] - c_pos[i]) 
                new_position[i] = np.clip(raw_position, problem.lb, problem.ub)
                # update velocity
                new_velocity[i] =  self.w * self.particles['velocity'][i] + self.c1 * rand1 * (self.particles['pbest_position'][i] - self.particles['current_position'][i]) + \
                            self.c2 * rand2 * (self.particles['current_position'][better] - self.particles['current_position'][i])
                # clip the velocity if exceeding the boarder
                new_velocity[i] = np.clip(new_velocity[i], -self.max_velocity, self.max_velocity)
                # update position according the boarding method
                
        # calculate the new costs
        new_cost = self.get_costs(new_position, problem)
        particles_filter = new_cost < self.particles['c_cost']
        accept_new_position = self.particles['current_position'].copy()
        accept_new_cost = self.particles['c_cost'].copy()
        accept_new_position[particles_filter] = new_position[particles_filter].copy()
        accept_new_cost[particles_filter] = new_cost[particles_filter].copy()

        chosen_pop = new_position[particles_filter].copy()
        chosen_val = new_cost[particles_filter].copy()
        self.archive = np.vstack((self.archive, chosen_pop[chosen_val < 1e-1].copy()))
        self.archive_val = np.hstack((self.archive_val, chosen_val[chosen_val<1e-1].copy()))

        # update particles
        filters = accept_new_cost < self.particles['pbest']

        new_cbest_val = np.min(accept_new_cost)
        new_cbest_index = np.argmin(accept_new_cost)
        filters_best_val = new_cbest_val < self.particles['gbest_val']

        new_pop_dist = distance.cdist(accept_new_position, accept_new_position)

        new_particles = {'current_position': accept_new_position.copy(),
                         'c_cost': accept_new_cost.copy(),
                         'pbest_position': np.where(np.expand_dims(filters, axis = -1),
                                                    accept_new_position.copy(),
                                                    self.particles['pbest_position']),
                         'pbest': np.where(filters,
                                           accept_new_cost.copy(),
                                           self.particles['pbest']),
                         'velocity': new_velocity.copy(),
                         'gbest_val': new_cbest_val if filters_best_val else self.particles['gbest_val'],
                         'gbest_position': np.where(np.expand_dims(filters_best_val, axis = -1),
                                                    accept_new_position[new_cbest_index],
                                                    self.particles['gbest_position']),
                         'pop_dist': new_pop_dist.copy()
                         }

        # update the population
        self.particles = new_particles

        if self.__config.full_meta_data:
            self.meta_X.append(self.particles['current_position'].copy())
            self.meta_Cost.append(self.particles['c_cost'].copy())
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.meta_Pr.append(raw_pr.copy())
            self.meta_Sr.append(raw_sr.copy())


        # see if the end condition is satisfied
        is_end = np.array([self.fes >= self.max_fes] * self.ps)

        # cal the reward
        reward = self.cal_reward(new_cost, parent_cost)

        # get the population next_state
        next_state = self.observe()  # ps, 9

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.particles['gbest_val'])
            raw_pr, raw_sr = self.cal_pr_sr(problem)
            self.pr.append(raw_pr.copy())
            self.sr.append(raw_sr.copy())

            
        if is_end[0]:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.particles['gbest_val']
                raw_pr, raw_sr = self.cal_pr_sr(problem)
                self.pr[-1] = raw_pr.copy()
                self.sr[-1] = raw_sr.copy()
            else:
                self.cost.append(self.particles['gbest_val'])
                raw_pr, raw_sr = self.cal_pr_sr(problem)
                self.pr.append(raw_pr.copy())
                self.sr.append(raw_sr.copy())
            


        info = {}
        return next_state, reward, is_end, info