from .learnable_optimizer import Learnable_Optimizer
import numpy as np
import math

class NRLPSO_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    NRLPSO is a reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy.
    # Original paper
    "[**Reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy**](https://www.sciencedirect.com/science/article/pii/S2210650223000482)." Swarm and Evolutionary Computation (2023).
    # Official Implementation
    None
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration and sets up default parameters for the optimization process.
        # Args:
        - config (object): Config object containing optimizer settings.
            - The Attributes needed for the NRLPSO are the following:
                - maxFEs (int): Maximum number of function evaluations.
                - log_interval (int): Interval for logging progress.
                - n_logpoint (int): Number of log points to record.
                - full_meta_data (bool): Flag for using full meta data.
        # Built-in Attribute:
        - NP (int): Number of particles, default is 100.
        - k (int): Number of neighbors or clusters, default is 5.
        - total_state (int): Total number of states, default is 4.
        - w_max (float): Maximum inertia weight, default is 1.
        - w_min (float): Minimum inertia weight, default is 0.4.
        - u (float): Parameter for algorithm control, default is 0.6.
        - v (float): Parameter for algorithm control, default is 0.33.
        - v_ratio (float): Ratio parameter for velocity, default is 0.1.
        - n_state (int): Number of states, default is 4.
        - __maxFEs (int): Maximum number of function evaluations, taken from config.
        - cost (list or None): List to store costs for backbone optimizers.
        - log_index (int or None): Index for logging.
        - log_interval (int): Interval for logging, taken from config.
        # Returns:
        - None
        # Raises:
        - AttributeError: If required attributes are missing in the `config` object.
        """
        
        super().__init__(config)
        self.__config = config
        
        self.NP = 100
        self.k = 5
        self.total_state = 4
        self.w_max = 1
        self.w_min = 0.4
        self.u = 0.6
        self.v = 0.33
        self.v_ratio = 0.1
        self.n_state = 4

        self.__maxFEs = config.maxFEs

        self.cost = None  # a list of costs that need to be maintained by EVERY backbone optimizers
        self.log_index = None
        self.log_interval = config.log_interval
        

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and related attributes for the optimizer based on the given problem definition.
        # Args:
        - problem: An object representing the optimization problem, which must provide `lb` (lower bounds), `ub` (upper bounds), `eval` (evaluation function), and optionally `optimum` (known optimum value).
        # Built-in Attributes:
        - __dim (int): Dimensionality of the problem.
        - pointer (int): Current index in the population.Default is 0.
        - __population (np.ndarray): The current population of particles.
        - __velocity (np.ndarray): The current velocity of particles.
        - __cost (np.ndarray): The cost of each particle in the population.
        - __pbest_pos (np.ndarray): The personal best position of each particle.
        - __pbest_cost (np.ndarray): The personal best cost of each particle.
        - __gbest_pos (np.ndarray): The global best position found so far.
        - __gbest_cost (float): The global best cost found so far.
        - pbest_stag_count (np.ndarray): Stagnation count for each particle.
        - fes (int): Function evaluation count.
        - log_index (int): Index for logging progress.
        - cost (list): List to store costs at each logging interval.
        - meta_X (list): List to store population positions for meta-data logging.
        - meta_Cost (list): List to store costs for meta-data logging.
        - meta_tmp_x (list): Temporary list to store population positions for meta-data logging.
        - meta_tmp_cost (list): Temporary list to store costs for meta-data logging.
        - r_w (float): Random weight for exploration.
        - __state (np.ndarray): The current state of each particle.
        # Returns:
        - int: The initial state of the individual at the current pointer index.
        # Notes:
        - Initializes population positions, velocities, personal and global bests, and other tracking variables.
        - Handles optional meta-data logging if enabled in the configuration.
        - Sets up the initial state for each individual in the population.
        """
        self.__dim = problem.dim
        self.pointer = 0
        # init population
        self.__population = self.rng.rand(self.NP, self.__dim) * (problem.ub - problem.lb) + problem.lb
        
        self.v_min = -0.1 * (problem.ub - problem.lb)
        self.v_max = -self.v_min
        self.__velocity = np.zeros(shape=(self.NP, self.__dim))
        if problem.optimum is None:
            self.__cost = problem.eval(self.__population)
        else:
            self.__cost = problem.eval(self.__population) - problem.optimum
        self.__pbest_pos = self.__population.copy()
        self.__pbest_cost = self.__cost.copy()

        gbest_index = np.argmin(self.__cost)
        self.__gbest_cost = self.__cost[gbest_index]
        self.__gbest_pos = self.__population[gbest_index]

        self.pbest_stag_count = np.zeros((self.NP, ))
        
        self.fes = self.NP
        self.log_index = 1
        self.cost = [self.__gbest_cost]

        if self.__config.full_meta_data:
            self.meta_X = [self.__population.copy()]
            self.meta_Cost = [self.__cost.copy()]
            self.meta_tmp_x = []
            self.meta_tmp_cost = []

        self.r_w = self.rng.rand()

        # init state
        self.__state = self.rng.randint(low=0, high=self.n_state, size=self.NP)

        return self.__state[self.pointer]

    
    def update_construct_neighborhood(self):
        """
        # Introduction
        Updates the neighborhood information for both personal best (pbest) and global best (gbest) particles in the population. This method computes the k-nearest neighbors for each particle based on Euclidean distance, and updates the corresponding neighborhood indices and positions.
        # Args:
        None
        # Updates:
        - self.pbest_neb_index (np.ndarray): Indices of the k-nearest neighbors for each particle's personal best position.
        - self.pbest_neb (np.ndarray): Positions of the k-nearest neighbors for each particle's personal best.
        - self.gbest_neb_index (np.ndarray): Indices of the k-nearest neighbors for the global best position.
        - self.gbest_neb (np.ndarray): Positions of the k-nearest neighbors for the global best.
        # Notes:
        - Assumes that `self.__population`, `self.__pbest_pos`, `self.__gbest_pos`, `self.NP`, and `self.k` are properly initialized.
        - Uses Euclidean distance to determine neighborhood proximity.
        """

        # pbest neb
        pbest_to_every_distance = np.sqrt(np.sum((self.__pbest_pos[None, :] - self.__population[:, None])**2, axis=-1))
        id_index = np.arange(self.NP)

        pbest_to_every_distance[id_index, id_index] = math.inf


        sort_index = np.argsort(pbest_to_every_distance, -1)

        n_index = sort_index[:, :self.k]
        self.pbest_neb_index = n_index
        self.pbest_neb = self.__population[n_index]

        # gbest neb
        gbest_to_every_distance = np.sqrt(np.sum((self.__gbest_pos[None, :] - self.__population)**2, axis=-1))

        sort_index = np.argsort(gbest_to_every_distance, -1)

        n_index = sort_index[:self.k]

        self.gbest_neb = self.__population[n_index]
        self.gbest_neb_index = n_index

    def cal_w(self):
        """
        # Introduction
        Calculates the inertia weight (`w`) for the optimizer based on the current function evaluations and random weight.
        # Built-in Attribute:
        - self.r_w (float): Random weight, updated using a logistic map.
        - self.u (float): Upper bound parameter for inertia weight calculation.
        - self.fes (int): Current number of function evaluations.
        - self.__maxFEs (int): Maximum number of function evaluations.
        - self.w_min (float): Minimum inertia weight.
        - self.w_max (float): Maximum inertia weight.
        - self.v (float): Scaling parameter for inertia weight adjustment.
        # Returns:
        - float: The calculated inertia weight (`w`) for the current iteration.
        """

        self.r_w = 4 * self.r_w * (1 - self.r_w)

        w = self.u - ((self.fes / self.__maxFEs) * self.r_w * self.w_min + self.v * (self.w_max - self.w_min) * (self.fes / self.__maxFEs))

        return w

    def cal_reward(self, f_new, f_old, ef_new, ef_old):
        """
        # Introduction
        Calculates a reward value based on the comparison of new and old fitness and error values.
        # Args:
        - f_new (float): The new fitness value.
        - f_old (float): The previous (old) fitness value.
        - ef_new (float): The new error value.
        - ef_old (float): The previous (old) error value.
        # Returns:
        - int: The reward value, determined as follows:
            - 2 if the new fitness is better (lower) and the new error is worse (higher).
            - 1 if the new fitness is better (lower) but the new error is not worse.
            - 0 if the new fitness is not better but the new error is worse (higher).
            - -2 if neither the new fitness is better nor the new error is worse.
        """
        
        cond1 = f_new < f_old
        cond2 = ef_new > ef_old

        r = None
        if cond1 and cond2:
            r = 2
        elif cond1 and not cond2:
            r = 1
        elif not cond1 and cond2:
            r = 0
        elif not cond1 and not cond2:
            r = -2
        return r

    def cal_ef(self, ith):
        """
        # Introduction
        Calculates the normalized effectiveness factor for the `ith` element based on its distance.
        # Args:
        - ith (int): The index of the element for which the effectiveness factor is to be calculated.
        # Built-in Attribute:
        - self.distance (list or np.ndarray): The list or array of distances for all elements.
        - self.d_min (float): The minimum distance among all elements.
        - self.d_max (float): The maximum distance among all elements.
        # Returns:
        - float: The normalized effectiveness factor for the specified element.
        # Raises:
        - ZeroDivisionError: If `self.d_max` equals `self.d_min`, resulting in division by zero.
        """
        
        self.update_distance()
        return (self.distance[ith] - self.d_min) / (self.d_max - self.d_min)
    
    def update_distance(self):
        """
        # Introduction
        Computes the pairwise Euclidean distances between individuals in the population and updates distance-related attributes.
        # Args:
        None
        # Built-in Attribute:
        - self.__population (np.ndarray): The current population array of shape (NP, D).
        - self.NP (int): The number of individuals in the population.
        # Updates:
        - self.distance (np.ndarray): The average distance from each individual to all others.
        - self.d_min (float): The minimum average distance among all individuals.
        - self.d_max (float): The maximum average distance among all individuals.
        # Returns:
        None
        """
        
        p1 = self.__population[None, :]
        p2 = self.__population[:, None]
        distance = np.sqrt(np.sum((p1 - p2)**2, -1))

        self.distance = np.sum(distance, -1) / (self.NP - 1)

        self.d_min = np.min(self.distance)
        self.d_max = np.max(self.distance)

    def cal_cs(self, p1, p2):
        """
        # Introduction
        Calculates the cosine similarity between two vectors.
        # Args:
        - p1 (np.ndarray): The first input vector.
        - p2 (np.ndarray): The second input vector.
        # Returns:
        - float: The cosine similarity value between `p1` and `p2`.
        # Raises:
        - ValueError: If either `p1` or `p2` is a zero vector, resulting in division by zero.
        """
        
        return np.sum(p1 * p2) / (np.sqrt(np.sum(p1**2)) * np.sqrt(np.sum(p2**2)))

    def get_p_b(self, ith):
        """
        # Introduction
        Selects and returns a personal best neighbor for the given particle index.
        # Args:
        - ith (int): The index of the particle for which to retrieve a personal best neighbor.
        # Returns:
        - Any: The personal best neighbor selected for the specified particle index.
        # Notes:
        - The neighbor is chosen randomly from the `pbest_neb` list using the random number generator `rng`.
        """
        
        r_idx = self.rng.randint(0, self.k)
        return self.pbest_neb[ith][r_idx]

    def get_p_a(self):
        """
        # Introduction
        Selects and returns a global best neighbor from the population using a random index.
        # Returns:
        - The selected global best neighbor from `self.gbest_neb` at a randomly chosen index.
        """
        
        r_idx = self.rng.randint(0, self.k)
        return self.gbest_neb[r_idx]
    

    def generate_v_vector(self, action, ith, w):
        """
        # Introduction
        Updates the velocity vector for a particle in the NR-LPSO (Non-Redundant Learning Particle Swarm Optimization) algorithm based on the selected action strategy.
        The method modifies the velocity of the `ith` particle in the swarm according to the specified action, which determines the exploration, exploitation, convergence, or jumping-out behavior. The update uses personal best, global best, and additional reference positions, with random coefficients and inertia weight.
        # Args:
        - action (int): The action type indicating the update strategy.  
            - 0: Exploration  
            - 1: Exploitation  
            - 2: Convergence  
            - 3: Jumping-out
        - ith (int): The index of the particle in the population whose velocity is to be updated.
        - w (float): The inertia weight applied to the previous velocity.
        # Built-in Attribute:
        - self.__population (np.ndarray): The current positions of all particles.
        - self.__velocity (np.ndarray): The current velocities of all particles.
        - self.__pbest_pos (np.ndarray): The personal best positions of all particles.
        - self.__gbest_pos (np.ndarray): The global best position found by the swarm.
        - self.rng (np.random.Generator): Random number generator for stochastic updates.
        - self.v_min (float or np.ndarray): Minimum allowed velocity.
        - self.v_max (float or np.ndarray): Maximum allowed velocity.
        - self.__dim (int): Dimensionality of the search space.
        # Returns:
        - None: The method updates the velocity of the specified particle in-place.
        # Raises:
        - IndexError: If `ith` is out of bounds for the population arrays.
        - AttributeError: If required attributes are not initialized.
        """
        
        c1, c2, P1, P2 = None, None, None, None
        r1 = self.rng.rand()
        r2 = self.rng.rand()

        cur_p = self.__population[ith]

        cs = self.cal_cs(self.__pbest_pos[ith], self.__gbest_pos)
        # exploration
        if action == 0:
            c1 = 2.2
            c2 = 1.8
            if cs < 0:
                P1 = self.__pbest_pos[ith]
                P2 = self.get_p_a()

                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
            else:
                P1 = self.get_p_b(ith)
                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p)
        # exploitation
        elif action == 1:
            c1 = 2.1
            c2 = 1.8
            if cs < 0:
                P1 = self.get_p_b(ith)
                P2 = self.__gbest_pos

                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
            else:
                P2 = self.get_p_a()

                self.__velocity[ith] = w * self.__velocity[ith] + c2 * r2 * (P2 - cur_p)
        # convergence
        elif action == 2:
            c1 = 2
            c2 = 2
            
            if cs < 0:
                P1 = self.__pbest_pos[ith]
                P2 = self.__gbest_pos

                self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
            else:
                P2 = self.__gbest_pos
                self.__velocity[ith] = w * self.__velocity[ith] + c2 * r2 * (P2 - cur_p)
        
        # jumping-out
        elif action == 3:
            c1 = 1.8
            c2 = 2.2
            P1 = self.get_p_b(ith)
            P2 = self.get_p_a()
            r1 = self.rng.rand(self.__dim)
            r2 = self.rng.rand(self.__dim)
            self.__velocity[ith] = w * self.__velocity[ith] + c1 * r1 * (P1 - cur_p) + c2 * r2 * (P2 - cur_p)
        
        # clip velocity
        self.__velocity[ith] = np.clip(self.__velocity[ith], self.v_min, self.v_max)

    def cal_cost(self, x, problem):
        """
        # Introduction
        Calculates the cost of a solution vector `x` for a given optimization problem. The cost is either the raw evaluation of `x` or the difference between the evaluation and the known optimum, if available. Also increments the function evaluation counter.
        # Args:
        - x (Any): The solution vector to be evaluated.
        - problem (object): The optimization problem instance, which must have `eval(x)` and `optimum` attributes.
        # Returns:
        - float: The computed cost value for the solution vector `x`.
        # Built-in Attribute:
        - self.fes (int): Increments the function evaluation count by 1.
        # Raises:
        - AttributeError: If `problem` does not have the required attributes (`eval` method or `optimum`).
        """
        
        self.fes += 1
        if problem.optimum is None:
            cost = problem.eval(x)
        else:
            cost = problem.eval(x) - problem.optimum
        return cost

    def neb_mutation(self, ith, problem):
        """
        # Introduction
        Performs neighborhood-based mutation on the particle at index `ith` within a population for a given optimization problem. This mutation is applied to both the personal best (pbest) and global best (gbest) neighborhoods, potentially updating the best positions and costs or replacing a neighbor in the population.
        # Args:
        - ith (int): Index of the particle in the population to apply the mutation.
        - problem (object): The optimization problem instance, used to evaluate the cost of candidate solutions.
        # Built-in Attribute:
        - self.__pbest_pos (np.ndarray): Personal best positions of all particles.
        - self.pbest_neb (np.ndarray): Neighborhood personal best positions for each particle.
        - self.pbest_neb_index (np.ndarray): Indices of neighborhood personal bests.
        - self.__pbest_cost (np.ndarray): Personal best costs of all particles.
        - self.__gbest_pos (np.ndarray): Global best position.
        - self.gbest_neb (np.ndarray): Neighborhood global best positions.
        - self.gbest_neb_index (np.ndarray): Indices of neighborhood global bests.
        - self.__gbest_cost (float): Global best cost.
        - self.__population (np.ndarray): Current population of particles.
        - self.__cost (np.ndarray): Current costs of all particles.
        - self.rng (np.random.Generator): Random number generator.
        - self.__dim (int): Dimensionality of the problem.
        # Returns:
        - None
        # Raises:
        - None
        """
        
        # for pbest ith neibourhood
        # find out P1 P2
        distance = np.sqrt(np.sum((self.__pbest_pos[ith][None, :] - self.pbest_neb[ith])**2, axis=-1))
        sort_idx = np.argsort(distance)
        P1 = self.pbest_neb[ith][sort_idx[0]]
        P2 = self.pbest_neb[ith][sort_idx[-1]]

        P3 = self.__pbest_pos[ith] + self.rng.rand(self.__dim) * (P1 - P2)
        cost = self.cal_cost(P3, problem)
        if cost < self.__pbest_cost[ith]:
            self.__pbest_pos[ith] = P3
            self.__pbest_cost[ith] = cost
        else:
            # replace P2 by P3
            P2_idx = self.pbest_neb_index[ith][sort_idx[-1]]
            self.__population[P2_idx] = P3
            self.__cost[P2_idx] = cost


        # for gbest neibourhood
        distance = np.sqrt(np.sum((self.__gbest_pos[None, :] - self.gbest_neb)**2, axis=-1))
        sort_idx = np.argsort(distance)
        P1 = self.gbest_neb[sort_idx[0]]
        P2 = self.gbest_neb[sort_idx[-1]]

        P3 = self.__gbest_pos + self.rng.rand(self.__dim) * (P1 - P2)
        cost = self.cal_cost(P3, problem)
        if cost < self.__gbest_cost:
            self.__gbest_pos = P3
            self.__gbest_cost = cost
        else:
            # replace P2 by P3
            P2_idx = self.gbest_neb_index[sort_idx[-1]]
            self.__population[P2_idx] = P3
            self.__cost[P2_idx] = cost

    def update(self, action, problem):
        """
        # Introduction
        Updates the state of the optimizer for a single agent/particle based on the provided action and the optimization problem. This method handles velocity and position updates, reward calculation, personal and global best tracking, neighborhood mutation, logging, and episode termination checks.
        # Args:
        - action (Any): The action to be applied for updating the agent's state (typically from a reinforcement learning policy or heuristic).
        - problem (object): The optimization problem instance, which should provide lower and upper bounds (`lb`, `ub`), and optionally an optimum value.
        # Returns:
        - state (Any): The updated state of the agent after applying the action.
        - reward (float): The reward calculated based on the improvement in cost and efficiency.
        - is_done (bool): Whether the optimization episode should be terminated.
        - info (dict): Additional information (currently empty, but can be extended).
        # Notes:
        - Updates the velocity and position of the current agent/particle.
        - Applies boundary constraints to the position.
        - Updates personal and global bests if improvements are found.
        - Applies neighborhood mutation if stagnation is detected.
        - Logs progress at specified intervals.
        - Handles episode termination based on function evaluations or optimum achievement.
        - Optionally stores meta-data if configured.
        """
        
        if self.pointer == 0:
            self.update_construct_neighborhood()
            # dynamic w
            self.w = self.cal_w()

        # generate velocity vector
        self.generate_v_vector(action, self.pointer, self.w)

        ef_old = self.cal_ef(self.pointer)
        # update position
        self.__population[self.pointer] = self.__population[self.pointer] + self.__velocity[self.pointer]
        self.__population[self.pointer] = np.clip(self.__population[self.pointer], problem.lb, problem.ub)

        ef_new = self.cal_ef(self.pointer)
        f_old = self.__cost[self.pointer]

        
        f_new = self.cal_cost(self.__population[self.pointer], problem)
        
        reward = self.cal_reward(f_new, f_old, ef_new, ef_old)
        
        self.__cost[self.pointer] = f_new
        # perform neighborhood diffenent mutation
        if f_new < self.__pbest_cost[self.pointer]:
            self.__pbest_pos[self.pointer] = self.__population[self.pointer]
            self.pbest_stag_count[self.pointer] = 0
        else:
            self.pbest_stag_count[self.pointer] += 1

        if self.pbest_stag_count[self.pointer] >= 2:
            self.neb_mutation(self.pointer, problem)

        if f_new < self.__gbest_cost:
            self.__gbest_cost = f_new
            self.__gbest_pos = self.__population[self.pointer]

        self.__state[self.pointer] = action

        if self.__config.full_meta_data:
            self.meta_tmp_x.append(self.__population[self.pointer].copy())
            self.meta_tmp_cost.append(self.__cost[self.pointer].copy())

            # 在某一轮迭代结束后（例如在 for j in range(NP) 之后）
            if len(self.meta_tmp_cost) == self.NP:  # 或 len(self.meta_tmp_x) == NP
                self.meta_X.append(np.array(self.meta_tmp_x))
                self.meta_Cost.append(np.array(self.meta_tmp_cost))

                self.meta_tmp_x.clear()
                self.meta_tmp_cost.clear()

        self.pointer = (self.pointer + 1) % self.NP

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
        
        return self.__state[self.pointer], reward, is_done , info

