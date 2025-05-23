import numpy as np

from .learnable_optimizer import Learnable_Optimizer

crossover_operators = ["CR1", "CR2", "CR3"]
mutation_operators = ['DE1', 'DE2', 'DE3', 'DE4', 'DE5', 'DE6', 'DE7', 'DE8', 'DE9', 'DE10', 'DE11', 'DE12', 'DE13', 'DE14']

class select_mutation:
    def __init__(self, rng):
        """
        # Introduction
        Initializes the optimizer with a given random number generator and sets up mutation operators.
        # Args:
        - rng: A random number generator instance used for stochastic operations.
        # Built-in Attributes:
        - `self.rng`: Stores the provided random number generator.
        - `self.mutation_operators` (list): List of mutation operator names to be used.
        - `self.operators` (dict): Dictionary mapping operator names to their instantiated objects.
        - `self.n_operator` (int): Number of mutation operators.
        # Notes:
        - The mutation_operators variable is expected to be defined elsewhere in the class or module.
        - Each operator is instantiated using its name and the provided random number generator.
        """
        
        # print(mutation_operators)
        operators = {}
        self.rng = rng
        for operator_name in mutation_operators:
            operators[operator_name] = eval(operator_name)(rng)
        self.mutation_operators = mutation_operators

        self.operators = operators
        self.n_operator = len(self.mutation_operators)

    def select_mutation_operator(self, mutation_operator):
        """
        # Introduction
        Selects and returns the mutation operator class based on the provided mutation operator key.
        # Args:
        - mutation_operator (str or int): The key or index used to identify the desired mutation operator.
        # Returns:
        - type: The class corresponding to the selected mutation operator.
        # Raises:
        - KeyError: If the provided mutation_operator does not exist in the mutation_operators mapping.
        """
        
        mutation_operator_name = self.mutation_operators[mutation_operator]
        operator_class = self.operators[mutation_operator_name]
        return operator_class

class select_crossover:
    def __init__(self, rng):
        """
        # Introduction
        Initializes the optimizer with a random number generator and sets up crossover operators.
        # Args:
        - rng: A random number generator instance used for stochastic operations.
        # Attributes:
        - `self.rng`: Stores the provided random number generator.
        - `self.crossover_operators`: A list or collection of crossover operator names.
        - `self.operators`: A dictionary mapping operator names to their instantiated objects.
        - `self.n_operator`: The number of crossover operators available.
        # Notes:
        Assumes that `crossover_operators` is defined in the scope and that each operator name corresponds to a callable class or function.
        """
        
        self.rng = rng
        # print(crossover_operators)
        operators = {}
        for operator_name in crossover_operators:
            operators[operator_name] = eval(operator_name)(rng)
        self.crossover_operators = crossover_operators

        self.operators = operators
        self.n_operator = len(self.crossover_operators)

    def select_crossover_operator(self, crossover_operator):
        """
        # Introduction
        Selects and returns the crossover operator class based on the provided operator key.
        # Args:
        - crossover_operator (str): The key or identifier for the desired crossover operator.
        # Returns:
        - type: The class corresponding to the selected crossover operator.
        # Raises:
        - KeyError: If the provided `crossover_operator` is not found in the available operators.
        """
        
        crossover_operator_name = self.crossover_operators[crossover_operator]
        operator_class = self.operators[crossover_operator_name]
        return operator_class

class RLDEAFL_Optimizer(Learnable_Optimizer):
    """
    # Introduction:
    RLDEAFL_Optimizer is a reinforcement learning-based Differential Evolution with Adaptive Feature Learning optimizer. It is designed to solve continuous optimization problems by adaptively selecting mutation and crossover operators using reinforcement learning strategies. The optimizer maintains a population of candidate solutions, applies evolutionary operators, and tracks the best solution found during the optimization process.
    # Paper:
    [Reinforcement Learning-based Self-adaptive Differential Evolution through Automated Landscape Feature Learning](https://arxiv.org/abs/2503.18061)
    # Implementation:
    [RLDEAFL](https://github.com/MetaEvo/RLDE-AFL)
    """
    def __init__(self, config):
        """
        # Introduction
        Initializes the RLDEAFLOptimizer with the provided configuration, setting up internal parameters for mutation, crossover, population size, and logging.
        # Args:
        - config (object): Configuration object containing optimizer settings such as `maxFEs` and `log_interval`.
            - The Attribute needed for the RLDEAFL_Optimizer:
                - maxFEs (int): Maximum number of function evaluations.
                - log_interval (int): Interval for logging progress.
                - n_logpoint (int): Number of log points to record.
                - full_meta_data (bool): Flag indicating whether to use full meta data.
        # Built-in Attribute:
        - self.__config: Stores the configuration object.
        - self.__mu_operator (int): Number of mutation operators. Default is 14.
        - self.__cr_operator (int): Number of crossover operators. Default is 3.
        - self.__n_mutation (int): Number of mutation strategies. Default is 3.
        - self.__n_crossover (int): Number of crossover strategies. Default is 2.
        - self.__NP (int): Population size. Default is 100.
        - self.__reward_ratio (int): Reward ratio for operator selection. Default is 1.
        - self.__mu_selector: Placeholder for mutation operator selector.Default is None.
        - self.__cr_selector: Placeholder for crossover operator selector.Default is None.
        - self.log_index: Index for logging.Default is None.
        # Returns:
        - None
        """
        
        super().__init__(config)
        self.__config = config

        self.__mu_operator = 14
        self.__cr_operator = 3

        self.__n_mutation = 3
        self.__n_crossover = 2

        self.__NP = 100
        self.max_fes = config.maxFEs
        self.__reward_ratio = 1

        self.__mu_selector = None
        self.__cr_selector = None


        self.log_index = None
        self.log_interval = config.log_interval

    def __str__(self):
        """
        Returns a string representation of the RLDEAFL_Optimizer class.
        # Returns:
            str: The name of the optimizer, "RLDEAFL_Optimizer".
        """
        
        return "RLDEAFL_Optimizer"

    # calculate costs of solutions
    def get_costs(self, position, problem):
        """
        # Introduction
        Evaluates the cost(s) of a given position or set of positions for a specified optimization problem, applying boundary constraints and adjusting for the problem's optimum if available.
        # Args:
        - position (np.ndarray): The position(s) to be evaluated, typically in normalized [0, 1] space.
        - problem (object): The optimization problem instance, expected to have `lb`, `ub`, `eval`, and `optimum` attributes.
        # Built-in Attribute:
        - self.fes (int): Increments the function evaluation counter by the number of positions evaluated.
        # Returns:
        - cost (float or np.ndarray): The evaluated cost(s) for the given position(s), optionally shifted by the problem's optimum.
        # Raises:
        - AttributeError: If the `problem` object does not have the required attributes (`lb`, `ub`, `eval`, `optimum`).
        """
        
        ps = position.shape[0]
        self.fes += ps
        # return problem bound
        position = np.clip(position, 0, 1)
        position = (problem.ub - problem.lb) * position + problem.lb

        if problem.optimum is None:
            cost = problem.eval(position)
        else:
            cost = problem.eval(position) - problem.optimum
        return cost

    def observe(self):
        """
        # Introduction
        Observes the current state of the optimizer by normalizing the current vector, recording the current fitness, and tracking the progress as a fraction of function evaluations.
        # Returns:
        - np.ndarray: A 2D array where each row contains the normalized current vector, its corresponding fitness value, and the normalized number of function evaluations.
        # Notes:
        - The normalization of the current vector assumes the lower and upper bounds are 0 and 1, respectively.
        - The function evaluation step (`fes`) is normalized by the maximum allowed function evaluations (`max_fes`).
        """
        
        xs = (self.current_vector - 0) / (1 - 0)
        fes = self.fes / self.max_fes
        pop = np.column_stack((xs, self.current_fitness, np.full(xs.shape[0], fes)))
        return pop

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the RLDEAF optimizer, setting up the initial candidate solutions, their fitness values, and relevant optimizer state variables.
        # Args:
        - problem (object): An optimization problem instance that must have `dim`, `ub`, and `lb` attributes, representing the dimensionality, upper bounds, and lower bounds of the search space, respectively.
        # Built-in Attribute:
        - self.__dim (int): Dimensionality of the problem.
        - self.__mu_selector, self.__cr_selector: Mutation and crossover selector functions, initialized if not already set.
        - self.fes (int): Function evaluation counter, reset to zero.
        - self.__archive (np.ndarray): Archive of solutions, initialized as empty.
        - self.current_vector (np.ndarray): Current population of candidate solutions.
        - self.current_fitness (np.ndarray): Fitness values of the current population.
        - self.gbest_val (float): Best fitness value found in the current population.
        - self.__gbest_index (int): Index of the best individual in the current population.
        - self.__gbest_vector (np.ndarray): Best solution vector found in the current population.
        - self.log_index (int): Logging index, initialized to 1.
        - self.cost (list): List of best fitness values per generation.
        - self.__init_gbest (float): Initial best fitness value.
        - self.meta_X (list, optional): List of population vectors for meta-data logging (if enabled).
        - self.meta_Cost (list, optional): List of population fitness values for meta-data logging (if enabled).
        # Returns:
        - object: The result of `self.observe()`, typically an observation or summary of the initialized population state.
        # Raises:
        - None explicitly, but may raise exceptions if `problem` is not properly defined or if array operations fail.
        """
        
        self.__dim = problem.dim
        if self.__mu_selector is None:
            self.__mu_selector = select_mutation(self.rng)
            self.__cr_selector = select_crossover(self.rng)

        self.fes = 0
        NP = self.__NP
        dim = self.__dim
        rand_vector = self.rng.uniform(low = 0, high = 1, size = (NP, dim))
        c_cost = self.get_costs(rand_vector, problem)

        self.__archive = np.array([])
        self.current_vector = rand_vector
        self.current_fitness = c_cost

        self.gbest_val = np.min(self.current_fitness)
        self.__gbest_index = np.argmin(self.current_fitness)
        self.__gbest_vector = self.current_vector[self.__gbest_index]

        self.log_index = 1
        self.cost = [self.gbest_val]
        self.__init_gbest = self.gbest_val

        if self.__config.full_meta_data:
            self.meta_X = [self.current_vector.copy() * (problem.ub - problem.lb) + problem.lb]
            self.meta_Cost = [self.current_fitness.copy()]

        return self.observe()

    def __update_archive(self, old_id):
        """
        # Introduction
        Updates the archive of solution vectors by either appending a new vector or replacing an existing one, depending on the archive's current size.
        # Args:
        - old_id (int): The index of the vector in `current_vector` to be added to the archive.
        # Built-in Attribute:
        - `self.__archive` (np.ndarray): The archive of solution vectors.
        - `self.__NP` (int): The maximum allowed size of the archive.
        - `self.current_vector` (np.ndarray): The current population of solution vectors.
        - `self.__dim` (int): The dimensionality of each solution vector.
        - `self.rng` (np.random.Generator): Random number generator for selecting replacement indices.
        # Returns:
        - None
        # Raises:
        - None
        """
        if self.__archive.shape[0] < self.__NP:
            self.__archive = np.append(self.__archive, self.current_vector[old_id]).reshape(-1, self.__dim)
        else:
            self.__archive[self.rng.randint(self.__archive.shape[0])] = self.current_vector[old_id]

    def update(self, action, problem):
        """
        # Introduction
        Updates the optimizer's population based on the provided actions, applies mutation and crossover operators, evaluates new solutions, updates the archive, and tracks the best solution found so far.
        # Args:
        - action (np.ndarray): An array representing the actions to be taken, including mutation and crossover operator indices and their parameters for each individual in the population.
        - problem (object): The optimization problem instance, which should provide methods for evaluating solutions and contain problem-specific attributes such as bounds and optimum.
        # Returns:
        - observation (np.ndarray): The current observation/state after the update.
        - reward (float): The reward computed based on the improvement in the global best value.
        - is_done (bool): A flag indicating whether the optimization process has reached its termination condition.
        - info (dict): An empty dictionary reserved for additional information (for compatibility).
        # Raises:
        - None explicitly, but may raise exceptions if the action array is malformed or if operator selection fails.
        """
        
        _, n_action = action.shape
        mutation_operator = action[:, 0]
        crossover_operator = action[:, 1]
        mutation_parameters = action[:, 2: 2 + self.__n_mutation]
        crossover_parameters = action[:, -self.__n_crossover:]

        pre_gbest = self.gbest_val

        # classification
        mu_operators_dict = {}
        for i in range(self.__mu_operator):
            indexs = np.where(mutation_operator == i)[0]
            mu_operators_dict[i] = indexs

        cr_operators_dict = {}
        for i in range(self.__cr_operator):
            indexs = np.where(crossover_operator == i)[0]
            cr_operators_dict[i] = indexs

        # apply mutation
        origin_vector = self.current_vector
        origin_fitness = self.current_fitness
        v = np.zeros_like(origin_vector)
        for de in mu_operators_dict:
            indexs = mu_operators_dict[de]
            if indexs.shape[0] == 0:
                continue
            # parametrers = mutation_parameters[indexs]
            operator = self.__mu_selector.select_mutation_operator(int(de))
            updated_sub_vector = operator.mutation(origin_vector, origin_fitness, indexs, mutation_parameters, self.__archive)
            v[indexs] = updated_sub_vector

        # bound
        v = np.where(v < 0, (origin_vector + 0) / 2, v)
        v = np.where(v > 1, (origin_vector + 1) / 2, v)

        # apply crossover
        u = np.zeros_like(v)
        for cr in cr_operators_dict:
            indexs = cr_operators_dict[cr]
            if indexs.shape[0] == 0:
                continue
            parametrers = crossover_parameters[indexs]
            operator = self.__cr_selector.select_crossover_operator(int(cr))
            updated_sub_vector = operator.crossover(origin_vector[indexs], v[indexs],
                                                    parametrers, origin_vector, origin_fitness, self.__archive)
            u[indexs] = updated_sub_vector

        # cost
        new_cost = self.get_costs(u, problem)
        optim = np.where(new_cost < self.current_fitness)[0]
        for i in optim:
            self.__update_archive(i)

        self.current_vector[optim] = u[optim]
        self.current_fitness = np.minimum(self.current_fitness, new_cost)

        self.gbest_val = np.min(self.current_fitness)
        self.__gbest_index = np.argmin(self.current_fitness)
        self.__gbest_vector = self.current_vector[self.__gbest_index]


        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest_val)

        if self.__config.full_meta_data:
            self.meta_X.append(self.current_vector.copy() * (problem.ub - problem.lb) + problem.lb)
            self.meta_Cost.append(self.current_fitness.copy())

        if problem.optimum is None:
            is_done = self.fes >= self.max_fes
        else:
            is_done = self.fes >= self.max_fes

        reward = self.__reward_ratio * (pre_gbest - self.gbest_val) / self.__init_gbest

        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.gbest_val
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.gbest_val)
        return self.observe(), reward, is_done, {}


# [best/1] [best/2] [rand/1] [rand/2] [current-to-best/1] [rand-to-best/1] [current-to-rand/1] [current-to-pbest/1] [ProDE-rand/1]
# [TopoDE-rand/1] [current-to-pbest/1+archive] [HARDDE-current-to-pbest/2] [current-to-rand/1+archive] [weighted-rand-to-pbest/1]
class Basic_mutation:
    """
    This class represents a basic mutation.
    Methods:
    - get_parameters_numbers: Returns the number of parameters.
    - mutation: Performs the mutation.
    """

    # individual version
    # def mutation(self,env,individual_indice):
    #     """
    #     Perform mutation on the given individual.
    #     Parameters:
    #     - env: The environment object.
    #     - individual_indice: The index of the individual to mutate.
    #     Returns:
    #     - None
    #     """

    #     pass

    # population version
    def __init__(self, rng):
        self.rng = rng
    def mutation(self, group, cost, indexs, parameters, archive):
        pass

    def construct_random_indices(self, pop_size, indexs, x_num):
        indices = np.arange(pop_size)
        if x_num == 1:
            Indices = np.zeros(len(indexs), dtype = int)
        else:
            Indices = np.zeros((len(indexs), x_num), dtype = int)
        for i, index in enumerate(indexs):
            temp_indices = indices[indices != index]
            Indices[i] = self.rng.choice(temp_indices, x_num, replace = False)
        return Indices

    def construct_extra_random_indices(self, pop_size, indexs, x_num, extra):
        indices = np.arange(pop_size)
        if x_num == 1:
            Indices = np.zeros(len(indexs), dtype = int)
        else:
            Indices = np.zeros((len(indexs), x_num), dtype = int)
        extra_n = extra.shape[1]
        for i, index in enumerate(indexs):
            filters = indices != index
            for j in range(extra_n):
                filters = filters & (indices != extra[i, j])
            temp_indices = indices[filters]
            Indices[i] = self.rng.choice(temp_indices, x_num, replace = False)
        return Indices

    def construct_pbest(self, group, p):
        p = np.mean(p)
        pbest = group[:max(int(p * group.shape[0]), 2)]
        return pbest

# [binomial] [exponential] [p-binomial]
class Basic_crossover:
    def __init__(self, rng):
        self.rng = rng
    def crossover(self, x, v, parameters, group, cost, archive):

        pass

class DE1(Basic_mutation):

    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 2)
        x1, x2 = group[random_indices.T]
        mutated_vector = best_vector + F * (x1 - x2)
        return mutated_vector

class DE2(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 4)
        x1, x2, x3, x4 = group[random_indices.T]
        mutated_vector = best_vector + F * (x1 - x2) + F * (x3 - x4)
        return mutated_vector

class DE3(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 3)
        x1, x2, x3 = group[random_indices.T]
        mutated_vector = x1 + F * (x2 - x3)
        return mutated_vector

class DE4(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 5)
        x1, x2, x3, x4, x5 = group[random_indices.T]
        mutated_vector = x1 + F * (x2 - x3) + F * (x4 - x5)
        return mutated_vector

class DE5(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        current_vector = group[indexs]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 2)
        x1, x2 = group[random_indices.T]
        mutated_vector = current_vector + F * (best_vector - current_vector) + F * (x1 - x2)
        return mutated_vector

class DE6(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        best_index = np.argmin(cost)
        best_vector = group[best_index]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 4)
        x1, x2, x3, x4 = group[random_indices.T]
        mutated_vector = x1 + F * (best_vector - x2) + F * (x3 - x4)
        return mutated_vector

class DE7(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        current_vector = group[indexs]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 3)
        x1, x2, x3 = group[random_indices.T]
        mutated_vector = current_vector + F * (x1 - current_vector) + F * (x2 - x3)
        return mutated_vector

class DE8(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    # current-to-pbest/1
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]
        current_vector = temp_group[temp_indexs]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]
        rb = self.rng.randint(NB, size = len(indexs))
        random_indices = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 2, extra = rb[:, None])
        x1, x2 = temp_group[random_indices.T]
        mutated_vector = current_vector + F * (pbest[rb] - current_vector) + F * (x1 - x2)
        return mutated_vector

class DE9(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def EuclideanDistance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def cal_R_d(self, group):
        pop_size = group.shape[0]
        R_d = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                R_d[i, j] = self.EuclideanDistance(group[i], group[j])
                R_d[j, i] = R_d[i, j]
        return R_d

    def construct_r(self, group, indexs):
        R_d = self.cal_R_d(group)
        Sum = np.sum(R_d, axis = 0) # 1D NP
        Sum = np.where(Sum == 0, 1, Sum)
        R_p = 1 - (R_d / Sum) # NP * NP

        p = np.sum(R_p, axis = 1)
        Indices = np.zeros((len(indexs), 3), dtype = int)
        for i, index in enumerate(indexs):
            temp_p = p
            temp_p[index] = 0
            if np.sum(temp_p) == 0:
                temp_p = np.ones_like(temp_p)
                temp_p[index] = 0
            temp_p = temp_p / np.sum(temp_p)
            Indices[i] = self.rng.choice(len(temp_p), size = 3, p = temp_p, replace = False)
        return Indices

    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_r(group, indexs)
        x1, x2, x3 = group[random_indices.T]
        matuated_vector = x1 + F * (x2 - x3)
        return matuated_vector

class DE10(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def Topograph(self, group, cost):
        """
        generate a kNN matrix indicating the nearest neighbors of each individual
        """
        current_vector = group
        pop_size, dim = current_vector.shape
        # generate N*N distance matrix
        distance_matrix = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                distance_matrix[i, j] = np.linalg.norm(current_vector[i] - current_vector[j])
                distance_matrix[j, i] = distance_matrix[i, j]
            distance_matrix[i, i] = np.inf
        # generate kNN matrix
        k = pop_size // 10  # number of nearest neighbors
        kNN_matrix = np.zeros((pop_size, k))
        for i in range(pop_size):
            kNN_matrix[i] = np.argsort(distance_matrix[i])[:k]
        for i in range(pop_size):
            for j in range(k):
                if cost[i] < cost[int(kNN_matrix[i, j])]:
                    kNN_matrix[i, j] = kNN_matrix[i, j]
                else:
                    kNN_matrix[i, j] = -kNN_matrix[i, j]
        return kNN_matrix

    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        random_indices = self.construct_random_indices(group.shape[0], indexs, 2)
        x2, x3 = group[random_indices.T]
        knn_matrix = self.Topograph(group, cost)
        flag = np.zeros(knn_matrix.shape[0], dtype = bool)
        negative_indices = knn_matrix < 0
        purpose = np.arange(knn_matrix.shape[0])

        for i in range(group.shape[0]):
            if np.any(negative_indices[i]):
                flag[i] = True
                valid_indices = np.where(negative_indices[i])[0]
                fitness_values = cost[knn_matrix[i, valid_indices].astype(int)]
                purpose[i] = valid_indices[np.argmin(fitness_values)]

        purpose[~flag] = np.arange(knn_matrix.shape[0])[~flag]

        topu = group[purpose]
        x1 = topu[indexs]

        mutated_vector = x1 + F * (x2 - x3)
        return mutated_vector

class DE11(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    # current-to-pbest/1+archive
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]
        current_vector = temp_group[temp_indexs]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]
        NA = archive.shape[0]
        rb = self.rng.randint(NB, size = len(temp_indexs))
        r1 = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 1, extra = rb[:, None])
        r2 = self.construct_extra_random_indices(temp_group.shape[0] + NA, temp_indexs, 1, extra = np.concatenate((rb[:, None], r1[:,None]), 1))

        xb = pbest[rb]
        x1 = temp_group[r1]
        if NA > 0:
            x2 = np.concatenate((temp_group, archive), 0)[r2]
        else:
            x2 = temp_group[r2]
        mutated_vector = current_vector + F * (xb - current_vector) + F * (x1 - x2)
        return mutated_vector

class DE12(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]

        Fa = temp_parameters[:, 2]
        Fa = Fa[:, np.newaxis]
        current_vector = temp_group[temp_indexs]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]
        NA = archive.shape[0]
        rb = self.rng.randint(NB, size = len(temp_indexs))
        r1 = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 1, extra = rb[:, None])
        r2 = self.construct_extra_random_indices(temp_group.shape[0] + NA, temp_indexs, 1, extra = np.concatenate((rb[:, None], r1[:,None]), 1))
        r3 = self.construct_extra_random_indices(temp_group.shape[0] + NA, temp_indexs, 1, extra = np.concatenate((rb[:, None], r1[:,None], r2[:, None]), 1))
        xb = pbest[rb]
        x1 = temp_group[r1]
        if NA > 0:
            x2 = np.concatenate((temp_group, archive), 0)[r2]
            x3 = np.concatenate((temp_group, archive), 0)[r3]
        else:
            x2 = temp_group[r2]
            x3 = temp_group[r3]
        mutated_vector = current_vector + F * (xb - current_vector) + Fa * (x1 - x2) + Fa * (x1 - x3)
        return mutated_vector

class DE13(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        F = parameters[indexs][:, 0]
        F = F[:, np.newaxis]
        NA = archive.shape[0]
        current_vector = group[indexs]
        r1 = self.construct_random_indices(group.shape[0], indexs, 1)
        r2 = self.construct_extra_random_indices(group.shape[0] + NA, indexs, 1, extra = r1[:, None])

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]

        mutated_vector = current_vector + F * (x1 - x2)
        return mutated_vector

class DE14(Basic_mutation):
    def __init__(self, rng):
        super().__init__(rng)
    def mutation(self, group, cost, indexs, parameters, archive):
        # Sort
        ind = np.argsort(cost)
        temp_group = group[ind]

        ind_2 = np.argsort(ind)
        temp_indexs = ind_2[indexs]
        temp_parameters = parameters[temp_indexs]
        F = temp_parameters[:, 0]
        F = F[:, np.newaxis]

        pbest = self.construct_pbest(temp_group, temp_parameters[:, 1])
        NB = pbest.shape[0]

        Fa = temp_parameters[:, 2]
        Fa = Fa[:, np.newaxis]
        rb = self.rng.randint(NB, size = len(temp_indexs))
        random_indices = self.construct_extra_random_indices(temp_group.shape[0], temp_indexs, 2, extra = rb[:, None])

        xb = pbest[rb]
        x1, x2 = temp_group[random_indices.T]
        mutated_vector = F * x1 + F * Fa * (xb - x2)
        return mutated_vector

class CR1(Basic_crossover):
    def __init__(self, rng):
        super().__init__(rng)
    def crossover(self, x, v, parameters, group, cost, archive):
        CR = parameters[:, 0]
        NP, dim = x.shape
        jrand = self.rng.randint(dim, size = NP)
        CRs = np.repeat(CR, dim).reshape(NP, dim)
        u = np.where(self.rng.rand(NP, dim) < CRs, v, x)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u

class CR2(Basic_crossover):
    def __init__(self, rng):
        super().__init__(rng)
    def crossover(self, x, v, parameters, group, cost, archive):
        CR = parameters[:, 0]
        NP, dim = x.shape
        u = x.copy()
        L = self.rng.randint(dim, size = NP).repeat(dim).reshape(NP, dim)
        L = L <= np.arange(dim)
        rvs = self.rng.rand(NP, dim)
        CRs = np.repeat(CR, dim).reshape(NP, dim)
        L = np.where(rvs > CRs, L, 0)
        u = u * (1 - L) + v * L
        return u

class CR3(Basic_crossover):
    def __init__(self, rng):
        super().__init__(rng)
    def crossover(self, x, v, parameters, group, cost, archive):
        CR = parameters[:, 0]
        p = parameters[:, 1]

        p = np.mean(p)
        ind = np.argsort(cost)
        temp_group = group[ind]
        pbest = temp_group[:max(int(p * group.shape[0]), 2)]
        if archive.shape[0] > 0:
            pbest = np.concatenate((temp_group, archive), 0)[:max(int(p * (group.shape[0] + archive.shape[0])), 2)]

        NP, dim = x.shape
        cross_pbest = pbest[self.rng.randint(pbest.shape[0], size = NP)]
        jrand = self.rng.randint(dim, size = NP)
        CRs = np.repeat(CR, dim).reshape(NP, dim)
        u = np.where(self.rng.rand(NP, dim) < CRs, v, cross_pbest)
        u[np.arange(NP), jrand] = v[np.arange(NP), jrand]
        return u
