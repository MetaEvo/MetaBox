import numpy as np
import torch
from .learnable_optimizer import Learnable_Optimizer


class LDE_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    LDE:Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient
    # Original paper
    "[**Learning Adaptive Differential Evolution Algorithm from Optimization Experiences by Policy Gradient**](https://ieeexplore.ieee.org/abstract/document/9359652)." IEEE Transactions on Evolutionary Computation (2021).
    # Official Implementation
    [LDE](https://github.com/yierh/LDE)
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the provided configuration and sets default values for several optimization parameters.
        # Args:
        - config (object): Configuration object containing optimizer settings.
        # Built-in Attribute:
        - self.__config: Stores the configuration object.
        - self.__BATCH_SIZE: Batch size for optimization (default is 1).
        - self.fes: Function evaluation counter (initialized as None).
        - self.cost: Cost value (initialized as None).
        - self.log_index: Logging index (initialized as None).
        - self.log_interval: Interval for logging, taken from the configuration.
        # Returns:
        - None
        # Raises:
        - None
        """
        
        super().__init__(config)
        self.__config = config
        self.__config.NP = 50
        self.__config.BINS = 5
        self.__config.P_INI = 1
        self.__config.P_NUM_MIN = 2
        self.__config.P_MIN = self.__config.P_NUM_MIN/self.__config.NP
        self.__BATCH_SIZE = 1
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

    def __get_cost(self, batch, pop):
        """
        # Introduction
        Computes the cost for each item in a batch given a corresponding population, optionally normalizing by the optimum value if available.
        # Args:
        - batch (list): A list of objects, each with an `optimum` attribute and an `eval` method that evaluates a population member.
        - pop (list): A list of population members, one for each item in the batch.
        # Returns:
        - numpy.ndarray: A 2D array where each row corresponds to the cost for a batch item.
        # Notes:
        - If `batch[p].optimum` is `None`, the raw evaluation is used as the cost.
        - If `batch[p].optimum` is not `None`, the cost is normalized by subtracting the optimum value.
        """
        
        bs = len(batch)
        cost = []
        for p in range(bs):
            if batch[p].optimum is None:
                cost.append(batch[p].eval(pop[p]))  # [NP]
            else:
                cost.append(batch[p].eval(pop[p]) - batch[p].optimum)  # [NP]
        return np.vstack(cost)

    def __modifyChildwithParent(self, cross_pop, parent_pop, x_max, x_min):
        """
        # Introduction
        Modifies the offspring population (`cross_pop`) based on boundary constraints and the parent population (`parent_pop`). 
        If an offspring value is out of bounds, it is adjusted using the parent value and the respective bound.
        # Args:
        - cross_pop (np.ndarray): The offspring population array to be modified.
        - parent_pop (np.ndarray): The parent population array used for boundary correction.
        - x_max (float or np.ndarray): The upper bound(s) for the population values.
        - x_min (float or np.ndarray): The lower bound(s) for the population values.
        # Returns:
        - np.ndarray: The modified offspring population with boundary violations corrected.
        """
        
        cro_lb = cross_pop < x_min
        cro_ub = cross_pop > x_max
        no_cro = ~(cro_lb | cro_ub)

        cross_pop = no_cro * cross_pop + cro_lb * (parent_pop + x_min) / 2. + cro_ub * (parent_pop + x_max) / 2.

        return cross_pop

    def __de_crosselect_random_dataset(self, pop, m_pop, fit, cr_vector, nfes, batch):
        """
        # Introduction
        Performs the crossover and selection operations for a Differential Evolution (DE) optimizer on a batch of populations, using random datasets for crossover. This function generates offspring via crossover, applies boundary control, evaluates offspring fitness, and selects the next generation based on fitness.
        # Args:
        - pop (np.ndarray): The current population array of shape (batch_size, pop_size, problem_size).
        - m_pop (np.ndarray): The mutated population array of the same shape as `pop`.
        - fit (np.ndarray): The fitness values of the current population, shape (batch_size, pop_size).
        - cr_vector (np.ndarray): The crossover rate vector, shape (batch_size, pop_size).
        - nfes (int): The current number of function evaluations.
        - batch (object): An object containing problem-specific data, including upper and lower bounds (`ub`, `lb`).
        # Returns:
        - n_pop (np.ndarray): The next generation population after selection, shape (batch_size, pop_size, problem_size).
        - n_fit (np.ndarray): The fitness values of the next generation, shape (batch_size, pop_size).
        - nfes (int): The updated number of function evaluations after evaluating offspring.
        """
        
        batch_size, pop_size, problem_size = pop.shape
        
        # Crossover
        r = self.rng.uniform(size=(batch_size, pop_size, problem_size))
        r[np.arange(batch_size)[:, None].repeat(pop_size, axis=1),
          np.arange(pop_size)[None, :].repeat(batch_size, axis=0),
          self.rng.randint(low=0, high=problem_size, size=[batch_size, self.__config.NP])] = 0.
        cross_pop = np.where(r <= cr_vector[:, :, None].repeat(problem_size, axis=-1), m_pop, pop)

        # Boundary Control
        cross_pop = self.__modifyChildwithParent(cross_pop, pop, batch.ub, batch.lb)

        # Get costs
        cross_fit = self.__get_cost([batch], cross_pop)

        nfes += pop_size

        # Selection
        surv_filters = cross_fit <= fit  # survive_filter is true if the offspring is better than or equal to its parent
        n_pop = np.where(surv_filters[:, :, None].repeat(problem_size, axis=-1), cross_pop, pop)
        n_fit = np.where(surv_filters, cross_fit, fit)

        return n_pop, n_fit, nfes

    def __mulgenerate_pop(self, p, NP, input_dimension, x_min, x_max, same_per_problem):
        """
        # Introduction
        Generates a population of candidate solutions for an optimization algorithm, with options for generating the same or different populations per problem.
        # Args:
        - p (int): Number of problems or populations to generate.
        - NP (int): Number of individuals in each population.
        - input_dimension (int): Dimensionality of each individual.
        - x_min (float or np.ndarray): Lower bound(s) for initialization.
        - x_max (float or np.ndarray): Upper bound(s) for initialization.
        - same_per_problem (bool): If True, generates the same population for all problems; otherwise, generates different populations.
        # Returns:
        - np.ndarray: Generated population(s) with shape (p, NP, input_dimension).
        """

        if same_per_problem:
            pop = x_min + self.rng.uniform(size=(NP, input_dimension)) * (x_max - x_min)
            pop = pop[None, :, :].repeat(p, axis=0)
        else:
            pop = x_min + self.rng.uniform(size=(p, NP, input_dimension)) * (x_max - x_min)
        return pop

    def __order_by_f(self, pop, fit):
        """
        # Introduction
        Sorts a population and its corresponding fitness values in ascending order of fitness for each batch.
        # Args:
        - pop (np.ndarray): The population array of shape (batch_size, pop_size, ...).
        - fit (np.ndarray): The fitness array of shape (batch_size, pop_size).
        # Returns:
        - temp_pop (p, NP, input_dimension): The population array sorted by fitness for each batch.
        - temp_fit (p, NP, input_dimension): The fitness array sorted in ascending order for each batch.
        """
        
        batch_size, pop_size = pop.shape[0], pop.shape[1]
        sorted_array = np.argsort(fit, axis=1)
        temp_pop = pop[np.arange(batch_size)[:, None].repeat(pop_size, axis=1), sorted_array]
        temp_fit = fit[np.arange(batch_size)[:, None].repeat(pop_size, axis=1), sorted_array]
        return temp_pop, temp_fit

    def __maxmin_norm(self, a):
        """
        # Introduction
        Applies max-min normalization to each batch in the input array, scaling values to the [0, 1] range per batch.
        # Args:
        - a (np.ndarray): A 2D NumPy array of shape (batch_size, n_features), where each row represents a batch to be normalized.
        # Returns:
        - np.ndarray: A NumPy array of the same shape as `a`, with each batch normalized using max-min scaling.
        # Notes:
        - If all values in a batch are equal, the batch is left as zeros (no normalization applied).
        """
        
        batch_size = a.shape[0]
        normed = np.zeros_like(a)
        for b in range(batch_size):
            if np.max(a[b]) != np.min(a[b]):
                normed[b] = (a[b] - np.min(a[b])) / (np.max(a[b]) - np.min(a[b]))
        return normed

    def __con2mat_current2pbest_Nw(self, mutation_vector, p):
        """
        # Introduction
        Constructs a mutation matrix for the "current-to-pbest" strategy in a differential evolution optimizer, 
        supporting batch operations and stochastic selection of p-best individuals.
        # Args:
        - mutation_vector (np.ndarray): A 2D array of shape (batch_size, pop_size) containing mutation coefficients for each individual in the population.
        - p (float): The proportion (0 < p <= 1) of top individuals to consider as p-best for mutation.
        # Returns:
        - np.ndarray: A 3D array of shape (batch_size, pop_size, pop_size) representing the mutation matrix for each batch and individual.
        # Notes:
        - The method uses a random number generator (`self.rng`) to select p-best indices.
        - For each individual, the diagonal of the mutation matrix is set based on the mutation vector, and the selected p-best index is updated accordingly.
        """
        
        batch_size, pop_size = mutation_vector.shape[0], mutation_vector.shape[1]
        p_index_array = self.rng.randint(0, int(np.ceil(pop_size*p)), size=(batch_size, pop_size))
        mutation_mat = np.zeros((batch_size, pop_size, pop_size))
        for i in range(pop_size):
            mutation_mat[:, i, i] = 1 - mutation_vector[:, i]
            for b in range(batch_size):
                if p_index_array[b, i] != i:
                    mutation_mat[b, i, p_index_array[b, i]] = mutation_vector[b, i]
                else:
                    mutation_mat[b, i, i] = 1
        return mutation_mat

    def __con2mat_rand2pbest_Nw(self, mutation_vector, nfes, MaxFEs):
        """
        # Introduction
        Generates a mutation matrix using the "rand-to-pbest" strategy with a dynamically adjusted p-rate based on the current number of function evaluations.
        # Args:
        - mutation_vector (np.ndarray): The mutation vector to be transformed into a mutation matrix.
        - nfes (int): The current number of function evaluations.
        - MaxFEs (int): The maximum number of function evaluations allowed.
        # Returns:
        - np.ndarray: A 3D array of shape (batch_size, pop_size, pop_size) representing the mutation matrix for each batch and individual.
        # Notes:
        The p-rate is linearly interpolated between `P_INI` and `P_MIN` from the configuration as the optimization progresses.
        """
        
        #        ( 0.4  -   1  ) * nfes/MAXFE + 1
        p_rate = (self.__config.P_MIN - self.__config.P_INI) * nfes/MaxFEs + self.__config.P_INI
        mutation_mat = self.__con2mat_current2pbest_Nw(mutation_vector, max(0, p_rate))
        return mutation_mat

    def __add_random(self, m_pop, pop, mu):
        """
        # Introduction
        Generates a mutated population by adding scaled differences between randomly selected individuals from the current population, ensuring that indices are unique and do not repeat within each selection.
        # Args:
        - m_pop (np.ndarray): The mean population array, typically representing the current mean of the population.
        - pop (np.ndarray): The current population array of individuals.
        - mu (np.ndarray): The mutation factor(s) to scale the difference between individuals.
        # Returns:
        - np.ndarray: The mutated population A 2D array of shape (NP, dim) array after applying the random differential mutation.
        # Notes:
        - The function ensures that for each selection, the indices used are unique and do not overlap with the current index.
        """
        
        batch_size = pop.shape[0]
        r = torch.randint(high=self.__config.NP, size=[batch_size, self.__config.NP, 2])

        # validity checking and modification for r
        pop_index = torch.arange(self.__config.NP)
        for col in range(0, 2):
            while True:
                is_repeated = [torch.eq(r[:, :, col], r[:, :, i]) for i in range(col)]
                is_repeated.append(torch.eq(r[:, :, col], pop_index))
                repeated_index = torch.nonzero(torch.any(torch.stack(is_repeated), dim=0))
                repeated_sum = repeated_index.size(0)
                if repeated_sum != 0:
                    r[repeated_index[:, 0], repeated_index[:, 1], col] = torch.randint(high=self.__config.NP,
                                                                                    size=[repeated_sum])
                else:
                    break
        r = r.numpy()

        batch_index = np.arange(batch_size)[:, None].repeat(self.__config.NP, axis=1)

        mur_pop = m_pop + np.expand_dims(mu, -1).repeat(pop.shape[-1], axis=-1) * (pop[batch_index, r[:, :, 0]] - pop[batch_index, r[:, :, 1]])

        return mur_pop

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the LDE_Optimizer object.
        # Returns:
        - str: The name of the optimizer, "LDE_Optimizer".
        """
        
        return "LDE_Optimizer"

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimizer, evaluates their fitness, and sets up internal tracking variables.
        # Args:
        - problem (object): The optimization problem object, which has attributes `lb` (lower bounds) and `ub` (upper bounds).
        # Returns:
        - np.ndarray: Feature representation of the initialized population, as returned by `self.__get_feature()`.
        # Notes:
        - Initializes the population using the problem's bounds and configuration parameters.
        - Evaluates the initial fitness of the population.
        - Sets up tracking for best cost, function evaluations, logging, and historical data.
        - Optionally stores meta-data if configured.
        """
        self.__dim = problem.dim
        self.__pop = self.__mulgenerate_pop(self.__BATCH_SIZE, self.__config.NP, self.__dim, problem.lb, problem.ub, True)   # [bs, NP, dim]
        self.__fit = self.__get_cost([problem], self.__pop)
        self.gbest_cost = np.min(self.__fit)

        self.fes = self.__config.NP
        self.log_index = 1
        self.cost = [self.gbest_cost]
        self.__past_histo = (self.__config.NP/self.__config.BINS) * np.ones((self.__BATCH_SIZE, 1, self.__config.BINS))

        if self.__config.full_meta_data:
            self.meta_X = [self.__pop.copy()[0]]
            self.meta_Cost = [self.__fit.copy()[0]]
        return self.__get_feature()

    def get_best(self):
        """
        # Introduction
        Retrieves the best (global best) cost found by the optimizer.
        # Returns:
        - float: The lowest cost value (gbest_cost) discovered during the optimization process.
        """
        
        return self.gbest_cost

    def __get_feature(self):
        """
        # Introduction
        Computes and returns the input features for the optimizer's neural network, combining normalized fitness values, fitness histograms, and historical histogram means.
        # Returns:
        - np.ndarray: A 2D array of shape [batch_size, NP + BINS * 2], where each row contains the concatenated normalized fitness, current fitness histogram, and mean of past histograms for each batch.
        # Notes:
        - Assumes that `self.__pop`, `self.__fit`, `self.__order_by_f`, `self.__maxmin_norm`, `self.__BATCH_SIZE`, `self.__config.BINS`, and `self.__past_histo` are properly initialized and available as class attributes.
        """
        
        self.__pop, self.__fit = self.__order_by_f(self.__pop, self.__fit)
        fitness = self.__maxmin_norm(self.__fit)
        hist_fit = []
        for b in range(self.__BATCH_SIZE):
            hist_fit.append(np.histogram(fitness[b], self.__config.BINS)[0])
        hist_fit = np.vstack(hist_fit)  # [bs, BINS]

        mean_past_histo = np.mean(self.__past_histo, axis=1)   # [bs, BINS]

        # [bs, NP+BINS*2]
        input_net = np.concatenate((fitness, hist_fit, mean_past_histo), axis=1)
        return input_net

    def update(self, action, problem):
        """
        # Introduction
        Updates the population and fitness values in the LDE optimizer using the provided action and problem instance. This method performs one iteration of the optimization process, applying mutation, crossover, and selection operations, and computes the reward and termination status.
        # Args:
        - action (np.ndarray): The action tensor containing scale factors and crossover rates for the population, typically output from a policy network. Shape: [batch_size, NP*2].
        - problem (object): The optimization problem instance, which should provide an evaluation method and may contain an optimum attribute.
        # Returns:
        - feature (np.ndarray): The extracted feature representation of the current population state.
        - reward (np.ndarray): The reward signal computed based on the improvement in best-so-far fitness.
        - is_done (bool): Flag indicating whether the optimization process has reached its termination condition.
        - info (dict): Additional information dictionary (currently empty).
        # Notes:
        - Updates internal state variables such as population, fitness, best-so-far cost, and historical fitness distribution.
        - Handles logging and meta-data collection if enabled in the configuration.
        - The method assumes that the action tensor is properly formatted and that the problem instance provides necessary evaluation functionality.
        """
        
        self.__pop, self.__fit = self.__order_by_f(self.__pop, self.__fit)
        fitness = self.__maxmin_norm(self.__fit)
        # sf_cr = np.squeeze(action.cpu().numpy(), axis=0)  # [bs, NP*2]
        sf = action[:, 0:self.__config.NP]  # scale factor [bs, NP]
        cr = action[:, self.__config.NP:2*self.__config.NP]  # crossover rate  [bs, NP]
        sf_mat = self.__con2mat_rand2pbest_Nw(sf, self.fes, self.__config.maxFEs)  # [NP, NP]
        mu_pop = self.__add_random(np.matmul(sf_mat, self.__pop), self.__pop, sf)  # [NP, dim]

        pop_next, fit_next, self.fes = self.__de_crosselect_random_dataset(self.__pop, mu_pop, self.__fit, cr, self.fes, problem)  # DE
        bsf = self.__fit.min(1)
        bsf_next = fit_next.min(1)

        reward = (bsf - bsf_next)/bsf  # reward

        if problem.optimum is None:
            is_done = self.fes >= self.__config.maxFEs
        else:
            is_done = self.fes >= self.__config.maxFEs

        self.__pop = pop_next
        self.__fit = fit_next

        hist_fit = []
        for b in range(self.__BATCH_SIZE):
            hist_fit.append(np.histogram(fitness[b], self.__config.BINS)[0])
        hist_fit = np.vstack(hist_fit)  # [bs, BINS]
        self.__past_histo = np.concatenate((self.__past_histo, hist_fit[:, None, :]), axis=1)
        self.gbest_cost = np.min(self.__fit)

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest_cost)

        if self.__config.full_meta_data:
            self.meta_X.append(self.__pop.copy()[0])
            self.meta_Cost.append(self.__fit.copy()[0])
        if is_done:
            if len(self.cost) >= self.__config.n_logpoint + 1:
                self.cost[-1] = self.gbest_cost
            else:
                while len(self.cost) < self.__config.n_logpoint + 1:
                    self.cost.append(self.gbest_cost)
        
        info = {}
        
        return self.__get_feature(), reward, is_done, info
