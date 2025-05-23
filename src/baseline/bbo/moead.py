import copy
import functools
import numpy as np
import math
import sys

from operator import itemgetter
from scipy.spatial.distance import cdist
from ...environment.optimizer.basic_optimizer import Basic_Optimizer
# from baseline.bbo.moo_utils import *
from ...environment.problem.MOO.MOO_synthetic.dtlz_numpy import *



POSITIVE_INFINITY = float("inf")
EPSILON = sys.float_info.epsilon


class PlatypusError(Exception):
    pass



class MOEAD(Basic_Optimizer):
    """
    # Introduction
    MOEAD is a multiobjective evolutionary algorithm based on decomposition.It decomposes a multiobjective optimization problem into a number of scalar optimization subproblems and optimizes them simultaneously. Each subproblem is optimized by only using information from its several neighboring subproblems.
    # Original paper
    "[**MOEA/D: A multiobjective evolutionary algorithm based on decomposition**](https://ieeexplore.ieee.org/abstract/document/4358754/)." IEEE Transactions on Evolutionary Computation 11.6 (2007): 712-731.
    # Official Implementation
    None
    """
    
    def __init__(self, config):
        """
        Initializes the MOEA/D algorithm with the provided configuration.
        # Introduction
        Sets up problem-related and algorithm-specific parameters for the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) baseline implementation.
        # Args:
        - config (object): Config object containing algorithm and problem settings.
            - The Attributes needed for the MOEAD are the following:
                - maxFEs (int): Maximum number of function evaluations allowed.
                - n_logpoint (int): Number of log points to record.Default is 50.
                - log_interval (int): Interval at which logs are recorded.Default is config.maxFEs/config.n_logpoint.
                
        # Built-in Attributes:
        - `self.n_ref_points` (int): Number of reference points for the algorithm. Default is 1000.
        - `self.population_size` (int): Size of the population.Default is 100.
        - `self.moead_neighborhood_size` (int): Size of the neighborhood for each subproblem.Default is 8.
        - `self.moead_neighborhood_maxsize` (int): Maximum size of the neighborhood.Default is 30.
        - `self.moead_delta` (float): Probability of using the neighborhood for mating.Default is 0.8.
        - `self.moead_eta` (int): Number of neighbors to consider for solution replacement.Default is 2.
        - `self.max_fes` (int): Maximum number of function evaluations allowed.Default is config.maxFEs.
        
        # Raises:
        - AttributeError: If `config` does not contain the required attributes.
        """
        
        super().__init__(config)
        # Problem Related
        self.n_ref_points = 1000
        # # MOEA/D Algorithm Related
        self.population_size = 100
        self.moead_neighborhood_size = 8
        self.moead_neighborhood_maxsize = 30
        self.moead_delta = 0.8
        self.moead_eta = 2
        # self.max_fes=config.maxFEs
        self.max_fes = config.maxFEs
    def __str__(self):
        """
        # Introduction
        Returns the string representation of the MOEAD class.
        # Returns:
        - str: The string 'MOEAD', representing the name of the class.
        """
        
        return 'MOEAD'

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and related attributes for the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) algorithm.
        # Args:
        - problem (object): An instance representing the optimization problem, which must provide attributes such as `n_obj` (number of objectives), `dim` (number of variables), `lb` (lower bounds), `ub` (upper bounds), and methods like `eval()` and `get_ref_set()`.
        # Built-in Attribute:
        - self.problem: Stores the problem instance.
        - self.n_obj: Number of objectives.
        - self.n_var: Number of decision variables.
        - self.weights: Weight vectors for decomposition.
        - self.population_size: Number of individuals in the population.
        - self.population: The initialized population of solutions.
        - self.population_obj: Objective values of the population.
        - self.neighborhoods: Neighborhood structure for each individual.
        - self.done: Boolean flag indicating if the optimization is finished.
        - self.fes: Number of function evaluations so far.
        - self.episode_limit: Maximum number of generations (episodes).
        - self.moead_generation: Current generation counter.
        - self.archive_maximum: Maximum objective values found so far.
        - self.archive_minimum: Minimum objective values found so far.
        - self.ideal_point: Current ideal point in the objective space.
        - self.problem_ref_points: Reference points for performance indicators.
        - self.igd_his: History of IGD (Inverted Generational Distance) values.
        - self.initial_igd: IGD value at initialization.
        - self.last_igd: Most recent IGD value.
        - self.best_igd: Best IGD value found so far.
        - self.hv_his: History of HV (Hypervolume) values.
        - self.initial_hv: HV value at initialization.
        - self.last_hv: Most recent HV value.
        - self.best_hv: Best HV value found so far.
        - self.metadata: Dictionary for storing additional metadata.
        # Returns:
        - None
        # Raises:
        - None
        """
        
        # problem
        self.problem = problem
        self.n_obj = problem.n_obj
        self.n_var = problem.dim
        # population
        self.weights = self.get_weights(self.n_obj)
        if self.population_size!=len(self.weights):
            self.population_size = len(self.weights)
        self.population = self.rng.uniform(low=problem.lb, high=problem.ub, size=(self.population_size, problem.dim))
        self.population_obj = problem.eval(self.population)
        self.neighborhoods = self.get_neighborhoods()
        # budget
        self.done = False
        self.fes = len(self.population)
        self.episode_limit = self.max_fes // self.population_size
        self.moead_generation = 0 
        # reference
        self.archive_maximum = np.max(self.population_obj, axis=0)
        self.archive_minimum = np.min(self.population_obj, axis=0)
        self.ideal_point = copy.deepcopy(self.archive_minimum)
        self.problem_ref_points = self.problem.get_ref_set(
            n_ref_points=self.n_ref_points)
        # indicators
        self.igd_his = []
        self.initial_igd = self.get_igd()
        self.last_igd = self.initial_igd
        self.best_igd = self.initial_igd
        self.hv_his = []
        self.initial_hv = self.get_hv()
        self.last_hv = self.initial_hv
        self.best_hv = self.initial_hv
        self.metadata = {'X':[],'cost':[]}
        self.update_information()

    def get_neighborhoods(self):
        """
        # Introduction
        Computes the neighborhoods for each weight vector in the MOEA/D algorithm by sorting all weight vectors according to their distance to the current weight and selecting the closest ones.
        # Returns:
        - list[list[int]]: A list where each element contains the indices of the nearest neighbors for the corresponding weight vector.
        # Notes:
        - The number of neighbors is determined by `self.moead_neighborhood_maxsize`.
        - The sorting of weights is performed using `self.moead_sort_weights`.
        """
        
        neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(len(self.weights)):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])
        return neighborhoods

    def get_weights(self, n_obj):
        """
        # Introduction
        Generates a set of weight vectors for multi-objective optimization based on the number of objectives.
        # Args:
        - n_obj (int): The number of objectives for which the weight vectors are to be generated.
        # Returns:
        - np.ndarray: An array of weight vectors suitable for the specified number of objectives.
        # Notes:
        - For specific values of `n_obj` (2, 3, 5, 7, 8, 10), predefined boundary weights are generated using `normal_boundary_weights` with preset parameters.
        - For other values of `n_obj`, random weights are generated using `random_weights` with the current population size.
        """
        
        weights = None
        if n_obj == 2:
            weights = self.normal_boundary_weights(n_obj,99, 0)
        elif n_obj == 3:
            weights = self.normal_boundary_weights(n_obj, 13, 0)
        elif n_obj == 5:
            weights = self.normal_boundary_weights(n_obj, 5, 0)
        elif n_obj == 7:
            weights = self.normal_boundary_weights(n_obj, 3, 2)
        elif n_obj == 8:
            weights = self.normal_boundary_weights(n_obj, 3, 1)
        elif n_obj == 10:
            weights = self.normal_boundary_weights(n_obj,2, 2)
        else:
            weights = self.random_weights(n_obj, self.population_size)
        return weights

    def moead_update_ideal(self, solution_obj):
        """
        # Introduction
        Updates the ideal point in the MOEA/D algorithm based on a new solution's objective values.
        # Args:
        - solution_obj (np.ndarray): A 1D array containing the objective values of the new solution.
        # Built-in Attribute:
        - self.ideal_point (np.ndarray): The current ideal point vector, updated in-place.
        # Returns:
        - None
        # Raises:
        - IndexError: If the dimensions of `solution_obj` do not match `self.ideal_point`.
        """
        
        for i in range(solution_obj.shape[-1]):
            self.ideal_point[i] = min(
                self.ideal_point[i], solution_obj[i])

    def run_episode(self, problem):
        """
        # Introduction
        Executes one episode of the MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) optimization process for a given problem instance. This involves initializing the population, generating offspring through crossover and mutation, evaluating solutions, updating ideal points and solutions, and tracking performance metrics such as IGD (Inverted Generational Distance) and HV (Hypervolume).
        # Args:
        - problem (object): The optimization problem instance, which must provide an `eval` method for evaluating solutions.
        # Built-in Attribute:
        - self.population (list): The current population of solutions.
        - self.fes (int): The number of function evaluations performed.
        - self.max_fes (int): The maximum number of function evaluations allowed.
        - self.done (bool): Indicates whether the optimization process is complete.
        - self.offspring_list (list): Stores the offspring generated in the current episode.
        - self.offspring_obj_list (list): Stores the objective values of the offspring.
        - self.moead_generation (int): The current generation count.
        - self.last_igd (float): The IGD value of the last generation.
        - self.best_igd (float): The best IGD value observed so far.
        - self.last_hv (float): The HV value of the last generation.
        - self.best_hv (float): The best HV value observed so far.
        - self.cost (float): The cost associated with the optimization process.
        - self.metadata (dict): Additional metadata collected during optimization.
        - self.hv_his (list): History of HV values.
        - self.igd_his (list): History of IGD values.
        # Returns:
        - dict: A dictionary containing the final cost, number of function evaluations, metadata, and histories of HV and IGD values when the episode is complete.
        # Raises:
        - None explicitly, but may raise exceptions if the problem instance or internal methods are misconfigured.
        """
        
        self.init_population(problem)
        
        while not self.done:
            subproblems = self.moead_get_subproblems()
            self.offspring_list = []
            self.offspring_obj_list = []
            for index in subproblems:
                mating_indices = self.moead_get_mating_indices(index)
                mating_population = [self.population[i] for i in mating_indices]
                if index in mating_indices:
                    mating_indices.remove(index)

                parents = [self.population[index]] + \
                            [self.population[i] for i in
                            self.rng.choice(mating_indices, 2, replace=False)]
                offspring = self.sbx(problem, parents)[0]
                offspring = self.pm(problem, offspring)
                offspring_obj = problem.eval(offspring)
                self.fes += 1
                safe_extend(self.offspring_list,offspring)
                safe_extend(self.offspring_obj_list,offspring_obj)
                # if offspring.ndim == 1:  # Check if offspring is a 1D array
                for child, child_obj in zip([offspring], [offspring_obj]):
                    self.moead_update_ideal(child_obj)
                    self.moead_update_solution(child, child_obj, mating_indices)
                # else:
                #     for child, child_obj in zip(offspring, offspring_obj):
                #         self.moead_update_ideal(child_obj)
                #         self.moead_update_solution(child, child_obj, mating_indices)
            self.moead_generation += 1
            self.last_igd = self.get_igd()
            self.best_igd = min(self.best_igd, self.last_igd)
            self.last_hv = self.get_hv()
            self.best_hv = max(self.best_hv, self.last_hv)
            self.update_information()
            print("igd:{},hv:{}".format(self.last_igd,self.last_hv))
            if self.fes >= self.max_fes:
                self.done = True
                print("fes:{},last_igd:{},last_hv:{}".format(self.fes,self.last_igd,self.last_hv))
                results = {'cost': self.cost, 'fes': self.fes, 'metadata': self.metadata,'hv_his':self.hv_his,'igd_his':self.igd_his}
                return results
            else:
                self.done = False
            
    def update_information(self):
        """
        # Introduction
        Updates the internal state of the object by identifying the non-dominated solutions (Pareto front) in the current population and storing relevant metadata.
        # Args:
        None
        # Built-in Attribute:
        - self.population_obj (list): Objective values of the current population.
        - self.population (list): Current population of solutions.
        - self.metadata (dict): Dictionary to store historical populations and objective values.
        - self.cost (list): Stores the current Pareto front objective values.
        # Returns:
        None
        # Raises:
        None
        """
        
        index =  self.find_non_dominated_indices(self.population_obj)
        self.cost = [copy.deepcopy(self.population_obj[i]) for i in index] # parato front
        self.metadata['X'].append(copy.deepcopy(self.population))
        self.metadata['cost'].append(copy.deepcopy(self.population_obj))
        
    def find_non_dominated_indices(self, population_list):
        """
        # Introduction
        Identifies the indices of non-dominated solutions (Pareto optimal solutions) in a given population based on their objective values.
        # Args:
        - population_list (List[List[float]]): A list where each element is a list representing the objective values of a solution in the population.
        # Returns:
        - np.ndarray: An array of indices corresponding to the non-dominated solutions in the population.
        # Raises:
        - None
        """
    
        # 将列表转换为 numpy 数组
        population = np.array(population_list)
        n_solutions = population.shape[0]
        is_dominated = np.zeros(n_solutions, dtype=bool)

        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # 检查是否存在解 j 支配解 i
                    if np.all(population[j] <= population[i]) and np.any(population[j] < population[i]):
                        is_dominated[i] = True
                        break

        # 找出非支配解的索引
        non_dominated_indices = np.where(~is_dominated)[0]
        return non_dominated_indices

    def moead_calculate_fitness(self, solution_obj, weights):
        """
        # Introduction
        Calculates the fitness value of a solution in the MOEA/D algorithm using the Chebyshev scalarizing function.
        # Args:
        - solution_obj (list or np.ndarray): The objective values of the solution to be evaluated.
        - weights (list or np.ndarray): The weight vector associated with the subproblem.
        # Built-in Attribute:
        - self.ideal_point (list or np.ndarray): The ideal point in the objective space, used as a reference for scalarization.
        # Returns:
        - float: The computed fitness value of the solution based on the Chebyshev approach.
        # Raises:
        - None
        """
        
        return chebyshev(solution_obj, self.ideal_point, weights)

    def moead_update_solution(self, solution, solution_obj, mating_indices):
        """
        # Introduction

        Updates the MOEA/D population by potentially replacing individuals in the mating set with a new solution if it improves the scalarized fitness value under the corresponding weight vector.

        # Args:

        - solution (Any): The candidate solution to potentially insert into the population.
        - solution_obj (Any): The objective values associated with the candidate solution.
        - mating_indices (List[int]): Indices of population members considered for replacement.

        # Returns:

        - None

        # Notes:

        - The method shuffles the mating indices, then iterates through them, replacing individuals if the new solution yields a better scalarized fitness.
        - Replacement stops after `self.moead_eta` successful replacements.
        """
        c = 0
        self.rng.shuffle(mating_indices)

        for i in mating_indices:
            candidate = self.population[i]
            candidate_obj = self.population_obj[i]
            weights = self.weights[i]
            replace = False
            if self.moead_calculate_fitness(solution_obj, weights) < self.moead_calculate_fitness(candidate_obj,
                                                                                                  weights):
                replace = True

            if replace:
                self.population[i] = copy.deepcopy(solution)
                self.population_obj[i] = copy.deepcopy(solution_obj)
                c = c + 1

            if c >= self.moead_eta:
                break

    @staticmethod
    def moead_sort_weights(base, weights):
        """
        # Introduction
        Sorts a list of weight vectors by their Euclidean distance to a given base weight vector, returning the indices of the weights in ascending order of distance.
        # Args:
        - base (list[float]): The reference weight vector to which distances are computed.
        - weights (list[list[float]]): A list of weight vectors to be sorted.
        # Returns:
        - list[int]: A list of indices representing the order of `weights` sorted by increasing distance to `base`.
        # Raises:
        - ValueError: If the dimensions of `base` and any weight vector in `weights` do not match.
        """

        def compare(weight1, weight2):
            dist1 = math.sqrt(
                sum([math.pow(base[i] - weight1[1][i], 2.0) for i in range(len(base))]))
            dist2 = math.sqrt(
                sum([math.pow(base[i] - weight2[1][i], 2.0) for i in range(len(base))]))

            if dist1 < dist2:
                return -1
            elif dist1 > dist2:
                return 1
            else:
                return 0

        sorted_weights = sorted(
            enumerate(weights), key=functools.cmp_to_key(compare))
        return [i[0] for i in sorted_weights]

    def moead_get_subproblems(self):
        """
        # Introduction

        Determines the order of subproblems to be searched in the MOEA/D algorithm. 
        If utility-based updating is enabled, it follows the utility-based search; 
        otherwise, it uses the original MOEA/D specification.

        # Returns:

        - list[int]: A shuffled list of indices representing the subproblems to be searched.

        # Built-in Attribute:

        - self.population_size (int): The total number of subproblems (population size).
        - self.rng (random.Random): Random number generator used for shuffling.
        """
        indices = list(range(self.population_size))
        self.rng.shuffle(indices)
        return indices

    def moead_get_mating_indices(self, index):
        """
        # Introduction

        Determines the mating indices for the MOEA/D algorithm based on the current individual's index.

        With probability `moead_delta`, the method returns the neighborhood indices for the given individual; otherwise, it returns the indices of the entire population.

        # Args:

        - index (int): The index of the current individual in the population.

        # Built-in Attribute:

        - self.rng (np.random.Generator): Random number generator for stochastic decisions.
        - self.moead_delta (float): Probability of selecting the neighborhood.
        - self.neighborhoods (List[List[int]]): Precomputed neighborhoods for each individual.
        - self.moead_neighborhood_size (int): Number of neighbors to consider.
        - self.population_size (int): Total number of individuals in the population.

        # Returns:

        - List[int]: Indices of individuals considered for mating (either the neighborhood or the entire population).
        """
        
        if self.rng.uniform(0.0, 1.0) <= self.moead_delta:
            return self.neighborhoods[index][:self.moead_neighborhood_size]
        else:
            return list(range(self.population_size))

    def get_hv(self,n_samples=1e5):
        """
        # Introduction
        Computes the hypervolume (HV) indicator for the current population of solutions, either exactly or approximately, depending on the problem's dimensionality and population size.
        # Args:
        - n_samples (int, optional): Number of Monte Carlo samples to use for HV estimation when using the approximate method. Defaults to 1e5.
        # Returns:
        - float: The computed or estimated hypervolume value for the current population.
        # Details:
        - If the number of objectives (`self.problem.n_obj`) is 3 or fewer, or the population size is 50 or fewer, the exact hypervolume is calculated.
        - Otherwise, the hypervolume is estimated using a Monte Carlo sampling approach for efficiency.
        # Raises:
        - AssertionError: If the normalized population objectives are not all less than 1 during the approximate calculation.
        """
        
        if self.problem.n_obj <= 3 or self.population_size <= 50:
            hv_fast = False
        else:
            hv_fast = True
        if not hv_fast:
            # Calculate the exact hv value
            hyp = Hypervolume(minimum=[0 for _ in range(
                self.n_obj)], maximum=self.archive_maximum)
            hv_value = hyp.calculate(np.array(self.population_obj))
        else:
            # Estimate the hv value by Monte Carlo
            popobj = copy.deepcopy(self.population_obj)
            optimum = self.problem_ref_points
            fmin = np.clip(np.min(popobj, axis=0), np.min(popobj), 0)
            fmax = np.max(optimum, axis=0)

            popobj = (popobj - np.tile(fmin, (self.population_size, 1))) / (
                np.tile(1.1 * (fmax - fmin), (self.population_size, 1)))
            index = np.all(popobj < 1, 1).tolist()
            popobj = popobj[index]
            if popobj.shape[0] <= 1:
                hv_value = 0
                self.hv_his.append(hv_value)
                # self.hv_last5.add(hv_value)
                # self.hv_running.update(np.array([hv_value]))
                return hv_value
            assert np.max(popobj) < 1
            hv_maximum = np.ones([self.n_obj])
            hv_minimum = np.min(popobj, axis=0)
            n_samples_hv = int(n_samples)
            samples = np.zeros([n_samples_hv, self.n_obj])
            for i in range(self.n_obj):
                samples[:, i] = self.rng.uniform(
                    hv_minimum[i], hv_maximum[i], n_samples_hv)
            for i in range(popobj.shape[0]):
                domi = np.ones([samples.shape[0]], dtype=bool)
                m = 0
                while m < self.n_obj and any(domi):
                    domi = np.logical_and(domi, popobj[i, m] <= samples[:, m])
                    m += 1
                save_id = np.logical_not(domi)
                samples = samples[save_id, :]
            hv_value = np.prod(hv_maximum - hv_minimum) * (
                    1 - samples.shape[0] / n_samples_hv)
        self.hv_his.append(hv_value)
        return hv_value

    def get_igd(self):
        """
        # Introduction
        Calculates the Inverted Generational Distance (IGD) between the current population objectives and a reference set, appends the result to the IGD history, and returns the computed IGD value.
        # Args:
        None
        # Returns:
        - float: The calculated IGD value for the current population.
        # Raises:
        - Any exception raised by the `InvertedGenerationalDistance` calculation if the input data is invalid.
        """
        
        igd_calculator = InvertedGenerationalDistance(reference_set=self.problem_ref_points)
        igd_value = igd_calculator.calculate(self.population_obj)
        self.igd_his.append(igd_value)

        return igd_value

    def close(self):
        """
        # Introduction
        Closes the current instance by resetting its internal state.
        # Args:
        None
        # Returns:
        None
        # Raises:
        None
        """
        
        self.reset()
    
    def normal_boundary_weights(self,nobjs, divisions_outer, divisions_inner=0):
        """
        # Introduction

        Generates a set of uniformly distributed weight vectors on the unit simplex using the normal boundary intersection method. This is commonly used in multi-objective optimization algorithms such as MOEA/D to decompose the objective space.

        # Args:

        - nobjs (int): The number of objectives (dimensions) for which to generate the weights.
        - divisions_outer (int): The number of divisions for the outer set of weights, controlling the granularity of the weight vectors on the simplex boundary.
        - divisions_inner (int, optional): The number of divisions for the inner set of weights, used to generate additional weights inside the simplex. Defaults to 0 (no inner weights).

        # Returns:

        - list[list[float]]: A list of weight vectors, where each vector is a list of floats summing to 1, representing a point on the unit simplex.

        # Notes:

        - The outer weights are distributed on the boundary of the simplex, while the optional inner weights are distributed inside the simplex and averaged with the uniform vector.
        - Useful for decomposition-based multi-objective evolutionary algorithms.
        """

        def generate_recursive(weights, weight, left, total, index):
            if index == nobjs - 1:
                weight[index] = float(left) / float(total)
                weights.append(copy.copy(weight))
            else:
                for i in range(left + 1):
                    weight[index] = float(i) / float(total)
                    generate_recursive(weights, weight, left - i, total, index + 1)

        def generate_weights(divisions):
            weights = []
            generate_recursive(weights, [0.0] * nobjs, divisions, divisions, 0)
            return weights

        weights = generate_weights(divisions_outer)

        if divisions_inner > 0:
            inner_weights = generate_weights(divisions_inner)

            for i in range(len(inner_weights)):
                weight = inner_weights[i]

                for j in range(len(weight)):
                    weight[j] = (1.0 / nobjs + weight[j]) / 2.0

                weights.append(weight)

        return weights

    def random_weights(self,nobjs, population_size):
        """
        # Introduction
        Generates a set of randomly-generated but uniformly distributed weight vectors for multi-objective optimization.
        This method ensures that the generated weights are as uniformly distributed as possible by maximizing the minimum distance between selected weights. For two objectives, weights are generated directly. For more than two objectives, a large pool of candidate weights is created, and the most distant candidates are iteratively selected.
        # Args:
        - nobjs (int): The number of objectives (dimensions) for the weight vectors.
        - population_size (int): The number of weight vectors to generate.
        # Returns:
        - list[list[float]]: A list of weight vectors, each of length `nobjs`, where each vector sums to 1.
        # Raises:
        - None explicitly, but may raise exceptions if invalid arguments are provided (e.g., population_size < nobjs).
        """
        
        weights = []
        
        if nobjs == 2:
            weights = [[1, 0], [0, 1]]
            weights.extend([(i/(population_size-1.0), 1.0-i/(population_size-1.0)) for i in range(1, population_size-1)])
        else:
            # generate candidate weights
            candidate_weights = []
            
            for i in range(population_size*50):
                random_values = [np.random.uniform(0.0, 1.0) for _ in range(nobjs)]
                candidate_weights.append([x/sum(random_values) for x in random_values])
            
            # add weights for the corners
            for i in range(nobjs):
                weights.append([0]*i + [1] + [0]*(nobjs-i-1))
                
            # iteratively fill in the remaining weights by finding the candidate
            # weight with the largest distance from the assigned weights
            while len(weights) < population_size:
                max_index = -1
                max_distance = -POSITIVE_INFINITY
                
                for i in range(len(candidate_weights)):
                    distance = POSITIVE_INFINITY
                    
                    for j in range(len(weights)):
                        temp = math.sqrt(sum([math.pow(candidate_weights[i][k]-weights[j][k], 2.0) for k in range(nobjs)]))
                        distance = min(distance, temp)
                        
                    if distance > max_distance:
                        max_index = i
                        max_distance = distance
                        
                weights.append(candidate_weights[max_index])
                del candidate_weights[max_index]
                
        return weights

    def sbx(self, problem,parents,probability=1.0, distribution_index=20.0):
        """
        # Introduction
        Performs Simulated Binary Crossover (SBX) on a pair of parent solutions to generate two offspring solutions for evolutionary algorithms. SBX is commonly used in real-coded genetic algorithms to recombine parent solutions while preserving variable bounds.
        # Args:
        - problem: An object representing the optimization problem, which must have the following attributes:
            - dim (int): The number of decision variables.
            - lb (list or array-like): Lower bounds for each variable.
            - ub (list or array-like): Upper bounds for each variable.
        - parents (list): A list containing two parent solutions (each as a list or array-like of variable values).
        - probability (float, optional): The probability of applying crossover to the parents. Defaults to 1.0.
        - distribution_index (float, optional): The distribution index (η) controlling the spread of offspring. Higher values result in offspring closer to parents. Defaults to 20.0.
        # Returns:
        - list: A list containing two offspring solutions (each as a list of variable values).
        # Notes:
        - The method uses a random number generator (`self.rng`) for stochastic operations.
        - Variable values are clipped to remain within the specified bounds after crossover.
        - If crossover is not applied (based on `probability`), the offspring are deep copies of the parents.
        """
        
        def sbx_crossover(x1, x2, lb, ub, distribution_index):
            dx = x2 - x1

            if dx > EPSILON:
                if x2 > x1:
                    y2 = x2
                    y1 = x1
                else:
                    y2 = x1
                    y1 = x2

                beta = 1.0 / (1.0 + (2.0 * (y1 - lb) / (y2 - y1)))
                alpha = 2.0 - pow(beta, distribution_index + 1.0)
                rand = self.rng.uniform(0.0, 1.0)

                if rand <= 1.0 / alpha:
                    alpha = alpha * rand
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0))
                else:
                    alpha = alpha * rand;
                    alpha = 1.0 / (2.0 - alpha)
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0))

                x1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                beta = 1.0 / (1.0 + (2.0 * (ub - y2) / (y2 - y1)));
                alpha = 2.0 - pow(beta, distribution_index + 1.0);

                if rand <= 1.0 / alpha:
                    alpha = alpha * rand
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0));
                else:
                    alpha = alpha * rand
                    alpha = 1.0 / (2.0 - alpha)
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0));

                x2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                # randomly swap the values
                if bool(self.rng.randint(0, 2)):
                    x1, x2 = x2, x1

                x1 = np.clip(x1, lb, ub)
                x2 = np.clip(x2, lb, ub)

            return x1, x2
        
        child1 = copy.deepcopy(parents[0])
        child2 = copy.deepcopy(parents[1])
        if self.rng.uniform(0.0, 1.0) <= probability:
            nvars = problem.dim

            for i in range(nvars):
                if self.rng.uniform(0.0, 1.0) <= 0.5:
                    x1 = float(child1[i])
                    x2 = float(child2[i])
                    lb = problem.lb[i]
                    ub = problem.ub[i]

                    x1, x2 = sbx_crossover(x1, x2, lb, ub,distribution_index=distribution_index)
                    child1[i] = x1
                    child2[i]= x2

        return [child1, child2]
 
    def pm(self, problem,parent, probability=1.0, distribution_index=20.0):
        """
        # Introduction
        Applies the Polynomial Mutation (PM) operator to a parent solution vector for evolutionary algorithms, producing a mutated child solution. This operator introduces diversity by perturbing solution variables within their bounds according to a specified probability and distribution index.
        # Args:
        - problem: An object representing the optimization problem, expected to have attributes `dim` (int, number of dimensions), `lb` (list or array of lower bounds), and `ub` (list or array of upper bounds).
        - parent (list or array-like): The parent solution vector to be mutated.
        - probability (float, optional): The overall mutation probability (default is 1.0). The per-variable mutation probability is computed as `probability / problem.dim`.
        - distribution_index (float, optional): The distribution index controlling the spread of the mutation (default is 20.0). Higher values result in smaller mutations.
        # Returns:
        - child (list): A new solution vector resulting from applying polynomial mutation to the parent.
        # Notes:
        - The method uses a random number generator `self.rng` for stochasticity.
        - Each variable in the parent has an independent chance to be mutated.
        - The mutated values are clipped to remain within the specified bounds.
        """
        
        def pm_mutation(x, lb, ub,distribution_index):
            u = self.rng.uniform(0, 1)
            dx = ub - lb

            if u < 0.5:
                bl = (x - lb) / dx
                b = 2.0 * u + (1.0 - 2.0 * u) * pow(1.0 - bl, distribution_index + 1.0)
                delta = pow(b, 1.0 / (distribution_index + 1.0)) - 1.0
            else:
                bu = (ub - x) / dx
                b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * pow(1.0 - bu, distribution_index + 1.0)
                delta = 1.0 - pow(b, 1.0 / (distribution_index + 1.0))

            x = x + delta * dx
            x = np.clip(x, lb, ub)

            return x
        child = copy.deepcopy(parent)
        probability /= float(problem.dim)

        for i in range(problem.dim):
            if self.rng.uniform(0.0, 1.0) <= probability:
                child[i] = pm_mutation(float(child[i]),
                                        problem.lb[i],
                                        problem.ub[i],
                                        distribution_index=distribution_index)
        return child

    

## Aggregate functions
def chebyshev(solution_obj, ideal_point, weights, min_weight=0.0001):
    """
    # Introduction

    Calculates the Chebyshev (Tchebycheff) fitness value for a multi-objective solution, assuming all objectives are to be minimized. This metric is commonly used in multi-objective optimization to aggregate multiple objectives into a single scalar value, emphasizing the worst (maximum) weighted deviation from the ideal point.

    # Args:

    - solution_obj (np.ndarray or list of float): The objective values of the solution.
    - ideal_point (list of float): The ideal (best known) values for each objective.
    - weights (list of float): The weights assigned to each objective.
    - min_weight (float, optional): The minimum allowable weight for any objective (default is 0.0001).

    # Returns:

    - float: The Chebyshev fitness value, representing the maximum weighted deviation from the ideal point.

    # Raises:

    - IndexError: If the lengths of `solution_obj`, `ideal_point`, or `weights` do not match.
    """
    objs = solution_obj
    n_obj = objs.shape[-1]
    return max([max(weights[i], min_weight) * (objs[i] - ideal_point[i]) for i in range(n_obj)])

def pbi(solution_obj, ideal_point, weights, theta=5):
    """
    # Introduction
    Calculates the Penalty-based Boundary Intersection (PBI) fitness value for a solution in multi-objective optimization, assuming all objectives are to be minimized. This metric is commonly used in decomposition-based evolutionary algorithms such as MOEA/D.
    # Args:
    - solution_obj (list or array-like of float): The objective values of the solution.
    - ideal_point (list or array-like of float): The ideal (reference) point in the objective space.
    - weights (list or array-like of float): The weight vector associated with the subproblem.
    - theta (float, optional): The penalty parameter controlling the balance between convergence and diversity (default is 5).
    # Returns:
    - float: The computed PBI fitness value for the given solution.
    # Raises:
    - ImportError: If numpy is not installed.
    """
    try:
        import numpy as np
    except:
        print("The pbi function requires numpy.", file=sys.stderr)
        raise

    w = np.array(weights)
    z_star = np.array(ideal_point)
    F = np.array(solution_obj)

    d1 = np.linalg.norm(np.dot((F - z_star), w)) / np.linalg.norm(w)
    d2 = np.linalg.norm(F - (z_star + d1 * w))

    return (d1 + theta * d2).tolist()

class Indicator(object):
    #__metaclass = ABCMeta

    def __init__(self):
        super(Indicator, self).__init__()

    def __call__(self, set):
        return self.calculate(set)

    def calculate(self, set):
        raise NotImplementedError("method not implemented")


class Hypervolume(Indicator):
    """
    # Introduction
    The `Hypervolume` class is an indicator used to calculate the hypervolume metric for multi-objective optimization problems. The hypervolume measures the volume of the objective space dominated by a set of solutions, bounded by a reference point. This implementation is specifically designed for minimization problems.
    """
    
    # 只适用于最小化问题

    def __init__(self, reference_set=None, minimum=None, maximum=None):
        """
        Initializes the Hypervolume object with either a reference set or explicit minimum and maximum values.
        # Args:
        - reference_set (array-like, optional): A set of reference points used for normalization. If provided, `minimum` and `maximum` must not be specified.
        - minimum (array-like, optional): The minimum values for normalization. Must be specified if `reference_set` is not provided.
        - maximum (array-like, optional): The maximum values for normalization. Must be specified if `reference_set` is not provided.
        # Returns:
        - None
        # Raises:
        - ValueError: If both `reference_set` and (`minimum` or `maximum`) are specified.
        - ValueError: If neither `reference_set` nor both `minimum` and `maximum` are specified.
        """
        
        super(Hypervolume, self).__init__()
        if reference_set is not None:
            if minimum is not None or maximum is not None:
                raise ValueError("minimum and maximum must not be specified if reference_set is defined")
            self.minimum, self.maximum = normalize(reference_set)
        else:
            if minimum is None or maximum is None:
                raise ValueError("minimum and maximum must be specified when no reference_set is defined")
            self.minimum, self.maximum = minimum, maximum

    def invert(self, solution_normalized_obj: np.ndarray):
        """
        # Introduction
        Inverts the normalized objective values for each objective in the solution array.
        Each value is clipped to the [0.0, 1.0] range before inversion.
        # Args:
        - solution_normalized_obj (np.ndarray): A 2D NumPy array of shape (n_solutions, n_objectives)
          containing normalized objective values in the range [0.0, 1.0].
        # Returns:
        - np.ndarray: The input array with each objective value inverted (i.e., 1.0 - value).
        # Raises:
        - None
        """
        
        for i in range(solution_normalized_obj.shape[1]):
            solution_normalized_obj[:, i] = 1.0 - np.clip(solution_normalized_obj[:, i], 0.0, 1.0)
        return solution_normalized_obj

    def dominates(self, solution1_obj, solution2_obj, nobjs):
        """
        # Introduction
        Determines whether the first solution dominates the second solution in a multi-objective optimization context.
        # Args:
        - solution1_obj (list or array-like): Objective values of the first solution.
        - solution2_obj (list or array-like): Objective values of the second solution.
        - nobjs (int): Number of objectives.
        # Returns:
        - bool: True if `solution1_obj` dominates `solution2_obj`, False otherwise.
        # Notes:
        - Dominance is determined by comparing each objective value. The function assumes a maximization problem, where higher objective values are better.
        """
        
        better = False
        worse = False

        for i in range(nobjs):
            if solution1_obj[i] > solution2_obj[i]:
                better = True
            else:
                worse = True
                break
        return not worse and better

    def swap(self, solutions_obj, i, j):
        """
        # Introduction
        Swaps the positions of two elements in the given solutions object at indices `i` and `j`.
        # Args:
        - solutions_obj (array-like): The collection (e.g., numpy array) containing the solutions.
        - i (int): The index of the first element to swap.
        - j (int): The index of the second element to swap.
        # Returns:
        - array-like: The solutions object with the elements at indices `i` and `j` swapped.
        # Raises:
        - IndexError: If `i` or `j` are out of bounds for `solutions_obj`.
        """
        
        solutions_obj[[i, j]] = solutions_obj[[j, i]]
        return solutions_obj

    def filter_nondominated(self, solutions_obj, nsols, nobjs):
        """
        # Introduction
        Filters out dominated solutions from a set of solutions based on their objective values, leaving only the non-dominated (Pareto optimal) solutions.
        # Args:
        - solutions_obj (list or np.ndarray): A collection of solution objective vectors to be filtered.
        - nsols (int): The number of solutions in the input collection.
        - nobjs (int): The number of objectives for each solution.
        # Returns:
        - int: The number of non-dominated solutions remaining after filtering.
        # Raises:
        - None
        """
        
        i = 0
        n = nsols
        while i < n:
            j = i + 1
            while j < n:
                if self.dominates(solutions_obj[i], solutions_obj[j], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, j, n)
                elif self.dominates(solutions_obj[j], solutions_obj[i], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, i, n)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1
        return n

    def surface_unchanged_to(self, solutions_normalized_obj, nsols, obj):
        """
        # Introduction
        Computes the minimum value of a specified objective across a subset of normalized solutions.
        # Args:
        - solutions_normalized_obj (np.ndarray): A 2D array containing normalized objective values for all solutions.
        - nsols (int): The number of solutions to consider from the beginning of the array.
        - obj (int): The index of the objective to evaluate.
        # Returns:
        - float: The minimum value of the specified objective among the first `nsols` solutions.
        # Raises:
        - IndexError: If `obj` is out of bounds for the number of objectives in `solutions_normalized_obj`.
        - ValueError: If `nsols` is greater than the number of available solutions.
        """
        
        return np.min(solutions_normalized_obj[:nsols, obj])

    def reduce_set(self, solutions, nsols, obj, threshold):
        """
        # Introduction
        Reduces the set of solutions by removing those whose value for a specified objective is less than or equal to a given threshold. The removal is performed in-place by swapping qualifying solutions to the end of the array.
        # Args:
        - solutions (np.ndarray): Array of candidate solutions, where each row represents a solution and columns represent objective values.
        - nsols (int): The current number of solutions in the set.
        - obj (int): The index of the objective to be considered for threshold comparison.
        - threshold (float): The threshold value for the specified objective. Solutions with values less than or equal to this threshold are removed.
        # Returns:
        - int: The updated number of solutions after reduction.
        # Notes:
        - The function modifies the `solutions` array in-place by swapping removed solutions to the end.
        """
        
        i = 0
        n = nsols
        while i < n:
            if solutions[i, obj] <= threshold:
                n -= 1
                solutions = self.swap(solutions, i, n)
            else:
                i += 1
        return n

    def calc_internal(self, solutions_obj: np.ndarray, nsols, nobjs):
        """
        # Introduction
        Recursively calculates the hypervolume (or a similar metric) of a set of solutions in a multi-objective optimization context using a divide-and-conquer approach.
        # Args:
        - solutions_obj (np.ndarray): A 2D array containing the objective values of the solutions.
        - nsols (int): The number of solutions to consider from `solutions_obj`.
        - nobjs (int): The number of objectives.
        # Returns:
        - float: The calculated hypervolume (or related metric) for the given set of solutions.
        # Raises:
        - None explicitly, but may raise exceptions if input arrays are malformed or if called recursively with invalid parameters.
        """
        
        volume = 0.0
        distance = 0.0
        n = nsols

        while n > 0:
            nnondom = self.filter_nondominated(solutions_obj, n, nobjs - 1)

            if nobjs < 3:
                temp_volume = solutions_obj[0][0]
            else:
                temp_volume = self.calc_internal(solutions_obj, nnondom, nobjs - 1)

            temp_distance = self.surface_unchanged_to(solutions_obj, n, nobjs - 1)
            volume += temp_volume * (temp_distance - distance)
            distance = temp_distance
            n = self.reduce_set(solutions_obj, n, nobjs - 1, distance)

        return volume

    def calculate(self, solutions_obj: np.ndarray):
        """
        # Introduction
        Calculates the hypervolume indicator for a set of solution objective values, considering only feasible solutions within normalized bounds.
        # Args:
        - solutions_obj (np.ndarray): A 2D array of shape (n_solutions, n_objectives) containing the objective values of the solutions.
        # Returns:
        - float: The computed hypervolume for the feasible solutions. Returns 0.0 if no feasible solutions are found.
        # Notes:
        - Solutions are first normalized using predefined minimum and maximum values.
        - Only solutions with all normalized objectives less than or equal to 1.0 are considered feasible.
        - The feasible solutions are inverted before hypervolume calculation.
        """
        
        # 对可行解进行归一化
        solutions_normalized_obj = normalize(solutions_obj, self.minimum, self.maximum)

        # 筛选出所有目标值都小于等于 1.0 的解
        valid_mask = np.all(solutions_normalized_obj <= 1.0, axis=1)
        valid_feasible = solutions_normalized_obj[valid_mask]

        if valid_feasible.size == 0:
            return 0.0

        # 对可行解进行反转操作
        inverted_feasible = self.invert(valid_feasible)

        # 计算超体积
        nobjs = inverted_feasible.shape[1]
        return self.calc_internal(inverted_feasible, len(inverted_feasible), nobjs)


class InvertedGenerationalDistance(Indicator):
    """
    # Introduction
    Represents the Inverted Generational Distance (IGD) indicator, which measures the average distance from points in a reference set to their nearest point in a given solution set. IGD is commonly used in multi-objective optimization to assess how well a set of solutions approximates the true Pareto front.
    # Args:
    - reference_set (Iterable): The set of reference points (typically representing the true Pareto front) against which the solution set will be evaluated.
    # Methods:
    - calculate(set): Computes the IGD value for a given solution set.
        - set (Iterable): The solution set to be evaluated.
    # Returns:
    - float: The IGD value, representing the average distance from each reference point to its nearest solution in the set.
    # Raises:
    - ZeroDivisionError: If the reference set is empty, as division by zero will occur.
    """

    def __init__(self, reference_set):
        super(InvertedGenerationalDistance, self).__init__()
        self.reference_set = reference_set


    def calculate(self, set):
        return sum([distance_to_nearest(s, set) for s in self.reference_set])/ len(self.reference_set)
                        

def distance_to_nearest(solution_obj, set):
    """
    # Introduction
    Calculates the minimum Euclidean distance between a given solution's objective vector and a set of other objective vectors.
    # Args:
    - solution_obj (array-like): The objective vector of the solution for which the nearest distance is to be calculated.
    - set (Iterable[array-like]): A collection of objective vectors to compare against.
    # Returns:
    - float: The smallest Euclidean distance between `solution_obj` and any element in `set`. Returns `POSITIVE_INFINITY` if `set` is empty.
    # Raises:
    - NameError: If `POSITIVE_INFINITY` or `euclidean_dist` are not defined in the scope.
    """
    if len(set) == 0:
        return POSITIVE_INFINITY

    return min([euclidean_dist(solution_obj, s) for s in set])


def euclidean_dist(x, y):
    """
    # Introduction
    Computes the Euclidean (L2) distance between two vectors `x` and `y`. Both inputs must be either lists or NumPy arrays of equal length.
    # Args:
    - x (list or numpy.ndarray): The first vector.
    - y (list or numpy.ndarray): The second vector.
    # Returns:
    - float: The Euclidean distance between `x` and `y`.
    # Raises:
    - TypeError: If `x` or `y` is not a list or numpy.ndarray.
    - IndexError: If `x` and `y` are not of the same length.
    """
    
    if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        print("x:", x)
        print("y:", y)
        raise TypeError("x and y must be lists or tuples.")

    return math.sqrt(sum([math.pow(x[i] - y[i], 2.0) for i in range(len(x))]))


def normalize(solutions_obj: np.ndarray, minimum: np.ndarray = None, maximum: np.ndarray = None) -> np.ndarray:
    """
    # Introduction

    This function scales each objective in the input solution array to the [0, 1] range using provided or computed minimum and maximum values for each objective. If minimum or maximum bounds are not provided, they are computed from the input data.

    # Args:

    - solutions_obj (np.ndarray): A 2D numpy array of shape (n_solutions, n_objectives) representing the objective values of solutions to be normalized.
    - minimum (np.ndarray, optional): A 1D numpy array of minimum values for each objective. If not provided, computed from `solutions_obj`.
    - maximum (np.ndarray, optional): A 1D numpy array of maximum values for each objective. If not provided, computed from `solutions_obj`.

    # Returns:

    - np.ndarray: A 2D numpy array of the same shape as `solutions_obj`, containing the normalized objective values.

    # Raises:

    - ValueError: If any objective has an empty range (i.e., maximum - minimum < EPSILON).
    """
    # 如果输入数组为空，直接返回空数组
    if len(solutions_obj) == 0:
        return solutions_obj

    # 获取目标的数量
    n_obj = solutions_obj.shape[1]

    # 如果 minimum 或 maximum 未提供，则计算它们
    if minimum is None or maximum is None:
        if minimum is None:
            minimum = np.min(solutions_obj, axis=0)
        if maximum is None:
            maximum = np.max(solutions_obj, axis=0)

    # 检查是否有目标的范围为空
    if np.any(maximum - minimum < EPSILON):
        raise ValueError("objective with empty range")

    # 进行归一化操作
    solutions_normalized_obj = (solutions_obj - minimum) / (maximum - minimum)

    return solutions_normalized_obj
    
def safe_extend(lst, items):
    """
    # Introduction
    Safely extends or appends items to a list, handling various input types such as lists, NumPy arrays, or single elements.
    # Args:
    - lst (list): The list to which items will be added.
    - items (Any): The items to add. Can be None, a list, a NumPy ndarray, or a single element.
    # Returns:
    - None: The function modifies `lst` in place and does not return a value.
    # Notes:
    - If `items` is None, the function does nothing.
    - If `items` is a multi-dimensional list or ndarray, it extends `lst` with its elements.
    - If `items` is a one-dimensional list or ndarray, it appends the entire object as a single element.
    - For other types, it appends `items` as a single element.
    """
    
    # 如果 items 是 None，跳过处理
    if items is None:
        return
    # 判断是否是 list 或 ndarray
    if isinstance(items, (list, np.ndarray)):
        # 获取维度
        shape = np.shape(items)
        if len(shape) > 1:  # 多维的：用 extend
            lst.extend(items)
        else:               # 一维的：用 append
            lst.append(items)
    else:
        # 不是列表/数组，默认 append
        lst.append(items)
    
