import numpy as np
import copy
from ...environment.optimizer.basic_optimizer import Basic_Optimizer


class JDE21(Basic_Optimizer):
    """
    # Introduction
    A DE for solving single-objective real-parameter bound-constrained optimization problems. It uses several mechanisms to tackle optimization problems efficiently: two populations with different sizes, restart mechanism in both populations, self-adaptive control parameters F and CR, the extended range of values for CR in thebigger population, migration of the best individual from the big population into the small population, modified mutation strategy in the bigger population, crowding mechanism and population size reduction in the bigger population.
    # Original paper
    "[**Self-adaptive differential evolution algorithm with population size reduction for single objective bound-constrained optimization: Algorithm j21**](https://ieeexplore.ieee.org/abstract/document/9504782/)." 2021 IEEE Congress on Evolutionary Computation (CEC). IEEE, 2021.
    """
    
    def __init__(self, config):
        """
        # Introduction
        Initializes the JDE21 optimizer with configuration parameters and sets up internal variables according to the JDE21 algorithm.
        # Args:
        - config (object): Configuration object containing algorithm parameters.
            - The Attributes needed for the JDE21 optimizer in config are the following:
                - maxFEs (int): Maximum number of function evaluations allowed. Default value depends on the type of the problem.
                - n_logpoint (int): Number of log points for tracking progress. Default is 50.
                - log_interval (int): Interval at which logs are recorded. Default is maxFEs // n_logpoint.
                - full_meta_data (bool): Flag indicating whether to store complete solution history. Default is False.
                - seed (int): Random seed for reproducibility. Used for initializing populations and control parameters.

        # Attributes:
        - __sNP (int): Size of the small population. Default is 10.
        - __bNP (int): Size of the big population. Default is 160.
        - __NP (int): Total population size. Default is 170.
        - __tao1 (float): Parameter for mutation strategy. Default is 0.1.
        - __tao2 (float): Parameter for crossover strategy. Default is 0.1.
        - __Finit (float): Initial scaling factor. Default is 0.5.
        - __CRinit (float): Initial crossover rate. Default is 0.9.
        - __Fl_b (float): Lower bound for scaling factor in big population. Default is 0.1.
        - __Fl_s (float): Lower bound for scaling factor in small population. Default is 0.17.
        - __Fu (float): Upper bound for scaling factor. Default is 1.1.
        - __CRl_b (float): Lower bound for crossover rate in big population. Default is 0.0.
        - __CRl_s (float): Lower bound for crossover rate in small population. Default is 0.1.
        - __CRu_b (float): Upper bound for crossover rate in big population. Default is 1.1.
        - __CRu_s (float): Upper bound for crossover rate in small population. Default is 0.8.
        # Notes:
        The meaning and usage of the parameters are based on the JDE21 paper. This constructor prepares all necessary internal state for running the JDE21 optimization algorithm.
        """
        
        super(JDE21, self).__init__(config)
        self.__sNP = 10       # size of small population
        self.__bNP = 160      # size of big population
        self.__NP = self.__sNP + self.__bNP
        # meaning of following parameters reference from the JDE21 paper
        self.__tao1 = 0.1
        self.__tao2 = 0.1
        self.__Finit = 0.5
        self.__CRinit = 0.9
        self.__Fl_b = 0.1
        self.__Fl_s = 0.17
        self.__Fu = 1.1
        self.__CRl_b = 0.0
        self.__CRl_s = 0.1
        self.__CRu_b = 1.1
        self.__CRu_s = 0.8
        self.__eps = 1e-12
        self.__MyEps = 0.25
        # record number of operation called
        self.__nReset = 0
        self.__sReset = 0
        self.__cCopy = 0
        # self.__terminateErrorValue = 1e-8
        self.__MaxFEs = config.maxFEs
        self.__FEs = 0
        self.gbest = 1e15
        self.__F = np.ones(self.__NP) * 0.5
        self.__Cr = np.ones(self.__NP) * 0.9
        self.__n_logpoint = config.n_logpoint
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        
    def __str__(self):
        """
        Returns the string representation of the JDE21 class.
        # Returns:
        - str: The string 'JDE21'.
        """
        
        return 'JDE21'
    # check whether the optimization stuck(global best doesn't improve for a while)
    def __prevecEnakih(self, cost, best):
        """
        # Introduction
        Determines if there are a significant number of elements in `cost` that are approximately equal to `best`, based on specified tolerances.
        # Args:
        - cost (np.ndarray): Array of cost values.
        - best (float): The reference value to compare against.
        # Returns:
        - bool: True if the number of elements in `cost` close to `best` exceeds both 2 and a fraction (`__MyEps`) of the total number of elements; otherwise, False.
        """
        
        eqs = len(cost[np.fabs(cost - best) < self.__eps])
        return eqs > 2 and eqs > len(cost) * self.__MyEps

    # crowding operation describe in JDE21
    def __crowding(self, group, vs):
        """
        # Introduction
        Computes the index of the closest vector in `vs` to each vector in `group` based on squared Euclidean distance.
        # Args:
        - group (np.ndarray): An array representing a group of vectors, shape (NP, dim).
        - vs (np.ndarray): An array of vectors to compare against, shape (NP, dim).
        # Returns:
        - np.ndarray: An array of indices indicating, for each vector in `group`, the index of the closest vector in `vs`.
        # Notes:
        - The function assumes that `group` and `vs` have the same shape.
        """
        
        NP, dim = vs.shape
        dist = np.sum(((group * np.ones((NP, NP, dim))).transpose(1, 0, 2) - vs) ** 2, -1).transpose()
        return np.argmin(dist, -1)

    def __evaluate(self, problem, Xs):
        """
        # Introduction
        Evaluates the cost of a solution or set of solutions `Xs` for a given optimization `problem`, optionally normalizing by the problem's known optimum. Also stores meta-data if enabled.
        # Args:
        - problem:The problem object representing the optimization problem.
        - Xs: The candidate solution(s) to be evaluated, typically as a NumPy array or compatible structure.
        # Returns:
        - cost: The evaluated cost(s) of the solution(s).
        # Notes:
        - If `self.full_meta_data` is `True`, the method appends the cost and solution to internal meta-data lists.
        """
        
        if problem.optimum is None:
            cost = problem.eval(Xs)
        else:
            cost = problem.eval(Xs) - problem.optimum
        return cost

    def __sort(self):
        """
        # Introduction
        Sorts the population and corresponding cost arrays in ascending order based on the cost values.
        # Args:
        None
        # Returns:
        None
        # Side Effects:
        - Updates `self.__cost` and `self.__population` so that both are sorted according to the ascending order of `self.__cost`.
        """
        
        # new index after sorting
        ind = np.argsort(self.__cost)
        self.__cost = self.__cost[ind]
        self.__population = self.__population[ind]

    def __reinitialize(self, size, problem):
        """
        # Introduction
        Reinitializes a population of candidate solutions within the problem's bounds using uniform random sampling.
        # Args:
        - size (int): The number of candidate solutions to generate.
        - problem (object): The problem object representing the optimization problem. Must have the following attributes:
            - dim (int): Dimensionality of the problem.
            - ub (float or np.ndarray): Upper bound(s) for each dimension.
            - lb (float or np.ndarray): Lower bound(s) for each dimension.
        # Returns:
        - np.ndarray: An array of shape (size, problem.dim) containing the reinitialized candidate solutions.
        # Notes:
        The method uses the instance's random number generator (`self.rng`) to ensure reproducibility.
        """
        
        return self.rng.random((size, problem.dim)) * (problem.ub - problem.lb) + problem.ub

    def __init_population(self, problem):
        """
        # Introduction
        Initializes the population and related attributes for the evolutionary optimization algorithm.
        # Args:
        - problem (object): The problem object representing the optimization problem.
        # Side Effects:
        - Initializes the population matrix with random values within the problem bounds.
        - Evaluates the initial population and stores their costs.
        - Sets up internal counters and parameters for the algorithm, such as population size, best cost, scaling factors, and crossover rates.
        - Initializes logging variables for tracking optimization progress.
        # Attributes Set:
        - __sNP (int): Size of the small population.Default is 10.
        - __bNP (int): Size of the big population. Default is 160.
        - __NP (int): Total population size. Default is 170.
        - __population (np.ndarray): The initial population matrix, shape (NP, problem.dim).
        - __cost (np.ndarray): The cost of each individual in the population, shape (NP,).
        - __FEs (int): Total number of function evaluations performed. Initialized to NP.
        - __cbest (float): The best cost found so far.
        - __cbest_id (int): The index of the best individual in the population.
        - __F (np.ndarray): Scaling factors for each individual in the population, shape (NP,).
        - __Cr (np.ndarray): Crossover rates for each individual in the population, shape (NP,).
        - log_index (int): Index for logging progress, initialized to 1.
        - cost (list): List to store the best cost found at each logging interval.
        
        """
        
        self.__sNP = 10
        self.__bNP = 160
        self.__NP = self.__sNP + self.__bNP
        self.__population = self.rng.rand(self.__NP, problem.dim) * (problem.ub - problem.lb) + problem.lb
        self.__cost = self.__evaluate(problem, self.__population)
        self.__FEs = self.__NP
        self.__cbest = self.gbest = np.min(self.__cost)
        self.__cbest_id = np.argmin(self.__cost)
        self.__F = np.ones(self.__NP) * 0.5
        self.__Cr = np.ones(self.__NP) * 0.9

        self.log_index = 1
        self.cost = [self.gbest]

    def __update(self,
                 problem,       # the problem instance
                 ):
        """
        # Introduction
        Performs one iteration of the population update for a differential evolution algorithm variant (likely NL-SHADE-RSP), including mutation, crossover, selection, population reinitialization, and population reduction. Handles both "big" and "small" subpopulations, manages best solution tracking, and logs progress.
        # Args:
        - problem: The problem object, which must provide at least the following attributes:
            - dim (int): Dimensionality of the problem.
            - lb (array-like): Lower bounds for each dimension.
            - ub (array-like): Upper bounds for each dimension.
            - optimum (optional): The known optimum value for early stopping (can be None).
        # Returns:
        - None
        """
        # initialize population
        NP = self.__NP
        dim = problem.dim
        sNP = self.__sNP
        bNP = NP - sNP
        age = 0

        def __mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big):
            if big:
                xNP = bNP
                randF = self.rng.rand(xNP) * self.__Fu + self.__Fl_b
                randCr = self.rng.rand(xNP) * self.__CRu_b + self.__CRl_b
                pF = self.__F[:xNP]
                pCr = self.__Cr[:xNP]
            else:
                xNP = sNP
                randF = self.rng.rand(xNP) * self.__Fu + self.__Fl_s
                randCr = self.rng.rand(xNP) * self.__CRu_b + self.__CRl_s
                pF = self.__F[bNP:]
                pCr = self.__Cr[bNP:]

            rvs = self.rng.rand(xNP)
            F = np.where(rvs < self.__tao1, randF, pF)
            rvs = self.rng.rand(xNP)
            Cr = np.where(rvs < self.__tao2, randCr, pCr)
            Fs = F.repeat(dim).reshape(xNP, dim)
            Cr[Cr > 1] = 0
            Crs = Cr.repeat(dim).reshape(xNP, dim)
            v = self.__population[r1] + Fs * (self.__population[r2] - self.__population[r3])
            v = np.where(v > problem.ub, (v - problem.lb) % (problem.ub - problem.lb) + problem.lb, v)
            v = np.where(v < problem.ub, (v - problem.ub) % (problem.ub - problem.lb) + problem.lb, v)
            # v = np.clip(v, problem.lb, problem.ub)
            jrand = self.rng.randint(dim, size=xNP)
            u = np.where(self.rng.rand(xNP, dim) < Crs, v, (self.__population[:bNP] if big else self.__population[bNP:]))
            u[np.arange(xNP), jrand] = v[np.arange(xNP), jrand]
            cost = self.__evaluate(problem, u)
            if big:
                crowding_ids = self.__crowding(self.__population[:xNP], u)
            else:
                crowding_ids = np.arange(xNP) + bNP
            age += xNP
            for i in range(xNP):
                id = crowding_ids[i]
                if cost[i] < self.__cost[id]:
                    # update and record
                    self.__population[id] = u[i]
                    self.__cost[id] = cost[i]
                    self.__F[id] = F[i]
                    self.__Cr[id] = Cr[i]
                    SF = np.append(SF, F[i])
                    SCr = np.append(SCr, Cr[i])
                    d = (self.__cost[i] - cost[i]) / (self.__cost[i] + 1e-9)
                    df = np.append(df, d)
                    if cost[i] < self.__cbest:
                        age = 0
                        self.__cbest_id = id
                        self.__cbest = cost[i]
                        if cost[i] < self.gbest:
                            self.gbest = cost[i]

            if self.full_meta_data:
                self.meta_X.append(self.__population[crowding_ids].copy())
                self.meta_Cost.append(self.__cost[crowding_ids].copy())

            return SF, SCr, df, age

        # self.__sort()
        # initialize temp records
        # small population evaluates same times as big one thus the total evaluations for a loop is doubled big one
        df = np.array([])
        SF = np.array([])
        SCr = np.array([])
        if self.__prevecEnakih(self.__cost[:bNP], self.gbest) or age > self.__MaxFEs / 10:
            self.__nReset += 1
            self.__population[:bNP] = self.__reinitialize(bNP, problem)
            self.__F[:bNP] = self.__Finit
            self.__Cr[:bNP] = self.__CRinit
            self.__cost[:bNP] = 1e15
            age = 0
            self.__cbest = np.min(self.__cost)
            self.__cbest_id = np.argmin(self.__cost)

        if self.__FEs < self.__MaxFEs / 3:
            mig = 1
        elif self.__FEs < 2 * self.__MaxFEs / 3:
            mig = 2
        else:
            mig = 3

        r1 = self.rng.randint(bNP, size=bNP)
        count = 0
        duplicate = np.where((r1 == np.arange(bNP)) * (r1 == self.__cbest_id))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = self.rng.randint(bNP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(bNP)) * (r1 == self.__cbest_id))[0]
            count += 1

        r2 = self.rng.randint(bNP + mig, size=bNP)
        count = 0
        duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = self.rng.randint(bNP + mig, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(bNP)) + (r2 == r1))[0]
            count += 1

        r3 = self.rng.randint(bNP + mig, size=bNP)
        count = 0
        duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r3[duplicate] = self.rng.randint(bNP + mig, size=duplicate.shape[0])
            duplicate = np.where((r3 == np.arange(bNP)) + (r3 == r1) + (r3 == r2))[0]
            count += 1

        SF, SCr, df, age = __mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big=True)
        self.__FEs += bNP

        if self.__cbest_id >= bNP and self.__prevecEnakih(self.__cost[bNP:], self.__cbest):
            self.__sReset += 1
            cbest = self.__cbest
            cbest_id = self.__cbest_id
            tmp = copy.deepcopy(self.__population[cbest_id])
            self.__population[bNP:] = self.__reinitialize(sNP, problem)
            self.__F[bNP:] = self.__Finit
            self.__Cr[bNP:] = self.__CRinit
            self.__cost[bNP:] = 1e15
            self.__cbest = cbest
            self.__cbest_id = cbest_id
            self.__population[cbest_id] = tmp
            self.__cost[cbest_id] = cbest

        if self.__cbest_id < bNP:
            self.__cCopy += 1
            self.__cost[bNP] = self.__cbest
            self.__population[bNP] = self.__population[self.__cbest_id]
            self.__cbest_id = bNP

        for i in range(bNP // sNP):

            r1 = self.rng.randint(sNP, size=sNP) + bNP
            count = 0
            duplicate = np.where(r1 == (np.arange(sNP) + bNP))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r1[duplicate] = self.rng.randint(sNP, size=duplicate.shape[0]) + bNP
                duplicate = np.where(r1 == (np.arange(sNP) + bNP))[0]
                count += 1

            r2 = self.rng.randint(sNP, size=sNP) + bNP
            count = 0
            duplicate = np.where((r2 == (np.arange(sNP) + bNP)) + (r2 == r1))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r2[duplicate] = self.rng.randint(sNP, size=duplicate.shape[0]) + bNP
                duplicate = np.where((r2 == (np.arange(sNP) + bNP)) + (r2 == r1))[0]
                count += 1

            r3 = self.rng.randint(sNP, size=sNP) + bNP
            count = 0
            duplicate = np.where((r3 == (np.arange(sNP) + bNP)) + (r3 == r1) + (r3 == r2))[0]
            while duplicate.shape[0] > 0 and count < 25:
                r3[duplicate] = self.rng.randint(sNP, size=duplicate.shape[0]) + bNP
                duplicate = np.where((r3 == (np.arange(sNP) + bNP)) + (r3 == r1) + (r3 == r2))[0]
                count += 1

            SF, SCr, df, age = __mutate_cross_select(r1, r2, r3, SF, SCr, df, age, big=False)
            self.__FEs += sNP

        # update and record information for NL-SHADE-RSP and reduce population
        self.gbest = np.min(self.__cost)
        if self.__FEs - self.__NP <= 0.25 * self.__MaxFEs <= self.__FEs or self.__FEs - self.__NP <= 0.5 * self.__MaxFEs <= self.__FEs or self.__FEs - self.__NP <= 0.75 * self.__MaxFEs <= self.__FEs:
            self.__bNP //= 2
            self.__population = self.__population[self.__bNP:]
            self.__cost = self.__cost[self.__bNP:]
            self.__F = self.__F[self.__bNP:]
            self.__Cr = self.__Cr[self.__bNP:]
            self.__NP = self.__bNP + self.__sNP
            self.__cbest_id = np.argmin(self.__cost)

        if self.__FEs >= self.log_index * self.log_interval:
            self.log_index += 1
            self.cost.append(self.gbest)
            
        # 取消早停
        # if problem.optimum is None:
        #     return False
        # else:
        #     return self.gbest <= 1e-8 

    def run_episode(self, problem):
        """
        # Introduction
        Executes a single optimization episode for the given problem, managing population initialization, iterative updates, and result logging. Optionally collects and returns full meta-data for analysis.
        # Args:
        - problem (object): The optimization problem instance to be solved. Must provide necessary interfaces for population initialization and evaluation.
        # Returns:
        - dict: A dictionary containing:
            - 'cost' (list): The cost history or best cost found during the episode.
            - 'fes' (int): The number of function evaluations performed.
            - 'metadata' (dict, optional): Contains 'X' (list of solutions) and 'Cost' (list of costs) if `full_meta_data` is enabled.
        # Notes:
        - The method resets and initializes the population at the start of each episode.
        - Iteratively updates the population until the maximum number of function evaluations is reached.
        - Logs the best solution and optionally collects detailed meta-data for further analysis.
        """
        
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []
        self.__init_population(problem)
        
        while self.__FEs < self.__MaxFEs:
              self.__update(problem)

        if len(self.cost) >= self.__n_logpoint + 1:
            self.cost[-1] = self.gbest
        else:
            self.cost.append(self.gbest)
            
        results = {'cost': self.cost, 'fes': self.__FEs}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata
        return results
