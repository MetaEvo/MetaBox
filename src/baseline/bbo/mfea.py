import numpy as np
import copy
from ...environment.optimizer.basic_optimizer import Basic_Optimizer

def SBX(parent1,parent2,n):
    """
    # Introduction
    Performs Simulated Binary Crossover (SBX) on two parent populations to generate two offspring populations. SBX is a genetic algorithm operator used to recombine two parent solutions, producing offspring that inherit characteristics from both parents, with diversity controlled by the distribution index `n`.
    # Args:
    - parent1 (np.ndarray): The first parent population, typically a 2D array where each row represents an individual.
    - parent2 (np.ndarray): The second parent population, with the same shape as `parent1`.
    - n (float): The distribution index that controls the spread of the offspring around the parents. Higher values result in offspring closer to the parents.
    # Returns:
    - Tuple[np.ndarray, np.ndarray]: A tuple containing two offspring populations, each with the same shape as the input parents.
    # Notes:
    - The function assumes that `parent1` and `parent2` are NumPy arrays of the same shape.
    - Offspring values are clipped to the range [0, 1].
    """
    
    pop_cnt = parent1.shape[0]
    beta = [0] * pop_cnt
    for i in range(pop_cnt):
        rand = np.random.random()
        if rand<=0.5:
            beta[i] = (2*rand) ** (1/(n+1))
        if rand>0.5:
            beta[i] = (1/(2-2*rand)) ** (1/(n+1))

    offspring1 = copy.deepcopy(parent1)
    offspring2 = copy.deepcopy(parent2)
    for i in range(pop_cnt):
        offspring1[i] = 1/2 * (parent1[i]+parent2[i]) - 1/2 * beta[i] * (parent2[i]-parent1[i])
        offspring2[i] = 1/2 * (parent1[i]+parent2[i]) + 1/2 * beta[i] * (parent2[i]-parent1[i])
        offspring1[i] = np.clip(offspring1[i],0,1)
        offspring2[i] = np.clip(offspring2[i],0,1)

    return offspring1, offspring2

def gaussian_mutation(parent, mutate_probability, sigma):
    """
    # Introduction
    Applies Gaussian mutation to a parent solution vector, producing an offspring by perturbing each gene with a specified probability.
    # Args:
    - parent (np.ndarray): The parent solution vector to be mutated.
    - mutate_probability (float): The probability of mutating each gene in the parent vector.
    - sigma (float): The standard deviation of the Gaussian distribution used for mutation.
    # Returns:
    - np.ndarray: The mutated offspring vector, with values clipped to the range [0, 1].
    # Notes:
    - Each gene in the parent vector has an independent chance (mutate_probability) to be mutated.
    - Mutated genes are sampled from a normal distribution centered at the original gene value with standard deviation sigma.
    - All gene values are clipped to the range [0, 1] after mutation.
    """
    
    dimension = parent.shape[0]
    offspring = copy.deepcopy(parent)
    for i in range(dimension):
        rand = np.random.random()
        if rand<mutate_probability:
            offspring[i] = np.clip(np.random.normal(parent[i],sigma),0,1)

    return offspring

def polinomial_mutation(parent, mu):
    """
    # Introduction
    Applies polynomial mutation to a parent solution vector, generating an offspring with mutated genes based on a specified distribution index.
    # Args:
    - parent (np.ndarray): The parent solution vector to be mutated. Each element should be within the range [0, 1].
    - mu (float): The distribution index controlling the mutation's spread. Higher values result in smaller mutations.
    # Returns:
    - np.ndarray: The mutated offspring vector, with each gene potentially altered and clipped to the [0, 1] range.
    # Notes:
    - Mutation occurs independently for each gene with a probability of 0.05.
    - The function uses deep copy to avoid modifying the original parent vector.
    """
    
    dim = parent.shape[0]
    offspring = copy.deepcopy(parent)
    for i in range(dim):
        if np.random.random() < 0.05 :
            u = np.random.random()
            if u < 0.5:
                delta = (2 * u) ** (1 / (1 + mu)) - 1
                offspring[i] = offspring[i] + delta * offspring[i]
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (1 + mu))
                offspring[i] = offspring[i] + delta * (1-offspring[i])
            offspring[i] = np.clip(offspring[i],0,1)
    return offspring

class Individual():
    def __init__(self, D_multitask, tasks):
        """
        Initializes an instance for multitask evolutionary optimization.
        # Args:
        - D_multitask (int): The dimensionality of the multitask problem.
        - tasks (list): A list of task objects or definitions to be optimized.
        # Attributes:
        - dim (int): Stores the dimensionality of the multitask problem.
        - tasks (list): Stores the list of tasks.
        - tasks_count (int): The number of tasks.
        - genes (np.ndarray): Randomly initialized gene vector of length `D_multitask`.
        - scalar_fitness (float or None): Placeholder for the individual's scalar fitness value.
        - skill_factor (int or None): Placeholder for the individual's skill factor.
        """
        
        self.dim = D_multitask
        self.tasks = tasks
        self.tasks_count = len(tasks)
        self.genes = np.random.uniform(size=D_multitask)
        self.scalar_fitness = None
        self.skill_factor = None

    def update_evaluate(self):
        """
        # Introduction
        Evaluates the individual's fitness on its assigned task using its current genes.
        # Args:
        None
        # Built-in Attribute:
        - self.skill_factor (int): Index of the task assigned to this individual.
        - self.tasks (List[Task]): List of task objects, each with a `dim` attribute and an `eval` method.
        - self.genes (np.ndarray): The individual's genetic representation.
        # Returns:
        - Tuple[int, np.ndarray]: A tuple containing the skill factor (task index) and the evaluated fitness as a 1D numpy array.
        # Raises:
        - AttributeError: If required attributes (`tasks`, `genes`, or `skill_factor`) are missing or improperly set.
        """
        
        task = self.tasks[self.skill_factor]
        task_genes = self.genes[:task.dim].reshape(1,-1)
        fitness = task.eval(task_genes).reshape(-1)
        return self.skill_factor, fitness

    def first_evaluate(self):
        """
        # Introduction
        Evaluates the fitness of the current individual's genes on all tasks and returns a list of fitness values for each task.
        # Args:
        None
        # Built-in Attribute:
        - self.genes (np.ndarray): The gene representation of the individual.
        - self.tasks (List[Task]): List of task objects, each with a `dim` attribute and an `eval` method.
        - self.tasks_count (int): The number of tasks.
        # Returns:
        - List[np.ndarray]: A list where each element is a 1D numpy array containing the fitness value(s) for the corresponding task.
        # Raises:
        None
        """
        
        factorial_cost_list_j = []
        for j in range(self.tasks_count):
            task_j_genes = self.genes[:self.tasks[j].dim].reshape(1,-1)
            fitness = self.tasks[j].eval(task_j_genes).reshape(-1)
            factorial_cost_list_j.append(fitness)

        return factorial_cost_list_j


class MFEA(Basic_Optimizer):
    """
    # Introduction
    MFEA:Multifactorial Evolution: Toward Evolutionary Multitasking
    Multifactorial Evolution is an emerging evolutionary computing paradigm that aims to achieve evolutionary multitasking. Traditional evolutionary algorithms usually target a single optimization problem, while multifactorial evolution allows a single population to optimize multiple target tasks simultaneously. This method can exploit the correlation between different tasks and improve the overall performance of the algorithm through cross-task knowledge transfer.
    Multifactorial evolution algorithms achieve evolutionary multitasking by simulating the multifactorial characteristics of biological evolution, such as gene expression and inheritance. This adaptive multitasking optimization method can be applied to various complex practical problems, such as production optimization, reinforcement learning, etc.
    # Original Paper
    "[**Multifactorial Evolution: Toward Evolutionary Multitasking**](https://ieeexplore.ieee.org/abstract/document/7161358)." 
    # Official Implementation
    None
    # Application Scenario
    multi-task optimization problems(MTOP)
    """
    def __init__(self, config):
        """
        # Introduction
        Initializes the class with the provided configuration, setting up logging intervals, meta data, and default values for optimization parameters.
        # Args:
        - config (object): Config object.
            - The Attributes needed for the MFEA are the following:
                - log_interval (int): Interval at which logs are recorded, defaulted to config.maxFEs/config.n_logpoint.
                - full_meta_data (bool): Flag indicating whether to use full meta data, defaulted to False.
                - maxFEs (int): Maximum number of function evaluations allowed.
                - n_logpoint (int): Number of log points to record, defaulted to 50.
                
        # Built-in Attributes:
        - log_interval (int): Interval at which logs are recorded, taken from the configuration.
        - full_meta_data (any): Meta data used for the optimization process, from the configuration.
        - total_generation (int): Total number of generations for the optimization, defaulted to 250.
        - cost (any): Placeholder for cost value, initialized as None.
        - _fes (any): Placeholder for function evaluation count, initialized as None.
        - log_index (any): Placeholder for log index, initialized as None.
        """
        
        super().__init__(config)
        self.__config = config
        self.log_interval = config.log_interval
        self.full_meta_data = config.full_meta_data
        self.total_generation = 250

        self.cost = None
        self._fes = None
        self.log_index = None

    def __str__(self):
        """
        Returns a string representation of the MFEA class.
        # Returns:
            str: The name of the class, "MFEA".
        """
        
        return "MFEA"
    
    
    def run_episode(self, mto_tasks):
        """
        # Introduction
        Executes a single episode of the Multifactorial Evolutionary Algorithm (MFEA) for multitask optimization. This function manages the evolutionary process, including population initialization, evaluation, selection, crossover, mutation, and logging of progress for a set of optimization tasks.
        # Args:
        - mto_tasks (object): An object containing the multitask optimization tasks and related methods. It should provide access to the list of tasks, each with its own dimensionality and evaluation function.
        # Built-in Attribute:
        - self.full_meta_data (bool): If True, collects and stores metadata (population and costs) during the run.
        - self._fes (int): Tracks the number of function evaluations performed.
        - self.log_index (int): Index for logging progress at specified intervals.
        - self.log_interval (int): Interval (in function evaluations) at which to log progress.
        - self.__config (object): Configuration object containing parameters such as `maxFEs` (maximum function evaluations) and `n_logpoint` (number of log points).
        - self.total_generation (int): Maximum number of generations to run.
        - self.meta_Cost (list): Stores cost metadata if `full_meta_data` is enabled.
        - self.meta_X (list): Stores population metadata if `full_meta_data` is enabled.
        - self.cost (list): Stores the best fitness values found at each log point.
        - self.rng (np.random.Generator): Random number generator for reproducibility.
        # Returns:
        - dict: A dictionary containing:
            - 'cost' (list): Best fitness values for each task at each log point.
            - 'fes' (int): Total number of function evaluations performed.
            - 'metadata' (dict, optional): If `full_meta_data` is True, includes:
                - 'X' (list): Population genes at each log point.
                - 'Cost' (list): Corresponding costs at each log point.
        # Raises:
        - None explicitly. Assumes that all required attributes and methods are properly defined and that the input `mto_tasks` object is valid.
        """
        
        rmp = 0.3
        population_cnt = 50
        generation = 0
        mu = 2  
        self._fes = 0
        self.log_index = 1
    
        if self.full_meta_data:
            self.meta_Cost = []
            self.meta_X = []

        tasks = mto_tasks.tasks
        task_count = len(tasks)
        D = np.zeros(shape=task_count)
        for i in range(task_count):
            D[i] = tasks[i].dim
        D_multitask = int(np.max(D))

        population = np.array([Individual(D_multitask, tasks) for _ in range(2*population_cnt)])
        factorial_costs = np.full(shape=(2*population_cnt, task_count), fill_value=np.inf)
        factorial_ranks = np.empty(shape=(2*population_cnt, task_count))
        best_fitness = np.full(shape=task_count,fill_value=np.inf)
        
        for i, individual in enumerate(population[:population_cnt]):
            factorial_all_cost = individual.first_evaluate()
            factorial_costs[i] = np.array(factorial_all_cost).reshape(-1)

        for j in range(task_count):
            factorial_cost_j = factorial_costs[:, j]
            index = np.argsort(factorial_cost_j)
            for i, x in enumerate(index):
                factorial_ranks[x, j] = i + 1

        for i in range(population_cnt):
            population[i].scalar_fitness = 1 / np.min(factorial_ranks[i])
            population[i].skill_factor = np.argmin(factorial_ranks[i])
        
        if self.full_meta_data:
            list_pop = [population[i].genes for i in range(0,population_cnt)]
            self.meta_Cost.append(factorial_costs[:population_cnt])
            self.meta_X.append(list_pop)   
        
        self.cost = [copy.deepcopy(best_fitness)]

        done = False
        while not done:
            order = self.rng.permutation(population_cnt)
            count = population_cnt
            factorial_costs[population_cnt:,:] = np.inf
            for i in range(0,population_cnt,2):
                parent1 = population[order[i]]
                parent2 = population[order[i+1]]
                offspring1 = Individual(D_multitask, tasks)
                offspring2 = Individual(D_multitask, tasks)

                if(parent1.skill_factor == parent2.skill_factor or self.rng.random()<rmp):
                    offspring1.genes,offspring2.genes = SBX(parent1.genes,parent2.genes,mu)

                    rand1 = self.rng.random()
                    rand2 = self.rng.random()
                    if rand1 <0.5:
                        offspring1.skill_factor = parent1.skill_factor
                    else:
                        offspring1.skill_factor = parent2.skill_factor

                    if rand2 < 0.5:
                        offspring2.skill_factor = parent1.skill_factor
                    else:
                        offspring2.skill_factor = parent2.skill_factor

                else:
                    offspring1.genes = gaussian_mutation(parent1.genes,0.05,0.5)
                    offspring1.skill_factor = parent1.skill_factor
                        
                    offspring2.genes = gaussian_mutation(parent2.genes,0.05,0.5)
                    offspring2.skill_factor = parent2.skill_factor

                population[count] = offspring1
                population[count+1] = offspring2
                count+=2

            for i, individual in enumerate(population[population_cnt:]):
                j, factorial_cost = individual.update_evaluate()
                factorial_costs[population_cnt + i, j] = factorial_cost

            for j in range(task_count):
                factorial_cost_j = factorial_costs[:,j]
                index = np.argsort(factorial_cost_j)
                for i, x in enumerate(index):
                    factorial_ranks[x,j] = i+1

            for i in range(2 * population_cnt):
                population[i].scalar_fitness = 1 / np.min(factorial_ranks[i])
                population[i].skill_factor = np.argmin(factorial_ranks[i])

            scalar_fitness_list = np.array([individual.scalar_fitness for individual in population])
            select_list = np.argsort(scalar_fitness_list)[::-1]
            population = population[select_list]
            factorial_costs = factorial_costs[select_list]
            factorial_ranks = factorial_ranks[select_list]


            for j in range(task_count):
                best_j = np.argmin(factorial_costs[:, j])
                if (best_fitness[j] > factorial_costs[best_j, j]):
                    best_fitness[j] = factorial_costs[best_j, j]

            self._fes += population_cnt
            generation += 1
            if self._fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(copy.deepcopy(best_fitness))

            done = self._fes >= self.__config.maxFEs or generation >= self.total_generation

            if done:
                if len(self.cost) >= self.__config.n_logpoint + 1:
                    self.cost[-1] = copy.deepcopy(best_fitness)
                else:
                    self.cost.append(copy.deepcopy(best_fitness))
                break    

            if self.full_meta_data: 
                list_pop = [population[i].genes for i in range(0,population_cnt)]
                self.meta_Cost.append(factorial_costs[:population_cnt])
                self.meta_X.append(list_pop)
               
        results = {'cost': self.cost, 'fes': self._fes}

        if self.full_meta_data:
            metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
            results['metadata'] = metadata

        mto_tasks.update_T1()
        return results 