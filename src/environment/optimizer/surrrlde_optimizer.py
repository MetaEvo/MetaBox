import torch
import numpy as np
from collections import deque
from .learnable_optimizer import Learnable_Optimizer
from ..problem.SOO.COCO_BBOB.bbob_surrogate import bbob_surrogate_model

class SurrRLDE_Optimizer(Learnable_Optimizer):
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
		Initializes the optimizer with the given configuration, setting up algorithm parameters, population attributes, and logging controls.
		# Args:
		- config (object): Configuration object.
		    - The Attributes needed for the surrrlde optimizer:
			    - device (str): The device to be used for computation (e.g., 'cpu', 'cuda').
				- maxFEs (int): The maximum number of function evaluations.
				- upperbound (float): The upper bound for the optimization problem.
				- log_interval (int): The interval for logging progress.
				- n_logpoint (int): The number of log points.
				- full_meta_data (bool): Flag to indicate whether to store full meta data.
	
		# Built-in Attribute:
		- `self.F` (float): The mutation factor for DE.Default is 0.5.
  		- `self.Cr` (float): The crossover probability for DE. Default is 0.7.
   		- `self.pop_size` (int): The size of the population. Default is 100.
		- `self.maxFEs` (int): The maximum number of function evaluations. 
		- `self.ub` (float): The upper bound for the optimization problem.
		- `self.lb` (float): The lower bound for the optimization problem.
		- `self.population` (torch.Tensor): The current population of solutions.
		- `self.fitness` (torch.Tensor): The fitness values of the current population.
		- `self.pop_cur_best` (torch.Tensor): The best solution in the current population.
		- `self.fit_cur_best` (torch.Tensor): The fitness value of the best solution in the current population.
		- `self.pop_history_best` (torch.Tensor): The best solution found so far.Default is None.
		- `self.fit_history_best` (torch.Tensor): The fitness value of the best solution found so far.Default is None.
 		- `self.fit_init_best` (torch.Tensor): The initial best fitness value.Default is None.
		- `self.improved_gen` (int): The number of generations since the last improvement.Default is 0.
		- `self.fes` (int): The number of function evaluations used.Default is None.
		- `self.cost` (list): The cost history of the best solution found so far.Default is None.
		- `self.cur_logpoint` (int): The current log point.Default is None.
		# Returns:
		- None
		"""
     
		super().__init__(config)

		config.F = 0.5
		config.Cr = 0.7
		config.NP = 100
		self.device = config.device
		self.config = config

		self.F = config.F
		self.Cr = config.Cr
		self.pop_size = config.NP
		self.maxFEs = config.maxFEs
		self.ub = config.upperbound
		self.lb = -config.upperbound

		self.population = None
		self.fitness = None
		self.pop_cur_best = None
		self.fit_cur_best = None
		self.pop_history_best = None
		self.fit_history_best = None
		self.fit_init_best = None

		self.improved_gen = 0

		self.fes = None  # record the number of function evaluations used
		self.cost = None
		self.cur_logpoint = None  # record the current logpoint
		self.log_interval = config.log_interval

	def __str__(self):
		"""
		Returns a string representation of the SurrRLDE_Optimizer instance.
		# Returns:
		- str: The name of the optimizer, "SurrRLDE_Optimizer".
		"""
		return "SurrRLDE_Optimizer"

	def get_state(self, problem):
		"""
		# Introduction
		Computes a 9-dimensional state vector representing various statistics and progress indicators of the optimizer's current population and fitness landscape.
		# Args:
		- problem: The optimization problem instance (not directly used in this method, but may be required for interface compatibility).
		# Returns:
		- torch.Tensor: A 1D tensor of length 9 containing the following state features:
			1. Mean pairwise Euclidean distance between individuals in the population.
			2. Mean Euclidean distance between each individual and the current best individual.
			3. Mean Euclidean distance between each individual and the historical best individual.
			4. Mean Euclidean distance between current fitness values and historical best fitness values.
			5. Mean Euclidean distance between current fitness values and current best fitness value.
			6. Standard deviation of the current fitness values.
			7. Normalized remaining function evaluations: (maxFEs - fes) / maxFEs.
			8. Number of generations since last improvement in best fitness.
			9. Binary indicator (1 if current best fitness improved over historical best, else 0).
		# Built-in Attribute:
		- Uses internal attributes such as `self.population`, `self.fitness`, `self.pop_cur_best`, `self.pop_history_best`, `self.fit_cur_best`, `self.fit_history_best`, `self.maxFEs`, `self.fes`, and `self.improved_gen`.
		# Raises:
		- None explicitly, but assumes all internal attributes are properly initialized and shaped.
		"""
     
		state = torch.zeros(9)
		# state 1
		diff = self.population.unsqueeze(0) - self.population.unsqueeze(1)
		distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
		state[0] = torch.sum(distances) / (self.population.shape[0] * (self.population.shape[0] - 1))

		# state 2
		diff = self.population - self.pop_cur_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[1] = torch.sum(distances) / (self.population.shape[0])

		# state 3
		diff = self.population - self.pop_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=1))
		state[2] = torch.sum(distances) / (self.population.shape[0])

		# state 4
		diff = self.fitness - self.fit_history_best
		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 5
		diff = self.fitness - self.fit_cur_best
		# print

		distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
		state[3] = torch.sum(distances) / (self.fitness.shape[0])

		# state 6 std(y)
		state[5] = torch.std(self.fitness)

		# state 7 (T - t)/T
		state[6] = (self.maxFEs - self.fes) / self.maxFEs

		# state 8
		if self.fit_cur_best < self.fit_history_best:
			self.improved_gen = 0
		else:
			self.improved_gen += 1

		state[7] = self.improved_gen

		# state 9 bool
		if self.fit_cur_best < self.fit_history_best:
			state[8] = 1
		else:
			state[8] = 0
		return state

	def init_population(self, problem):
		"""
		# Introduction
		Initializes the population for the optimizer based on the provided problem definition, evaluates the initial fitness, and sets up tracking for the best solutions and meta-data.
		# Args:
		- problem (object): The optimization problem object which has attributes `dim`, `ub`, `lb`, and methods `eval()`.
		# Built-in Attribute:
		- self.dim (int): Dimensionality of the problem.
		- self.rng_torch (torch.Generator): Random number generator for torch, set according to device.
		- self.population (torch.Tensor): The initialized population of candidate solutions.
		- self.fitness (torch.Tensor): Fitness values of the population.
		- self.pop_cur_best (torch.Tensor): Current best solution in the population.
		- self.pop_history_best (torch.Tensor): Historical best solution found.
		- self.fit_init_best (torch.Tensor): Initial best fitness value.
		- self.fit_cur_best (torch.Tensor): Current best fitness value.
		- self.fit_history_best (torch.Tensor): Historical best fitness value.
		- self.fes (int): Number of function evaluations performed.
		- self.cost (list): List of best cost values per generation.
		- self.cur_logpoint (int): Current logpoint for logging.
		- self.meta_X (list, optional): Meta-data of populations if `full_meta_data` is enabled.
		- self.meta_Cost (list, optional): Meta-data of costs if `full_meta_data` is enabled.
		# Returns:
		- state (Any): The current state of the optimizer, as returned by `self.get_state(problem)`.
		# Raises:
		- AttributeError: If the `problem` object does not have required attributes or methods.
		- TypeError: If the fitness evaluation returns an unexpected type.
		"""
     
		self.dim = problem.dim
		self.rng_torch = self.rng_cpu
		if self.device != "cpu":
			self.rng_torch = self.rng_gpu

		self.population = (torch.rand(self.pop_size, self.dim, generator = self.rng_torch, device = self.device)
						   * (problem.ub - problem.lb) + problem.lb)
		#(-5,5)
		self.population = self.population.to(self.device)

		if isinstance(problem, bbob_surrogate_model):
			# print(self.population.clone().to(self.device))
			self.fitness = problem.eval(self.population.clone().to(self.device))

		else:
			if problem.optimum is None:
				self.fitness = problem.eval(self.population.clone().cpu().numpy())
			else:
				self.fitness = problem.eval(self.population.clone().cpu().numpy()) - problem.optimum

		if isinstance(self.fitness, np.ndarray):
			self.fitness = torch.from_numpy(self.fitness).to(self.device)
		if self.fitness.shape == (self.pop_size,):
			self.fitness = self.fitness.unsqueeze(1)

		self.pop_cur_best = self.population[torch.argmin(self.fitness)].clone()
		self.pop_history_best = self.population[torch.argmin(self.fitness)].clone()


		self.fit_init_best = torch.min(self.fitness).clone()
		self.fit_cur_best = torch.min(self.fitness).clone()
		self.fit_history_best = torch.min(self.fitness).clone()

		self.fes = self.pop_size
		self.cost = [self.fit_cur_best.clone().cpu().item()]  # record the best cost of first generation
		self.cur_logpoint = 1  # record the current logpoint
		state = self.get_state(problem)
		if self.config.full_meta_data:
			self.meta_X = [self.population.clone().cpu().numpy()]
			self.meta_Cost = [self.fitness.clone().cpu().numpy()]

		return state

	def update(self, action, problem):
		"""
		# Introduction
		Updates the optimizer's state based on the selected action and the given problem instance. 
		This includes mutation and crossover operations, fitness evaluation, population update, 
		reward calculation, logging, and meta-data collection.
		# Args:
		- action (int): An integer representing the chosen mutation strategy and scaling factor (F).
		- problem (object): The problem instance to be optimized. Must provide an `eval` method and may have an `optimum` attribute.
		# Returns:
		- next_state (Any): The next state representation after the update, as returned by `get_state(problem)`.
		- reward (float): The reward signal computed based on improvement in best fitness.
		- is_done (bool): Whether the optimization process has reached its maximum number of function evaluations.
		- info (dict): Additional information (currently empty).
		# Raises:
		- ValueError: If the provided `action` is not in the valid range (0-14).
		"""

		if action == 0:
			mut_way = 'DE/rand/1'
			self.F = 0.1
		elif action == 1:
			mut_way = 'DE/rand/1'
			self.F = 0.5
		elif action == 2:
			mut_way = 'DE/rand/1'
			self.F = 0.9
		elif action == 3:
			mut_way = 'DE/best/1'
			self.F = 0.1
		elif action == 4:
			mut_way = 'DE/best/1'
			self.F = 0.5
		elif action == 5:
			mut_way = 'DE/best/1'
			self.F = 0.9
		elif action == 6:
			mut_way = 'DE/current-to-rand'
			self.F = 0.1
		elif action == 7:
			mut_way = 'DE/current-to-rand'
			self.F = 0.5
		elif action == 8:
			mut_way = 'DE/current-to-rand'
			self.F = 0.9
		elif action == 9:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.1
		elif action == 10:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.5
		elif action == 11:
			mut_way = 'DE/current-to-pbest'
			self.F = 0.9
		elif action == 12:
			mut_way = 'DE/current-to-best'
			self.F = 0.1
		elif action == 13:
			mut_way = 'DE/current-to-best'
			self.F = 0.5
		elif action == 14:
			mut_way = 'DE/current-to-best'
			self.F = 0.9
		else:
			raise ValueError(f'action error: {action}')

		mut_population = self.mutation(mut_way)
		crossover_population = self.crossover(mut_population)

		if isinstance(problem, bbob_surrogate_model):
			temp_fit = problem.eval(crossover_population.clone().to(self.device))
		else:
			if problem.optimum is None:
				temp_fit = problem.eval(crossover_population.clone().cpu().numpy())
			else:
				temp_fit = problem.eval(crossover_population.clone().cpu().numpy()) - problem.optimum

		if isinstance(temp_fit, np.ndarray):
			temp_fit = torch.from_numpy(temp_fit).to(self.device)
		if temp_fit.shape == (self.pop_size,):
			temp_fit = temp_fit.unsqueeze(1)

		for i in range(self.pop_size):
			if temp_fit[i].item() < self.fitness[i].item():
				self.fitness[i] = temp_fit[i]
				self.population[i] = crossover_population[i]

		reward = self.fit_history_best > torch.min(self.fit_history_best, torch.min(self.fitness).clone())
		reward = reward / 200
		best_index = torch.argmin(self.fitness)

		self.pop_cur_best = self.population[best_index].clone()
		self.fit_cur_best = self.fitness[best_index].clone()

		next_state = self.get_state(problem)

		if self.fit_cur_best < self.fit_history_best:
			self.fit_history_best = self.fit_cur_best.clone()
			self.pop_history_best = self.pop_cur_best.clone()

		is_done = (self.fes >= self.maxFEs)

		self.fes += self.pop_size

		if self.fes >= self.cur_logpoint * self.config.log_interval:
			self.cur_logpoint += 1
			self.cost.append(self.fit_history_best.clone().cpu().item())

		if self.config.full_meta_data:
			self.meta_X.append(self.population.clone().cpu().numpy())
			self.meta_Cost.append(self.fitness.clone().cpu().numpy())

		if is_done:
			if len(self.cost) >= self.config.n_logpoint + 1:
				self.cost[-1] = self.fit_history_best.clone().cpu().item()
			else:
				while len(self.cost) < self.__config.n_logpoint + 1:
					self.cost.append(self.fit_history_best.clone().cpu().item())

		info = {}
		return next_state, reward.item(), is_done, info

	def mutation(self, mut_way):
		"""
		# Introduction
		Applies various Differential Evolution (DE) mutation strategies to the current population and returns the mutated population.
		# Args:
		- mut_way (str): The mutation strategy to use. Supported values are:
			- 'DE/rand/1'
			- 'DE/best/1'
			- 'DE/current-to-rand'
			- 'DE/current-to-pbest'
			- 'DE/current-to-best'
		# Built-in Attribute:
		- self.population (torch.Tensor): The current population of candidate solutions.
		- self.pop_size (int): The number of individuals in the population.
		- self.F (float): The mutation scaling factor.
		- self.lb (float or torch.Tensor): The lower bound(s) for the variables.
		- self.ub (float or torch.Tensor): The upper bound(s) for the variables.
		- self.pop_cur_best (torch.Tensor): The current best individual in the population.
		- self.fitness (torch.Tensor): The fitness values of the population.
		- self.device (torch.device): The device on which tensors are allocated.
		- self.rng_torch (torch.Generator): The random number generator for torch operations.
		# Returns:
		- torch.Tensor: The mutated population tensor, with the same shape as `self.population`.
		# Raises:
		- ValueError: If `mut_way` is not one of the supported mutation strategies.
		"""
		
		mut_population = torch.zeros_like(self.population, device=self.device)

		if mut_way == 'DE/rand/1':

			r = self.generate_random_int(self.pop_size, 3)  # Shape: [pop_size, 3]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]

			v = a + self.F * (b - c)

			v = torch.clamp(v, min=self.lb, max=self.ub)
			mut_population = v

		elif mut_way == 'DE/best/1':
			r = self.generate_random_int(self.pop_size, 2)  # Shape: [pop_size, 2]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			v = self.pop_cur_best + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-rand':
			r = self.generate_random_int(self.pop_size, 3)  # Shape: [pop_size, 3]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			v = self.population + self.F * (a - self.population) + self.F * (b - c)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-pbest':
			p = 0.1
			p_num = max(1, int(p * self.pop_size))
			sorted_indices = torch.argsort(self.fitness.clone().flatten())
			pbest_indices = sorted_indices[:p_num]
			r = self.generate_random_int(self.pop_size, 2)  # Shape: [pop_size, 2]

			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]

			pbest_index = pbest_indices[torch.randint(0, p_num, (self.pop_size,), generator = self.rng_torch, device = self.device)]
			pbest = self.population[pbest_index]

			v = self.population + self.F * (pbest - self.population) + self.F * (a - b)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		elif mut_way == 'DE/current-to-best':
			r = self.generate_random_int(self.pop_size, 4)  # Shape: [pop_size, 4]
			a = self.population[r[:, 0]]
			b = self.population[r[:, 1]]
			c = self.population[r[:, 2]]
			d = self.population[r[:, 3]]
			v = self.population + self.F * (self.pop_cur_best - self.population) + self.F * (a - b) + self.F * (c - d)
			v = torch.clamp(v, self.lb, self.ub)
			mut_population = v

		else:
			raise ValueError(f'mutation error: {mut_way} is not defined')

		mut_population = torch.tensor(mut_population, device=self.device)
		return mut_population

	def crossover(self, mut_population):
		"""
		# Introduction
		Performs the crossover operation in a differential evolution algorithm, combining the current population with a mutated population to generate a new candidate population.
		# Args:
		- mut_population (torch.Tensor): The mutated population tensor with the same shape as the current population.
		# Built-in Attribute:
		- self.population (torch.Tensor): The current population tensor.
		- self.pop_size (int): The number of individuals in the population.
		- self.dim (int): The dimensionality of each individual.
		- self.Cr (float): The crossover probability.
		- self.rng_torch (torch.Generator): The random number generator for reproducibility.
		- self.device (torch.device): The device on which tensors are allocated.
		# Returns:
		- torch.Tensor: The new population tensor after crossover, with the same shape as the input population.
		# Raises:
		- None
		"""
	
		crossover_population = self.population.clone()
		for i in range(self.pop_size):

			select_dim = torch.randint(0, self.dim, (1,), generator = self.rng_torch, device = self.device)
			for j in range(self.dim):
				if torch.rand(1, generator = self.rng_torch, device = self.device) < self.Cr or j == select_dim:
					crossover_population[i][j] = mut_population[i][j]
		return crossover_population

	def generate_random_int(self, NP: int, cols: int) -> torch.Tensor:
		"""
		# Introduction
		Generates a random integer tensor of shape (NP, cols) where each row contains unique indices not equal to the row index, suitable for population-based optimization algorithms.
		# Args:
		- NP (int): The population size, determines the number of rows in the output tensor.
		- cols (int): The number of columns in the output tensor, typically the number of unique indices required per row.
		# Returns:
		- torch.Tensor: A tensor of shape (NP, cols) containing random integers in the range [0, NP), with the constraint that no row contains its own index.
		# Raises:
		- None
		"""
    
		r = torch.randint(0, NP, (NP, cols), dtype = torch.long, generator = self.rng_torch, device = self.device)  # [NP, 3]

		for i in range(NP):
			while r[i, :].eq(i).any():
				r[i, :] = torch.randint(0, NP, (cols,), dtype = torch.long, generator = self.rng_torch, device = self.device)

		return r
