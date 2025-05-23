from .networks import MLP
from ...rl.ddqn import *


class SurrRLDE(DDQN_Agent):
	"""
	# Introduction
	The paper "Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study" introduces a novel framework, Surr-RLDE, that combines surrogate modeling and reinforcement learning to enhance Meta-Black-Box Optimization (MetaBBO),the authors propose a two-stage learning process: (1) Surrogate learning, where a Kolmogorov-Arnold Network (KAN) is trained using a relative-order-aware loss to accurately approximate objective functions, and (2) Policy learning, where reinforcement learning dynamically configures mutation operators in a Differential Evolution (DE) algorithm. By integrating the surrogate model into policy training, Surr-RLDE significantly reduces evaluation costs while maintaining competitive performance.
	# Original Paper
	"[**Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study**](https://arxiv.org/abs/2503.18060)." The Genetic and Evolutionary Computation Conference (GECCO 2025)
	# Official Implementation
	[Surr-RLDE](https://github.com/MetaEvo/Surr-RLDE)
	# Application Scenario
	single-object optimization problems(SOOP), in this implementation, the built-in DDQN_Agent is used as the parent class, since SurrRLDE is based on DDQN. 
	# Raises:
	None explicitly raised in the provided code, but potential exceptions may occur during tensor operations, environment interactions, or model updates.
	"""
	def __init__(self, config):
		"""
        # Args:
        -config: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
		# Built-in Attributes:
		-config (object): Stores the configuration object.
		-device (str): Device to be used for computation ('cpu' or 'cuda').
		-memory_size (int): Size of the replay buffer.
		-n_act (int): Number of possible actions.
		-epsilon (float): Initial epsilon value for epsilon-greedy policy.
		-gamma (float): Discount factor for future rewards.
		-max_learning_step (int): Maximum number of learning steps.
		-cur_checkpoint (int): Current checkpoint index for saving the model.
		-replay_buffer (ReplayBuffer_torch): Replay buffer for storing experiences.
		-model (MLP): Neural network model for Q-value prediction.
        """

 
		self.config = config
		self.config.state_size = 9
		self.config.n_act = 15
		self.config.lr_model = 1e-5
		self.config.lr_decay = 1
		self.config.batch_size = 512
		self.config.epsilon = 0.5  # 0.5 - 0.05
		self.config.gamma = 0.999
		self.config.target_update_interval = 10
		self.config.memory_size = 2048
		self.config.warm_up_size = config.batch_size
		self.config.net_config = [{'in': config.state_size, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 32, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 32, 'out': config.n_act, 'drop_out': 0, 'activation': 'Sigmoid'}]
		self.device = config.device
		self.memory_size = self.config.memory_size = 2048


		self.config.max_grad_norm = math.inf
		# self.pred_Qnet = MLP(config.net_config).to(self.device)
		# self.target_Qnet = copy.deepcopy(self.pred_Qnet).to(self.device)
		# self.optimizer = torch.optim.AdamW(self.pred_Qnet.parameters(), lr=config.lr)
		# self.criterion = torch.nn.MSELoss()

		self.n_act = config.n_act
		self.epsilon = config.epsilon
		self.gamma = config.gamma
		# self.update_target_steps = config.update_target_steps
		# self.batch_size = config.batch_size
		# self.replay_buffer = ReplayBuffer(config.memory_size, device=self.device, state_dim=self.config.state_size)
		# self.warm_up_size = config.warm_up_size
		self.max_learning_step = config.max_learning_step

		self.cur_checkpoint = 0

		self.config.optimizer = 'AdamW'
		# origin code does not have lr_scheduler
		self.config.lr_scheduler = 'ExponentialLR'
		self.config.criterion = 'MSELoss'
		model = MLP(self.config.net_config).to(self.config.device)

		self.config.agent_save_dir = os.path.join(
			self.config.agent_save_dir,
			self.__str__(),
			self.config.train_name
		)
		super().__init__(self.config, {'model': model}, self.config.lr_model)
		self.replay_buffer = ReplayBuffer_torch(self.memory_size, 9, self.device)

	def __str__(self):
		return "Surr_RLDE"

	def get_epsilon(self, step, start=0.5, end=0.05):
		"""
  		Calculates the epsilon value for epsilon-greedy policy based on the current step.
		-step (int): Current training step.
		-start (float): Starting epsilon value.
		-end (float): Minimum epsilon value.
  		"""
		total_steps = self.config.max_learning_step
		return max(end, start - (start - end) * (step / total_steps))

	def get_action(self, state, epsilon_greedy=False):
		"""
  		Selects an action based on the current state using epsilon-greedy policy.
		-state (array-like): Current state.
		-epsilon_greedy (bool): Whether to use epsilon-greedy policy.
  		"""
		state = torch.Tensor(state).to(self.device)
		self.epsilon = self.get_epsilon(self.learning_time)
		with torch.no_grad():
			Q_list = self.model(state)
		if epsilon_greedy and np.random.rand() < self.epsilon:
			action = np.random.randint(low=0, high=self.n_act, size=len(state))
		else:
			action = torch.argmax(Q_list, -1).detach().cpu().numpy()
		return action


	def train_episode(self,
					  envs,
					  seeds: Optional[Union[int, List[int], np.ndarray]],
					  para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
					  # todo: asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
					  # num_cpus: Optional[Union[int, None]] = 1,
					  # num_gpus: int = 0,
					  compute_resource = {},
					  tb_logger = None,
					  required_info = {}):
		"""
  		Trains the agent for one episode.
		-envs (list): List of environments.
		-seeds (int, list, or np.ndarray): Seeds for environment initialization.
		-para_mode (str): Parallelization mode ('dummy', 'subproc', 'ray', 'ray-subproc').
		-compute_resource (dict): Dictionary specifying computational resources.
		-tb_logger (object): TensorBoard logger for logging training metrics.
		-required_info (dict): Additional information required from the environment.
  		"""
		num_cpus = None
		num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
		if 'num_cpus' in compute_resource.keys():
			num_cpus = compute_resource['num_cpus']
		if 'num_gpus' in compute_resource.keys():
			num_gpus = compute_resource['num_gpus']
		env = ParallelEnv(envs, para_mode, num_cpus = num_cpus, num_gpus = num_gpus)
		env.seed(seeds)
		# params for training
		gamma = self.gamma

		state = env.reset()
		try:
			state = torch.Tensor(state)
		except:
			pass

		_R = torch.zeros(len(env))
		_loss = []
		_reward = []
		# sample trajectory
		while not env.all_done():
			action = self.get_action(state = state, epsilon_greedy = True)

			# state transient
			next_state, reward, is_end, info = env.step(action)
			_R += reward
			_reward.append(torch.Tensor(reward))
			# store info
			# convert next_state into tensor
			try:
				next_state = torch.Tensor(next_state).to(self.device)
			except:
				pass
			for s, a, r, ns, d in zip(state, action, reward, next_state, is_end):

				self.replay_buffer.append(s, a, r, ns, d)
			try:
				state = next_state
			except:
				state = copy.deepcopy(next_state)

			# begin update
			if len(self.replay_buffer) >= self.warm_up_size:
				batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(self.batch_size)
				pred_Vs = self.model(batch_obs.to(self.device))  # [batch_size, n_act]
				action_onehot = torch.nn.functional.one_hot(batch_action.to(self.device), self.n_act)  # [batch_size, n_act]

				_avg_predict_Q = (pred_Vs * action_onehot).mean(0)  # [n_act]
				predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]

				target_output = self.target_model(batch_next_obs.to(self.device))
				_avg_target_Q = batch_reward.to(self.device)[:, None] + (1 - batch_done.to(self.device))[:, None] * gamma * target_output
				target_Q = batch_reward.to(self.device) + (1 - batch_done.to(self.device)) * gamma * target_output.max(1)[0].detach()
				_avg_target_Q = _avg_target_Q.mean(0)  # [n_act]

				self.optimizer.zero_grad()
				loss = self.criterion(predict_Q, target_Q)
				loss.backward()
				grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
				self.optimizer.step()

				_loss.append(loss.item())
				self.learning_time += 1
				if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
					save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
					self.cur_checkpoint += 1

				if self.learning_time % self.target_update_interval == 0:
					for target_parma, parma in zip(self.target_model.parameters(), self.model.parameters()):
						target_parma.data.copy_(parma.data)

				if not self.config.no_tb:
					self.log_to_tb_train(tb_logger, self.learning_time,
										 grad_norms,
										 loss,
										 _R, _reward,
										 _avg_predict_Q, _avg_target_Q)

				if self.learning_time >= self.config.max_learning_step:
					_Rs = _R.detach().numpy().tolist()
					return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
					env_cost = env.get_env_attr('cost')
					return_info['normalizer'] = env_cost[0]
					return_info['gbest'] = env_cost[-1]
					for key in required_info.keys():
						return_info[key] = env.get_env_attr(required_info[key])
					env.close()
					return self.learning_time >= self.config.max_learning_step, return_info

		is_train_ended = self.learning_time >= self.config.max_learning_step
		_Rs = _R.detach().numpy().tolist()
		return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
		env_cost = env.get_env_attr('cost')
		return_info['normalizer'] = env_cost[0]
		return_info['gbest'] = env_cost[-1]
		for key in required_info.keys():
			return_info[key] = env.get_env_attr(required_info[key])
		env.close()

		return is_train_ended, return_info
