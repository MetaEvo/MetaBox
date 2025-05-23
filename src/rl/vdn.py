from typing import Tuple
import torch
import math, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from .utils import *
from .basic_agent import Basic_Agent


from ..environment.parallelenv.parallelenv import ParallelEnv

class VDN_Agent(Basic_Agent):
    """
    # Introduction
    The `VDN_Agent` class implements a Value Decomposition Network (VDN) agent for multi-agent reinforcement learning. This agent is designed to handle cooperative multi-agent environments by decomposing the joint action-value function into individual agent value functions. It supports experience replay, target networks, epsilon-greedy exploration, and parallelized environments. The class provides methods for training, action selection, and evaluation.

    # Original paper
    "[**Value-Decomposition Networks For Cooperative Multi-Agent Learning**](https://arxiv.org/abs/1706.05296)."

    # Args
    - `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    - `network` (dict): A dictionary of neural networks used by the agent, where keys are network names and values are the corresponding network objects.
    - `learning_rates` (float): Learning rate or a list of learning rates for the optimizer(s).

    # Attributes
    - `n_agent` (int): Number of agents in the environment.(default: 4)
    - `n_act` (int): Number of actions available to each agent.(default: 4)
    - `available_action` (list): List of available actions for each agent.(default: 4)
    - `memory_size` (int): Size of the replay buffer.(default: 10000)
    - `warm_up_size` (int): Number of experiences required in the replay buffer before training starts.(default: 1000)
    - `gamma` (float): Discount factor for future rewards.(default: 0.99)
    - `epsilon` (float): Epsilon value for epsilon-greedy exploration.(default: 0.5)
    - `epsilon_start` (float): Initial epsilon value for exploration.(default: 1)
    - `epsilon_end` (float): Final epsilon value for exploration.(default: 0.1)
    - `epsilon_decay_steps` (int): Number of steps for epsilon decay.(default: 10000)
    - `max_grad_norm` (float): Maximum gradient norm for gradient clipping.(default: 10.0)
    - `batch_size` (int): Batch size for training.(default: 64)
    - `chunk_size` (int): Chunk size for sampling trajectories from the replay buffer.(default: 1)
    - `update_iter` (int): Number of update iterations per training step.(default: 10)
    - `device` (str): Device used for computation (e.g., 'cpu' or 'cuda').
    - `replay_buffer` (MultiAgent_ReplayBuffer): Replay buffer for storing experiences.
    - `network` (list): List of network names used by the agent.
    - `optimizer` (torch.optim.Optimizer): Optimizer for training the networks.(default: Adam)
    - `criterion` (torch.nn.Module): Loss function used for training.(default: MSELoss)
    - `learning_time` (int): Counter for the number of training steps.
    - `cur_checkpoint` (int): Counter for the current checkpoint index.

    # Methods
    - `set_network(networks: dict, learning_rates: float)`: Sets up the networks, optimizer, and loss function for the agent.
    - `get_step() -> int`: Returns the current training step.
    - `update_setting(config)`: Updates the agent's configuration and resets training-related attributes.
    - `get_action(state, epsilon_greedy=False) -> np.ndarray`: Selects an action based on the current state and exploration strategy.
    - `train_episode(...)`: Trains the agent for one episode in a parallelized environment.
    - `rollout_episode(env, seed=None, required_info={}) -> dict`: Executes a single episode in the environment and returns the results.
    - `log_to_tb_train(...)`: Logs training metrics and information to TensorBoard.
    """

    def __init__(self, config, networks, learning_rates):
        """
        Initializes the VDN agent with the given configuration, networks, and learning rates.Store the initial agent in the checkpoint directory.

        # Args:
        - config: Configuration object containing all necessary parameters for the experiment.
        - networks (dict): A dictionary of neural networks used by the agent.
        - learning_rates (float): Learning rate for the optimizer.
        """
        super().__init__(config)
        self.config = config

        # define parameters
        self.n_agent = self.config.n_agent
        self.n_act = self.config.n_act
        self.available_action = self.config.available_action
        self.memory_size = self.config.memory_size
        self.warm_up_size = self.config.warm_up_size
        self.gamma = self.config.gamma
        self.epsilon_start = self.config.epsilon_start
        self.epsilon_end = self.config.epsilon_end
        self.epsilon_decay_steps = self.config.epsilon_decay_steps
        self.epsilon = self.epsilon_start
        self.max_grad_norm = self.config.max_grad_norm
        self.batch_size = self.config.batch_size
        self.chunk_size = self.config.chunk_size
        self.update_iter = self.config.update_iter
        self.device = self.config.device

        self.replay_buffer = MultiAgent_ReplayBuffer(self.memory_size)
        self.set_network(networks, learning_rates)

        # figure out the actor network
        # self.model = None
        # assert hasattr(self, 'model')

        # # figure out the optimizer
        # assert hasattr(torch.optim, self.config.optimizer)
        # self.optimizer = eval('torch.optim.' + self.config.optimizer)(
        #     [{'params': self.model.parameters(), 'lr': self.config.lr_model}])
        # # figure out the lr schedule
        # assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        # self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer,
        #                                                                                  self.config.lr_decay,
        #                                                                                  last_epoch=-1, )

        # assert hasattr(torch.nn, self.config.criterion)
        # self.criterion = eval('torch.nn.' + self.config.criterion)()

        # self.replay_buffer = MultiAgent_ReplayBuffer(self.memory_size)

        # # move to device
        # self.model.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def set_network(self, networks: dict, learning_rates: float):
        """
        Sets up the networks, optimizer, and loss function for the agent.

        # Args:
        - networks (dict): A dictionary of neural networks used by the agent.
        - learning_rates (float): Learning rate for the optimizer.

        # Raises:
        - ValueError: If the length of the learning rates list does not match the number of networks.
        """
        Network_name = []
        if networks:
            for name, network in networks.items():
                Network_name.append(name)
                setattr(self, name, network)  # Assign each network in the dictionary to the class instance
        self.network = Network_name

        assert hasattr(self, 'model')  # Ensure that 'model' is set as an attribute of the class
        self.target_model = copy.deepcopy(self.model).to(self.device)
        if isinstance(learning_rates, (int, float)):
            learning_rates = [learning_rates] * len(networks)
        elif len(learning_rates) != len(networks):
            raise ValueError("The length of the learning rates list must match the number of networks!")

        all_params = []
        for id, network_name in enumerate(networks):
            network = getattr(self, network_name)
            all_params.append({'params': network.parameters(), 'lr': learning_rates[id]})

        assert hasattr(torch.optim, self.config.optimizer)
        self.optimizer = eval('torch.optim.' + self.config.optimizer)(all_params)

        assert hasattr(torch.nn, self.config.criterion)
        self.criterion = eval('torch.nn.' + self.config.criterion)()

        for network_name in networks:
            getattr(self, network_name).to(self.device)

    def get_step(self):
        """
        Returns the current training step.

        #Returns:
        - int: The current training step.
        """
        return self.learning_time


    def update_setting(self, config):
        """
        Updates the agent's configuration and resets training-related attributes.

        # Args:
        - config: Configuration object containing updated parameters.
        """
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    def get_action(self, state, epsilon_greedy = False):
        """
        Selects an action based on the current state and exploration strategy.

        # Args:
        - state (torch.Tensor): The current state.
        - epsilon_greedy (bool): Whether to use epsilon-greedy exploration.

        # Returns:
        - np.ndarray: The selected action(s).
        """
        state = torch.tensor(state, dtype = torch.float64).to(self.device)
        with torch.no_grad():
            Q_list = self.model(state)
        action = np.zeros((len(state), self.n_agent), dtype = int)
        if epsilon_greedy and np.random.rand() < self.epsilon:
            for i in range(self.n_agent):
                action[:, i] = np.random.randint(low = 0, high = self.available_action[i], size = len(state))
        else:
            for i in range(self.n_agent):
                action[:, i] = torch.argmax(Q_list[:, i, :self.available_action[i]], -1).detach().cpu().numpy().astype(int)
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
        Trains the agent for one episode in a parallelized environment.

        # Args:
        - envs: List of environments for training.
        - seeds: Seeds for reproducibility.
        - para_mode (str): Parallelization mode for the environments.
        - compute_resource (dict): Resources for computation (e.g., CPUs, GPUs).
        - tb_logger: TensorBoard logger for logging training metrics.
        - required_info (dict): Additional information required from the environment.

        # Returns:
        - tuple: A boolean indicating whether training has ended and a dictionary with training metrics.
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
            state = torch.FloatTensor(state)
        except:
            pass

        _R = torch.zeros(len(env))
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            self.epsilon = max(self.epsilon_end, \
                               self.epsilon_start - (self.epsilon_start - self.epsilon_end) * (self.learning_time / self.epsilon_decay_steps))
            action = self.get_action(state = state, epsilon_greedy = True)
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward[:, 0]
            _reward.append(torch.FloatTensor(reward[:, 0]))
            # store info
            # convert next_state into tensor
            for s, a, r, ns, d in zip(state.cpu().numpy(), action, reward, next_state, is_end):
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = torch.FloatTensor(next_state).to(self.device)
            except:
                state = copy.deepcopy(next_state)
            # begin update
            if len(self.replay_buffer) >= self.warm_up_size:
                for _ in range(self.update_iter):
                    batch_obs, batch_action, batch_reward, batch_next_obs, batch_done \
                        = self.replay_buffer.sample_chunk(self.batch_size, self.chunk_size)
                    loss = 0
                    for step_i in range(self.chunk_size):
                        q_out = self.model(batch_obs[:, step_i, :, :].to(self.device))
                        q_a = q_out.gather(2, batch_action[:, step_i, :].unsqueeze(-1).long().to(self.device)).squeeze(-1)
                        sum_q = q_a.sum(dim = 1, keepdims = True)
                        max_q_prime = self.target_model(batch_next_obs[:, step_i, :, :].to(self.device))
                        max_q_prime = max_q_prime.max(dim = 2)[0].squeeze(-1)
                        target_q = batch_reward[:, step_i, :].sum(dim = 1, keepdims = True).to(self.device)
                        target_q += self.gamma * max_q_prime.sum(dim = 1, keepdims = True) * (1 - batch_done[:, step_i].to(self.device))

                        loss += self.criterion(sum_q, target_q.detach())
                    loss = loss / self.chunk_size
                    _loss.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                    self.optimizer.step()
                    self.learning_time += 1

                    if self.config.target_update_interval is not None and self.learning_time % self.config.target_update_interval == 0:
                        self.target_model.load_state_dict(self.model.state_dict())

                    if not self.config.no_tb:
                        self.log_to_tb_train(tb_logger, self.learning_time,
                                             grad_norms,
                                             loss,
                                             _R, _reward,
                                             q_out, max_q_prime)
                    if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                        self.cur_checkpoint += 1

                    if self.learning_time >= self.config.max_learning_step:
                        _Rs = _R.detach().numpy().tolist()
                        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time}
                        env_cost = env.get_env_attr('cost')
                        return_info['gbest'] = env_cost[-1]
                        return_info['loss'] = _loss
                        for key in required_info.keys():
                            return_info[key] = env.get_env_attr(required_info[key])
                        env.close()
                        return self.learning_time >= self.config.max_learning_step, return_info

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time}
        env_cost = env.get_env_attr('cost')
        return_info['gbest'] = env_cost[-1]
        return_info['loss'] = _loss

        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed = None,
                        required_info = {}):
        """
        Executes a single episode in the environment without training.

        # Args:
        - env: The environment for the rollout.
        - seed (int, optional): Seed for reproducibility.
        - required_info (dict): Additional information required from the environment.

        # Returns:
        - dict: A dictionary containing episode results such as return, cost, and metadata.
        """
        if hasattr(self.config,required_info):
            required_info = self.config.required_info
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                except:
                    state = [state]
                action = self.get_action(state = state, epsilon_greedy = True)
                action = action[0]
                state, reward, is_done, info = env.step(action)
                R += reward[0]
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': R}
            if self.config.full_meta_data:
                env_metadata = env.get_env_attr('metadata')
                results['metadata'] = env_metadata
            for key in required_info.keys():
                results[key] = env.get_env_attr(required_info[key])
            return results

    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms,
                        loss,
                        Return, Reward,
                        predict_Q, target_Q,
                        extra_info = {}):
        """
        Logs training metrics to TensorBoard.

        # Args:
        - tb_logger: TensorBoard logger for logging training metrics.
        - mini_step (int): Current mini-batch step.
        - grad_norms (tuple): Gradient norms for the networks.
        - loss (torch.Tensor): Training loss.
        - Return (torch.Tensor): Episode return.
        - Reward (torch.Tensor): Target reward.
        - predict_Q (torch.Tensor): Predicted Q-values.
        - target_Q (torch.Tensor): Target Q-values.
        - extra_info (dict): Additional information to log.
        """

        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # learning rate
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'learnrate/{network_name}', self.optimizer.param_groups[id]['lr'], mini_step)
        #
        # # grad and clipped grad
        grad_norms, grad_norms_clipped = grad_norms
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'grad/{network_name}', grad_norms[id], mini_step)
            tb_logger.add_scalar(f'grad_clipped/{network_name}', grad_norms_clipped[id], mini_step)

        # loss
        tb_logger.add_scalar('loss', loss.item(), mini_step)

        # Q
        for i in range(self.n_agent):
            tb_logger.add_scalar(f"Q/action_{i}", predict_Q[:, i].mean().item(), mini_step)
            tb_logger.add_scalar(f"Q/action_{i}_target", target_Q[:, i].mean().item(), mini_step)

        # train metric
        avg_reward = torch.stack(Reward).mean().item()
        max_reward = torch.stack(Reward).max().item()
        tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)
        tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
        tb_logger.add_scalar('train/max_reward', max_reward, mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)
