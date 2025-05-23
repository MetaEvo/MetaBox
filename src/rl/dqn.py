import copy
import math
from typing import Optional, Union, Literal, List
import torch.nn.functional as F

from ..environment.parallelenv.parallelenv import ParallelEnv
from .basic_agent import Basic_Agent
from .utils import *
import torch
import numpy as np

class DQN_Agent(Basic_Agent):
    """
    # Introduction
    The `DQN_Agent` class implements a Deep Q-Network (DQN) agent for reinforcement learning. This agent uses experience replay, epsilon-greedy exploration, and gradient clipping to learn optimal policies in a given environment. It supports parallelized environments and provides methods for training, action selection, and evaluation.

    # Original paper
    "[**Playing Atari with Deep Reinforcement Learning**](https://arxiv.org/abs/1312.5602)."

    # Args
    - `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    - `networks` (dict): A dictionary of neural networks used by the agent, with keys as network names (e.g., 'actor', 'critic') and values as the corresponding network instances.
    - `learning_rates` (float): Learning rate for the optimizer.

    # Attributes
    - `gamma` (float): Discount factor for future rewards.(default: 0.8)
    - `n_act` (int): Number of possible actions in the environment.(default: 3)
    - `epsilon` (float): Exploration rate for epsilon-greedy policy.(default: 0.1)
    - `max_grad_norm` (float): Maximum gradient norm for gradient clipping.(default: infinity)
    - `memory_size` (int): Size of the replay buffer.(default: 100)
    - `batch_size` (int): Batch size for training.(default: 64)
    - `warm_up_size` (int): Minimum number of experiences required in the replay buffer before training starts.(default: training batch size)
    - `device` (str): Device to run computations on (e.g., 'cpu' or 'cuda').
    - `replay_buffer` (ReplayBuffer): Replay buffer for storing experiences.
    - `network` (list): List of network names used by the agent.
    - `optimizer` (torch.optim.Optimizer): Optimizer for training the networks.(default: 'AdamW')
    - `criterion` (torch.nn.Module): Loss function used for training.(default: 'MSELoss')
    - `learning_time` (int): Counter for the number of training steps.
    - `cur_checkpoint` (int): Counter for the current checkpoint index.

    # Methods
    - `set_network`(networks, learning_rates): Sets up the neural networks, optimizers, and loss functions for the agent.
    - `update_setting`(config): Updates the agent's configuration and resets training-related attributes.
    - `get_action`(state, epsilon_greedy=False): Selects an action based on the current state using the epsilon-greedy policy.
    - `train_episode`(envs, seeds, para_mode, compute_resource, tb_logger, required_info): Trains the agent for one episode in a parallelized environment.
    - `rollout_episode`(env, seed, required_info): Executes a single rollout episode in the environment and collects results.
    - `log_to_tb_train`(tb_logger, mini_step, grad_norms, loss, Return, Reward, predict_Q, target_Q, extra_info): Logs training metrics to TensorBoard.

    """
    def __init__(self, config, network: dict, learning_rates: float):
        """
        Initializes the DQN agent with the given configuration, networks, and learning rates.Store the initial agent in the checkpoint directory.

        # Args:
        - config: Configuration object containing all necessary parameters for the experiment.
        - network (dict): A dictionary of neural networks used by the agent.
        - learning_rates (float): Learning rate for the optimizer.
        """
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_act = self.config.n_act
        self.epsilon = self.config.epsilon
        self.max_grad_norm = self.config.max_grad_norm
        self.memory_size = self.config.memory_size
        self.batch_size = self.config.batch_size
        self.warm_up_size = self.config.warm_up_size
        self.device = self.config.device

        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.set_network(network, learning_rates)
        # figure out the actor network
        # self.model = None
        # assert hasattr(self, 'model')
        #
        # # figure out the optimizer
        # assert hasattr(torch.optim, self.config.optimizer)
        # self.optimizer = eval('torch.optim.' + self.config.optimizer)(
        #     [{'params': self.model.parameters(), 'lr': self.config.lr_model}])
        # # figure out the lr schedule
        # # assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        # # self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer, self.config.lr_decay, last_epoch=-1,)
        #
        # assert hasattr(torch.nn, self.config.criterion)
        # self.criterion = eval('torch.nn.' + self.config.criterion)()
        #
        # self.replay_buffer = ReplayBuffer(self.memory_size)
        #
        # # move to device
        # self.model.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def get_step(self):
        """
        Returns the current training step.

        # Returns:
        - int: The current training step.
        """
        return self.learning_time

    def set_network(self, networks: dict, learning_rates: float):
        """
        Sets up the neural networks, optimizers, and loss functions for the agent.

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

    def get_action(self, state, epsilon_greedy=False):
        """
        Selects an action based on the current state using the epsilon-greedy policy.

        # Args:
        - state (torch.Tensor): The current state.
        - epsilon_greedy (bool): Whether to use epsilon-greedy exploration.

        # Returns:
        - np.ndarray: The selected action(s).
        """
        state = torch.Tensor(state).to(self.device)
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
                      compute_resource={},
                      tb_logger=None,
                      required_info={}):
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
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
        env.seed(seeds)
        # params for training
        gamma = self.gamma

        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device)
        except:
            pass

        _R = torch.zeros(len(env))
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state=state, epsilon_greedy=True)

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
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = torch.Tensor(next_state).to(self.device)
            except:
                state = copy.deepcopy(next_state)

            # begin update
            if len(self.replay_buffer) >= self.warm_up_size:
                batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(
                    self.batch_size)
                pred_Vs = self.model(batch_obs.to(self.device))  # [batch_size, n_act]
                action_onehot = torch.nn.functional.one_hot(batch_action.to(self.device),
                                                            self.n_act)  # [batch_size, n_act]

                _avg_predict_Q = (pred_Vs * action_onehot).mean(0)  # [n_act]
                predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]

                target_output = self.model(batch_next_obs.to(self.device))
                _avg_target_Q = batch_reward.to(self.device)[:, None] + (1 - batch_done.to(self.device))[:,
                                                                        None] * gamma * target_output
                target_Q = batch_reward.to(self.device) + (1 - batch_done.to(self.device)) * gamma * \
                           target_output.max(1)[0]
                _avg_target_Q = _avg_target_Q.mean(0)  # [n_act]

                self.optimizer.zero_grad()
                loss = self.criterion(predict_Q, target_Q)
                loss.backward()
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                self.optimizer.step()

                _loss.append(loss.item())

                self.learning_time += 1
                if self.learning_time >= (
                        self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                    save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if not self.config.no_tb :
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
                    for key in required_info:
                        return_info[key] = env.get_env_attr(key)
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time}
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info:
            return_info[key] = env.get_env_attr(key)
            # print(f"{key} : {return_info[key]}")
        env.close()

        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed=None,
                        required_info=['normalizer', 'gbest']):
        """
        Executes a single rollout episode in the environment and collects results.

        # Args:
        - env: The environment for the rollout.
        - seed (int, optional): Seed for reproducibility.
        - required_info (dict): Additional information required from the environment.

        # Returns:
        - dict: A dictionary containing results of the rollout episode, including return and environment-specific metrics.
        """
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.Tensor(state).unsqueeze(0).to(self.device)
                except:
                    st = [state]
                action = self.get_action(state)[0]
                action = action.squeeze()
                state, reward, is_done, info = env.step(action)
                R += reward
            _Rs = R
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': _Rs}

            if self.config.full_meta_data:
                meta_X = env.get_env_attr('meta_X')
                meta_Cost = env.get_env_attr('meta_Cost')
                metadata = {'X': meta_X, 'Cost': meta_Cost}
                results['metadata'] = metadata
            for key in required_info:
                results[key] = getattr(env, key)
            return results


    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms,
                        loss,
                        Return, Reward,
                        predict_Q, target_Q,
                        extra_info={}):
        """
        Logs training metrics to TensorBoard.

        # Args:
        - tb_logger: TensorBoard logger for logging training metrics.
        - mini_step (int): Current mini-batch step.
        - grad_norms (tuple): Gradient norms for the networks.
        -  loss (torch.Tensor): Training loss.
        -  Return (torch.Tensor): Episode return.
        -  Reward (torch.Tensor): Target reward.
        -  predict_Q (torch.Tensor): Predicted Q-values.
        -  target_Q (torch.Tensor): Target Q-values.
        -  extra_info (dict): Additional information to log.
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
        for id, (p_q, t_q) in enumerate(zip(predict_Q, target_Q)):
            tb_logger.add_scalar(f"Predict_Q/action_{id}", p_q.item(), mini_step)
            tb_logger.add_scalar(f"Target_Q/action_{id}", t_q.item(), mini_step)

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