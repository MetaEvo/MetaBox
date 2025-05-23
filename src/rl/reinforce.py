import math
from typing import Optional, Union, Literal, List

from ..environment.parallelenv.parallelenv import ParallelEnv
from .basic_agent import Basic_Agent
from .utils import *
import torch
import numpy as np


# memory for recording transition during training process
class Memory:
    """
    # Introduction

    A class to store and manage the memory required for reinforcement learning algorithms.
    It keeps track of log probabilities and rewards during an episode
    and provides functionality to clear the stored memory.

    # Methods:
    - __init__(): Initializes the memory by creating empty lists for log probabilities and rewards.
    - clear_memory(): Clears the stored memory by deleting the lists of log probabilities and rewards.

    """
    def __init__(self):
        """
        Initializes the memory by creating empty lists for log probabilities and rewards.
        """
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        """
        Clears the stored memory by deleting the lists of log probabilities and rewards.
        """
        del self.logprobs[:]
        del self.rewards[:]

class REINFORCE_Agent(Basic_Agent):
    """
    # Introduction
    The `REINFORCE_Agent` class implements a REINFORCE algorithm-based agent for reinforcement learning. This agent uses policy gradient methods to optimize the policy directly by maximizing the expected cumulative reward. It supports parallelized environments, logging to TensorBoard, and saving/loading checkpoints for training continuation.
    # Original paper
    "[**Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning**](https://link.springer.com/article/10.1007/BF00992696)." 1992 (in Machine Learning journal)

    # Args
    - `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    - `networks` (dict): A dictionary of neural networks used by the agent, with keys as network names (e.g., 'actor', 'critic') and values as the corresponding network instances.
    - `learning_rates` (float): Learning rate for the optimizer.

    # Attributes
    - `gamma` (float): Discount factor for future rewards.
    - `max_grad_norm` (float): Maximum gradient norm for gradient clipping.
    - `device` (str): Device to run the computations on (e.g., 'cpu' or 'cuda').
    - `network` (list): List of network names used by the agent.
    - `optimizer` (torch.optim.Optimizer): Optimizer for training the networks.
    - `learning_time` (int): Counter for the number of training steps completed.
    - `cur_checkpoint` (int): Counter for the current checkpoint index.

    # Methods

    - `set_network`(networks, learning_rates): Configures the networks and optimizer for the agent.
    - `update_setting`(config): Updates the agent's configuration and resets training-related attributes.
    - `train_episode`(envs, seeds, para_mode, compute_resource, tb_logger, required_info): Trains the agent for one episode using the REINFORCE algorithm.
    - `rollout_episode`(env, seed, required_info): Executes a single rollout episode in a given environment without training.
    - `log_to_tb_train`(tb_logger, mini_step, grad_norms, loss, Return, Reward, logprobs, extra_info): Logs training metrics and additional information to TensorBoard.
    """
    def __init__(self, config, networks: dict, learning_rates: float):
        """
        Initializes the REINFORCE agent with the given configuration, networks, and learning rates.Store the initial agent in the checkpoint directory.

        # Args:
        - config: Configuration object containing all necessary parameters for the experiment.
        - networks (dict): A dictionary of neural networks used by the agent.
        - learning_rates (float): Learning rate for the optimizer.
        """
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.max_grad_norm = self.config.max_grad_norm
        self.device = self.config.device

        self.set_network(networks, learning_rates)

        # figure out the lr schedule
        # assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        # self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer, self.config.lr_decay, last_epoch=-1,)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def set_network(self, networks: dict, learning_rates: float):
        """
        Configures the networks and optimizer for the agent.

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

        # make sure has model or network
        assert hasattr(self, 'model') or hasattr(self, 'net')

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
        Trains the agent for one episode using the REINFORCE algorithm.

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
        memory = Memory()

        # params for training
        gamma = self.gamma

        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass

        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            entropy = []

            action, log_lh, entro_p = self.model(state)

            memory.logprobs.append(log_lh)

            entropy.append(entro_p.detach().cpu())

            # state transient
            state, rewards, is_end, info = env.step(action)
            memory.rewards.append(torch.FloatTensor(rewards).to(self.device))
            _R += rewards
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass

        # begin update
        logprobs = torch.stack(memory.logprobs).view(-1).to(self.device)
        Reward = []
        reward_reversed = memory.rewards[::-1]
        R = torch.zeros(len(envs))
        for r in range(len(reward_reversed)):
            R = R * gamma + reward_reversed[r]
            Reward.append(R)
        # clip the target:
        Reward = torch.stack(Reward[::-1], 0)
        Reward = Reward.view(-1).to(self.device)
        loss = - torch.mean(logprobs * Reward)
        self.optimizer.zero_grad()
        loss.backward()
        grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
        self.optimizer.step()

        memory.clear_memory()

        self.learning_time += 1
        if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
            save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
            self.cur_checkpoint += 1

        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed=None,
                        required_info={}):
        """
        Executes a single rollout episode in a given environment without training.

        # Args:
        - env: The environment for the rollout.
        - seed (int, optional): Seed for reproducibility.
        - required_info (dict): Additional information required from the environment.

        # Returns:
        - dict: A dictionary containing rollout results such as return, cost, and metadata.
        """
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
                self.model = self.model.float()
                action, _ = self.model(state)
                action = action.cpu().numpy().squeeze()
                state, reward, is_done, info = env.step(action)
                R += reward
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': R}

            if self.config.full_meta_data:
                meta_X = env.get_env_attr('meta_X')
                meta_Cost = env.get_env_attr('meta_Cost')
                metadata = {'X': meta_X, 'Cost': meta_Cost}
                results['metadata'] = metadata
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results


    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms,
                        loss,
                        Return, Reward,
                        logprobs,
                        extra_info={}):
        """
        Logs training metrics and additional information to TensorBoard.

        # Args:
        - tb_logger: TensorBoard logger for logging training metrics.
        - mini_step (int): Current mini-batch step.
        - grad_norms (tuple): Gradient norms for the networks.
        - loss (torch.Tensor): Training loss.
        - Return (torch.Tensor): Episode return.
        - Reward (torch.Tensor): Target reward.
        - logprobs (torch.Tensor): Log probabilities.
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

        # grad and clipped grad
        grad_norms, grad_norms_clipped = grad_norms
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'grad/{network_name}', grad_norms[id], mini_step)
            tb_logger.add_scalar(f'grad_clipped/{network_name}', grad_norms_clipped[id], mini_step)

        # loss
        tb_logger.add_scalar('loss', loss.item(), mini_step)
        #
        # train metric
        avg_reward = torch.stack(Reward).mean().item()
        max_reward = torch.stack(Reward).max().item()
        #
        tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)
        tb_logger.add_scalar('train/-avg_logprobs', -logprobs.mean().item(), mini_step)
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