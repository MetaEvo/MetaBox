from typing import Tuple
from .basic_agent import Basic_Agent
import torch
import math, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from .utils import *
from ..environment.parallelenv.parallelenv import ParallelEnv

# memory for recording transition during training process
class Memory:
    """
    # Introduction

    A class to store and manage the memory required for reinforcement learning algorithms.
    It keeps track of actions, states, log probabilities, and rewards during an episode
    and provides functionality to clear the stored memory.

    # Methods:
    - __init__(): Initializes the memory by creating empty lists for actions, states, log probabilities, and rewards.
    - clear_memory(): Clears the stored memory by deleting the lists of actions, states, log probabilities, and rewards.

    """
    def __init__(self):
        """
        Initializes the memory by creating empty lists for actions, states, log probabilities, and rewards.
        """
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        """
        Clears the stored memory by deleting the lists of actions, states, log probabilities, and rewards.
        """
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]

class A2C_Agent(Basic_Agent):
    """
    # Introduction
    The `A2C_Agent` class implements an Advantage Actor-Critic (A2C) agent for reinforcement learning. This agent uses actor and critic networks to optimize policies and guide the low_level optimizer to optimize.

    # Original paper
    "[**Actor-Critic Algorithms**](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)." Advances in Neural Information Processing Systems (NIPS), 1999

    # Args
    - `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    - `networks` (dict): A dictionary of neural networks used by the agent, with keys as network names (e.g., 'actor', 'critic') and values as the corresponding network instances.
    - `learning_rates` (float): Learning rate for the optimizer.

    # Attributes
    - `gamma` (float): Discount factor for future rewards.
    - `n_step` (int): Number of steps for multi-step returns.
    - `max_grad_norm` (float): Maximum gradient norm for gradient clipping.
    - `device` (str): Device to run the computations on (e.g., 'cpu' or 'cuda').
    - `network` (list): List of network names used by the agent.
    - `optimizer` (torch.optim.Optimizer): Optimizer for training the networks.
    - `learning_time` (int): Counter for the number of training steps completed.
    - `cur_checkpoint` (int): Counter for the current checkpoint index.

    # Methods
    - `set_network`(networks, learning_rates): Initializes the networks and optimizer for the agent.
    - `get_step`(): Returns the current training step count.
    - `update_setting`(config): Updates the agent's configuration and resets training-related attributes.
    - `train_episode`(envs, para_mode, compute_resource, tb_logger, required_info): Trains the agent for one episode in a parallelized environment.
    - `log_to_tb_train`(tb_logger, mini_step, grad_norms, reinforce_loss, baseline_loss, Return, Reward, memory_reward, critic_output, logprobs, entropy, approx_kl_divergence, extra_info): Logs training metrics to TensorBoard.
    - `rollout_episode`(env, seed, required_info): Executes a single rollout episode in the environment and returns the results.
    - `rollout_batch_episode`(envs, seeds, para_mode, compute_resource, required_info): Executes batch rollout episodes in parallelized environments and returns the results.

    """

    def __init__(self, config, networks: dict, learning_rates: float):
        """
        Initializes the A2C_Agent with the given configuration, networks, and learning rates.

        # Args:
        - config: Configuration object containing all necessary parameters for the experiment.
        - networks (dict): A dictionary of neural networks used by the agent.
        - learning_rates (float): Learning rate for the optimizer.
        """
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_step = self.config.n_step
        self.max_grad_norm = self.config.max_grad_norm
        self.device = self.config.device
        
        self.set_network(networks, learning_rates)
        
        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0
        
    def set_network(self, networks: dict, learning_rates: float):
        """
        Initializes the networks and optimizer for the agent.

        # Args:
        - networks (dict): A dictionary of neural networks used by the agent.
        - learning_rates (float): Learning rate for the optimizer.

        # Raises:
        - AssertionError: If required network attributes (e.g., 'actor', 'critic') are not set.
        - ValueError: If the length of the learning rates list does not match the number of networks.
        - AttributeError: If the specified optimizer in the configuration is not available in `torch.optim`.
        """
        Network_name = []
        if networks:
            for name, network in networks.items():
                Network_name.append(name)
                setattr(self, name, network)   # Assign each network in the dictionary to the class instance
        self.network = Network_name

        # make sure actor and critic network
        assert hasattr(self, 'actor') and hasattr(self, 'critic')

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

    def get_step(self):
        """
        Returns the current training step count.

        # Returns:
        - int: The current training step count.
        """
        return self.learning_time

    def update_setting(self, config):
        """
        Updates the agent's configuration and resets training-related attributes.Store the initial agent in the checkpoint directory.

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
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
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
        
        memory = Memory()

        # params for training
        gamma = self.gamma
        n_step = self.n_step
        
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        t = 0
        # initial_cost = obj
        _R = torch.zeros(len(env))
        loss_ = []
        # sample trajectory
        while not env.all_done():
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step :  
                
                memory.states.append(state.clone())
                action, log_lh, entro_p = self.actor(state)
                
                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))
                memory.logprobs.append(log_lh)
                
                entropy.append(entro_p.detach().cpu())

                baseline_val = self.critic(state)
                baseline_val_detached = baseline_val.detach()
                
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action)
                memory.rewards.append(torch.FloatTensor(rewards).to(self.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards
                # store info

                # next
                t = t + 1

                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    pass
            
            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            logprobs = memory.logprobs

            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)

            # get traget value for critic
            Reward = []
            reward_reversed = memory.rewards[::-1]
            # get next value
            R = self.critic(self.actor(state))[0]

            for r in range(len(reward_reversed)):
                R = R * gamma + reward_reversed[r]
                Reward.append(R)
            # clip the target:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)
            
            # Finding Surrogate Loss:
            advantages = Reward - bl_val_detached
            reinforce_loss = -(logprobs * advantages).mean()

            # define baseline loss
            baseline_loss = ((bl_val - Reward) ** 2).mean()
            # calculate loss
            loss = baseline_loss + reinforce_loss

            # update gradient step
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm and get (clipped) gradient norms for logging
            # current_step = int(pre_step + t//n_step * K_epochs  + _k)
            grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
            loss_.append(loss.detach())
            # perform gradient descent
            self.optimizer.step()
            self.learning_time += 1
            if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                save_class(self.config.agent_save_dir, 'checkpoint-'+str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if self.learning_time >= self.config.max_learning_step:
                memory.clear_memory()
                return_info = {'return': _R, 'loss': loss_, 'learn_steps': self.learning_time, }
                for key in required_info.keys():
                    return_info[key] = env.get_env_attr(required_info[key])
                env.close()
                return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()
        
        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'loss': loss_, 'learn_steps': self.learning_time, }
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()
        
        return is_train_ended, return_info
    
    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms, # network grad
                        reinforce_loss, baseline_loss, # actor loss critic loss
                        Return, Reward, memory_reward,
                        critic_output,
                        logprobs, entropy,
                        approx_kl_divergence,
                        extra_info = {}):
        """
        Logs training metrics to TensorBoard.

        # Args:
        - tb_logger: TensorBoard logger for logging training metrics.
        - mini_step (int): Current mini-batch step.
        - grad_norms (tuple): Gradient norms for the networks.
        - reinforce_loss (torch.Tensor): Actor loss.
        - baseline_loss (torch.Tensor): Critic loss.
        - Return (torch.Tensor): Episode return.
        - Reward (torch.Tensor): Target reward.
        - memory_reward (list): List of rewards from memory.
        - critic_output (torch.Tensor): Critic network output.
        - logprobs (torch.Tensor): Log probabilities of actions.
        - entropy (torch.Tensor): Entropy of the policy.
        - approx_kl_divergence (torch.Tensor): Approximate KL divergence.
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
        tb_logger.add_scalar('loss/actor_loss', reinforce_loss.item(), mini_step)
        tb_logger.add_scalar('loss/critic_loss', baseline_loss.item(), mini_step)
        tb_logger.add_scalar('loss/total_loss', (reinforce_loss + baseline_loss).item(), mini_step)

        # train metric
        avg_reward = torch.stack(memory_reward).mean().item()
        max_reward = torch.stack(memory_reward).max().item()

        tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)
        tb_logger.add_scalar('train/target_avg_return_changed', Reward.mean().item(), mini_step)
        tb_logger.add_scalar('train/critic_avg_output', critic_output.mean().item(), mini_step)
        tb_logger.add_scalar('train/avg_entropy', entropy.mean().item(), mini_step)
        tb_logger.add_scalar('train/-avg_logprobs', -logprobs.mean().item(), mini_step)
        tb_logger.add_scalar('train/approx_kl', approx_kl_divergence.item(), mini_step)
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
    
    def rollout_episode(self, 
                        env,
                        seed=None,
                        required_info={}):
        """
        Executes a single rollout episode in the environment and returns the results.

        # Args:
        - env: The environment for the rollout.
        - seed (int, optional): Seed for reproducibility.
        - required_info (dict): Additional information required from the environment.

        # Returns:
        - dict: A dictionary containing the total return and additional requested information.
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
                action = self.actor(state)
                action = action.cpu().numpy().squeeze()
                state, reward, is_done = env.step(action)
                R += reward
            results = {'return': R}
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results


