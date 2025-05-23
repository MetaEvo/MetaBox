import math
from typing import Optional, Union, Literal, List

from ..environment.parallelenv.parallelenv import ParallelEnv
from .basic_agent import Basic_Agent
from .utils import *
import torch
import numpy as np


class QLearning_Agent(Basic_Agent):
    """
    # Introduction
    The `QLearning_Agent` class implements a Q-Learning agent for reinforcement learning. This agent uses a tabular Q-learning approach to learn optimal policies in discrete state and action spaces. It supports parallelized environments, epsilon-greedy exploration, and provides methods for training, action selection, and evaluation.

    # Original paper
    "Learning from Delayed Rewards" (Chapter 5 introduces Q-Learning)
    The First Journal Publication：
    "[**Q-Learning**](https://link.springer.com/article/10.1007/BF00992698)." 1992 (in Machine Learning journal)

    # Args
    - `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.

    # Attributes
    - `gamma` (float): Discount factor for future rewards.(default: 0.8)
    - `n_act` (int): Number of possible actions.(default: 4)
    - `n_state` (int): Number of possible states.(default: 4)
    - `epsilon` (float): Exploration rate for epsilon-greedy policy.(default: None)
    - `lr_model` (float): Learning rate for updating the Q-table.(default: 1)
    - `q_table` (torch.Tensor): Q-table storing the state-action values.
    - `learning_time` (int): Counter for the number of learning steps taken.
    - `cur_checkpoint` (int): Counter for the current checkpoint index.
    - `config` (object): Configuration object passed during initialization.

    # Methods
    - `__init__(config)`: Initializes the Q-Learning agent with the given configuration.
    - `update_setting`(config): Updates the agent's settings and resets learning time and checkpoints.
    - `get_action`(state, epsilon_greedy=False): Selects an action based on the current state using an epsilon-greedy policy.
    - `train_episode`(envs, seeds, para_mode, compute_resource, tb_logger, required_info): Trains the agent for one episode in the given environment(s).
    - `rollout_episode`(env, seed, required_info): Executes a single episode in the environment without training and returns the results.
    - `log_to_tb_train`(tb_logger, mini_step, loss, Return, Reward, extra_info): Logs training metrics and additional information to TensorBoard.

    """
    def __init__(self, config):
        """
        Initializes the Q-Learning agent with the given configuration.Store the initial agent in the checkpoint directory.

        # Args:
        - config: Configuration object containing all necessary parameters for the experiment.
        """
        super().__init__(config)
        self.config = config

        # define parameters
        self.gamma = self.config.gamma
        self.n_act = self.config.n_act
        self.n_state = self.config.n_state
        self.epsilon = self.config.epsilon
        self.lr_model = self.config.lr_model

        self.q_table = torch.zeros(self.n_state, self.n_act).to(self.config.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def update_setting(self, config):
        """
        Updates the agent's settings and resets learning time and checkpoints.

        # Args:
        - config: Configuration object containing updated parameters.
        """
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    # def get_action(self, state, epsilon_greedy=False):
    #     Q_list = self.q_table(state)
    #     if epsilon_greedy and np.random.rand() < self.epsilon:
    #         action = np.random.randint(low=0, high=self.n_act, size=len(state))
    #     else:
    #         action = torch.argmax(Q_list, -1).numpy()
    #     return action

    def get_action(self, state, epsilon_greedy=False):
        """
        Selects an action based on the current state using an epsilon-greedy policy.

        # Args:
        - state (torch.Tensor): The current state.
        - epsilon_greedy (bool): Whether to use epsilon-greedy exploration.

        # Returns:
        - np.ndarray: The selected action(s).
        """
        Q_list = torch.stack([self.q_table[st] for st in state])
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
        Trains the agent for one episode in the given environment(s).

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
        state = torch.FloatTensor(state)

        _R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state=state, epsilon_greedy=True)

            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward

            # error = reward + gamma * self.q_table[next_state].max() - self.q_table[state][action]

            error = [reward[i] + gamma * self.q_table[next_state[i]].max() - self.q_table[state[i]][action[i]] \
                     for i in range(len(state))]

            for i in range(len(state)):
                self.q_table[state[i]][action[i]] += self.lr_model * error[i]

            # store info
            state = torch.FloatTensor(next_state)

            self.learning_time += 1
            if self.learning_time >= (
                    self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if self.learning_time >= self.config.max_learning_step:
                return_info = {'return': _R, 'learn_steps': self.learning_time, }
                env_cost = env.get_env_attr('cost')
                return_info['gbest'] = env_cost[-1]
                for key in required_info.keys():
                    return_info[key] = env.get_env_attr(required_info[key])
                env.close()
                return self.learning_time >= self.config.max_learning_step, return_info

        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        # 返回：奖励_R, 学习步数_learning_time, 最优评估值：gbest
        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed=None,
                        required_info={}):
        """
        Executes a single episode in the environment without training and returns the results.

        # Args:
        - env: The environment for the rollout.
        - seed (int, optional): Seed for reproducibility.
        - required_info (dict): Additional information required from the environment.

        # Returns:
        - dict: A dictionary containing evaluation results such as cumulative rewards, environment costs, and metadata.
        """
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                action = self.get_action([state])[0]
                state, reward, is_done, info = env.step(action)
                R += reward
            env_cost = env.get_env_attr('cost')  # 只是最优的评估值，还需要加每一代最优的解，所有评估值
            env_fes = env.get_env_attr('fes')
            results = {'cost': env_cost, 'fes': env_fes, 'return': R}
            # 加metadata：每一代最优（解和值）, 所有评估值

            if self.config.full_meta_data:
                meta_X = env.get_env_attr('meta_X')
                meta_Cost = env.get_env_attr('meta_Cost')
                metadata = {'X': meta_X, 'Cost': meta_Cost}
                results['metadata'] = metadata

            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results


    def log_to_tb_train(self, tb_logger, mini_step,
                        loss,
                        Return, Reward,
                        extra_info={}):
        """
        Logs training metrics and additional information to TensorBoard.

        # Args:
        - tb_logger: TensorBoard logger for logging training metrics.
        - mini_step (int): Current mini-batch step.
        - loss (torch.Tensor): Training loss.
        - Return (torch.Tensor): Episode return.
        - Reward (torch.Tensor): Target reward.
        - extra_info (dict): Additional information to log.
        """
        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # lr_model
        tb_logger.add_scalar('learnrate', self.lr_model, mini_step)

        # loss
        tb_logger.add_scalar('loss', loss.item(), mini_step)

        # Q
        Q = self.q_table.mean(0)  # [n_act]
        for id, q in enumerate(Q):
            tb_logger.add_scalar("Q_values", q.item(), mini_step)
        # tb_logger.add_scalars("Q_values", {f"action_{id}": q.item() for id, q in enumerate(Q)}, mini_step)

        # train
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
