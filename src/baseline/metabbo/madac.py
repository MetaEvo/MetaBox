import collections
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

from ...rl.vdn import VDN_Agent


class MultiAgentQNet(nn.Module):
    def __init__(self, input_shape, agent_configs):
        """
        Args:
            input_shape (int): 输入特征维度
            agent_configs (list of dict): 每个 agent 的配置字典，包括 'name', 'n_actions', 'n_valid_actions'
        """
        super(MultiAgentQNet, self).__init__()
        self.agents = nn.ModuleList()

        for config in agent_configs:
            agent = nn.Sequential(
                nn.Linear(input_shape, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, config['n_actions'])
            )
            # 保存每个 agent 的有效动作数
            agent.n_valid_actions = config['n_valid_actions']
            self.agents.append(agent)

        self.n_agents = len(agent_configs)

    def forward(self, obs):
        q_values = []
        for i, agent in enumerate(self.agents):
            x = agent(obs[:, i, :])
            # 无效动作置为-999
            x[:, agent.n_valid_actions:] = -999
            q_values.append(x.unsqueeze(1))
        return th.cat(q_values, dim = 1)


class MADAC(VDN_Agent):
    """
    # Introduction
    Multi-agent dynamic algorithm configuration in which one agent works for one type of configuration hyperparameter.It rmulates the dynamic configuration of a complex algorithm with multiple types of hyperparameters as a contextual multi-agent Markov decision process and solves it by a cooperative multi-agent RL (MARL) algorithm.
    # Original paper
    "[**Multi-agent dynamic algorithm configuration**](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)." Advances in Neural Information Processing Systems 35 (2022): 20147-20161.
    # Official Implementation
    [MADAC](https://github.com/lamda-bbo/madac)
    # Args:
    - config (Namespace): A configuration object containing all necessary hyperparameters and settings for the agent and environment.
    # Attributes:
    - gamma (float): Discount factor for future rewards.
    - n_act (int): Number of actions per agent.
    - epsilon_start (float): Initial value of epsilon for epsilon-greedy exploration.
    - epsilon_end (float): Final value of epsilon after decay.
    - epsilon_decay_steps (int): Number of steps over which epsilon decays.
    - max_grad_norm (float): Maximum norm for gradient clipping.
    - memory_size (int): Size of the replay buffer.
    - batch_size (int): Number of samples per training batch.
    - warm_up_size (int): Number of steps before training starts.
    - chunk_size (int): Size of sequence chunks for training.
    - update_iter (int): Number of update iterations per training step.
    - device (str): Device to use for computation ('cuda' or 'cpu').
    - n_agent (int): Number of agents in the environment.
    - available_action (list): List specifying the number of available actions for each agent.
    - optimizer (str): Optimizer to use for training.
    - criterion (str): Loss function to use for training.
    - target_update_interval (int): Frequency (in steps) to update the target network.
    - required_info (dict): Dictionary specifying required information for logging or evaluation.
    - agent_save_dir (str): Directory path for saving agent checkpoints.
    # Methods:
    - __init__(self, config): Initializes the MADAC agent with the specified configuration.
    - __str__(self): Returns the string representation of the agent ("MADAC").
    """
    def __init__(self, config):
        config.gamma = 0.99
        config.n_act = 4
        config.epsilon_start = 1
        config.epsilon_end = 0.1
        config.epsilon_decay_steps = 10000
        config.max_grad_norm = 5
        config.memory_size = 5000
        config.batch_size = 64
        config.warm_up_size = 500
        config.chunk_size = 1
        config.update_iter = 10
        config.device = 'cuda' if th.cuda.is_available() else 'cpu'
        config.n_agent = 4
        config.available_action = [4, 4, 4, 2]
        config.optimizer = 'Adam'
        # config.lr_model = 1e-3
        # config.lr_scheduler = 'StepLR'
        # config.lr_decay = 0.99
        config.criterion = 'MSELoss'

        config.target_update_interval = 2000
        config.required_info = {
            'hv_his':'hv_his',
            'igd_his': 'igd_his'
        }

        agent_configs = [
            {'name': 'ns_agent', 'n_actions': 4, 'n_valid_actions': 4},
            {'name': 'os_agent', 'n_actions': 4, 'n_valid_actions': 4},
            {'name': 'pc_agent', 'n_actions': 4, 'n_valid_actions': 4},
            {'name': 'weight_agent', 'n_actions': 4, 'n_valid_actions': 2}
        ]
        model = MultiAgentQNet(input_shape = 22, agent_configs = agent_configs)
        config.agent_save_dir = os.path.join(
            config.agent_save_dir,
            self.__str__(),
            config.train_name
        )
        super().__init__(config, {'model': model}, 0.001)

    def __str__(self):
        return "MADAC"


class Config:
    def __init__(self):
        # 设置默认值，具体数值可以根据需要进行调整
        self.gamma = 0.99  # 折扣因子
        self.n_act = 4  # 动作空间大小
        self.epsilon = 0.5  # epsilon-greedy 策略中的 epsilon 值
        self.max_grad_norm = 10.0  # 最大梯度裁剪
        self.memory_size = 10000  # 经验回放池大小
        self.batch_size = 64  # 批量大小
        self.warm_up_size = 1000  # 预热步数
        self.device = 'cuda' if th.cuda.is_available() else 'cpu'  # 使用的设备，默认为 'cuda' 或 'cpu'
        self.n_agent = 4  # 代理数量
        self.available_action = 4  # 可用动作数量


if __name__ == '__main__':
    # 输入数据 shape: (batch_size, n_agents, input_shape)

    # 创建一个 config 实例
    config = Config()
    obs = th.randn(32, 4, 22)
    madac_agent = MADAC(config)

    print()


