from ...rl.ddqn import *
from .networks import MLP

class DEDDQN(DDQN_Agent):
    """
    # Introduction
    DE-DDQN is an adaptive operator selection method based on Double Deep Q-Learning (DDQN), a Deep Reinforcement Learning method, to control the mutation strategies of Differential Evolution (DE).
    # Original paper
    "[**Deep reinforcement learning based parameter control in differential evolution**](https://dl.acm.org/doi/abs/10.1145/3321707.3321813)." Proceedings of the Genetic and Evolutionary Computation Conference (2019).
    # Official Implementation
    [DE-DDQN](https://github.com/mudita11/DE-DDQN)
    # Args:
    - config (Namespace): Configuration object containing environment and agent parameters. 
      The constructor modifies several attributes of `config` to set up the DEDDQN agent.
    # Attributes Set in Config:
    - state_size (int): Size of the input state vector (default: 99).
    - n_act (int): Number of possible actions (default: 4).
    - lr_model (float): Learning rate for the optimizer (default: 1e-4).
    - lr_decay (float): Learning rate decay factor (default: 1).
    - batch_size (int): Batch size for training (default: 64).
    - epsilon (float): Exploration rate for epsilon-greedy policy (default: 0.1).
    - gamma (float): Discount factor for future rewards (default: 0.99).
    - target_update_interval (int): Frequency of target network updates (default: 1000).
    - memory_size (int): Size of the replay buffer (default: 100000).
    - warm_up_size (int): Number of experiences to collect before training (default: 10000).
    - net_config (list): List of dictionaries specifying the neural network architecture.
    - device (str or torch.device): Device to run the model on.
    - max_grad_norm (float): Maximum norm for gradient clipping (default: infinity).
    - optimizer (str): Optimizer type (default: 'Adam').
    - lr_scheduler (str): Learning rate scheduler type (default: 'ExponentialLR').
    - criterion (str): Loss function (default: 'MSELoss').
    - agent_save_dir (str): Directory to save agent checkpoints.
    # Methods
    - __str__(): Returns the string "DEDDQN" representing the agent type.
    # Usage
    Instantiate with a configuration object and use as a reinforcement learning agent for environments
    with discrete action spaces.
    # Raises
    - Inherits exceptions from DDQN_Agent and underlying PyTorch modules.
    """
    
    def __init__(self, config):
        self.config = config
        self.config.state_size = 99
        self.config.n_act = 4
        self.config.lr_model = 1e-4
        # origin DEDDQN doesn't have decay
        self.config.lr_decay = 1
        self.config.batch_size = 64
        self.config.epsilon = 0.1
        self.config.gamma = 0.99
        self.config.target_update_interval = 1000
        self.config.memory_size = 100000
        self.config.warm_up_size = 10000
        self.config.net_config = [{'in': config.state_size, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': 100, 'drop_out': 0, 'activation': 'ReLU'},
                                  {'in': 100, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]

        self.config.device = config.device
        # origin DEDDQN doesn't have clip
        self.config.max_grad_norm = math.inf

        # self.target_model is defined in DDQN_Agent

        self.config.optimizer = 'Adam'
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

    def __str__(self):
        return "DEDDQN"