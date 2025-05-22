from .networks import MLP
from ...rl.dqn import *


class DEDQN(DQN_Agent):
    """
    # Introduction
    DEDQN is a mixed mutation strategy Differential Evolution (DE) algorithm based on deep Q-network (DQN), in which a deep reinforcement learning approach realizes the adaptive selection of mutation strategy in the evolution process.
    # Original paper
    "[**Differential evolution with mixed mutation strategy based on deep reinforcement learning**](https://www.sciencedirect.com/science/article/abs/pii/S1568494621005998)." Applied Soft Computing (2021).
    # Official Implementation
    None
    # Args:
    - config (object): Configuration object containing agent and environment parameters. 
      The constructor modifies and extends this configuration with DEDQN-specific settings, 
      such as state size, action space, learning rate, optimizer, and neural network architecture.
    # Attributes:
    - config (object): The configuration object with updated DEDQN parameters.
    - model (MLP): The neural network model used for Q-value approximation.
    # Example:
    ```python
    agent = DEDQN(config)
    ```
    # Notes:
    - The agent uses an MLP with two hidden layers of 10 units each and ReLU activations.
    - The optimizer is set to AdamW, and the loss criterion is MSELoss.
    - Learning rate decay and gradient clipping are not used by default, following the original DEDQN design.
    - The agent's save directory is automatically constructed based on its string representation and training name.
    # See Also:
    - DQN_Agent: The base class for DEDQN.
    - MLP: The neural network class used for function approximation.
    """
    
    def __init__(self, config):
        
        self.config = config
        self.config.state_size = 4
        self.config.n_act = 3
        self.config.mlp_config = [{'in': config.state_size, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': 10, 'drop_out': 0, 'activation': 'ReLU'},
                             {'in': 10, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
        self.config.lr_model = 1e-4
        # origin DEDDQN doesn't have decay
        self.config.lr_decay = 1
        self.config.epsilon = 0.1
        self.config.gamma = 0.8
        self.config.memory_size = 100
        self.config.batch_size = 64
        self.config.warm_up_size = config.batch_size

        self.config.device = config.device
        # origin DEDDQN doesn't have clip 
        self.config.max_grad_norm = math.inf
        self.config.optimizer = 'AdamW'
        # origin code does not have lr_scheduler
        self.config.lr_scheduler = 'ExponentialLR'
        self.config.criterion = 'MSELoss'
        model = MLP(self.config.mlp_config).to(self.config.device)

        # self.__cur_checkpoint=0
        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config, {'model': model}, self.config.lr_model)

    def __str__(self):
        return "DEDQN"

