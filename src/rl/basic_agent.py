"""
This is a basic agent class for MetaBBO agents. All agents should inherit from this class.
Your own agent should have the following methods:
    - 1. __init__(self, config) : to initialize the agent
    - 2. train_episode(self, env, epoch_id, logger) : to train the agent for an episode by using env.reset() and env.step() to interact with environment. It's expected to return a Tuple[bool, dict] whose first element indicates whether the learned step has exceeded the max_learning_step and second element is a dictionary that contains:
       {
        'normalizer' : the best cost in initial population.
        'gbest' : the best cost found in this episode.
        'return' : total reward in this episode.
        'learn_steps' : the number of accumulated learned steps of the agent.
       }
    - 3. rollout_episode(self, env, epoch_id, logger) : to rollout the agent for an episode by using env.reset() and env.step() to interact with environment. It's expected to return a dictionary that contains:
       {
        'cost' : a list of costs that need to be maintained in backbone optimizer. See learnable_optimizer.py for more details.
        'fes' : times of function evaluations used by optimizer.
        'return' : total reward in this episode.
       }
"""
from typing import Tuple
from typing import Optional, Union, Literal
class Basic_Agent:
    """
    # Introduction
    The basic agent class for MetaBBO agents. All agents should inherit from this class.
    """
    def __init__(self, config):
        """
        Initialize the basic_agent with config.
        """
        self.__config = config


    def update_setting(self, config):
        """
        # Introduction
        Updates the configuration settings of the agent and initializes the learning process.

        # Args
        - config: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.

        """
        pass

    def train_episode(self,
                      env,
                      seeds: int) -> Tuple[bool, dict]:
        """
        # Introduction
        Executes a single training episode for the agent within the provided environment using the specified random seed.

        # Args:
        - env: The environment in which the agent will be trained. Must support standard reinforcement learning environment interfaces.
        - seeds (int): The random seed to ensure reproducibility of the episode.

        # Returns:
        - Tuple[bool, dict]: A tuple where the first element indicates whether the episode was successful, and the second element is a dictionary containing episode statistics or additional information.

        # Raises:
        - NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def rollout_episode(self,
                        env) -> dict:
        """
        # Introduction
        Executes a single rollout (episode) in the provided environment using the agent's policy.

        # Args:
        - env: The ParallelEnv object environment in which the episode will be executed. 

        # Returns:
        - dict: A dictionary containing episode statistics such as 'cost', 'fes','return',and other required info.

        # Raises:
        - NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def train_epoch(self):
        """
        # Introduction
        Executes a single epoch of training for the reinforcement learning agent.

        # Args:
        None

        # Returns:
        None

        # Raises:
        This method does not explicitly raise any exceptions.
        """
        
        pass

    def set_network(self, networks: dict, learning_rates: int):
        """
        # Introduction
        Configures the agent's neural network(s), initialize the network and set learning rates.

        # Args:
        - networks (dict): A dictionary containing the neural network architectures or instances to be used by the agent.
        - learning_rates (int): The learning rate(s) to be applied to the corresponding networks.

        # Returns:
        - None

        # Raises:
        - TypeError: If the input types do not match the expected types.
        """
        
        pass

    def get_step(self):
        """
        # Introduction
        get the current learning_time/learning_step of the agent.

        # Args:
        None

        # Returns:
        - int: The current learning time/learning step of the agent.

        # Raises:
        NotImplementedError: If the method is not implemented by a subclass.
        """
        
        pass

    @classmethod
    def log_to_tb_train(self):
        """
        # Introduction
        Logs training metrics and statistics to TensorBoard for visualization and monitoring.

        # Args:
        None

        # Returns:
        None

        # Raises:
        This method does not raise any exceptions.
        """
        
        pass
