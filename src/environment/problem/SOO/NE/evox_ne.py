import sys
import subprocess
import numpy as np
import time
import torch
import torch.nn as nn
from ....problem.basic_problem import Basic_Problem


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer_num):
        super(MLP, self).__init__()
        self.networks = nn.ModuleList()
        # self.in_layer = nn.Sequential(nn.Linear(state_dim,32),nn.Tanh())
        self.networks.append(nn.Sequential(nn.Linear(state_dim, 32), nn.Tanh()))
        # self.hidden_layers = []
        for _ in range(hidden_layer_num):
            self.networks.append(nn.Sequential(nn.Linear(32, 32), nn.Tanh()))
        # self.out_layer = nn.Linear(32,action_dim)
        self.networks.append(nn.Linear(32, action_dim))

    def forward(self, state):
        # h = self.in_layer(state)
        for layer in self.networks:
            state = layer(state)
        return torch.tanh(state)


envs = {
    'ant': {'state_dim': 27, 'action_dim': 8, },  # https://github.com/google/brax/blob/main/brax/envs/ant.py
    'halfcheetah': {'state_dim': 18, 'action_dim': 6, },  # https://github.com/google/brax/blob/main/brax/envs/half_cheetah.py
    'hopper': {'state_dim': 11, 'action_dim': 3, },  # https://github.com/google/brax/blob/main/brax/envs/hopper.py
    'humanoid': {'state_dim': 376, 'action_dim': 17, },  # https://github.com/google/brax/blob/main/brax/envs/humanoid.py
    'humanoidstandup': {'state_dim': 376, 'action_dim': 17, },  # https://github.com/google/brax/blob/main/brax/envs/humanoidstandup.py
    'inverted_pendulum': {'state_dim': 4, 'action_dim': 1, },  # https://github.com/google/brax/blob/main/brax/envs/inverted_pendulum.py
    'inverted_double_pendulum': {'state_dim': 8, 'action_dim': 1, },  # https://github.com/google/brax/blob/main/brax/envs/inverted_double_pendulum.py
    'pusher': {'state_dim': 23, 'action_dim': 7, },  # https://github.com/google/brax/blob/main/brax/envs/pusher.py
    'reacher': {'state_dim': 11, 'action_dim': 2, },  # https://github.com/google/brax/blob/main/brax/envs/reacher.py
    'swimmer': {'state_dim': 8, 'action_dim': 2, },  # https://github.com/google/brax/blob/main/brax/envs/swimmer.py
    'walker2d': {'state_dim': 17, 'action_dim': 6, },  # https://github.com/google/brax/blob/main/brax/envs/ant.py
}

model_depth = [
    0,
    1,
    2,
    3,
    4,
    5
]


class NE_Problem(Basic_Problem):
    """
    # Introduction
    This problem set is based on the neuroevolution interfaces in <a href="https://evox.readthedocs.io/en/latest/examples/brax.html">EvoX</a>. The goal is to optimize the parameters of neural network-based RL agents for a series of Robotic Control tasks. We pre-define 11 control tasks (e.g., swimmer, ant, walker2D etc.), and 6 MLP structures with 0~5 hidden layers. The combinations of task & network structure result in 66 problem instances, which feature extremely high-dimensional problems (>=1000D).

    # Original paper
    "[EvoX: A distributed GPU-accelerated framework for scalable evolutionary computation.](https://ieeexplore.ieee.org/abstract/document/10499977)" IEEE Transactions on Evolutionary Computation (2024).
    # Official Implementation
    [NE](https://github.com/EMI-Group/evox)
    # License
    None    
    """

    def __init__(self, env_name, model_depth, seed):
        """
        # Introduction
        Initializes the environment and neural network model for a single-objective optimization (SOO) problem using neuroevolution.
        # Args:
        - env_name (str): The name of the environment to be used.
        - model_depth (int): The number of layers or depth of the neural network model.
        - seed (int): The random seed for reproducibility.
        # Attributes:
        - env_state_dim (int): Dimension of the environment's state space.
        - env_action_dim (int): Dimension of the environment's action space.
        - nn_model (MLP): The neural network model used for policy representation.
        - dim (int): Total number of parameters in the neural network model.
        - ub (float): Upper bound for parameter initialization.
        - lb (float): Lower bound for parameter initialization.
        - seed (int): Random seed for reproducibility.
        - env_name (str): Name of the environment.
        - model_depth (int): Depth of the neural network model.
        - optimum (Any): Placeholder for the optimum solution (default: None).
        - init (bool): Flag indicating if initialization is complete (default: False).
        """
        self.env_state_dim = envs[env_name]['state_dim']
        self.env_action_dim = envs[env_name]['action_dim']
        self.nn_model = MLP(self.env_state_dim, self.env_action_dim, model_depth)
        self.dim = sum(p.numel() for p in self.nn_model.parameters())
        self.ub = 0.2
        self.lb = -0.2
        self.seed = seed
        self.env_name = env_name
        self.model_depth = model_depth
        self.optimum = None
        self.init = False

    def reset(self):
        """
        # Introduction
        Resets the state of the object by initializing the neural network model, adapter, evaluator, and a timer variable.
        # Args:
        None
        # Returns:
        None
        # Raises:
        - RuntimeError: If CUDA is not available or the model cannot be moved to CUDA.
        - ImportError: If `evox.utils.ParamsAndVector` cannot be imported.
        """
        from evox.utils import ParamsAndVector
        self.nn_model.to("cuda")
        self.adapter = ParamsAndVector(dummy_model = self.nn_model)
        self.evaluator = None
        self.T1 = 0

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the environment, combining its name and model depth.
        # Returns:
        - str: A string in the format "{env_name}-{model_depth}" representing the environment.
        """
        
        return f"{self.env_name}-{self.model_depth}"

    def func(self, x):  # x is a batch of neural network parameters: bs * num_params, type: numpy.array
        """
        # Introduction
        Evaluates a batch of neural network parameter sets in a Brax-based neuroevolution environment and returns their fitness scores.
        # Args:
        - x (numpy.ndarray): A batch of neural network parameters with shape (batch_size, num_params).
        # Returns:
        - numpy.ndarray: An array of fitness scores for each parameter set in the batch, where higher values indicate better performance.
        # Raises:
        - AssertionError: If the last dimension of `x` does not match the expected problem dimension (`self.dim`).
        # Notes:
        - The function initializes or updates the evaluator (`BraxProblem`) if necessary, based on the population size.
        - Handles NaN and infinite rewards by assigning a large negative value.
        - Converts rewards to a minimization objective by subtracting them from a large constant (1e5).
        """
        # x_cuda = torch.from_numpy(x).double().to(torch.get_default_device())
        # x_cuda = torch.from_numpy(x)
        # print(1)
        
        from evox.problems.neuroevolution.brax import BraxProblem
        # print(torch.cuda.is_available())
        torch.set_default_device("cuda")
        torch.set_float32_matmul_precision('high')

        if self.init:
            pop_size = x.shape[0]
            if pop_size != self.evaluator.pop_size:
                self.evaluator = BraxProblem(
                    policy = self.nn_model,
                    env_name = self.env_name,
                    max_episode_length = 200,
                    num_episodes = 10,
                    pop_size = pop_size,
                    reduce_fn = torch.mean,
                )

        if self.evaluator == None:
            pop_size = x.shape[0]
            self.evaluator = BraxProblem(
                policy = self.nn_model,
                env_name = self.env_name,
                max_episode_length = 200,
                num_episodes = 10,
                pop_size = pop_size,
                seed = self.seed,
                reduce_fn = torch.mean,
            )
            self.init = True

        assert x.shape[-1] == self.dim, "solution dimension not equal to problem dimension!"
        x = torch.tensor(x, device = torch.get_default_device()).float()
        nn_population = self.adapter.batched_to_params(x)
        # for key in nn_population.keys():
        #     print(nn_population[key].shape)
        rewards = self.evaluator.evaluate(nn_population)

        rewards[torch.isnan(rewards)] = -5 * 200
        rewards[torch.isinf(rewards)] = -5 * 200

        torch.set_default_device("cpu")

        rewards = rewards.cpu().numpy()
        rewards = 1e5 - rewards

        return rewards


