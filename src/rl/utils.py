import collections
import torch
import random
import numpy as np
import pickle 
import os
import math

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


class ReplayBuffer:
    """
    # Introduction
    The `ReplayBuffer` class is a utility for storing and sampling experiences in reinforcement learning. It uses a fixed-size buffer to store transitions (state, action, reward, next state, done) and provides methods to append new experiences and sample mini-batches for training. This class is essential for implementing experience replay, which helps stabilize and improve the learning process in reinforcement learning algorithms.

    # Args
    - `max_size` (int): The maximum number of experiences the buffer can hold.

    # Attributes
    - `buffer` (collections.deque): A deque object that stores the experiences with a fixed maximum size.

    # Methods
    - `append`(exp): Adds a new experience to the buffer.
    - `sample`(batch_size): Samples a mini-batch of experiences from the buffer.
    - `__len__()`: Returns the current number of experiences stored in the buffer.
    """
    def __init__(self, max_size):
        """
        Initializes the ReplayBuffer with a fixed maximum size.

        # Args:
        - max_size (int): The maximum number of experiences the buffer can hold.
        """
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        """
        Adds a new experience to the buffer.

        # Args:
        - exp (tuple): A tuple representing a transition (state, action, reward, next state, done).
        """
        self.buffer.append(exp)

    def sample(self, batch_size):
        """
        Samples a mini-batch of experiences from the buffer.

        # Args:
        - batch_size (int): The number of experiences to sample.

        # Returns:
        - tuple: A tuple containing batches of observations, actions, rewards, next observations, and done flags.

        # Raises:
        - ValueError: If the requested batch size exceeds the number of stored experiences.
        """
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = zip(*mini_batch)
        # print(type(obs_batch),type(action_batch),type(reward_batch),type(next_obs_batch),type(done_batch))
        # print(type(action_batch[0]))
        # obs_batch = torch.FloatTensor(np.array(obs_batch))
        obs_batch = torch.stack(obs_batch)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.Tensor(reward_batch)

        # 兼容操作，满足MOO和SOO等需求
        if isinstance(next_obs_batch, (list, np.ndarray)):
            next_obs_batch = torch.Tensor(np.array(next_obs_batch))
        else:
            next_obs_batch = torch.stack(next_obs_batch)

        # next_obs_batch = torch.FloatTensor(np.array(next_obs_batch))
        done_batch = torch.Tensor(done_batch)
        return obs_batch, action_batch, reward_batch, next_obs_batch, done_batch

    def __len__(self):
        """
        Returns the current number of experiences stored in the buffer.

        # Returns:
        - int: The number of experiences in the buffer.
        """
        return len(self.buffer)


class ReplayBuffer_torch:
    """
    # Introduction
    The `ReplayBuffer_torch` class implements a replay buffer for reinforcement learning using PyTorch. It is designed to store and sample transitions (state, action, reward, next_state, done) efficiently, enabling agents to learn from past experiences. The buffer supports fixed capacity and operates in a circular manner, overwriting old transitions when full.

    # Args
    - `capacity` (int): The maximum number of transitions the buffer can store.
    - `state_dim` (int): The dimensionality of the state space.
    - `device` (torch.device): The device (CPU or GPU) on which the buffer's tensors are stored.

    # Attributes
    - `capacity` (int): The maximum number of transitions the buffer can store.
    - `device` (torch.device): The device (CPU or GPU) on which the buffer's tensors are stored.
    - `position` (int): The current position in the buffer where the next transition will be stored.
    - `size` (int): The current number of transitions stored in the buffer.
    - `states` (torch.Tensor): A tensor storing the states of transitions.
    - `actions` (torch.Tensor): A tensor storing the actions of transitions.
    - `rewards` (torch.Tensor): A tensor storing the rewards of transitions.
    - `next_states` (torch.Tensor): A tensor storing the next states of transitions.
    - `dones` (torch.Tensor): A tensor storing the done flags of transitions.

    # Methods
    - `append`(state, action, reward, next_state, done): Adds a new transition to the buffer. Overwrites the oldest transition if the buffer is full.
    - `sample`(batch_size): Samples a batch of transitions from the buffer.
    - `__len__()`: Returns the current number of transitions stored in the buffer.
    """
    def __init__(self, capacity, state_dim, device):
        """
        Initializes the ReplayBuffer_torch with a fixed capacity and state dimensionality.

        # Args:
        - capacity (int): The maximum number of transitions the buffer can store.
        - state_dim (int): The dimensionality of the state space.
        - device (torch.device): The device (CPU or GPU) on which the buffer's tensors are stored.
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0  

    
        self.states = torch.zeros((capacity, state_dim), device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, device=device, dtype=torch.long)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.dones = torch.zeros(capacity, dtype=torch.long, device=device)

    def append(self, state, action, reward, next_state, done):
        """
        Adds a new transition to the buffer. Overwrites the oldest transition if the buffer is full.

        # Args:
        - state (torch.Tensor): The current state.
        - action (int): The action taken.
        - reward (float): The reward received.
        - next_state (torch.Tensor): The next state.
        - done (bool): Whether the episode is done.
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = int(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position] = int(done)


        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.

        # Args:
        - batch_size (int): The number of transitions to sample.

        # Returns:
        - tuple: A tuple of tensors `(states, actions, rewards, next_states, dones)` representing a batch of sampled transitions.
        """

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current number of transitions stored in the buffer.

        # Returns:
        - int: The number of transitions in the buffer.
        """
        return self.size

# MOO特有,这个放在这里是需要的吗？
class MultiAgent_ReplayBuffer:
    """
    # Introduction
    The `MultiAgent_ReplayBuffer` class is designed for multi-agent reinforcement learning. It stores transitions for multiple agents and supports sampling chunks of transitions for training.

    # Args
    - `max_size` (int): The maximum number of transitions the buffer can hold.

    # Attributes
    - `buffer` (collections.deque): A deque object that stores the transitions with a fixed maximum size.

    # Methods
    - `append`(transition): Adds a new transition to the buffer.
    - `sample_chunk`(batch_size, chunk_size): Samples chunks of transitions for training.
    - `__len__()`: Returns the current number of transitions stored in the buffer.
    """
    def __init__(self, max_size):
        """
        Initializes the MultiAgent_ReplayBuffer with a fixed maximum size.

        # Args:
        - max_size (int): The maximum number of transitions the buffer can hold.
        """
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, transition):
        """
        Adds a new transition to the buffer.

        # Args:
        - transition (tuple): A tuple representing a transition for multiple agents.
        """
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        """
        Samples chunks of transitions for training.

        # Args:
        - batch_size (int): The number of chunks to sample.
        - chunk_size (int): The size of each chunk.

        # Returns:
        - tuple: Tensors representing sampled chunks of transitions for states, actions, rewards, next states, and done flags.
        """
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)

        n_agents, obs_size = len(s_lst[0]), len(s_lst[0][0])
        return torch.tensor(s_lst, dtype=torch.float64).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(a_lst, dtype=torch.float64).view(batch_size, chunk_size, n_agents), \
               torch.tensor(r_lst, dtype=torch.float64).view(batch_size, chunk_size, n_agents), \
               torch.tensor(s_prime_lst, dtype=torch.float64).view(batch_size, chunk_size, n_agents, obs_size), \
               torch.tensor(done_lst, dtype=torch.float64).view(batch_size, chunk_size, 1)

    def __len__(self):
        """
        Returns the current number of transitions stored in the buffer.

        # Returns:
        - int: The number of transitions in the buffer.
        """
        return len(self.buffer)

def clip_grad_norms(param_groups, max_norm = math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    # Args:
    - param_groups (list): A list of parameter groups, typically from an optimizer.
    - max_norm (float): The maximum allowable norm for gradients.

    # Returns:
    - tuple: A tuple containing lists of gradient norms before and after clipping.
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type = 2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

def save_class(dir, file_name, saving_class):
    """
    # Introduction
    Saves a Python object (class instance) to a file in pickle format.

    # Args:
    - dir (str): The directory where the file will be saved. If the directory
                   does not exist, it will be created.
    - file_name (str): The name of the file (without extension) to save the object.
    - saving_class (object): The Python object (class instance) to be saved.

    # Raises:
    - OSError: If there is an issue creating the directory or writing the file.

    # Notes:
    - The saved file will have a `.pkl` extension.
    """
    if not dir.endswith('/') or not dir.endswith('\\'):
        dir += '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir+file_name+'.pkl', 'wb') as f:
        pickle.dump(saving_class, f, -1)
