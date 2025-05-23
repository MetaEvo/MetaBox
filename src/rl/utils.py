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

    # Raises:

    This class does not raise any exceptions.
    """
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
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
    - `append(exp)`: Adds a new experience to the buffer.
    - `sample(batch_size)`: Samples a mini-batch of experiences from the buffer.
    - `__len__()`: Returns the current number of experiences stored in the buffer.
    # Raises
    - `ValueError`: Raised in the `sample` method if the requested batch size exceeds the number of stored experiences.
    """
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
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
    - `append(state, action, reward, next_state, done)`: Adds a new transition to the buffer. Overwrites the oldest transition if the buffer is full.
    - `sample(batch_size)`: Samples a batch of transitions from the buffer.
    - `__len__()`: Returns the current number of transitions stored in the buffer.
    # Returns
    - `sample(batch_size)`: Returns a tuple of tensors `(states, actions, rewards, next_states, dones)` representing a batch of sampled transitions.
    # Raises
    - No explicit exceptions are raised by this class.
    """
    def __init__(self, capacity, state_dim, device):

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

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = int(reward)
        self.next_states[self.position] = next_state
        self.dones[self.position] = int(done)


        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        states = self.states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        next_states = self.next_states[indices]
        dones = self.dones[indices]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

# MOO特有,这个放在这里是需要的吗？
class MultiAgent_ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, transition):
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
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
        return len(self.buffer)

def clip_grad_norms(param_groups, max_norm = math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
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
