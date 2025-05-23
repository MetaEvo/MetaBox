from torch.utils.data import Dataset
import sys
import subprocess
import numpy as np
from .evox_ne import *


class NE_Dataset(Dataset):
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
    
    def __init__(self,
                 data,
                 batch_size=1):
        """
        Initializes the dataset object for single-objective optimization (SOO) problems.
        # Args:
        - data (list): A list of data items, where each item is expected to have a `dim` attribute.
        - batch_size (int, optional): The number of samples per batch. Defaults to 1.
        # Built-in Attributes:
        - data (list): Stores the input data.
        - batch_size (int): Stores the batch size.Defaults to 1.
        - N (int): The total number of data items.
        - ptr (list): List of starting indices for each batch.
        - index (np.ndarray): Array of indices for the data items.
        - maxdim (int): The maximum dimension found among all data items.Defaults to 0.
        # Notes:
        Iterates through the data to determine the maximum dimension (`maxdim`) among all items.
        """
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)

    @staticmethod
    def get_datasets(
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty='easy',
                     user_train_list = None,
                     user_test_list = None,
                     instance_seed=3849):
        """
        # Introduction
        Generates training and testing datasets for the NE_Problem environment based on specified difficulty or user-provided lists.
        # Args:
        - train_batch_size (int, optional): Batch size for the training dataset. Defaults to 1.
        - test_batch_size (int, optional): Batch size for the testing dataset. Defaults to 1.
        - difficulty (str, optional): Difficulty level of the datasets to generate. Must be one of ['all', 'easy', 'difficult']. Defaults to 'easy'.
        - user_train_list (list of str, optional): List of environment-depth identifiers to include in the training set. If provided, overrides `difficulty` for training set selection.
        - user_test_list (list of str, optional): List of environment-depth identifiers to include in the testing set. If provided, overrides `difficulty` for testing set selection.
        - instance_seed (int, optional): Random seed for instance generation. Defaults to 3849.
        # Returns:
        - NE_Dataset: The training dataset.
        - NE_Dataset: The testing dataset.
        # Raises:
        - AssertionError: If `difficulty` is not one of ['all', 'easy', 'difficult'].
        """
        assert difficulty in ['all','easy','difficult']
        train_set = []
        test_set = []
        if difficulty == 'all':
            for env in envs.keys():
                for depth in model_depth:
                    train_set.append(NE_Problem(env, depth, instance_seed))
                    test_set.append(NE_Problem(env, depth, instance_seed))
        elif user_train_list is not None or user_test_list is not None:
            for env in envs.keys():
                for depth in model_depth:
                    if user_train_list is not None and user_test_list is not None:
                        if f'{env}_{depth}' in user_train_list:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                        if f'{env}_{depth}' in user_test_list:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                    elif user_train_list is not None:
                        if f'{env}_{depth}' in user_train_list:
                            train_set.append(NE_Problem(env, depth, instance_seed))
                        else:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                    elif user_test_list is not None:
                        if f'{env}_{depth}' in user_test_list:
                            test_set.append(NE_Problem(env, depth, instance_seed))
                        else:
                            train_set.append(NE_Problem(env, depth, instance_seed))
        elif difficulty == 'easy':
            for env in envs.keys():
                for depth in model_depth:
                    if depth <=2:
                        test_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        train_set.append(NE_Problem(env, depth, instance_seed))
        elif difficulty == 'difficult':
            for env in envs.keys():
                for depth in model_depth:
                    if depth <=2:
                        train_set.append(NE_Problem(env, depth, instance_seed))
                    else:
                        test_set.append(NE_Problem(env, depth, instance_seed))
        return NE_Dataset(train_set, train_batch_size), NE_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        """
        # Introduction
        Retrieves a batch of data samples corresponding to the given index.
        # Args:
        - item (int): The index of the batch to retrieve.
        # Built-in Attribute:
        - self.ptr (list or array-like): Maps batch indices to starting positions in the dataset.
        - self.index (list or array-like): Contains indices of the data samples.
        - self.batch_size (int): The number of samples in each batch.
        - self.N (int): The total number of data samples.
        - self.data (list or array-like): The dataset from which samples are retrieved.
        # Returns:
        - list: A list containing the data samples for the specified batch.
        # Raises:
        - IndexError: If `item` is out of range of available batches.
        """
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        """
        # Introduction
        Returns the number of elements in the dataset.
        # Returns:
        - int: The total number of elements in the dataset.
        """
        return self.N

    def __add__(self, other: 'NE_Dataset'):
        """
        # Introduction
        Combines two `NE_Dataset` instances by adding their data attributes and returns a new `NE_Dataset` with the same batch size.
        # Args:
        - other (NE_Dataset): Another `NE_Dataset` instance to be added.
        # Returns:
        - NE_Dataset: A new dataset containing the combined data of both instances.
        # Raises:
        - AttributeError: If `other` does not have a `data` attribute.
        """
        return NE_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly shuffles the indices of the dataset, updating the internal index array.
        # Built-in Attribute:
        - self.N (int): The number of elements in the dataset.
        - self.index (np.ndarray): The array storing the current order of indices.
        # Returns:
        - None
        # Notes:
        This method uses `np.random.permutation` to generate a new random ordering of indices for the dataset.
        """
        self.index = np.random.permutation(self.N)

