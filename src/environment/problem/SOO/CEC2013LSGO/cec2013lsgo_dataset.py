import numpy as np
from torch.utils.data import Dataset
from .cec2013lsgo_numpy import *
from .cec2013lsgo_torch import *
class CEC2013LSGO_Dataset(Dataset):
    """
    # Introduction
      CEC2013LSGO proposes 15 large-scale benchmark problems to represent a wider range of realworld large-scale optimization problems.
    # Original paper
      "[Benchmark functions for the CEC 2013 special session and competition on large-scale global optimization](https://al-roomi.org/multimedia/CEC_Database/CEC2015/LargeScaleGlobalOptimization/CEC2015_LargeScaleGO_TechnicalReport.pdf)." gene 7.33 (2013): 8.
    # Official Implementation
    [CEC2013LSGO](https://github.com/dmolina/cec2013lsgo)
    # License
    GPL-3.0
    # Problem Suite Composition
      CEC2013LSGO contains four major categories of large-scale problems:
      1. Fully-separable functions (F1-F3) 
      2. Two types of partially separable functions: 
          1. Partially separable functions with a set of non-separable subcomponents and one fully-separable subcomponents (F4-F7) 
          2. Partially separable functions with only a set of non-separable subcomponents and no fullyseparable subcomponent (F8-F11) 
      3. Two types of overlapping functions: 
          1. Overlapping functions with conforming subcomponents (F12-F13)
          2. Overlapping functions with conflicting subcomponents (F14)
      4. Fully-nonseparable functions (F15)

    """
    
    def __init__(self,
                 data,
                 batch_size=1):
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
    def get_datasets(version='numpy',
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     user_train_list=None,
                     user_test_list=None):
        """
        # Introduction
        Generates training and testing datasets for the CEC2013 LSGO benchmark suite based on specified difficulty or user-defined function lists.
        # Args:
        - version (str, optional): Specifies the implementation version to use for function instances. 
          Accepts 'numpy' or any other string for alternative (e.g., 'torch'). Defaults to 'numpy'.
        - train_batch_size (int, optional): Batch size for the training dataset. Defaults to 1.
        - test_batch_size (int, optional): Batch size for the testing dataset. Defaults to 1.
        - difficulty (str, optional): Difficulty level for dataset split. Accepts 'easy', 'difficult', 'all', or None. 
          If None, `user_train_list` and `user_test_list` must be provided.
        - user_train_list (list of int, optional): List of function IDs to include in the training set. Used if `difficulty` is None.
        - user_test_list (list of int, optional): List of function IDs to include in the testing set. Used if `difficulty` is None.
        # Returns:
        - tuple: A tuple containing two `CEC2013LSGO_Dataset` objects:
            - The first is the training dataset.
            - The second is the testing dataset.
        # Raises:
        - ValueError: If neither `difficulty` nor both `user_train_list` and `user_test_list` are provided.
        - ValueError: If an invalid `difficulty` value is specified.
        """
        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all']:
            raise ValueError(f'{difficulty} difficulty is invalid.')
        func_id = [i for i in range(1, 16)]
        train_set = []
        test_set = []

        if difficulty == 'easy':
            train_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif difficulty == 'difficult':
            train_id = [7, 8, 9, 10, 11, 12, 13, 14, 15]

        instance_list = []
        for id in func_id:
            if version == 'numpy':
                instance = eval(f"F{id}")()
            else:
                instance = eval(f"F{id}_Torch")()

            if difficulty == "all":
                instance_list.append(instance)
                continue

            if user_train_list is None and user_test_list is None:
                if id in train_id:
                    train_set.append(instance)
                else:
                    test_set.append(instance)
            else:
                if user_train_list is not None and user_test_list is not None:
                    if id in user_train_list:
                        train_set.append(instance)
                    if id in user_test_list:
                        test_set.append(instance)
                elif user_train_list is not None:
                    if id in user_train_list:
                        train_set.append(instance)
                    else:
                        test_set.append(instance)
                elif user_test_list is not None:
                    if id in user_test_list:
                        test_set.append(instance)
                    else:
                        train_set.append(instance)

        if difficulty == 'all':
            train_set = instance_list.copy()
            test_set = instance_list.copy()

        return CEC2013LSGO_Dataset(train_set, train_batch_size), CEC2013LSGO_Dataset(test_set, test_batch_size)

    # get a batch of data
    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    # get the number of data
    def __len__(self):
        return self.N

    def __add__(self, other: 'CEC2013LSGO_Dataset'):
        return CEC2013LSGO_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
