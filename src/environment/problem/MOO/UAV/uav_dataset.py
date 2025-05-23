from .uav_numpy import Terrain as Terrain_Numpy
from torch.utils.data import Dataset
from .utils import createmodel
import numpy as np
import pickle

class UAV_Dataset(Dataset):
    """
    # Introduction
    The `UAV_Dataset` class is designed to handle datasets for Unmanned Aerial Vehicle (UAV) optimization problems. 
    # Original Paper
    "[Benchmarking global optimization techniques for unmanned aerial vehicle path planning](https://arxiv.org/abs/2501.14503)." 
    # Official Implementation
    None
    # License
    None
    # Problem Suite Composition
    The UAV dataset is composed of instances that simulate UAV optimization problems. The dataset can be configured to include different levels of difficulty (`easy`, `difficult`, or `all`) and supports both user-defined and random splits for training and testing. The dataset can be generated using surrogate models or custom terrain generation.
    # Args:
    - `data` (list): A list of data instances representing the UAV optimization problems.
    - `batch_size` (int, optional): The size of each batch for data retrieval. Defaults to 1.
    # Attributes:
    - `data` (list): The dataset containing UAV problem instances.
    - `batch_size` (int): The size of each batch for data retrieval.
    - `N` (int): The total number of instances in the dataset.
    - `ptr` (list): A list of pointers for batching.
    - `index` (numpy.ndarray): An array of indices for shuffling and batching.
    - `maxdim` (int): The maximum dimensionality of the problem instances in the dataset.
    # Methods:
    - `get_datasets(version, train_batch_size, test_batch_size, difficulty, user_train_list, user_test_list, dv, j_pen, seed, num, mode, path)`: 
        Static method to generate training and testing datasets based on the specified parameters.
    - `__getitem__(item)`: 
        Retrieves a batch of data instances based on the given index.
    - `__len__()`: 
        Returns the total number of instances in the dataset.
    - `__add__(other)`: 
        Combines the current dataset with another `UAV_Dataset` instance.
    - `shuffle()`: 
        Shuffles the dataset indices to randomize the order of data retrieval.
    # Raises:
    - `ValueError`: Raised in `get_datasets` if `difficulty`, `user_train_list`, or `user_test_list` are not properly set or if an invalid difficulty level is provided.
    """

    def __init__(self, data, batch_size = 1):
        """
        # Introduction
        Initialize the UAV dataset with problem instances and batch size.

        # Args
        - `data` (list): List of UAV problem instances.
        - `batch_size` (int, optional): Batch size. Defaults to 1.

        # Attributes
        - Sets `N`, `ptr`, `index`, and computes `maxdim` based on problem dimensionality.
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
    def get_datasets(version = 'numpy',
                     train_batch_size = 1,
                     test_batch_size = 1,
                     difficulty = None,
                     user_train_list = None,
                     user_test_list = None,
                     dv = 5.0,
                     j_pen = 1e4,
                     seed = 3849,
                     num = 56,
                     mode = "standard",
                     path = None
                     ):
        """
        # Introduction
        Generate training and testing UAV datasets according to difficulty, user-specified splits, and generation modes.

        # Args
        - `version` (str, optional): 'numpy' or 'torch' implementation version. Defaults to 'numpy'.
        - `train_batch_size` (int, optional): Batch size for training data. Defaults to 1.
        - `test_batch_size` (int, optional): Batch size for testing data. Defaults to 1.
        - `difficulty` (str, optional): Difficulty level ('easy', 'difficult', 'all'). Defaults to None.
        - `user_train_list` (list, optional): List of training instance IDs specified by user. Defaults to None.
        - `user_test_list` (list, optional): List of testing instance IDs specified by user. Defaults to None.
        - `dv` (float, optional): Dimensionality parameter for problem instances. Defaults to 5.0.
        - `j_pen` (float, optional): Penalty parameter for objective function. Defaults to 1e4.
        - `seed` (int, optional): Random seed for reproducibility. Defaults to 3849.
        - `num` (int, optional): Total number of problem instances. Defaults to 56.
        - `mode` (str, optional): Dataset generation mode ('standard' or 'custom'). Defaults to "standard".
        - `path` (str, optional): File path to precomputed data for 'standard' mode.

        # Returns
        - Tuple of two `UAV_Dataset` instances: (train_dataset, test_dataset).

        # Raises
        - `ValueError`: If difficulty or user lists are not set correctly or invalid difficulty string provided.
        """
        # easy 15 diff 30
        easy_id = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54] # 28
        diff_id = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55] # 28
        # if easy |train| = 42 = 14 + 28(easy + diff)
        # if diff |train| = 42 = 28 + 14(easy + diff)

        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')

        train_set = []
        test_set = []
        rng = np.random.RandomState(seed)

        Terrain = "Terrain_Numpy" if version == "numpy" else "Terrain_Torch"

        train_len = int(num * 0.75)  # if num = 56 train_len = 42
        test_len = num - train_len

        if difficulty == 'easy':
            # 训练集先尽量从 diff_id 取（最多28个）
            if train_len <= len(diff_id):
                train_diff = list(rng.choice(diff_id, train_len, replace = False))
                train_easy = []
            else:
                train_diff = diff_id.copy()
                train_easy = list(rng.choice(easy_id, train_len - len(diff_id), replace = False))
            train_id = train_diff + train_easy

            # 测试集从 easy_id 取出剩下没用于 train 的那部分
            remaining_easy = list(set(easy_id) - set(train_easy))
            test_easy_count = min(test_len, len(remaining_easy))
            test_easy = list(rng.choice(remaining_easy, test_easy_count, replace = False))

            test_id = test_easy
        elif difficulty == 'difficult':
            # 训练集先尽量从 easy_id 取（最多28个）
            if train_len <= len(easy_id):
                train_easy = list(rng.choice(easy_id, train_len, replace = False))
                train_diff = []
            else:
                train_easy = easy_id.copy()
                train_diff = list(rng.choice(diff_id, train_len - len(easy_id), replace = False))
            train_id = train_diff + train_easy

            # 测试集从 diff_id 取出剩下没用于 train 的那部分
            remaining_diff = list(set(diff_id) - set(train_diff))
            test_diff_count = min(test_len, len(remaining_diff))
            test_diff = list(rng.choice(remaining_diff, test_diff_count, replace = False))

            test_id = test_diff

        instance_list = []

        if mode == "standard":
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            for id in range(num):
                terrain_data = model_data[id]
                terrain_data['n'] = dv
                terrain_data['J_pen'] = j_pen
                instance = eval(Terrain)(terrain_data, id + 1)

                if difficulty == 'all':
                    instance_list.append(instance)
                    continue

                if user_train_list == None and user_test_list == None:
                    if id in train_id:
                        train_set.append(instance)
                    if id in test_id:
                        test_set.append(instance)

                elif user_train_list is not None and user_test_list is not None:
                    if id in user_train_list:
                        train_set.append(instance)
                    elif id in user_test_list:
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

                # if id in train_id:
                #     if not user_train_list or id in user_train_list:
                #         terrain_data = model_data[id]
                #         terrain_data['n'] = dv
                #         terrain_data['J_pen'] = j_pen
                #         instance = eval(Terrain)(terrain_data, id + 1)
                #         train_set.append(instance)
                #
                # if id in test_id:
                #     if not user_test_list or id in user_test_list:
                #         terrain_data = model_data[id]
                #         terrain_data['n'] = dv
                #         terrain_data['J_pen'] = j_pen
                #         instance = eval(Terrain)(terrain_data, id + 1)
                #         test_set.append(instance)

        elif mode == "custom":
            for id in range(num):
                if id < 0.5 * num:
                    num_threats = 15
                else:
                    num_threats = 30
                terrain_data = createmodel(map_size = 900,
                                           r = rng.rand() * 600 - 200, #[-200, 400]
                                           rr = rng.rand() * 800 + 100, # [100, 900]
                                           num_threats = num_threats,
                                           rng = rng)
                terrain_data['n'] = dv
                terrain_data['J_pen'] = j_pen
                instance = eval(Terrain)(terrain_data, id + 1)

                if difficulty == 'all':
                    instance_list.append(instance)
                    continue

                if user_train_list == None and user_test_list == None:
                    if id in train_id:
                        train_set.append(instance)
                    if id in test_id:
                        test_set.append(instance)

                elif user_train_list is not None and user_test_list is not None:
                    if id in user_train_list:
                        train_set.append(instance)
                    elif id in user_test_list:
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

        return UAV_Dataset(train_set, train_batch_size), \
               UAV_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        """
        # Introduction
        Retrieve a batch of UAV problem instances by batch index.

        # Args
        - `item` (int): Batch index.

        # Returns
        - List of UAV problem instances for the batch.
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
        Return the total number of UAV problem instances.

        # Returns
        - `int`: Total dataset size.
        """
        return self.N

    def __add__(self, other: 'UAV_Dataset'):
        """
        # Introduction
        Combine two UAV_Dataset instances into one.

        # Args
        - `other` (UAV_Dataset): Another UAV_Dataset instance.

        # Returns
        - A new UAV_Dataset instance containing combined data.
        """
        return UAV_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly permute the dataset indices for shuffling.

        """
        self.index = np.random.permutation(self.N)
