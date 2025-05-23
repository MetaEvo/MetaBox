from .uav_numpy import Terrain as Terrain_Numpy
from torch.utils.data import Dataset
from .utils import createmodel
import numpy as np
import pickle

class UAV_Dataset(Dataset):
    """
    # Introduction
    UAV provides 56 terrain-based landscapes as realistic Unmanned Aerial Vehicle(UAV) path planning problems, each of which is 30D. The objective is to select given number of path nodes (x,y,z coordinates) from the 3D space, so the the UAV could fly as shortly as possible in a collision-free way.
    # Original paper
    "[Benchmarking global optimization techniques for unmanned aerial vehicle path planning.](https://arxiv.org/abs/2501.14503)" arXiv preprint arXiv:2501.14503 (2025).
    # Official Implementation
    [UAV](https://zenodo.org/records/12793991)
    # License
    None
    """
    
    def __init__(self, data, batch_size = 1):
        """
        # Introduction
        Initializes the dataset object for UAV problems, setting up batching and dimension tracking.
        # Args:
        - data (list): A list of data items, each expected to have a `dim` attribute.
        - batch_size (int, optional): The number of samples per batch. Defaults to 1.
        # Built-in Attribute:
        - data (list): Stores the input data.
        - batch_size (int): Stores the batch size.
        - N (int): The total number of data items.
        - ptr (list): List of starting indices for each batch.
        - index (np.ndarray): Array of indices for the data.
        - maxdim (int): The maximum dimension found among all data items.
        # Returns:
        - None
        # Raises:
        - AttributeError: If any item in `data` does not have a `dim` attribute.
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
        Generates and returns training and testing datasets for UAV (Unmanned Aerial Vehicle) terrain navigation problems, supporting both standard (from file) and custom (generated) modes. Allows flexible selection of dataset difficulty, batch sizes, and custom train/test splits.
        # Args:
        - version (str, optional): Dataset version, either 'numpy' or 'torch'. Defaults to 'numpy'.
        - train_batch_size (int, optional): Batch size for the training dataset. Defaults to 1.
        - test_batch_size (int, optional): Batch size for the testing dataset. Defaults to 1.
        - difficulty (str, optional): Difficulty level of the dataset. One of ['easy', 'difficult', 'all', None]. If None, user_train_list and user_test_list must be provided.
        - user_train_list (list of int, optional): Custom list of instance indices for the training set. Defaults to None.
        - user_test_list (list of int, optional): Custom list of instance indices for the testing set. Defaults to None.
        - dv (float, optional): Parameter for terrain data (e.g., number of divisions). Defaults to 5.0.
        - j_pen (float, optional): Penalty parameter for the terrain data. Defaults to 1e4.
        - seed (int, optional): Random seed for reproducibility. Defaults to 3849.
        - num (int, optional): Total number of terrain instances to generate or load. Defaults to 56.
        - mode (str, optional): Dataset mode, either 'standard' (load from file) or 'custom' (generate on the fly). Defaults to "standard".
        - path (str, optional): Path to the dataset file (required if mode is "standard"). Defaults to None.
        # Returns:
        - UAV_Dataset: Training dataset object.
        - UAV_Dataset: Testing dataset object.
        # Raises:
        - ValueError: If neither `difficulty` nor both `user_train_list` and `user_test_list` are provided.
        - ValueError: If `difficulty` is not one of ['easy', 'difficult', 'all', None].
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
        Retrieves a batch of data samples corresponding to the specified index.
        # Args:
        - item (int): The index of the batch to retrieve.
        # Returns:
        - list: A list containing the data samples for the specified batch.
        # Raises:
        - IndexError: If `item` is out of range of the available batches.
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

    def __add__(self, other: 'UAV_Dataset'):
        """
        # Introduction
        Combines two UAV_Dataset instances by concatenating their data attributes.
        # Args:
        - other (UAV_Dataset): Another UAV_Dataset instance to be added.
        # Returns:
        - UAV_Dataset: A new UAV_Dataset instance containing the combined data from both datasets, with the same batch size as the original.
        # Raises:
        - AttributeError: If `other` does not have a `data` attribute.
        - TypeError: If `other` is not an instance of UAV_Dataset.
        """
        
        return UAV_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly shuffles the indices of the dataset to change the order of data access.
        # Built-in Attribute:
        - self.N (int): The total number of data samples in the dataset.
        # Returns:
        - None
        # Side Effects:
        - Updates `self.index` with a new permutation of indices for the dataset.
        """
        self.index = np.random.permutation(self.N)
