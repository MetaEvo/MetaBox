import random

from torch.utils.data import Dataset
import numpy as np
from .zdt_numpy import *
from .uf_numpy import *
from .wfg_numpy import *
from .dtlz_numpy import *
from .zdt_torch import *
from .uf_torch import *
from .dtlz_torch import *
from .wfg_torch import *


class MOO_Synthetic_Dataset(Dataset):
    """
    # Introduction
    The `MOO_Synthetic_Dataset` class is designed to handle synthetic multi-objective optimization (MOO) datasets. MOO-Synthetic provides a more comprehensive problem set for multi-objective optimization by combining multiple mainstream problem sets (ZDT、UF、DTLZ、WFG).
    # Problem Suite Composition
    MOO-Synthetic contains 187 questions, consisting of the ZDT, UF, DTLZ, and WFG question sets.
    - **UF (Unconstrained Functions)**: UF1 to UF10.
    - **ZDT (Zitzler-Deb Thiele)**: ZDT1 to ZDT6.
    - **DTLZ (Deb Thiele Laumanns Zitzler)**: DTLZ1 to DTLZ7.
    - **WFG (Walking Fish Group)**: WFG1 to WFG9.
    Each problem is parameterized by the number of objectives (`n_obj`) and the number of decision variables (`n_var`). The problems can be instantiated in either NumPy or PyTorch versions.
    # Args:
    - `data` (list): A list of problem instances to be included in the dataset.
    - `batch_size` (int, optional): The size of each batch. Defaults to 1.
    # Attributes:
    - `data` (list): The list of problem instances in the dataset.
    - `batch_size` (int): The size of each batch.
    - `N` (int): The total number of problem instances in the dataset.
    - `ptr` (list): A list of indices for batching.
    - `index` (numpy.ndarray): An array of shuffled indices for accessing the dataset.
    - `maxdim` (int): The maximum number of decision variables (`n_var`) across all problem instances.
    # Methods:
    - `get_datasets(version='numpy', train_batch_size=1, test_batch_size=1, difficulty=None, user_train_list=None, user_test_list=None)`: 
        Static method to generate training and testing datasets based on the specified difficulty or user-provided problem lists.
    - `__getitem__(item)`: 
        Retrieves a batch of problem instances based on the given index.
    - `__len__()`: 
        Returns the total number of problem instances in the dataset.
    - `__add__(other)`: 
        Combines the current dataset with another `MOO_Synthetic_Dataset` instance.
    - `shuffle()`: 
        Shuffles the dataset indices for random access.
    # Raises:
    - `ValueError`: Raised in the `get_datasets` method if:
        - Neither `difficulty` nor `user_train_list` and `user_test_list` are provided.
        - An invalid `difficulty` value is specified.
    """

    def __init__(self,
                 data,
                 batch_size = 1):
        """
        # Introduction
        Initializes the dataset with a list of problem instances and a batch size.

        # Args
        - `data` (list): List of problem instances.
        - `batch_size` (int, optional): Batch size for data loading. Defaults to 1.

        # Attributes
        - Sets `N`, `ptr`, `index`, and computes `maxdim` from `data`.
        """
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.n_var)

    @staticmethod
    def get_datasets(version = 'numpy',
                     train_batch_size = 1,
                     test_batch_size = 1,
                     difficulty = None,
                     user_train_list = None,
                     user_test_list = None,
                     ):
        """
        # Introduction
        Generates training and testing datasets of multi-objective optimization problems based on difficulty or user-defined lists.

        # Args
        - `version` (str, optional): Specifies the implementation ('numpy' or 'torch'). Defaults to 'numpy'.
        - `train_batch_size` (int, optional): Batch size for training dataset. Defaults to 1.
        - `test_batch_size` (int, optional): Batch size for testing dataset. Defaults to 1.
        - `difficulty` (str, optional): Difficulty level ('easy', 'difficult', 'all'). Defaults to None.
        - `user_train_list` (list, optional): User-specified training problem names. Defaults to None.
        - `user_test_list` (list, optional): User-specified testing problem names. Defaults to None.

        # Returns
        - Tuple of two `MOO_Synthetic_Dataset` instances: (train_dataset, test_dataset).

        # Raises
        - `ValueError`: If neither `difficulty` nor user lists are provided, or if difficulty is invalid.
        """
        # get functions ID of indicated suit
        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')
        instance_set = []
        train_set = []
        test_set = []
        if difficulty == None:
            for problem in user_train_list:
                parts = problem.split("_")
                problem_name = parts[0]
                n_var = int([p for p in parts if p.startswith("d")][0][1:])  
                n_obj = int([p for p in parts if p.startswith("n")][0][1:])  
                train_set.append(eval(problem_name)(n_obj = n_obj, n_var = n_var))
            for problem in user_test_list:
                parts = problem.split("_")
                problem_name = parts[0]
                n_var = int([p for p in parts if p.startswith("d")][0][1:])
                n_obj = int([p for p in parts if p.startswith("n")][0][1:])
                test_set.append(eval(problem_name)(n_obj = n_obj, n_var = n_var))
        elif difficulty != None:
            # UF1-7
            for id in range(1, 8):
                if version == 'numpy':
                    instance_set.append(eval(f"UF{id}")())
                elif version == 'torch':
                    instance_set.append(eval(f"UF{id}_Torch")())

            # UF8-10
            for id in range(8, 11):
                if version == 'numpy':
                    instance_set.append(eval(f"UF{id}")())
                elif version == 'torch':
                    instance_set.append(eval(f"UF{id}_Torch")())

            # ZDT1-3
            for id in range(1, 4):
                if version == 'numpy':
                    instance_set.append(eval(f"ZDT{id}")(n_var = 30))
                elif version == 'torch':
                    instance_set.append(eval(f"ZDT{id}_Torch")(n_var = 30))

            # ZDT4 & ZDT6
            for id in [4, 6]:
                if version == 'numpy':
                    instance_set.append(eval(f"ZDT{id}")(n_var = 10))
                elif version == 'torch':
                    instance_set.append(eval(f"ZDT{id}_Torch")(n_var = 10))

            # DTLZ1
            dtlz1_settings = {
                2: [6],
                3: [7],
                5: [9],
                7: [11],
                8: [12],
                10: [14]
            }
            for n_obj, n_var_list in dtlz1_settings.items():
                for n_var in n_var_list:
                    if version == 'numpy':
                        instance_set.append(eval("DTLZ1")(n_obj = n_obj, n_var = n_var))
                    elif version == 'torch':
                        instance_set.append(eval("DTLZ1_Torch")(n_obj = n_obj, n_var = n_var))

            # DTLZ2-6
            for dtlz_id in range(2, 7):
                dtlz_settings = {
                    2: [11],
                    3: [11, 12] if dtlz_id != 3 and dtlz_id != 5 else [12],
                    5: [14],
                    7: [16],
                    8: [17],
                    10: [19]
                }
                for n_obj, n_var_list in dtlz_settings.items():
                    for n_var in n_var_list:
                        if version == 'numpy':
                            instance_set.append(eval(f"DTLZ{dtlz_id}")(n_obj = n_obj, n_var = n_var))
                        elif version == 'torch':
                            instance_set.append(eval(f"DTLZ{dtlz_id}_Torch")(n_obj = n_obj, n_var = n_var))

            # DTLZ7
            dtlz7_settings = {
                2: [21],
                3: [22],
                5: [24],
                7: [16, 26],
                8: [27],
                10: [29]
            }
            for n_obj, n_var_list in dtlz7_settings.items():
                for n_var in n_var_list:
                    if version == 'numpy':
                        instance_set.append(eval("DTLZ7")(n_obj = n_obj, n_var = n_var))
                    elif version == 'torch':
                        instance_set.append(eval("DTLZ7_Torch")(n_obj = n_obj, n_var = n_var))

            # WFG1-9
            for wfg_id in range(1, 10):
                wfg_settings = {
                    2: [12, 22],
                    3: [12, 14, 24],
                    5: [14, 18, 28],
                    7: [16],
                    8: [24, 34],
                    10: [28, 38]
                }
                for n_obj, n_var_list in wfg_settings.items():
                    for n_var in n_var_list:
                        if version == 'numpy':
                            instance_set.append(eval(f"WFG{wfg_id}")(n_obj = n_obj, n_var = n_var))
                        elif version == 'torch':
                            instance_set.append(eval(f"WFG{wfg_id}_Torch")(n_obj = n_obj, n_var = n_var))

            print(f"Total instances: {len(instance_set)}")
            instance_set.sort(key = lambda x: x.n_obj * x.n_var)
            if difficulty == 'easy':
                train_set = instance_set[:int(0.8 * len(instance_set))]
                test_set = instance_set[int(0.8 * len(instance_set)):]
            elif difficulty == 'difficult':
                train_set = instance_set[:int(0.2 * len(instance_set))]
                test_set = instance_set[int(0.2 * len(instance_set)):]
            elif difficulty == 'all':
                train_set = instance_set
                test_set = instance_set
        for i in range(len(train_set)):
            train_set[i].dim = train_set[i].n_var
        for i in range(len(test_set)):
            test_set[i].dim = test_set[i].n_var

        return MOO_Synthetic_Dataset(train_set, train_batch_size), MOO_Synthetic_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        """
        # Introduction
        Retrieves a batch of problem instances based on batch index.

        # Args
        - `item` (int): Batch index.

        # Returns
        - List of problem instances for the batch.
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
        Returns the total number of problem instances in the dataset.

        # Returns
        - `int`: Total dataset size.
        """
        return self.N

    def __add__(self, other: 'MOO_Synthetic_Dataset'):
        """
        # Introduction
        Combines this dataset with another to form a larger dataset.

        # Args
        - `other` (MOO_Synthetic_Dataset): Another dataset instance.

        # Returns
        - A new `MOO_Synthetic_Dataset` instance containing combined data.
        """
        return MOO_Synthetic_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly permutes the internal index order for dataset shuffling.
        """
        self.index = np.random.permutation(self.N)


if __name__ == '__main__':
    train_set, test_set = MOO_Synthetic_Dataset.get_datasets()
    print(train_set, test_set)
