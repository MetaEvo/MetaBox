from .augmented_wcci2020_numpy import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
from .augmented_wcci2020_torch import Sphere_Torch, Ackley_Torch, Rosenbrock_Torch, Rastrigin_Torch, Schwefel_Torch, Griewank_Torch, Weierstrass_Torch
import numpy as np
from torch.utils.data import Dataset
import os
from itertools import combinations


def rotate_gen(dim):  # Generate a rotate matrix
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

def get_combinations():
    numbers = list(range(1, 8))     
    all_combinations = []
    for r in range(1, len(numbers) + 1):
        all_combinations.extend(combinations(numbers, r))

    sorted_combinations = sorted(all_combinations, key=len)
    combinations_list = [list(comb) for comb in sorted_combinations]
    return combinations_list


class AugmentedWCCI2020_MTO_Tasks():
    def __init__(self, tasks):
        self.tasks = tasks
        self.T1 = None
        self.dim = 0
    
    def reset(self):
        for _ in range(len(self.tasks)):
            self.dim = max(self.dim, self.tasks[_].dim)
        for _ in range(len(self.tasks)):
            self.tasks[_].reset()
        self.T1 = 0
    
    def __str__(self):
        name = ''
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            name += task.__str__()
        return name

    def update_T1(self):
        eval_time = 0
        for _ in range(len(self.tasks)):
            eval_time += self.tasks[_].T1
        self.T1 = eval_time

class Augmented_WCCI2020_Dataset(Dataset):
    """
    # Introduction
      Augmented WCCI2020 proposes 127 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
    None
    # Official Implementation
    None
    # License
    None
    # Problem Suite Composition
      The Augmented WCCI2020 problem suite contains a total of 127 benchmark problems, with each problem consisting of multiple different basic functions with unique transformations(shifts and rotations).
      The number of basic functions can be specified according to the user's requirements. Defaults to 10.
      These 127 benchmark problems are composed based on all combinations of the seven basic functions as Shpere, Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass and Schwefel.
      For each benchmark problem, the basic functions in the correspondent combination are selected randomly and added with unique transformations(shifts and rotations) until the number of basic functions is reached.
    # Methods:
    - `get_datasets(version='numpy', train_batch_size=1, test_batch_size=1, difficulty=None, user_train_list=None, user_test_list=None)`: 
        Static method to generate training and testing datasets based on the specified difficulty or user-provided task lists.
    - `__getitem__(item)`: 
        Retrieves a batch of tasks based on the given index.
    - `__len__()`: 
        Returns the total number of tasks in the dataset.
    - `__add__(other)`: 
        Combines two datasets into a single dataset.
    - `shuffle()`: 
        Randomly shuffles the order of tasks in the dataset.
    # Raises:
    - `ValueError`: Raised in the `get_datasets` method if neither `difficulty` nor `user_train_list` and `user_test_list` are provided, or if an invalid difficulty level is specified.
    """

    def __init__(self,
                 data,
                 batch_size=1):
        """
        # Introduction
        Initializes the Augmented WCCI2020 Dataset with datas.
        # Args:
        - `data` (list): A list of tasks, where each task is a list of optimization problems.
        - `batch_size` (int, optional): The number of tasks to include in each batch. Defaults to 1.
        # Attributes:
        - `data` (list): The dataset containing tasks for optimization.
        - `batch_size` (int): The size of each batch. Defaults to 1.
        - `maxdim` (int): The maximum dimensionality of the tasks in the dataset.
        - `N` (int): The total number of tasks in the dataset.
        - `ptr` (list): A list of indices for batching.
        - `index` (numpy.ndarray): An array of indices used for shuffling the dataset.
        """
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.maxdim = 0
        for data_lis in self.data:
            for item in data_lis:
                self.maxdim = max(self.maxdim, item.dim)
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)


    @staticmethod
    def get_datasets(version='numpy',
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     user_train_list=None,
                     user_test_list=None):
        """
        # Introduction
        Generates training and testing datasets for the Augmented WCCI2020 benchmark suite based on specified difficulty or user-defined function lists.
        # Args:
        - `version` (str, optional): Specifies the implementation version to use for function instances. 
          Accepts 'numpy' or any other string for alternative (e.g., 'torch'). Defaults to 'numpy'.
        - `train_batch_size` (int, optional): Batch size for the training dataset. Defaults to 1.
        - `test_batch_size` (int, optional): Batch size for the testing dataset. Defaults to 1.
        - `difficulty` (str, optional): Difficulty level for dataset split. Accepts 'easy', 'difficult', 'all', or None. 
          If None, `user_train_list` and `user_test_list` must be provided.
        - `user_train_list` (list of int, optional): List of function IDs to include in the training set. Used if `difficulty` is None.
        - `user_test_list` (list of int, optional): List of function IDs to include in the testing set. Used if `difficulty` is None.
        # Returns:
        - tuple: A tuple containing two `Augmented_WCCI2020_Dataset` objects:
            - The first is the training dataset.
            - The second is the testing dataset.
        # Raises:
        - `ValueError`: If neither `difficulty` nor both `user_train_list` and `user_test_list` are provided.
        - `ValueError`: If an invalid `difficulty` value is specified.
        """

        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')

        task_cnt = 10
        dim = 50
        combinations = get_combinations()
        combination_cnt = len(combinations)
        task_set = []
        for combination in combinations:
            ub = 0
            lb = 0
            Tasks = []
            for _ in range(task_cnt):
                func_id = np.random.choice(combination)
                if func_id == 1:
                    ub = Sphere.UB
                    lb = Sphere.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Sphere(dim, shift, rotate_matrix)
                    else:
                        task = Sphere_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 2:
                    ub = Rosenbrock.UB
                    lb = Rosenbrock.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                         task = Rosenbrock(dim, shift, rotate_matrix)
                    else:
                        task = Rosenbrock_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 3:
                    ub = Ackley.UB
                    lb = Ackley.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                         task = Ackley(dim, shift, rotate_matrix)
                    else:
                        task = Ackley_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 4:
                    ub = Rastrigin.UB
                    lb = Rastrigin.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Rastrigin(dim, shift, rotate_matrix)
                    else:
                        task = Rastrigin_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 5:
                    ub = Griewank.UB
                    lb = Griewank.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Griewank(dim, shift, rotate_matrix)
                    else:
                        task = Griewank_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 6:
                    ub = Weierstrass.UB
                    lb = Weierstrass.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Weierstrass(dim, shift, rotate_matrix)
                    else:
                        task = Weierstrass_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 7:
                    ub = Schwefel.UB
                    lb = Schwefel.LB
                    shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
                    rotate_matrix = rotate_gen(dim)
                    if version == 'numpy':
                        task = Schwefel(dim, shift, rotate_matrix)
                    else:
                        task = Schwefel_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)
            
            task_set.append(Tasks)

        if difficulty == 'easy':
            dataset_list = np.arange(0,combination_cnt)
            train_select_list = np.random.choice(dataset_list,size=int(combination_cnt*0.2), replace=False)
            test_select_list = dataset_list[~np.isin(dataset_list, train_select_list)]  
        elif difficulty == 'difficult':
            dataset_list = np.arange(0,combination_cnt)
            train_select_list = np.random.choice(dataset_list,size=int(combination_cnt*0.8), replace=False)
            test_select_list = dataset_list[~np.isin(dataset_list, train_select_list)]  
        elif difficulty == 'all':
            dataset_list = np.arange(0,combination_cnt)
            train_select_list = dataset_list
            test_select_list = dataset_list
        elif difficulty is None:
            train_select_list = user_train_list
            test_select_list = user_test_list

        train_set = [AugmentedWCCI2020_MTO_Tasks(task_set[i]) for i in train_select_list]
        test_set = [AugmentedWCCI2020_MTO_Tasks(task_set[i]) for i in test_select_list]

        return Augmented_WCCI2020_Dataset(train_set, train_batch_size), Augmented_WCCI2020_Dataset(test_set, test_batch_size)


    def __getitem__(self, item):
        """
        # Introduction
        Retrieves a batch of tasks of the the Augmented WCCI2020 benchmark suite based on the given index.
        # Args:
        - `item` (int, optional): Specifies which batch of tasks of the Augmented WCCI2020 benchmark is selected.
        # Returns:
        - list: A list containing a batch of tasks of the Augmented WCCI2020 benchmark.
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
        Returns the total number of tasks in the Augmented WCCI2020 benchmark suite.
        # Returns:
        - int: The size of the Augmented WCCI2020 benchmark suite.
        """
        return self.N

    def __add__(self, other: 'Augmented_WCCI2020_Dataset'):
        """
        # Introduction
        Combines two datasets into a single dataset.
        # Returns:
        - Object: The combined new dataset of the Augmented WCCI2020 benchmark suite.
        """
        return Augmented_WCCI2020_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly shuffles the order of tasks in the dataset.
        """
        self.index = np.random.permutation(self.N)


