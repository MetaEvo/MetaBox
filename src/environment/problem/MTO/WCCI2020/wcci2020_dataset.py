from .wcci2020_numpy import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
from .wcci2020_torch import Sphere_Torch, Ackley_Torch, Rosenbrock_Torch, Rastrigin_Torch, Schwefel_Torch, Griewank_Torch, Weierstrass_Torch
import numpy as np
from torch.utils.data import Dataset
import os
import importlib.util
import importlib.resources as pkg_resources

class WCCI2020MTO_Tasks():
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

class WCCI2020_Dataset(Dataset):
    """
    # Introduction
      WCCI2020 proposes 10 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
    None
    # Official Implementation
      [WCCI2020](http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html)
    # License
    None
    # Problem Suite Composition
      The WCCI2020 problem suite contains a total of 10 benchmark problems, each consisting of 50 different basic functions with unique transformations(shifts and rotations).
      For each benchmark problem, fifty basic functions are added sequentially and cyclically to constitute the problem.
      These ten benchmark problems are classified according to the specific combination of different types of basic functions:
        P1: Shpere
        P2: Rosenbrock
        P3: Rastrigin
        P4: Shpere, Rosenbrock, Ackley
        P5: Rastrigin, Griewank, Weierstrass
        P6: Rosenbrock, Griewank, Schwefel
        P7: Rastrigin, Ackley, Weierstrass
        P8: Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass
        P9: Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass, Schwefel
        P10:Rastrigin, Ackley, Griewank, Weierstrass, Schwefel
    # Methods:
    - `__getitem__(item)`: Retrieves a batch of tasks based on the given index.
    - `__len__()`: Returns the total number of task datasets.
    - `__add__(other)`: Combines the current dataset with another `WCCI2020_Dataset` instance.
    - `shuffle()`: Shuffles the dataset indices to randomize the order of tasks.
    - `get_datasets(version, train_batch_size, test_batch_size, difficulty, user_train_list, user_test_list)`: A static method to generate training and testing datasets based on the specified difficulty or user-defined task lists.
    # Raises:
    - `ValueError`: Raised in the `get_datasets` method if:
        - Neither `difficulty` nor `user_train_list` and `user_test_list` are provided.
        - An invalid `difficulty` value is specified.
    """

    def __init__(self,
                 data,
                 batch_size=1):
        """
        # Introduction
        Initializes the WCCI2020 Dataset with datas.
        # Args:
        - `data` (list): A list of task datasets, where each dataset contains multiple tasks.
        - `batch_size` (int, optional): The size of each batch when retrieving data. Defaults to 1.
        # Attributes:
        - `data` (list): The dataset containing tasks.
        - `batch_size` (int): The size of each batch. Defaults to 1.
        - `maxdim` (int): The maximum dimensionality across all tasks in the dataset.
        - `N` (int): The total number of task datasets.
        - `ptr` (list): A list of indices for batching.
        - `index` (numpy.ndarray): An array of indices for shuffling and accessing data.
        """
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.maxdim = 0
        for data_lis in self.data:
            for item in data_lis.tasks:
                self.maxdim = max(self.maxdim, item.dim)
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    def __getitem__(self, item):
        """
        # Introduction
        Retrieves a batch of tasks of the the WCCI2020 benchmark suite based on the given index.
        # Args:
        - `item` (int, optional): Specifies which batch of tasks of the WCCI2020 benchmark is selected.
        # Returns:
        - list: A list containing a batch of tasks of the WCCI2020 benchmark.
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
        Returns the total number of tasks in the WCCI2020 benchmark suite.
        # Returns:
        - int: The size of the WCCI2020 benchmark suite.
        """
        return self.N

    
    def __add__(self, other: 'WCCI2020_Dataset'):
        """
        # Introduction
        Combines two datasets into a single dataset.
        # Returns:
        - Object: The combined new dataset of the WCCI2020 benchmark suite.
        """
        return WCCI2020_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly shuffles the order of tasks in the dataset.
        """
        self.index = np.random.permutation(self.N)

    @staticmethod
    def get_datasets(version='numpy',
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     user_train_list=None,
                     user_test_list=None):
        """
        # Introduction
        Generates training and testing datasets for the WCCI2020 benchmark suite based on specified difficulty or user-defined function lists.
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
        - tuple: A tuple containing two `WCCI2020_Dataset` objects:
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

        func_id = [i for i in range(0, 10)]
        if difficulty == 'easy':
            train_id = [0, 1, 2, 3, 4, 5]
            test_id = [6, 7, 8, 9]
        elif difficulty == 'difficult':
            train_id = [6, 7, 8, 9]
            test_id = [0, 1, 2, 3, 4, 5]
        elif difficulty == 'all':
            train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            test_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif difficulty is None:
            train_id = user_train_list
            test_id = user_test_list

        train_set = []
        test_set = []
        for task_ID in func_id:
            dim = 50
            task_size = 50
            choice_functions = []
            if task_ID == 0:
                choice_functions = [1]
            if task_ID == 1:
                choice_functions = [2]
            if task_ID == 2:
                choice_functions = [4]
            if task_ID == 3:
                choice_functions = [1,2,3]
            if task_ID == 4:
                choice_functions = [4,5,6]
            if task_ID == 5:
                choice_functions = [2,5,7]
            if task_ID == 6:
                choice_functions = [3,4,6]
            if task_ID == 7:
                choice_functions = [2,3,4,5,6]
            if task_ID == 8:
                choice_functions = [2,3,4,5,6,7]
            if task_ID == 9:
                choice_functions = [3,4,5,6,7]

            Tasks = []
            for task_id in range(1, task_size+1):
                id = (task_id-1) % len(choice_functions)
                func_id = choice_functions[id]
                try:
                    folder_package = f"metaevobox.environment.problem.MTO.WCCI2020.datafile.benchmark_{task_ID + 1}"
                    if importlib.util.find_spec(folder_package) is not None:
                        shift_file_path = pkg_resources.files(folder_package).joinpath(f'bias_{task_id}')
                        rotate_file_path = pkg_resources.files(folder_package).joinpath(f'matrix_{task_id}')
                        shift_file_obj = shift_file_path.open('r')
                        rotate_file_obj = rotate_file_path.open('r')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', f"benchmark_{task_ID + 1}")
                    shift_file_obj = open(os.path.join(local_path, f'bias_{task_id}'), 'r')
                    rotate_file_obj = open(os.path.join(local_path, f'matrix_{task_id}'), 'r')

                with shift_file_obj as f:
                    shift = np.loadtxt(f)
                with rotate_file_obj as f:
                    rotate_matrix = np.loadtxt(f)

                if func_id == 1:
                    if version == 'numpy': 
                        task = Sphere(dim,shift, rotate_matrix)
                    else:
                        task = Sphere_Torch(dim,shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 2:
                    if version == 'numpy': 
                        task = Rosenbrock(dim, shift, rotate_matrix)
                    else:
                        task = Rosenbrock_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 3:
                    if version == 'numpy':
                         task = Ackley(dim, shift, rotate_matrix)
                    else:
                        task = Ackley_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 4:
                    if version == 'numpy':
                        task = Rastrigin(dim, shift, rotate_matrix)
                    else:
                        task = Rastrigin_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 5:
                    if version == 'numpy':
                        task = Griewank(dim, shift, rotate_matrix)
                    else:
                        task = Griewank_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 6:
                    if version == 'numpy':
                        task = Weierstrass(dim, shift, rotate_matrix)
                    else:
                        task = Weierstrass_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)

                if func_id == 7:
                    if version == 'numpy':
                        task = Schwefel(dim, shift, rotate_matrix)
                    else:
                        task = Schwefel_Torch(dim, shift, rotate_matrix)
                    Tasks.append(task)
            
            if task_ID in train_id:
                train_set.append(WCCI2020MTO_Tasks(Tasks))
            if task_ID in test_id:
                test_set.append(WCCI2020MTO_Tasks(Tasks))

        return WCCI2020_Dataset(train_set, train_batch_size), WCCI2020_Dataset(test_set, test_batch_size)
