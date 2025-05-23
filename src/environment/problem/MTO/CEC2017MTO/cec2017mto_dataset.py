from .cec2017mto_numpy import Sphere, Ackley, Rosenbrock, Rastrigin, Schwefel, Griewank, Weierstrass
from .cec2017mto_torch import Sphere_Torch, Ackley_Torch, Rosenbrock_Torch, Rastrigin_Torch, Schwefel_Torch,Griewank_Torch, Weierstrass_Torch
import numpy as np
from torch.utils.data import Dataset
import os
import scipy.io as sio
import importlib.util
import importlib.resources as pkg_resources

def mat2np(file_obj):
    with file_obj as f:
        data = sio.loadmat(f)
    return data


class CEC2017MTO_Tasks():
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

class CEC2017MTO_Dataset(Dataset):
    """
    # Introduction
      CEC2017MTO proposes 9 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
      "[Evolutionary Multitasking for Single-objective Continuous Optimization: Benchmark Problems, Performance Metric, and Bseline Results](https://arxiv.org/pdf/1706.03470)."
    # Official Implementation
      [CEC2017MTO](http://www.bdsc.site/websites/MTO/index.html)
    # License
    None
    # Problem Suite Composition
      The CEC2017MTO problem suite contains a total of 9 benchmark problems, each consisting of two basic functions.
      These nine benchmark problems are classified according to the degree of intersection and the inter-task similarity between the two constitutive functions:
        P1. Complete intersection and high similarity(CI+HS)
        P2. Complete intersection and medium similarity(CI+MS)  
        P3. Complete intersection and low similarity(CI+LS)
        P4. Partial intersection and high similarity(PI+HS)
        P5. Partial intersection and medium similarity(PI+MS)  
        P6. Partial intersection and low similarity(PI+LS)
        P7. No intersection and high similarity(NI+HS)
        P8. No intersection and medium similarity(NI+MS) 
        P9. No intersection and low similarity(NI+LS)
    # Methods:
    - `__getitem__(item)`: Retrieves a batch of data based on the specified index.
    - `__len__()`: Returns the total number of datasets in the collection.
    - `__add__(other)`: Combines the current dataset with another `CEC2017MTO_Dataset` instance.
    - `shuffle()`: Randomly shuffles the dataset indices.
    - `get_datasets(version, train_batch_size, test_batch_size, difficulty, user_train_list, user_test_list)`: 
        Static method to generate training and testing datasets based on the specified difficulty level or user-defined task lists.
    # Raises:
    - `ValueError`: Raised in the following cases:
        - If `difficulty`, `user_train_list`, and `user_test_list` are all `None`.
        - If an invalid `difficulty` value is provided.
    """

    def __init__(self,
                 data,
                 batch_size=1):
        """
        # Introduction
        Initializes the CEC2017MTO Dataset with datas.
        # Args:
        - `data` (list): A list of task datasets, where each dataset contains multiple tasks.
        - `batch_size` (int, optional): The size of each batch for data retrieval. Defaults to 1.
        # Attributes:
        - `data` (list): The dataset containing tasks for the CEC2017MTO problem suite.
        - `batch_size` (int): The size of each batch for data retrieval. Defaults to 1.
        - `maxdim` (int): The maximum dimensionality across all tasks in the dataset.
        - `N` (int): The total number of datasets in the collection.
        - `ptr` (list): A list of indices for batching the dataset.
        - `index` (numpy.ndarray): An array of shuffled indices for dataset access.
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
        Retrieves a batch of tasks of the the CEC2017MTO benchmark suite based on the given index.
        # Args:
        - `item` (int, optional): Specifies which batch of tasks of the CEC2017MTO benchmark is selected.
        # Returns:
        - list: A list containing a batch of tasks of the CEC2017MTO benchmark.
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
        Returns the total number of tasks in the CEC2017MTO benchmark suite.
        # Returns:
        - int: The size of the CEC2017MTO benchmark suite.
        """
        return self.N

    
    def __add__(self, other: 'CEC2017MTO_Dataset'):
        """
        # Introduction
        Combines two datasets into a single dataset.
        # Returns:
        - Object: The combined new dataset of the CEC2017MTO benchmark suite.
        """
        return CEC2017MTO_Dataset(self.data + other.data, self.batch_size)

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
        Generates training and testing datasets for the CEC2017MTO Dataset benchmark suite based on specified difficulty or user-defined function lists.
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
        - tuple: A tuple containing two `CEC2017MTO_Dataset` objects:
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

        func_id = [i for i in range(0, 9)]
        if difficulty == 'easy':
            train_id = [0, 1, 2, 3, 4, 5]
            test_id = [6, 7, 8]
        elif difficulty == 'difficult':
            train_id = [6, 7, 8]
            test_id = [0, 1, 2, 3, 4, 5]
        elif difficulty == 'all':
            train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            test_id = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        elif difficulty is None:
            train_id = user_train_list
            test_id = user_test_list
        
        train_set = []
        test_set = []
        for task_ID in func_id:
            Tasks = []
            if task_ID == 0:
                file_name = 'CI_H.mat'
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')

                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Griewank(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Griewank_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 1:
                file_name = 'CI_M.mat'
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Rastrigin_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 2:
                file_name = 'CI_L.mat'
                keys = ['GO_Task1', None, 'Rotation_Task1',None]
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel(50)
                else :
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel_Torch(50)
                Tasks = [task1, task2]

            if task_ID == 3:
                file_name = 'PI_H.mat'
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', None]
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Rastrigin(50, data[keys[0]], data[keys[2]])
                    task2 = Sphere(50, data[keys[1]])
                else:
                    task1 = Rastrigin_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Sphere_Torch(50, data[keys[1]])
                Tasks = [task1, task2]

            if task_ID == 4:
                file_name = 'PI_M.mat'
                keys = ['GO_Task1',None, 'Rotation_Task1', None]
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Rosenbrock(50)
                else:
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Rosenbrock_Torch(50)
                Tasks = [task1, task2]

            if task_ID == 5:
                file_name = 'PI_L.mat'
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Ackley(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass(25, data[keys[1]], data[keys[3]])
                else:
                    task1 = Ackley_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass_Torch(25, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 6:
                file_name = 'NI_H.mat'
                keys = [None, 'GO_Task2', None, 'Rotation_Task2']
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Rosenbrock(50)
                    task2 = Rastrigin(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Rosenbrock_Torch(50)
                    task2 = Rastrigin_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]
            
            if task_ID == 7:
                file_name = 'NI_M.mat'
                keys = ['GO_Task1', 'GO_Task2', 'Rotation_Task1', 'Rotation_Task2']
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Griewank(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass(50, data[keys[1]], data[keys[3]])
                else:
                    task1 = Griewank_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Weierstrass_Torch(50, data[keys[1]], data[keys[3]])
                Tasks = [task1, task2]

            if task_ID == 8:
                file_name = 'NI_L.mat'
                keys = ['GO_Task1',None, 'Rotation_Task1',None]
                try:
                    folder_dir = 'metaevobox.environment.problem.MTO.CEC2017MTO.datafile'
                    if importlib.util.find_spec(folder_dir) is not None:
                        file_path = pkg_resources.files(folder_dir).joinpath(file_name)
                        file_obj = file_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', file_name)
                    file_obj = open(local_path, 'rb')
                data = mat2np(file_obj)
                if version == 'numpy':
                    task1 = Rastrigin(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel(50)
                else:
                    task1 = Rastrigin_Torch(50, data[keys[0]], data[keys[2]])
                    task2 = Schwefel_Torch(50)
                Tasks = [task1, task2]
            
            if task_ID in train_id:
                train_set.append(CEC2017MTO_Tasks(Tasks))
            if task_ID in test_id:
                test_set.append(CEC2017MTO_Tasks(Tasks))

        return CEC2017MTO_Dataset(train_set, train_batch_size), CEC2017MTO_Dataset(test_set, test_batch_size)
