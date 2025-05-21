import torch
import numpy as np
from torch.utils.data import Dataset
from ....problem.basic_problem import Basic_Problem
import time
from .protein_docking import Protein_Docking_Torch_Problem, Protein_Docking_Numpy_Problem
import importlib.util
import importlib.resources as pkg_resources
import os
class Protein_Docking_Dataset(Dataset):
    """
    # Attributes:
    - proteins_set (dict): Dictionary containing protein IDs categorized by difficulty ('rigid', 'medium', 'difficult').
    - n_start_points (int): Number of starting models per protein (default: 10).
    - data (list): List of problem instances.
    - batch_size (int): Number of instances per batch.
    - N (int): Total number of problem instances.
    - ptr (list): List of starting indices for each batch.
    - index (np.ndarray): Array of indices for data access and shuffling.
    - maxdim (int): Maximum dimension among all problem instances.
    # Methods:
    - __init__(self, data, batch_size=1): Initializes the dataset with provided data and batch size.
    - get_datasets(version, train_batch_size=1, test_batch_size=1, difficulty='easy', dataset_seed=1035): 
        Static method to generate training and testing datasets based on difficulty and random seed.
    - __getitem__(self, item): Returns a batch of problem instances at the specified batch index.
    - __len__(self): Returns the total number of problem instances.
    - __add__(self, other): Concatenates two Protein_Docking_Dataset instances.
    - shuffle(self): Randomly permutes the order of the dataset.
    # Args:
    - data (list): List of problem instances to be included in the dataset.
    - batch_size (int, optional): Number of instances per batch (default: 1).
    # Returns:
    - Protein_Docking_Dataset: An instance of the dataset for protein docking problems.
    # Raises:
    - ValueError: If an unsupported difficulty or version is provided in `get_datasets`.
    """
    
    proteins_set = {'rigid': ['1AVX', '1BJ1', '1BVN', '1CGI', '1DFJ', '1EAW', '1EWY', '1EZU', '1IQD', '1JPS',
                              '1KXQ', '1MAH', '1N8O', '1PPE', '1R0R', '2B42', '2I25', '2JEL', '7CEI', '1AY7'],
                    'medium': ['1GRN', '1IJK', '1M10', '1XQS', '2HRK'],
                    'difficult': ['1ATN', '1IBR', '2C0L']
                    }
    n_start_points = 10  # top models from ZDOCK

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
    def get_datasets(version,
                     train_batch_size=1,
                     test_batch_size=1,
                     user_train_list = None,
                     user_test_list = None,
                     difficulty='easy',
                     dataset_seed=1035):
        # apart train set and test set
        if difficulty == 'easy':
            train_set_ratio = 0.75
        elif difficulty == 'difficult':
            train_set_ratio = 0.25
        else:
            train_set_ratio = 0 # 全在test上
        rng = np.random.RandomState(dataset_seed)
        train_proteins_set = []
        test_proteins_set = []
        for key in Protein_Docking_Dataset.proteins_set.keys():
            permutated = rng.permutation(Protein_Docking_Dataset.proteins_set[key])
            n_train_proteins = max(1, min(int(len(permutated) * train_set_ratio), len(permutated) - 1))
            train_proteins_set.extend(permutated[:n_train_proteins])
            test_proteins_set.extend(permutated[n_train_proteins:])
        # construct problem instances
        train_set = []
        test_set = []
        instance_list = []
        for id in train_proteins_set + test_proteins_set:
            tmp_set = []
            for j in range(Protein_Docking_Dataset.n_start_points):
                problem_id = id + '_' + str(j + 1)

                try:
                    data_folder = 'metaevobox.environment.problem.SOO.PROTEIN_DOCKING.datafile'
                    if importlib.util.find_spec(data_folder) is not None:
                        f = pkg_resources.files(data_folder).joinpath(problem_id)
                        open_fn = lambda filename: f.joinpath(filename).open('r')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    baseline_path = os.path.dirname(os.path.abspath(__file__))
                    data_folder = os.path.join(baseline_path, "datafile", f"{problem_id}")
                    open_fn = lambda filename: open(os.path.join(data_folder, filename), 'r')

                coor_init = np.loadtxt(open_fn('coor_init'))
                q = np.loadtxt(open_fn('q'))
                e = np.loadtxt(open_fn('e'))
                r = np.loadtxt(open_fn('r'))
                basis = np.loadtxt(open_fn('basis'))
                eigval = np.loadtxt(open_fn('eigval'))


                q = np.tile(q, (1, 1))
                e = np.tile(e, (1, 1))
                r = np.tile(r, (len(r), 1))

                q = np.matmul(q.T, q)
                e = np.sqrt(np.matmul(e.T, e))
                r = (r + r.T) / 2
                if version == 'numpy':
                    tmp_set.append(Protein_Docking_Numpy_Problem(coor_init, q, e, r, basis, eigval, problem_id))
                elif version == 'torch':
                    tmp_set.append(Protein_Docking_Torch_Problem(coor_init, q, e, r, basis, eigval, problem_id))
                else:
                    raise ValueError(f'{version} version is invalid or is not supported yet.')
            if difficulty == "all":
                instance_list.extend(tmp_set)
            if user_train_list is None and user_test_list is None:
                if id in train_proteins_set:
                    train_set.extend(tmp_set)
                elif id in test_proteins_set:
                    test_set.extend(tmp_set)
            else:
                if user_train_list is not None and user_test_list is not None:
                    if id in user_train_list:
                        train_set.extend(tmp_set)
                    if id in user_test_list:
                        test_set.extend(tmp_set)
                elif user_train_list is not None:
                    if id in user_train_list:
                        train_set.extend(tmp_set)
                    else:
                        test_set.extend(tmp_set)
                elif user_test_list is not None:
                    if id in user_test_list:
                        test_set.extend(tmp_set)
                    else:
                        train_set.extend(tmp_set)
        if difficulty == 'all':
            train_set = instance_list.copy()
            test_set = instance_list.copy()

        return Protein_Docking_Dataset(train_set, train_batch_size), Protein_Docking_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Protein_Docking_Dataset'):
        return Protein_Docking_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)
