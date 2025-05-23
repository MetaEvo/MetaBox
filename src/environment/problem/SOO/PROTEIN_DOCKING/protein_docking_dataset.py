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
    # Introduction
    
    Protein-Docking benchmark, where the objective is to minimize the Gibbs free energy resulting from protein-protein interaction between a given complex and any other conformation. We select 28 protein complexes and randomly initialize 10 starting points for each complex, resulting in 280 problem instances. To simplify the problem structure, we only optimize 12 interaction points in a complex instance (12D problem).

    # Original paper
    "[Protein–protein docking benchmark version 4.0.](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22830)" Proteins: Structure, Function, and Bioinformatics 78.15 (2010): 3111-3114.
    # Official Implementation
    [Protein-Docking](https://zlab.wenglab.org/benchmark/)
    # License
    None
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
        """
        Initializes the protein docking dataset object with provided data and batch size.
        # Args:
        - data (list): A list of data items, each expected to have a `dim` attribute.
        - batch_size (int, optional): The number of samples per batch. Defaults to 1.
        # Built-in Attributes:
        - data (list): Stores the input data.
        - batch_size (int): Stores the batch size.
        - N (int): The total number of data items.
        - ptr (list): List of starting indices for each batch.
        - index (np.ndarray): Array of indices for the data items.Defaults to a range from 0 to N.
        - maxdim (int): The maximum `dim` value among all data items.Defaults to 0.
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
    def get_datasets(version,
                     train_batch_size=1,
                     test_batch_size=1,
                     user_train_list = None,
                     user_test_list = None,
                     difficulty='easy',
                     dataset_seed=1035):
        """
        # Introduction
        Generates training and testing datasets for the protein docking problem, partitioning protein instances based on the specified difficulty level or user-provided lists. Supports both NumPy and PyTorch problem representations.
        # Args:
        - version (str): The backend to use for problem instances. Must be either 'numpy' or 'torch'.
        - train_batch_size (int, optional): Batch size for the training dataset. Defaults to 1.
        - test_batch_size (int, optional): Batch size for the testing dataset. Defaults to 1.
        - user_train_list (list, optional): List of protein IDs to include in the training set. If None, uses automatic partitioning. Defaults to None.
        - user_test_list (list, optional): List of protein IDs to include in the testing set. If None, uses automatic partitioning. Defaults to None.
        - difficulty (str, optional): Difficulty level for dataset partitioning. Can be 'easy', 'difficult', or 'all'. Defaults to 'easy'.
        - dataset_seed (int, optional): Random seed for reproducible dataset partitioning. Defaults to 1035.
        # Returns:
        - Protein_Docking_Dataset: The training dataset.
        - Protein_Docking_Dataset: The testing dataset.
        # Raises:
        - ValueError: If the specified `version` is not supported.
        """
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
        """
        # Introduction
        Retrieves a batch of data samples corresponding to the given index.
        # Args:
        - item (int): The batch index to retrieve data for.
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
        Returns the number of items in the dataset.
        # Returns:
        - int: The total number of items in the dataset.
        """
        
        return self.N

    def __add__(self, other: 'Protein_Docking_Dataset'):
        """
        # Introduction
        Implements the addition operator for the `Protein_Docking_Dataset` class, allowing two datasets to be combined into a new dataset.
        # Args:
        - other (Protein_Docking_Dataset): Another instance of `Protein_Docking_Dataset` to be added.
        # Returns:
        - Protein_Docking_Dataset: A new dataset instance containing the combined data from both datasets, using the current instance's batch size.
        # Raises:
        - AttributeError: If `other` does not have a `data` attribute.
        - TypeError: If `other` is not an instance of `Protein_Docking_Dataset`.
        """
        
        return Protein_Docking_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly shuffles the indices of the dataset, updating the internal index order.
        # Built-in Attribute:
        - self.N (int): The total number of items in the dataset.
        - self.index (np.ndarray): The array storing the current order of indices.
        # Returns:
        - None
        # Notes:
        This method uses `np.random.permutation` to generate a new random ordering of indices for the dataset.
        """
        
        self.index = np.random.permutation(self.N)
