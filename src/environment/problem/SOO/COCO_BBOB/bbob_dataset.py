from .bbob_numpy import *
from .bbob_torch import *
from torch.utils.data import Dataset

class BBOB_Dataset(Dataset):
    """
    # Introduction
    BBOB-Surrogate investigates the integration of surrogate modeling techniques into MetaBBO , enabling data-driven approximation of expensive objective functions while maintaining optimization fidelity.
    # Original paper
    "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
    # Official Implementation
    [BBOB-Surrogate](https://github.com/GMC-DRL/Surr-RLDE)
    # License
    None
    # Problem Suite Composition
    BBOB-Surrogate contains a total of 72 optimization problems, corresponding to three dimensions (2, 5, 10), each dimension contains 24 problems. Each problem consists of a trained KAN or MLP network, which is used to fit 24 black box functions in the COCO-BBOB benchmark. The network here is a surrogate model of the original function.
    # Args:
    - data (list): A list of BBOB problem instances.
    - batch_size (int, optional): Number of instances per batch. Defaults to 1.
    # Attributes:
    - data (list): The list of BBOB problem instances.
    - batch_size (int): The batch size for sampling.
    - N (int): Total number of problem instances.
    - ptr (list): List of starting indices for each batch.
    - index (np.ndarray): Permuted indices for shuffling.
    - maxdim (int): Maximum dimensionality among all problem instances.
    # Methods:
    - get_datasets(...): Static method to generate train and test datasets based on suit, difficulty, and other configuration options.
    - __getitem__(item): Returns a batch of problem instances at the specified batch index.
    - __len__(): Returns the total number of problem instances.
    - __add__(other): Concatenates two BBOB_Dataset objects.
    - shuffle(): Randomly permutes the order of the dataset.
    # Raises:
    - ValueError: If required arguments are missing or invalid (e.g., unsupported suit, invalid difficulty, or upperbound too low).
    batch = train_set[0]
    train_set.shuffle()
    ```
    """
    
    def __init__(self,
                 data,
                 batch_size=1):
        """
        # Introduction
        Initializes the dataset object with provided data and batch size, and computes relevant attributes for batching and dimensionality.
        # Args:
        - data (list): The dataset to be managed, where each item is expected to have a `dim` attribute.
        - batch_size (int, optional): The number of samples per batch. Defaults to 1.
        # Built-in Attribute:
        - self.data (list): Stores the input dataset.
        - self.batch_size (int): Stores the batch size.
        - self.N (int): The total number of data items.
        - self.ptr (list): List of starting indices for each batch.
        - self.index (np.ndarray): Array of indices for the dataset.
        - self.maxdim (int): The maximum dimension found among all items in the dataset.
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
    def get_datasets(suit,
                     upperbound,
                     shifted=True,
                     rotated=True,
                     biased=True,
                     train_batch_size=1,
                     test_batch_size=1,
                     difficulty=None,
                     version='numpy',
                     instance_seed=3849,
                     user_train_list=None,
                     user_test_list=None,
                     device = None):
        """
        # Introduction
        Generates training and testing datasets for BBOB (Black-Box Optimization Benchmarking) or BBOB-noisy function suites, with configurable properties such as shifting, rotation, bias, and difficulty level.
        # Args:
        - suit (str): The function suite and dimension, e.g., 'bbob10' or 'bbob-noisy20'.
        - upperbound (float): The upper bound for the function domain (must be at least 5).
        - shifted (bool, optional): Whether to apply a random shift to the function. Defaults to True.
        - rotated (bool, optional): Whether to apply a random rotation to the function. Defaults to True.
        - biased (bool, optional): Whether to add a random bias to the function. Defaults to True.
        - train_batch_size (int, optional): Batch size for the training dataset. Defaults to 1.
        - test_batch_size (int, optional): Batch size for the testing dataset. Defaults to 1.
        - difficulty (str, optional): Difficulty level of the functions to include ('easy', 'difficult', 'all', or None). Defaults to None.
        - version (str, optional): Version of the function implementation ('numpy' or other). Defaults to 'numpy'.
        - instance_seed (int, optional): Random seed for reproducibility. Defaults to 3849.
        - user_train_list (list, optional): List of function IDs to include in the training set. Defaults to None.
        - user_test_list (list, optional): List of function IDs to include in the testing set. Defaults to None.
        - device (torch.device, optional): Device for torch tensors if using the torch version. Defaults to None.
        # Returns:
        - Tuple[BBOB_Dataset, BBOB_Dataset]: A tuple containing the training and testing datasets.
        # Raises:
        - ValueError: If neither `difficulty` nor both `user_train_list` and `user_test_list` are provided.
        - ValueError: If the function suite is invalid or not supported.
        - ValueError: If the difficulty level is invalid.
        - AssertionError: If `upperbound` is less than 5.
        """
        # get functions ID of indicated suit
        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')

        dim = int(suit[-3:-1])
        suit = suit[:4]
        if suit == 'bbob':
            func_id = [i for i in range(1, 25)]     # [1, 24]
            small_set_func_id = [1, 2, 3, 5, 15, 16, 17, 21]
        elif suit == 'bbob-noisy':
            func_id = [i for i in range(101, 131)]  # [101, 130]
            small_set_func_id = [101, 105, 115, 116, 117, 119, 120, 125]
        else:
            raise ValueError(f'{suit} function suit is invalid or is not supported yet.')
        if difficulty != 'easy' and difficulty != 'difficult' and difficulty != 'all' and difficulty is not None:
            raise ValueError(f'{difficulty} difficulty is invalid.')
        # get problem instances
        if instance_seed > 0:
            np.random.seed(instance_seed)
            torch.manual_seed(instance_seed)

        train_set = []
        test_set = []
        assert upperbound >= 5., f'Argument upperbound must be at least 5, but got {upperbound}.'
        ub = upperbound
        lb = -upperbound

        instance_list = []

        for id in func_id:
            if shifted:
                shift = 0.8 * (np.random.random(dim) * (ub - lb) + lb)
            else:
                shift = np.zeros(dim)
            if rotated:
                H = rotate_gen(dim)
            else:
                H = np.eye(dim)
            if biased:
                bias = np.random.randint(1, 26) * 100
            else:
                bias = 0
            if version == 'numpy':
                instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)
            else:
                shift = torch.Tensor(shift)
                H = torch.Tensor(H)
                bias = torch.Tensor([bias])
                instance = eval(f'F{id}_torch')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub, device = device)

            if difficulty == "all":
                instance_list.append(instance)
                continue # all 优先级最高

            if user_train_list is None and user_test_list is None and difficulty is not None:
                if (difficulty == 'easy' and id not in small_set_func_id) or (difficulty == 'difficult' and id in small_set_func_id):
                    train_set.append(instance)
                else:
                    # 如果difficulty 是 all test_set这里是全部
                    test_set.append(instance)
            else:
                if user_train_list is not None and user_test_list is not None:
                    if (id in user_train_list):
                        train_set.append(instance)
                    if (id in user_test_list):
                        test_set.append(instance)
                elif user_train_list is not None:
                    # 如果只选择了train，不在train的都去test，不然用户自己选
                    if (id in user_train_list):
                        train_set.append(instance)
                    else:
                        test_set.append(instance)
                elif user_test_list is not None:
                    if (id in user_test_list):
                        test_set.append(instance)
                    else:
                        train_set.append(instance)

        if difficulty == 'all':
            train_set = instance_list.copy()
            test_set = instance_list.copy()

        return BBOB_Dataset(train_set, train_batch_size), BBOB_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        """
        # Introduction
        Retrieves a batch of data items corresponding to the given index or indices.
        # Args:
        - item (int or slice): The index or indices specifying which batch to retrieve.
        # Returns:
        - list: A list containing the data items from the dataset for the specified batch.
        # Raises:
        - IndexError: If the provided index is out of range.
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
        - int: The total number of elements contained in the dataset.
        """
        
        return self.N

    def __add__(self, other: 'BBOB_Dataset'):
        """
        # Introduction
        Implements the addition operator for the `BBOB_Dataset` class, allowing two datasets to be combined.
        # Args:
        - other (BBOB_Dataset): Another instance of `BBOB_Dataset` to be added to the current dataset.
        # Returns:
        - BBOB_Dataset: A new `BBOB_Dataset` instance containing the combined data from both datasets, with the same batch size as the original.
        # Raises:
        - AttributeError: If `other` does not have a `data` attribute.
        - TypeError: If `other` is not an instance of `BBOB_Dataset`.
        """
        
        return BBOB_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly shuffles the indices of the dataset, updating the internal index order.
        # Built-in Attribute:
        - self.N (int): The total number of elements in the dataset.
        - self.index (np.ndarray): The array storing the current order of indices.
        # Returns:
        - None
        # Notes:
        This method modifies the `self.index` attribute in-place using a random permutation.
        """
        
        self.index = np.random.permutation(self.N)
