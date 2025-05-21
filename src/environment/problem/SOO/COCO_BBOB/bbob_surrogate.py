from .kan import *
from ....problem.basic_problem import Basic_Problem
from ....problem.SOO.COCO_BBOB.bbob_numpy import *
from os import path
from torch.utils.data import Dataset
import time
import torch.nn as nn
import importlib.util
import importlib.resources as pkg_resources
# MLP
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.ln1 = nn.Linear(input_dim, 32)
        self.ln2 = nn.Linear(32, 64)
        self.ln3 = nn.Linear(64, 32)
        self.ln4 = nn.Linear(32, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.ln3(x)
        x = self.relu3(x)
        x = self.ln4(x)
        return x

class bbob_surrogate_model(Basic_Problem):
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
    - dim (int): Dimensionality of the problem.
    - func_id (int): Identifier for the BBOB function.
    - lb (float or np.ndarray): Lower bound(s) of the input domain.
    - ub (float or np.ndarray): Upper bound(s) of the input domain.
    - shift (np.ndarray): Shift vector for the function.
    - rotate (np.ndarray): Rotation matrix for the function.
    - bias (float): Bias term for the function.
    - config (object): Configuration object containing device information.
    # Attributes:
    - dim (int): Problem dimensionality.
    - func_id (int): BBOB function identifier.
    - instance (object): Instantiated BBOB function.
    - device (str or torch.device): Device for computation (CPU or GPU).
    - optimum (Any): Placeholder for the optimum value (not set in this class).
    - model (KAN or MLP): Loaded surrogate model for the function.
    - ub (float or np.ndarray): Upper bound(s) of the input domain.
    - lb (float or np.ndarray): Lower bound(s) of the input domain.
    # Methods:
    - func(x): Evaluates the surrogate model for a given input `x`, supporting both numpy arrays and torch tensors.
    - eval(x): General evaluation method that adapts to both individual and population inputs, measuring evaluation time.
    - __str__(): Returns a string representation of the surrogate model instance.
    # Raises:
    - ValueError: If the specified dimension is not supported for training.
    """
    def __init__(self, dim, func_id, lb, ub, shift, rotate, bias, config):
        self.dim = dim
        self.func_id = func_id

        self.instance = eval(f'F{func_id}')(dim=dim, shift=shift, rotate=rotate, bias=bias, lb=lb, ub=ub)
        self.device = config.device
        self.optimum = None

        if dim == 2:

            if func_id in [1, 6, 8, 9, 12, 14, 19, 20, 23]:

                model_dir = f'Dim{dim}/KAN/{self.instance}/model'
                try:
                    base_dir = 'metaevobox.environment.problem.SOO.COCO_BBOB.datafile'
                    if importlib.util.find_spec(base_dir) is not None:
                        model_path = pkg_resources.files(base_dir).joinpath(model_dir)
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_datafile_dir = os.path.join(base_path, 'datafile')
                    model_path = os.path.join(local_datafile_dir, model_dir)

                self.model = KAN.loadckpt(str(model_path))
            # elif func_id in [2, 3, 4, 5, 7, 10, 11, 13, 15, 16, 17, 18, 21, 22, 23]:
            else:
                self.model = MLP(dim)

                model_file = f'Dim{dim}/MLP/{self.instance}/model.pth'
                try:
                    base_dir = 'metaevobox.environment.problem.SOO.COCO_BBOB.datafile'
                    if importlib.util.find_spec(base_dir) is not None:
                        model_path = pkg_resources.files(base_dir).joinpath(model_file)
                        model_file_obj = model_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', model_file)
                    model_file_obj = open(local_path, 'rb')

                with model_file_obj as f:
                    if torch.cuda.is_available():
                        self.model.load_state_dict(torch.load(f))
                    else:
                        self.model.load_state_dict(torch.load(f, map_location = 'cpu'))

        elif dim == 5:

            if func_id in [1, 2, 4, 6, 8, 9, 11, 12, 14, 20, 23]:
                model_dir = f'Dim{dim}/KAN/{self.instance}/model'
                try:
                    base_dir = 'metaevobox.environment.problem.SOO.COCO_BBOB.datafile'
                    if importlib.util.find_spec(base_dir) is not None:
                        model_path = pkg_resources.files(base_dir).joinpath(model_dir)
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_datafile_dir = os.path.join(base_path, 'datafile')
                    model_path = os.path.join(local_datafile_dir, model_dir)

                self.model = KAN.loadckpt(str(model_path))
            else:
                self.model = MLP(dim)

                model_file = f'Dim{dim}/MLP/{self.instance}/model.pth'
                try:
                    base_dir = 'metaevobox.environment.problem.SOO.COCO_BBOB.datafile'
                    if importlib.util.find_spec(base_dir) is not None:
                        model_path = pkg_resources.files(base_dir).joinpath(model_file)
                        model_file_obj = model_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', model_file)
                    model_file_obj = open(local_path, 'rb')

                with model_file_obj as f:
                    if torch.cuda.is_available():
                        self.model.load_state_dict(torch.load(f))
                    else:
                        self.model.load_state_dict(torch.load(f, map_location = 'cpu'))


        elif dim == 10:

            if func_id in [1, 2, 4, 6, 9, 12, 14, 23]:
                model_dir = f'Dim{dim}/KAN/{self.instance}/model'
                try:
                    base_dir = 'metaevobox.environment.problem.SOO.COCO_BBOB.datafile'
                    if importlib.util.find_spec(base_dir) is not None:
                        model_path = pkg_resources.files(base_dir).joinpath(model_dir)
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_datafile_dir = os.path.join(base_path, 'datafile')
                    model_path = os.path.join(local_datafile_dir, model_dir)

                self.model = KAN.loadckpt(str(model_path))
            # elif func_id in [2, 5, 8, 9, 11, 16, 17, 18, 19, 20, 21, 22]:
            else:
                self.model = MLP(dim)

                model_file = f'Dim{dim}/MLP/{self.instance}/model.pth'
                try:
                    base_dir = 'metaevobox.environment.problem.SOO.COCO_BBOB.datafile'
                    if importlib.util.find_spec(base_dir) is not None:
                        model_path = pkg_resources.files(base_dir).joinpath(model_file)
                        model_file_obj = model_path.open('rb')
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    local_path = os.path.join(base_path, 'datafile', model_file)
                    model_file_obj = open(local_path, 'rb')

                with model_file_obj as f:
                    if torch.cuda.is_available():
                        self.model.load_state_dict(torch.load(f))
                    else:
                        self.model.load_state_dict(torch.load(f, map_location = 'cpu'))

        else:
            raise ValueError(f'training on dim{dim} is not supported yet.')

        self.model.to(self.device)
        # KAN: 1,3,4,6,7,10,12,13,14,15,23,24  MLP:2,5,8,9,11,16,17,18,19,20,21,22

        self.ub = ub
        self.lb = lb

    def func(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x).to(self.device)
            input_x = (x - self.lb) / (self.ub - self.lb)
            input_x = input_x.to(torch.float64)
            with torch.no_grad():
                y = self.model(input_x)

            return y.flatten().cpu().numpy()

        elif isinstance(x, torch.Tensor):
            input_x = (x - self.lb) / (self.ub - self.lb)
            input_x = input_x.to(torch.float64)
            with torch.no_grad():
                y = self.model(input_x)
            return y

    # return y
    def eval(self, x):
        """
        A general version of func() with adaptation to evaluate both individual and population.
        """
        start=time.perf_counter()

        if x.ndim == 1:  # x is a single individual
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
    def __str__(self):
        return f'Surrogate_{self.instance}'


class bbob_surrogate_Dataset(Dataset):
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
    - data (list): List of surrogate or BBOB function instances.
    - batch_size (int, optional): Number of items per batch. Defaults to 1.
    # Attributes:
    - data (list): The dataset containing function instances.
    - batch_size (int): The batch size for data loading.
    - N (int): Total number of items in the dataset.
    - ptr (list): List of starting indices for each batch.
    - index (np.ndarray): Array of indices for shuffling and sampling.
    - maxdim (int): Maximum dimensionality among all function instances.
    # Methods:
    - get_datasets(...): Static method to generate train and test datasets based on configuration, difficulty, and user-specified splits.
    - __len__(): Returns the number of items in the dataset.
    - __getitem__(item): Returns a batch of data at the specified batch index.
    - __add__(other): Concatenates two datasets.
    - shuffle(): Randomly permutes the dataset indices for shuffling.
    # Raises:
    - ValueError: If configuration or arguments are invalid (e.g., unsupported suit, missing difficulty, or conflicting train/test splits).
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
    def get_datasets(version='torch', suit='bbob-surrogate-10D',
                     train_batch_size=1,
                     test_batch_size=1, difficulty='easy',
                     user_train_list=None, user_test_list=None,
                     seed=3849, shifted=True, biased=True, rotated=True,
                     config=None, upperbound=5):

        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty != 'easy' and difficulty != 'difficult' and difficulty != 'all' and difficulty is not None:
            raise ValueError(f'{difficulty} difficulty is invalid.')
        if difficulty in ['easy', 'difficult', 'all'] and user_test_list is not None and user_train_list is not None:
            raise ValueError('If you have specified the training/test set, the difficulty should be None.')
        if suit == 'bbob-surrogate-10D':
            dim = config.dim = 10
        elif suit == 'bbob-surrogate-5D':
            dim = config.dim = 5
        elif suit == 'bbob-surrogate-2D':
            dim = config.dim = 2
        else:
            raise ValueError(f'{suit} is not supported yet.')

        if difficulty == 'easy':
            if dim == 2:
                train_id = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15,
                            20, 22]
            elif dim == 5 or dim == 10:
                train_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            20]
        # test_id = [16, 17, 18, 19, 21, 22, 23, 24]
        elif difficulty == 'difficult':
            if dim == 2 or dim == 5:
                train_id = [1, 2, 5, 6, 10, 11, 13, 14]
            elif dim == 10:
                train_id = [1, 2, 5, 6, 10, 11, 13, 20]
        elif difficulty == None:
            train_id = user_train_list
            test_id = user_test_list
        elif difficulty == 'all':
            test_id = train_id = [i for i in range(1, 25)]

        np.random.seed(seed)
        train_set = []
        test_set = []
        ub = upperbound
        lb = -upperbound

        func_id = [i for i in range(1, 25)]
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
            surrogate_instance = bbob_surrogate_model(dim, id, ub=ub, lb=lb, shift=shift, rotate=H, bias=bias, config=config)
            bbob_instance = eval(f'F{id}')(dim=dim, shift=shift, rotate=H, bias=bias, lb=lb, ub=ub)

            if difficulty == 'all':
                train_set.append(surrogate_instance)
                test_set.append(bbob_instance)
                continue
            if user_train_list is None and user_test_list is None and difficulty is not None:
                if id in train_id:
                    train_set.append(surrogate_instance)
                else:
                    test_set.append(bbob_instance)
            else:
                if user_train_list is not None and user_test_list is not None:
                    if id in train_id:
                        train_set.append(surrogate_instance)
                    if id in test_id:
                        test_set.append(bbob_instance)
                elif user_train_list is not None:
                    if id in train_id:
                        train_set.append(surrogate_instance)
                    else:
                        test_set.append(bbob_instance)
                elif user_test_list is not None:
                    if id in test_id:
                        test_set.append(bbob_instance)
                    else:
                        train_set.append(surrogate_instance)

        return bbob_surrogate_Dataset(train_set, train_batch_size), bbob_surrogate_Dataset(test_set, test_batch_size)

    def __len__(self):
        return self.N

    def __getitem__(self, item):

        
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __add__(self, other: 'bbob_surrogate_Dataset'):
        return bbob_surrogate_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = torch.randperm(self.N)
