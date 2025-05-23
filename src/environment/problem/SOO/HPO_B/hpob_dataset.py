import numpy as np
from torch.utils.data import Dataset
import subprocess, sys, os
from .hpo_b import HPOB_Problem
from tqdm import tqdm
import json
import xgboost as xgb


class HPOB_Dataset(Dataset):
    """
    # Introduction
    HPO-B is an autoML hyper-parameter optimization benchmark which includes a wide range of hyperparameter optimization tasks for 16 different model types (e.g., SVM, XGBoost, etc.), resulting in a total of 935 problem instances. The dimension of these problem instances range from 2 to 16. We also note that HPO-B represents problems with ill-conditioned landscape such as huge flattern.
    # Original paper
    "[Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml.](https://arxiv.org/pdf/2106.06257)" arXiv preprint arXiv:2106.06257 (2021).
    # Official Implementation
    [HPO-B](https://github.com/machinelearningnuremberg/HPO-B)
    # License
    None
    """
    
    def __init__(self,
                 data,
                 batch_size = 1):
        """
        # Introduction
        Initializes the dataset object for handling batches of data items, determining the maximum dimension among items, and setting up batch pointers and indices.
        # Args:
        - data (list): A list of data items, each expected to have a `dim` attribute.
        - batch_size (int, optional): The number of items per batch. Defaults to 1.
        # Built-in Attribute:
        - self.data (list): Stores the input data.
        - self.maxdim (int): The maximum dimension found among all data items.Defaults to 0.
        - self.batch_size (int): The batch size for processing data.
        - self.N (int): The total number of data items.
        - self.ptr (list): List of starting indices for each batch.
        - self.index (np.ndarray): Array of indices for the data items.
        # Returns:
        - None
        """
        super().__init__()
        self.data = data
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)

    @staticmethod
    def get_datasets(datapath = None,
                     train_batch_size = 1,
                     test_batch_size = 1,
                     upperbound = None,
                     difficulty = None,
                     user_train_list = None,
                     user_test_list = None,
                     cost_normalize = False, ):
        """
        # Introduction
        Loads and processes the HPO-B benchmark datasets, returning train and test sets as `HPOB_Dataset` objects. Handles dataset downloading, extraction, and filtering based on user-specified lists or difficulty level.
        # Args:
        - datapath (str, optional): Path to the root directory containing the HPO-B data. If `None`, defaults to a subdirectory in the current working directory.
        - train_batch_size (int, optional): Batch size for the training dataset. Defaults to 1.
        - test_batch_size (int, optional): Batch size for the test dataset. Defaults to 1.
        - upperbound (float, optional): Upper bound for the problem domain. Used to set the search space limits.
        - difficulty (str, optional): If set to 'all', merges train and test sets. Otherwise, uses user-specified lists for filtering.
        - user_train_list (list of str, optional): List of problem identifiers to include in the training set.
        - user_test_list (list of str, optional): List of problem identifiers to include in the test set.
        - cost_normalize (bool, optional): Whether to normalize the cost values in the problems. Defaults to False.
        # Returns:
        - tuple: A tuple containing:
            - HPOB_Dataset: The training dataset.
            - HPOB_Dataset: The test dataset.
        # Raises:
        - NotImplementedError: If neither `user_train_list` nor `user_test_list` is provided when required for filtering.
        """
        # get functions ID of indicated suit
        if datapath is None:
            datapath = os.path.join(os.getcwd(), "metabox_data")
        root_dir = datapath + "HPO-B-main/hpob-data/"
        surrogates_dir = datapath + "HPO-B-main/saved-surrogates/"

        # if not os.path.exists(root_dir) or len(os.listdir(root_dir)) < 7 or not os.path.exists(surrogates_dir) or len(os.listdir(surrogates_dir)) < 1909:
        #     try:
        #         from huggingface_hub import snapshot_download
        #     except ImportError:
        #         # check the required package, if not exists, pip install it
        #         try:
        #             subprocess.check_call([sys.executable,'-m', "pip", "install", 'huggingface_hub'])
        #             # print("huggingface_hub has been installed successfully!")
        #             from huggingface_hub import snapshot_download
        #         except subprocess.CalledProcessError as e:
        #             print(f"Install huggingface_hub leads to errors: {e}")

        #     snapshot_download(repo_id='GMC-DRL/MetaBox-HPO-B', repo_type="dataset", local_dir=datapath)
        #     print("Extract data...")
        #     os.system(f'tar -xf {datapath}HPO-B-main.tar.gz -C {datapath}')
        #     os.system(f'rm {datapath}HPO-B-main.tar.gz')
        #     os.system(f'rm {datapath}.gitattributes')

        meta_train_data, meta_vali_data, meta_test_data, bo_initializations, surrogates_stats = get_data(root_dir = root_dir, mode = "v3", surrogates_dir = surrogates_dir)

        if (user_train_list is None and user_test_list is None) or difficulty == 'all':

            def process_data(data, name, n):
                problems = []
                pbar = tqdm(desc = f'Loading {name}', total = n, leave = False)
                for search_space_id in data.keys():
                    for dataset_id in data[search_space_id].keys():
                        bst_model, y_min, y_max = get_bst(surrogates_dir = datapath + 'HPO-B-main/saved-surrogates/', search_space_id = search_space_id, dataset_id = dataset_id,
                                                          surrogates_stats = surrogates_stats)
                        X = np.array(data[search_space_id][dataset_id]["X"])
                        dim = X.shape[1]
                        p = HPOB_Problem(bst_surrogate = bst_model, dim = dim, y_min = y_min, y_max = y_max, lb = -upperbound, ub = upperbound, normalized = cost_normalize, name=str(search_space_id)+'-'+ str(dataset_id))
                        problems.append(p)
                        pbar.update()
                pbar.close()
                return problems

            train_set = process_data(meta_train_data, 'meta_train_data', 758)
            test_set = process_data(meta_vali_data, 'meta_vali_data', 91) + process_data(meta_test_data, 'meta_test_data', 86)
            if difficulty == 'all':
                train_set = test_set = train_set + test_set

        else:
            train_set = []
            test_set = []

            def process_data(data, name, n):
                pbar = tqdm(desc = f'Loading {name}', total = n, leave = False)
                for search_space_id in data.keys():
                    for dataset_id in data[search_space_id].keys():
                        bst_model, y_min, y_max = get_bst(surrogates_dir = datapath + 'HPO-B-main/saved-surrogates/', search_space_id = search_space_id, dataset_id = dataset_id,
                                                          surrogates_stats = surrogates_stats)
                        X = np.array(data[search_space_id][dataset_id]["X"])
                        dim = X.shape[1]
                        p = HPOB_Problem(bst_surrogate = bst_model, dim = dim, y_min = y_min, y_max = y_max,  lb = -upperbound, ub = upperbound, normalized = cost_normalize, name=str(search_space_id)+'-'+ str(dataset_id))
                        if user_train_list is not None and user_test_list is not None:
                            if search_space_id + '-' + dataset_id in user_train_list:
                                train_set.append(p)
                            if search_space_id + '-' + dataset_id in user_test_list:
                                test_set.append(p)
                        elif user_train_list is not None:
                            if search_space_id + '-' + dataset_id in user_train_list:
                                train_set.append(p)
                            else:
                                test_set.append(p)
                        elif user_test_list is not None:
                            if search_space_id + '-' + dataset_id in user_test_list:
                                test_set.append(p)
                            else:
                                train_set.append(p)
                        else:
                            raise NotImplementedError
                        pbar.update()
                pbar.close()

            process_data(meta_train_data, 'meta_train_data', 758)
            process_data(meta_vali_data, 'meta_vali_data', 91)
            process_data(meta_test_data, 'meta_test_data', 86)

        return HPOB_Dataset(train_set, train_batch_size), HPOB_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        """
        # Introduction
        Retrieves a batch of data samples corresponding to the given index.
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
        Returns the number of elements in the dataset.
        # Returns:
            int: The total number of elements in the dataset.
        """
        return self.N

    def __add__(self, other: 'HPOB_Dataset'):
        """
        # Introduction
        Combines two `HPOB_Dataset` instances by concatenating their data attributes.
        # Args:
        - other (HPOB_Dataset): Another dataset instance to be added.
        # Returns:
        - HPOB_Dataset: A new dataset containing the combined data from both instances.
        # Raises:
        - AttributeError: If `other` does not have a `data` attribute.
        """
        return HPOB_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        """
        # Introduction
        Randomly shuffles the indices of the dataset to change the order of data access.
        # Built-in Attribute:
        - self.N (int): The total number of data points in the dataset.
        # Returns:
        - None
        # Side Effects:
        - Updates `self.index` with a new permutation of indices from 0 to `self.N - 1`.
        """
        self.index = np.random.permutation(self.N)


def get_data(mode, surrogates_dir, root_dir):
    """
    # Introduction
    Loads and returns training, validation, and test datasets along with Bayesian Optimization (BO) initializations and surrogate statistics based on the specified mode and directory paths.
    # Args:
    - mode (str): The mode specifying which dataset version or configuration to load. Supported values are "v1", "v2", "v3", "v3-test", and "v3-train-augmented".
    - surrogates_dir (str): Directory path where the surrogate statistics file ("summary-stats.json") is located.
    - root_dir (str): Root directory path containing the dataset files.
    # Returns:
    - train_set: The training dataset, or None if not applicable for the selected mode.
    - vali_set: The validation dataset, or None if not applicable for the selected mode.
    - test_set: The test dataset.
    - bo_initializations: Initializations for Bayesian Optimization.
    - surrogates_stats (dict): Dictionary containing surrogate statistics loaded from "summary-stats.json".
    # Raises:
    - ValueError: If an invalid mode is provided.
    """
    train_set, vali_set, test_set = None, None, None
    if mode == "v3-test":
        train_set, vali_set, test_set, bo_initializations = load_data(root_dir, only_test = True)
    elif mode == "v3-train-augmented":
        train_set, vali_set, test_set, bo_initializations = load_data(root_dir, only_test = False, augmented_train = True)
    elif mode in ["v1", "v2", "v3"]:
        train_set, vali_set, test_set, bo_initializations = load_data(root_dir, version = mode, only_test = False)
    else:
        raise ValueError("Provide a valid mode")

    surrogates_file = surrogates_dir + "summary-stats.json"
    if os.path.isfile(surrogates_file):
        with open(surrogates_file) as f:
            surrogates_stats = json.load(f)

    return train_set, vali_set, test_set, bo_initializations, surrogates_stats


def load_data(rootdir = "", version = "v3", only_test = True, augmented_train = False):
    """
    # Introduction
    Loads HPOB benchmark datasets according to specified parameters, supporting different dataset versions, test/train splits, and augmented training data.
    # Args:
    - rootdir (str, optional): Path to the directory containing the benchmark data files. Defaults to "".
    - version (str, optional): HPOB dataset version to use. Options are "v1", "v2", or "v3". Defaults to "v3".
    - only_test (bool, optional): If True, loads only the test data (valid only for version "v3"). Defaults to True.
    - augmented_train (bool, optional): If True, loads the augmented training data (valid only for version "v3"). Defaults to False.
    # Returns:
    - meta_train_data (dict or None): The meta-training dataset, or None if only_test is True.
    - meta_validation_data (dict or None): The meta-validation dataset, or None if only_test is True.
    - meta_test_data (dict): The meta-testing dataset.
    - bo_initializations (dict): The Bayesian optimization initializations.
    # Raises:
    - FileNotFoundError: If any of the required dataset files are missing in the specified root directory.
    - json.JSONDecodeError: If any of the dataset files are not valid JSON.
    """

    print("Reading data...")
    meta_train_augmented_path = os.path.join(rootdir, "meta-train-dataset-augmented.json")
    meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
    meta_test_path = os.path.join(rootdir, "meta-test-dataset.json")
    meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
    bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

    with open(meta_test_path, "rb") as f:
        meta_test_data = json.load(f)

    with open(bo_initializations_path, "rb") as f:
        bo_initializations = json.load(f)

    meta_train_data = None
    meta_validation_data = None

    if not only_test:
        if augmented_train or version == "v1":
            with open(meta_train_augmented_path, "rb") as f:
                meta_train_data = json.load(f)
        else:
            with open(meta_train_path, "rb") as f:
                meta_train_data = json.load(f)
        with open(meta_validation_path, "rb") as f:
            meta_validation_data = json.load(f)

    if version != "v3":
        temp_data = {}
        for search_space in meta_train_data.keys():
            temp_data[search_space] = {}

            for dataset in meta_train_data[search_space].keys():
                temp_data[search_space][dataset] = meta_train_data[search_space][dataset]

            if search_space in meta_test_data.keys():
                for dataset in meta_test_data[search_space].keys():
                    temp_data[search_space][dataset] = meta_test_data[search_space][dataset]

                for dataset in meta_validation_data[search_space].keys():
                    temp_data[search_space][dataset] = meta_validation_data[search_space][dataset]

        meta_train_data = None
        meta_validation_data = None
        meta_test_data = temp_data

    search_space_dims = {}

    for search_space in meta_test_data.keys():
        dataset = list(meta_test_data[search_space].keys())[0]
        X = meta_test_data[search_space][dataset]["X"][0]
        search_space_dims[search_space] = len(X)

    return meta_train_data, meta_validation_data, meta_test_data, bo_initializations


def get_bst(surrogates_dir, search_space_id, dataset_id, surrogates_stats):
    """
    # Introduction
    Loads a pre-trained XGBoost surrogate model and retrieves its associated normalization statistics for a given search space and dataset.
    # Args:
    - surrogates_dir (str): Directory path where surrogate models are stored.
    - search_space_id (str): Identifier for the search space.
    - dataset_id (str): Identifier for the dataset.
    - surrogates_stats (dict): Dictionary containing normalization statistics for each surrogate model.
    # Returns:
    - bst_surrogate (xgboost.Booster): Loaded XGBoost surrogate model.
    - y_min (float): Minimum target value used for normalization.
    - y_max (float): Maximum target value used for normalization.
    # Raises:
    - AssertionError: If `y_min` is None for the specified surrogate model.
    """
    surrogate_name = 'surrogate-' + search_space_id + '-' + dataset_id
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(surrogates_dir + surrogate_name + '.json')

    y_min = surrogates_stats[surrogate_name]["y_min"]
    y_max = surrogates_stats[surrogate_name]["y_max"]
    assert y_min is not None, 'y_min is None!!'

    return bst_surrogate, y_min, y_max

