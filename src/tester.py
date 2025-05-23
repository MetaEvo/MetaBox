import copy
import importlib

from .environment.problem.utils import construct_problem_set
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time
from tqdm import tqdm
import os, psutil
from .environment.basic_environment import PBO_Env
from .logger import *
from .environment.parallelenv.parallelenv import ParallelEnv
import json
import torch
import gym, os
from typing import Optional, Union, Literal, List
from .environment.optimizer.basic_optimizer import Basic_Optimizer
from .rl import Basic_Agent
from .environment.problem.basic_problem import Basic_Problem
from dill import dumps, loads
import importlib.resources as pkg_resources
import pprint
import importlib.util
from .environment.optimizer import (
    DEDDQN_Optimizer,
    DEDQN_Optimizer,
    RLHPSDE_Optimizer,
    LDE_Optimizer,
    QLPSO_Optimizer,
    RLEPSO_Optimizer,
    RLPSO_Optimizer,
    RNNOPT_Optimizer,
    GLEET_Optimizer,
    RLDAS_Optimizer,
    LES_Optimizer,
    NRLPSO_Optimizer,
    SYMBOL_Optimizer,
    RLDEAFL_Optimizer,
    SurrRLDE_Optimizer,
    RLEMMO_Optimizer,

    GLHF_Optimizer,
    B2OPT_Optimizer,
    PSORLNS_Optimizer,
    L2T_Optimizer,
    MADAC_Optimizer,
    LGA_Optimizer,
    OPRO_Optimizer
)

from .baseline.bbo import (
    DE,
    JDE21,
    MADDE,
    NLSHADELBC,
    PSO,
    GLPSO,
    SDMSPSO,
    SAHLPSO,
    CMAES,
    Random_search,
    SHADE,
    MOEAD,
    MFEA
)

from .baseline.metabbo import (
    GLEET,
    DEDDQN,
    DEDQN,
    QLPSO,
    NRLPSO,
    RLHPSDE,
    RLDEAFL,
    LDE,
    RLPSO,
    SYMBOL,
    RLDAS,
    SurrRLDE,
    RLEMMO,
    GLHF,
    B2OPT,
    LGA,
    PSORLNS,
    LES,
    L2T,
    MADAC,
    RNNOPT,
    OPRO
)

def cal_t0(dim, fes):
    """
    # Introduction
    Estimates the average time (in milliseconds) required to perform a set of basic NumPy vectorized operations.
    '''T0 will be used to calculate the complexity of algorithms.'''
    # Args:
    - dim (int): The dimensionality of the random NumPy arrays to generate.
    - fes (int): The number of function evaluations (iterations of operations) to perform in each timing loop.
    # Returns:
    - float: The average elapsed time in milliseconds over 10 runs for performing the specified operations.
    # Notes:
    - The function performs addition, division, multiplication, square root, logarithm, and exponential operations on randomly generated NumPy arrays.
    - The timing is measured using `time.perf_counter()` for higher precision.
    """
    
    T0 = 0
    for i in range(10):
        start = time.perf_counter()
        for _ in range(fes):
            x = np.random.rand(dim)
            x + x
            x / (x+2)
            x * x
            np.sqrt(x)
            np.log(x)
            np.exp(x)
        end = time.perf_counter()
        T0 += (end - start) * 1000
    # ms
    return T0/10


def cal_t1(problem, dim, fes):
    """
    # Introduction
    Measures the average time (in milliseconds) required to evaluate a problem's objective function over a batch of randomly generated solutions.
    T1 will be used to calculate the complexity of the algorithm.
    # Args:
    - problem: a problem object 
    - dim (int): The dimensionality of each solution vector.
    - fes (int): The number of function evaluations
    # Returns:
    - float: The average elapsed time (in milliseconds) to evaluate the batch, computed over 10 runs.
    # Notes:
    - The function generates random solutions using `np.random.rand`.
    - Timing is performed using `time.perf_counter`.
    """
    T1 = 0
    for i in range(10):
        x = np.random.rand(fes, dim)
        start = time.perf_counter()
        # for i in range(fes):
        #     problem.eval(x[i])
        problem.eval(x)
        end = time.perf_counter()
        T1 += (end - start) * 1000
    # ms
    return T1/10


def record_data(data, test_set, agent_for_rollout, checkpoints, results, meta_results, config):
    """
    # Introduction
    Processes a list of data items, updating results and meta_results dictionaries with information extracted from each item. Handles both standard result keys and metadata, organizing results by problem and agent.
    # Args:
    todo:这里写完了，有个问题，这个metadata具体的结构写在哪比较好
    - data(dict): Metadata, a dict contain the rollout test result,similar to test result but has more details.
    - test_set (object): The problem dataset for the test process.
    - agent_for_rollout (str): The base name or identifier for the agent used during rollout.
    - checkpoints (list): List of checkpoint identifiers for agents.
    - results (dict): A dictionary to store or update results initialized only with the config information.
    - meta_results (dict): An empty dictionary to store or update metadata results.
    - config (object): Configuration object with attributes such as `full_meta_data` to control metadata processing.
    # Returns:
    - tuple: A tuple containing the updated `results` and `meta_results` dictionaries.
    """
    
    for item in data:
        for key in item.keys():
            if key == 'metadata' and config.full_meta_data:
                metadata = item[key]
                metadata['T'] = item['T2'] # ms
                meta_results[item['problem_name']][item['agent_name']].append(metadata)
                continue
            if key not in ['agent_name', 'problem_name']:
                if key not in results.keys():
                    results[key] = {}
                    for problem in test_set.data:
                        results[key][problem.__str__()] = {}
                        for agent_id in checkpoints:
                            results[key][problem.__str__()][agent_for_rollout+f'-{agent_id}'] = []  # 51 np.arrays
                results[key][item['problem_name']][item['agent_name']].append(item[key])
    return results, meta_results


def store_meta_data(log_dir, meta_data_results):
    """
    # Introduction
    Stores and updates meta data results for different process names into pickle files within a specified log directory. Ensures that meta data is accumulated and persisted across multiple calls, and clears in-memory storage after saving.
    # Args:
    - log_dir (str): The directory path where the metadata should be stored.
    - meta_data_results (dict): A dictionary where keys are process names and values are dictionaries mapping agent names to lists of meta data.
    # Returns:
    - dict: The updated `meta_data_results` dictionary with in-memory lists cleared after saving.
    # Raises:
    - OSError: If the function fails to create the required directories or write to files.
    - pickle.PickleError: If there is an error during pickling or unpickling the data.
    """
    
    if not os.path.exists(log_dir+'/metadata/'):
        os.makedirs(log_dir+'/metadata/')

    for pname in meta_data_results.keys():
        problem_data = meta_data_results[pname]
        for baseline in problem_data.keys():
            if problem_data[baseline]:
                # not empty
                if not os.path.exists(log_dir + f"/metadata/{baseline}/"):
                    os.makedirs(log_dir + f"/metadata/{baseline}/")
                if not os.path.exists(log_dir + f"/metadata/{baseline}/{pname}.pkl"):
                    with open(log_dir + f"/metadata/{baseline}/{pname}.pkl", 'wb') as f:
                        pickle.dump(problem_data[baseline], f, -1)
                else:
                    with open(log_dir + f"/metadata/{baseline}/{pname}.pkl", 'rb') as f:
                        data_results = pickle.load(f)
                    data_results += problem_data[baseline] # list + list
                    with open(log_dir + f"/metadata/{baseline}/{pname}.pkl", 'wb') as f:
                        pickle.dump(data_results, f, -1)
                meta_data_results[pname][baseline] = []

    # for pname in meta_data_results.keys():
    #     if not os.path.exists(log_dir+f'/metadata/{pname}.pkl'):
    #         with open(log_dir + f'/metadata/{pname}.pkl', 'wb') as f:
    #             pickle.dump(meta_data_results[pname], f, -1)
    #         for agent in meta_data_results[pname].keys():  # clear memory storage
    #             meta_data_results[pname][agent] = []
    #     else:
    #         with open(log_dir + f'/metadata/{pname}.pkl', 'rb') as f:
    #             data_results = pickle.load(f)
    #         for key in meta_data_results[pname].keys():
    #             if key in data_results.keys():
    #                 data_results[key] += meta_data_results[pname][key]  # list + list
    #             else:
    #                 data_results[key] = meta_data_results[pname][key]
    #             meta_data_results[pname][key] = []  # clear memory storage
    #         with open(log_dir + f'/metadata/{pname}.pkl', 'wb') as f:
    #             pickle.dump(data_results, f, -1)
    return meta_data_results
                    
                    
class BBO_TestUnit():
    """
    Introduction:
    BBO_TestUnit is a test unit designed for running batch episodes of black-box optimization (BBO) algorithms in parallel using RAY. It encapsulates a problem instance and an optimizer, and ensures reproducibility by managing random seeds and PyTorch settings.


    - optimizer (Basic_Optimizer): The optimizer instance to be tested.
    - problem (Basic_Problem): The problem instance on which the optimizer will be evaluated.
    - seed (int): The random seed for reproducibility.

    # Methods:

    - run_batch_episode(): Runs a single batch episode of the optimizer on the problem, returning a dictionary of results and timing information.

    # Attributes:

    - optimizer (Basic_Optimizer): The optimizer used in the test unit.
    - problem (Basic_Problem): The problem instance for evaluation.
    - seed (int): The random seed for reproducibility.
    """
    """
        A test unit for RAY parallel with a problem and a basic optimizer.
    """

    def __init__(self,
                 optimizer: Basic_Optimizer,
                 problem: Basic_Problem,
                 seed: int,
                 ):
        self.optimizer = optimizer
        self.problem = problem
        self.seed = seed

    def run_batch_episode(self):
        """
        # Introduction

        Runs a single batch episode for the optimizer on the given problem, ensuring reproducibility by setting random seeds and configuring PyTorch settings.

        # Args:

        None

        # Returns:

        - dict: A dictionary containing the results of the optimizer's episode, including timing information, agent and problem names, and additional metrics.

        # Raises:

        None
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.set_default_dtype(torch.float64)
        self.optimizer.seed(self.seed)
        self.problem.reset()
        start_time = time.perf_counter()
        res = self.optimizer.run_episode(self.problem)
        end_time = time.perf_counter()
        res['T1'] = self.problem.T1
        res['T2'] = (end_time - start_time) * 1000
        res['agent_name'] = self.optimizer.test_name
        res['problem_name'] = self.problem.__str__()
        return res


class MetaBBO_TestUnit():
    """
    # Introduction
    MetaBBO_TestUnit is a test unit designed for parallel execution using RAY, encapsulating an agent, an environment, and a random seed for reproducibility. It facilitates the evaluation of agent performance on a given environment, with optional checkpointing.


    - agent (Basic_Agent): The agent to be evaluated.
    - env (PBO_Env): The environment in which the agent operates.
    - seed (int): The random seed for reproducibility.
    - checkpoint (int, optional): An optional checkpoint identifier for the agent. Defaults to None.

    # Methods:

    - run_batch_episode(required_info: dict = {}): Runs a single batch episode with the specified agent and environment, ensuring reproducibility by setting random seeds and configuring PyTorch settings. Returns a dictionary containing episode results, timing, agent and problem names, and any additional rollout results.
    """
    """
    
        A test unit for RAY parallel with an agent, an env and a seed.
    """

    
    def __init__(self,
                 agent: Basic_Agent,
                 env: PBO_Env,
                 seed: int,
                 ):
        self.agent = agent
        self.env = env
        self.seed = seed

    def run_batch_episode(self, required_info = {}):
        """
        # Introduction

        Runs a single batch episode using the agent and environment, ensuring reproducibility by setting random seeds and configuring PyTorch settings.

        # Args:
        todo:需不需要example
        - required_info (dict, optional): Additional information required for the episode rollout. Defaults to an empty dictionary.

        # Returns:

        - dict: A dictionary containing the results of the episode, including timing information, agent and problem names, and any additional rollout results.
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.set_default_dtype(torch.float64)

        start_time = time.perf_counter()
        res = self.agent.rollout_episode(self.env, self.seed, required_info)
        end_time = time.perf_counter()
        res['T1'] = self.env.problem.T1
        res['T2'] = (end_time - start_time) * 1000
        agent_name = self.env.optimizer.test_name
        res['agent_name'] = agent_name
        res['problem_name'] = self.env.problem.__str__()
        return res


def get_baseline(config):
    agents_for_cp=[]
    agents_optimizers_for_cp=[]
    traditional_optimizers_for_cp=[]
    baselines = config.baselines
    assert baselines is not None
    for bsl in baselines.keys():
        if 'agent' in baselines[bsl].keys():  # metabbo
            agents_optimizers_for_cp.append(baselines[bsl]['optimizer'](config))
            if 'model_load_path' in baselines[bsl].keys() and baselines[bsl]['model_load_path'] is not None:
                with open(os.path.join(os.getcwd(), baselines[bsl]['model_load_path']), 'rb') as f:
                    agents_for_cp.append(pickle.load(f, fix_imports=False))
            else:
                try:
                    base_dir = f'metaevobox.model.{config.test_problem}.{config.test_difficulty}'
                    if importlib.util.find_spec(base_dir) is not None:
                        model_path = pkg_resources.files(base_dir).joinpath(f"{baselines[bsl]['agent']}.pkl")
                        with model_path.open('rb') as f:
                            agents_for_cp.append(pickle.load(f))
                    else:
                        raise ModuleNotFoundError
                except ModuleNotFoundError:
                    base_path = os.path.dirname(os.path.abspath(__file__))
                    model_path = os.path.join(base_path, f"model/{config.test_problem}/{config.test_difficulty}", f"{baselines[bsl]['agent']}.pkl")
                    with open(model_path, 'rb') as f:
                        agents_for_cp.append(pickle.load(f))
        else:  # bbo
            traditional_optimizers_for_cp.append(baselines[bsl]['optimizer'](config))
    config.baselines = None
    return (agents_for_cp, agents_optimizers_for_cp, traditional_optimizers_for_cp), config


class Tester(object):
    def __init__(self, config, baselines, user_datasets = None):
        # self.key_list = config.agent
        self.test_log_dir = config.test_log_dir
        self.rollout_log_dir = config.rollout_log_dir
        self.mgd_test_log_dir = config.mgd_test_log_dir
        self.mte_test_log_dir = config.mte_test_log_dir
        self.config = config

        _, self.test_set = user_datasets

        # if user_datasets is None:
        #     self.train_set, self.test_set = construct_problem_set(self.config)
        # else:
        #     self.train_set, self.test_set = user_datasets(config)
        # self.config.dim = max(self.train_set.maxdim, self.test_set.maxdim)

        # initialize the dataframe for logging
        self.test_results = {'cost': {},
                             'fes': {},
                             'T1': {},
                             'T2': {},
                             }
        self.meta_data_results = {}
        # prepare experimental optimizers and agents
        self.agent_for_cp = []
        self.agent_name_list = []
        self.l_optimizer_for_cp = []
        self.t_optimizer_for_cp = []

        # 先append 用户的
        agents_for_cp, agents_optimizers_for_cp, traditional_optimizers_for_cp = baselines
        agents_for_cp = agents_for_cp if isinstance(agents_for_cp, list) else [agents_for_cp]
        agents_optimizers_for_cp = agents_optimizers_for_cp if isinstance(agents_optimizers_for_cp, list) else [agents_optimizers_for_cp]
        traditional_optimizers_for_cp = traditional_optimizers_for_cp if isinstance(traditional_optimizers_for_cp, list) else [traditional_optimizers_for_cp]

        name_count = dict()
        for id, agent in enumerate(agents_for_cp):
            name = agent.__str__()
            self.agent_for_cp.append(copy.deepcopy(agent))
            self.agent_name_list.append(name)
            if name not in name_count:
                name_count[name] = [id]
            else:
                name_count[name].append(id)
        metabbo = []
        for id, opt in enumerate(agents_optimizers_for_cp):
            name = self.agent_name_list[id]
            if len(name_count[name]) > 1:
                for i in range(1, len(name_count[name])):
                    updated_name = f"{i}_" + name
                    self.agent_name_list[name_count[name][i]] = updated_name
            setattr(opt, "test_name", self.agent_name_list[id])
            metabbo.append(self.agent_name_list[id])
            self.l_optimizer_for_cp.append(copy.deepcopy(opt))

        name_count = dict()
        for id, opt in enumerate(traditional_optimizers_for_cp):
            name = opt.__str__()
            if name not in name_count:
                name_count[name] = 0
            else:
                name_count[name] += 1
        bbo = []
        for id in reversed(range(len(traditional_optimizers_for_cp))):
            opt = traditional_optimizers_for_cp[id]
            name = opt.__str__()
            count = name_count[name]
            if count:
                name_count[name] -= 1
                name = f"{count}_" + name
            setattr(opt, "test_name", name)
            self.t_optimizer_for_cp.insert(0, copy.deepcopy(opt))
            bbo.insert(0, name)

        if 'mmo' in self.config.test_problem:
            pass
        elif 'mto' in self.config.test_problem or 'wcci2020' in self.config.test_problem:
            pass
        elif 'moo' in self.config.test_problem:
            pass
        else:
            if "CMAES" not in name_count:
                cmaes = CMAES(self.config)
                setattr(cmaes, "test_name", "CMAES")
                self.t_optimizer_for_cp.append(cmaes)
                bbo.append("CMAES")

            if "Random_search" not in name_count:
                rs = Random_search(self.config)
                setattr(rs, "test_name", "Random_search")
                self.t_optimizer_for_cp.append(rs)
                bbo.append("Random_search")
        # logging
        if len(self.agent_for_cp) == 0:
            print('None of learnable agent')
        else:
            print(f'there are {len(self.agent_for_cp)} agent')
            for a, l_optimizer in zip(self.agent_name_list, self.l_optimizer_for_cp):
                print(f'learnable_agent:{a},l_optimizer:{l_optimizer.test_name} optimizer')

        if len(self.t_optimizer_for_cp) == 0:
            print('None of traditional optimizer')
        else:
            print(f'there are {len(self.t_optimizer_for_cp)} traditional optimizer')
            for t_optimizer in self.t_optimizer_for_cp:
                print(f't_optimizer:{t_optimizer.test_name}')

        self.config.baselines = {'metabbo': metabbo, 'bbo': bbo}

        for key in self.test_results.keys():
            self.initialize_record(key)
        self.test_results['config'] = copy.deepcopy(self.config)
        self.test_results['T0'] = np.mean([cal_t0(p.dim, config.maxFEs) for p in self.test_set.data])
        if config.full_meta_data:
            for problem in self.test_set.data:
                self.meta_data_results[problem.__str__()] = {}
                for agent_name in self.agent_name_list:
                    self.meta_data_results[problem.__str__()][agent_name] = []  # test_run x fes
                for optimizer in self.t_optimizer_for_cp:
                    self.meta_data_results[problem.__str__()][optimizer.test_name] = []

        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def initialize_record(self, key):
        """
        # Introduction

        Initializes a record in the `test_results` dictionary for a given key, setting up nested dictionaries for each problem, agent, and optimizer.

        # Args:

        - key (str): The identifier for the test record to initialize.

        # Side Effects:

        - Modifies the `self.test_results` attribute by adding a new entry for `key` if it does not already exist. For each problem in `self.test_set.data`, creates sub-entries for each agent in `self.agent_name_list` and each optimizer in `self.t_optimizer_for_cp`, initializing them as empty lists.
        """
        if key not in self.test_results.keys():
            self.test_results[key] = {}
        for problem in self.test_set.data:
            self.test_results[key][problem.__str__()] = {}
            for agent_name in self.agent_name_list:
                self.test_results[key][problem.__str__()][agent_name] = []  # 51 np.arrays
            for optimizer in self.t_optimizer_for_cp:
                self.test_results[key][problem.__str__()][optimizer.test_name] = []  # 51 np.arrays
        
    def record_test_data(self, data: list):
        """
        # Introduction

        Records test data from a list of dictionaries, organizing results by problem and agent names.
        Handles both metadata and other test result keys, updating internal result structures accordingly.

        # Args:

        - data (list): Metadata, a dict contain the rollout test result,similar to test result but has more details.
        # Side Effects:

        - Updates `self.meta_data_results` with metadata if `self.config.full_meta_data` is True.
        - Updates `self.test_results` with other test result metrics, initializing records as needed.

        # Notes:

        - Assumes that `self.meta_data_results`, `self.test_results`, and `self.config.full_meta_data` are properly initialized.
        - Ignores keys 'agent_name' and 'problem_name' when recording test results.
        """

        for item in data:
            for key in item.keys():
                if key == 'metadata' and self.config.full_meta_data:
                    self.meta_data_results[item['problem_name']][item['agent_name']].append(item[key])
                    continue
                if key not in ['agent_name', 'problem_name']:
                    if key not in self.test_results.keys():
                        self.initialize_record(key)
                    self.test_results[key][item['problem_name']][item['agent_name']].append(item[key])            

    def test(self, log = True):
        """
        # Introduction
        Runs tests on agents and optimizers using different parallelization strategies and records the results.
        # Args:
        None
        # Side Effects:
        - Records test data and stores meta data results after each test run.
        - Saves the final test results to a pickle file in the log directory.
        # Raises:
        - NotImplementedError: If an unsupported parallelization mode is specified in the configuration.
        """

        print(f'start testing: {self.config.run_time}_{self.config.test_problem}_{self.config.test_difficulty}')
        print("following config:")
        pprint.pprint(vars(self.config))
        test_log_dir = self.test_log_dir

        if not os.path.exists(test_log_dir):
            os.makedirs(test_log_dir)

        if not os.path.exists(test_log_dir + '/metadata/'):
            os.makedirs(test_log_dir + '/metadata/')
        with open(test_log_dir + f'/metadata/config.pkl', 'wb') as f:
            pickle.dump(self.config, f, -1)

        test_parallel_mode = self.config.test_parallel_mode  # 'Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch'
        test_run = self.config.test_run
        seed_list = list(range(1, test_run + 1)) # test_run
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()

        test_start_time = time.perf_counter()
        if test_parallel_mode == 'Full':
            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp)
                                                                                                                               for p in self.test_set.data
                                                                                                                               for seed in seed_list]
            testunit_list += [BBO_TestUnit(copy.deepcopy(optimizer), copy.deepcopy(p), seed) for optimizer in self.t_optimizer_for_cp
                                                                                        for p in self.test_set.data
                                                                                        for seed in seed_list]
            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            self.record_test_data(meta_test_data)
            self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
                
        elif test_parallel_mode == 'Baseline_Problem':
            pbar = tqdm(total=len(seed_list), desc="Baseline_Problem Testing")
            for seed in seed_list:
                testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp)
                                                                                                                                   for p in self.test_set.data
                                                                                                                                    ]
                testunit_list += [BBO_TestUnit(copy.deepcopy(optimizer), copy.deepcopy(p), seed) for optimizer in self.t_optimizer_for_cp
                                                                                                for p in self.test_set.data
                                                                                                ]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                self.record_test_data(meta_test_data)
                self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
                pbar.update()
            pbar.close()
                
        elif test_parallel_mode == 'Problem_Testrun':
            pbar = tqdm(total=len(self.agent_for_cp) + len(self.t_optimizer_for_cp), desc="Problem_Testrun Testing")
            for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp):
                pbar.set_description(f"Problem_Testrun Testing {agent.__str__()}")
                testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for p in self.test_set.data for seed in seed_list]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                self.record_test_data(meta_test_data)
                self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
                pbar.update()
            for optimizer in self.t_optimizer_for_cp:
                pbar.set_description(f"Problem_Testrun Testing {optimizer.__str__()}")
                testunit_list = [BBO_TestUnit(copy.deepcopy(optimizer), copy.deepcopy(p), seed) for p in self.test_set.data
                                                                                                 for seed in seed_list]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                self.record_test_data(meta_test_data)
                self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
                pbar.update()
            pbar.close()
                
        elif test_parallel_mode == 'Batch':
            pbar_len = (len(self.agent_for_cp) + len(self.t_optimizer_for_cp)) * np.ceil(self.test_set.N / self.config.test_batch_size) * self.config.test_run
            pbar = tqdm(total=pbar_len, desc="Batch Testing")
            for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp):
                for ip, problem in enumerate(self.test_set):
                    for i, seed in enumerate(seed_list):
                        pbar.set_description_str(f"Batch Testing Agent {agent.__str__()} with Problem Batch {ip}, Run {i}")
                        testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for p in problem]
                        MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                        meta_test_data = MetaBBO_test.rollout()
                        self.record_test_data(meta_test_data)
                        pbar.update()
                    self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
            for optimizer in self.t_optimizer_for_cp:
                for ip, problem in enumerate(self.test_set):
                    for i, seed in enumerate(seed_list):
                        pbar.set_description_str(f"Batch Testing Optimizer {optimizer.__str__()} with Problem Batch {ip}, Run {i}")
                        testunit_list = [BBO_TestUnit(copy.deepcopy(optimizer), copy.deepcopy(p), seed) for p in problem]
                        MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                        meta_test_data = MetaBBO_test.rollout()
                        self.record_test_data(meta_test_data)
                        pbar.update()
                    self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
            pbar.close()

        elif test_parallel_mode == "Serial":
            pbar_len = (len(self.agent_for_cp) + len(self.t_optimizer_for_cp)) * self.test_set.N * self.config.test_run
            pbar = tqdm(total = pbar_len, desc = "Serial Testing")
            for (agent, optimizer) in zip(self.agent_for_cp, self.l_optimizer_for_cp):
                for ip, problem in enumerate(self.test_set.data):
                    for i, seed in enumerate(seed_list):
                        pbar.set_description(f"Batch Testing Agent {agent.__str__()} with Problem Batch {ip}, Run {i}")
                        env = PBO_Env(copy.deepcopy(problem), copy.deepcopy(optimizer))

                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        np.random.seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False

                        torch.set_default_dtype(torch.float64)
                        tmp_agent = copy.deepcopy(agent)

                        start_time = time.perf_counter()
                        res = tmp_agent.rollout_episode(env, seed, {})
                        end_time = time.perf_counter()
                        res['T1'] = env.problem.T1
                        res['T2'] = (end_time - start_time) * 1000
                        agent_name = optimizer.test_name
                        res['agent_name'] = agent_name
                        res['problem_name'] = problem.__str__()
                        meta_test_data = [res]
                        self.record_test_data(meta_test_data)
                        pbar.update()
                    self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
            for optimizer in self.t_optimizer_for_cp:
                for ip, problem in enumerate(self.test_set.data):
                    for i, seed in enumerate(seed_list):
                        pbar.set_description_str(f"Batch Testing Optimizer {optimizer.__str__()} with Problem Batch {ip}, Run {i}")

                        torch.manual_seed(seed)
                        torch.cuda.manual_seed(seed)
                        np.random.seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False

                        torch.set_default_dtype(torch.float64)

                        tmp_optimizer = copy.deepcopy(optimizer)
                        tmp_problem = copy.deepcopy(problem)
                        tmp_optimizer.seed(seed)
                        tmp_problem.reset()

                        start_time = time.perf_counter()
                        res = tmp_optimizer.run_episode(tmp_problem)
                        end_time = time.perf_counter()

                        res['T1'] = tmp_problem.T1
                        res['T2'] = (end_time - start_time) * 1000
                        res['agent_name'] = tmp_optimizer.test_name
                        res['problem_name'] = tmp_problem.__str__()

                        meta_test_data = [res]
                        self.record_test_data(meta_test_data)
                        pbar.update()
                    self.meta_data_results = store_meta_data(test_log_dir, self.meta_data_results)
        else:
            raise NotImplementedError

        test_end_time = time.perf_counter()

        with open(test_log_dir + '/test_time_log.txt', 'a') as f:
            f.write(f"Test time: {test_end_time - test_start_time} seconds\n")

        with open(test_log_dir + '/test_results.pkl', 'wb') as f:
            pickle.dump(self.test_results, f, -1)

        if log:
            if 'mmo' in self.config.test_problem:
                logger = MMO_Logger(self.config)
            elif 'mto' in self.config.test_problem or 'wcci2020' in self.config.test_problem:
                logger = MTO_Logger(self.config)
            elif 'moo' in self.config.test_problem:
                logger = MOO_Logger(self.config)
            else:
                logger = Basic_Logger(self.config)
            logger.post_processing_test_statics(test_log_dir + '/') # todo

    def test_for_random_search(self):
        """
        # Introduction
        Executes a comprehensive test suite for the Random Search optimizer across a set of benchmark problems, logging performance metrics and timing information for analysis.
        # Args:
        None (uses self.config for configuration).
        # Returns:
        todo: 这里就是test_result,同样的问题，它的具体结构是否写在这？感觉应该写在这，毕竟是在这里构建的,要放图在这里吗？
        - dict: A dictionary `test_results` containing the metrics list:
            - 'cost': Nested dict mapping problem names to optimizer names to lists of cost arrays (one per run).
            - 'fes': Nested dict mapping problem names to optimizer names to lists of function evaluation counts (one per run).
            - 'T0': Baseline timing value computed from problem dimension and max function evaluations.
            - 'T1': Dict mapping optimizer names to average problem-specific timing metric.
            - 'T2': Dict mapping optimizer names to average wall-clock time per run (in milliseconds).
        # Notes:
        - Runs 51 independent trials per problem.
        - Uses tqdm for progress visualization.
        - Seeds numpy's RNG for reproducibility.
        - Pads cost arrays to length 51 if necessary.
        """
        
        config = self.config
        # get entire problem set
        train_set, test_set = construct_problem_set(config)
        entire_set = train_set + test_set
        # get optimizer
        optimizer = Random_search(copy.deepcopy(config))
        # initialize the dataframe for logging
        test_results = {'cost': {},
                        'fes': {},
                        'T0': 0.,
                        'T1': {},
                        'T2': {}}
        test_results['T1'][type(optimizer).__name__] = 0.
        test_results['T2'][type(optimizer).__name__] = 0.
        for problem in entire_set:
            test_results['cost'][problem.__str__()] = {}
            test_results['fes'][problem.__str__()] = {}
            test_results['cost'][problem.__str__()][type(optimizer).__name__] = []  # 51 np.arrays
            test_results['fes'][problem.__str__()][type(optimizer).__name__] = []  # 51 scalars
        # calculate T0
        test_results['T0'] = cal_t0(config.dim, config.maxFEs)
        # begin testing
        seed = list(range(1, self.config.test_run + 1))
        pbar_len = len(entire_set) * self.config.test_run
        with tqdm(range(pbar_len), desc='test for random search') as pbar:
            for i, problem in enumerate(entire_set):
                T1 = 0
                T2 = 0
                for run in range(self.config.test_run):
                    np.random.seed(seed[run])

                    tmp_optimizer = copy.deepcopy(optimizer)
                    tmp_problem = copy.deepcopy(problem)

                    tmp_optimizer.seed(seed[run])
                    tmp_problem.reset()

                    start = time.perf_counter()
                    info = tmp_optimizer.run_episode(tmp_problem)
                    end = time.perf_counter()

                    cost = info['cost']
                    while len(cost) < 51:
                        cost.append(cost[-1])
                    fes = info['fes']
                    if i == 0:
                        T1 += problem.T1
                        T2 += (end - start) * 1000  # ms
                    test_results['cost'][problem.__str__()][type(optimizer).__name__].append(cost)
                    test_results['fes'][problem.__str__()][type(optimizer).__name__].append(fes)
                    pbar_info = {'problem': problem.__str__(),
                                'optimizer': type(optimizer).__name__,
                                'run': run,
                                'cost': cost[-1],
                                'fes': fes, }
                    pbar.set_postfix(pbar_info)
                    pbar.update(1)
                if i == 0:
                    test_results['T1'][type(optimizer).__name__] = T1 / self.config.test_run
                    test_results['T2'][type(optimizer).__name__] = T2 / self.config.test_run
        return test_results


    @ staticmethod
    def name_translate(problem):
        """
        # Introduction
        Translates a given problem identifier into a human-readable problem name.
        # Args:
        - problem (str): The identifier of the problem to be translated. Expected values include 'bbob', 'bbob-torch', 'bbob-noisy', 'bbob-noisy-torch', 'protein', or 'protein-torch'.
        # Returns:
        - str: The human-readable name corresponding to the given problem identifier.
        # Raises:
        - ValueError: If the provided problem identifier is not recognized.
        """

        return problem[:-6] if problem.endswith('-torch') else problem

    @ staticmethod
    def mgd_test(config, agent: str, from_problem:str, from_difficulty:str, from_test_path: str, to_test_path: str):
        with open(from_test_path, 'rb') as f:
            from_data = pickle.load(f)
        with open(to_test_path, 'rb') as f:
            to_data = pickle.load(f)

        mgd_test_log_dir = f"{config.mgd_test_log_dir}/{agent}"
        #
        data = {}
        for metric in from_data.keys():
            if metric not in data.keys():
                data[metric] = {}
            metric_value = from_data[metric]
            for problem in metric_value.keys():
                if problem not in data[metric].keys():
                    data[metric][problem] = {}
                from_agent_data = from_data[metric][problem]
                to_agent_data = to_data[metric][problem]
                if agent not in from_agent_data.keys() or agent not in to_agent_data.keys():
                    assert "Agent '{agent}' not found in from_data or to_data for metric '{metric}', problem '{problem}'."
                data[metric][problem][f"{agent}_from"] = from_agent_data[agent]
                data[metric][problem][f"{agent}_to"] = to_agent_data[agent]
                data[metric][problem]['Random_Search'] = from_agent_data['Random_Search']
        logger = Basic_Logger(config) # Only SOO
        aei, aei_std = logger.aei_metric(data, config.maxFEs)
        print(f"AEI: {aei}")
        print(f"AEI_std: {aei_std}")
        print(f'MGD({Tester.name_translate(from_problem)}_{from_difficulty}, {Tester.name_translate(config.test_problem)}_{config.test_difficulty}) of {agent}: '
              f'{100 * (1 - aei[agent + "_from"] / aei[agent + "_to"])}%')

    # def mgd_test(self, user_from, user_to, user_opt, user_datasets):
    #     """
    #     todo:重写注释
    #     # Introduction
    #     Executes the Meta Generalization Domain (MGD) test for evaluating agent performance across different problem domains and configurations. This method loads pre-trained agents, sets up test environments, and runs parallelized test episodes to collect and log performance metrics.
    #     # Args:
    #     None (uses instance attributes and configuration).
    #     # Side Effects:
    #     - Loads agent models and configuration from files.
    #     - Runs parallelized test episodes using various batching strategies.
    #     - Logs test results and meta-data to disk.
    #     - Updates instance attributes `self.test_results` and `self.meta_data_results`.
    #     # Raises:
    #     - FileNotFoundError: If required model or configuration files are missing.
    #     - KeyError: If specified agent or optimizer keys are not found in the configuration.
    #     - Exception: Propagates exceptions from environment setup, agent loading, or parallel execution.
    #     # Notes:
    #     - Supports multiple parallelization strategies: 'Full', 'Baseline_Problem', 'Problem_Testrun', and 'Batch'.
    #     - Stores results as pickled files in the specified log directory.
    #     - Designed for use in meta-learning and black-box optimization benchmarking.
    #     """
    #
    #     config = self.config
    #     print(f'start MGD_test: {config.run_time}_{config.test_problem}_{config.test_difficulty}')
    #     # get test set
    #     num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
    #     mgd_test_log_dir = f"{self.mgd_test_log_dir}_{self.config.test_problem}_{self.config.test_difficulty}"
    #
    #     if not os.path.exists(mgd_test_log_dir):
    #         os.makedirs(mgd_test_log_dir)
    #     with open(mgd_test_log_dir + '/config.pkl', 'wb') as f:
    #         pickle.dump(config, f, -1)
    #
    #     _, test_set = user_datasets
    #     self.test_set = test_set
    #
    #     agent_name = user_from.__str__()
    #     agent_from = user_from
    #     agent_to = user_to
    #     l_optimizer = copy.deepcopy(user_opt)
    #
    #     # initialize the dataframe for logging
    #     self.test_results = {'cost': {},
    #                          'fes': {},
    #                          'T1': {},
    #                          'T2': {},
    #                          }
    #     self.meta_data_results = {}
    #     agent_name_list = [f'{agent_name}_from', f'{agent_name}_to']
    #     l_optimizer_cp = []
    #     for agent_name in agent_name_list:
    #         opt = copy.deepcopy(l_optimizer)
    #         setattr(opt, 'test_name', agent_name)
    #         l_optimizer_cp.append(opt)
    #
    #     self.agent_name_list = agent_name_list
    #
    #
    #     for key in self.test_results.keys():
    #         self.initialize_record(key)
    #
    #     if config.full_meta_data:
    #         for problem in self.test_set.data:
    #             self.meta_data_results[problem.__str__()] = {}
    #             for agent_name in self.agent_name_list:
    #                 self.meta_data_results[problem.__str__()][agent_name] = []  # test_run x fes
    #
    #     # calculate T0
    #     self.test_results['T0'] = np.mean([cal_t0(p.dim, config.maxFEs) for p in self.test_set.data])
    #     # begin mgd_test
    #
    #     test_run = config.test_run
    #     test_parallel_mode = config.test_parallel_mode
    #     seed_list = list(range(1, test_run + 1))
    #
    #     if test_parallel_mode == 'Full':
    #         testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[0])), seed) for p in test_set.data for seed in seed_list]
    #         testunit_list += [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[1])), seed) for p in test_set.data for seed in seed_list]
    #
    #         MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
    #         meta_test_data = MetaBBO_test.rollout()
    #         self.record_test_data(meta_test_data)
    #         self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #
    #     elif test_parallel_mode == 'Baseline_Problem':
    #         pbar = tqdm(total = len(seed_list), desc = "Baseline_Problem Testing")
    #         for seed in seed_list:
    #             testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[0])), seed) for p in test_set.data]
    #             testunit_list += [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[1])), seed) for p in test_set.data]
    #             MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
    #             meta_test_data = MetaBBO_test.rollout()
    #             self.record_test_data(meta_test_data)
    #             self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #             pbar.update()
    #         pbar.close()
    #
    #     elif test_parallel_mode == 'Problem_Testrun':
    #         pbar_len = 2
    #         pbar = tqdm(total = pbar_len, desc = "Problem_Testrun Testing")
    #         pbar.set_description(f"Problem_Testrun Testing from {agent_from.__str__()} to {agent_to.__str__()}")
    #         testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[0])), seed)
    #                          for p in self.test_set.data
    #                          for seed in seed_list]
    #         MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
    #         meta_test_data = MetaBBO_test.rollout()
    #         self.record_test_data(meta_test_data)
    #         self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #         pbar.update()
    #
    #         testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[1])), seed)
    #                          for p in self.test_set.data
    #                          for seed in seed_list]
    #         MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
    #         meta_test_data = MetaBBO_test.rollout()
    #         self.record_test_data(meta_test_data)
    #         self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #         pbar.update()
    #         pbar.close()
    #
    #     elif test_parallel_mode == 'Batch':
    #         pbar_len = 2 * np.ceil(test_set.N / config.test_batch_size) * test_run
    #         pbar = tqdm(total = pbar_len, desc = "Batch Testing")
    #         for ip, problem in enumerate(test_set):
    #             for i, seed in enumerate(seed_list):
    #                 pbar.set_description_str(f"Batch Testing From Agent {agent_from.__str__()} with Problem Batch {ip}, Run {i}")
    #                 testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_from), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[0])), seed) for p in problem]
    #                 MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
    #                 meta_test_data = MetaBBO_test.rollout()
    #                 self.record_test_data(meta_test_data)
    #                 pbar.update()
    #             self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #         for ip, problem in enumerate(test_set):
    #             for i, seed in enumerate(seed_list):
    #                 pbar.set_description_str(f"Batch Testing To Agent {agent_to.__str__()} with Problem Batch {ip}, Run {i}")
    #                 testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent_to), PBO_Env(copy.deepcopy(p), copy.deepcopy(l_optimizer_cp[1])), seed) for p in problem]
    #                 MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
    #                 meta_test_data = MetaBBO_test.rollout()
    #                 self.record_test_data(meta_test_data)
    #                 pbar.update()
    #             self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #         pbar.close()
    #
    #     elif test_parallel_mode == 'Serial':
    #         pbar_len = 2 * test_set.N * self.config.test_run
    #         pbar = tqdm(total = pbar_len, desc = "Serial Testing")
    #         for ip, problem in enumerate(test_set.data):
    #             for i, seed in enumerate(seed_list):
    #                 pbar.set_description_str(f"Batch Testing From Agent {agent_from.__str__()} with Problem Batch {ip}, Run {i}")
    #                 env = PBO_Env(copy.deepcopy(problem), copy.deepcopy(l_optimizer_cp[0]))
    #
    #                 torch.manual_seed(seed)
    #                 torch.cuda.manual_seed(seed)
    #                 np.random.seed(seed)
    #                 torch.backends.cudnn.deterministic = True
    #                 torch.backends.cudnn.benchmark = False
    #
    #                 torch.set_default_dtype(torch.float64)
    #                 tmp_agent = copy.deepcopy(agent_from)
    #
    #                 start_time = time.perf_counter()
    #                 res = tmp_agent.rollout_episode(env, seed, {})
    #                 end_time = time.perf_counter()
    #                 res['T1'] = env.problem.T1
    #                 res['T2'] = (end_time - start_time) * 1000
    #                 agent_name = l_optimizer_cp[0].test_name
    #                 res['agent_name'] = agent_name
    #                 res['problem_name'] = problem.__str__()
    #                 meta_test_data = [res]
    #                 self.record_test_data(meta_test_data)
    #                 pbar.update()
    #             self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #         for ip, problem in enumerate(test_set.data):
    #             for i, seed in enumerate(seed_list):
    #                 pbar.set_description_str(f"Batch Testing TO Agent {agent_to.__str__()} with Problem Batch {ip}, Run {i}")
    #                 env = PBO_Env(copy.deepcopy(problem), copy.deepcopy(l_optimizer_cp[1]))
    #
    #                 torch.manual_seed(seed)
    #                 torch.cuda.manual_seed(seed)
    #                 np.random.seed(seed)
    #                 torch.backends.cudnn.deterministic = True
    #                 torch.backends.cudnn.benchmark = False
    #
    #                 torch.set_default_dtype(torch.float64)
    #                 tmp_agent = copy.deepcopy(agent_to)
    #
    #                 start_time = time.perf_counter()
    #                 res = tmp_agent.rollout_episode(env, seed, {})
    #                 end_time = time.perf_counter()
    #                 res['T1'] = env.problem.T1
    #                 res['T2'] = (end_time - start_time) * 1000
    #                 agent_name = l_optimizer_cp[1].test_name
    #                 res['agent_name'] = agent_name
    #                 res['problem_name'] = problem.__str__()
    #                 meta_test_data = [res]
    #                 self.record_test_data(meta_test_data)
    #                 pbar.update()
    #             self.meta_data_results = store_meta_data(mgd_test_log_dir, self.meta_data_results)
    #
    #     with open(mgd_test_log_dir + '/mgd_test_results.pkl', 'wb') as f:
    #         pickle.dump(self.test_results, f, -1)

    @ staticmethod
    def mte_test(config, agent: str, pre_train_problem: str, pre_train_difficulty: str, pre_train_data_path: str, scratch_data_path: str, pdf_fig: bool = True):
        """
        # Introduction
        Evaluates and visualizes the Model Transfer Efficiency (MTE) between a pre-trained agent and a scratch agent on a transfer learning task. The method loads experiment results, processes performance data, computes MTE, and generates a comparative plot of average returns over learning steps.
        # Args:
        None. Uses configuration from `self.config`.
        # Returns:
        None. Prints the computed MTE value and saves a plot comparing pre-trained and scratch agent performance.
        # Raises:
        - FileNotFoundError: If the required JSON or pickle files are not found.
        - KeyError: If expected keys are missing in the loaded data.
        - Exception: For errors during data processing or plotting.
        """
        with open(pre_train_data_path, 'rb') as f:
            pre_train_data = pickle.load(f)
        with open(scratch_data_path, 'rb') as f:
            scratch_data = pickle.load(f)

        pre_train_steps = pre_train_data['steps']

        fig_type = 'pdf' if pdf_fig else 'png'

        mte_test_log_dir = f"{config.mte_test_log_dir}/{agent}"
        transferred_problem = config.test_problem
        transferred_difficulty = config.test_difficulty
        print(f'start MTE_test: {config.run_time}, agent {agent} from {pre_train_difficulty}_{pre_train_problem} to {transferred_difficulty}_{transferred_problem}')

        # with open('model.json', 'r', encoding = 'utf-8') as f:
        #     json_data = json.load(f)
        # pre_train = json_data[config.pre_train_rollout]
        # scratch_rollout = json_data[config.scratch_rollout]
        #
        # pre_train_file = pre_train['dir']
        # scratch_file = scratch_rollout['dir']

        # preprocess data for agent
        def preprocess(data):
            # aggregate all problem's data together
            returns = data['return']
            results = []
            for problem in returns.keys():
                results.append([])
                for agt in returns[problem].keys():
                    results[-1].append(np.array(returns[problem][agt]))  
            results = np.array(results).transpose(1, 0, 2)  # num_problems x n_ckeckpoints x rollout_run -> n_ckeckpoints x num_problems x rollout_run
            return results.mean((1, 2)), results.std(-1).mean(-1)  # mean(n_ckeckpoints x rollout_run) ; std(rollout_run).mean(num_problems)

        pre_train_mean, pre_train_std = preprocess(pre_train_data)
        scratch_mean, scratch_std = preprocess(scratch_data)
        pre_train_mean = savgol_filter(pre_train_mean, 13, 5)
        scratch_mean = savgol_filter(scratch_mean, 13, 5)
        plt.figure(figsize=(40, 15))        
        x = np.arange(min(len(pre_train_mean), len(scratch_mean)))
        idx = len(x)
        smooth = 1
        smooth_pre_train_mean = np.zeros(len(pre_train_mean))
        a = smooth_pre_train_mean[0] = pre_train_mean[0]
        norm = smooth + 1
        for i in range(1, len(pre_train_mean)):
            a = a * smooth + pre_train_mean[i]
            smooth_pre_train_mean[i] = a / norm if norm > 0 else a
            norm *= smooth
            norm += 1

        smooth_scratch_mean = np.zeros(len(scratch_mean))
        a = smooth_scratch_mean[0] = scratch_mean[0]
        norm = smooth + 1
        for i in range(1, len(scratch_mean)):
            a = a * smooth + scratch_mean[i]
            smooth_scratch_mean[i] = a / norm if norm > 0 else a
            norm *= smooth
            norm += 1
            
        plt.plot(x[:idx], smooth_pre_train_mean[:idx], label='pre-train', marker='*', markersize=30, markevery=1, c='blue', linewidth=5)
        plt.fill_between(x[:idx], smooth_pre_train_mean[:idx] - pre_train_std[:idx], smooth_pre_train_mean[:idx] + pre_train_std[:idx], alpha=0.2, facecolor='blue')
        plt.plot(x[:idx], smooth_scratch_mean[:idx], label='scratch', marker='*', markersize=30, markevery=1, c='red', linewidth=5)
        plt.fill_between(x[:idx], smooth_scratch_mean[:idx] - scratch_std[:idx], smooth_scratch_mean[:idx] + scratch_std[:idx], alpha=0.2, facecolor='red')
        # Search MTE
        scratch = smooth_scratch_mean[:idx]
        pretrain = smooth_pre_train_mean[:idx]
        topx = np.argmax(scratch)
        topy = scratch[topx]
        T = topx / idx
        t = 0
        if pretrain[0] < topy:
            for i in range(1, idx):
                if pretrain[i - 1] < topy <= pretrain[i]:
                    t = ((topy - pretrain[i - 1]) / (pretrain[i] - pretrain[i - 1]) + i - 1) / idx
                    break
        if np.max(pretrain[-1]) < topy:
            t = 1
        MTE = 1 - t / T

        print(f'MTE({Tester.name_translate(pre_train_problem)}_{pre_train_difficulty}, {Tester.name_translate(transferred_problem)}_{transferred_difficulty}) of {agent}: '
            f'{MTE}')

        ax = plt.gca()
        ax.xaxis.get_offset_text().set_fontsize(45)
        plt.xticks(fontsize=45, ticks = np.arange(idx)[::2], labels = pre_train_steps[::2])
        plt.yticks(fontsize=45)
        plt.legend(loc=0, fontsize=60)
        plt.xlabel('Learning Steps', fontsize=55)
        plt.ylabel('Avg Return', fontsize=55)
        plt.title(f'Fine-tuning ({Tester.name_translate(pre_train_problem)} $\\rightarrow$ {Tester.name_translate(transferred_problem)})',
                fontsize=60)
        plt.tight_layout()
        plt.grid()
        plt.subplots_adjust(wspace=0.2)

        if not os.path.exists(mte_test_log_dir):
            os.makedirs(mte_test_log_dir)
        plt.savefig(f'{mte_test_log_dir}/MTE_{pre_train_difficulty}_{pre_train_problem}_to_{transferred_difficulty}_{transferred_problem}.{fig_type}', bbox_inches='tight')

    @ staticmethod
    def rollout_batch(config, rollout_dir, rollout_opt, rollout_datasets, checkpoints = None, log = True):
        """
        todo:重写注释
        # Introduction
        Executes a batch rollout of agents on a test set of problems using various parallelization strategies. The function loads agent checkpoints, sets up environments, and evaluates agent performance across multiple seeds and problems, storing the results for further analysis.
        # Args:
        - config (object): Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
        # Returns:
        - None: The function saves the rollout results and metadata to disk but does not return any value.
        # Raises:
        - KeyError: If the specified agent key is missing in the `model.json` file.
        - NotImplementedError: If the specified parallelization mode in `config.test_parallel_mode` is not supported.
        """
        print(f'start rollout: {config.run_time}_{config.test_problem}_{config.test_difficulty}')

        rollout_log_dir = f"{config.rollout_log_dir}_{config.test_problem}_{config.test_difficulty}"

        num_gpus = 0 if config.device == 'cpu' else 1
        train_set, test_set = rollout_datasets

        test_parallel_mode = config.test_parallel_mode

        agents = []
        optimizer_for_rollout = []

        l_optimizer = copy.deepcopy(rollout_opt)
        upper_dir = rollout_dir
        if not os.path.isdir(upper_dir):  # path to .pkl files
            upper_dir = os.path.join(*tuple(str.split(upper_dir, '/')[:-1]))

        if checkpoints is None:
            epoch_list = [f for f in os.listdir(upper_dir) if f.endswith('.pkl')]
            checkpoints = np.arange(len(epoch_list))
        n_checkpoint = len(checkpoints)

        # get agent
        # learning_step
        steps = []
        agent_name = None
        for agent_id in checkpoints:
            with open(os.path.join(upper_dir, f'checkpoint-{agent_id}.pkl'), 'rb') as f:
                agent = pickle.load(f)
                agent_name = agent.__str__()
                steps.append(agent.get_step())
                agents.append(agent)
                opt = copy.deepcopy(l_optimizer)
                setattr(opt, 'test_name', agent_name + f'-{agent_id}')
                optimizer_for_rollout.append(opt)

        agent_for_rollout = agent_name
        rollout_results = {'cost': {},
                           'return': {},
                           }
        meta_data_results = {}
        for key in rollout_results.keys():
            if key not in rollout_results.keys():
                rollout_results[key] = {}
            for problem in test_set.data:
                rollout_results[key][problem.__str__()] = {}
                meta_data_results[problem.__str__()] = {}
                for agent_id in checkpoints:
                    rollout_results[key][problem.__str__()][agent_name + f'-{agent_id}'] = []  # 51 np.arrays
                    meta_data_results[problem.__str__()][agent_name + f'-{agent_id}'] = []

        rollout_results['config'] = copy.deepcopy(config)

        pbar_len = int(np.ceil(test_set.N * n_checkpoint / test_set.batch_size))
        seed_list = list(range(1, config.rollout_run + 1))

        if test_parallel_mode == 'Full':
            testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout)
                                                                                                                                for p in test_set.data
                                                                                                                                for seed in seed_list]
            MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
            meta_test_data = MetaBBO_test.rollout()
            rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
            meta_data_results = store_meta_data(rollout_log_dir, meta_data_results)
        elif test_parallel_mode == 'Baseline_Problem':
            pbar = tqdm(total=len(seed_list), desc="Baseline_Problem Rollouting")
            for seed in seed_list:
                testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout)
                                                                                                                                for p in test_set.data
                                                                                                                                ]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
                meta_data_results = store_meta_data(rollout_log_dir, meta_data_results)
                pbar.update()
            pbar.close()

        elif test_parallel_mode == 'Problem_Testrun':
            pbar = tqdm(total=len(agents), desc="Problem_Testrun Rollouting")
            for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout):
                pbar.set_description(f"Problem_Testrun Rollouting Checkpoint {ckp}")
                testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed)
                                                                                                                                for p in test_set.data
                                                                                                                                for seed in seed_list]
                MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                meta_test_data = MetaBBO_test.rollout()
                rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
                meta_data_results = store_meta_data(rollout_log_dir, meta_data_results)
                pbar.update()
            pbar.close()

        elif test_parallel_mode == 'Batch':
            pbar_len = len(agents)  * np.ceil(test_set.N / config.test_batch_size) * config.rollout_run
            pbar = tqdm(total=pbar_len, desc="Batch Rollouting")
            for (ckp, agent, optimizer) in zip(checkpoints, agents, optimizer_for_rollout):
                for ip, problem in enumerate(test_set):
                    for i, seed in enumerate(seed_list):
                        pbar.set_description_str(f"Batch Rollouting Checkpoint {ckp} with Problem Batch {ip}, Run {i}")
                        testunit_list = [MetaBBO_TestUnit(copy.deepcopy(agent), PBO_Env(copy.deepcopy(p), copy.deepcopy(optimizer)), seed) for p in problem]
                        MetaBBO_test = ParallelEnv(testunit_list, para_mode = 'ray', num_gpus=num_gpus)
                        meta_test_data = MetaBBO_test.rollout()
                        rollout_results, meta_data_results = record_data(meta_test_data, test_set, agent_for_rollout, checkpoints, rollout_results, meta_data_results, config)
                        pbar.update()
                meta_data_results = store_meta_data(rollout_log_dir, meta_data_results)
            pbar.close()
        else:
            raise NotImplementedError

        rollout_results['steps'] = steps
        rollout_results['agent_for_rollout'] = agent_name

        if not os.path.exists(rollout_log_dir):
            os.makedirs(rollout_log_dir)
        with open(rollout_log_dir + '/rollout.pkl', 'wb') as f:
            pickle.dump(rollout_results, f, -1)
        with open(rollout_log_dir + '/config.pkl', 'wb') as f:
            pickle.dump(config, f, -1)

        if log:
            if 'mmo' in config.test_problem:
                logger = MMO_Logger(config)
            elif 'mto' in config.test_problem or 'wcci2020' in config.test_problem:
                logger = MTO_Logger(config)
            elif 'moo' in config.test_problem:
                logger = MOO_Logger(config)
            else:
                logger = Basic_Logger(config)
            logger.post_processing_rollout_statics(rollout_log_dir + '/')



