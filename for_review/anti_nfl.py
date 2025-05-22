from ..src import Tester, Config, construct_problem_set, get_baseline
from ..src.baseline.metabbo import *
from ..src.baseline.bbo import *
from ..src.environment.optimizer import *

import pickle
import numpy as np
import matplotlib.pyplot as plt


zero_shot_list = ['DEDDQN', 'DEDQN', 'LDE', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'SYMBOL', 'RLDEAFL']
no_zero_shot_list = ['RNNOPT', 'B2OPT', 'RLPSO', 'RLDAS']
# using test.py testing result
from_dir = "" # test_dir
zero_shot_problem = ['bbob-noisy-30D', 'protein', 'uav', 'hpo-b', 'bbob-30D', 'bbob-noisy-10D']
zero_shot_dir = [] # using zero_shot.py test_dir for bbob-noisy-30D, protein, uav, hpo-b

# additional run bbob-30D and bbob-noisy-10D

zero_shot_path = [] # you trained model path
no_zero_shot_path = [] # you trained model path

# bbob-30D
config = {
    'test_problem': 'bbob-30D',
    'test_difficulty': 'all',
    'test_batch_size': 16,
    'full_meta_data': True,
    'baselines': {},
}
for baseline, path in zip(zero_shot_list, zero_shot_path):
    config['baselines'][baseline] = {
        'agent': baseline,
        'optimizer': eval(f"{baseline}_Optimizer"),
        'model_load_path': path,
    }
config = Config(config)
config, datasets = construct_problem_set(config)
baselines, config = get_baseline(config)
tester = Tester(config, baselines, datasets)
tester.test(log = False)
zero_shot_dir.append(f"{config.test_log_dir}_{config.test_problem}_{config.test_difficulty}") # bbob-30D

# bbob-noisy-10D
config = {
    'test_problem': 'bbob-noisy-10D',
    'test_difficulty': 'all',
    'test_batch_size': 16,
    'full_meta_data': True,
    'baselines': {},
}
for baseline, path in zip(zero_shot_list + no_zero_shot_list, zero_shot_path + no_zero_shot_path):
    config['baselines'][baseline] = {
        'agent': baseline,
        'optimizer': eval(f"{baseline}_Optimizer"),
        'model_load_path': path,
    }
config = Config(config)
config, datasets = construct_problem_set(config)
baselines, config = get_baseline(config)
tester = Tester(config, baselines, datasets)
tester.test(log = False)
zero_shot_dir.append(f"{config.test_log_dir}_{config.test_problem}_{config.test_difficulty}") # bbob-noisy-10D


# cal from problem performance
baseline_list = zero_shot_list + no_zero_shot_list
from_result = {}

with open(from_dir + '/test_results.pkl', 'rb') as f:
    data = pickle.load(f)
    data = data['cost']

    problem_list = list(data.keys())
    total_result = np.zeros((len(baseline_list), len(problem_list)))
    for j, problem in enumerate(problem_list):
        problem_data = data[problem]
        for i, baseline in enumerate(baseline_list):
            baseline_data = problem_data[baseline] # List[run]
            performance = []
            for run in baseline_data:
                performance.append((run[-1] - run[0]) / (0 - run[0] + 1e-20))
            performance = np.mean(performance)
            total_result[i][j] = performance
    for i, baseline in enumerate(baseline_list):
        from_result[baseline] = np.mean(total_result[i])

# zero-shot

# -------- find gbest for each problem----------
gbest = {}
for problem in zero_shot_problem:
    gbest[problem] = 1e32
gbest['bbob-noisy-30D'] = 0.0 # synthetic problem
gbest['bbob-30D'] = 0.0 # synthetic problem
gbest['bbob-noisy-10D'] = 0.0 # synthetic problem
for dir, problem in zip(zero_shot_dir, zero_shot_problem):
    if problem == "bbob-noisy-30D" or problem == "bbob-30D" or problem == "bbob-noisy-10D":
        continue
    with open(dir + '/test_results.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data['cost']

        problem_list = list(data.keys())
        for j, problem in enumerate(problem_list):
            problem_data = data[problem]
            # only find baseline can zero-shot
            for i, baseline in enumerate(zero_shot_list):
                baseline_data = problem_data[baseline]
                for run in baseline_data:
                    gbest[problem] = min(gbest[problem], run[-1])

# --------- cal performance -----------
zero_shot_result = {}

for problem in zero_shot_problem:
    zero_shot_result[problem] = {}
    for baseline in baseline_list:
        zero_shot_result[problem][baseline] = 0.0

for dir, problem in zip(zero_shot_dir, zero_shot_problem):

    zero_shot_baseline_list = zero_shot_list
    if problem == "bbob-noisy-10D":
        zero_shot_baseline_list = zero_shot_list + no_zero_shot_list

    with open(dir + '/test_results.pkl', 'rb') as f:
        data = pickle.load(f)
        data = data['cost']

        problem_list = list(data.keys())
        total_result = np.zeros((len(zero_shot_baseline_list), len(problem_list)))
        for j, problem in enumerate(problem_list):
            problem_data = data[problem]
            for i, baseline in enumerate(zero_shot_baseline_list):
                baseline_data = problem_data[baseline] # List[run]
                performance = []
                for run in baseline_data:
                    performance.append((run[-1] - run[0]) / (gbest[problem] - run[0] + 1e-20))
                performance = np.mean(performance)
                total_result[i][j] = performance
        for i, baseline in enumerate(zero_shot_baseline_list):
            zero_shot_result[problem][baseline] = np.mean(total_result[i])
                        
anti_nfl_result = []
for baseline in baseline_list:
    baseline_anti_nfl = []
    for problem in zero_shot_problem:
        baseline_anti_nfl.append((zero_shot_result[problem][baseline] - from_result[baseline]) / from_result[baseline])
    anti_nfl_result.append(np.exp(np.mean(baseline_anti_nfl)))
    print(f"{baseline}: {anti_nfl_result[-1]}")

# plot
plt.figure(figsize=(10, 5))
plt.bar(baseline_list, anti_nfl_result)
plt.xticks(rotation=45)
plt.ylabel('Anti-NFL')
plt.tight_layout()
plt.show()

