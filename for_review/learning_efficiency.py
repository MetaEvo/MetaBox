from ..src import Tester, Config, construct_problem_set, get_baseline
from ..src.baseline.metabbo import *
from ..src.baseline.bbo import *
from ..src.environment.optimizer import *

import pickle
import numpy as np
import matplotlib.pyplot as plt

metabbo_list = ['RNNOPT', 'DEDDQN', 'DEDQN', 'LDE', 'RLPSO', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'RLDAS', 'SYMBOL', 'B2OPT', 'RLDEAFL']
metabbo_path = []  # you trained model path 20250421T122939_bbob-10D_difficult

T0_list = [0.0] * len(metabbo_list)  # default T0
Tg_list = []

# get time
for path, baseline in zip(metabbo_path, metabbo_list):
    path = f"agent_model/train/{baseline}/{path}/checkpoint_log.txt"
    last_time = None
    with open(path, 'r') as f:
        for line in f:
            if "Time:" in line:
                time_str = line.strip().split("Time: ")[1].split("s")[0].strip()  # get seconds
                last_time = float(time_str)
    Tg_list.append(last_time)  # last time

# test checkpoint-0

config = {
    'test_problem': 'bbob-10D',
    'test_difficulty': 'difficult',
    'test_batch_size': 16,
    'baselines': {},
}
for path, baseline in zip(metabbo_path, metabbo_list):
    config['baselines'][baseline] = {
        'agent': baseline,
        'optimizer': eval(f"{baseline}_Optimizer"),
        'model_load_path': f"agent_model/train/{baseline}/{path}/checkpoint-0.pkl", # default checkpoint
    }
config = Config(config)
config, datasets = construct_problem_set(config)
baselines, config = get_baseline(config)
tester = Tester(config, baselines, datasets)
tester.test(log=False)

c0_dir = f"{config.test_log_dir}_{config.test_problem}_{config.test_difficulty}"

# test checkpoint-20

config = {
    'test_problem': 'bbob-10D',
    'test_difficulty': 'difficult',
    'test_batch_size': 16,
    'baselines': {},
}
for path, baseline in zip(metabbo_path, metabbo_list):
    config['baselines'][baseline] = {
        'agent': baseline,
        'optimizer': eval(f"{baseline}_Optimizer"),
        'model_load_path': f"agent_model/train/{baseline}/{path}/checkpoint-20.pkl", # default checkpoint
    }
config = Config(config)
config, datasets = construct_problem_set(config)
baselines, config = get_baseline(config)
tester = Tester(config, baselines, datasets)
tester.test(log=False)
c20_dir = f"{config.test_log_dir}_{config.test_problem}_{config.test_difficulty}"

# cal performance
c0_result = {}
c20_result = {}


with open(c0_dir + '/test_results.pkl', 'rb') as f:
    data = pickle.load(f)
    data = data['cost']

    problem_list = list(data.keys())
    total_performance = np.zeros((len(metabbo_list), len(problem_list)))
    for j, problem in enumerate(problem_list):
        problem_data = data[problem]
        for i, baseline in enumerate(metabbo_list):
            baseline_data = problem_data[baseline]  # List[run]
            run_performance = []
            for run in baseline_data:
                run_performance.append((run[-1] - run[0])/ (0 - run[0] + 1e-20))
            mean_performance = np.mean(run_performance)
            total_performance[i][j] = mean_performance
    for i, baseline in enumerate(metabbo_list):
        c0_result[baseline] = np.mean(total_performance[i])

with open(c20_dir + '/test_results.pkl', 'rb') as f:
    data = pickle.load(f)
    data = data['cost']

    problem_list = list(data.keys())
    total_performance = np.zeros((len(metabbo_list), len(problem_list)))
    for j, problem in enumerate(problem_list):
        problem_data = data[problem]
        for i, baseline in enumerate(metabbo_list):
            baseline_data = problem_data[baseline]  # List[run]
            run_performance = []
            for run in baseline_data:
                run_performance.append((run[-1] - run[0])/ (0 - run[0] + 1e-20))
            mean_performance = np.mean(run_performance)
            total_performance[i][j] = mean_performance
    for i, baseline in enumerate(metabbo_list):
        c20_result[baseline] = np.mean(total_performance[i])

learning_efficiency = []
for i, baseline in enumerate(metabbo_list):
    learning_efficiency.append((c20_result[baseline] - c0_result[baseline]) / (Tg_list[i] - T0_list[i]))
    print(f"{baseline}: {learning_efficiency[i]}")

plt.bar(metabbo_list, learning_efficiency)
plt.ylabel('Learning Efficiency')
plt.title('Learning Efficiency by Baseline')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
