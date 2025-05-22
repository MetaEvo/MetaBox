from ..src import Tester, Config, construct_problem_set, get_baseline
from ..src.baseline.metabbo import *
from ..src.baseline.bbo import *
from ..src.environment.optimizer import *

import pickle
import numpy as np
import pandas as pd
from openpyxl import Workbook
# test bbob-10D difficult
metabbo_list = ['RNNOPT', 'DEDDQN', 'DEDQN', 'LDE', 'RLPSO', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'RLDAS', 'SYMBOL', 'B2OPT', 'RLDEAFL']
metabbo_path = [] # you trained model path 20250421T122939_bbob-10D_difficult
bbo_list = ['PSO', 'DE', 'SHADE', 'JDE21', 'MADDE']

config = {
    'test_problem': 'bbob-10D',
    'test_difficulty': 'difficult',
    'test_batch_size': 16,
    'full_meta_data': True,
    'baselines': {},
}

for baseline, path in zip(metabbo_list, metabbo_path):
    config['baselines'][baseline] = {
        'agent': baseline,
        'optimizer': eval(f"{baseline}_Optimizer"),
        'model_load_path': f"agent_model/train/{baseline}/{path}/checkpoint-20.pkl", # default checkpoint
    }
for baseline in bbo_list:
    config['baselines'][baseline] = {
        'optimizer': eval(f"{baseline}"),
    }
config = Config(config)
config, datasets = construct_problem_set(config)
baselines, config = get_baseline(config)

tester = Tester(config, baselines, datasets)
tester.test(log = False)

baseline_list = metabbo_list + bbo_list

# ------------ cal rank ------------
test_dir = f"{config.test_log_dir}_{config.test_problem}_{config.test_difficulty}"
print(test_dir)
# cal baseline mean and std for each problem

with open(test_dir + '/test_results.pkl', 'rb') as f:
    data = pickle.load(f)
    data = data['cost']

    problem_list = list(data.keys())

mean_result = np.zeros((len(baseline_list), len(problem_list)))
std_result = np.zeros((len(baseline_list), len(problem_list)))

for j, problem in enumerate(problem_list):
    problem_data = data[problem]
    for i, baseline in enumerate(baseline_list):
        baseline_data = problem_data[baseline] # List[run]
        run_gbest = []
        for run in baseline_data:
            run_gbest.append(run[-1])
        mean_result[i][j] = np.mean(run_gbest)
        std_result[i][j] = np.std(run_gbest)

m = len(baseline_list)
n = len(problem_list)
ranks = np.zeros((m, n))

for col in range(n):
    problem_data = list(zip(mean_result[:, col], std_result[:, col], range(m)))
    problem_data.sort()
    
    col_ranks = [0] * m
    rank = 1
    i = 0
    while i < m:
        j = i
        while j + 1 < m and problem_data[j][:2] == problem_data[j + 1][:2]:
            j += 1
        for k in range(i, j + 1):
            idx = data[k][2]
            col_ranks[idx] = rank
        rank += (j - i + 1)
        i = j + 1
    ranks[:, col] = col_ranks

avg_rank = np.mean(ranks, axis=1)
sorted_idx = np.argsort(avg_rank)
final_rank = np.empty_like(sorted_idx)
final_rank[sorted_idx] = np.arange(1, len(baseline_list) + 1)

df_mean = pd.DataFrame(mean_result, index=baseline_list, columns=problem_list)
df_std = pd.DataFrame(std_result, index=baseline_list, columns=problem_list)

df_rank = pd.DataFrame(ranks, index=baseline_list, columns=problem_list)
df_rank['avg_rank'] = avg_rank
df_rank['final_rank'] = final_rank

with pd.ExcelWriter('result_summary.xlsx', engine='openpyxl') as writer:
    df_mean.to_excel(writer, sheet_name='mean')
    df_std.to_excel(writer, sheet_name='std')
    df_rank.to_excel(writer, sheet_name='rank')
