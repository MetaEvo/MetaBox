import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from openpyxl import Workbook

# bbob-10D
test_dir = "test/bbob-10D/metadata"
metabbo_list = ['RNNOPT', 'DEDDQN', 'DEDQN', 'LDE', 'RLPSO', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'RLDAS', 'SYMBOL', 'B2OPT', 'RLDEAFL']
bbo_list = ['PSO', 'DE', 'SHADE', 'JDE21', 'MADDE']

# get problem_list
problem_list = [f for f in os.listdir(f"{test_dir}/{metabbo_list[0]}/") if f.endswith('.pkl')]
problem_list = [problem.split('.')[0] for problem in problem_list]

baseline_list = metabbo_list + bbo_list
mean_result = np.zeros((len(baseline_list), len(problem_list)))
std_result = np.zeros((len(baseline_list), len(problem_list)))

for i, baseline in enumerate(baseline_list):
    for j, problem in enumerate(problem_list):
        with open(f"{test_dir}/{baseline}/{problem}.pkl", 'rb') as f:
            data = pickle.load(f) # list[run]
            total_ = []
            for run in data:
                run_cost = run['Cost']
                y_0 = np.min(run_cost[0])
                y_min = np.min(run_cost[0])
                for cost in run_cost:
                    y_min = np.minimum(y_min, np.min(cost))
                total_.append(y_min)
            mean_result[i][j] = np.mean(total_)
            std_result[i][j] = np.std(total_)


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
            idx = problem_data[k][2]
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
