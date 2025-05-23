import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt


zero_shot_problem = ['bbob-noisy-30D', 'protein', 'uav', 'hpo-b']
metabbo_list = ['DEDDQN', 'DEDQN', 'LDE', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'SYMBOL', 'RLDEAFL']
bbo_list = ['PSO', 'DE', 'SHADE', 'JDE21', 'MADDE']
baseline_list = metabbo_list + bbo_list
total_result = {}
for test_problem in zero_shot_problem:
    # only draw protein zero-shot curve 
    test_dir = f"test/{test_problem}/metadata"

    # get problem_list
    problem_list = [f for f in os.listdir(f"{test_dir}/{metabbo_list[0]}/") if f.endswith('.pkl')]
    problem_list = [problem.split('.')[0] for problem in problem_list]

    result = {}
    gbest = {}

    for problem in problem_list:
        gbest[problem] = 1e32
        if test_problem == "bbob-noisy-30D":
            gbest[problem] = 0.0
    
    test_run = 51
    # ------------ find gbest for each problem------------
    maxlen_index = {}
    for baseline in baseline_list:
        maxlen_index[baseline] = 0
        max_len = 0
        for problem in problem_list:
            metabbo_path = f"{test_dir}/{baseline}/{problem}.pkl"
            with open(metabbo_path, 'rb') as f:
                metadata = pickle.load(f)
            test_run = len(metadata)
            for i, metarun in enumerate(metadata):
                cost_run = metarun['Cost']
                y_0 = np.min(cost_run[0])
                y_min = np.min(cost_run[0])
                
                for cost in cost_run:
                    y_min = np.minimum(y_min, np.min(cost))
                    gbest[problem] = min(gbest[problem], y_min)

                if max_len < len(cost_run):
                    max_len = len(cost_run)
                    maxlen_index[baseline] = i

    for baseline in baseline_list:
        result[baseline] = {
            'fes': [],
            'mean': [],
            'std': [],
        }
        for problem in problem_list:
            metadata_path = f"{test_dir}/{baseline}/{problem}.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f) # List[run]
            
            if result[baseline]['fes'] == []:
                runs = metadata[maxlen_index[baseline]]['Cost']
                for cost in runs:
                    result[baseline]['fes'].append(len(cost) + result[baseline]['fes'][-1] if result[baseline]['fes'] else 0)
            
            performance_total = [[] for _ in test_run]

            for i, metarun in enumerate(metadata):
                cost_run = metarun['Cost']
                y_0 = np.min(cost_run[0])
                y_min = np.min(cost_run[0])
                for cost in cost_run:
                    y_min = np.minimum(y_min, np.min(cost))
                
                    performance = (y_min - y_0) / (gbest[problem] - y_0 + 1e-20)
                    performance_total[i].append(performance)
                while len(performance_total[i]) < len(result[baseline]['fes']):
                    performance_total[i].append(performance_total[i][-1])
            
            performance_total = np.array(performance_total)
            performance_mean = np.mean(performance_total, axis=0)
            performance_std = np.std(performance_total, axis=0)
            result[baseline]['mean'].append(performance_mean)
            result[baseline]['std'].append(performance_std)
        
        result[baseline]['mean'] = np.array(result[baseline]['mean'])
        result[baseline]['std'] = np.array(result[baseline]['std'])
        result[baseline]['mean'] = np.mean(result[baseline]['mean'], axis=0)
        result[baseline]['std'] = np.mean(result[baseline]['std'], axis=0)
    total_result[test_problem] = result

# plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, problem in enumerate(zero_shot_problem):
    ax = axes[i]
    result = total_result[problem]
    
    for baseline, data in result.items():
        fes = np.array(data['fes'])
        mean = np.array(data['mean'])
        std = np.array(data['std'])

        ax.plot(fes, mean, label=baseline)
        ax.fill_between(fes, mean - std, mean + std, alpha=0.3)
    
    ax.set_title(f'Zero-Shot Performance in {problem}')
    ax.set_xlabel('FEs')
    ax.set_ylabel('Performance')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()