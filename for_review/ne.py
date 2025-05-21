from metaevobox import Config, Tester, get_baseline, construct_problem_set
from metaevobox.environment.optimizer import *
from metaevobox.baseline.metabbo import *
from metaevobox.baseline.bbo import *
import pickle
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

metabbo_list = ['GLEET', 'RLEPSO', 'LDE']
metabbo_path = []

bbo_list = ['GLPSO', 'CMAES']
test_problem_list = ['ant-3', 'ant-4', 'ant-5']

config = {
    'test_problem':'ne', # specify the problem set you want to benchmark
    'baselines':{},
    'user_test_problem_list': test_problem_list,
    'test_run': 10,
    'full_meta_data': True, # need metadata to record
    'test_parallel_mode': "Serial",
}

for path, metabbo in zip(metabbo_path, metabbo_list):
    config['baselines'][metabbo] = {
        'agent': metabbo,
        'optimizer': eval(f"{metabbo}_Optimizer"),
        'model_load_path': path,
    }

for bbo in bbo_list:
    config['baselines'][bbo] = {
        'optimizer': eval(f"{bbo}"),
    }

config = Config(config)
config, datasets = construct_problem_set(config)
baselines, config = get_baseline(config)
tester = Tester(config, baselines, datasets)
tester.test(log = False)

# plot

test_output_dir = f"{config.test_log_dir}_{config.test_problem}_{config.test_difficulty}"

baseline_list = metabbo_list + bbo_list

result = {}
for baseline in baseline_list:
    result[baseline] = {}
    for problem in test_problem_list:

        result[baseline][problem] = {
            'fes': [],
            'mean': [[] for _ in range(config.test_run)],
            'std': [[] for _ in range(config.test_run)],
        }


        metadata_path = f"{test_output_dir}/metadata/{baseline}/{problem}.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f) # List[run]

        if result[baseline]['fes'] == []:
            run0 = metadata[0]['Cost']
            for cost in run0:
                result[baseline]['fes'].append(len(cost) + result[baseline]['fes'][-1] if result[baseline]['fes'] else 0)

        for i, metarun in enumerate(metadata):
            cost_run = metarun['Cost']
            y_0 = np.min(cost_run[0])
            y_min = y_0.copy()
            for cost in cost_run:
                y_min = np.minimum(y_min, np.min(cost))

                reward = 1e5 - y_min # convert to reward
                result[baseline]['mean'][i].append(reward)



FEs = {}
result = {}

for baseline in baseline_list:
    FEs[baseline] = []

for problem in test_problem_list:
    result[problem] = {}
    for baseline in baseline_list:
        result[problem][baseline] = {}

        metadata_path = f"{test_output_dir}/metadata/{baseline}/{problem}.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f) # List[run]

        if FEs[baseline] == []:
            run0 = metadata[0]['Cost']
            for cost in run0:
                FEs[baseline].append(len(cost) + FEs[baseline][-1] if FEs[baseline] else 0)

        total_reward = [[] for _ in range(config.test_run)]

        for i, metarun in enumerate(metadata):
            cost_run = metarun['Cost']
            y_0 = np.min(cost_run[0])
            y_min = y_0.copy()
            for cost in cost_run:
                y_min = np.minimum(y_min, np.min(cost))

                reward = 1e5 - y_min # convert to reward
                total_reward[i].append(reward)
        total_reward = np.array(total_reward) # run * Gen
        total_reward_mean = np.mean(total_reward, axis=0) # [Gen]
        total_reward_std = np.std(total_reward, axis=0) # [Gen]
        result[problem][baseline]['mean'] = total_reward_mean
        result[problem][baseline]['std'] = total_reward_std

# plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for idx, problem in enumerate(test_problem_list):
    ax = axes[idx]
    for baseline in baseline_list:
        x = FEs[baseline]
        y = result[problem][baseline]['mean']
        yerr = result[problem][baseline]['std']

        # y if negative, ant can't move in fact
        is_negative = y < 0
        y[is_negative] = -0.5
        yerr[is_negative] = 0

        ax.plot(x, y, label=baseline)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    ax.set_title(problem)
    ax.set_xlabel('FEs')
    ax.set_ylabel('Total Reward')
    ax.grid(True)
    ax.legend(fontsize='small')

plt.tight_layout()
plt.show()

