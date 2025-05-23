import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


metabbo_list = ['RNNOPT', 'DEDDQN', 'DEDQN', 'LDE', 'RLPSO', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'RLDAS', 'SYMBOL', 'B2OPT', 'RLDEAFL']
no_zero_shot_list = ['RNNOPT', 'B2OPT', 'RLPSO', 'RLDAS']

# --------------------------------------- learning efficiency --------------------------------------


T0_list = [0.0] * len(metabbo_list)
Tg_list = []

for baseline in metabbo_list:
    path = f"train/{baseline}/checkpoint_log.txt"
    last_time = None
    with open(path, 'r') as f:
        for line in f:
            if "Time:" in line:
                time_str = line.strip().split("Time: ")[1].split("s")[0].strip()
                last_time = float(time_str)
    Tg_list.append(last_time)  # last time

c0_dir = f"test/bbob-10D-checkpoint-0/metadata"
c20_dir = f"test/bbob-10D/metadata"
c0_result = {}
c20_result = {}

problem_list = [f for f in os.listdir(f"{c0_dir}/{metabbo_list[0]}/") if f.endswith('.pkl')]
problem_list = [problem.split('.')[0] for problem in problem_list]

for baseline in metabbo_list:
    c0_total_performance = []
    c20_total_performance = []
    for problem in problem_list:
        c0_path = f"{c0_dir}/{baseline}/{problem}.pkl"
        c20_path = f"{c20_dir}/{baseline}/{problem}.pkl"
        with open(c0_path, 'rb') as f:
            c0_metadata = pickle.load(f)
        with open(c20_path, 'rb') as f:
            c20_metadata = pickle.load(f)
        c0_performance = []
        c20_performance = []
        for metarun in c0_metadata:
            cost_run = metarun['Cost']
            y_0 = np.min(cost_run[0])
            y_min = np.min(cost_run[0])
            for cost in cost_run:
                y_min = np.minimum(y_min, np.min(cost))
            c0_performance.append((y_min - y_0) / (0 - y_0 + 1e-20))
        for metarun in c20_metadata:
            cost_run = metarun['Cost']
            y_0 = np.min(cost_run[0])
            y_min = np.min(cost_run[0])
            for cost in cost_run:
                y_min = np.minimum(y_min, np.min(cost))
            c20_performance.append((y_min - y_0) / (0 - y_0 + 1e-20))
        c0_total_performance.append(np.mean(c0_performance))
        c20_total_performance.append(np.mean(c20_performance))
    c0_result[baseline] = np.mean(c0_total_performance)
    c20_result[baseline] = np.mean(c20_total_performance)

learning_efficiency = []
print("------------learning efficiency------------")
for i, baseline in enumerate(metabbo_list):
    learning_efficiency.append((c20_result[baseline] - c0_result[baseline]) / (Tg_list[i] - T0_list[i]))
    print(f"{baseline}: {learning_efficiency[i]}")

# ------------------------------------------------ anti-nfl ------------------------------------------------

zeroshot_problem_list = ['bbob-10D', 'bbob-noisy-10D', 'bbob-30D', 'bbob-noisy-30D', 'protein', 'uav', 'hpo-b']

# find gbest
gbest = {}

zero_shot_result = {}
for zero_shot_problem in zeroshot_problem_list:
    gbest[zero_shot_problem] = {}
    zero_shot_result[zero_shot_problem] = {}
    for baseline in metabbo_list:
        zero_shot_result[zero_shot_problem][baseline] = 0.0

    path = f"test/{zero_shot_problem}/metadata"
    problem_list = [f for f in os.listdir(f"{path}/{metabbo_list[0]}/") if f.endswith('.pkl')]
    problem_list = [problem.split('.')[0] for problem in problem_list]

    for problem in problem_list:
        gbest[zero_shot_problem][problem] = 1e32
        if "bbob" in zero_shot_problem:
            gbest[zero_shot_problem][problem] = 0.0 # synthetic problem

    for baseline in metabbo_list:
        if "10D" not in zero_shot_problem and baseline in no_zero_shot_list:
            continue
        total_performance = np.zeros((len(problem_list)))
        for problem in problem_list:
            baseline_path = f"{path}/{baseline}/{problem}.pkl"
            with open(baseline_path, 'rb') as f:
                metadata = pickle.load(f) # List[run]
            for metarun in metadata:
                cost_run = metarun['Cost']
                y_0 = np.min(cost_run[0])
                y_min = np.min(cost_run[0])
                for cost in cost_run:
                    y_min = np.minimum(y_min, np.min(cost))
                gbest[zero_shot_problem][problem] = min(gbest[zero_shot_problem][problem], y_min)

    for baseline in metabbo_list:
        if "10D" not in zero_shot_problem and baseline in no_zero_shot_list:
            continue
        total_performance = []
        for problem in problem_list:
            metadata_path = f"{path}/{baseline}/{problem}.pkl"
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f) # List[run]
            performance = []
            for metarun in metadata:
                cost_run = metarun['Cost']
                y_0 = np.min(cost_run[0])
                y_min = np.min(cost_run[0])
                for cost in cost_run:
                    y_min = np.minimum(y_min, np.min(cost))
                performance.append((y_min - y_0) / (gbest[zero_shot_problem][problem] - y_0 + 1e-20))
            performance = np.mean(performance)
            total_performance.append(performance)
        zero_shot_result[zero_shot_problem][baseline] = np.mean(total_performance)

anti_nfl_result = []
print("------------anti nfl------------")
for baseline in metabbo_list:
    baseline_anti_nfl = []
    for problem in zeroshot_problem_list:
        if problem == "bbob-10D":
            continue
        baseline_anti_nfl.append((zero_shot_result[problem][baseline] - zero_shot_result["bbob-10D"][baseline]) / zero_shot_result["bbob-10D"][baseline])
    anti_nfl_result.append(np.exp(np.mean(baseline_anti_nfl)))
    print(f"{baseline}: {anti_nfl_result[-1]}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].bar(metabbo_list, learning_efficiency, color='skyblue')
axes[0].set_ylabel("Learning Efficiency")
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(metabbo_list, anti_nfl_result, color='salmon')
axes[1].set_ylabel("Anti-NFL")
axes[1].tick_params(axis='x', rotation=45)

axes[2].scatter(learning_efficiency, anti_nfl_result, color='green')
for i, label in enumerate(metabbo_list):
    axes[2].annotate(label, (learning_efficiency[i], anti_nfl_result[i]), fontsize=9, xytext=(5, 2), textcoords='offset points')
axes[2].set_xlabel("Learning Efficiency")
axes[2].set_ylabel("Anti-NFL")
axes[2].set_title("Efficiency vs Anti-NFL")

plt.tight_layout()
plt.savefig('figure_7.pdf', dpi=300)
print("Figure_7 saved as learning_efficiency.pdf")
plt.show()

