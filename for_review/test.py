from metaevobox import Tester, Config, construct_problem_set, get_baseline
from metaevobox.baseline.metabbo import *
from metaevobox.baseline.bbo import *
from metaevobox.environment.optimizer import *

# test bbob-10D difficult
metabbo_list = ['RNNOPT', 'DEDDQN', 'DEDQN', 'LDE', 'RLPSO', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'RLDAS', 'SYMBOL', 'B2OPT', 'RLDEAFL']
metabbo_path = "train/"
bbo_list = ['PSO', 'DE', 'SHADE', 'JDE21', 'MADDE']
no_zero_shot_list = ['RNNOPT', 'B2OPT', 'RLPSO', 'RLDAS']


test_problem_list = ['bbob-10D', 'bbob-noisy-10D', 'bbob-30D', 'bbob-noisy-30D', 'protein', 'uav', 'hpo-b']
test_problem_difficulty = ['difficult', 'all', 'all', 'all', 'all', 'all', 'all']
test_batch_size_list = [16, 16, 16, 16, 64, 56, 64]

for problem, difficulty, batch_size in zip(test_problem_list, test_problem_difficulty, test_batch_size_list):
    config = {
        'test_problem': problem,
        'test_difficulty': difficulty,
        'test_batch_size': batch_size,
        'full_meta_data': True,
        'log_dir': 'test/',
        'baselines': {},
    }
    for baseline in metabbo_list:
        if "10D" not in problem and baseline in no_zero_shot_list:
            continue
        config['baselines'][baseline] = {
            'agent': baseline,
            'optimizer': eval(f"{baseline}_Optimizer"),
            'model_load_path': f"{metabbo_path}{baseline}/checkpoint-20.pkl", # default checkpoint
        }
    for baseline in bbo_list:
        config['baselines'][baseline] = {
            'optimizer': eval(f"{baseline}"),
        }
    config = Config(config)
    config.test_log_dir = config.log_dir + f"{problem}"
    # test/bbob-10D
    config, datasets = construct_problem_set(config)
    baselines, config = get_baseline(config)
    tester = Tester(config, baselines, datasets)
    tester.test(log = False)

for i in range(len(test_problem_list)):
    print(f"Test {test_problem_list[i]}: test/{test_problem_list[i]}")

print("Now test checkpoint-0 to plot the learning efficiency")
# test checkpoint-0
config = {
    'test_problem': "bbob-10D",
    'test_difficulty': "difficult",
    'test_batch_size': 16,
    'full_meta_data': True,
    'log_dir': 'test/',
    'baselines': {},
}
for baseline in metabbo_list:
    config['baselines'][baseline] = {
        'agent': baseline,
        'optimizer': eval(f"{baseline}_Optimizer"),
        'model_load_path': f"{metabbo_path}{baseline}/checkpoint-0.pkl",
    }
config = Config(config)
config.test_log_dir = config.log_dir + f"bbob-10D-checkpoint-0"
# test/bbob-10D-checkpoint-0
config, datasets = construct_problem_set(config)
baselines, config = get_baseline(config)
tester = Tester(config, baselines, datasets)
tester.test(log = False)