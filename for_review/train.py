from metaevobox import Config, Trainer
from metaevobox.baseline.metabbo import *
from metaevobox.environment.optimizer import *
from metaevobox.environment.problem.utils import construct_problem_set

# put user-specific configuration
config = {'train_problem': 'bbob-10D',
          'train_difficulty': 'difficult',
          'train_batch_size': 16,
          'train_parallel_mode':'subproc',
          'agent_save_dir': 'train/',
          }

baseline_list = ['RNNOPT', 'DEDDQN', 'DEDQN', 'LDE', 'RLPSO', 'RLEPSO', 'NRLPSO', 'LES', 'GLEET', 'GLHF', 'RLDAS', 'SYMBOL', 'B2OPT', 'RLDEAFL']

dir_results = []

for baseline in baseline_list:
    baseline_config = config.copy()
    if baseline == "GLHF" or baseline == "B2OPT" or baseline == "RNNOPT":
        baseline_config['train_problem'] = "bbob-torch-10D"
        baseline_config['train_parallel_mode'] = 'dummy' # these baselines are not compatible with subproc

    tmp_config = Config(baseline_config)
    tmp_config.train_name = ""
    # construct dataset
    tmp_config, datasets = construct_problem_set(tmp_config)

    metabbo = eval(baseline)(tmp_config)
    metabbo_opt = eval(f"{baseline}_Optimizer")(tmp_config)
    trainer = Trainer(tmp_config, metabbo, metabbo_opt, datasets)
    trainer.train()
    dir_results.append(f"train/{baseline}/")

for i in range(len(dir_results)):
    print(f"MetaBBO {baseline_list[i]}: {dir_results[i]}")

# train/GLEET/checkpoint-0.pkl