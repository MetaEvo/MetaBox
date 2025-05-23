import argparse
import time
import os
import subprocess, sys
import torch
def Config(user_config, datasets_dir = None):
    default_config = get_config()
    for key, value in user_config.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)

    default_config = init_config(default_config)

    torch.set_default_dtype(torch.float64)
    # find out if it is an HPO-B task
    is_hpo_b = 'hpo-b' in default_config.train_problem or 'hpo-b' in default_config.test_problem

    if is_hpo_b:
        print("Detected HPO-B Problem.")

        # 数据集目录默认使用当前工作目录
        if datasets_dir is None:
            datasets_dir = os.path.join(os.getcwd(), "metabox_data")
        os.makedirs(datasets_dir, exist_ok = True)

        # 检查该目录下是否已有对应数据文件
        data_dir = datasets_dir + "/HPO-B-main/hpob-data/"
        surrogates_dir = datasets_dir + "/HPO-B-main/saved-surrogates/"
        # expected_files = ['hpob-data/meta-train-dataset.json', 'hpob-data/meta-test-dataset.json', 'hpob-data/meta-validation-dataset.json']  # 你可以换成真实文件名
        # missing_files = [f for f in expected_files if not os.path.exists(os.path.join(datasets_dir, f))]
        missing_files = not os.path.exists(data_dir) or len(os.listdir(data_dir)) != 7 or not os.path.exists(surrogates_dir) or len(os.listdir(surrogates_dir)) != 1909

        if missing_files:
            print(f"[Warning] HPO-B dataset files not found")  # Too many files to display
            print(f"Expected in directory: {datasets_dir}")
            # 可以在这里加入自动下载逻辑 if you want
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                # check the required package, if not exists, pip install it
                try:
                    subprocess.check_call([sys.executable, '-m', "pip", "install", 'huggingface_hub'])
                    # print("huggingface_hub has been installed successfully!")
                    from huggingface_hub import snapshot_download
                except subprocess.CalledProcessError as e:
                    print(f"Install huggingface_hub leads to errors: {e}")

            snapshot_download(repo_id = 'GMC-DRL/MetaBox-HPO-B', repo_type = "dataset", local_dir = datasets_dir)
            print("Extract data...")
            os.system(f'tar -xf {datasets_dir}/HPO-B-main.tar.gz -C {datasets_dir}')
            os.remove(f'{datasets_dir}/HPO-B-main.tar.gz')
            os.remove(f"{datasets_dir}/.gitattributes")
        else:
            print(f"HPO-B dataset is ready in: {datasets_dir}/HPO-B-main")
        default_config.hpob_path = datasets_dir + '/'

    # 判断是不是 uav 任务
    is_uav = 'uav' in default_config.train_problem or 'uav' in default_config.test_problem
    if is_uav:
        print("Detected UAV Problem.")

        # 数据集目录默认使用当前工作目录
        if datasets_dir is None:
            datasets_dir = os.path.join(os.getcwd(), "metabox_data", "uav")
        os.makedirs(datasets_dir, exist_ok = True)

        # 检查该目录下是否已有对应数据文件
        expected_files = ['Model56.pkl']  # 你可以换成真实文件名
        missing_files = [f for f in expected_files if not os.path.exists(os.path.join(datasets_dir, f))]

        if missing_files:
            print(f"[Warning] UAV dataset files not found: {missing_files}")
            print(f"Expected in directory: {datasets_dir}")
            # 可以在这里加入自动下载逻辑 if you want
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                # check the required package, if not exists, pip install it
                try:
                    subprocess.check_call([sys.executable, '-m', "pip", "install", 'huggingface_hub'])
                    # print("huggingface_hub has been installed successfully!")
                    from huggingface_hub import snapshot_download
                except subprocess.CalledProcessError as e:
                    print(f"Install huggingface_hub leads to errors: {e}")
            snapshot_download(repo_id = 'GMC-DRL/MetaBox-uav', repo_type = "dataset", local_dir = datasets_dir)
        else:
            print(f"UAV dataset is ready in: {datasets_dir}")
        default_config.uav_path = datasets_dir + '/Model56.pkl'

    return default_config

def init_config(config):
    config.n_logpoint = 50

    if config.test_problem is None:
        config.test_problem = config.train_problem
    if config.test_difficulty is None:
        config.test_difficulty = config.train_difficulty
    if config.end_mode == 'epoch':
        config.max_learning_step = 1e9

    if 'protein' in config.train_problem or 'protein' in config.test_problem:
        config.dim = 12
        config.maxFEs = 2000
    elif 'hpo-b' in config.train_problem or 'hpo-b' in config.test_problem:
        config.maxFEs = 2000
    elif 'uav' in config.train_problem or 'uav' in config.test_problem:
        config.maxFEs = 2500
    elif 'lsgo' in config.train_problem or 'lsgo' in config.test_problem:
        config.maxFEs = 3e6
    elif "30D" in config.train_problem or '30D' in config.test_problem:
        config.maxFEs = 50000
    elif "ne" in config.train_problem or "ne" in config.test_problem:
        config.maxFEs = 2500
    else:
        config.maxFEs = 20000

    config.run_time = time.strftime("%Y%m%dT%H%M%S")
    config.train_name = f'{config.run_time}_{config.train_problem}_{config.train_difficulty}'
    config.test_log_dir = config.log_dir + 'test/' + f'{config.run_time}_{config.test_problem}_{config.test_difficulty}'
    config.rollout_log_dir = config.log_dir + 'rollout/' + f'{config.run_time}'
    config.mgd_test_log_dir = config.log_dir + 'mgd_test/' + f'{config.run_time}'
    config.mte_test_log_dir = config.log_dir + 'mte_test/' + f'{config.run_time}'

    if config.end_mode == "step":
        config.save_interval = config.max_learning_step // config.n_checkpoint
    elif config.end_mode == "epoch":
        config.save_interval = config.max_epoch // config.n_checkpoint
    config.log_interval = config.maxFEs // config.n_logpoint
    return config

def get_config(args=None):
    parser = argparse.ArgumentParser()
    # ------------------------------ The Config of Problem Setup ------------------------------
    parser.add_argument('--train_problem', default = 'bbob-10D',
                        choices = ['bbob-10D', 'bbob-30D', 'bbob-torch-10D', 'bbob-torch-30D', 'bbob-noisy-10D', 'bbob-noisy-30D',
                                   'bbob-noisy-torch-10D', 'bbob-noisy-torch-30D', 'bbob-surrogate-2D','bbob-surrogate-5D','bbob-surrogate-10D',
                                   'hpo-b', 'lsgo', 'lsgo-torch', 'protein', 'protein-torch', 'uav',
                                   'mmo', 'mmo-torch', 'wcci2020', 'cec2017mto', 'moo-synthetic', 'moo-uav'],
                        help='specify the problem suite for training')
    parser.add_argument('--test_problem', default = None,
                        choices = [None, 'bbob-10D', 'bbob-30D', 'bbob-torch-10D', 'bbob-torch-30D', 'bbob-noisy-10D', 'bbob-noisy-30D',
                                   'bbob-noisy-torch-10D', 'bbob-noisy-torch-30D', 'bbob-surrogate-2D', 'bbob-surrogate-5D', 'bbob-surrogate-10D',
                                   'hpo-b', 'lsgo', 'lsgo-torch', 'protein', 'protein-torch', 'uav',
                                   'mmo', 'mmo-torch', 'wcci2020', 'cec2017mto', 'moo-synthetic', 'moo-uav'],
                        help='specify the problem suite for testing, default to be consistent with training')
    parser.add_argument('--train_difficulty', default='easy', choices=['all', 'easy', 'difficult'],
                        help='difficulty level for training problems')
    parser.add_argument('--test_difficulty', default=None, choices=['all', 'easy', 'difficult'],
                        help='difficulty level for testing problems, default to be consistent with training')

    parser.add_argument('--user_train_problem_list', nargs='+', default=None, help = 'user define training problem list')
    parser.add_argument('--user_test_problem_list', nargs='+', default=None, help = 'user define testing problem list')
    parser.add_argument('--device', default='cpu', help='device to use')
    parser.add_argument('--upperbound', type=float, default=5, help='upperbound of search space')

    # ------------------------------ The Config of Training Mode ------------------------------
    parser.add_argument('--max_learning_step', type=int, default=1500000, help='the maximum learning step for training')
    parser.add_argument('--max_epoch', type = int, default = 100, help = 'the maximum number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size of train set')
    parser.add_argument('--agent_save_dir', type = str, default = 'agent_model/train/',
                        help = 'save your own trained agent model')
    parser.add_argument('--n_checkpoint', type=int, default=20, help='number of training checkpoints')
    parser.add_argument('--train_parallel_mode', type=str, default='subproc', choices=['dummy', 'subproc', 'ray'],
                        help='the parellel processing method for batch env step in training')
    parser.add_argument('--train_mode', default='single', type = str, choices = ['single', 'multi'],
                        help = 'training mode：single fixed learning step, multi fixed number of problems encountered')
    parser.add_argument('--end_mode', type = str, default = 'epoch', choices = ['step', 'epoch'])
    parser.add_argument('--no_tb', action='store_true', default = False, help = 'disable tensorboard logging')

    # ------------------------------ The Config of Testing Mode ------------------------------
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'batch size of test set')
    parser.add_argument('--test_parallel_mode', type = str, default = 'Batch', choices = ['Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch', 'Serial'],
                        help = 'the parellel processing mode for testing')
    parser.add_argument('--baselines', type = dict, default = None, help = 'the baselines for testing test')
    parser.add_argument('--test_run', type = int, default = 51, help = 'the run number of test')
    parser.add_argument('--rollout_run', type = int, default = 10, help = 'The run number of rollout')

    # ------------------------------ General Parameters ------------------------------
    parser.add_argument('--seed', type = int, default = 3849, help = 'Random seed for training and test to fixed agent')
    parser.add_argument('--full_meta_data', type=bool, default=False, help='store the metadata')
    parser.add_argument('--log_dir', type=str, default='output/',
                        help='logging output')

    config = parser.parse_args(args)

    # # for bo, maxFEs is relatively smaller due to time limit
    # config.n_logpoint = 50
    #
    # if config.test_problem is None:
    #     config.test_problem = config.train_problem
    # if config.test_difficulty is None:
    #     config.test_difficulty = config.train_difficulty
    # if config.end_mode == 'epoch':
    #     config.max_learning_step = 1e9
    # # if config.run_experiment and len(config.agent_for_cp) >= 1:
    # #     assert config.agent_load_dir is not None, "Option --agent_load_dir must be given since you specified option --agent_for_cp."
    #
    # if config.mgd_test or config.mte_test:
    #     config.train_problem = config.problem_to
    #     config.train_difficulty = config.difficulty_to
    #
    # if config.train_problem in ['protein', 'protein-torch']:
    #     config.dim = 12
    #     config.maxFEs = 1000
    #     config.n_logpoint = 5
    #
    # config.run_time = f'{time.strftime("%Y%m%dT%H%M%S")}_{config.train_problem}_{config.train_difficulty}'
    # config.test_log_dir = config.log_dir + 'test/' + config.run_time + '/'
    # config.rollout_log_dir = config.log_dir + 'rollout/' + config.run_time + '/'
    # config.mgd_test_log_dir = config.log_dir + 'mgd_test/' + config.run_time + '/'
    # config.mte_test_log_dir = config.log_dir + 'mte_test/' + config.run_time + '/'
    #
    # if config.end_mode == "step":
    #     config.save_interval = config.max_learning_step // config.n_checkpoint
    # elif config.end_mode == "epoch":
    #     config.save_interval = config.max_epoch // config.n_checkpoint
    # config.log_interval = config.maxFEs // config.n_logpoint

    return config
