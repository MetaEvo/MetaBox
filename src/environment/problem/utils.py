import pickle
import os

from .MMO.CEC2013MMO.cec2013mmo_dataset import CEC2013MMO_Dataset
from .MOO.MOO_synthetic.moo_synthetic_dataset import MOO_Synthetic_Dataset
from .MTO.WCCI2020.wcci2020_dataset import WCCI2020_Dataset
from .MTO.CEC2017MTO.cec2017mto_dataset import CEC2017MTO_Dataset
from .SOO.COCO_BBOB.bbob_dataset import BBOB_Dataset
from .SOO.COCO_BBOB.bbob_surrogate import bbob_surrogate_Dataset
from .SOO.CEC2013LSGO.cec2013lsgo_dataset import CEC2013LSGO_Dataset
from .SOO.UAV.uav_dataset import UAV_Dataset
from .SOO.PROTEIN_DOCKING.protein_docking_dataset import Protein_Docking_Dataset
from .SOO.HPO_B.hpob_dataset import HPOB_Dataset
from .MOO.UAV.uav_dataset import UAV_Dataset as MMO_UAV_Dataset

def construct_problem_set(config):
    if config.train_problem == config.test_problem and config.train_difficulty == config.test_difficulty:
        train_set, test_set = get_problem_set(config, config.train_problem, config.train_difficulty, config.user_train_problem_list, config.user_test_problem_list)
        config.dim = max(train_set.maxdim, test_set.maxdim)
        return config, (train_set, test_set)

    train_set = get_problem_set(config, config.train_problem, config.train_difficulty, config.user_train_problem_list, config.user_test_problem_list)[0]
    test_set = get_problem_set(config, config.test_problem, config.test_difficulty, config.user_train_problem_list, config.user_test_problem_list)[1]
    config.dim = max(train_set.maxdim, test_set.maxdim)
    return config, (train_set, test_set)


def get_problem_set(config, problem, difficulty, train_list, test_list):
    if problem in ['bbob-10D', 'bbob-30D', 'bbob-torch-10D', 'bbob-torch-30D',
                   'bbob-noisy-10D', 'bbob-noisy-30D', 'bbob-noisy-torch-10D', 'bbob-noisy-torch-30D']:
        return BBOB_Dataset.get_datasets(suit=problem,
                                        upperbound=config.upperbound,
                                        train_batch_size=config.train_batch_size,
                                        test_batch_size=config.test_batch_size,
                                        difficulty=difficulty,
                                        user_train_list=train_list,
                                        user_test_list=test_list,
                                        version='torch' if 'torch' in problem else 'numpy',
                                        device = config.device)

    elif problem in ['protein', 'protein-torch']:
        return Protein_Docking_Dataset.get_datasets(version='torch' if 'torch' in problem else 'numpy',
                                                    train_batch_size=config.train_batch_size,
                                                    test_batch_size=config.test_batch_size,
                                                    user_train_list=train_list,
                                                    user_test_list=test_list,
                                                    difficulty=difficulty)

    elif problem in ['bbob-surrogate-2D','bbob-surrogate-5D','bbob-surrogate-10D']:
        return bbob_surrogate_Dataset.get_datasets(config=config,
                                                   suit=problem,
                                                    upperbound=config.upperbound,
                                                    train_batch_size=config.train_batch_size,
                                                    test_batch_size=config.test_batch_size,
                                                    difficulty=difficulty,
                                                    user_train_list=train_list,
                                                    user_test_list=test_list,
                                                    version='torch' if 'torch' in problem else 'numpy')

    elif problem in ['lsgo', 'lsgo-torch']:
        return CEC2013LSGO_Dataset.get_datasets(train_batch_size = config.train_batch_size,
                                                 test_batch_size = config.test_batch_size,
                                                 difficulty = difficulty,
                                                 version='torch' if 'torch' in problem else 'numpy',
                                                 user_train_list=train_list,
                                                 user_test_list=test_list,
                                                 )
    elif problem in ['uav']:
        return UAV_Dataset.get_datasets(train_batch_size = config.train_batch_size,
                                        test_batch_size = config.test_batch_size,
                                        dv = 10,
                                        j_pen = 1e4,
                                        mode = "standard",
                                        num = 56,
                                        difficulty = difficulty, 
                                        user_train_list=train_list,
                                        user_test_list=test_list,
                                        version='torch' if 'torch' in problem else 'numpy',
                                        path = config.uav_path)
    elif problem in ['hpo-b']:
        return HPOB_Dataset.get_datasets(train_batch_size = config.train_batch_size,
                                        test_batch_size = config.test_batch_size,
                                        upperbound=config.upperbound,
                                        difficulty = difficulty,
                                        user_train_list=train_list,
                                        user_test_list=test_list,
                                        datapath = config.hpob_path
                                        )
    elif problem in ['ne']:
        from .SOO.NE.ne_dataset import NE_Dataset
        return NE_Dataset.get_datasets(train_batch_size = config.train_batch_size,
                                        test_batch_size = config.test_batch_size,
                                        difficulty = difficulty,
                                        user_train_list=train_list,
                                        user_test_list=test_list,
                                        )
    elif problem in ['mmo', 'mmo-torch']:
        return CEC2013MMO_Dataset.get_datasets(
                                            train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=difficulty,
                                            user_train_list=train_list,
                                            user_test_list=test_list,
                                            version='torch' if 'torch' in problem else 'numpy')
    elif problem in ['wcci2020']:
        return WCCI2020_Dataset.get_datasets(train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=difficulty,
                                            user_train_list=train_list,
                                            user_test_list=test_list,
                                            version='torch' if 'torch' in problem else 'numpy')
    elif problem in ['cec2017mto']:
        return CEC2017MTO_Dataset.get_datasets(train_batch_size=config.train_batch_size,
                                            test_batch_size=config.test_batch_size,
                                            difficulty=difficulty,
                                            user_train_list=train_list,
                                            user_test_list=test_list,
                                            version='torch' if 'torch' in problem else 'numpy')
    elif problem in ['moo-synthetic']:
        return MOO_Synthetic_Dataset.get_datasets(train_batch_size=config.train_batch_size,
                                        test_batch_size=config.test_batch_size,
                                        difficulty=difficulty,
                                        user_train_list=train_list,
                                        user_test_list=test_list,
                                        version='torch' if 'torch' in problem else 'numpy')
    elif problem in ['moo-uav']:
        return MMO_UAV_Dataset.get_datasets(train_batch_size = config.train_batch_size,
                                            test_batch_size = config.test_batch_size,
                                            dv = 10,
                                            j_pen = 1e4,
                                            mode = "standard",
                                            num = 56,
                                            difficulty = difficulty,
                                            user_train_list=train_list,
                                            user_test_list=test_list,
                                            version='torch' if 'torch' in problem else 'numpy',
                                            path = config.uav_path)

    else:
        raise ValueError(problem + ' is not defined!')
    
        
