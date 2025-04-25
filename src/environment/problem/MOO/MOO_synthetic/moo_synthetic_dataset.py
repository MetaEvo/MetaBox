import random

from torch.utils.data import Dataset
import numpy as np
from ....problem.MOO.MOO_synthetic.zdt_numpy import *
from ....problem.MOO.MOO_synthetic.uf_numpy import *
from ....problem.MOO.MOO_synthetic.dtlz_numpy import *
from ....problem.MOO.MOO_synthetic.wfg_numpy import *
from ....problem.MOO.MOO_synthetic.zdt_torch import *
from ....problem.MOO.MOO_synthetic.uf_torch import *
from ....problem.MOO.MOO_synthetic.dtlz_torch import *
from ....problem.MOO.MOO_synthetic.wfg_torch import *


class MOO_Synthetic_Dataset(Dataset):
    def __init__(self,
                 data,
                 batch_size = 1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.n_var)

    @staticmethod
    def get_datasets(version = 'numpy',
                     train_batch_size = 1,
                     test_batch_size = 1,
                     difficulty = None,
                     user_train_list = None,
                     user_test_list = None,
                     ):
        # get functions ID of indicated suit
        if difficulty == None and user_test_list == None and user_train_list == None:
            raise ValueError('Please set difficulty or user_train_list and user_test_list.')
        if difficulty not in ['easy', 'difficult', 'all', None]:
            raise ValueError(f'{difficulty} difficulty is invalid.')
        instance_set = []
        if difficulty == None:
            for problem in user_train_list:
                parts = problem.split("_")
                problem_name = parts[0]
                # 提取维数和目标数
                n_var = int([p for p in parts if p.startswith("d")][0][1:])  # 去掉前缀 'd'
                n_obj = int([p for p in parts if p.startswith("n")][0][1:])  # 去掉前缀 'n'
                train_set.append(eval(problem_name)(n_obj = n_obj, n_var = n_var))
            for problem in user_test_list:
                parts = problem.split("_")
                problem_name = parts[0]
                # 提取维数和目标数
                n_var = int([p for p in parts if p.startswith("d")][0][1:])
                n_obj = int([p for p in parts if p.startswith("n")][0][1:])
                test_set.append(eval(problem_name)(n_obj = n_obj, n_var = n_var))
        elif difficulty != None:
            # UF1-7
            for id in range(1, 8):
                if version == 'numpy':
                    instance_set.append(eval(f"UF{id}")())
                elif version == 'torch':
                    instance_set.append(eval(f"UF{id}_Torch")())

            # UF8-10
            for id in range(8, 11):
                if version == 'numpy':
                    instance_set.append(eval(f"UF{id}")())
                elif version == 'torch':
                    instance_set.append(eval(f"UF{id}_Torch")())

            # ZDT1-3
            for id in range(1, 4):
                if version == 'numpy':
                    instance_set.append(eval(f"ZDT{id}")(n_var = 30))
                elif version == 'torch':
                    instance_set.append(eval(f"ZDT{id}_Torch")(n_var = 30))

            # ZDT4 & ZDT6
            for id in [4, 6]:
                if version == 'numpy':
                    instance_set.append(eval(f"ZDT{id}")(n_var = 10))
                elif version == 'torch':
                    instance_set.append(eval(f"ZDT{id}_Torch")(n_var = 10))

            # DTLZ1
            dtlz1_settings = {
                2: [6],
                3: [7],
                5: [9],
                7: [11],
                8: [12],
                10: [14]
            }
            for n_obj, n_var_list in dtlz1_settings.items():
                for n_var in n_var_list:
                    if version == 'numpy':
                        instance_set.append(eval("DTLZ1")(n_obj = n_obj, n_var = n_var))
                    elif version == 'torch':
                        instance_set.append(eval("DTLZ1_Torch")(n_obj = n_obj, n_var = n_var))

            # DTLZ2-6
            for dtlz_id in range(2, 7):
                dtlz_settings = {
                    2: [11],
                    3: [11, 12] if dtlz_id != 3 and dtlz_id != 5 else [12],
                    5: [14],
                    7: [16],
                    8: [17],
                    10: [19]
                }
                for n_obj, n_var_list in dtlz_settings.items():
                    for n_var in n_var_list:
                        if version == 'numpy':
                            instance_set.append(eval(f"DTLZ{dtlz_id}")(n_obj = n_obj, n_var = n_var))
                        elif version == 'torch':
                            instance_set.append(eval(f"DTLZ{dtlz_id}_Torch")(n_obj = n_obj, n_var = n_var))

            # DTLZ7
            dtlz7_settings = {
                2: [21],
                3: [22],
                5: [24],
                7: [16, 26],
                8: [27],
                10: [29]
            }
            for n_obj, n_var_list in dtlz7_settings.items():
                for n_var in n_var_list:
                    if version == 'numpy':
                        instance_set.append(eval("DTLZ7")(n_obj = n_obj, n_var = n_var))
                    elif version == 'torch':
                        instance_set.append(eval("DTLZ7_Torch")(n_obj = n_obj, n_var = n_var))

            # WFG1-9
            for wfg_id in range(1, 10):
                wfg_settings = {
                    2: [12, 22],
                    3: [12, 14, 24],
                    5: [14, 18, 28],
                    7: [16],
                    8: [24, 34],
                    10: [28, 38]
                }
                for n_obj, n_var_list in wfg_settings.items():
                    for n_var in n_var_list:
                        if version == 'numpy':
                            instance_set.append(eval(f"WFG{wfg_id}")(n_obj = n_obj, n_var = n_var))
                        elif version == 'torch':
                            instance_set.append(eval(f"WFG{wfg_id}_Torch")(n_obj = n_obj, n_var = n_var))

            print(f"Total instances: {len(instance_set)}")
            # === 排序 ===
            instance_set.sort(key = lambda x: x.n_obj * x.n_var)
            train_set = []
            test_set = []
            if difficulty == 'easy':
                train_set = instance_set[:int(0.8 * len(instance_set))]
                test_set = instance_set[int(0.8 * len(instance_set)):]
            elif difficulty == 'difficult':
                train_set = instance_set[:int(0.2 * len(instance_set))]
                test_set = instance_set[int(0.2 * len(instance_set)):]
            elif difficulty == 'all':
                train_set = instance_set
                test_set = instance_set
            for i in range(len(train_set)):
                train_set[i].dim = train_set[i].n_var
            for i in range(len(test_set)):
                test_set[i].dim = test_set[i].n_var

        return MOO_Synthetic_Dataset(train_set, train_batch_size), MOO_Synthetic_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):

        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'MOO_Synthetic_Dataset'):
        return MOO_Synthetic_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)


if __name__ == '__main__':
    train_set, test_set = MOO_Synthetic_Dataset.get_datasets()
    print(train_set, test_set)
