import itertools
import numpy as np
from scipy.special import comb
from ....problem.basic_problem import Basic_Problem


def crtup(n_obj, n_ref_points = 1000):
    def find_H_for_closest_points(N, M):
        """
        根据目标点数 N 和维数 M，找到最接近的 H，使得生成的点数不超过 N。
        """
        # 设定初始搜索范围
        H_min, H_max = 1, 100000  # 假设 H 的范围在 1 到 100 之间，具体可根据实际情况调整
        closest_H = H_min
        closest_diff = float('inf')
        closest_N = 0
        # 搜索最接近 N 的 H
        for H in range(H_min, H_max + 1):
            generated_points = int(comb(H + M - 1, M - 1))  # 计算生成的点数

            # 如果生成的点数超过目标 N，跳过此 H
            if generated_points > N:
                break

            diff = abs(generated_points - N)  # 计算与目标 N 的差异

            # 如果当前差异更小，则更新最接近的 H 和差异
            if diff < closest_diff:
                closest_H = H
                closest_diff = diff
                closest_N = generated_points

        return closest_H, closest_N

    M = n_obj
    H, closest_N = find_H_for_closest_points(n_ref_points, M)
    n_comb = int(comb(H + M - 1, M - 1))
    combinations = list(itertools.combinations(range(1, H + M), M - 1))
    temp = np.array([np.arange(0, M - 1)] * n_comb)
    if len(combinations) == len(temp):
        result = []
        for combination, arr in zip(combinations, temp):
            # 元组元素与数组元素相减
            sub_result = np.array(combination) - arr - 1
            result.append(sub_result)
    else:
        print("两个列表长度不一致，无法相减。")
    result = np.array(result)
    W = np.zeros((n_comb, M))
    W[:, 0] = result[:, 0] - 0  # 第一列直接是 Temp 的第一列
    for i in range(1, M - 1):
        W[:, i] = result[:, i] - result[:, i - 1]  # 后续列是 Temp 当前列减去前一列
    W[:, -1] = H - result[:, -1]  # 最后一列是 H - Temp 最后一列

    W = W / H
    return W, n_comb


# Basic_Problem
class UF1(Basic_Problem):
    def __init__(self):
        self.n_obj = 2
        self.n_var = 30
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        f1 = x1 + 2 * np.mean((Vars[:, J1] - np.sin(6 * np.pi * x1 + (J1 + 1) * np.pi / self.n_var)) ** 2, 1,
                              keepdims = True)
        f2 = 1 - np.sqrt(np.abs(x1)) + 2 * np.mean(
            (Vars[:, J2] - np.sin(6 * np.pi * x1 + (J2 + 1) * np.pi / self.n_var)) ** 2, 1, keepdims = True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF2(Basic_Problem):
    def __init__(self):
        self.n_obj = 2  # 初始化（目标维数）
        self.n_var = 30  # 初始化（决策变量维数）
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 得到决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        yJ1 = Vars[:, J1] - (
                0.3 * x1 ** 2 * np.cos(24 * np.pi * x1 + 4 * (J1 + 1) * np.pi / self.n_var) + 0.6 * x1) * np.cos(
            6 * np.pi * x1 + (J1 + 1) * np.pi / self.n_var)
        yJ2 = Vars[:, J2] - (
                0.3 * x1 ** 2 * np.cos(24 * np.pi * x1 + 4 * (J2 + 1) * np.pi / self.n_var) + 0.6 * x1) * np.sin(
            6 * np.pi * x1 + (J2 + 1) * np.pi / self.n_var)
        f1 = x1 + 2 * np.mean((yJ1) ** 2, 1, keepdims = True)
        f2 = 1 - np.sqrt(np.abs(x1)) + 2 * np.mean((yJ2) ** 2, 1, keepdims = True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF3(Basic_Problem):  # 继承Problem的父类
    def __init__(self):
        self.n_obj = 2  # 目标维数
        self.n_var = 30  # 决策变量维数
        self.lb = np.array([0] * self.n_var)
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - x1 ** (0.5 * (1 + (3 * (J - 2) / (self.n_var - 2))))
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1 + (2 / len(J1)) * (4 * np.sum(yJ1 ** 2, 1, keepdims = True) -
                                   2 * (np.prod(np.cos((20 * yJ1 * np.pi) / (np.sqrt(J1))), 1, keepdims = True)) + 2)
        f2 = 1 - np.sqrt(x1) + (2 / len(J2)) * (4 * np.sum(yJ2 ** 2, 1, keepdims = True) -
                                                2 * (np.prod(np.cos((20 * yJ2 * np.pi) / (np.sqrt(J2))), 1,
                                                             keepdims = True)) + 2)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF4(Basic_Problem):
    def __init__(self):
        self.n_obj = 2  # 初始化（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-2] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([2] * self.n_var)
        self.ub[0] = 1
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        hy = np.abs(y) / (1 + np.exp(2 * (np.abs(y))))
        hy1 = hy[:, J1]
        hy2 = hy[:, J2]
        f1 = x1 + 2 * np.mean(hy1, 1, keepdims = True)
        f2 = 1 - x1 ** 2 + 2 * np.mean(hy2, 1, keepdims = True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 2
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF5(Basic_Problem):
    def __init__(self):
        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        hy = 2 * y ** 2 - np.cos(4 * np.pi * y) + 1
        # print(hy)
        hy1 = hy[:, J1]
        hy2 = hy[:, J2]
        f1 = x1 + (1 / 20 + 0.1) * np.abs(np.sin(20 * np.pi * x1)) + 2 * (np.mean(hy1, 1, keepdims = True))
        f2 = 1 - x1 + (1 / 20 + 0.1) * np.abs(np.sin(20 * np.pi * x1)) + 2 * (np.mean(hy2, 1, keepdims = True))
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF6(Basic_Problem):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        # hy    = 2*y**2 - np.cos(4*np.pi*y) + 1
        # print(hy)
        # hy1   = hy[:, J1]
        # hy2   = hy[:, J2]
        f1 = x1 + np.maximum(0, 2 * (1 / 4 + 0.1) * np.sin(4 * np.pi * x1)) + \
             (2 / len(J1)) * (4 * np.sum(yJ1 ** 2, 1, keepdims = True) - \
                              2 * (np.prod(np.cos((20 * yJ1 * np.pi) / (np.sqrt(J1))), 1, keepdims = True)) + 2)
        f2 = 1 - x1 + np.maximum(0, 2 * (1 / 4 + 0.1) * np.sin(4 * np.pi * x1)) + \
             (2 / len(J2)) * (4 * np.sum(yJ2 ** 2, 1, keepdims = True) - \
                              2 * (np.prod(np.cos((20 * yJ2 * np.pi) / (np.sqrt(J2))), 1, keepdims = True)) + 2)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        idx = ((ObjV1 > 0) & (ObjV1 < 1 / 4)) | ((ObjV1 > 1 / 2) & (ObjV1 < 3 / 4))
        ObjV1 = ObjV1[~idx]
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF7(Basic_Problem):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 2  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1 ** 0.2 + 2 * np.mean(yJ1 ** 2, 1, keepdims = True)
        f2 = 1 - x1 ** 0.2 + 2 * np.mean(yJ2 ** 2, 1, keepdims = True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF8(Basic_Problem):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([0] * 2 + [-2] * (self.n_var - 2))
        self.ub = np.array([1] * 2 + [2] * (self.n_var - 2))
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = np.array(list(range(3, self.n_var, 3)))
        J2 = np.array(list(range(4, self.n_var, 3)))
        J3 = np.array(list(range(2, self.n_var, 3)))
        # print(J1, J2, J3)
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        # f    = 2*np.mean((Vars-2*x2*np.sin(2*np.pi*x1+J*np.pi/self.Dim))**2 ,1,keepdims = True)
        f = (Vars - 2 * x2 * np.sin(2 * np.pi * x1 + J * np.pi / self.n_var)) ** 2
        # print(f.shape)
        f1 = np.cos(0.5 * x1 * np.pi) * np.cos(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J1], 1, keepdims = True)
        f2 = np.cos(0.5 * x1 * np.pi) * np.sin(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J2], 1, keepdims = True)
        f3 = np.sin(0.5 * x1 * np.pi) + 2 * np.mean(f[:, J3], 1, keepdims = True)
        ObjV = np.hstack([f1, f2, f3])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points
        ObjV, N = crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = ObjV / np.sqrt(np.sum(ObjV ** 2, 1, keepdims = True))
        referenceObjV = ObjV
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF9(Basic_Problem):  # 继承Problem父类
    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([0] * 2 + [-2] * (self.n_var - 2))
        self.ub = np.array([1] * 2 + [2] * (self.n_var - 2))
        # 调用父类构造方法完成实例化
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = np.array(list(range(3, self.n_var, 3)))
        J2 = np.array(list(range(4, self.n_var, 3)))
        J3 = np.array(list(range(2, self.n_var, 3)))
        # print(J1, J2, J3)
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        f = (Vars - 2 * x2 * np.sin(2 * np.pi * x1 + J * np.pi / self.n_var)) ** 2
        f1 = 0.5 * (np.maximum(0, (1.1 * (1 - 4 * (2 * x1 - 1) ** 2))) + 2 * x1) * x2 + 2 * np.mean(f[:, J1], 1,
                                                                                                    keepdims = True)
        f2 = 0.5 * (np.maximum(0, (1.1 * (1 - 4 * (2 * x1 - 1) ** 2))) - 2 * x1 + 2) * x2 + 2 * np.mean(f[:, J2], 1,
                                                                                                        keepdims = True)
        f3 = 1 - x2 + 2 * np.mean(f[:, J3], 1, keepdims = True)
        ObjV = np.hstack([f1, f2, f3])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV, N = crtup(self.n_obj, N)  # ObjV.shape=N,3
        idx = (ObjV[:, 0] > (1 - ObjV[:, 2]) / 4) & (ObjV[:, 0] < (1 - ObjV[:, 2]) * 3 / 4)
        referenceObjV = ObjV[~idx]
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF10(Basic_Problem):  # 继承Problem父类

    def __init__(self):
        self.n_obj = 3  # 初始化M（目标维数）
        self.n_var = 30  # 初始化Dim（决策变量维数）
        self.lb = np.array([0] * 2 + [-2] * (self.n_var - 2))
        self.ub = np.array([1] * 2 + [2] * (self.n_var - 2))
        self.vtype = float

    def func(self, x):  # 目标函数
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  # 决策变量矩阵
        x1 = Vars[:, [0]]
        x2 = Vars[:, [1]]
        J1 = np.array(list(range(3, self.n_var, 3)))
        J2 = np.array(list(range(4, self.n_var, 3)))
        J3 = np.array(list(range(2, self.n_var, 3)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - 2 * x2 * np.sin(2 * np.pi * x1 + (J * np.pi) / self.n_var)
        f = 4 * y ** 2 - np.cos(8 * np.pi * y) + 1
        f1 = np.cos(0.5 * x1 * np.pi) * np.cos(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J1], 1, keepdims = True)
        f2 = np.cos(0.5 * x1 * np.pi) * np.sin(0.5 * x2 * np.pi) + 2 * np.mean(f[:, J2], 1, keepdims = True)
        f3 = np.sin(0.5 * x1 * np.pi) + 2 * np.mean(f[:, J3], 1, keepdims = True)
        ObjV = np.hstack([f1, f2, f3])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):  # 理论最优值
        N = n_ref_points  # 生成10000个参考点
        ObjV, N = crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = ObjV / np.sqrt(np.sum(ObjV ** 2, 1, keepdims = True))
        referenceObjV = ObjV
        return referenceObjV

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


if __name__ == '__main__':
    uf1 = UF1()
    uf2 = UF2()
    uf3 = UF3()
    uf4 = UF4()
    uf5 = UF5()
    uf6 = UF6()
    uf7 = UF7()
    uf8 = UF8()
    uf9 = UF9()
    uf10 = UF10()
    x = np.ones((30,))
    print(uf1.func(x))
    print(uf2.func(x))
    print(uf3.func(x))
    print(uf4.func(x))
    print(uf5.func(x))
    print(uf6.func(x))
    print(uf7.func(x))
    print(uf8.func(x))
    print(uf9.func(x))
    print(uf10.func(x))
    s1 = uf1.get_ref_set()
    s2 = uf2.get_ref_set()
    s3 = uf3.get_ref_set()
    s4 = uf4.get_ref_set()
    s5 = uf5.get_ref_set()
    s6 = uf6.get_ref_set()
    s7 = uf7.get_ref_set()
    s8 = uf8.get_ref_set()
    s9 = uf9.get_ref_set()
    s10 = uf10.get_ref_set()
