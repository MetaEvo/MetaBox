import torch as th
import math
from ....problem.basic_problem import Basic_Problem_Torch
import itertools
import numpy as np
from scipy.special import comb


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


def find_non_dominated_indices(Point):
    """
    此函数用于找出种群中的支配解
    :param population_list: 种群的目标值的列表，列表中的每个元素是一个代表单个解目标值的列表
    :return: 支配解的列表
    """
    # 将列表转换为 numpy 数组
    n_points = Point.shape[0]
    is_dominated = th.zeros(n_points, dtype = bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # 检查是否存在解 j 支配解 i
                if th.all(Point[j] <= Point[i]) and th.any(Point[j] < Point[i]):
                    is_dominated[i] = True
                    break

    # 找出非支配解的索引
    non_dominated_indices = th.where(~is_dominated)[0]
    return non_dominated_indices


class WFG_Torch(Basic_Problem_Torch):

    def __init__(self, n_var, n_obj, k = None, l = None, **kwargs):
        self.n_obj = n_obj
        self.n_var = n_var
        self.lb = th.zeros(n_var)
        self.ub = 2 * th.arange(1, n_var + 1).to(th.float32)
        self.vtype = th.float32
        self.S = th.arange(2, 2 * self.n_obj + 1, 2).to(th.float32)
        self.A = th.ones(self.n_obj - 1)

        if k:
            self.k = k
        else:
            if n_obj == 2:
                self.k = 4
            else:
                self.k = 2 * (n_obj - 1)

        if l:
            self.l = l
        else:
            self.l = n_var - self.k

        self.validate(self.l, self.k, self.n_obj)

    def validate(self, l, k, n_obj):
        if n_obj < 2:
            raise ValueError('WFG problems must have two or more objectives.')
        if not k % (n_obj - 1) == 0:
            raise ValueError('Position parameter (k) must be divisible by number of objectives minus one.')
        if k < 4:
            raise ValueError('Position parameter (k) must be greater or equal than 4.')
        if (k + l) < n_obj:
            raise ValueError('Sum of distance and position parameters must be greater than num. of objs. (k + l >= M).')

    def _post(self, t, a):
        x = []
        for i in range(t.shape[1] - 1):
            x.append(th.maximum(t[:, -1], a[i]) * (t[:, i] - 0.5) + 0.5)
        x.append(t[:, -1])
        return th.column_stack(x)

    def _calculate(self, x, s, h):
        return x[:, -1][:, None] + s * th.column_stack(h)

    def _rand_optimal_position(self, n):
        return th.random.random((n, self.k))

    def _positional_to_optimal(self, K):
        suffix = th.full((len(K), self.l), 0.35)
        X = th.column_stack([K, suffix])
        return X * self.ub

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class WFG1_Torch(WFG_Torch):

    @staticmethod
    def t1(x, n, k):
        x[:, k:n] = _transformation_shift_linear(x[:, k:n], 0.35)
        return x

    @staticmethod
    def t2(x, n, k):
        x[:, k:n] = _transformation_bias_flat(x[:, k:n], 0.8, 0.75, 0.85)
        return x

    @staticmethod
    def t3(x, n):
        x[:, :n] = _transformation_bias_poly(x[:, :n], 0.02)
        return x

    @staticmethod
    def t4(x, m, n, k):
        w = th.arange(2, 2 * n + 1, 2).to(th.float32)
        gap = k // (m - 1)
        t = []
        for m in range(1, m):
            _y = x[:, (m - 1) * gap: (m * gap)]
            _w = w[(m - 1) * gap: (m * gap)]
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(x[:, k:n], w[k:n]))
        return th.column_stack(t)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y = WFG1_Torch.t1(y, self.n_var, self.k)
        y = WFG1_Torch.t2(y, self.n_var, self.k)
        y = WFG1_Torch.t3(y, self.n_var)
        y = WFG1_Torch.t4(y, self.n_obj, self.n_var, self.k)

        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_mixed(y[:, 0], alpha = 1.0, A = 5.0))

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        M = self.n_obj
        c = th.ones((num, M))
        for i in range(num):
            for j in range(1, M):
                temp = Point[i, j] / Point[i, 0] * th.prod(1 - c[i, M - j: M - 1])
                c[i, M - j - 1] = (temp ** 2 - temp + th.sqrt(2 * temp)) / (temp ** 2 + 1)
        x = th.arccos(c) * 2 / math.pi
        temp = (1 - th.sin(math.pi / 2 * x[:, [1]])) * Point[:, [M - 1]] / Point[:, [M - 2]]
        a = th.linspace(0, 1, 10000 + 1)
        for i in range(num):
            E = th.abs(
                temp[i] * (1 - th.cos(math.pi / 2 * a)) - 1 + a + th.cos(10 * math.pi * a + math.pi / 2) / 10 / math.pi)
            rank = th.argsort(E)
            x[i, 0] = a[th.min(rank[0: 10])]
        Point = convex(x)
        Point[:, [M - 1]] = mixed(x)
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (num, 1)) * Point
        return referenceObjV


class WFG2_Torch(WFG_Torch):

    def validate(self, l, k, n_obj):
        super().validate(l, k, n_obj)
        validate_wfg2_wfg3(l)

    @staticmethod
    def t2(x, n, k):
        y = [x[:, i] for i in range(k)]

        l = n - k
        ind_non_sep = k + l // 2

        i = k + 1
        while i <= ind_non_sep:
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y.append(_reduction_non_sep(x[:, head:tail], 2))
            i += 1

        return th.column_stack(y)

    @staticmethod
    def t3(x, m, n, k):
        ind_r_sum = k + (n - k) // 2
        gap = k // (m - 1)

        t = [_reduction_weighted_sum_uniform(x[:, (m - 1) * gap: (m * gap)]) for m in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:ind_r_sum]))

        return th.column_stack(t)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y = WFG1_Torch.t1(y, self.n_var, self.k)
        y = WFG2_Torch.t2(y, self.n_var, self.k)
        y = WFG2_Torch.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_disconnected(y[:, 0], alpha = 1.0, beta = 1.0, A = 5.0))

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        M = self.n_obj
        c = th.ones((num, M))
        for i in range(num):
            for j in range(1, M):
                temp = Point[i, j] / Point[i, 0] * th.prod(1 - c[i, M - j: M - 1])
                c[i, M - j - 1] = (temp ** 2 - temp + th.sqrt(2 * temp)) / (temp ** 2 + 1)
        x = th.arccos(c) * 2 / math.pi
        temp = (1 - th.sin(math.pi / 2 * x[:, [1]])) * Point[:, [M - 1]] / Point[:, [M - 2]]
        a = th.linspace(0, 1, 10000 + 1)
        for i in range(num):
            E = th.abs(temp[i] * (1 - th.cos(math.pi / 2 * a)) - 1 + a * th.cos(5 * math.pi * a) ** 2)
            rank = th.argsort(E)
            x[i, 0] = a[th.min(rank[0: 10])]
        Point = convex(x)
        Point[:, [M - 1]] = disc(x)
        # [levels, criLevel] = ea.ndsortESS(Point.numpy(), None, 1)  # 非支配分层，只分出第一层即可
        # levels = th.tensor(levels)
        # Point = Point[th.where(levels == 1)[0], :]  # 只保留点集中的非支配点
        index = find_non_dominated_indices(Point)
        Point = Point[index]
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG3_Torch(WFG_Torch):

    def __init__(self, n_var, n_obj, k = None, **kwargs):
        super().__init__(n_var, n_obj, k = k, **kwargs)
        self.A[1:] = 0

    def validate(self, l, k, n_obj):
        super().validate(l, k, n_obj)
        validate_wfg2_wfg3(l)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y = WFG1_Torch.t1(y, self.n_var, self.k)
        y = WFG2_Torch.t2(y, self.n_var, self.k)
        y = WFG2_Torch.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_linear(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)

        return out

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        X = th.hstack([th.linspace(0, 1, N).unsqueeze(1), th.zeros((N, self.n_obj - 2)) + 0.5, th.zeros((N, 1))])
        Point = linear(X)
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG4_Torch(WFG_Torch):

    @staticmethod
    def t1(x):
        return _transformation_shift_multi_modal(x, 30.0, 10.0, 0.35)

    @staticmethod
    def t2(x, m, k):
        gap = k // (m - 1)
        t = [_reduction_weighted_sum_uniform(x[:, (m - 1) * gap: (m * gap)]) for m in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:]))
        return th.column_stack(t)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y = WFG4_Torch.t1(y)
        y = WFG4_Torch.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        Point = Point / th.tile(th.sqrt(th.sum(Point ** 2, dim = 1, keepdim = True)), (1, self.n_obj))

        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG5_Torch(WFG_Torch):

    @staticmethod
    def t1(x):
        return _transformation_param_deceptive(x, A = 0.35, B = 0.001, C = 0.05)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y = WFG5_Torch.t1(y)
        y = WFG4_Torch.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        Point = Point / th.tile(th.sqrt(th.sum(Point ** 2, dim = 1, keepdim = True)), (1, self.n_obj))
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG6_Torch(WFG_Torch):

    @staticmethod
    def t2(x, m, n, k):
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return th.column_stack(t)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y = WFG1_Torch.t1(y, self.n_var, self.k)
        y = WFG6_Torch.t2(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        Point = Point / th.tile(th.sqrt(th.sum(Point ** 2, dim = 1, keepdim = True)), (1, self.n_obj))
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG7_Torch(WFG_Torch):

    @staticmethod
    def t1(x, k):
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1:])
            x[:, i] = _transformation_param_dependent(x[:, i], aux)
        return x

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y = WFG7_Torch.t1(y, self.k)
        y = WFG1_Torch.t1(y, self.n_var, self.k)
        y = WFG4_Torch.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        Point = Point / th.tile(th.sqrt(th.sum(Point ** 2, dim = 1, keepdim = True)), (1, self.n_obj))
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG8_Torch(WFG_Torch):

    @staticmethod
    def t1(x, n, k):
        ret = []
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:, :i])
            ret.append(_transformation_param_dependent(x[:, i], aux, A = 0.98 / 49.98, B = 0.02, C = 50.0))
        return th.column_stack(ret)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y[:, self.k:self.n_var] = WFG8_Torch.t1(y, self.n_var, self.k)
        y = WFG1_Torch.t1(y, self.n_var, self.k)
        y = WFG4_Torch.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def _positional_to_optimal(self, K):
        k, l = self.k, self.l

        for i in range(k, k + l):
            u = K.sum(axis = 1) / K.shape[1]
            tmp1 = th.abs(th.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0 * u) * tmp1)
            suffix = th.pow(0.35, th.pow(tmp2, -1.0))

            K = th.column_stack([K, suffix[:, None]])

        ret = K * (2 * (th.arange(self.n_var) + 1))
        return ret

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        Point = Point / th.tile(th.sqrt(th.sum(Point ** 2, dim = 1, keepdim = True)), (1, self.n_obj))
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG9_Torch(WFG_Torch):

    @staticmethod
    def t1(x, n):
        ret = []
        for i in range(0, n - 1):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1:])
            ret.append(_transformation_param_dependent(x[:, i], aux))
        return th.column_stack(ret)

    @staticmethod
    def t2(x, n, k):
        a = [_transformation_shift_deceptive(x[:, i], 0.35, 0.001, 0.05) for i in range(k)]
        b = [_transformation_shift_multi_modal(x[:, i], 30.0, 95.0, 0.35) for i in range(k, n)]
        return th.column_stack(a + b)

    @staticmethod
    def t3(x, m, n, k):
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return th.column_stack(t)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        y = x / self.ub
        y[:, :self.n_var - 1] = WFG9_Torch.t1(y, self.n_var)
        y = WFG9_Torch.t2(y, self.n_var, self.k)
        y = WFG9_Torch.t3(y, self.n_obj, self.n_var, self.k)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def _positional_to_optimal(self, K):
        k, l = self.k, self.l

        suffix = th.full((len(K), self.l), 0.0)
        X = th.column_stack([K, suffix])
        X[:, self.k + self.l - 1] = 0.35

        for i in range(self.k + self.l - 2, self.k - 1, -1):
            m = X[:, i + 1:k + l]
            val = m.sum(axis = 1) / m.shape[1]
            X[:, i] = 0.35 ** ((0.02 + 1.96 * val) ** -1)

        ret = X * (2 * (th.arange(self.n_var) + 1))
        return ret

    def get_ref_set(self, n_ref_points = 1000):
        N = n_ref_points  # 设置所要生成的全局最优解的个数
        Point, num = crtup(self.n_obj, N)  # 生成N个在各目标的单位维度上均匀分布的参考点
        Point = th.tensor(Point)
        Point = Point / th.tile(th.sqrt(th.sum(Point ** 2, dim = 1, keepdim = True)), (1, self.n_obj))
        referenceObjV = th.tile(th.tensor([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


## ---------------------------------------------------------------------------------------------------------
# tool for get reference point
# ---------------------------------------------------------------------------------------------------------

def convex(x):
    return th.fliplr(
        th.cumprod(th.hstack([th.ones((x.shape[0], 1)), 1 - th.cos(x[:, :-1] * math.pi / 2)]), 1)) * th.hstack(
        [th.ones((x.shape[0], 1)), 1 - th.sin(x[:, list(range(x.shape[1] - 1 - 1, -1, -1))] * math.pi / 2)])


def mixed(x):
    return 1 - x[:, [0]] - th.cos(10 * math.pi * x[:, [0]] + math.pi / 2) / 10 / math.pi


def linear(x):
    return th.fliplr(th.cumprod(th.hstack([th.ones((x.shape[0], 1)), x[:, :-1]]), 1)) * th.hstack([th.ones((x.shape[0], 1)), 1 - x[:, list(range(x.shape[1] - 1 - 1, -1, -1))]])


def s_linear(x, A):
    return th.abs(x - A) / th.abs(th.floor(A - x) + A)


def b_flat(x, A, B, C):
    Output = A + th.min([0 * th.floor(x - B), th.floor(x - B)], 0) * A * (B - x) / B - th.min(
        [0 * th.floor(C - x), th.floor(C - x)], 0) * (1 - A) * (x - C) / (1 - C)
    return th.round(Output, 6)


def b_poly(x, a):
    return th.sign(x) * th.abs(x) ** a


def r_sum(x, w):
    Output = th.sum(x * th.tile(w, (x.shape[0], 1)), 1) / th.sum(w)
    return Output


def disc(x):
    return 1 - x[:, [0]] * (th.cos(5 * math.pi * x[:, [0]])) ** 2


# ---------------------------------------------------------------------------------------------------------
# TRANSFORMATIONS
# ---------------------------------------------------------------------------------------------------------


def _transformation_shift_linear(value, shift = 0.35):
    return correct_to_01(th.abs(value - shift) / th.abs(th.floor(shift - value) + shift))


def _transformation_shift_deceptive(y, A = 0.35, B = 0.005, C = 0.05):
    tmp1 = th.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = th.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (th.abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


def _transformation_shift_multi_modal(y, A, B, C):
    tmp1 = th.abs(y - C) / (2.0 * (th.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * math.pi * (0.5 - tmp1)
    ret = (1.0 + th.cos(tmp2) + 4.0 * B * th.pow(tmp1, 2.0)) / (B + 2.0)
    return correct_to_01(ret)


def _transformation_bias_flat(y, a, b, c):
    ret = a + th.minimum(th.tensor(0), th.floor(y - b)) * (a * (b - y) / b) \
          - th.minimum(th.tensor(0), th.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
    return correct_to_01(ret)


def _transformation_bias_poly(y, alpha):
    return correct_to_01(y ** alpha)


def _transformation_param_dependent(y, y_deg, A = 0.98 / 49.98, B = 0.02, C = 50.0):
    aux = A - (1.0 - 2.0 * y_deg) * th.abs(th.floor(0.5 - y_deg) + A)
    ret = th.pow(y, B + (C - B) * aux)
    return correct_to_01(ret)


def _transformation_param_deceptive(y, A = 0.35, B = 0.001, C = 0.05):
    tmp1 = th.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = th.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (th.abs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


# ---------------------------------------------------------------------------------------------------------
# REDUCTION
# ---------------------------------------------------------------------------------------------------------


def _reduction_weighted_sum(y, w):
    return correct_to_01(th.matmul(y, w) / w.sum())


def _reduction_weighted_sum_uniform(y):
    return correct_to_01(y.mean(axis = 1))


def _reduction_non_sep(y, A):
    n, m = y.shape
    val = th.ceil(th.tensor(A) / 2.0)

    num = th.zeros(n)
    for j in range(m):
        num += y[:, j]
        for k in range(A - 1):
            num += th.abs(y[:, j] - y[:, (1 + j + k) % m])

    denom = m * val * (1.0 + 2.0 * A - 2 * val) / A

    return correct_to_01(num / denom)


# ---------------------------------------------------------------------------------------------------------
# SHAPE
# ---------------------------------------------------------------------------------------------------------


def _shape_concave(x, m):
    M = x.shape[1]
    if m == 1:
        ret = th.prod(th.sin(0.5 * x[:, :M] * math.pi), axis = 1)
    elif 1 < m <= M:
        ret = th.prod(th.sin(0.5 * x[:, :M - m + 1] * math.pi), axis = 1)
        ret *= th.cos(0.5 * x[:, M - m + 1] * math.pi)
    else:
        ret = th.cos(0.5 * x[:, 0] * math.pi)
    return correct_to_01(ret)


def _shape_convex(x, m):
    M = x.shape[1]
    if m == 1:
        ret = th.prod(1.0 - th.cos(0.5 * x[:, :M] * math.pi), axis = 1)
    elif 1 < m <= M:
        ret = th.prod(1.0 - th.cos(0.5 * x[:, :M - m + 1] * math.pi), axis = 1)
        ret *= 1.0 - th.sin(0.5 * x[:, M - m + 1] * math.pi)
    else:
        ret = 1.0 - th.sin(0.5 * x[:, 0] * math.pi)
    return correct_to_01(ret)


def _shape_linear(x, m):
    M = x.shape[1]
    if m == 1:
        ret = th.prod(x, axis = 1)
    elif 1 < m <= M:
        ret = th.prod(x[:, :M - m + 1], axis = 1)
        ret *= 1.0 - x[:, M - m + 1]
    else:
        ret = 1.0 - x[:, 0]
    return correct_to_01(ret)


def _shape_mixed(x, A = 5.0, alpha = 1.0):
    aux = 2.0 * A * math.pi
    ret = th.pow(1.0 - x - (th.cos(aux * x + 0.5 * math.pi) / aux), alpha)
    return correct_to_01(ret)


def _shape_disconnected(x, alpha = 1.0, beta = 1.0, A = 5.0):
    aux = th.cos(A * math.pi * x ** beta)
    return correct_to_01(1.0 - x ** alpha * aux ** 2)


# ---------------------------------------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------------------------------------

def validate_wfg2_wfg3(l):
    if not l % 2 == 0:
        raise ValueError('In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.')


def correct_to_01(X, epsilon = 1.0e-10):
    X[th.logical_and(X < 0, X >= 0 - epsilon)] = 0
    X[th.logical_and(X > 1, X <= 1 + epsilon)] = 1
    return X


if __name__ == '__main__':
    wfg1 = WFG1_Torch(10, 3)
    wfg2 = WFG2_Torch(10, 3)
    wfg3 = WFG3_Torch(10, 3)
    wfg4 = WFG4_Torch(10, 3)
    wfg5 = WFG5_Torch(10, 3)
    wfg6 = WFG6_Torch(10, 3)
    wfg7 = WFG7_Torch(10, 3)
    wfg8 = WFG8_Torch(10, 3)
    wfg9 = WFG9_Torch(10, 3)
    x = th.ones(10)
    print(wfg1.func(x))
    print(wfg2.func(x))
    print(wfg3.func(x))
    print(wfg4.func(x))
    print(wfg5.func(x))
    print(wfg6.func(x))
    print(wfg7.func(x))
    print(wfg8.func(x))
    print(wfg9.func(x))

    s1 = wfg1.get_ref_set()
    s2 = wfg2.get_ref_set()
    s3 = wfg3.get_ref_set()
    s4 = wfg4.get_ref_set()
    s5 = wfg5.get_ref_set()
    s6 = wfg6.get_ref_set()
    s7 = wfg7.get_ref_set()
    s8 = wfg8.get_ref_set()
    s9 = wfg9.get_ref_set()


