
import numpy as np
from ....problem.basic_problem import Basic_Problem


def find_non_dominated_indices(Point):
    """
    此函数用于找出种群中的支配解
    :param population_list: 种群的目标值的列表，列表中的每个元素是一个代表单个解目标值的列表
    :return: 支配解的列表
    """
    # 将列表转换为 numpy 数组
    n_points = Point.shape[0]
    is_dominated = np.zeros(n_points, dtype = bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # 检查是否存在解 j 支配解 i
                if np.all(Point[j] <= Point[i]) and np.any(Point[j] < Point[i]):
                    is_dominated[i] = True
                    break

    # 找出非支配解的索引
    non_dominated_indices = np.where(~is_dominated)[0]
    return non_dominated_indices


class ZDT(Basic_Problem):

    def __init__(self, n_var = 30, **kwargs):
        self.n_var = n_var
        self.n_obj = 2
        self.vtype = float
        self.lb = np.zeros(n_var)
        self.ub = np.ones(n_var)

    def __str__(self):
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class ZDT1(ZDT):

    def func(self, x, *args, **kwargs):
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis = 1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points  #
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT2(ZDT):

    def func(self, x, *args, **kwargs):
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis = 1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 2
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT3(ZDT):

    def func(self, x, *args, **kwargs):
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis = 1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 0.5 - ObjV1 * np.sin(10 * np.pi * ObjV1)
        f = np.array([ObjV1, ObjV2]).T
        index = find_non_dominated_indices(f)
        referenceObjV = f[index]
        # levels, criLevel = ea.ndsortESS(f, None, 1)
        # referenceObjV = f[np.where(levels == 1)[0]]
        return referenceObjV


class ZDT4(ZDT):
    def __init__(self, n_var = 10):
        super().__init__(n_var)
        self.lb = -5 * np.ones(self.n_var)
        self.lb[0] = 0.0
        self.ub = 5 * np.ones(self.n_var)
        self.ub[0] = 1.0
        # self.func = self._evaluate

    def func(self, x, *args, **kwargs):
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * np.cos(4.0 * np.pi * x[:, i])
        h = 1.0 - np.sqrt(f1 / g)
        f2 = g * h

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT5(ZDT):

    def __init__(self, m = 11, n = 5, normalize = True, **kwargs):
        self.m = m
        self.n = n
        self.normalize = normalize
        super().__init__(n_var = (30 + n * (m - 1)), **kwargs)

    def func(self, x, *args, **kwargs):
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        x = x.astype(float)

        _x = [x[:, :30]]
        for i in range(self.m - 1):
            _x.append(x[:, 30 + i * self.n: 30 + (i + 1) * self.n])

        u = np.column_stack([x_i.sum(axis = 1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[:, 1:].sum(axis = 1)

        f1 = 1 + u[:, 0]
        f2 = g * (1 / f1)

        if self.normalize:
            f1 = normalize(f1, 1, 31)
            f2 = normalize(f2, (self.m - 1) * 1 / 31, (self.m - 1))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):
        x = 1 + np.linspace(0, 1, n_ref_points) * 30
        pf = np.column_stack([x, (self.m - 1) / x])
        if self.normalize:
            pf = normalize(pf)
        return pf


class ZDT6(ZDT):

    def __init__(self, n_var = 10, **kwargs):
        super().__init__(n_var = n_var, **kwargs)

    def func(self, x, *args, **kwargs):
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = 1 - np.exp(-4 * x[:, 0]) * np.power(np.sin(6 * np.pi * x[:, 0]), 6)
        g = 1 + 9.0 * np.power(np.sum(x[:, 1:], axis = 1) / (self.n_var - 1.0), 0.25)
        f2 = g * (1 - np.power(f1 / g, 2))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = np.linspace(0.280775, 1, N)
        ObjV2 = 1 - ObjV1 ** 2;
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZeroToOneNormalization():

    def __init__(self, lb = None, ub = None) -> None:

        # if both are None we are basically done because normalization is disabled
        if lb is None and ub is None:
            self.lb, self.ub = None, None
            return

        # if not set simply fall back no nan values
        if lb is None:
            lb = np.full_like(ub, np.nan)
        if ub is None:
            ub = np.full_like(lb, np.nan)

        lb, ub = np.copy(lb).astype(float), np.copy(ub).astype(float)

        # if both are equal then set the upper bound to none (always the 0 or lower bound will be returned then)
        ub[lb == ub] = np.nan

        # store the lower and upper bounds
        self.lb, self.ub = lb, ub

        # check out when the input values are nan
        lb_nan, ub_nan = np.isnan(lb), np.isnan(ub)

        # now create all the masks that are necessary
        self.lb_only, self.ub_only = np.logical_and(~lb_nan, ub_nan), np.logical_and(lb_nan, ~ub_nan)
        self.both_nan = np.logical_and(np.isnan(lb), np.isnan(ub))
        self.neither_nan = ~self.both_nan

        # if neither is nan than ub must be greater or equal than lb
        any_nan = np.logical_or(np.isnan(lb), np.isnan(ub))
        assert np.all(np.logical_or(ub >= lb, any_nan)), "lb must be less or equal than ub."

    def forward(self, X):
        if X is None or (self.lb is None and self.ub is None):
            return X

        lb, ub, lb_only, ub_only = self.lb, self.ub, self.lb_only, self.ub_only
        both_nan, neither_nan = self.both_nan, self.neither_nan

        # simple copy the input
        N = np.copy(X)

        # normalize between zero and one if neither of them is nan
        N[..., neither_nan] = (X[..., neither_nan] - lb[neither_nan]) / (ub[neither_nan] - lb[neither_nan])

        N[..., lb_only] = X[..., lb_only] - lb[lb_only]

        N[..., ub_only] = 1.0 - (ub[ub_only] - X[..., ub_only])

        return N


def normalize(X, lb = None, ub = None, return_bounds = False, estimate_bounds_if_none = True):
    if estimate_bounds_if_none:
        if lb is None:
            lb = np.min(X, axis = 0)
        if ub is None:
            ub = np.max(X, axis = 0)

    if isinstance(lb, float) or isinstance(lb, int):
        lb = np.full(X.shape[-1], lb)

    if isinstance(ub, float) or isinstance(ub, int):
        ub = np.full(X.shape[-1], ub)

    norm = ZeroToOneNormalization(lb, ub)
    X = norm.forward(X)

    if not return_bounds:
        return X
    else:
        return X, norm.lb, norm.ub


if __name__ == '__main__':
    x1 = np.random.rand(30)
    zdt1 = ZDT1()
    zdt2 = ZDT2()
    zdt3 = ZDT3()
    print(zdt1.eval(x1))
    print(zdt2.eval(x1))
    print(zdt3.eval(x1))
    x2 = np.random.rand(10)
    zdt4 = ZDT4()
    zdt6 = ZDT6()
    print(zdt4.eval(x2))
    print(zdt6.eval(x2))
    x3 = np.random.rand(45)
    zdt5 = ZDT5()
    print(zdt5.eval(x3))
    s1 = zdt1.get_ref_set()
    s2 = zdt2.get_ref_set()
    s3 = zdt3.get_ref_set()
    s4 = zdt4.get_ref_set()
    s5 = zdt5.get_ref_set()
    s6 = zdt6.get_ref_set()

