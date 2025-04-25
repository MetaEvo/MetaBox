
import torch as th
from ....problem.basic_problem import Basic_Problem_Torch
import math
def find_non_dominated_indices(Point):
    """
    此函数用于找出种群中的支配解
    :param population_list: 种群的目标值的列表，列表中的每个元素是一个代表单个解目标值的列表
    :return: 支配解的列表
    """
    # 将列表转换为 numpy 数组
    n_points = Point.shape[0]
    is_dominated = th.zeros(n_points, dtype=bool)

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

class ZDT_Torch(Basic_Problem_Torch):

    def __init__(self, n_var=30, **kwargs):
        self.n_var = n_var
        self.n_obj = 2
        self.vtype = float
        self.lb = th.zeros(n_var)
        self.ub = th.ones(n_var)
    def __str__(self):
        return  self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class ZDT1_Torch(ZDT_Torch):


    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * th.sum(x[:, 1:], axis=1)
        f2 = g * (1 - th.pow((f1 / g), 0.5))

        out = th.column_stack([f1, f2])
        return out

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points #
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - th.sqrt(ObjV1)
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV

class ZDT2_Torch(ZDT_Torch):


    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f1 = x[:, 0]
        c = th.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - th.pow((f1 * 1.0 / g), 2))

        out = th.column_stack([f1, f2])
        return out

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 2
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT3_Torch(ZDT_Torch):

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f1 = x[:, 0]
        c = th.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - th.pow(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * th.sin(10 * math.pi * f1))

        out = th.column_stack([f1, f2])
        return out

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 0.5 - ObjV1 * th.sin(10 * math.pi * ObjV1)
        f = th.stack([ObjV1, ObjV2]).T
        index = find_non_dominated_indices(f)
        referenceObjV = f[index,:]
        # levels, criLevel = ea.ndsortESS(f.numpy(), None, 1)
        # levels = th.tensor(levels)
        # referenceObjV = f[th.where(levels == 1)[0]]
        return referenceObjV


class ZDT4_Torch(ZDT_Torch):
    def __init__(self, n_var=10):
        super().__init__(n_var)
        self.lb = -5 * th.ones(self.n_var)
        self.lb[0] = 0.0
        self.ub = 5 * th.ones(self.n_var)
        self.ub[0] = 1.0
        # self.func = self._evaluate


    def func(self, x,*args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f1 = x[:, 0]
        g = 1.0
        g += 10 * (self.n_var - 1)
        for i in range(1, self.n_var):
            g += x[:, i] * x[:, i] - 10.0 * th.cos(4.0 * math.pi * x[:, i])
        h = 1.0 - th.sqrt(f1 / g)
        f2 = g * h

        out = th.column_stack([f1, f2])
        return out

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = th.linspace(0, 1, N)
        ObjV2 = 1 - th.sqrt(ObjV1)
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT5_Torch(ZDT_Torch):

    def __init__(self, m=11, n=5, normalize=True, **kwargs):
        self.m = m
        self.n = n
        self.normalize = normalize
        super().__init__(n_var=(30 + n * (m - 1)), **kwargs)



    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(th.float32)

        _x = [x[:, :30]]
        for i in range(self.m - 1):
            _x.append(x[:, 30 + i * self.n: 30 + (i + 1) * self.n])

        u = th.column_stack([x_i.sum(axis=1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[:, 1:].sum(axis=1)

        f1 = 1 + u[:, 0]
        f2 = g * (1 / f1)

        if self.normalize:
            f1 = normalize(f1, 1, 31)
            f2 = normalize(f2, (self.m-1) * 1/31, (self.m-1))

        out = th.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points=1000):
        x = 1 + th.linspace(0, 1, n_ref_points) * 30
        pf = th.column_stack([x, (self.m - 1) / x])
        if self.normalize:
            pf = normalize(pf)
        return pf

class ZDT6_Torch(ZDT_Torch):

    def __init__(self, n_var=10, **kwargs):
        super().__init__(n_var=n_var, **kwargs)


    def func(self, x,  *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f1 = 1 - th.exp(-4 * x[:, 0]) * th.pow(th.sin(6 * math.pi * x[:, 0]), 6)
        g = 1 + 9.0 * th.pow(th.sum(x[:, 1:], axis=1) / (self.n_var - 1.0), 0.25)
        f2 = g * (1 - th.pow(f1 / g, 2))

        out= th.column_stack([f1, f2])
        return out

    def get_ref_set(self,n_ref_points=1000):  # 设定目标数参考值（本问题目标函数参考值设定为理论最优值，即“真实帕累托前沿点”）
        N = n_ref_points
        ObjV1 = th.linspace(0.280775, 1, N)
        ObjV2 = 1 - ObjV1 ** 2;
        referenceObjV = th.stack([ObjV1, ObjV2]).T
        return referenceObjV

class ZeroToOneNormalization():

    def __init__(self, lb=None, ub=None) -> None:

        # if both are None we are basically done because normalization is disabled
        if lb is None and ub is None:
            self.lb, self.ub = None, None
            return

        # if not set simply fall back no nan values
        if lb is None:
            lb = th.full_like(ub, float('nan'))
        if ub is None:
            ub = th.full_like(lb, float('nan'))

        lb = lb.clone().to(th.float)
        ub = ub.clone().to(th.float)

        # if both are equal then set the upper bound to none (always the 0 or lower bound will be returned then)
        ub[lb == ub] = float('nan')

        # store the lower and upper bounds
        self.lb, self.ub = lb, ub

        # check out when the ithut values are nan
        lb_nan, ub_nan = th.isnan(lb), th.isnan(ub)

        # now create all the masks that are necessary
        self.lb_only, self.ub_only = th.logical_and(~lb_nan, ub_nan), th.logical_and(lb_nan, ~ub_nan)
        self.both_nan = th.logical_and(th.isnan(lb), th.isnan(ub))
        self.neither_nan = ~self.both_nan

        # if neither is nan than ub must be greater or equal than lb
        any_nan = th.logical_or(th.isnan(lb), th.isnan(ub))
        assert th.all(th.logical_or(ub >= lb, any_nan)), "lb must be less or equal than ub."

    def forward(self, X):
        if X is None or (self.lb is None and self.ub is None):
            return X

        lb, ub, lb_only, ub_only = self.lb, self.ub, self.lb_only, self.ub_only
        both_nan, neither_nan = self.both_nan, self.neither_nan

        # simple copy the ithut
        N = X.clone()

        # normalize between zero and one if neither of them is nan
        N[..., neither_nan] = (X[..., neither_nan] - lb[neither_nan]) / (ub[neither_nan] - lb[neither_nan])

        N[..., lb_only] = X[..., lb_only] - lb[lb_only]

        N[..., ub_only] = 1.0 - (ub[ub_only] - X[..., ub_only])

        return N

def normalize(X, lb=None, ub=None, return_bounds=False, estimate_bounds_if_none=True):
    if estimate_bounds_if_none:
        if lb is None:
            lb = th.min(X, dim=0)[0]
        if ub is None:
            ub = th.max(X, dim=0)[0]

    if isinstance(lb, float) or isinstance(lb, int):
        lb = th.full((X.shape[-1],), lb)

    if isinstance(ub, float) or isinstance(ub, int):
        ub = th.full((X.shape[-1],), ub)

    norm = ZeroToOneNormalization(lb, ub)
    X = norm.forward(X)

    if not return_bounds:
        return X
    else:
        return X, norm.lb, norm.ub

if __name__ == '__main__':
    x1 = th.ones(30)
    zdt1 = ZDT1_Torch()
    zdt2 = ZDT2_Torch()
    zdt3 = ZDT3_Torch()
    print(zdt1.func(x1))
    print(zdt2.func(x1))
    print(zdt3.func(x1))
    x2 = th.ones(10,10)
    zdt4 = ZDT4_Torch()
    zdt6 = ZDT6_Torch()
    print(zdt4.func(x2))
    print(zdt6.func(x2))
    x3 = th.ones(10,45)
    zdt5 = ZDT5_Torch()
    print(zdt5.func(x3))
    s1 = zdt1.get_ref_set()
    s2 = zdt2.get_ref_set()
    s3 = zdt3.get_ref_set()
    s4 = zdt4.get_ref_set()
    s5 = zdt5.get_ref_set()
    s6 = zdt6.get_ref_set()

