
from ....problem.basic_problem import Basic_Problem_Torch
import torch as th
import math
import numpy as np
import itertools
from scipy.special import comb

def crtup(n_obj, n_ref_points=1000):
    """
    # Introduction
    Generates a set of uniformly distributed reference points (weight vectors) for a given number of objectives.
    This function is typically used in multi-objective optimization algorithms such as NSGA-III and RVEA,
    where reference points are required to guide the selection process.

    # Args:
    - n_obj (int): Number of objectives (i.e., the dimensionality of the reference points).
    - n_ref_points (int): Approximate number of desired reference points (default: 1000).

    # Returns:
    - W (np.ndarray): A 2D array of shape (n_comb, n_obj) representing the generated reference vectors.
    - n_comb (int): Actual number of generated reference vectors.
    """
    def find_H_for_closest_points(N, M):
        """
        # Introduction
        Finds the closest integer H such that the number of combinations C(H+M-1, M-1)
        is as close as possible to N without exceeding it.

        # Args:
        - N (int): Desired number of reference points.
        - M (int): Number of objectives.

        # Returns:
        - closest_H (int): The value of H that generates the closest number of points ≤ N.
        - closest_N (int): The actual number of points generated using closest_H.
        """
        H_min, H_max = 1, 100000
        closest_H = H_min
        closest_diff = float('inf')
        closest_N = 0

        for H in range(H_min, H_max + 1):
            generated_points = int(comb(H + M - 1, M - 1))


            if generated_points > N:
                break

            diff = abs(generated_points - N) 

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
            sub_result = np.array(combination) - arr - 1
            result.append(sub_result)
    else:
        print("Length mismatch between combinations and temp array. Cannot compute reference points.")
    result = np.array(result)
    W = np.zeros((n_comb, M))
    W[:, 0] = result[:, 0] - 0
    for i in range(1, M - 1):
        W[:, i] = result[:, i] - result[:, i - 1]
    W[:, -1] = H - result[:, -1]

    W = W / H
    return W, n_comb

def crtgp(dim, N):
    """
    # Introduction
    Generate a set of evenly distributed grid points in a unit hypercube of specified dimension.

    This function tries to generate at most N points that are uniformly spread in a `dim`-dimensional unit cube [0,1]^dim
    by using a Cartesian grid (meshgrid) approach.

    # Args:
    - dim (int): Dimensionality of the grid (number of variables).
    - N (int): Maximum number of grid points to generate.

    # Returns:
    - grid_points (np.ndarray): A 2D array of shape (total_points, dim) representing the generated grid points.
    - total_points (int): Actual number of points generated (≤ N).
    """
    n_points_per_dim = int(np.floor(N ** (1 / dim)))

    total_points = n_points_per_dim ** dim

    while total_points > N:
        n_points_per_dim -= 1
        total_points = n_points_per_dim ** dim

    grid_points = np.meshgrid(*[np.linspace(0, 1, n_points_per_dim)] * dim)
    grid_points = np.vstack([g.ravel() for g in grid_points]).T 

    return grid_points, total_points

class DTLZ_Torch(Basic_Problem_Torch):
    """
    # Introduction
    A PyTorch version of the DTLZ (Deb-Thiele-Laumanns-Zitzler) test suite for multi-objective optimization problems.
    """


    def __init__(self, n_var, n_obj, k=None, **kwargs):

        if n_var:
            self.k = n_var - n_obj + 1
        elif k:
            self.k = k
            n_var = k + n_obj - 1
        else:
            raise Exception("Either provide number of variables or k!")
        self.n_var = n_var
        self.n_obj = n_obj
        self.vtype = float
        self.lb = th.zeros(n_var)
        self.ub = th.ones(n_var)

    def g1(self, X_M):
        return 100 * (self.k + th.sum(th.square(X_M - 0.5) - th.cos(20 * math.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        return th.sum(th.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= th.prod(th.cos(th.pow(X_[:, :X_.shape[1] - i], alpha) * math.pi / 2.0), axis=1)
            if i > 0:
                _f *= th.sin(th.pow(X_[:, X_.shape[1] - i], alpha) * math.pi / 2.0)

            f.append(_f)

        f = th.column_stack(f)
        return f
    def __str__(self):
        return  self.__class__.__name__ + "_" + str(self.n_obj) + "_" + str(self.n_var)

class DTLZ1_Torch(DTLZ_Torch):
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def obj_func(self, X_, g):
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= th.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return th.column_stack(f)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out = self.obj_func(X_, g)
        return out

    def get_ref_set(self,n_ref_points=1000):
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        referenceObjV = uniformPoint / 2
        return referenceObjV


class DTLZ2_Torch(DTLZ_Torch):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out= self.obj_func(X_, g, alpha=1)
        return out

    def get_ref_set(self,n_ref_points=1000): 
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        uniformPoint = th.tensor(uniformPoint, dtype=th.float32)
        referenceObjV = uniformPoint / th.tile(th.sqrt(th.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV


class DTLZ3_Torch(DTLZ_Torch):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out = self.obj_func(X_, g, alpha=1)
        return out

    def get_ref_set(self,n_ref_points=1000):
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        uniformPoint = th.tensor(uniformPoint, dtype=th.float32)
        referenceObjV = uniformPoint / th.tile(th.sqrt(th.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV


class DTLZ4_Torch(DTLZ_Torch):
    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)
        self.alpha = alpha
        self.d = d


    def func(self, x,  *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out =  self.obj_func(X_, g, alpha=self.alpha)
        return out

    def get_ref_set(self,n_ref_points=1000):
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        uniformPoint = th.tensor(uniformPoint, dtype=th.float32)
        referenceObjV = uniformPoint / th.tile(th.sqrt(th.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV


class DTLZ5_Torch(DTLZ_Torch):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = th.column_stack([x[:, 0], theta[:, 1:]])

        out = self.obj_func(theta, g)
        return out
    def get_ref_set(self,n_ref_points=1000):
        
        N = n_ref_points
        P = th.vstack([th.linspace(0, 1, N), th.linspace(1, 0, N)]).T
        P = P / th.tile(th.sqrt(th.sum(P ** 2, 1, keepdims=True)), (1, P.shape[1]))
        P = th.hstack([P[:, th.zeros(self.n_obj - 2, dtype=th.long)], P])
        referenceObjV = P / th.sqrt(th.tensor(2, dtype=th.float32)) ** th.tile(th.hstack([th.tensor(self.n_obj - 2), th.linspace(self.n_obj - 2, 0, self.n_obj - 1)]),
                                                  (P.shape[0], 1))
        return referenceObjV

class DTLZ6_Torch(DTLZ_Torch):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x, *args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = th.sum(th.pow(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = th.column_stack([x[:, 0], theta[:, 1:]])

        out = self.obj_func(theta, g)
        return out

    def get_ref_set(self,n_ref_points = 1000):
        N = n_ref_points  #
        P = th.vstack([th.linspace(0, 1, N), th.linspace(1, 0, N)]).T
        P = P / th.tile(th.sqrt(th.sum(P ** 2, 1, keepdims=True)), (1, P.shape[1]))
        P = th.hstack([P[:, th.zeros(self.n_obj - 2, dtype=th.long)], P])
        referenceObjV = P / th.sqrt(th.tensor(2,dtype=th.float32)) ** th.tile(th.hstack([th.tensor(self.n_obj - 2), th.linspace(self.n_obj - 2, 0, self.n_obj - 1)]),
                                                  (P.shape[0], 1))
        return referenceObjV


class DTLZ7_Torch(DTLZ_Torch):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x,*args, **kwargs):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = th.column_stack(f)

        g = 1 + 9 / self.k * th.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - th.sum(f / (1 + g[:, None]) * (1 + th.sin(3 * math.pi * f)), axis=1)

        out = th.column_stack([f, (1 + g) * h])
        return out
    def get_ref_set(self,n_ref_points = 1000):
        
        N = n_ref_points  
        
        a = 0.2514118360889171
        b = 0.6316265307000614
        c = 0.8594008566447239
        Vars, Sizes = crtgp(self.n_obj - 1, N)  
        Vars = th.tensor(Vars)
        middle = 0.5
        left = Vars <= middle
        right = Vars > middle
        maxs_Left = th.max(Vars[left])
        if maxs_Left > 0:
            Vars[left] = Vars[left] / maxs_Left * a
        Vars[right] = (Vars[right] - middle) / (th.max(Vars[right]) - middle) * (c - b) + b
        P = th.hstack([Vars, (2 * self.n_obj - th.sum(Vars * (1 + th.sin(3 * math.pi * Vars)), 1, keepdims=True))])
        referenceObjV = P
        return referenceObjV

if __name__ == '__main__':
    x = th.ones((10))
    dtlz1 = DTLZ1_Torch(n_var=10, n_obj=5)
    dtlz2 = DTLZ2_Torch(n_var=10, n_obj=5)
    dtlz3 = DTLZ3_Torch(n_var=10, n_obj=5)
    dtlz4 = DTLZ4_Torch(n_var=10, n_obj=5)
    dtlz5 = DTLZ5_Torch(n_var=10, n_obj=5)
    dtlz6 = DTLZ6_Torch(n_var=10, n_obj=5)
    dtlz7 = DTLZ7_Torch(n_var=10, n_obj=5)
    print(dtlz1.func(x))
    print(dtlz2.func(x))
    print(dtlz3.func(x))
    print(dtlz4.func(x))
    print(dtlz5.func(x))
    print(dtlz6.func(x))
    print(dtlz7.func(x))
    s1=dtlz1.get_ref_set()
    s2=dtlz2.get_ref_set()
    s3=dtlz3.get_ref_set()
    s4=dtlz4.get_ref_set()
    s5=dtlz5.get_ref_set()
    s6=dtlz6.get_ref_set()
    s7=dtlz7.get_ref_set()
