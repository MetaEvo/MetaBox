
from ....problem.basic_problem import Basic_Problem
import itertools
import numpy as np
from scipy.special import comb
import math

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

class DTLZ(Basic_Problem):
    """
    # Introduction
    The `DTLZ` class represents a numpy-based family of multi-objective optimization problems commonly used in benchmarking optimization algorithms. These problems are designed to evaluate the performance of algorithms in handling trade-offs between multiple conflicting objectives. The class provides a flexible implementation of the DTLZ problem suite, allowing users to specify the number of variables, objectives, and other parameters.
    # Original paper
    "[Scalable multi-objective optimization test problems](https://ieeexplore.ieee.org/abstract/document/1007032)." Proceedings of the 2002 congress on evolutionary computation. CEC'02 (Cat. No. 02TH8600). Vol. 1. IEEE, 2002.
    # Official Implementation
    [pymoo](https://github.com/anyoptimization/pymoo)
    # License
    Apache-2.0
    # Problem Suite Composition
    The DTLZ problem suite consists of a set of scalable multi-objective optimization problems. Each problem is parameterized by the number of decision variables (`n_var`) and the number of objectives (`n_obj`). The problems are designed to test the scalability and performance of optimization algorithms in high-dimensional objective spaces.
    # Args:
    - `n_var` (int): The number of decision variables. If not provided, it is computed using `k` and `n_obj`.
    - `n_obj` (int): The number of objectives.
    - `k` (int, optional): The number of distance-related variables. If not provided, it is computed using `n_var` and `n_obj`.
    - `**kwargs`: Additional keyword arguments for customization.
    # Attributes:
    - `n_var` (int): The number of decision variables.
    - `n_obj` (int): The number of objectives.
    - `k` (int): The number of distance-related variables.
    - `vtype` (type): The type of variables (default is `float`).
    - `lb` (numpy.ndarray): The lower bounds of the decision variables.
    - `ub` (numpy.ndarray): The upper bounds of the decision variables.
    # Methods:
    - `g1(X_M)`: Computes the `g1` function, which is a component of the DTLZ problem.
    - `g2(X_M)`: Computes the `g2` function, which is another component of the DTLZ problem.
    - `obj_func(X_, g, alpha=1)`: Computes the objective function values for the given decision variables and `g` function.
    - `__str__()`: Returns a string representation of the problem, including the number of objectives and decision variables.
    # Raises:
    - `Exception`: Raised if neither `n_var` nor `k` is provided during initialization.
    """

    def __init__(self, n_var, n_obj, k=None, **kwargs):
        """
        # Introduction
        Initializes a specific instance of the DTLZ problem suite.  
        If `k` is not provided, it is computed based on the number of decision variables and objectives.

        # Args:
        - n_var (int): Number of decision variables. If not provided explicitly, will be computed from `k` and `n_obj`.
        - n_obj (int): Number of objectives.
        - k (int, optional): Number of distance-related variables. Optional; computed if not given.
        - **kwargs: Additional keyword arguments for extension or metadata (currently unused).
        
        # Raises:
        - Exception: If neither `n_var` nor `k` is provided.
        """

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
        self.lb = np.zeros(n_var)
        self.ub = np.ones(n_var)

    def g1(self, X_M):
        """
        # Introduction
        Computes the `g1` function of the DTLZ problem, which includes a complex multimodal landscape to test algorithm robustness.

        # Args:
        - X_M (np.ndarray): A 2D numpy array representing the distance-related variables (shape: [n_samples, k]).

        # Returns:
        - np.ndarray: Computed g1 values for each input vector.
        """
        return 100 * (self.k + np.sum(np.square(X_M - 0.5) - np.cos(20 * np.pi * (X_M - 0.5)), axis=1))

    def g2(self, X_M):
        """
        # Introduction
        Computes the `g2` function of the DTLZ problem, representing a simpler sphere-like landscape.

        # Args:
        - X_M (np.ndarray): A 2D numpy array representing the distance-related variables (shape: [n_samples, k]).

        # Returns:
        - np.ndarray: Computed g2 values for each input vector.
        """
        return np.sum(np.square(X_M - 0.5), axis=1)

    def obj_func(self, X_, g, alpha=1):
        """
        # Introduction
        Computes the multi-objective values for a population of decision variables using the DTLZ objective function formulation.

        # Args:
        - X_ (np.ndarray): A 2D array of decision variables (shape: [n_samples, n_var - k]).
        - g (np.ndarray): A 1D array of `g` values (shape: [n_samples, ]) computed by `g1` or `g2`.
        - alpha (float, optional): An exponent applied to the decision variables (default is 1, i.e., linear).

        # Returns:
        - np.ndarray: A 2D array of shape [n_samples, n_obj], where each row is a vector of objective values.
        """
        f = []

        for i in range(0, self.n_obj):
            _f = (1 + g)
            _f *= np.prod(np.cos(np.power(X_[:, :X_.shape[1] - i], alpha) * np.pi / 2.0), axis=1)
            if i > 0:
                _f *= np.sin(np.power(X_[:, X_.shape[1] - i], alpha) * np.pi / 2.0)

            f.append(_f)

        f = np.column_stack(f)
        return f
    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return  self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class DTLZ1(DTLZ):
    """
    # Introduction
    DTLZ1 is a scalable benchmark problem in multi-objective optimization, designed to evaluate an algorithm's ability to converge to and maintain a diverse set of solutions along a linear Pareto front.

    # Args:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - **kwargs: Additional keyword arguments passed to the parent class.

    # Attributes:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - k (int): Number of distance-related variables.
    - lb (np.ndarray): Lower bound of decision variables (all zeros).
    - ub (np.ndarray): Upper bound of decision variables (all ones).
    - vtype (type): Variable type, default is float.

    # Methods:
    - obj_func(X_, g): Compute objective values given shape-related variables X_ and function g.
    - func(x): Evaluate the full decision vector x and return its objective values.
    - get_ref_set(n_ref_points): Generate a reference Pareto front (true PF) consisting of uniformly spaced points.

    # Raises:
    - Exception: Raised during parent initialization if neither `n_var` nor `k` is properly specified.
    """
    
    def __init__(self, n_var=7, n_obj=3, **kwargs):
        """
        # Introduction
        Initialize the DTLZ1 problem instance.

        # Args:
        - n_var (int): Number of decision variables.Default is 7.
        - n_obj (int): Number of objectives.Default is 3.
        - **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def obj_func(self, X_, g):
        """
        # Introduction
        Compute the objective function values for the DTLZ1 problem.

        # Args:
        - X_ (np.ndarray): The shape-related decision variables (first n_obj-1 columns).
        - g (np.ndarray): The distance function value computed from the remaining variables.

        # Returns:
        - np.ndarray: Objective values for each solution in the population.
        """
        f = []

        for i in range(0, self.n_obj):
            _f = 0.5 * (1 + g)
            _f *= np.prod(X_[:, :X_.shape[1] - i], axis=1)
            if i > 0:
                _f *= 1 - X_[:, X_.shape[1] - i]
            f.append(_f)

        return np.column_stack(f)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the DTLZ1 objective function given a set of decision variables.

        # Args:
        - x (np.ndarray): Decision variable array, can be 1D or 2D.

        # Returns:
        - np.ndarray: Evaluated objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out = self.obj_func(X_, g)
        return out

    def get_ref_set(self,n_ref_points=1000): 
        """
        # Introduction
        Generate a reference set of uniformly distributed points on the true Pareto front.

        # Args:
        - n_ref_points (int): Number of reference points to generate.

        # Returns:
        - np.ndarray: Reference objective values on the Pareto front.
        """
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        referenceObjV = uniformPoint / 2
        return referenceObjV


class DTLZ2(DTLZ):
    """
    # Introduction
    DTLZ2 is a scalable benchmark problem in multi-objective optimization. It is designed to test an algorithm's ability to maintain a uniform distribution of solutions on a spherical Pareto front.

    # Args:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - **kwargs: Additional keyword arguments passed to the parent class.

    # Attributes:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - k (int): Number of distance-related variables.
    - lb (np.ndarray): Lower bound of decision variables (all zeros).
    - ub (np.ndarray): Upper bound of decision variables (all ones).
    - vtype (type): Variable type, default is float.

    # Methods:
    - func(x): Evaluate the objective values for input decision vector(s).
    - get_ref_set(n_ref_points): Generate reference points uniformly distributed on the true spherical Pareto front.

    # Raises:
    - Exception: Raised during parent initialization if parameters are invalid.
    """
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        """
        # Introduction
        Initialize the DTLZ2 problem instance.

        # Args:
        - n_var (int): Number of decision variables. Default is 10.
        - n_obj (int): Number of objectives. Default is 3.
        - **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the DTLZ2 objective function given a set of decision variables.

        # Args:
        - x (np.ndarray): Decision variable array. Can be 1D or 2D.

        # Returns:
        - np.ndarray: Evaluated objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out= self.obj_func(X_, g, alpha=1)
        return out

    def get_ref_set(self,n_ref_points=1000): 
        """
        # Introduction
        Generate a reference set of uniformly distributed points on the true spherical Pareto front.

        # Args:
        - n_ref_points (int): Number of reference points to generate.

        # Returns:
        - np.ndarray: Reference objective values on the Pareto front.
        """
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        referenceObjV = uniformPoint / np.tile(np.sqrt(np.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV


class DTLZ3(DTLZ):
    """
    # Introduction
    DTLZ3 is a scalable benchmark problem in multi-objective optimization. It is designed to test an algorithm’s ability to maintain convergence and diversity in the presence of many local Pareto-optimal fronts.

    # Args:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - **kwargs: Additional keyword arguments passed to the parent class.

    # Attributes:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - k (int): Number of distance-related variables.
    - lb (np.ndarray): Lower bound of decision variables (all zeros).
    - ub (np.ndarray): Upper bound of decision variables (all ones).
    - vtype (type): Variable type, default is float.

    # Methods:
    - func(x): Evaluate the objective values for input decision vector(s).
    - get_ref_set(n_ref_points): Generate reference points uniformly distributed on the true spherical Pareto front.

    # Raises:
    - Exception: Raised during parent initialization if parameters are invalid.
    """
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        """
        # Introduction
        Initialize the DTLZ3 problem instance.

        # Args:
        - n_var (int): Number of decision variables. Default is 10.
        - n_obj (int): Number of objectives. Default is 3.
        - **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the DTLZ3 objective function given a set of decision variables.

        # Args:
        - x (np.ndarray): Decision variable array. Can be 1D or 2D.

        # Returns:
        - np.ndarray: Evaluated objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g1(X_M)
        out = self.obj_func(X_, g, alpha=1)
        return out

    def get_ref_set(self,n_ref_points=1000): 
        """
        # Introduction
        Generate a reference set of uniformly distributed points on the true spherical Pareto front.

        # Args:
        - n_ref_points (int): Number of reference points to generate.

        # Returns:
        - np.ndarray: Reference objective values on the Pareto front.
        """
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        referenceObjV = uniformPoint / np.tile(np.sqrt(np.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV


class DTLZ4(DTLZ):
    """
    # Introduction
    DTLZ4 is a benchmark problem in multi-objective optimization that introduces a parameterized distortion to bias the distribution of solutions along the Pareto front, challenging the diversity maintenance of optimization algorithms.

    # Args:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - alpha (float): Exponent used to bias the distribution of decision variables. Default is 100.
    - d (int): Number of distance-related variables. Default is 100.
    - **kwargs: Additional keyword arguments passed to the parent class.

    # Attributes:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - alpha (float): Distribution distortion parameter.
    - d (int): Number of distance-related variables.
    - lb (np.ndarray): Lower bound of decision variables (all zeros).
    - ub (np.ndarray): Upper bound of decision variables (all ones).
    - vtype (type): Variable type, default is float.

    # Methods:
    - func(x): Evaluate the objective values for input decision vector(s).
    - get_ref_set(n_ref_points): Generate reference points uniformly distributed on the true spherical Pareto front.

    # Raises:
    - Exception: Raised during parent initialization if parameters are invalid.
    """

    def __init__(self, n_var=10, n_obj=3, alpha=100, d=100, **kwargs):
        """
        # Introduction
        Initialize the DTLZ4 problem instance with the specified number of variables, objectives, and distortion parameters.

        # Args:
        - n_var (int): Number of decision variables. Default is 10.
        - n_obj (int): Number of objectives. Default is 3.
        - alpha (float): Exponent to control the distribution of solutions. Default is 100.
        - d (int): Number of distance-related variables. Default is 100.
        - **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)
        self.alpha = alpha
        self.d = d


    def func(self, x,  *args, **kwargs):
        """
        # Introduction
        Evaluate the DTLZ4 objective function given a set of decision variables.

        # Args:
        - x (np.ndarray): Decision variable array. Can be 1D or 2D.

        # Returns:
        - np.ndarray: Evaluated objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)
        out =  self.obj_func(X_, g, alpha=self.alpha)
        return out

    def get_ref_set(self,n_ref_points=1000):
        """
        # Introduction
        Generate a reference set of uniformly distributed points on the true spherical Pareto front.

        # Args:
        - n_ref_points (int): Number of reference points to generate.

        # Returns:
        - np.ndarray: Reference objective values on the Pareto front.
        """
        uniformPoint, ans = crtup(self.n_obj, n_ref_points)
        referenceObjV = uniformPoint / np.tile(np.sqrt(np.sum(uniformPoint ** 2, 1, keepdims=True)), (1, self.n_obj))
        return referenceObjV


class DTLZ5(DTLZ):
    """
    # Introduction
    DTLZ5 is a benchmark problem in multi-objective optimization that introduces a non-linear transformation of the decision variables to reduce the dimensionality of the objective space, thereby increasing the difficulty for algorithms to maintain diversity.

    # Args:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - **kwargs: Additional keyword arguments passed to the parent class.

    # Attributes:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - lb (np.ndarray): Lower bound of decision variables (all zeros).
    - ub (np.ndarray): Upper bound of decision variables (all ones).
    - vtype (type): Variable type, default is float.

    # Methods:
    - func(x): Evaluate the objective values for input decision vector(s).
    - get_ref_set(n_ref_points): Generate reference points on the true Pareto front with a degenerate shape.

    # Raises:
    - Exception: Raised during parent initialization if parameters are invalid.
    """
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        """
        # Introduction
        Initialize the DTLZ5 problem instance with specified variables and objectives.

        # Args:
        - n_var (int): Number of decision variables. Default is 10.
        - n_obj (int): Number of objectives. Default is 3.
        - **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the DTLZ5 objective function using transformed decision variables to reduce the effective dimensionality.

        # Args:
        - x (np.ndarray): Decision variable array. Can be 1D or 2D.

        # Returns:
        - np.ndarray: Evaluated objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = self.g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack([x[:, 0], theta[:, 1:]])

        out = self.obj_func(theta, g)
        return out
    def get_ref_set(self,n_ref_points=1000):
        """
        # Introduction
        Generate a reference set of solutions on the true Pareto front for DTLZ5, which lies on a lower-dimensional manifold.

        # Args:
        - n_ref_points (int): Number of reference points to generate.

        # Returns:
        - np.ndarray: Reference objective values on the Pareto front.
        """
        N = n_ref_points
        P = np.vstack([np.linspace(0, 1, N), np.linspace(1, 0, N)]).T
        P = P / np.tile(np.sqrt(np.sum(P ** 2, 1, keepdims=True)), (1, P.shape[1]))
        P = np.hstack([P[:, np.zeros(self.n_obj - 2, dtype=np.int64)], P])
        referenceObjV = P / np.sqrt(2) ** np.tile(np.hstack([self.n_obj - 2, np.linspace(self.n_obj - 2, 0, self.n_obj - 1)]),
                                                  (P.shape[0], 1))
        return referenceObjV

class DTLZ6(DTLZ):
    """
    # Introduction
    DTLZ6 is a benchmark problem in multi-objective optimization with a deceptive Pareto-optimal front, 
    designed to test the convergence and diversity capabilities of algorithms under non-uniform mappings.

    # Args:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - **kwargs: Additional keyword arguments passed to the parent class.

    # Attributes:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - lb (np.ndarray): Lower bound of decision variables (all zeros).
    - ub (np.ndarray): Upper bound of decision variables (all ones).
    - vtype (type): Variable type, default is float.

    # Methods:
    - func(x): Evaluate the objective values for input decision vector(s).
    - get_ref_set(n_ref_points): Generate reference points on the true Pareto front.

    # Raises:
    - Exception: Raised during parent initialization if parameters are invalid.
    """
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        """
        # Introduction
        Initialize the DTLZ6 problem instance.

        # Args:
        - n_var (int): Number of decision variables. Default is 10.
        - n_obj (int): Number of objectives. Default is 3.
        - **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the DTLZ6 objective function with a non-linear transformation and biased distribution.

        # Args:
        - x (np.ndarray): Decision variable array. Can be 1D or 2D.

        # Returns:
        - np.ndarray: Evaluated objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = np.sum(np.power(X_M, 0.1), axis=1)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = np.column_stack([x[:, 0], theta[:, 1:]])

        out = self.obj_func(theta, g)
        return out

    def get_ref_set(self,n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set of solutions lying on the true Pareto front for DTLZ6.

        # Args:
        - n_ref_points (int): Number of reference points to generate.

        # Returns:
        - np.ndarray: Reference objective values on the Pareto front.
        """
        N = n_ref_points  #
        P = np.vstack([np.linspace(0, 1, N), np.linspace(1, 0, N)]).T
        P = P / np.tile(np.sqrt(np.sum(P ** 2, 1, keepdims=True)), (1, P.shape[1]))
        P = np.hstack([P[:, np.zeros(self.n_obj - 2, dtype=np.int64)], P])
        referenceObjV = P / np.sqrt(2) ** np.tile(np.hstack([self.n_obj - 2, np.linspace(self.n_obj - 2, 0, self.n_obj - 1)]),
                                                  (P.shape[0], 1))
        return referenceObjV


class DTLZ7(DTLZ):
    def __init__(self, n_var=10, n_obj=3, **kwargs):
        """
    # Introduction
    DTLZ7 is a benchmark problem featuring a disconnected and non-convex Pareto front, 
    designed to test the algorithm’s ability to maintain diversity across multiple isolated regions.

    # Args:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - **kwargs: Additional keyword arguments passed to the parent class.

    # Attributes:
    - n_var (int): Number of decision variables.
    - n_obj (int): Number of objectives.
    - lb (np.ndarray): Lower bound of decision variables (all zeros).
    - ub (np.ndarray): Upper bound of decision variables (all ones).
    - vtype (type): Variable type, default is float.

    # Methods:
    - func(x): Evaluate the objective values for input decision vector(s).
    - get_ref_set(n_ref_points): Generate reference points on the true disconnected Pareto front.

    # Raises:
    - Exception: Raised during parent initialization if parameters are invalid.
    """
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)


    def func(self, x,*args, **kwargs):
        """
        # Introduction
        Initialize the DTLZ7 problem instance.

        # Args:
        - n_var (int): Number of decision variables. Default is 10.
        - n_obj (int): Number of objectives. Default is 3.
        - **kwargs: Additional arguments passed to the parent class.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis=0)
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = np.column_stack(f)

        g = 1 + 9 / self.k * np.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - np.sum(f / (1 + g[:, None]) * (1 + np.sin(3 * np.pi * f)), axis=1)

        out = np.column_stack([f, (1 + g) * h])
        return out
    def get_ref_set(self,n_ref_points = 1000):
        """
        # Introduction
        Evaluate the DTLZ7 objective function featuring a disconnected Pareto front.

        # Args:
        - x (np.ndarray): Decision variable array. Can be 1D or 2D.

        # Returns:
        - np.ndarray: Evaluated objective values.
        """
        N = n_ref_points  
        a = 0.2514118360889171
        b = 0.6316265307000614
        c = 0.8594008566447239
        Vars, Sizes = crtgp(self.n_obj - 1, N) 
        middle = 0.5
        left = Vars <= middle
        right = Vars > middle
        maxs_Left = np.max(Vars[left])
        if maxs_Left > 0:
            Vars[left] = Vars[left] / maxs_Left * a
        Vars[right] = (Vars[right] - middle) / (np.max(Vars[right]) - middle) * (c - b) + b
        P = np.hstack([Vars, (2 * self.n_obj - np.sum(Vars * (1 + np.sin(3 * np.pi * Vars)), 1, keepdims=True))])
        referenceObjV = P
        return referenceObjV

if __name__ == '__main__':
    x = np.ones((10,))
    dtlz1 = DTLZ1(n_var=10, n_obj=5)
    dtlz2 = DTLZ2(n_var=10, n_obj=5)
    dtlz3 = DTLZ3(n_var=10, n_obj=5)
    dtlz4 = DTLZ4(n_var=10, n_obj=5)
    dtlz5 = DTLZ5(n_var=10, n_obj=5)
    dtlz6 = DTLZ6(n_var=10, n_obj=5)
    dtlz7 = DTLZ7(n_var=10, n_obj=5)
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
