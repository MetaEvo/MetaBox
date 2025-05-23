
import numpy as np
from ....problem.basic_problem import Basic_Problem


def find_non_dominated_indices(Point):
    """
    # Introduction
    Find the indices of non-dominated solutions in a population.

    A solution is said to be non-dominated if no other solution in the population
    dominates it. This function performs a pairwise comparison between all solutions.

    # Args:
    - Point (np.ndarray): A 2D array of shape (n_points, n_objectives), where each row 
                          represents the objective values of a solution.

    # Returns:
    - non_dominated_indices (np.ndarray): Indices of the non-dominated solutions.
    """
    n_points = Point.shape[0]
    is_dominated = np.zeros(n_points, dtype = bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                if np.all(Point[j] <= Point[i]) and np.any(Point[j] < Point[i]):
                    is_dominated[i] = True
                    break

    non_dominated_indices = np.where(~is_dominated)[0]
    return non_dominated_indices


class ZDT(Basic_Problem):
    """
    # Introduction
    The `ZDT` class represents a numpy-based synthetic multi-objective optimization problem from the ZDT problem suite. 
    These problems are widely used as benchmarks in the field of evolutionary multi-objective optimization.
    # Original paper
    "[Comparison of multiobjective evolutionary algorithms: Empirical results](https://ieeexplore.ieee.org/abstract/document/6787994)." Evolutionary computation 8.2 (2000): 173-195.
    # Official Implementation
    [pymoo](https://github.com/anyoptimization/pymoo)
    # License
    Apache-2.0
    # Problem Suite Composition
    The ZDT problem suite consists of six benchmark problems (ZDT1 to ZDT6) designed to test the performance of multi-objective optimization algorithms. Each problem has two or three objectives and varying levels of complexity in terms of Pareto front shapes and decision space characteristics.
    # Args:
    - `n_var` (int, optional): Number of decision variables. Defaults to 30.
    - `**kwargs`: Additional keyword arguments for customization.
    # Attributes:
    - `n_var` (int): Number of decision variables.
    - `n_obj` (int): Number of objectives. Defaults to 2.
    - `vtype` (type): Variable type, set to `float`.
    - `lb` (numpy.ndarray): Lower bounds of the decision variables, initialized to zeros.
    - `ub` (numpy.ndarray): Upper bounds of the decision variables, initialized to ones.
    # Methods:
    - `__str__() -> str`: Returns a string representation of the problem, including the class name, number of objectives, and number of decision variables.
    # Raises:
    - No specific exceptions are raised by this class, but errors may occur if the input arguments are invalid or if the methods are used improperly.
    """

    def __init__(self, n_var = 30, **kwargs):
        """
        # Introduction
        Initialize a ZDT problem with specified number of decision variables.

        # Args
        - n_var (int, optional): Number of decision variables. Default is 30.
        - **kwargs: Additional keyword arguments for further customization.

        """
        self.n_var = n_var
        self.n_obj = 2
        self.vtype = float
        self.lb = np.zeros(n_var)
        self.ub = np.ones(n_var)

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Args
        None

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class ZDT1(ZDT):
    """
    # Introduction
    The `ZDT1` class represents the first problem in the ZDT benchmark suite. It features a convex Pareto front and is commonly used to evaluate the convergence ability of multi-objective optimization algorithms.

    # Methods
    - func(x): Compute the objective values of ZDT1.
    - get_ref_set(n_ref_points): Generate a reference Pareto front for ZDT1.
    """

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Compute the two objective values of the ZDT1 problem for a given set of decision variable vectors.

        # Args
        - x (np.ndarray): A 1D or 2D NumPy array representing decision variables. 
                          If 1D, it will be reshaped to (1, n_var).
        - *args: Unused positional arguments.
        - **kwargs: Unused keyword arguments.

        # Returns
        - np.ndarray: A 2D NumPy array of shape (n_samples, 2), containing the objective values for each solution.

        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = x[:, 0]
        g = 1 + 9.0 / (self.n_var - 1) * np.sum(x[:, 1:], axis = 1)
        f2 = g * (1 - np.power((f1 / g), 0.5))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):  
        """
        # Introduction
        Generate the reference Pareto front for ZDT1, which has a convex shape and is analytically defined.

        # Args
        - n_ref_points (int, optional): Number of points to sample along the Pareto front. Defaults to 1000.

        # Returns
        - np.ndarray: A 2D NumPy array of shape (n_ref_points, 2), representing the true Pareto front.

        """
        N = n_ref_points  #
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT2(ZDT):
    """
    # Introduction
    The `ZDT2` class represents the second problem in the ZDT benchmark suite. It is characterized by a non-convex 
    Pareto front and is designed to test the ability of optimization algorithms to converge and maintain diversity 
    in non-convex regions.

    # Methods
    - func(x): Compute the objective values of ZDT2.
    - get_ref_set(n_ref_points): Generate a reference Pareto front for ZDT2.
    """

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Compute the two objective values of the ZDT2 problem for a given set of decision variable vectors.

        # Args
        - x (np.ndarray): A 1D or 2D NumPy array representing decision variables. 
                          If 1D, it will be reshaped to (1, n_var).
        - *args: Unused positional arguments.
        - **kwargs: Unused keyword arguments.

        # Returns
        - np.ndarray: A 2D NumPy array of shape (n_samples, 2), containing the objective values for each solution.

        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis = 1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power((f1 * 1.0 / g), 2))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000): 
        """
        # Introduction
        Generate the reference Pareto front for ZDT2, which is known to be non-convex and continuous.

        # Args
        - n_ref_points (int, optional): Number of points to sample along the Pareto front. Defaults to 1000.

        # Returns
        - np.ndarray: A 2D NumPy array of shape (n_ref_points, 2), representing the true Pareto front.

        """
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 2
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT3(ZDT):
    """
    # Introduction
    The `ZDT3` class represents the third problem in the ZDT multi-objective benchmark suite.
    This problem features a disconnected Pareto front and is used to evaluate the diversity-preserving ability of algorithms.

    # Args
    - `n_var` (int, optional): Number of decision variables. Default is 30.
    - `**kwargs`: Additional keyword arguments for customization.

    # Attributes
    - `n_var` (int): Number of decision variables.
    - `n_obj` (int): Number of objectives, fixed at 2.
    - `vtype` (type): Variable type, set to `float`.
    - `lb` (np.ndarray): Lower bounds of the decision variables, set to 0.
    - `ub` (np.ndarray): Upper bounds of the decision variables, set to 1.

    # Methods
    - `func(x, *args, **kwargs)`: Computes the objective values for input `x`.
    - `get_ref_set(n_ref_points=1000)`: Returns reference Pareto-optimal points for ZDT3.
    """

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the objective values for the ZDT3 problem. ZDT3 features a multi-modal, non-convex, and 
        discontinuous Pareto front. The objective is to map the decision variables to their respective 
        objective values in the two-objective space.

        # Args
        - x (np.ndarray): A 1D or 2D numpy array representing the decision variables.
        - *args: Additional positional arguments (unused).
        - **kwargs: Additional keyword arguments (unused).

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, 2) containing the computed objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = x[:, 0]
        c = np.sum(x[:, 1:], axis = 1)
        g = 1.0 + 9.0 * c / (self.n_var - 1)
        f2 = g * (1 - np.power(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * np.sin(10 * np.pi * f1))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a set of true Pareto-optimal objective vectors for the ZDT3 problem.
        ZDT3 has a discontinuous front composed of several convex segments.

        # Args
        - n_ref_points (int): Number of uniformly sampled candidate points. Default is 1000.

        # Returns
        - np.ndarray: A 2D array containing non-dominated Pareto-optimal objective vectors.
        """
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
    """
    # Introduction
    The `ZDT4` class represents the fourth problem in the ZDT multi-objective benchmark suite.
    This problem introduces a multimodal landscape with many local Pareto-optimal solutions to evaluate
    the ability of optimization algorithms to escape local optima.

    
    # Args
    - `n_var` (int, optional): Number of decision variables. Default is 10.

    # Attributes
    - `n_var` (int): Number of decision variables.
    - `n_obj` (int): Number of objectives, fixed at 2.
    - `vtype` (type): Variable type, set to `float`.
    - `lb` (np.ndarray): Lower bounds of the decision variables. x₀ in [0,1], others in [-5,5].
    - `ub` (np.ndarray): Upper bounds of the decision variables. x₀ in [0,1], others in [-5,5].

    # Methods
    - `func(x, *args, **kwargs)`: Computes the objective values for ZDT4.
    - `get_ref_set(n_ref_points=1000)`: Returns reference Pareto-optimal points for ZDT4.
    """

    def __init__(self, n_var = 10):
        """
        # Introduction
        Initialize the ZDT4 problem by setting custom bounds for the decision variables.
        x₀ is in [0, 1], while x₁~xₙ are in [-5, 5].

        # Args
        - n_var (int): Number of decision variables. Default is 10.
        """
        super().__init__(n_var)
        self.lb = -5 * np.ones(self.n_var)
        self.lb[0] = 0.0
        self.ub = 5 * np.ones(self.n_var)
        self.ub[0] = 1.0
        # self.func = self._evaluate

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the objective values for the ZDT4 problem. ZDT4 is characterized by a large number of
        local Pareto-optimal fronts, testing the global search capability of optimization algorithms.

        # Args
        - x (np.ndarray): A 1D or 2D array representing decision variable(s).
        - *args: Additional unused positional arguments.
        - **kwargs: Additional unused keyword arguments.

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, 2) containing the computed objective vectors.
        """
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

    def get_ref_set(self, n_ref_points = 1000):  
        """
        # Introduction
        Generate a set of true Pareto-optimal objective vectors for the ZDT4 problem.ZDT4 shares the same Pareto front shape as ZDT1, but has a rugged decision space.

        # Args
        - n_ref_points (int): Number of uniformly sampled points to generate. Default is 1000.

        # Returns
        - np.ndarray: A 2D array of shape (n_ref_points, 2) containing Pareto-optimal reference points.
        """
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZDT5(ZDT):
    """
    # Introduction
    The `ZDT5` class represents the fifth problem in the ZDT benchmark suite.
    This problem features binary decision variables and a discrete search space.
    It evaluates the ability of optimization algorithms to handle discrete and combinatorial problems.

    # Args
    - `m` (int, optional): Number of subcomponents (objectives related). Default is 11.
    - `n` (int, optional): Number of bits per subcomponent. Default is 5.
    - `normalize` (bool, optional): Whether to normalize objectives to [0,1]. Default is True.
    - `**kwargs`: Additional keyword arguments passed to the base class.

    # Attributes
    - `m` (int): Number of subcomponents.
    - `n` (int): Number of bits per subcomponent.
    - `normalize` (bool): Whether to normalize objectives.
    - `n_var` (int): Number of decision variables, computed as 30 + n * (m-1).
    - Other attributes inherited from `ZDT`.

    # Methods
    - `__init__(m=11, n=5, normalize=True, **kwargs)`: Initializes ZDT5 with given parameters.
    - `func(x, *args, **kwargs)`: Evaluates objectives for the input decision variables.
    - `get_ref_set(n_ref_points=1000)`: Generates reference Pareto front points.
    """

    def __init__(self, m = 11, n = 5, normalize = True, **kwargs):
        """
        # Introduction
        Initialize the ZDT5 problem with given parameters for subcomponents and bit-lengths.

        # Args
        - m (int): Number of subcomponents, default 11.
        - n (int): Number of bits per subcomponent, default 5.
        - normalize (bool): Whether to normalize objectives, default True.
        - **kwargs: Additional keyword arguments passed to the base class.
        """
        self.m = m
        self.n = n
        self.normalize = normalize
        super().__init__(n_var = (30 + n * (m - 1)), **kwargs)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the objectives for the ZDT5 problem. Converts input to float and splits
        decision variables into subcomponents. Calculates the objectives according to the ZDT5 definition.

        # Args
        - x (np.ndarray): Decision variable array, 1D or 2D.
        - *args: Additional positional arguments.
        - **kwargs: Additional keyword arguments.

        # Returns
        - np.ndarray: 2D array of shape (n_samples, 2) with objective values.
        """
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
        """
        # Introduction
        Generate reference Pareto front points for ZDT5.

        # Args
        - n_ref_points (int): Number of points to generate, default 1000.

        # Returns
        - np.ndarray: 2D array of reference Pareto front points.
        """
        x = 1 + np.linspace(0, 1, n_ref_points) * 30
        pf = np.column_stack([x, (self.m - 1) / x])
        if self.normalize:
            pf = normalize(pf)
        return pf


class ZDT6(ZDT):
    """
    # Introduction
    The `ZDT6` class represents the sixth problem in the ZDT benchmark suite.This problem features a non-uniformly distributed Pareto front and a complicated shape that challenges multi-objective optimization algorithms.


    # Args
    - `n_var` (int, optional): Number of decision variables. Default is 10.
    - `**kwargs`: Additional keyword arguments passed to the base class.

    # Attributes
    - `n_var` (int): Number of decision variables.
    - Other attributes inherited from `ZDT`.

    # Methods
    - `__init__(n_var=10, **kwargs)`: Initializes the ZDT6 problem.
    - `func(x, *args, **kwargs)`: Evaluates the objective functions.
    - `get_ref_set(n_ref_points=1000)`: Returns the theoretical Pareto front reference set.
    """

    def __init__(self, n_var = 10, **kwargs):
        """
        # Introduction
        Initialize ZDT6 with a specified number of decision variables.

        # Args
        - `n_var` (int): Number of decision variables, default 10.
        - `**kwargs`: Additional keyword arguments passed to the base class.
        """
        super().__init__(n_var = n_var, **kwargs)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluate the two objectives of the ZDT6 problem for given decision variables.

        # Args
        - `x` (np.ndarray): Decision variable array, can be 1D or 2D.
        - `*args`: Additional positional arguments.
        - `**kwargs`: Additional keyword arguments.

        # Returns
        - `np.ndarray`: Array of shape (n_samples, 2) containing objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        f1 = 1 - np.exp(-4 * x[:, 0]) * np.power(np.sin(6 * np.pi * x[:, 0]), 6)
        g = 1 + 9.0 * np.power(np.sum(x[:, 1:], axis = 1) / (self.n_var - 1.0), 0.25)
        f2 = g * (1 - np.power(f1 / g, 2))

        out = np.column_stack([f1, f2])
        return out

    def get_ref_set(self, n_ref_points = 1000):  
        """
        # Introduction
        Generate the theoretical Pareto front reference points for ZDT6.

        # Args
        - `n_ref_points` (int): Number of reference points to generate. Default is 1000.

        # Returns
        - `np.ndarray`: Array of shape (n_ref_points, 2) representing the Pareto front.
        """
        N = n_ref_points
        ObjV1 = np.linspace(0.280775, 1, N)
        ObjV2 = 1 - ObjV1 ** 2;
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV


class ZeroToOneNormalization():
    """
    # Introduction
    A utility class to normalize data features to the [0, 1] range based on provided lower and upper bounds.
    Supports cases where bounds may be partially or fully unspecified (NaN), handling normalization accordingly.

    # Args
    - `lb` (np.ndarray or None): Lower bounds for normalization. If None, treated as unspecified.
    - `ub` (np.ndarray or None): Upper bounds for normalization. If None, treated as unspecified.

    # Attributes
    - `lb` (np.ndarray or None): Stored lower bounds.
    - `ub` (np.ndarray or None): Stored upper bounds.
    - `lb_only` (np.ndarray of bool): Mask where only lower bound is specified.
    - `ub_only` (np.ndarray of bool): Mask where only upper bound is specified.
    - `both_nan` (np.ndarray of bool): Mask where both bounds are NaN.
    - `neither_nan` (np.ndarray of bool): Mask where neither bound is NaN.

    # Methods
    - `__init__(lb=None, ub=None)`: Initializes normalization bounds and masks.
    - `forward(X)`: Normalize input array `X` based on stored bounds.
    """

    def __init__(self, lb = None, ub = None) -> None:
        """
        # Introduction
        Initialize normalization with optional lower and upper bounds.

        # Args
        - `lb` (np.ndarray or None): Lower bounds.
        - `ub` (np.ndarray or None): Upper bounds.
        """

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
        """
        # Introduction
        Normalize input array `X` based on stored bounds.

        # Args
        - `X` (np.ndarray): Input data to be normalized.

        # Returns
        - `np.ndarray`: Normalized data in [0,1] range or adjusted according to specified bounds.
        """
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
    """
    # Introduction
    Normalize input data `X` to [0, 1] based on provided or estimated bounds.

    # Args
    - `X` (np.ndarray): Input data array.
    - `lb` (float, int, np.ndarray or None): Lower bounds. If None and `estimate_bounds_if_none` is True, estimated from `X`.
    - `ub` (float, int, np.ndarray or None): Upper bounds. If None and `estimate_bounds_if_none` is True, estimated from `X`.
    - `return_bounds` (bool): Whether to return the bounds along with normalized data.
    - `estimate_bounds_if_none` (bool): Whether to estimate bounds from `X` if they are not provided.

    # Returns
    - `np.ndarray` or tuple: Normalized data array; if `return_bounds` is True, also returns lower and upper bounds.
    """
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

