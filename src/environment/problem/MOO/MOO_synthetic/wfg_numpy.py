import numpy as np
from ....problem.basic_problem import Basic_Problem
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


class WFG(Basic_Problem):
    """
    # Introduction
    The `WFG` class represents a numpy-based family of multi-objective optimization problems known as the WFG (Walking Fish Group) test problems. These problems are widely used in the field of evolutionary multi-objective optimization to evaluate the performance of optimization algorithms. The WFG problems are characterized by their scalability, modularity, and ability to control problem difficulty through various parameters.
    # Original paper
    "[A review of multiobjective test problems and a scalable test problem toolkit](https://ieeexplore.ieee.org/abstract/document/1705400)." IEEE Transactions on Evolutionary Computation 10.5 (2006): 477-506.
    # Official Implementation
    [pymoo](https://github.com/anyoptimization/pymoo)
    # License
    Apache-2.0
    # Problem Suite Composition
    The WFG problem suite consists of a set of scalable multi-objective optimization problems. Each problem is defined by the number of decision variables (`n_var`), the number of objectives (`n_obj`), and two key parameters: `k` (position-related parameters) and `l` (distance-related parameters). These problems are designed to test the ability of optimization algorithms to handle complex Pareto fronts, including disconnected, degenerate, and mixed geometries.
    # Args:
    - `n_var` (int): Number of decision variables.
    - `n_obj` (int): Number of objectives.
    - `k` (int, optional): Position-related parameter. Defaults to `2 * (n_obj - 1)` if not provided.
    - `l` (int, optional): Distance-related parameter. Defaults to `n_var - k` if not provided.
    - `**kwargs`: Additional keyword arguments.
    # Attributes:
    - `n_var` (int): Number of decision variables.
    - `n_obj` (int): Number of objectives.
    - `lb` (numpy.ndarray): Lower bounds for decision variables.
    - `ub` (numpy.ndarray): Upper bounds for decision variables.
    - `vtype` (type): Data type of decision variables (default is `float`).
    - `S` (numpy.ndarray): Scaling factors for objectives.
    - `A` (numpy.ndarray): Shift parameters for the Pareto front.
    - `k` (int): Position-related parameter.
    - `l` (int): Distance-related parameter.
    # Methods:
    - `validate(l, k, n_obj)`: Validates the problem parameters to ensure they meet the constraints of the WFG problem suite.
    - `_post(t, a)`: Transforms the decision variables using a post-processing function.
    - `_calculate(x, s, h)`: Calculates the objective values based on the decision variables and scaling factors.
    - `_rand_optimal_position(n)`: Generates random optimal positions for the decision variables.
    - `_positional_to_optimal(K)`: Converts positional variables to optimal decision variables.
    - `__str__()`: Returns a string representation of the WFG problem instance.
    # Raises:
    - `ValueError`: Raised in the `validate` method if:
        - The number of objectives (`n_obj`) is less than 2.
        - The position parameter (`k`) is not divisible by the number of objectives minus one.
        - The position parameter (`k`) is less than 4.
        - The sum of the position and distance parameters (`k + l`) is less than the number of objectives (`n_obj`).
    """

    def __init__(self, n_var, n_obj, k = None, l = None, **kwargs):
        """
        # Introduction
        Initialize WFG problem parameters, bounds, and validate configuration.

        # Args
        - n_var (int): Number of decision variables.
        - n_obj (int): Number of objectives.
        - k (int, optional): Number of position parameters.
        - l (int, optional): Number of distance parameters.
        - **kwargs: Additional keyword arguments.

        """

        self.n_var = n_var
        self.n_obj = n_obj
        self.lb = np.zeros(self.n_var)
        self.ub = 2 * np.arange(1, n_var + 1).astype(float)
        self.vtype = float

        self.S = np.arange(2, 2 * self.n_obj + 1, 2).astype(float)
        self.A = np.ones(self.n_obj - 1)

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
        """
        # Introduction
        Validate the WFG problem configuration parameters.

        # Args
        - l (int): Number of distance parameters.
        - k (int): Number of position parameters.
        - n_obj (int): Number of objectives.


        # Raises
        - ValueError: If constraints are violated.
        """
        if n_obj < 2:
            raise ValueError('WFG problems must have two or more objectives.')
        if not k % (n_obj - 1) == 0:
            raise ValueError('Position parameter (k) must be divisible by number of objectives minus one.')
        if k < 4:
            raise ValueError('Position parameter (k) must be greater or equal than 4.')
        if (k + l) < n_obj:
            raise ValueError('Sum of distance and position parameters must be greater than num. of objs. (k + l >= M).')

    def _post(self, t, a):
        """
        # Introduction
        Apply post-processing transformation to intermediate variable vector.

        # Args
        - t (np.ndarray): Input matrix of shape (n, m).
        - a (np.ndarray): Parameter vector.

        # Returns
        - np.ndarray: Transformed decision matrix of shape (n, m).
        """
        x = []
        for i in range(t.shape[1] - 1):
            x.append(np.maximum(t[:, -1], a[i]) * (t[:, i] - 0.5) + 0.5)
        x.append(t[:, -1])
        return np.column_stack(x)

    def _calculate(self, x, s, h):
        """
        # Introduction
        Compute the final objective values for each solution.

        # Args
        - x (np.ndarray): Transformed decision vectors.
        - s (np.ndarray): Scaling factors.
        - h (list of np.ndarray): Shape function values.

        # Returns
        - np.ndarray: Final objective values of shape (n_samples, n_obj).
        """
        return x[:, -1][:, None] + s * np.column_stack(h)

    def _rand_optimal_position(self, n):
        """
        # Introduction
        Generate random optimal positional vectors.

        # Args
        - n (int): Number of samples to generate.

        # Returns
        - np.ndarray: Random vectors of shape (n, k).
        """
        return np.random.random((n, self.k))

    def _positional_to_optimal(self, K):
        """
        # Introduction
        Extend positional vector to a full decision vector using constant suffix.

        # Args
        - K (np.ndarray): Positional variable matrix of shape (n, k).

        # Returns
        - np.ndarray: Full decision variable matrix of shape (n, k + l).
        """
        suffix = np.full((len(K), self.l), 0.35)
        X = np.column_stack([K, suffix])
        return X * self.ub

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.


        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class WFG1(WFG):
    """
    # Introduction
    WFG1 is a standard multi-objective optimization test problem characterized by nonlinear and non-convex Pareto fronts.It is widely used to evaluate the performance of optimization algorithms. WFG1 includes multiple transformation stages such as bias and mixed shapes to increase problem complexity.

    # Attributes
    - Inherits all attributes from the WFG base class, including `k`, `l`, `n_obj`, `n_var`, `ub`, `S`, and `A`.

    # Methods
    - t1: Applies a shift linear transformation to the decision variables.
    - t2: Applies a flat bias transformation.
    - t3: Applies a polynomial bias transformation.
    - t4: Performs weighted sum reduction on variables.
    - func: Evaluates the WFG1 objective functions.
    - get_ref_set: Generates the true Pareto front reference points for benchmarking.
    """

    @staticmethod
    def t1(x, n, k):
        """
        # Introduction
        Shift linear transformation applied to the variables from index k to n.

        # Args
        - x (np.ndarray): Input decision variables.
        - n (int): Total number of variables.
        - k (int): Position-related parameter.

        # Returns
        - np.ndarray: Transformed variables.
        """
        x[:, k:n] = _transformation_shift_linear(x[:, k:n], 0.35)
        return x

    @staticmethod
    def t2(x, n, k):
        """
        # Introduction
        Flat bias transformation applied to variables from index k to n.

        # Args
        - x (np.ndarray): Input variables.
        - n (int): Total number of variables.
        - k (int): Position-related parameter.

        # Returns
        - np.ndarray: Transformed variables.
        """
        x[:, k:n] = _transformation_bias_flat(x[:, k:n], 0.8, 0.75, 0.85)
        return x

    @staticmethod
    def t3(x, n):
        """
        # Introduction
        Polynomial bias transformation applied to the first n variables.

        # Args
        - x (np.ndarray): Input variables.
        - n (int): Number of variables to transform.

        # Returns
        - np.ndarray: Transformed variables.
        """
        x[:, :n] = _transformation_bias_poly(x[:, :n], 0.02)
        return x

    @staticmethod
    def t4(x, m, n, k):
        """
        # Introduction
        Weighted sum reduction of variables for aggregation.

        # Args
        - x (np.ndarray): Input variables.
        - m (int): Number of objectives.
        - n (int): Total number of variables.
        - k (int): Position-related parameter.

        # Returns
        - np.ndarray: Aggregated variables for objective evaluation.
        """
        w = np.arange(2, 2 * n + 1, 2)
        gap = k // (m - 1)
        t = []
        for m in range(1, m):
            _y = x[:, (m - 1) * gap: (m * gap)]
            _w = w[(m - 1) * gap: (m * gap)]
            t.append(_reduction_weighted_sum(_y, _w))
        t.append(_reduction_weighted_sum(x[:, k:n], w[k:n]))
        return np.column_stack(t)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluates the WFG1 multi-objective functions for given decision variable vectors.

        # Args
        - x (np.ndarray): Decision variable input(s), shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Objective values of shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG1.t2(y, self.n_var, self.k)
        y = WFG1.t3(y, self.n_var)
        y = WFG1.t4(y, self.n_obj, self.n_var, self.k)

        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_mixed(y[:, 0], alpha = 1.0, A = 5.0))

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates the true Pareto front reference set for the WFG1 problem.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points of shape (N, n_obj).
        """
        N = n_ref_points 
        Point, num = crtup(self.n_obj, N)  
        M = self.n_obj
        c = np.ones((num, M))
        for i in range(num):
            for j in range(1, M):
                temp = Point[i, j] / Point[i, 0] * np.prod(1 - c[i, M - j: M - 1])
                c[i, M - j - 1] = (temp ** 2 - temp + np.sqrt(2 * temp)) / (temp ** 2 + 1)
        x = np.arccos(c) * 2 / np.pi
        temp = (1 - np.sin(np.pi / 2 * x[:, [1]])) * Point[:, [M - 1]] / Point[:, [M - 2]]
        a = np.linspace(0, 1, 10000 + 1)
        for i in range(num):
            E = np.abs(
                temp[i] * (1 - np.cos(np.pi / 2 * a)) - 1 + a + np.cos(10 * np.pi * a + np.pi / 2) / 10 / np.pi)
            rank = np.argsort(E, kind = 'mergesort')
            x[i, 0] = a[np.min(rank[0: 10])]
        Point = convex(x)
        Point[:, [M - 1]] = mixed(x)
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (num, 1)) * Point
        return referenceObjV


class WFG2(WFG):
    """
    # Introduction
    WFG2 is a multi-objective test problem featuring non-separable variables and disconnected Pareto fronts.It is designed to challenge optimization algorithms on handling variable dependencies and discontinuities.

    # Attributes
    - Inherits from the WFG base class with parameters `k`, `l`, `n_obj`, `n_var`, `ub`, `S`, and `A`.

    # Methods
    - validate: Checks parameter validity specific to WFG2.
    - t2: Applies a non-separable transformation.
    - t3: Performs weighted sum uniform reduction.
    - func: Evaluates the WFG2 objectives.
    - get_ref_set: Generates non-dominated Pareto front reference points.
    """

    def validate(self, l, k, n_obj):
        """
        # Introduction
        Validates parameters for WFG2, extending base WFG validation with additional constraints.

        # Args
        - l (int): Distance-related parameter.
        - k (int): Position-related parameter.
        - n_obj (int): Number of objectives.

        # Raises
        - ValueError: If validation fails.
        """
        super().validate(l, k, n_obj)
        validate_wfg2_wfg3(l)

    @staticmethod
    def t2(x, n, k):
        """
        # Introduction
        Non-separable transformation applied to certain variable pairs beyond position parameter k.

        # Args
        - x (np.ndarray): Input variables.
        - n (int): Total number of variables.
        - k (int): Position-related parameter.

        # Returns
        - np.ndarray: Transformed variables.
        """
        y = [x[:, i] for i in range(k)]

        l = n - k
        ind_non_sep = k + l // 2

        i = k + 1
        while i <= ind_non_sep:
            head = k + 2 * (i - k) - 2
            tail = k + 2 * (i - k)
            y.append(_reduction_non_sep(x[:, head:tail], 2))
            i += 1

        return np.column_stack(y)

    @staticmethod
    def t3(x, m, n, k):
        """
        # Introduction
        Weighted sum uniform reduction of transformed variables for aggregation.

        # Args
        - x (np.ndarray): Input variables.
        - m (int): Number of objectives.
        - n (int): Total number of variables.
        - k (int): Position-related parameter.

        # Returns
        - np.ndarray: Aggregated variables.
        """
        ind_r_sum = k + (n - k) // 2
        gap = k // (m - 1)

        t = [_reduction_weighted_sum_uniform(x[:, (m - 1) * gap: (m * gap)]) for m in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:ind_r_sum]))

        return np.column_stack(t)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluates the WFG2 multi-objective functions for given decision variable vectors.

        # Args
        - x (np.ndarray): Decision variable input(s), shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Objective values of shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_convex(y[:, :-1], m + 1) for m in range(self.n_obj - 1)]
        h.append(_shape_disconnected(y[:, 0], alpha = 1.0, beta = 1.0, A = 5.0))

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates the true Pareto front reference set for the WFG2 problem.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points of shape (N, n_obj).
        """
        N = n_ref_points  
        Point, num = crtup(self.n_obj, N)  
        M = self.n_obj
        c = np.ones((num, M))
        for i in range(num):
            for j in range(1, M):
                temp = Point[i, j] / Point[i, 0] * np.prod(1 - c[i, M - j: M - 1])
                c[i, M - j - 1] = (temp ** 2 - temp + np.sqrt(2 * temp)) / (temp ** 2 + 1)
        x = np.arccos(c) * 2 / np.pi
        temp = (1 - np.sin(np.pi / 2 * x[:, [1]])) * Point[:, [M - 1]] / Point[:, [M - 2]]
        a = np.linspace(0, 1, 10000 + 1)
        for i in range(num):
            E = np.abs(temp[i] * (1 - np.cos(np.pi / 2 * a)) - 1 + a * np.cos(5 * np.pi * a) ** 2)
            rank = np.argsort(E, kind = 'mergesort')
            x[i, 0] = a[np.min(rank[0: 10])]
        Point = convex(x)
        Point[:, [M - 1]] = disc(x)
        index = find_non_dominated_indices(Point)  
        Point = Point[index, :]
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG3(WFG):
    """
    # Introduction
    WFG3 is a multi-objective test problem from the WFG suite that incorporates non-separable variable interactions and linear shape functions. It uses transformations from WFG1 and WFG2 and modifies the problem's parameter matrix A to increase problem difficulty.

    # Attributes
    - Inherits all attributes from the WFG base class, including `k`, `l`, `n_obj`, `n_var`, `ub`, `S`, and `A`.
    - Modifies the `A` matrix by setting all elements except the first to zero.

    # Methods
    - validate: Ensures that parameters comply with WFG2 and WFG3 constraints.
    - func: Computes the objective values for given decision variable vectors.
    - get_ref_set: Generates the true Pareto front reference points for benchmarking.
    """

    def __init__(self, n_var, n_obj, k = None, **kwargs):
        """
        # Introduction
        Initializes the WFG3 problem, adjusting the A matrix to meet WFG3 specifications.

        # Args
        - n_var (int): Number of decision variables.
        - n_obj (int): Number of objectives.
        - k (int, optional): Position-related parameter, defaults to None.
        - kwargs: Additional keyword arguments passed to the WFG base class.
        """
        super().__init__(n_var, n_obj, k = k, **kwargs)
        self.A[1:] = 0

    def validate(self, l, k, n_obj):
        """
        # Introduction
        Validates the problem parameters, ensuring compliance with both WFG2 and WFG3 constraints.

        # Args
        - l (int): Distance-related parameter.
        - k (int): Position-related parameter.
        - n_obj (int): Number of objectives.

        # Raises
        - ValueError: If validation fails.
        """
        super().validate(l, k, n_obj)
        validate_wfg2_wfg3(l)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluates the WFG3 multi-objective functions for given decision variables.

        # Args
        - x (np.ndarray): Decision variable input(s), shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Objective values of shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG2.t2(y, self.n_var, self.k)
        y = WFG2.t3(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_linear(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)

        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates the true Pareto front reference set for the WFG3 problem.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points of shape (N, n_obj).
        """
        N = n_ref_points 
        X = np.hstack([np.array([np.linspace(0, 1, N)]).T, np.zeros((N, self.n_obj - 2)) + 0.5, np.zeros((N, 1))])
        Point = linear(X)
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG4(WFG):
    """
    # Introduction
    WFG4 is a multi-objective test problem from the WFG suite characterized by a multimodal transformation and concave shape functions. It introduces multimodality in the search space through a specialized transformation.

    # Methods
    - t1: Applies a multimodal shift transformation to decision variables.
    - t2: Aggregates variables into groups and applies a weighted sum reduction.
    - func: Evaluates the objective functions for given decision variables.
    - get_ref_set: Generates a normalized true Pareto front reference set.
    """

    @staticmethod
    def t1(x):
        """
        # Introduction
        Applies a multimodal shift transformation to the input vector.

        # Args
        - x (np.ndarray): Input decision variable matrix, shape (N, n_var).

        # Returns
        - np.ndarray: Transformed decision variables after multimodal shift.
        """
        return _transformation_shift_multi_modal(x, 30.0, 10.0, 0.35)

    @staticmethod
    def t2(x, m, k):
        """
        # Introduction
        Performs uniform weighted sum reduction of decision variables grouped by objectives.

        # Args
        - x (np.ndarray): Input decision variable matrix, shape (N, n_var).
        - m (int): Number of objectives.
        - k (int): Position-related parameter.

        # Returns
        - np.ndarray: Reduced variable matrix, shape (N, m).
        """
        gap = k // (m - 1)
        t = [_reduction_weighted_sum_uniform(x[:, (m - 1) * gap: (m * gap)]) for m in range(1, m)]
        t.append(_reduction_weighted_sum_uniform(x[:, k:]))
        return np.column_stack(t)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluates the WFG4 multi-objective functions for given decision variables.

        # Args
        - x (np.ndarray): Decision variable input(s), shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Objective values of shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y = WFG4.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates the normalized true Pareto front reference set for the WFG4 problem.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points of shape (N, n_obj).
        """
        N = n_ref_points  
        Point, num = crtup(self.n_obj, N)  
        Point = Point / np.tile(np.sqrt(np.array([np.sum(Point ** 2, 1)]).T), (1, self.n_obj))
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG5(WFG):
    """
    # Introduction
    WFG5 is a multi-objective test problem from the WFG suite featuring deceptive transformations.It challenges optimization algorithms by introducing deceptiveness in the decision space.

    # Methods
    - t1: Applies a parameterized deceptive transformation to decision variables.
    - func: Evaluates the WFG5 objective functions.
    - get_ref_set: Generates a normalized true Pareto front reference set.
    """

    @staticmethod
    def t1(x):
        """
        # Introduction
        Applies a parameterized deceptive transformation to input variables.

        # Args
        - x (np.ndarray): Input decision variable matrix, shape (N, n_var).

        # Returns
        - np.ndarray: Transformed variables after deceptive transformation.
        """
        return _transformation_param_deceptive(x, A = 0.35, B = 0.001, C = 0.05)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Evaluates the WFG5 multi-objective functions for given decision variables.

        # Args
        - x (np.ndarray): Decision variable input(s), shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Objective values of shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y = WFG5.t1(y)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates the normalized true Pareto front reference set for the WFG5 problem.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points of shape (N, n_obj).
        """
        N = n_ref_points 
        Point, num = crtup(self.n_obj, N)  
        Point = Point / np.tile(np.sqrt(np.array([np.sum(Point ** 2, 1)]).T), (1, self.n_obj))
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV

    # def _calc_pareto_front(self, ref_dirs=None):
    #     if ref_dirs is None:
    #         ref_dirs = get_ref_dirs(self.n_obj)
    #     return generic_sphere(ref_dirs) * self.S


class WFG6(WFG):
    """
    # Introduction
    WFG6 is a multi-objective test problem from the WFG suite characterized by non-separable reduction 
    transformations that increase problem difficulty by linking variables in groups.

    # Methods
    - t2: Applies a non-separable reduction transformation on variable groups.
    - func: Evaluates the WFG6 objectives based on transformations and shape functions.
    - get_ref_set: Generates normalized Pareto front reference points for performance benchmarking.
    """

    @staticmethod
    def t2(x, m, n, k):
        """
        # Introduction
        Applies non-separable reduction transformation on groups of variables.

        # Args
        - x (np.ndarray): Input variable matrix, shape (N, n).
        - m (int): Number of objectives.
        - n (int): Number of variables.
        - k (int): Position parameter separating position-related variables.

        # Returns
        - np.ndarray: Reduced variable matrix, shape (N, m).
        """
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Computes the WFG6 multi-objective function values for given input variables.

        # Args
        - x (np.ndarray): Input decision variables, shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Objective function values, shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG6.t2(y, self.n_obj, self.n_var, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates a normalized set of reference points approximating the true Pareto front for WFG6.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points of shape (N, n_obj).
        """
        N = n_ref_points  
        Point, num = crtup(self.n_obj, N)  
        Point = Point / np.tile(np.sqrt(np.array([np.sum(Point ** 2, 1)]).T), (1, self.n_obj))
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG7(WFG):
    """
    # Introduction
    WFG7 is a multi-objective test problem from the WFG suite that introduces parameter-dependent transformations increasing problem complexity by coupling decision variables.

    # Methods
    - t1: Applies a parameter-dependent transformation to the first k decision variables.
    - func: Computes the WFG7 objective values with defined transformations and shape functions.
    - get_ref_set: Generates normalized reference Pareto front points for benchmarking.
    """

    @staticmethod
    def t1(x, k):
        """
        # Introduction
        Applies a parameter-dependent transformation on the first k variables based on the weighted sum of the remaining variables.

        # Args
        - x (np.ndarray): Input variable matrix, shape (N, n).
        - k (int): Number of variables to apply the transformation on.

        # Returns
        - np.ndarray: Transformed variable matrix of the same shape as input.
        """
        for i in range(k):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1:])
            x[:, i] = _transformation_param_dependent(x[:, i], aux)
        return x

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Calculates WFG7 multi-objective function values.

        # Args
        - x (np.ndarray): Input decision variables, shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Computed objectives, shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y = WFG7.t1(y, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates a normalized set of reference points approximating the true Pareto front for WFG7.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points, shape (N, n_obj).
        """
        N = n_ref_points  
        Point, num = crtup(self.n_obj, N)  
        Point = Point / np.tile(np.sqrt(np.array([np.sum(Point ** 2, 1)]).T), (1, self.n_obj))
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG8(WFG):
    """
    # Introduction
    WFG8 is a complex multi-objective test problem from the WFG suite, featuring parameter-dependent transformations that increase problem difficulty and variable interactions.

    # Methods
    - t1: Applies a parameter-dependent transformation to variables from index k to n.
    - func: Computes the WFG8 objective values by applying transformations and shape functions.
    - _positional_to_optimal: Converts positional variables into optimal values using a specific formula.
    - get_ref_set: Generates normalized reference Pareto front points for benchmarking.
    """

    @staticmethod
    def t1(x, n, k):
        """
        # Introduction
        Applies a parameter-dependent transformation on decision variables from index k to n-1,
        using the weighted sum of preceding variables as a parameter.

        # Args
        - x (np.ndarray): Input variable matrix, shape (N, n).
        - n (int): Total number of decision variables.
        - k (int): Number of position-related variables (first k variables not transformed).

        # Returns
        - np.ndarray: Transformed variable matrix of shape (N, n - k).
        """
        ret = []
        for i in range(k, n):
            aux = _reduction_weighted_sum_uniform(x[:, :i])
            ret.append(_transformation_param_dependent(x[:, i], aux, A = 0.98 / 49.98, B = 0.02, C = 50.0))
        return np.column_stack(ret)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Calculates the WFG8 multi-objective function values.

        # Args
        - x (np.ndarray): Input decision variables, shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Computed objectives, shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y[:, self.k:self.n_var] = WFG8.t1(y, self.n_var, self.k)
        y = WFG1.t1(y, self.n_var, self.k)
        y = WFG4.t2(y, self.n_obj, self.k)
        y = self._post(y, self.A)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def _positional_to_optimal(self, K):
        """
        # Introduction
        Converts positional variables into optimal values based on a defined mathematical formula
        to enhance the problem's difficulty.

        # Args
        - K (np.ndarray): Input matrix of positional variables, shape (N, k).

        # Returns
        - np.ndarray: Matrix transformed to optimal positional values, shape (N, n_var).
        """
        k, l = self.k, self.l

        for i in range(k, k + l):
            u = K.sum(axis = 1) / K.shape[1]
            tmp1 = np.abs(np.floor(0.5 - u) + 0.98 / 49.98)
            tmp2 = 0.02 + 49.98 * (0.98 / 49.98 - (1.0 - 2.0 * u) * tmp1)
            suffix = np.power(0.35, np.power(tmp2, -1.0))

            K = np.column_stack([K, suffix[:, None]])

        ret = K * (2 * (np.arange(self.n_var) + 1))
        return ret

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates a normalized set of reference points approximating the true Pareto front for WFG8.

        # Args
        - n_ref_points (int): Number of reference points to generate. Default is 1000.

        # Returns
        - np.ndarray: Reference Pareto front points, shape (N, n_obj).
        """
        N = n_ref_points  
        Point, num = crtup(self.n_obj, N)  
        Point = Point / np.tile(np.sqrt(np.array([np.sum(Point ** 2, 1)]).T), (1, self.n_obj))
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


class WFG9(WFG):
    """
    # Introduction
    WFG9 is a challenging multi-objective optimization test problem from the WFG suite,featuring multiple parameter-dependent transformations, deceptive and multi-modal shifts,and non-separable reductions.

    # Methods
    - t1: Parameter-dependent transformation applied across variables except the last.
    - t2: Applies deceptive and multi-modal shift transformations.
    - t3: Performs non-separable reduction for objective calculation.
    - func: Computes the WFG9 objectives by applying transformations and shape functions.
    - _positional_to_optimal: Maps positional variables to optimal values using a backward calculation.
    - get_ref_set: Generates normalized reference Pareto front points for benchmarking.
    """


    @staticmethod
    def t1(x, n):
        """
        # Introduction
        Applies a parameter-dependent transformation on the first n-1 variables, 
        where each variable depends on a weighted sum of subsequent variables.

        # Args
        - x (np.ndarray): Input variable matrix of shape (N, n).
        - n (int): Total number of decision variables.

        # Returns
        - np.ndarray: Transformed variable matrix of shape (N, n-1).
        """
        ret = []
        for i in range(0, n - 1):
            aux = _reduction_weighted_sum_uniform(x[:, i + 1:])
            ret.append(_transformation_param_dependent(x[:, i], aux))
        return np.column_stack(ret)

    @staticmethod
    def t2(x, n, k):
        """
        # Introduction
        Applies shift transformations: deceptive shift on the first k variables,
        and multi-modal shift on the remaining variables.

        # Args
        - x (np.ndarray): Input variable matrix, shape (N, n).
        - n (int): Number of variables.
        - k (int): Number of position-related variables.

        # Returns
        - np.ndarray: Transformed variable matrix, shape (N, n).
        """
        a = [_transformation_shift_deceptive(x[:, i], 0.35, 0.001, 0.05) for i in range(k)]
        b = [_transformation_shift_multi_modal(x[:, i], 30.0, 95.0, 0.35) for i in range(k, n)]
        return np.column_stack(a + b)

    @staticmethod
    def t3(x, m, n, k):
        """
        # Introduction
        Performs non-separable reduction on grouped variables to produce m objective-related variables.

        # Args
        - x (np.ndarray): Input variable matrix, shape (N, n).
        - m (int): Number of objectives.
        - n (int): Number of variables.
        - k (int): Number of position-related variables.

        # Returns
        - np.ndarray: Reduced variable matrix, shape (N, m).
        """
        gap = k // (m - 1)
        t = [_reduction_non_sep(x[:, (m - 1) * gap: (m * gap)], gap) for m in range(1, m)]
        t.append(_reduction_non_sep(x[:, k:], n - k))
        return np.column_stack(t)

    def func(self, x, *args, **kwargs):
        """
        # Introduction
        Computes the WFG9 multi-objective function values by sequentially applying
        transformations t1, t2, and t3, then shape functions and scaling.

        # Args
        - x (np.ndarray): Decision variable input, shape (N, n_var) or (n_var,).

        # Returns
        - np.ndarray: Objective values, shape (N, n_obj).
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        y = x / self.ub
        y[:, :self.n_var - 1] = WFG9.t1(y, self.n_var)
        y = WFG9.t2(y, self.n_var, self.k)
        y = WFG9.t3(y, self.n_obj, self.n_var, self.k)

        h = [_shape_concave(y[:, :-1], m + 1) for m in range(self.n_obj)]

        out = self._calculate(y, self.S, h)
        return out

    def _positional_to_optimal(self, K):
        """
        # Introduction
        Maps positional variables into optimal values using a backward calculation 
        involving power functions and sums over subsequent variables.

        # Args
        - K (np.ndarray): Positional variable matrix, shape (N, k).

        # Returns
        - np.ndarray: Transformed variables matrix, shape (N, n_var).
        """
        k, l = self.k, self.l

        suffix = np.full((len(K), self.l), 0.0)
        X = np.column_stack([K, suffix])
        X[:, self.k + self.l - 1] = 0.35

        for i in range(self.k + self.l - 2, self.k - 1, -1):
            m = X[:, i + 1:k + l]
            val = m.sum(axis = 1) / m.shape[1]
            X[:, i] = 0.35 ** ((0.02 + 1.96 * val) ** -1)

        ret = X * (2 * (np.arange(self.n_var) + 1))
        return ret

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generates normalized reference points uniformly distributed on the Pareto front
        for benchmarking WFG9.

        # Args
        - n_ref_points (int): Number of reference points to generate, default 1000.

        # Returns
        - np.ndarray: Reference objective values matrix, shape (N, n_obj).
        """
        N = n_ref_points  
        Point, num = crtup(self.n_obj, N) 
        Point = Point / np.tile(np.sqrt(np.array([np.sum(Point ** 2, 1)]).T), (1, self.n_obj))
        referenceObjV = np.tile(np.array([list(range(2, 2 * self.n_obj + 1, 2))]), (Point.shape[0], 1)) * Point
        return referenceObjV


## ---------------------------------------------------------------------------------------------------------
# tool for get reference point
# ---------------------------------------------------------------------------------------------------------

def convex(x):
    """
    # Introduction
    Calculates the convex shape function used in multi-objective optimization.

    # Args
    - x (np.ndarray): Input array with shape (N, M).

    # Returns
    - np.ndarray: Output array with shape (N, M).
    """
    return np.fliplr(
        np.cumprod(np.hstack([np.ones((x.shape[0], 1)), 1 - np.cos(x[:, :-1] * np.pi / 2)]), 1)) * np.hstack(
        [np.ones((x.shape[0], 1)), 1 - np.sin(x[:, list(range(x.shape[1] - 1 - 1, -1, -1))] * np.pi / 2)])


def mixed(x):
    """
    # Introduction
    Calculates the mixed shape function.

    # Args
    - x (np.ndarray): Input array with shape (N, M).

    # Returns
    - np.ndarray: Output array with shape (N, 1).
    """
    return 1 - x[:, [0]] - np.cos(10 * np.pi * x[:, [0]] + np.pi / 2) / 10 / np.pi


def linear(x):
    """
    # Introduction
    Calculates the linear shape function.

    # Args
    - x (np.ndarray): Input array with shape (N, M).

    # Returns
    - np.ndarray: Output array with shape (N, M).
    """
    return np.fliplr(np.cumprod(np.hstack([np.ones((x.shape[0], 1)), x[:, :-1]]), 1)) * np.hstack([np.ones((x.shape[0], 1)), 1 - x[:, list(range(x.shape[1] - 1 - 1, -1, -1))]])


def s_linear(x, A):
    """
    # Introduction
    Calculates the s_linear shape function, a linear shift transformation.

    # Args
    - x (np.ndarray): Input array.
    - A (float): Shift parameter.

    # Returns
    - np.ndarray: Transformed array.
    """
    return np.abs(x - A) / np.abs(np.floor(A - x) + A)


def b_flat(x, A, B, C):
    """
    # Introduction
    Calculates the b_flat transformation function.

    # Args
    - x (np.ndarray): Input array.
    - A (float): Parameter A controlling flatness.
    - B (float): Parameter B defining interval start.
    - C (float): Parameter C defining interval end.

    # Returns
    - np.ndarray: Transformed array, rounded to 6 decimals.
    """
    Output = A + np.min([0 * np.floor(x - B), np.floor(x - B)], 0) * A * (B - x) / B - np.min(
        [0 * np.floor(C - x), np.floor(C - x)], 0) * (1 - A) * (x - C) / (1 - C)
    return np.round(Output, 6)


def b_poly(x, a):
    """
    # Introduction
    Calculates the b_poly transformation, a polynomial bias function.

    # Args
    - x (np.ndarray): Input array.
    - a (float): Exponent parameter.

    # Returns
    - np.ndarray: Transformed array.
    """
    return np.sign(x) * np.abs(x) ** a


def r_sum(x, w):
    """
    # Introduction
    Calculates the weighted sum reduction function.

    # Args
    - x (np.ndarray): Input array with shape (N, M).
    - w (np.ndarray): Weight vector with length M.

    # Returns
    - np.ndarray: Reduced values of shape (N,).
    """
    Output = np.sum(x * np.tile(w, (x.shape[0], 1)), 1) / np.sum(w)
    return Output


def disc(x):
    """
    # Introduction
    Calculates the discontinuous shape function.

    # Args
    - x (np.ndarray): Input array with shape (N, M).

    # Returns
    - np.ndarray: Output array with shape (N, 1).
    """
    return 1 - x[:, [0]] * (np.cos(5 * np.pi * x[:, [0]])) ** 2


# ---------------------------------------------------------------------------------------------------------
# TRANSFORMATIONS
# ---------------------------------------------------------------------------------------------------------


def _transformation_shift_linear(value, shift = 0.35):
    """
    # Introduction
    Applies a linear shift transformation to the input value.

    # Args
    - value (np.ndarray): Input array.
    - shift (float): Shift parameter (default 0.35).

    # Returns
    - np.ndarray: Transformed array corrected to [0, 1].
    """
    return correct_to_01(np.fabs(value - shift) / np.fabs(np.floor(shift - value) + shift))


def _transformation_shift_deceptive(y, A = 0.35, B = 0.005, C = 0.05):
    """
    # Introduction
    Applies a deceptive shift transformation to the input array.

    # Args
    - y (np.ndarray): Input array.
    - A (float): Parameter controlling the deceptive region center (default 0.35).
    - B (float): Parameter controlling the deceptive region width (default 0.005).
    - C (float): Parameter controlling the depth of deception (default 0.05).

    # Returns
    - np.ndarray: Transformed array corrected to [0, 1].
    """
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


def _transformation_shift_multi_modal(y, A, B, C):
    """
    # Introduction
    Applies a multi-modal shift transformation.

    # Args
    - y (np.ndarray): Input array.
    - A (float): Controls modality amplitude.
    - B (float): Controls modality frequency.
    - C (float): Controls modality position.

    # Returns
    - np.ndarray: Transformed array corrected to [0, 1].
    """
    tmp1 = np.fabs(y - C) / (2.0 * (np.floor(C - y) + C))
    tmp2 = (4.0 * A + 2.0) * np.pi * (0.5 - tmp1)
    ret = (1.0 + np.cos(tmp2) + 4.0 * B * np.power(tmp1, 2.0)) / (B + 2.0)
    return correct_to_01(ret)


def _transformation_bias_flat(y, a, b, c):
    """
    # Introduction
    Applies a flat bias transformation to input array.

    # Args
    - y (np.ndarray): Input array.
    - a (float): Bias parameter.
    - b (float): Left boundary of flat region.
    - c (float): Right boundary of flat region.

    # Returns
    - np.ndarray: Transformed array corrected to [0, 1].
    """
    ret = a + np.minimum(0, np.floor(y - b)) * (a * (b - y) / b) \
          - np.minimum(0, np.floor(c - y)) * ((1.0 - a) * (y - c) / (1.0 - c))
    return correct_to_01(ret)


def _transformation_bias_poly(y, alpha):
    """
    # Introduction
    Applies a polynomial bias transformation to input array.

    # Args
    - y (np.ndarray): Input array.
    - alpha (float): Exponent parameter controlling bias.

    # Returns
    - np.ndarray: Transformed array corrected to [0, 1].
    """
    return correct_to_01(y ** alpha)


def _transformation_param_dependent(y, y_deg, A = 0.98 / 49.98, B = 0.02, C = 50.0):
    """
    # Introduction
    Applies a parameter-dependent transformation.

    # Args
    - y (np.ndarray): Input array.
    - y_deg (np.ndarray): Parameter dependent variable.
    - A (float): Parameter A (default 0.98 / 49.98).
    - B (float): Parameter B (default 0.02).
    - C (float): Parameter C (default 50.0).

    # Returns
    - np.ndarray: Transformed array corrected to [0, 1].
    """
    aux = A - (1.0 - 2.0 * y_deg) * np.fabs(np.floor(0.5 - y_deg) + A)
    ret = np.power(y, B + (C - B) * aux)
    return correct_to_01(ret)


def _transformation_param_deceptive(y, A = 0.35, B = 0.001, C = 0.05):
    """
    # Introduction
    Applies a parameter-dependent deceptive transformation.

    # Args
    - y (np.ndarray): Input array.
    - A (float): Parameter controlling deceptive region center (default 0.35).
    - B (float): Parameter controlling deceptive region width (default 0.001).
    - C (float): Parameter controlling deception depth (default 0.05).

    # Returns
    - np.ndarray: Transformed array corrected to [0, 1].
    """
    tmp1 = np.floor(y - A + B) * (1.0 - C + (A - B) / B) / (A - B)
    tmp2 = np.floor(A + B - y) * (1.0 - C + (1.0 - A - B) / B) / (1.0 - A - B)
    ret = 1.0 + (np.fabs(y - A) - B) * (tmp1 + tmp2 + 1.0 / B)
    return correct_to_01(ret)


# ---------------------------------------------------------------------------------------------------------
# REDUCTION
# ---------------------------------------------------------------------------------------------------------


def _reduction_weighted_sum(y, w):
    """
    # Introduction
    Applies a weighted sum reduction to the input matrix.

    # Args
    - y (np.ndarray): Input matrix of shape (n_samples, n_features).
    - w (np.ndarray): Weight vector of shape (n_features,).

    # Returns
    - np.ndarray: Reduced values corrected to [0, 1].
    """
    return correct_to_01(np.dot(y, w) / w.sum())


def _reduction_weighted_sum_uniform(y):
    """
    # Introduction
    Applies a uniform (equal-weight) sum reduction along axis 1.

    # Args
    - y (np.ndarray): Input matrix of shape (n_samples, n_features).

    # Returns
    - np.ndarray: Reduced values corrected to [0, 1].
    """
    return correct_to_01(y.mean(axis = 1))


def _reduction_non_sep(y, A):
    """
    # Introduction
    Applies a non-separable reduction transformation.

    # Args
    - y (np.ndarray): Input matrix of shape (n_samples, n_features).
    - A (int): Non-separability parameter.

    # Returns
    - np.ndarray: Reduced values corrected to [0, 1].
    """
    n, m = y.shape
    val = np.ceil(A / 2.0)

    num = np.zeros(n)
    for j in range(m):
        num += y[:, j]
        for k in range(A - 1):
            num += np.fabs(y[:, j] - y[:, (1 + j + k) % m])

    denom = m * val * (1.0 + 2.0 * A - 2 * val) / A

    return correct_to_01(num / denom)


# ---------------------------------------------------------------------------------------------------------
# SHAPE
# ---------------------------------------------------------------------------------------------------------


def _shape_concave(x, m):
    """
    # Introduction
    Computes concave Pareto front shape.

    # Args
    - x (np.ndarray): Input decision variables (n_samples, n_features).
    - m (int): Objective index (1-based).

    # Returns
    - np.ndarray: Shape values corrected to [0, 1].
    """
    M = x.shape[1]
    if m == 1:
        ret = np.prod(np.sin(0.5 * x[:, :M] * np.pi), axis = 1)
    elif 1 < m <= M:
        ret = np.prod(np.sin(0.5 * x[:, :M - m + 1] * np.pi), axis = 1)
        ret *= np.cos(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = np.cos(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_convex(x, m):
    """
    # Introduction
    Computes convex Pareto front shape.

    # Args
    - x (np.ndarray): Input decision variables (n_samples, n_features).
    - m (int): Objective index (1-based).

    # Returns
    - np.ndarray: Shape values corrected to [0, 1].
    """
    M = x.shape[1]
    if m == 1:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, :M] * np.pi), axis = 1)
    elif 1 < m <= M:
        ret = np.prod(1.0 - np.cos(0.5 * x[:, :M - m + 1] * np.pi), axis = 1)
        ret *= 1.0 - np.sin(0.5 * x[:, M - m + 1] * np.pi)
    else:
        ret = 1.0 - np.sin(0.5 * x[:, 0] * np.pi)
    return correct_to_01(ret)


def _shape_linear(x, m):
    """
    # Introduction
    Computes linear Pareto front shape.

    # Args
    - x (np.ndarray): Input decision variables (n_samples, n_features).
    - m (int): Objective index (1-based).

    # Returns
    - np.ndarray: Shape values corrected to [0, 1].
    """
    M = x.shape[1]
    if m == 1:
        ret = np.prod(x, axis = 1)
    elif 1 < m <= M:
        ret = np.prod(x[:, :M - m + 1], axis = 1)
        ret *= 1.0 - x[:, M - m + 1]
    else:
        ret = 1.0 - x[:, 0]
    return correct_to_01(ret)


def _shape_mixed(x, A = 5.0, alpha = 1.0):
    """
    # Introduction
    Computes a mixed Pareto front shape with periodicity.

    # Args
    - x (np.ndarray): Input array.
    - A (float): Frequency factor (default 5.0).
    - alpha (float): Exponent for scaling (default 1.0).

    # Returns
    - np.ndarray: Shape values corrected to [0, 1].
    """
    aux = 2.0 * A * np.pi
    ret = np.power(1.0 - x - (np.cos(aux * x + 0.5 * np.pi) / aux), alpha)
    return correct_to_01(ret)


def _shape_disconnected(x, alpha = 1.0, beta = 1.0, A = 5.0):
    """
    # Introduction
    Computes a disconnected Pareto front shape.

    # Args
    - x (np.ndarray): Input array.
    - alpha (float): Exponent for x (default 1.0).
    - beta (float): Exponent for cosine input (default 1.0).
    - A (float): Frequency of cosine (default 5.0).

    # Returns
    - np.ndarray: Shape values corrected to [0, 1].
    """
    aux = np.cos(A * np.pi * x ** beta)
    return correct_to_01(1.0 - x ** alpha * aux ** 2)


# ---------------------------------------------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------------------------------------------

def validate_wfg2_wfg3(l):
    """
    # Introduction
    Validates if the distance parameter l is valid for WFG2/WFG3.

    # Args
    - l (int): Distance parameter.

    # Raises
    - ValueError: If l is not divisible by 2.
    """
    if not l % 2 == 0:
        raise ValueError('In WFG2/WFG3 the distance-related parameter (l) must be divisible by 2.')


def correct_to_01(X, epsilon = 1.0e-10):
    """
    # Introduction
    Corrects numerical errors to ensure values lie within [0, 1].

    # Args
    - X (np.ndarray): Input array.
    - epsilon (float): Tolerance for numerical error (default 1e-10).

    # Returns
    - np.ndarray: Corrected array.
    """
    X[np.logical_and(X < 0, X >= 0 - epsilon)] = 0
    X[np.logical_and(X > 1, X <= 1 + epsilon)] = 1
    return X


if __name__ == '__main__':
    wfg1 = WFG1(10, 3)
    wfg2 = WFG2(10, 3)
    wfg3 = WFG3(10, 3)
    wfg4 = WFG4(10, 3)
    wfg5 = WFG5(10, 3)
    wfg6 = WFG6(10, 3)
    wfg7 = WFG7(10, 3)
    wfg8 = WFG8(10, 3)
    wfg9 = WFG9(10, 3)
    x = np.random.rand(10)
    s1 = wfg1.get_ref_set()
    s2 = wfg2.get_ref_set()
    s3 = wfg3.get_ref_set()
    s4 = wfg4.get_ref_set()
    s5 = wfg5.get_ref_set()
    s6 = wfg6.get_ref_set()
    s7 = wfg7.get_ref_set()
    s8 = wfg8.get_ref_set()
    s9 = wfg9.get_ref_set()


