import itertools
import numpy as np
from scipy.special import comb
from ....problem.basic_problem import Basic_Problem


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




# Basic_Problem
class UF1(Basic_Problem):
    """
    # Introduction
    UF1 is a numpy-based implementation of the UF1 benchmark problem from the UF suite,a two-objective unconstrained multi-objective optimization problem.
    # Original paper
    "[Multiobjective optimization test instances for the CEC 2009 special session and competition](https://www.al-roomi.org/multimedia/CEC_Database/CEC2009/MultiObjectiveEA/CEC2009_MultiObjectiveEA_TechnicalReport.pdf)." (2008): 1-30.
    # Official Implementation
    [pymoo](https://github.com/anyoptimization/pymoo)
    # License
    Apache-2.0
    # Problem Suite Composition
    The UF problem suite contains a set of unconstrained multi-objective optimization problems designed for benchmarking optimization algorithms. 
    Each problem in the suite has a specific number of objectives and variables, with known theoretical Pareto fronts.

   # Attributes
    - n_obj (int): Number of objectives (default 2).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """

    def __init__(self):
        """
        # Introduction
        Initialize UF1 problem parameters.
        """
        self.n_obj = 2
        self.n_var = 30
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF1 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): 2D array of shape (n_samples, n_var) representing decision variables.

        # Returns
        - np.ndarray: 2D array of shape (n_samples, n_obj) representing objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        f1 = x1 + 2 * np.mean((Vars[:, J1] - np.sin(6 * np.pi * x1 + (J1 + 1) * np.pi / self.n_var)) ** 2, 1,
                              keepdims = True)
        f2 = 1 - np.sqrt(np.abs(x1)) + 2 * np.mean(
            (Vars[:, J2] - np.sin(6 * np.pi * x1 + (J2 + 1) * np.pi / self.n_var)) ** 2, 1, keepdims = True)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF1.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_ref_points, n_obj).
        """
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.
        
        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF2(Basic_Problem):
    """
    # Introduction
    UF2 is a numpy-based implementation of the UF2 benchmark problem from the UF suite,a two-objective unconstrained multi-objective optimization problem.
    
    # Attributes
    - n_obj (int): Number of objectives (default 2).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """
    def __init__(self):
        """
        # Introduction
        Initialize UF2 problem parameters.
        
        """
        self.n_obj = 2  
        self.n_var = 30  
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x): 
        """
        # Introduction
        Evaluate the UF2 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): 2D array of shape (n_samples, n_var) representing decision variables.

        # Returns
        - np.ndarray: 2D array of shape (n_samples, n_obj) representing objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x 
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

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF2.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_ref_points, n_obj).
        """  
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF3(Basic_Problem):  
    """
    # Introduction
    UF3 is a numpy-based implementation of the UF3 benchmark problem from the UF suite,
    a two-objective unconstrained multi-objective optimization problem.

    # Reference
    Deb, K., et al. "Multiobjective optimization test instances for the CEC 2009 special session and competition" (2008).

    # Attributes
    - n_obj (int): Number of objectives (default 2).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """

    def __init__(self):
        """
        # Introduction
        Initialize UF3 problem parameters.
        """
        self.n_obj = 2  
        self.n_var = 30  
        self.lb = np.array([0] * self.n_var)
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x): 
        """
        # Introduction
        Evaluate the UF3 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): 2D array of shape (n_samples, n_var) representing decision variables.

        # Returns
        - np.ndarray: 2D array of shape (n_samples, n_obj) representing objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  
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

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF3.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_ref_points, n_obj).
        """ 
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - np.sqrt(ObjV1)
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF4(Basic_Problem):
    """
    # Introduction
    UF4 is a numpy-based implementation of the UF4 benchmark problem from the UF suite,
    a two-objective unconstrained multi-objective optimization problem.

    # Reference
    Deb, K., et al. "Multiobjective optimization test instances for the CEC 2009 special session and competition" (2008).

    # Attributes
    - n_obj (int): Number of objectives (default 2).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """
    def __init__(self):
        """
        # Introduction
        Initialize UF4 problem parameters.

        """
        self.n_obj = 2  
        self.n_var = 30  
        self.lb = np.array([-2] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([2] * self.n_var)
        self.ub[0] = 1
        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF4 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): A 2D array of shape (n_samples, n_var), each row representing a decision vector.

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, n_obj), each row representing the objective values.
        """ 
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  
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

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF4.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_ref_points, n_obj).
        """
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1 ** 2
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF5(Basic_Problem):
    """
    # Introduction
    UF5 is a numpy-based implementation of the UF5 benchmark problem from the UF suite,
    characterized by a challenging Pareto set with discontinuities and multimodality.

    # Reference
    Deb, K., et al. "Multiobjective optimization test instances for the CEC 2009 special session and competition" (2008).

    # Attributes
    - n_obj (int): Number of objectives (default 2).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """
    def __init__(self):
        """
        # Introduction
        Initialize UF5 problem parameters.
        """
        self.n_obj = 2  
        self.n_var = 30  
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF5 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): A 2D array of shape (n_samples, n_var), each row representing a decision vector.

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, n_obj), each row representing the objective values.
        """ 
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x 
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        hy = 2 * y ** 2 - np.cos(4 * np.pi * y) + 1
        hy1 = hy[:, J1]
        hy2 = hy[:, J2]
        f1 = x1 + (1 / 20 + 0.1) * np.abs(np.sin(20 * np.pi * x1)) + 2 * (np.mean(hy1, 1, keepdims = True))
        f2 = 1 - x1 + (1 / 20 + 0.1) * np.abs(np.sin(20 * np.pi * x1)) + 2 * (np.mean(hy2, 1, keepdims = True))
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF5.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_ref_points, n_obj).
        """ 
        N = n_ref_points 
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF6(Basic_Problem):
    """
    # Introduction
    UF6 is a numpy-based implementation of the UF6 benchmark problem from the UF suite,
    designed to test an optimizer's ability to handle disconnected and deceptive Pareto fronts
    with multimodal landscape and discontinuities.


    # Attributes
    - n_obj (int): Number of objectives (default 2).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """
    def __init__(self):
        """
        # Introduction
        Initialize UF6 problem parameters.

        """
        self.n_obj = 2 
        self.n_var = 30 
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF6 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): A 2D array of shape (n_samples, n_var), each row representing a decision vector.

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, n_obj), each row representing the objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x
        x1 = Vars[:, [0]]
        J1 = np.array(list(range(2, self.n_var, 2)))
        J2 = np.array(list(range(1, self.n_var, 2)))
        J = np.arange(1, 31)
        J = J[np.newaxis, :]
        y = Vars - np.sin(6 * np.pi * x1 + (J * np.pi) / self.n_var)
        yJ1 = y[:, J1]
        yJ2 = y[:, J2]
        f1 = x1 + np.maximum(0, 2 * (1 / 4 + 0.1) * np.sin(4 * np.pi * x1)) + \
             (2 / len(J1)) * (4 * np.sum(yJ1 ** 2, 1, keepdims = True) - \
                              2 * (np.prod(np.cos((20 * yJ1 * np.pi) / (np.sqrt(J1))), 1, keepdims = True)) + 2)
        f2 = 1 - x1 + np.maximum(0, 2 * (1 / 4 + 0.1) * np.sin(4 * np.pi * x1)) + \
             (2 / len(J2)) * (4 * np.sum(yJ2 ** 2, 1, keepdims = True) - \
                              2 * (np.prod(np.cos((20 * yJ2 * np.pi) / (np.sqrt(J2))), 1, keepdims = True)) + 2)
        ObjV = np.hstack([f1, f2])
        return ObjV

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF6.
        The Pareto front is discontinuous, consisting of two valid segments in [0,0.25] and [0.75,1].

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_points, n_obj).
        """
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        idx = ((ObjV1 > 0) & (ObjV1 < 1 / 4)) | ((ObjV1 > 1 / 2) & (ObjV1 < 3 / 4))
        ObjV1 = ObjV1[~idx]
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF7(Basic_Problem):
    """
    # Introduction
    UF7 is a numpy-based implementation of the UF7 benchmark problem from the UF suite,
    designed to evaluate the capability of optimization algorithms to handle non-convex and
    non-uniform Pareto fronts with variable linkage and multimodal landscapes.

    # Attributes
    - n_obj (int): Number of objectives (default 2).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """

    def __init__(self):
        """
        # Introduction
        Initialize UF7 problem parameters.

        """
        self.n_obj = 2 
        self.n_var = 30 
        self.lb = np.array([-1] * self.n_var)
        self.lb[0] = 0
        self.ub = np.array([1] * self.n_var)
        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF7 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): A 2D array of shape (n_samples, n_var), each row representing a decision vector.

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, n_obj), each row representing the objective values.
        """
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x 
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

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF7.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_points, n_obj).
        """
        N = n_ref_points
        ObjV1 = np.linspace(0, 1, N)
        ObjV2 = 1 - ObjV1
        referenceObjV = np.array([ObjV1, ObjV2]).T
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF8(Basic_Problem):
    """
    # Introduction
    UF8 is a numpy-based implementation of the UF8 benchmark problem from the UF suite.
    It is a three-objective problem designed to test an algorithm's ability to handle
    multi-objective landscapes with complex variable linkage and diverse Pareto sets.


    # Attributes
    - n_obj (int): Number of objectives (default 3).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """
    def __init__(self):
        """
        # Introduction
        Initialize UF8 problem parameters, including bounds and variable types.

        """
        self.n_obj = 3 
        self.n_var = 30 
        self.lb = np.array([0] * 2 + [-2] * (self.n_var - 2))
        self.ub = np.array([1] * 2 + [2] * (self.n_var - 2))
        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF8 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): A 2D array of shape (n_samples, n_var), each row representing a decision vector.

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, n_obj), each row representing the objective values.
        """ 
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x 
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

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set approximating the theoretical Pareto front for UF8.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: Reference set of shape (n_points, n_obj), normalized to the unit hypersphere.
        """
        N = n_ref_points
        ObjV, N = crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = ObjV / np.sqrt(np.sum(ObjV ** 2, 1, keepdims = True))
        referenceObjV = ObjV
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF9(Basic_Problem):
    """
    # Introduction
    UF9 is a numpy-based implementation of the UF9 benchmark problem from the UF suite.It is a three-objective problem designed to evaluate the capability of algorithms in handling complex variable linkages and partially disconnected Pareto fronts.

    # Attributes
    - n_obj (int): Number of objectives (default 3).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """ 
    def __init__(self):
        """
        # Introduction
        Initialize UF9 problem parameters, including bounds and variable types.

        """
        self.n_obj = 3  
        self.n_var = 30  
        self.lb = np.array([0] * 2 + [-2] * (self.n_var - 2))
        self.ub = np.array([1] * 2 + [2] * (self.n_var - 2))

        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF9 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): A 2D array of shape (n_samples, n_var), each row representing a decision vector.

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, n_obj), each row representing the objective values.
        """ 
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x 
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

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a reference set that approximates the Pareto front of UF9.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: A 2D array of reference points on the approximated Pareto front.
        """ 
        N = n_ref_points  
        ObjV, N = crtup(self.n_obj, N)  # ObjV.shape=N,3
        idx = (ObjV[:, 0] > (1 - ObjV[:, 2]) / 4) & (ObjV[:, 0] < (1 - ObjV[:, 2]) * 3 / 4)
        referenceObjV = ObjV[~idx]
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.


        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
        return self.__class__.__name__ + "_n" + str(self.n_obj) + "_d" + str(self.n_var)


class UF10(Basic_Problem):
    """
    # Introduction
    UF10 is a numpy-based implementation of the UF10 benchmark problem from the UF suite.It is a three-objective test problem with complex linkages, designed to evaluate optimization algorithms on multi-objective problems with diverse Pareto fronts.

    # Attributes
    - n_obj (int): Number of objectives (default 3).
    - n_var (int): Number of decision variables (default 30).
    - lb (np.ndarray): Lower bounds for decision variables.
    - ub (np.ndarray): Upper bounds for decision variables.
    - vtype (type): Variable type, default float.

    # Methods
    - __init__(): Initialize problem parameters.
    - func(x): Calculate objective values from decision variables.
    - get_ref_set(n_ref_points=1000): Generate theoretical Pareto front samples.
    - __str__(): Return problem description string.
    """ 

    def __init__(self):
        """
        # Introduction
        Initialize UF10 problem parameters, including bounds and variable types.

        """
        self.n_obj = 3  
        self.n_var = 30  
        self.lb = np.array([0] * 2 + [-2] * (self.n_var - 2))
        self.ub = np.array([1] * 2 + [2] * (self.n_var - 2))
        self.vtype = float

    def func(self, x):
        """
        # Introduction
        Evaluate the UF10 objective functions for the input decision variables.

        # Args
        - x (np.ndarray): A 2D array of shape (n_samples, n_var), or a 1D array of shape (n_var,).

        # Returns
        - np.ndarray: A 2D array of shape (n_samples, n_obj), each row containing objective values.
        """  
        if x.ndim == 1:
            x = np.expand_dims(x, axis = 0)
        Vars = x  
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

    def get_ref_set(self, n_ref_points = 1000):
        """
        # Introduction
        Generate a set of uniformly distributed reference points on the unit sphere in 3D.

        # Args
        - n_ref_points (int): Number of reference points to generate (default 1000).

        # Returns
        - np.ndarray: A 2D array of shape (n_ref_points, n_obj), representing Pareto front samples.
        """  
        N = n_ref_points  
        ObjV, N = crtup(self.n_obj, N)  # ObjV.shape=N,3
        ObjV = ObjV / np.sqrt(np.sum(ObjV ** 2, 1, keepdims = True))
        referenceObjV = ObjV
        return referenceObjV

    def __str__(self):
        """
        # Introduction
        Return a string representation of the problem class.

        # Returns
        - str: A string describing the problem's name, number of objectives, and variables.
        """
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
