from ....problem.basic_problem import Basic_Problem
import numpy as np
import time 

class AUGMENTED_WCCI2020_Numpy_Problem(Basic_Problem):
    """
    # AUGMENTED_WCCI2020_Numpy_Problem
      A Numpy-based implementation of base class for defining basic functions in AUGMENTED WCCI2020 Multitask Optimization(MTO) benchmark problems.
    # Introduction
      Augmented WCCI2020 proposes 127 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
    None
    # Official Implementation
    None
    # License
    None
    # Problem Suite Composition
      The Augmented WCCI2020 problem suite contains a total of 127 benchmark problems, with each problem consisting of multiple different basic functions with unique transformations(shifts and rotations).
      The number of basic functions can be specified according to the user's requirements. Defaults to 10.
      These 127 benchmark problems are composed based on all combinations of the seven basic functions as Shpere, Rosenbrock, Rastrigin, Ackley, Griewank, Weierstrass and Schwefel.
      For each benchmark problem, the basic functions in the correspondent combination are selected randomly and added with unique transformations(shifts and rotations) until the number of basic functions is reached.
    # Methods:
    - `__init__(dim, shift, rotate, bias)`: Initializes the problem with the given parameters.
    - `get_optimal()`: Returns the optimal solution for the problem.
    - `func(x)`: Abstract method to define the problem's objective function. Must be implemented in subclasses.
    - `decode(x)`: Decodes a solution from the normalized space [0, 1] to the problem's actual search space.
    - `sr_func(x, shift, rotate)`: Applies shift and rotation transformations to the input solution.
    - `eval(x)`: Evaluates the solution(s) using the problem's objective function. Supports both individual and population evaluations.
    # Raises:
    - `NotImplementedError`: Raised if the `func` method is not implemented in a subclass.
    """
    def __init__(self, dim, shift, rotate, bias):
        """
        # Introduction
        Initializes the class AUGMENTED_WCCI2020_Numpy_Problem.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem.
        # Attributes:
        - `T1` (float): Tracks the cumulative evaluation time in milliseconds.
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem.
        - `lb` (float): The lower bound of the problem's search space.
        - `ub` (float): The upper bound of the problem's search space.
        - `FES` (int): The number of function evaluations performed.
        - `opt` (numpy.ndarray): The optimal solution for the problem.
        - `optimum` (float): The optimal function value for the problem.
        """
        self.T1 = 0
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.lb = -50
        self.ub = 50
        self.FES = 0
        self.opt = self.shift
        # self.optimum = self.eval(self.get_optimal())
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        """
        # Introduction
        Returns the optimal solution for the problem.
        # Returns:
        - numpy.ndarray: The optimal solution of the problem.
        """
        return self.opt

    def func(self, x):
        """
        # Introduction
        Abstract method to define the problem's objective function. Must be implemented in subclasses.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        """
        raise NotImplementedError
    
    def decode(self, x):
        """
        # Introduction
        Decodes a solution from the normalized space [0, 1] to the problem's actual search space.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The decoded solution of the problem.
        """
        return x * (self.ub - self.lb) + self.lb

    def sr_func(self, x, shift, rotate):
        """
        # Introduction
        Applies shift and rotation transformations to the input solution.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        # Returns:
        - numpy.ndarray: The solution being transformed by shift and rotatation.
        """
        y = x - shift
        return np.matmul(rotate, y.transpose()).transpose()
    
    def eval(self, x):
        """
        # Introduction
        A specific version of func() with adaptation to evaluate both individual and population in MTO.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        start=time.perf_counter()
        x = self.decode(x)  # the solution in MTO is constrained in a unified space [0,1]
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:  # x is a single individual
            x = x[:self.dim]
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            x = x[:, :self.dim]
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            x = x[:,:,:self.dim]
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y

class Sphere(AUGMENTED_WCCI2020_Numpy_Problem):
    LB = -100
    UB = 100
    def __init__(self, dim, shift, rotate, bias=0):
        """
        # Introduction
        Initializes the Shpere function.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem. Defaults to 0.
        # Attributes:
        - `lb` (float): The lower bound of the problem's search space. Defaults to -100.
        - `ub` (float): The upper bound of the problem's search space. Defaults to 100.
        """
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        """
        # Introduction
        The specific implementation of the Sphere's objective function.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        z = self.sr_func(x, self.shift, self.rotate)
        return np.sum(z ** 2, -1)
    
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "S" representing the object.
        """
        return 'S'

class Ackley(AUGMENTED_WCCI2020_Numpy_Problem):
    LB = -50
    UB = 50
    def __init__(self, dim, shift, rotate, bias=0):
        """
        # Introduction
        Initializes the Ackley function.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem. Defaults to 0.
        # Attributes:
        - `lb` (float): The lower bound of the problem's search space. Defaults to -50.
        - `ub` (float): The upper bound of the problem's search space. Defaults to 50.
        """
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        """
        # Introduction
        The specific implementation of the Ackley's objective function.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        z = self.sr_func(x, self.shift, self.rotate)
        sum1 = -0.2 * np.sqrt(np.sum(z ** 2, -1) / self.dim)
        sum2 = np.sum(np.cos(2 * np.pi * z), -1) / self.dim
        return np.round(np.e + 20 - 20 * np.exp(sum1) - np.exp(sum2), 15) + self.bias
    
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "A" representing the object.
        """
        return 'A'
    
class Griewank(AUGMENTED_WCCI2020_Numpy_Problem):
    LB = -100
    UB = 100
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        """
        # Introduction
        Initializes the Griewank function.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem. Defaults to 0.
        # Attributes:
        - `lb` (float): The lower bound of the problem's search space. Defaults to -100.
        - `ub` (float): The upper bound of the problem's search space. Defaults to 100.
        """
        super().__init__(dim, shift, rotate, bias)
        self.lb = -100
        self.ub = 100

    def func(self, x):
        """
        # Introduction
        The specific implementation of the Griewank's objective function.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        z = self.sr_func(x, self.shift, self.rotate)
        s = np.sum(z ** 2, -1)
        p = np.ones(x.shape[0])
        for i in range(self.dim):
            p *= np.cos(z[:, i] / np.sqrt(1 + i))
        return 1 + s / 4000 - p + self.bias
    
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "G" representing the object.
        """
        return 'G'

class Rastrigin(AUGMENTED_WCCI2020_Numpy_Problem):
    LB = -50
    UB = 50
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        """
        # Introduction
        Initializes the Rastrigin function.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem. Defaults to 0.
        # Attributes:
        - `lb` (float): The lower bound of the problem's search space. Defaults to -50.
        - `ub` (float): The upper bound of the problem's search space. Defaults to 50.
        """
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        """
        # Introduction
        The specific implementation of the Rastrigin's objective function.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        z = self.sr_func(x, self.shift, self.rotate)
        return np.sum(z ** 2 - 10 * np.cos(2 * np.pi * z) + 10, -1) + self.bias
    
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "R" representing the object.
        """
        return 'R'
    
class Rosenbrock(AUGMENTED_WCCI2020_Numpy_Problem):
    LB = -50
    UB = 50
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        """
        # Introduction
        Initializes the Rosenbrock function.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem. Defaults to 0.
        # Attributes:
        - `lb` (float): The lower bound of the problem's search space. Defaults to -50.
        - `ub` (float): The upper bound of the problem's search space. Defaults to 50.
        """
        super().__init__(dim, shift, rotate, bias)
        self.lb = -50
        self.ub = 50

    def func(self, x):
        """
        # Introduction
        The specific implementation of the Rosenbrock's objective function.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        z = self.sr_func(x, self.shift, self.rotate)
        z += 1
        z_ = z[:, 1:]
        z = z[:, :-1]
        tmp1 = z ** 2 - z_
        return np.sum(100 * tmp1 * tmp1 + (z - 1) ** 2, -1) + self.bias
    
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "Ro" representing the object.
        """
        return 'Ro'

class Weierstrass(AUGMENTED_WCCI2020_Numpy_Problem):
    LB = -0.5
    UB = 0.5
    def __init__(self, dim, shift, rotate, bias=0):
        """
        # Introduction
        Initializes the Weierstrass function.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem. Defaults to 0.
        # Attributes:
        - `lb` (float): The lower bound of the problem's search space. Defaults to -0.5.
        - `ub` (float): The upper bound of the problem's search space. Defaults to 0.5.
        """
        super().__init__(dim, shift, rotate, bias)
        self.lb = -0.5
        self.ub = 0.5

    def func(self, x):
        """
        # Introduction
        The specific implementation of the Weierstrass's objective function.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        z = self.sr_func(x, self.shift, self.rotate)
        a, b, k_max = 0.5, 3.0, 20
        sum1, sum2 = 0, 0
        for k in range(k_max + 1):
            sum1 += np.sum(np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (z + 0.5)), -1)
            sum2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
        return sum1 - self.dim * sum2 + self.bias
    
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "W" representing the object.
        """
        return 'W'
    
class Schwefel(AUGMENTED_WCCI2020_Numpy_Problem):
    LB = -500
    UB = 500
    def __init__(self, dim, shift=None, rotate=None, bias=0):
        """
        # Introduction
        Initializes the Schwefel function.
        # Args:
        - `dim` (int): The dimensionality of the problem.
        - `shift` (numpy.ndarray): The shift vector applied to the problem.
        - `rotate` (numpy.ndarray): The rotation matrix applied to the problem.
        - `bias` (float): The bias value added to the problem. Defaults to 0.
        # Attributes:
        - `lb` (float): The lower bound of the problem's search space. Defaults to -500.
        - `ub` (float): The upper bound of the problem's search space. Defaults to 500.
        """
        super().__init__(dim, shift, rotate, bias)
        self.lb = -500
        self.ub = 500

    def func(self, x):
        """
        # Introduction
        The specific implementation of the Schwefel's objective function.
        # Args:
        - `x` (numpy.ndarray): The solution of the problem.
        # Returns:
        - numpy.ndarray: The fitness value of the solution.
        """
        z = self.sr_func(x, self.shift, self.rotate)
        a = 4.209687462275036e+002
        b = 4.189828872724338e+002
        z += a
        z = np.clip(z, a_min=self.lb, a_max=self.ub)
        g = z * np.sin(np.sqrt(np.abs(z)))
        return b * self.dim - np.sum(g,-1)
    
    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The string "Sc" representing the object.
        """
        return 'Sc'