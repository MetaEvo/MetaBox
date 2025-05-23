from ....problem.basic_problem import Basic_Problem
import numpy as np
import time 

class CEC2017MTO_Numpy_Problem(Basic_Problem):
    """
    # CEC2017MTO_Numpy_Problem
      A Numpy-based implementation of base class for defining basic functions in CEC2017 Multitask Optimization(MTO) benchmark problems.
    # Introduction
      CEC2017MTO proposes 9 multi-task benchmark problems to represent a wider range of multi-task optimization problems.
    # Original Paper
      "[Evolutionary Multitasking for Single-objective Continuous Optimization: Benchmark Problems, Performance Metric, and Bseline Results](https://arxiv.org/pdf/1706.03470)."
    # Official Implementation
      [CEC2017MTO](http://www.bdsc.site/websites/MTO/index.html)
    # License
    None
    # Problem Suite Composition
      The CEC2017MTO problem suite contains a total of 9 benchmark problems, each consisting of two basic functions.
      These nine benchmark problems are classified according to the degree of intersection and the inter-task similarity between the two constitutive functions:
        P1. Complete intersection and high similarity(CI+HS)
        P2. Complete intersection and medium similarity(CI+MS)  
        P3. Complete intersection and low similarity(CI+LS)
        P4. Partial intersection and high similarity(PI+HS)
        P5. Partial intersection and medium similarity(PI+MS)  
        P6. Partial intersection and low similarity(PI+LS)
        P7. No intersection and high similarity(NI+HS)
        P8. No intersection and medium similarity(NI+MS) 
        P9. No intersection and low similarity(NI+LS)
    # Methods:
    - `get_optimal() -> np.ndarray`: Returns the optimal solution for the problem.
    - `func(x: np.ndarray) -> float`: Abstract method to compute the objective function value for a given solution. Must be implemented in subclasses.
    - `decode(x: np.ndarray) -> np.ndarray`: Decodes a solution from the normalized space [0, 1] to the problem's search space.
    - `sr_func(x: np.ndarray, shift: np.ndarray, rotate: np.ndarray) -> np.ndarray`: Applies shift and rotation transformations to a solution.
    - `eval(x: np.ndarray) -> float or np.ndarray`: Evaluates the objective function for a single solution or a population of solutions. Supports both individual and population-based evaluations.
    # Raises:
    - `NotImplementedError`: Raised if the `func` method is not implemented in a subclass.
    """
    def __init__(self, dim, shift, rotate, bias):
        """
        # Introduction
        Initializes the class CEC2017MTO_Numpy_Problem.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `shift` (np.ndarray or None): Shift vector for the problem. If `None`, a zero vector is used.
        - `rotate` (np.ndarray or None): Rotation matrix for the problem. If `None`, no rotation is applied.
        - `bias` (float): Bias value added to the objective function.
        # Attributes:
        - `T1` (float): Accumulated time (in milliseconds) for evaluating solutions.
        - `dim` (int): Dimensionality of the problem.
        - `shift` (np.ndarray): Shift vector for the problem.
        - `rotate` (np.ndarray): Rotation matrix for the problem.
        - `bias` (float): Bias value added to the objective function.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `FES` (int): Function evaluation count.
        - `opt` (np.ndarray): Optimal solution for the problem.
        - `optimum` (float): Objective function value at the optimal solution.
        """
        self.T1 = 0
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.lb = -50
        self.ub = 50
        self.FES = 0

        self.opt = self.shift if self.shift is not None else np.zeros(shape=(self.dim,))
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
        if shift is not None: 
            y = x - shift
        else:
            y = x
        
        if rotate is not None:
            z = np.matmul(rotate, y.transpose()).transpose()
        else:
            z = y 
        return z
    
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

class Sphere(CEC2017MTO_Numpy_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
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
        - str: The string "Sphere" representing the object.
        """
        return 'Sphere'

class Ackley(CEC2017MTO_Numpy_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
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
        - str: The string "Ackley" representing the object.
        """
        return 'Ackley'
    
class Griewank(CEC2017MTO_Numpy_Problem):
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
        - str: The string "Griewank" representing the object.
        """
        return 'Griewank'

class Rastrigin(CEC2017MTO_Numpy_Problem):
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
        - str: The string "Rastrigin" representing the object.
        """
        return 'Rastrigin'
    
class Rosenbrock(CEC2017MTO_Numpy_Problem):
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
        - str: The string "Rosenbrock" representing the object.
        """
        return 'Rosenbrock'

class Weierstrass(CEC2017MTO_Numpy_Problem):
    def __init__(self, dim, shift=None, rotate=None, bias=0):
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
        - str: The string "Weierstrass" representing the object.
        """
        return 'Weierstrass'
    
class Schwefel(CEC2017MTO_Numpy_Problem):
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
        - str: The string "Schwefel" representing the object.
        """
        return 'Schwefel'