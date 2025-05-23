from ....problem.basic_problem import Basic_Problem
import numpy as np
import math
from os import path
MINMAX = -1
import os
import importlib.util
import importlib.resources as pkg_resources
class CEC2013MMO_Numpy_Problem(Basic_Problem):
    """
    # CEC2013_MMO_Numpy_Problem
    A base class for CEC2013 Multi-Modal Optimization (MMO) problems implemented in NumPy.
    # Introduction
    CEC2013 MMO benchmark puts together 20 multimodal problems (including several identical functions with different dimension sizes), with different characteristics, for evaluating niching algorithms.
    # Original Paper
    "[Benchmark Functions for CEC’2013 Special Session and Competition on Niching Methods for Multimodal Function Optimization](https://al-roomi.org/multimedia/CEC_Database/CEC2015/NichingMultimodalOptimization/CEC2015_NichingMethods_TechnicalReport.pdf)"
    # Official Implementation
    [CEC2013MMO](https://github.com/mikeagn/CEC2013)
    # License
    Simplified BSD License
    # Problem Suite Composition
    The CEC2013 MMO problem suite contains 20 optimization problems, each with specific characteristics such as dimensionality, bounds, and multimodal properties. These problems are categorized into different difficulty levels (`easy`, `difficult`, and `all`) and can be used for benchmarking optimization algorithms.
    """

    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialize the cec2013 mmo problem with the settings.
        
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        # Attributes:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `FES` (int): Current number of function evaluations performed.
        - `optimum` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.

        
        """
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.FES = 0
        self.optimum = fopt
        self.rho = rho
        self.nopt = nopt
        self.maxfes = maxfes
    
    def func(self, x):
        """
        # Introduction:
        Abstract method to evaluate the fitness of a solution `x`. Must be implemented in a subclass.
        # Args:
        - `x` (np.ndarray) : A solution for evaluation.
        # Raises:
        - `NotImplementedError`: Raised when the `func` method is called without being implemented in a subclass.
        """
        raise NotImplementedError

    def how_many_goptima(self, pop, accuracy):
        """
        # Introduction:
        Determines the number of global optima found in a given population within a specified accuracy.
        # Args:
        - `pop` (np.ndarray) : A group of solutions for the calculation of found global optima.
        - `accuracy` (float) : The accuracy used to determin if a solution can be regarded as a satisfied global optimum.
        # Returns:
        - `count`(int): The number of global optima found within the specified accuracy.
        - `seeds` (np.ndarray): The representive solutions for found global optima.
        """
        NP, D = pop.shape[0], pop.shape[1]
        fits = self.eval(pop)
        order = np.argsort(fits)
        sorted_pop = pop[order, :]
        spopfits = fits[order]
        seeds_idx = self.__find_seeds_indices(sorted_pop, self.rho)
        count = 0
        goidx = []
        for idx in seeds_idx:
            seed_fitness = spopfits[idx]
            if math.fabs(seed_fitness - self.optimum) <= accuracy:
                count = count + 1
                goidx.append(idx)
            if count == self.nopt:
                break
        seeds = sorted_pop[goidx]
        return count, seeds


    def __find_seeds_indices(self, sorted_pop, radius):
        """
        # Introduction:
        Identifies seed points in a sorted population based on a given radius.
        # Args:
        - `sorted_pop` (np.ndarray): A group of solutions for indentifition.
        - `radius`(float) : Radius used to determine whether two solutions belong to different peaks.
        # Returns:
        - `seeds_idx` (list of int) : The index of the solutions regarded as the seed of peaks.
        """
        seeds = []
        seeds_idx = []
        for i, x in enumerate(sorted_pop):
            found = False
            for j, sx in enumerate(seeds):
                dist = math.sqrt(sum((x - sx) ** 2))
                if dist <= radius:
                    found = True
                    break
            if not found:
                seeds.append(x)
                seeds_idx.append(i)
        return seeds_idx

class CFunction(CEC2013MMO_Numpy_Problem):
    """
    # Introduction:
    The abstract class for problems with composition functions.
    # Attributes:
    - `__nofunc_` (int) : The number of basic functions.
    - `__C_` (float) : A predefined constant.
    - `__lambda_` (np.ndarray) : A parameter used to stretch or compress each basic function. 
    - `__sigma_` (np.ndarray) : A parameter to control the coverage range of each basic function, with small values to produce a narrow coverage range.
    - `__bias_` (np.ndarray) : Defines a function value bias for each basic function and denotes which optimum is the global optimum
    - `__O_` (np.ndarray) : The new shifted optimum of each basic function.
    - `__M_` (list) : The  linear transformation (rotation) matrix of each basic problem.
    - `__weight_` (np.ndarray) : the corresponding  weight of each basic function.
    - `__fi_` (np.ndarray) : The results of evaluation of basic functions.
    - `__z_` (np.ndarray) : The result after shifting, ratation and stretch/compress.
    - `__f_bias_` (int) : A function value bias for the constructed composition function.
    - `__fmaxi_` (np.ndarray) : The maximal value of basic functions.
    - `__function_` (dict) : The list of basic functions.
    """
    __nofunc_ = -1
    __C_ = 2000.0
    __lambda_ = None
    __sigma_ = None
    __bias_ = None
    __O_ = None
    __M_ = None
    __weight_ = None
    __fi_ = None
    __z_ = None
    __f_bias_ = 0
    __fmaxi_ = None
    __tmpx_ = None
    __function_ = None

    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes, nofunc):
        """
        # Introduction:
        Initialize a problem with a composition function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        - `nofunc` (int): the number of basic functions
        # Attributes:
        - `__nofunc_` (int): The number of basic functions
        """
        super(CFunction, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)
        self.__nofunc_ = nofunc

    def func(self, x):
        """
        # Introduction:
        Abstract method to evaluate the fitness of a solution `x`. Must be implemented in a subclass.
        # Args:
        - `x` (np.ndarray) : A solution for evaluation.
        # Raises:
        - `NotImplementedError`: Raised when the `func` method is called without being implemented in a subclass.
        """
        raise NotImplementedError

    def __evaluate_inner_(self, x):
        """
        # Introduction:
        Evaluate the given solutions with the composition function.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Return:
        - np.array: The result of evaluation.
        """
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")
        self.__fi_ = np.zeros((x.shape[0], self.__nofunc_))

        self.__calculate_weights(x)
        for i in range(self.__nofunc_):
            self.__transform_to_z(x, i)
            self.__fi_[:, i] = self.__function_[i](self.__z_)

        tmpsum = np.zeros((x.shape[0], self.__nofunc_))
        tmpsum = self.__weight_ * (
            self.__C_ * self.__fi_ / self.__fmaxi_[None, :] + self.__bias_[None, :]
        )
        return np.sum(tmpsum, axis = 1) * MINMAX + self.__f_bias_

    def __calculate_weights(self, x):
        """
        # Introduction:
        Calculate the weights of basic functions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        """
        self.__weight_ = np.zeros((x.shape[0], self.__nofunc_))
        for i in range(self.__nofunc_):
            mysum = np.sum((x - self.__O_[i]) ** 2, -1)
            self.__weight_[:, i] = np.exp(
                -mysum / (2.0 * self.dim * self.__sigma_[i] * self.__sigma_[i])
            )
        maxw = np.max(self.__weight_, -1)

        maxw10 = maxw ** 10
        mask = (self.__weight_ != maxw[:, None])
        self.__weight_[mask] = (self.__weight_ * (1.0 - maxw10[:, None]))[mask]

        mysum = np.sum(self.__weight_, -1)
        mask1 =  (mysum == 0.0)
        self.__weight_[mask1] = 1.0 / (1.0 * self.__nofunc_)
        mask2 = (mysum != 0.0)
        self.__weight_[mask2] = self.__weight_[mask2] / mysum[:, None][mask2]

    def __calculate_fmaxi(self):
        """
        # Introduction:
        Calculate the maximal values of each basic function.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        """
        self.__fmaxi_ = np.zeros(self.__nofunc_)
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")

        x5 = 5 * np.ones(self.dim)

        for i in range(self.__nofunc_):
            self.__transform_to_z_noshift(x5[None, :], i)
            self.__fmaxi_[i] = self.__function_[i](self.__z_)[0]

    def __transform_to_z_noshift(self, x, index):
        """
        # Introduction:
        Transform the global optima without shfit.
        # Args:
        - `x` (np.ndarray) : A solution for evaluation.
        - `index` (int): the index of the corresponding basic function.
        """
        tmpx = np.divide(x, self.__lambda_[index])
        self.__z_ = np.dot(tmpx, self.__M_[index])

    def __transform_to_z(self, x, index):
        """
        # Introduction:
        Transform the solution with shift, stretch/compress and rotation.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        - `index` (int): The index of the corresponding basic function.
        """
        tmpx = np.divide((x - self.__O_[index]), self.__lambda_[index])
        self.__z_ = np.dot(tmpx, self.__M_[index])

    def __load_rotmat(self, file_obj):
        """
        # Introduction:
        Load the rotation matrix.
        # Args:
        - `file_obj` (file object) : a file handler for reading the rotation matrix information.
        """
        self.__M_ = []

        with file_obj as f:
            tmp = np.zeros((self.dim, self.dim))
            cline = 0
            ctmp = 0
            for line in f:
                line = line.split()
                if line:
                    line = [float(i) for i in line]

                    if ctmp % self.dim == 0:
                        tmp = np.zeros((self.dim, self.dim))
                        ctmp = 0

                    tmp[ctmp] = line[: self.dim]
                    if cline >= self.__nofunc_ * self.dim - 1:
                        break
                    if cline % self.dim == 0:
                        self.__M_.append(tmp)
                    ctmp = ctmp + 1
                    cline = cline + 1

# Sphere function
def FSphere(x):
    """
    # Introduction:
    Sphere function, one of basic functions for the composition function.
    # Args:
    - `x` (np.ndarray) : A batch of solutions for evaluation.
    # Returns:
    - np.ndarray: The evaluation results.
    """
    return (x ** 2).sum(axis = 1)

# Rastrigin's function
def FRastrigin(x):
    """
    # Introduction:
    Rastrigin’s function, one of basic functions for the composition function.
    # Args:
    - `x` (np.ndarray) : A batch of solutions for evaluation.
    # Returns:
    - np.ndarray: The evaluation results.
    """
    return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10, axis=1)

# Griewank's function
def FGrienwank(x):
    """
    # Introduction:
    Grienwank’s function, one of basic functions for the composition function.
    # Args:
    - `x` (np.ndarray) : A batch of solutions for evaluation.
    # Returns:
    - np.ndarray: The evaluation results.
    """
    i = np.sqrt(np.arange(x.shape[1]) + 1.0)
    return np.sum(x ** 2, axis = 1) / 4000.0 - np.prod(np.cos(x / i[None, :]), axis = 1) + 1.0

# Weierstrass's function
def FWeierstrass(x):
    """
    # Introduction:
    Weierstrass function, one of basic functions for the composition function.
    # Args:
    - `x` (np.ndarray) : A batch of solutions for evaluation.
    # Returns:
    - np.ndarray: The evaluation results.
    """
    alpha = 0.5
    beta = 3.0
    kmax = 20
    D = x.shape[1]
    exprf = 0.0

    c1 = alpha ** np.arange(kmax + 1)
    c2 = 2.0 * np.pi * beta ** np.arange(kmax + 1)
    f = np.zeros(x.shape[0])
    c = -D * np.sum(c1 * np.cos(c2 * 0.5))

    for i in range(D):
        f += np.sum(c1[None, :] * np.cos(c2[None, :] * (x[:, i:i+1] + 0.5)), axis = 1)
    return f + c

def F8F2(x):
    """
    # Introduction:
    Auxiliary function for FEF8F2.
    # Args:
    - `x` (np.ndarray) : A batch of solutions for evaluation.
    # Returns:
    - np.ndarray: The evaluation results.
    """
    f2 = 100.0 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (1.0 - x[:, 0]) ** 2
    return 1.0 + (f2 ** 2) / 4000.0 - np.cos(f2)

# FEF8F2 function
def FEF8F2(x):
    """
    # Introduction:
    Expanded Griewank’s plus Rosenbrock’s function (EF8F2), one of basic functions for the composition function.
    # Args:
    - `x` (np.ndarray) : A batch of solutions for evaluation.
    # Returns:
    - np.ndarray: The evaluation results.
    """
    D = x.shape[1]
    f = np.zeros(x.shape[0])
    for i in range(D - 1):
        f += F8F2(x[:, [i, i + 1]] + 1)
    f += F8F2(x[:, [D - 1, 0]] + 1)
    return f



class F1(CEC2013MMO_Numpy_Problem): # five_uneven_peak_trap
    """
    # Introduction:
    The first test function: Five-Uneven-Peak Trap.
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F1, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'five_uneven_peak_trap'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        
        x = np.asarray(x)
        assert x.shape[1] == self.dim

        self.FES += x.shape[0]
        x = x[:, 0]
        result = np.zeros_like(x)
        
        mask1 = (x >= 0) & (x < 2.50)
        mask2 = (x >= 2.5) & (x < 5)
        mask3 = (x >= 5.0) & (x < 7.5)
        mask4 = (x >= 7.5) & (x < 12.5)
        mask5 = (x >= 12.5) & (x < 17.5)
        mask6 = (x >= 17.5) & (x < 22.5)
        mask7 = (x >= 22.5) & (x < 27.5)
        mask8 = (x >= 27.5) & (x <= 30)
        
        result[mask1] = 80 * (2.5 - x[mask1])
        result[mask2] = 64 * (x[mask2] - 2.5)
        result[mask3] = 64 * (7.5 - x[mask3])
        result[mask4] = 28 * (x[mask4] - 7.5)
        result[mask5] = 28 * (17.5 - x[mask5])
        result[mask6] = 32 * (x[mask6] - 17.5)
        result[mask7] = 32 * (27.5 - x[mask7])
        result[mask8] = 80 * (x[mask8] - 27.5)
        
        return -result

class F2(CEC2013MMO_Numpy_Problem): # 
    """
    # Introduction:
    The second test function: equal_maxima.
    """

    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F2, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'equal_maxima'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -np.sin(5.0 * np.pi * x[:, 0]) ** 6

class F3(CEC2013MMO_Numpy_Problem): # 
    """
    # Introduction:
    The third test function: uneven_decreasing_maxima
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F3, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'uneven_decreasing_maxima'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -(
            np.exp(-2.0 * np.log(2) * ((x[:, 0] - 0.08) / 0.854) ** 2)
            * (np.sin(5 * np.pi * (x[:, 0] ** 0.75 - 0.05))) ** 6
        )
        
class F4(CEC2013MMO_Numpy_Problem): # himmelblau
    """
    # Introduction:
    The 4th test function: himmelblau
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F4, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'himmelblau'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        result = 200 - (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 - (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
        return -result

class F5(CEC2013MMO_Numpy_Problem): # six_hump_camel_back
    """
    # Introduction:
    The 5th test function: six_hump_camel_back
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F5, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'six_hump_camel_back'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        x2 = x[:, 0] ** 2
        x4 = x[:, 0] ** 4
        y2 = x[:, 1] ** 2
        expr1 = (4.0 - 2.1 * x2 + x4 / 3.0) * x2
        expr2 = x[:, 0] * x[:, 1]
        expr3 = (4.0 * y2 - 4.0) * y2
        return -(-1.0 * (expr1 + expr2 + expr3))

class F6(CEC2013MMO_Numpy_Problem): # 
    """
    # Introduction:
    The 6th test function: shubert
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F6, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'shubert'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        result = np.ones(n)
        soma = np.zeros((n, D))
        
        for j in range(1, 6):
            soma = soma + (j * np.cos((j + 1) * x + j))
        result = np.prod(soma, axis = 1)

        return -(-result)

class F7(CEC2013MMO_Numpy_Problem): # 
    """
    # Introduction:
    The 7th test function: vincent
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F7, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'vincent'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        result = np.zeros(n)

        result = np.sum((np.sin(10 * np.log(x))) / D, axis = 1)
        return -result

class F8(CEC2013MMO_Numpy_Problem): # 
    """
    # Introduction:
    The 8th test function: modified_rastrigin_all
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        """
        super(F8, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'modified_rastrigin_all'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        if x is None:
            return None
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        
        if D == 2:
            k = [3, 4]
        elif D == 8:
            k = [1, 2, 1, 2, 1, 3, 1, 4]
        elif D == 16:
            k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]

        result = np.sum(10 + 9 * np.cos(2 * math.pi * np.array(k)[None, :] * x), axis=1)
        return -(-result)

class F9(CFunction): # CF1
    """
    # Introduction:
    The 9th test function: Composition function 1.
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        # Attibutes:
        - `_CFunction__sigma_` (np.ndarray): The __sigma_ attribute in the father class 'CFunction'.
        - `_CFunction__bias_`(np.ndarray): The __bias_ attribute in the father class 'CFunction'.
        - `_CFunction__weight_`(np.ndarray): The __weight_ attribute in the father class 'CFunction'.
        - `_CFunction__lambda_`(np.ndarray): The __lambda_ attribute in the father class 'CFunction'.
        - `_CFunction__O_`(np.ndarray): The __O_ attribute in the father class 'CFunction'.
        - `_CFunction__M_`(list): The __M_ attribute in the father class 'CFunction'.
        - `_CFunction__function_`(dict): The __function_ attribute in the father class 'CFunction'.
        """
        super(F9, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

        # Initialize data for composition
        self._CFunction__sigma_ = np.ones(self._CFunction__nofunc_)
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array([1.0, 1.0, 8.0, 8.0, 1.0 / 5.0, 1.0 / 5.0])

        try:
            folder_package = 'metaevobox.environment.problem.MMO.CEC2013MMO.datafile'
            if importlib.util.find_spec(folder_package) is not None:
                file_path = pkg_resources.files(folder_package).joinpath('optima.dat')
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_file_path = os.path.join(base_path, "datafile", 'optima.dat')
            file_obj = open(local_file_path, 'r')

        with file_obj as f:
            o = np.loadtxt(f)

        # o = np.loadtxt(path.join(path.dirname(__file__), 'datafile') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # M_: Identity matrices
        self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FGrienwank,
            1: FGrienwank,
            2: FWeierstrass,
            3: FWeierstrass,
            4: FSphere,
            5: FSphere,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'CF1'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

class F10(CFunction): # CF2
    """
    # Introduction:
    The 10th test function: Composition function 2.
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        # Attibutes:
        - `_CFunction__sigma_` (np.ndarray): The __sigma_ attribute in the father class 'CFunction'.
        - `_CFunction__bias_`(np.ndarray): The __bias_ attribute in the father class 'CFunction'.
        - `_CFunction__weight_`(np.ndarray): The __weight_ attribute in the father class 'CFunction'.
        - `_CFunction__lambda_`(np.ndarray): The __lambda_ attribute in the father class 'CFunction'.
        - `_CFunction__O_`(np.ndarray): The __O_ attribute in the father class 'CFunction'.
        - `_CFunction__M_`(list): The __M_ attribute in the father class 'CFunction'.
        - `_CFunction__function_`(dict): The __function_ attribute in the father class 'CFunction'.
        """
        super(F10, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

        # Initialize data for composition
        self._CFunction__sigma_ = np.ones(self._CFunction__nofunc_)
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array(
            [1.0, 1.0, 10.0, 10.0, 1.0 / 10.0, 1.0 / 10.0, 1.0 / 7.0, 1.0 / 7.0]
        )

        try:
            folder_package = 'metaevobox.environment.problem.MMO.CEC2013MMO.datafile'
            if importlib.util.find_spec(folder_package) is not None:
                file_path = pkg_resources.files(folder_package).joinpath('optima.dat')
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_file_path = os.path.join(base_path, "datafile", 'optima.dat')
            file_obj = open(local_file_path, 'r')

        with file_obj as f:
            o = np.loadtxt(f)

        # o = np.loadtxt(path.join(path.dirname(__file__), 'datafile') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # M_: Identity matrices
        self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FRastrigin,
            1: FRastrigin,
            2: FWeierstrass,
            3: FWeierstrass,
            4: FGrienwank,
            5: FGrienwank,
            6: FSphere,
            7: FSphere,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'CF2'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

class F11(CFunction): # CF3
    """
    # Introduction:
    The 11th test function: Composition function 3.
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        # Attibutes:
        - `_CFunction__sigma_` (np.ndarray): The __sigma_ attribute in the father class 'CFunction'.
        - `_CFunction__bias_`(np.ndarray): The __bias_ attribute in the father class 'CFunction'.
        - `_CFunction__weight_`(np.ndarray): The __weight_ attribute in the father class 'CFunction'.
        - `_CFunction__lambda_`(np.ndarray): The __lambda_ attribute in the father class 'CFunction'.
        - `_CFunction__O_`(np.ndarray): The __O_ attribute in the father class 'CFunction'.
        - `_CFunction__M_`(list): The __M_ attribute in the father class 'CFunction'.
        - `_CFunction__function_`(dict): The __function_ attribute in the father class 'CFunction'.
        """
        super(F11, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

        # Initialize data for composition
        self._CFunction__sigma_ = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array([1.0 / 4.0, 1.0 / 10.0, 2.0, 1.0, 2.0, 5.0])

        try:
            folder_package = 'metaevobox.environment.problem.MMO.CEC2013MMO.datafile'
            if importlib.util.find_spec(folder_package) is not None:
                file_path = pkg_resources.files(folder_package).joinpath('optima.dat')
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_file_path = os.path.join(base_path, "datafile", 'optima.dat')
            file_obj = open(local_file_path, 'r')

        with file_obj as f:
            o = np.loadtxt(f)

        # o = np.loadtxt(path.join(path.dirname(__file__), 'datafile') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # Load M_: Rotation matrices
        if dim == 2 or dim == 3 or dim == 5 or dim == 10 or dim == 20:
            try:
                folder_package = 'metaevobox.environment.problem.MMO.CEC2013MMO.datafile'
                if importlib.util.find_spec(folder_package) is not None:
                    file_path = pkg_resources.files(folder_package).joinpath(f'CF3_M_D{dim}.dat')
                    file_obj = file_path.open('r')
                else:
                    raise ModuleNotFoundError
            except ModuleNotFoundError:
                base_path = os.path.dirname(os.path.abspath(__file__))
                local_file_path = os.path.join(base_path, "datafile", f'CF3_M_D{dim}.dat')
                file_obj = open(local_file_path, 'r')

            self._CFunction__load_rotmat(file_obj)
        else:
            # M_ Identity matrices 
            self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FEF8F2,
            1: FEF8F2,
            2: FWeierstrass,
            3: FWeierstrass,
            4: FGrienwank,
            5: FGrienwank,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'CF3'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

class F12(CFunction): # CF4
    """
    # Introduction:
    The 12th test function: Composition function 4.
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        """
        # Introduction:
        Initialization the test function.
        # Args:
        - `dim` (int): Dimensionality of the problem.
        - `lb` (float): Lower bound of the search space.
        - `ub` (float): Upper bound of the search space.
        - `fopt` (float): The optimal fitness value for the problem.
        - `rho` (float): Radius used to determine proximity for seed identification.
        - `nopt` (int): Number of global optima in the problem.
        - `maxfes` (int): Maximum number of function evaluations allowed.
        # Attibutes:
        - `_CFunction__sigma_` (np.ndarray): The __sigma_ attribute in the father class 'CFunction'.
        - `_CFunction__bias_`(np.ndarray): The __bias_ attribute in the father class 'CFunction'.
        - `_CFunction__weight_`(np.ndarray): The __weight_ attribute in the father class 'CFunction'.
        - `_CFunction__lambda_`(np.ndarray): The __lambda_ attribute in the father class 'CFunction'.
        - `_CFunction__O_`(np.ndarray): The __O_ attribute in the father class 'CFunction'.
        - `_CFunction__M_`(list): The __M_ attribute in the father class 'CFunction'.
        - `_CFunction__function_`(dict): The __function_ attribute in the father class 'CFunction'.
        """
        super(F12, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

        # Initialize data for composition
        self._CFunction__sigma_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self._CFunction__bias_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__weight_ = np.zeros(self._CFunction__nofunc_)
        self._CFunction__lambda_ = np.array(
            [4.0, 1.0, 4.0, 1.0, 1.0 / 10.0, 1.0 / 5.0, 1.0 / 10.0, 1.0 / 40.0]
        )

        try:
            folder_package = 'metaevobox.environment.problem.MMO.CEC2013MMO.datafile'
            if importlib.util.find_spec(folder_package) is not None:
                file_path = pkg_resources.files(folder_package).joinpath('optima.dat')
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_file_path = os.path.join(base_path, "datafile", 'optima.dat')
            file_obj = open(local_file_path, 'r')

        with file_obj as f:
            o = np.loadtxt(f)

        # o = np.loadtxt(path.join(path.dirname(__file__), 'datafile') + "/optima.dat")
        if o.shape[1] >= dim:
            self._CFunction__O_ = o[: self._CFunction__nofunc_, :dim]
        else:  # randomly initialize
            self._CFunction__O_ = self.lb + (
                self.ub - self.lb
            ) * np.random.rand((self._CFunction__nofunc_, dim))

        # Load M_: Rotation matrices
        if dim == 2 or dim == 3 or dim == 5 or dim == 10 or dim == 20:
            try:
                folder_package = 'metaevobox.environment.problem.MMO.CEC2013MMO.datafile'
                if importlib.util.find_spec(folder_package) is not None:
                    file_path = pkg_resources.files(folder_package).joinpath(f'CF4_M_D{dim}.dat')
                    file_obj = file_path.open('r')
                else:
                    raise ModuleNotFoundError
            except ModuleNotFoundError:
                base_path = os.path.dirname(os.path.abspath(__file__))
                local_file_path = os.path.join(base_path, "datafile", f'CF4_M_D{dim}.dat')
                file_obj = open(local_file_path, 'r')

            # fname = path.join(path.dirname(__file__), 'datafile') + "/CF4_M_D" + str(dim) + ".dat"
            self._CFunction__load_rotmat(file_obj)
        else:
            # M_ Identity matrices
            self._CFunction__M_ = [np.eye(dim)] * self._CFunction__nofunc_

        # Initialize functions of the composition
        self._CFunction__function_ = {
            0: FRastrigin,
            1: FRastrigin,
            2: FEF8F2,
            3: FEF8F2,
            4: FWeierstrass,
            5: FWeierstrass,
            6: FGrienwank,
            7: FGrienwank,
        }

        # Calculate fmaxi
        self._CFunction__calculate_fmaxi()

    def __str__(self):
        """
        Returns a string representation of the object.
        # Returns:
        - str: The name with the dimension.
        """
        return 'CF4'+'_D'+str(self.dim)

    def func(self, x):
        """
        # Introduction:
        Evaluate the inputed solutions.
        # Args:
        - `x` (np.ndarray) : A group of solutions for evaluation.
        # Returns:
        - np.ndarray: The evaluation results.
        """
        x = np.asarray(x)
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)











