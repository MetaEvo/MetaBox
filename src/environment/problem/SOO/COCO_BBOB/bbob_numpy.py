from ....problem.basic_problem import Basic_Problem
import numpy as np


class BBOB_Numpy_Problem(Basic_Problem):
    """
    # Introduction
    BBOB-Surrogate investigates the integration of surrogate modeling techniques into MetaBBO , enabling data-driven approximation of expensive objective functions while maintaining optimization fidelity.
    # Original paper
    "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
    # Official Implementation
    [BBOB-Surrogate](https://github.com/GMC-DRL/Surr-RLDE)
    # License
    None
    # Problem Suite Composition
    BBOB-Surrogate contains a total of 72 optimization problems, corresponding to three dimensions (2, 5, 10), each dimension contains 24 problems. Each problem consists of a trained KAN or MLP network, which is used to fit 24 black box functions in the COCO-BBOB benchmark. The network here is a surrogate model of the original function.
    """
    
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        """
        # Introduction
        Initializes the problem instance with given parameters such as dimensionality, shift, rotation, bias, and bounds.
        # Args:
        - dim (int): Dimensionality of the problem.
        - shift (np.ndarray): Shift vector applied to the input space.
        - rotate (np.ndarray): Rotation matrix applied to the input space.
        - bias (float): Bias value added to the function output.
        - lb (float or np.ndarray): Lower bound(s) of the search space.
        - ub (float or np.ndarray): Upper bound(s) of the search space.
        # Built-in Attribute:
        - dim (int): Problem dimensionality.
        - shift (np.ndarray): Shift vector.
        - rotate (np.ndarray): Rotation matrix.
        - bias (float): Bias value.
        - lb (float or np.ndarray): Lower bound(s).
        - ub (float or np.ndarray): Upper bound(s).
        - FES (int): Function evaluation count, initialized to 0.
        - opt (np.ndarray): Optimal solution, set to the shift vector.
        - optimum (float): Function value at the optimal solution.
        # Returns:
        - None
        """
        self.dim = dim
        self.shift = shift
        self.rotate = rotate
        self.bias = bias
        self.lb = lb
        self.ub = ub
        self.FES = 0
        self.opt = self.shift
        # self.optimum = self.eval(self.get_optimal())
        self.optimum = self.func(self.get_optimal().reshape(1, -1))[0]

    def get_optimal(self):
        """
        # Introduction
        Returns the optimal value or solution associated with the current problem instance.
        # Returns:
        - Any: The optimal value or solution stored in the `opt` attribute.
        """
        
        return self.opt

    def func(self, x):
        """
        # Introduction
        Evaluates the objective function at the given input vector `x`.
        # Args:
        - x (numpy.ndarray): Input vector at which the function should be evaluated.
        # Returns:
        - float: The computed value of the objective function at `x`.
        # Raises:
        - NotImplementedError: This method must be implemented by subclasses.
        """
        
        raise NotImplementedError

class NoisyProblem:
    """
    # Introduction
    Represents a noisy optimization problem, providing methods to evaluate the true and noisy objective values, as well as handle boundary constraints.
    """
    
    def noisy(self, ftrue):
        raise NotImplementedError

    def eval(self, x):
        ftrue = super().eval(x)
        return self.noisy(ftrue)

    def boundaryHandling(self, x):
        return 100. * pen_func(x, self.ub)

class GaussNoisyProblem(NoisyProblem):
    """
    # Introduction
    Represents a Gaussian noisy optimization problem, where noise is multiplicatively applied to the unbiased function value using a Gaussian distribution. This class is intended to be subclassed, with the attribute `gauss_beta` defined in the subclass to control the noise intensity.
    """

    def noisy(self, ftrue):
        """
        Adds Gaussian noise to the true function value(s) to simulate a noisy optimization environment.
        # Args:
        - ftrue (float or np.ndarray): The true (noise-free) function value(s) to which noise will be added.
        # Returns:
        - np.ndarray: The noisy function value(s), with Gaussian noise applied multiplicatively to the unbiased value(s).
          If the unbiased value is less than 1e-8, returns the original value with a small offset.
        # Notes:
        - The noise is applied as: `fnoisy_unbiased = ftrue_unbiased * exp(gauss_beta * N(0,1))`, where `N(0,1)` is standard normal noise.
        - The function ensures that very small unbiased values (less than 1e-8) are not perturbed by noise.
        """
        
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * np.exp(self.gauss_beta * np.random.randn(*ftrue_unbiased.shape))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)

class UniformNoisyProblem(NoisyProblem):
    """
    # Introduction
    Represents a noisy optimization problem where uniform noise is applied to the true function value.
    The noise is controlled by the `uniform_alpha` and `uniform_beta` attributes, which must be defined in subclasses.
    """
    def noisy(self, ftrue):
        """
        # Introduction
        Applies a specific noise model to the true function value(s) `ftrue`, simulating noisy objective function evaluations as used in the BBOB (Black-Box Optimization Benchmarking) testbed.
        # Args:
        - ftrue (float or np.ndarray): The true (unbiased) function value(s) to which noise should be applied.
        # Built-in Attribute:
        - self.optimum (float): The bias (optimum value) of the function.
        - self.uniform_beta (float): Parameter controlling the strength of the uniform noise.
        - self.uniform_alpha (float): Parameter controlling the scaling of the noise.
        - self.dim (int): Dimensionality of the problem.
        # Returns:
        - np.ndarray: The noisy function value(s), with noise applied according to the BBOB noise model. For values of `ftrue` close to the optimum, the original value is returned.
        # Notes:
        - For `ftrue` values very close to the optimum (within 1e-8), no noise is added.
        - The noise model is multiplicative and depends on the distance from the optimum and random uniform variables.
        """
        
        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased * (np.random.rand(*ftrue_unbiased.shape) ** self.uniform_beta) * \
                          np.maximum(1., (1e9 / (ftrue_unbiased + 1e-99)) ** (
                                      self.uniform_alpha * (0.49 + 1. / self.dim) * np.random.rand(
                                  *ftrue_unbiased.shape)))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)

class CauchyNoisyProblem(NoisyProblem):
    """
    # Introduction
    Represents a noisy optimization problem where Cauchy-distributed noise is added to the true function value. 
    This class is typically used in benchmarking optimization algorithms under noisy conditions.
    """

    def noisy(self, ftrue):
        """
        # Introduction
        Adds Cauchy noise to the true function value(s) to simulate a noisy optimization problem.
        # Args:
        - ftrue (float or np.ndarray): The true (noise-free) function value(s) to which noise will be added.
        # Returns:
        - np.ndarray: The noisy function value(s), with Cauchy noise applied, except for values close to the optimum.
        # Notes:
        - The noise is only added to values sufficiently far from the optimum (i.e., where `ftrue_unbiased >= 1e-8`).
        - For values very close to the optimum, the original value is returned to avoid numerical issues.
        - The noise is generated using a Cauchy-like distribution, controlled by `self.cauchy_alpha` and `self.cauchy_p`.
        """

        if not isinstance(ftrue, np.ndarray):
            ftrue = np.array(ftrue)
        bias = self.optimum
        ftrue_unbiased = ftrue - bias
        fnoisy_unbiased = ftrue_unbiased + self.cauchy_alpha * np.maximum(0.,
                                                                          1e3 + (np.random.rand(
                                                                              *ftrue_unbiased.shape) < self.cauchy_p) * np.random.randn(
                                                                              *ftrue_unbiased.shape) / (np.abs(
                                                                              np.random.randn(
                                                                                  *ftrue_unbiased.shape)) + 1e-199))
        return np.where(ftrue_unbiased >= 1e-8, fnoisy_unbiased + bias + 1.01 * 1e-8, ftrue)

class _Sphere(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the abstract Sphere function for benchmarking optimization algorithms.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Methods:
    - func(x): Evaluates the shifted, rotated, and biased Sphere function on input `x` with boundary handling.
    # Returns:
    - float or np.ndarray: The evaluated Sphere function value(s) for the given input(s).
    # Raises:
    - None explicitly, but may raise exceptions from underlying numpy operations or if input shapes are incompatible.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        """
        Initializes the object with the given parameters for dimension, shift, rotation, bias, and bounds.
        # Args:
        - dim (int): The dimensionality of the problem.
        - shift (np.ndarray): The shift vector applied to the input.
        - rotate (np.ndarray): The rotation matrix applied to the input.
        - bias (float): The bias value added to the objective function.
        - lb (float or np.ndarray): The lower bound(s) for the input variables.
        - ub (float or np.ndarray): The upper bound(s) for the input variables.
        # Raises:
        - ValueError: If the dimensions of shift or rotate do not match `dim`.
        """

        super().__init__(dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        """
        Evaluates the objective function for a given input vector `x` in the context of a shifted and rotated BBOB benchmark problem.
        # Args:
        - x (np.ndarray): Input solution vector(s) of shape (..., D), where D is the problem dimension.
        # Returns:
        - np.ndarray: The computed objective value(s) after applying shift, rotation, boundary handling, and bias.
        # Side Effects:
        - Increments the function evaluation counter `self.FES` by the number of evaluated solutions.
        # Notes:
        - The function applies a shift and rotation transformation to `x` using `sr_func`.
        - The result is the sum of squares of the transformed vector, plus a bias and boundary handling penalty.
        """
        
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        return np.sum(z ** 2, axis=-1) + self.bias + self.boundaryHandling(x)

def sr_func(x, Os, Mr):  # shift and rotate
    """
    # Introduction
    Applies a shift and rotation transformation to the input vector(s).
    # Args:
    - x (np.ndarray): Input array of shape (n_samples, n_features), representing the data points to be transformed.
    - Os (np.ndarray): Shift vector of shape (n_features,), used to shift the input.
    - Mr (np.ndarray): Rotation matrix of shape (n_features, n_features), used to rotate the shifted input.
    # Returns:
    - np.ndarray: The shifted and rotated input array of shape (n_samples, n_features).
    # Raises:
    - ValueError: If the shapes of `x`, `Os`, or `Mr` are incompatible for the operations.
    """
    
    y = x[:, :Os.shape[-1]] - Os
    return np.matmul(Mr, y.transpose()).transpose()


def rotate_gen(dim):  # Generate a rotate matrix
    """
    Generates a random orthogonal rotation matrix of the specified dimension using Householder transformations.
    # Args:
    - dim (int): The dimension of the rotation matrix to generate.
    # Returns:
    - np.ndarray: A (dim, dim) orthogonal rotation matrix with determinant 1.
    # Notes:
    - The function uses Householder transformations to construct the rotation matrix.
    - The resulting matrix is suitable for use in optimization benchmarks such as COCO/BBOB.
    """
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        mat = np.eye(dim)
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
    # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H

def osc_transform(x):
    """
    # Introduction
    Applies an oscillating transformation to the input array, modifying its values based on their sign using logarithmic, exponential, and sinusoidal operations. This transformation is commonly used in benchmarking optimization problems to introduce non-linearity and complexity.
    # Args:
    - x (np.ndarray): Input array. If representing objective values, should be 1-D with shape [NP] for single-objective or 2-D with shape [NP, number_of_objectives] for multi-objective problems. If representing decision values, should be 2-D with shape [NP, dim].
    # Returns:
    - np.ndarray: Transformed array with the same shape as the input.
    # Notes:
    - Positive and negative values in `x` are transformed differently.
    - The transformation is element-wise and preserves the shape of the input.
    """
    y = x.copy()
    idx = (x > 0.)
    y[idx] = np.log(x[idx]) / 0.1
    y[idx] = np.exp(y[idx] + 0.49 * (np.sin(y[idx]) + np.sin(0.79 * y[idx]))) ** 0.1
    idx = (x < 0.)
    y[idx] = np.log(-x[idx]) / 0.1
    y[idx] = -np.exp(y[idx] + 0.49 * (np.sin(0.55 * y[idx]) + np.sin(0.31 * y[idx]))) ** 0.1
    return y


def asy_transform(x, beta):
    """
    # Introduction
    Applies an asymmetric transformation to the decision variable array, typically used in optimization benchmarking to introduce non-symmetry into the problem landscape.
    # Args:
    - x (np.ndarray): Decision values with shape [NP, dim], where NP is the number of points and dim is the dimensionality.
    - beta (float): Asymmetry factor controlling the degree of transformation.
    # Returns:
    - np.ndarray: Transformed array with the same shape as `x`, after applying the asymmetric transformation.
    # Notes:
    - The transformation is only applied to positive elements of `x`.
    - The transformation uses a position-dependent exponent based on `beta` and the variable index.
    """
    NP, dim = x.shape
    idx = (x > 0.)
    y = x.copy()
    y[idx] = y[idx] ** (1. + beta * np.linspace(0, 1, dim).reshape(1, -1).repeat(repeats=NP, axis=0)[idx] * np.sqrt(y[idx]))
    return y


def pen_func(x, ub):
    """
    # Introduction
    Computes a penalty value for decision variables that exceed a specified upper bound.
    # Args:
    - x (np.ndarray): Decision values with shape [NP, dim].
    - ub (float): The upper bound for each decision variable.
    # Returns:
    - np.ndarray: Penalty values for each solution in shape [NP].
    # Details:
    For each element in `x`, if its absolute value exceeds `ub`, the squared excess is added to the penalty. The penalties are summed across the last axis for each solution.
    """
    return np.sum(np.maximum(0., np.abs(x) - ub) ** 2, axis=-1)

class F1(_Sphere):
    """
    # Introduction
    Represents the Sphere function (F1) for single-objective optimization problems, inheriting from the `_Sphere` class..
    """
    
    def boundaryHandling(self, x):
        """
        # Introduction
        Handles boundary constraints for the input vector `x` in the optimization problem.
        # Args:
        - x (numpy.ndarray): The input vector to be checked or adjusted for boundary constraints.
        # Returns:
        - float: A penalty value or adjustment result based on the boundary handling logic.
        # Raises:
        - None
        """
        return 0.

    def __str__(self):
        """
        # Introduction
        Returns the string representation of the object, which is 'Sphere'.
        # Returns:
        - str: The string 'Sphere' representing the object.
        """
        return 'Sphere'

class F101(GaussNoisyProblem, _Sphere):
    """
    # Introduction
    Represents the BBOB Sphere function with moderate Gaussian noise, used as a single-objective optimization problem in benchmarking.
    # Inherits From:
    - GaussNoisyProblem: Adds Gaussian noise to the objective function.
    - _Sphere: Implements the noiseless Sphere function.
    # Class Attributes:
    - gauss_beta (float): Standard deviation of the Gaussian noise applied to the function (default: 0.01).
    # Methods:
    - __str__(): Returns a string identifier for the problem instance ("Sphere_moderate_gauss").
    # Usage:
    Instantiate this class to create a noisy Sphere optimization problem for benchmarking algorithms.
    """
    
    gauss_beta = 0.01
    def __str__(self):
        return 'Sphere_moderate_gauss'

class F102(UniformNoisyProblem, _Sphere):
    """
    # Introduction
    Represents the Sphere function with moderate uniform noise, as defined in the COCO BBOB test suite. Inherits from `UniformNoisyProblem` and `_Sphere` to provide a noisy optimization problem for benchmarking.
    # Class Attributes:
    - uniform_alpha (float): The alpha parameter controlling the level of uniform noise (default: 0.01).
    - uniform_beta (float): The beta parameter controlling the level of uniform noise (default: 0.01).
    # Methods:
    - __str__(): Returns a string representation of the problem ("Sphere_moderate_uniform").
    # Inheritance:
    - UniformNoisyProblem: Base class for problems with uniform noise.
    - _Sphere: Base class implementing the Sphere function.
    """
    uniform_alpha = 0.01
    uniform_beta = 0.01
    def __str__(self):
        return 'Sphere_moderate_uniform'

class F103(CauchyNoisyProblem, _Sphere):
    """
    # Introduction
    Represents the BBOB Sphere function with moderate Cauchy noise, used for single-objective optimization benchmarking.
    # Inheritance:
    - Inherits from `CauchyNoisyProblem` and `_Sphere`.
    # Class Attributes:
    - cauchy_alpha (float): Scale parameter for the Cauchy noise (default: 0.01).
    - cauchy_p (float): Probability of applying Cauchy noise (default: 0.05).
    # Methods:
    - __str__(): Returns a string representation of the problem ("Sphere_moderate_cauchy").
    # Usage:
    Instantiate this class to create a noisy Sphere optimization problem for benchmarking algorithms.
    """
    cauchy_alpha = 0.01
    cauchy_p = 0.05
    def __str__(self):
        return 'Sphere_moderate_cauchy'

class F107(GaussNoisyProblem, _Sphere):
    """
    # Introduction
    Represents the noisy Sphere function (BBOB function F107) with Gaussian noise, used for benchmarking optimization algorithms.
    # Inherits From:
    - GaussNoisyProblem: Adds Gaussian noise to the objective function.
    - _Sphere: Implements the Sphere function.
    # Class Attributes:
    - gauss_beta (float): The standard deviation of the Gaussian noise applied to the function (default is 1.0).
    # Methods:
    - __str__(): Returns the string representation 'Sphere_gauss'.
    # Usage:
    Typically used as part of a suite of single-objective optimization problems for algorithm benchmarking.
    """
    gauss_beta = 1.
    def __str__(self):
        return 'Sphere_gauss'

class F108(UniformNoisyProblem, _Sphere):
    """
    # Introduction
    Represents the BBOB Sphere function (F108) with uniform noise, used as a single-objective optimization problem in benchmarking.
    # Inherits From:
    - UniformNoisyProblem: Adds uniform noise to the objective function.
    - _Sphere: Implements the standard Sphere function.
    # Class Attributes:
    - uniform_alpha (float): The alpha parameter for the uniform noise distribution (default: 1.0).
    - uniform_beta (float): The beta parameter for the uniform noise distribution (default: 1.0).
    # Methods:
    - __str__(): Returns the string representation 'Sphere_uniform'.
    # Usage:
    Typically used in benchmarking optimization algorithms with noisy objective functions.
    """
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Sphere_uniform'

class F109(CauchyNoisyProblem, _Sphere):
    """
    # Introduction
    Represents the Sphere function with additive Cauchy noise, as defined in the BBOB COCO benchmark suite. Inherits from `CauchyNoisyProblem` and `_Sphere`, applying Cauchy-distributed noise to the standard Sphere problem.
    # Class Attributes:
    - cauchy_alpha (float): Scale parameter for the Cauchy noise distribution (default: 1.0).
    - cauchy_p (float): Probability of applying Cauchy noise to the function evaluation (default: 0.2).
    # Methods:
    - __str__(): Returns the string identifier for this problem variant ("Sphere_cauchy").
    # Inheritance:
    - CauchyNoisyProblem: Base class for problems with Cauchy noise.
    - _Sphere: Base class for the Sphere benchmark function.
    """
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Sphere_cauchy'

class F2(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Ellipsoidal function (BBOB F2) for single-objective optimization problems, as defined in the COCO BBOB benchmark suite. This class applies a shift, rotation, and oscillation transformation to the input and evaluates the Ellipsoidal function.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Methods:
    - __str__(): Returns the name of the function ("Ellipsoidal").
    - func(x): Evaluates the Ellipsoidal function on the input `x`.
    # Returns (func):
    - float or np.ndarray: The computed value(s) of the Ellipsoidal function for the given input(s).
    # Raises:
    - ValueError: If input dimensions do not match the problem's dimensionality.
    """

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Ellipsoidal'

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        i = np.arange(nx)
        return np.sum(np.power(10, 6 * i / (nx - 1)) * (z ** 2), -1) + self.bias

class F3(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Rastrigin function (BBOB F3) for benchmarking optimization algorithms, with support for shifting, rotation, scaling, and biasing in a numpy-based environment.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    # Methods:
    - __str__(): Returns the name of the function ("Rastrigin").
    - func(x): Evaluates the Rastrigin function on the input `x` with transformations.
    # Returns (func):
    - np.ndarray: The computed Rastrigin function values for the input(s) `x`.
    # Attributes:
    - scales (np.ndarray): Scaling factors for each dimension.
    - FES (int): Function evaluation count (incremented on each call to `func`).
    # Raises:
    - ValueError: If input dimensions do not match the problem's dimensionality.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rastrigin'

    def func(self, x):
        self.FES += x.shape[0]
        z = self.scales * asy_transform(osc_transform(sr_func(x, self.shift, self.rotate)), beta=0.2)
        return 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + self.bias

class F4(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Buche-Rastrigin function (BBOB F4) as a single-objective optimization problem, inheriting from `BBOB_Numpy_Problem`. This class applies specific transformations and scaling to the input vector and computes the function value according to the BBOB benchmark definition.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Attributes:
    - scales (np.ndarray): Scaling factors applied to the input vector.
    # Methods:
    - __str__(): Returns the name of the function ("Buche_Rastrigin").
    - func(x): Evaluates the Buche-Rastrigin function on the input `x`.
    # Returns:
    - func(x): Returns the computed function value(s) as a float or np.ndarray.
    # Raises:
    - None explicitly, but may raise exceptions from underlying numpy operations if input shapes are invalid.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift[::2] = np.abs(shift[::2])
        self.scales = ((10. ** 0.5) ** np.linspace(0, 1, dim))
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Buche_Rastrigin'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        even = z[:, ::2]
        even[even > 0.] *= 10.
        z *= self.scales
        return 10 * (self.dim - np.sum(np.cos(2 * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + 100 * pen_func(x, self.ub) + self.bias

class F5(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the BBOB Linear Slope function (F5) for single-objective optimization benchmarking, implemented using NumPy. This class is a specific problem instance used in black-box optimization benchmarking.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector for the problem, used to displace the optimum.
    - rotate (np.ndarray): Rotation matrix (not used in this function, but kept for interface consistency).
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    # Built-in Attribute:
    - FES (int): Function evaluation count, incremented with each call to `func`.
    - shift (np.ndarray): The shifted optimum location, adjusted to avoid zeros.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    - bias (float): Bias added to the function value.
    - dim (int): Dimensionality of the problem.
    # Methods:
    - __str__(): Returns the name of the function ("Linear_Slope").
    - func(x): Evaluates the Linear Slope function at the given input(s) `x`.
    # Returns:
    - func(x): Returns the function value(s) as a float or np.ndarray, depending on the input shape.
    # Raises:
    - None explicitly, but may raise exceptions if input shapes are incompatible or if NumPy operations fail.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift = np.sign(shift)
        shift[shift == 0.] = np.random.choice([-1., 1.], size=(shift == 0.).sum())
        shift = shift * ub
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Linear_Slope'

    def func(self, x):
        self.FES += x.shape[0]
        z = x.copy()
        exceed_bound = (x * self.shift) > (self.ub ** 2)
        z[exceed_bound] = np.sign(z[exceed_bound]) * self.ub  # clamp back into the domain
        s = np.sign(self.shift) * (10 ** np.linspace(0, 1, self.dim))
        return np.sum(self.ub * np.abs(s) - z * s, axis=-1) + self.bias

class F6(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Attractive Sector function (BBOB F6) for single-objective optimization, as part of the COCO BBOB benchmark suite. This class defines the problem's initialization, string representation, and objective function evaluation using numpy.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector for the problem.
    - rotate (np.ndarray): Rotation matrix for the problem.
    - bias (float): Bias added to the objective function value.
    - lb (float or np.ndarray): Lower bound(s) for the decision variables.
    - ub (float or np.ndarray): Upper bound(s) for the decision variables.
    # Raises:
    - None explicitly, but may raise exceptions if input shapes are incompatible or if required dependencies are missing.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Attractive_Sector'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        idx = (z * self.get_optimal()) > 0.
        z[idx] *= 100.
        return osc_transform(np.sum(z ** 2, -1)) ** 0.9 + self.bias

class _Step_Ellipsoidal(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents an abstract Step Ellipsoidal function for single-objective optimization, as part of the BBOB (Black-Box Optimization Benchmarking) suite. This class applies a step transformation to an ellipsoidal function, incorporating shift, rotation, and scaling, and is used to evaluate optimization algorithms on non-trivial, non-separable landscapes.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Constant bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Built-in Attribute:
    - Q_rotate (np.ndarray): Additional rotation matrix generated for the step transformation.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.diag(scales), rotate)
        self.Q_rotate = rotate_gen(dim)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z_hat = sr_func(x, self.shift, self.rotate)
        z = np.matmul(np.where(np.abs(z_hat) > 0.5, np.floor(0.5 + z_hat), np.floor(0.5 + 10. * z_hat) / 10.), self.Q_rotate.T)
        return 0.1 * np.maximum(np.abs(z_hat[:, 0]) / 1e4, np.sum(100 ** np.linspace(0, 1, self.dim) * (z ** 2), axis=-1)) + \
               self.boundaryHandling(x) + self.bias

class F7(_Step_Ellipsoidal):
    """
    # F7: Step Ellipsoidal Function
    Represents the F7 benchmark function (Step Ellipsoidal) from the COCO BBOB suite, used for single-objective optimization benchmarking.
    # Args
    - x (numpy.ndarray): Input vector to be evaluated by the boundary handling method.
    # Returns
    - boundaryHandling(x): float
        The penalized value of the input vector `x` if it exceeds the upper bound.
    - __str__(): str
        The name of the function, 'Step_Ellipsoidal'.
    # Inherits
    - _Step_Ellipsoidal: Base class implementing the core functionality of the step ellipsoidal function.
    """
    
    def boundaryHandling(self, x):
        return pen_func(x, self.ub)

    def __str__(self):
        return 'Step_Ellipsoidal'

class F113(GaussNoisyProblem, _Step_Ellipsoidal):
    """
    # Introduction
    Represents the noisy Step Ellipsoidal function (BBOB F113) with Gaussian noise, used as a single-objective optimization problem in benchmarking.
    # Inherits From:
    - GaussNoisyProblem: Adds Gaussian noise to the objective function.
    - _Step_Ellipsoidal: Implements the step ellipsoidal function logic.
    # Class Attributes:
    - gauss_beta (float): The standard deviation parameter for the Gaussian noise (default is 1.0).
    # Methods:
    - __str__(): Returns a string identifier for the problem ("Step_Ellipsoidal_gauss").
    # Usage:
    Instantiate this class to create a noisy step ellipsoidal optimization problem for benchmarking algorithms.
    """
    gauss_beta = 1.
    def __str__(self):
        return 'Step_Ellipsoidal_gauss'

class F114(UniformNoisyProblem, _Step_Ellipsoidal):
    """
    # Introduction
    Represents the Step Ellipsoidal function with uniform noise as defined in the COCO BBOB suite, implemented as a single-objective optimization problem.
    # Inherits From:
    - UniformNoisyProblem: Adds uniform noise to the objective function.
    - _Step_Ellipsoidal: Provides the step ellipsoidal function definition.
    # Class Attributes:
    - uniform_alpha (float): Scaling factor for the uniform noise (default: 1.0).
    - uniform_beta (float): Scaling factor for the uniform noise (default: 1.0).
    # Methods:
    - __str__(): Returns the string identifier for the problem ("Step_Ellipsoidal_uniform").
    # Usage:
    Instantiate this class to create a noisy step ellipsoidal optimization problem for benchmarking algorithms.
    """
    
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Step_Ellipsoidal_uniform'

class F115(CauchyNoisyProblem, _Step_Ellipsoidal):
    """
    # Introduction
    Represents the Step Ellipsoidal function with Cauchy noise (BBOB function F115) for single-objective optimization benchmarking.
    # Inherits From:
    - CauchyNoisyProblem: Adds Cauchy-distributed noise to the objective function.
    - _Step_Ellipsoidal: Implements the step ellipsoidal function structure.
    # Class Attributes:
    - cauchy_alpha (float): Scale parameter for the Cauchy noise (default: 1.0).
    - cauchy_p (float): Probability of applying Cauchy noise (default: 0.2).
    # Methods:
    - __str__(): Returns the string identifier 'Step_Ellipsoidal_cauchy'.
    # Usage:
    Instantiate this class to create a noisy step ellipsoidal optimization problem for benchmarking algorithms.
    """
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Step_Ellipsoidal_cauchy'

class _Rosenbrock(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the BBOB Rosenbrock function problem for single-objective optimization, with configurable dimension, shift, rotation, bias, and bounds. This class is used to evaluate the shifted and rotated Rosenbrock function as defined in the BBOB benchmark suite.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix (overwritten as identity).
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Built-in Attribute:
    - FES (int): Function evaluation count, incremented on each call to `func`.
    - dim (int): Problem dimensionality.
    - shift (np.ndarray): Shift vector.
    - rotate (np.ndarray): Rotation matrix.
    - bias (float): Bias term.
    - lb (float or np.ndarray): Lower bound(s).
    - ub (float or np.ndarray): Upper bound(s).
    # Methods:
    - func(x): Evaluates the shifted and rotated Rosenbrock function on input `x`.
    # Returns (for func):
    - np.ndarray: The computed Rosenbrock function values for each input in `x`.
    # Raises:
    - ValueError: If input `x` does not have the correct shape or type.
    """
    """
    Abstract Rosenbrock_original
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift *= 0.75  # range_of_shift=0.8*0.75*ub=0.6*ub
        rotate = np.eye(dim)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = max(1., self.dim ** 0.5 / 8.) * sr_func(x, self.shift, self.rotate) + 1
        return np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=-1) + self.bias + self.boundaryHandling(x)

class F8(_Rosenbrock):
    """
    # Introduction
    Represents the original Rosenbrock function (BBOB F8) as a single-objective optimization problem, inheriting from the `_Rosenbrock` base class.
    # Methods
    - boundaryHandling(x): Handles boundary constraints for the input vector `x`. For this implementation, always returns 0.
    - __str__(): Returns the string representation of the function.
    # Args:
    - x (numpy.ndarray): Input vector for the boundary handling method.
    # Returns:
    - float: For `boundaryHandling`, always returns 0.
    - str: For `__str__`, returns 'Rosenbrock_original'.
    # Notes:
    This class is typically used within the COCO BBOB benchmarking framework for optimization algorithms.
    """
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Rosenbrock_original'

class F104(GaussNoisyProblem, _Rosenbrock):
    """
    # Introduction
    Represents the BBOB F104 test function, which is a Rosenbrock function with moderate Gaussian noise, for single-objective optimization benchmarking.
    # Inherits From:
    - GaussNoisyProblem: Base class for problems with Gaussian noise.
    - _Rosenbrock: Implements the Rosenbrock function.
    # Class Attributes:
    - gauss_beta (float): The standard deviation of the Gaussian noise added to the function, set to 0.01.
    # Methods:
    - __str__(): Returns a string representation of the problem, 'Rosenbrock_moderate_gauss'.
    # Usage:
    This class is intended for benchmarking optimization algorithms on noisy variants of the Rosenbrock function as defined in the COCO BBOB suite.
    """
    gauss_beta = 0.01
    def __str__(self):
        return 'Rosenbrock_moderate_gauss'

class F105(UniformNoisyProblem, _Rosenbrock):
    """
    # Introduction
    Represents the Rosenbrock function with moderate uniform noise, as part of the COCO BBOB noisy single-objective optimization problems.
    # Inherits From:
    - UniformNoisyProblem: Adds uniform noise to the objective function.
    - _Rosenbrock: Implements the Rosenbrock benchmark function.
    # Class Attributes:
    - uniform_alpha (float): The alpha parameter controlling the noise level (default: 0.01).
    - uniform_beta (float): The beta parameter controlling the noise level (default: 0.01).
    # Methods:
    - __str__(): Returns a string representation of the problem ("Rosenbrock_moderate_uniform").
    """
    uniform_alpha = 0.01
    uniform_beta = 0.01
    def __str__(self):
        return 'Rosenbrock_moderate_uniform'

class F106(CauchyNoisyProblem, _Rosenbrock):
    """
    # Introduction
    Represents the Rosenbrock function with moderate Cauchy noise as defined in the COCO BBOB suite.
    # Inheritance:
    - Inherits from `CauchyNoisyProblem` and `_Rosenbrock`.
    # Class Attributes:
    - cauchy_alpha (float): Scale parameter for the Cauchy noise (default: 0.01).
    - cauchy_p (float): Probability of applying Cauchy noise (default: 0.05).
    # Methods:
    - __str__(): Returns a string representation of the problem ("Rosenbrock_moderate_cauchy").
    """
    
    cauchy_alpha = 0.01
    cauchy_p = 0.05
    def __str__(self):
        return 'Rosenbrock_moderate_cauchy'

class F110(GaussNoisyProblem, _Rosenbrock):
    """
    # Introduction
    Represents the noisy Rosenbrock function (F110) from the COCO BBOB suite, with Gaussian noise applied to the objective value.
    # Inherits From:
    - GaussNoisyProblem: Adds Gaussian noise to the objective function.
    - _Rosenbrock: Implements the Rosenbrock function.
    # Class Attributes:
    - gauss_beta (float): The standard deviation parameter for the Gaussian noise (default is 1.0).
    # Methods:
    - __str__(): Returns the string identifier 'Rosenbrock_gauss' for this problem instance.
    # Usage:
    Instantiate this class to create a noisy Rosenbrock optimization problem for benchmarking optimization algorithms.
    """
    
    gauss_beta = 1.
    def __str__(self):
        return 'Rosenbrock_gauss'

class F111(UniformNoisyProblem, _Rosenbrock):
    """
    # Introduction
    Represents the noisy Rosenbrock function with uniform noise, as part of the COCO BBOB single-objective optimization problems.
    # Attributes:
    - uniform_alpha (float): The alpha parameter for the uniform noise distribution (default: 1.0).
    - uniform_beta (float): The beta parameter for the uniform noise distribution (default: 1.0).
    # Inheritance:
    - Inherits from `UniformNoisyProblem` and `_Rosenbrock`, combining uniform noise handling and the Rosenbrock function definition.
    # Methods:
    - __str__(): Returns a string representation of the problem ("Rosenbrock_uniform").
    """
    
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Rosenbrock_uniform'

class F112(CauchyNoisyProblem, _Rosenbrock):
    """
    # Introduction
    Represents the noisy Rosenbrock function (BBOB F112) with Cauchy noise, as defined in the COCO BBOB benchmarking suite. Inherits from `CauchyNoisyProblem` and `_Rosenbrock` to provide the problem definition and noise characteristics.
    # Class Attributes:
    - cauchy_alpha (float): Scale parameter for the Cauchy noise distribution (default: 1.0).
    - cauchy_p (float): Probability of applying Cauchy noise (default: 0.2).
    # Methods:
    - __str__(): Returns the string identifier for the problem ("Rosenbrock_cauchy").
    # Inheritance:
    - CauchyNoisyProblem: Base class for problems with Cauchy-distributed noise.
    - _Rosenbrock: Base class implementing the Rosenbrock function.
    """
    
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Rosenbrock_cauchy'

class F9(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the rotated Rosenbrock function (F9) from the BBOB benchmark suite, implemented for use with numpy arrays. This class applies a linear transformation and shift to the input, and evaluates the rotated Rosenbrock function with an optional bias.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Built-in Attribute:
    - linearTF (np.ndarray): The linear transformation matrix used for rotation and scaling.
    - FES (int): Function evaluation count (inherited from BBOB_Numpy_Problem).
    # Returns:
    - float or np.ndarray: The evaluated rotated Rosenbrock function value(s) for the given input(s), with bias added.
    # Raises:
    - ValueError: If input dimensions do not match the problem's dimensionality.
    """
    """
    Rosenbrock_rotated
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = np.matmul(0.5 * np.ones(dim), self.linearTF) / (scale ** 2)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rosenbrock_rotated'

    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(x, self.linearTF.T) + 0.5
        return np.sum(100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=-1) + self.bias

class _Ellipsoidal(BBOB_Numpy_Problem):
    """
    # Introduction
    Abstract base class for Ellipsoidal functions in the BBOB (Black-Box Optimization Benchmarking) suite, implemented using NumPy. This class defines the common structure and transformation pipeline for ellipsoidal benchmark problems, including shifting, rotating, oscillating, and conditioning the input vector.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the final function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Attributes:
    - condition (float or np.ndarray): Conditioning parameter(s) for the ellipsoidal function.
    - FES (int): Function evaluation count, incremented with each call to `func`.
    # Methods:
    - func(x): Evaluates the ellipsoidal function on input `x` after applying shift, rotation, oscillation, and conditioning transformations.
    # Returns:
    - float or np.ndarray: The computed function value(s) after all transformations and boundary handling.
    # Notes:
    This is an abstract class and is intended to be subclassed with specific conditioning parameters.
    """
    condition = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        nx = self.dim
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        i = np.arange(nx)
        return np.sum((self.condition ** (i / (nx - 1))) * (z ** 2), -1) + self.bias + self.boundaryHandling(x)

class F10(_Ellipsoidal):
    """
    # Introduction
    Represents the high-conditioned Ellipsoidal benchmark function (F10) from the COCO BBOB suite.
    # Attributes:
    - condition (float): The conditioning parameter for the ellipsoidal function, set to 1e6.
    # Methods:
    - boundaryHandling(x): Handles boundary constraints for the input vector `x`. For this function, always returns 0.
    - __str__(): Returns a string representation of the function.
    # Inheritance:
    Inherits from `_Ellipsoidal`, which provides the core implementation of the ellipsoidal function.
    # Usage:
    Typically used as a test function for single-objective optimization algorithms to evaluate performance on ill-conditioned problems.
    """
    condition = 1e6
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Ellipsoidal_high_cond'

class F116(GaussNoisyProblem, _Ellipsoidal):
    """
    # Introduction
    Represents the noisy ellipsoidal function (F116) from the COCO BBOB benchmark suite, combining an ellipsoidal function with Gaussian noise.
    # Attributes:
    - condition (float): Conditioning parameter for the ellipsoidal function, controlling the axis scaling.
    - gauss_beta (float): Standard deviation of the Gaussian noise applied to the function.
    # Inheritance:
    - Inherits from `GaussNoisyProblem` for noise handling.
    - Inherits from `_Ellipsoidal` for the ellipsoidal function structure.
    # Methods:
    - __str__(): Returns the string identifier for this problem.
    # Usage:
    Typically used in benchmarking optimization algorithms under noisy conditions.
    """
    
    condition = 1e4
    gauss_beta = 1.
    def __str__(self):
        return 'Ellipsoidal_gauss'

class F117(UniformNoisyProblem, _Ellipsoidal):
    """
    # Introduction
    Represents the Ellipsoidal function with uniform noise for single-objective optimization, as part of the COCO BBOB benchmark suite.
    # Inherits From:
    - UniformNoisyProblem: Adds uniform noise to the objective function.
    - _Ellipsoidal: Provides the base implementation of the Ellipsoidal function.
    # Class Attributes:
    - condition (float): The conditioning parameter of the Ellipsoidal function (default: 1e4).
    - uniform_alpha (float): Alpha parameter for the uniform noise (default: 1.0).
    - uniform_beta (float): Beta parameter for the uniform noise (default: 1.0).
    # Methods:
    - __str__(): Returns the string representation 'Ellipsoidal_uniform'.
    # Usage:
    Instantiate this class to create an Ellipsoidal function with uniform noise for benchmarking optimization algorithms.
    """
    
    condition = 1e4
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Ellipsoidal_uniform'

class F118(CauchyNoisyProblem, _Ellipsoidal):
    """
    # Introduction
    Represents the Ellipsoidal function with Cauchy noise as defined in the BBOB COCO benchmark suite.
    # Attributes:
    - condition (float): The conditioning of the ellipsoidal function (default: 1e4).
    - cauchy_alpha (float): The scale parameter for the Cauchy noise (default: 1.0).
    - cauchy_p (float): The probability of applying Cauchy noise (default: 0.2).
    # Inheritance:
    Inherits from `CauchyNoisyProblem` and `_Ellipsoidal`, combining the ellipsoidal problem structure with Cauchy-distributed noise.
    # Methods:
    - __str__(): Returns the string identifier for the problem ("Ellipsoidal_cauchy").
    """
    condition = 1e4
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Ellipsoidal_cauchy'

class F11(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the BBOB F11 "Discus" benchmark function for single-objective optimization, implemented using NumPy.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector for the input transformation.
    - rotate (np.ndarray): Rotation matrix for the input transformation.
    - bias (float): Bias added to the final function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Discus'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = osc_transform(z)
        return np.power(10, 6) * (z[:, 0] ** 2) + np.sum(z[:, 1:] ** 2, -1) + self.bias

class F12(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Bent Cigar function (BBOB F12) for single-objective optimization benchmarking, implemented using numpy. Inherits from `BBOB_Numpy_Problem` and applies a sequence of transformations to the input vector before evaluating the Bent Cigar function.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector for the input transformation.
    - rotate (np.ndarray): Rotation matrix for the input transformation.
    - bias (float): Bias added to the final function value.
    - lb (float or np.ndarray): Lower bound(s) for the input domain.
    - ub (float or np.ndarray): Upper bound(s) for the input domain.
    # Built-in Attribute:
    - beta (float): Parameter for the asymmetric transformation, default is 0.5.
    """
    beta = 0.5

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Bent_Cigar'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = asy_transform(z, beta=self.beta)
        z = np.matmul(z, self.rotate.T)
        return z[:, 0] ** 2 + np.sum(np.power(10, 6) * (z[:, 1:] ** 2), -1) + self.bias

class F13(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Sharp Ridge function (BBOB F13) for single-objective optimization benchmarking, implemented using NumPy. This class is a part of the COCO BBOB test suite and inherits from `BBOB_Numpy_Problem`.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10 ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Sharp_Ridge'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        return z[:, 0] ** 2. + 100. * np.sqrt(np.sum(z[:, 1:] ** 2., axis=-1)) + self.bias

class _Dif_powers(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the abstract Different Powers function from the BBOB (Black-Box Optimization Benchmarking) suite, implemented using NumPy. This class defines a shifted and rotated version of the Different Powers test function, commonly used for benchmarking optimization algorithms.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Methods:
    - func(x): Evaluates the Different Powers function at the given input `x`.
    # Returns (for func):
    - float: The computed value of the Different Powers function at `x`, including bias and boundary handling.
    # Raises:
    - ValueError: If input dimensions do not match the problem's dimensionality.
    """
    
    """
    Abstract Different_Powers
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        i = np.arange(self.dim)
        return np.power(np.sum(np.power(np.fabs(z), 2 + 4 * i / max(1, self.dim - 1)), -1), 0.5) + self.bias + self.boundaryHandling(x)

class F14(_Dif_powers):
    """
    # Introduction
    Represents the Different Powers (F14) benchmark function from the COCO BBOB suite, inheriting from `_Dif_powers`.
    # Methods
    - boundaryHandling(x): Handles boundary constraints for the input vector `x`.
    - __str__(): Returns the string representation of the function.
    # Args:
    - x (numpy.ndarray): Input vector for the boundary handling method.
    # Returns:
    - boundaryHandling: Always returns 0.0, indicating no boundary penalty.
    - __str__: Returns the name 'Different_Powers'.
    # Notes:
    This class is typically used in single-objective optimization benchmarking scenarios.
    """
    
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Different_Powers'

class F119(GaussNoisyProblem, _Dif_powers):
    """
    # Introduction
    Represents the "Different Powers" function with Gaussian noise for the COCO BBOB single-objective optimization benchmark suite.
    # Inherits From:
    - GaussNoisyProblem: Adds Gaussian noise to the objective function.
    - _Dif_powers: Implements the "Different Powers" test function.
    # Class Attributes:
    - gauss_beta (float): The standard deviation parameter for the Gaussian noise (default: 1.0).
    # Methods:
    - __str__(): Returns the string identifier 'Different_Powers_gauss' for this problem instance.
    """
    
    gauss_beta = 1.
    def __str__(self):
        return 'Different_Powers_gauss'

class F120(UniformNoisyProblem, _Dif_powers):
    """
    # Introduction
    Represents the "Different Powers" function with uniform noise for the COCO BBOB single-objective optimization benchmark suite.
    # Inherits From:
    - UniformNoisyProblem: Adds uniform noise to the problem.
    - _Dif_powers: Implements the "Different Powers" test function.
    # Attributes:
    - uniform_alpha (float): Alpha parameter for the uniform noise distribution (default: 1.0).
    - uniform_beta (float): Beta parameter for the uniform noise distribution (default: 1.0).
    # Methods:
    - __str__(): Returns the string identifier 'Different_Powers_uniform' for this problem instance.
    """
    
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Different_Powers_uniform'

class F121(CauchyNoisyProblem, _Dif_powers):
    """
    # Introduction
    Represents the Different Powers function with Cauchy noise, as defined in the COCO BBOB suite.
    # Inheritance:
    - Inherits from `CauchyNoisyProblem` and `_Dif_powers`.
    # Class Attributes:
    - cauchy_alpha (float): The scale parameter for the Cauchy noise distribution (default: 1.0).
    - cauchy_p (float): The probability of applying Cauchy noise (default: 0.2).
    # Methods:
    - __str__(): Returns a string representation of the problem ("Different_Powers_cauchy").
    # Usage:
    Typically used as a single-objective optimization problem with Cauchy noise for benchmarking optimization algorithms.
    """
    
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Different_Powers_cauchy'

class F15(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the BBOB Rastrigin function (F15) problem for single-objective optimization, with configurable dimension, shift, rotation, scaling, and bias. This class implements the Rastrigin function with additional transformations as specified in the BBOB test suite.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (10. ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.matmul(rotate, np.diag(scales)), rotate_gen(dim))
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Rastrigin_F15'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = asy_transform(osc_transform(z), beta=0.2)
        z = np.matmul(z, self.linearTF.T)
        return 10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + np.sum(z ** 2, axis=-1) + self.bias

class F16(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Weierstrass function (BBOB F16) as a single-objective optimization problem in the COCO BBOB suite, implemented using NumPy. This class applies various transformations to the input and computes the Weierstrass function value, including oscillation, rotation, scaling, and penalization.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the final function value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Built-in Attributes:
    - linearTF (np.ndarray): Linear transformation matrix combining rotation and scaling.
    - aK (np.ndarray): Coefficient array for the Weierstrass function.
    - bK (np.ndarray): Frequency array for the Weierstrass function.
    - f0 (float): Reference value for normalization.
    - FES (int): Function evaluation count (inherited from base class).
    # Methods:
    - __str__(): Returns the name of the function ("Weierstrass").
    - func(x): Evaluates the transformed Weierstrass function on input `x`.
    # Returns:
    - func(x): Returns a NumPy array of function values for the input batch `x`.
    # Raises:
    - ValueError: If input dimensions do not match the problem definition.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (0.01 ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.matmul(rotate, np.diag(scales)), rotate_gen(dim))
        self.aK = 0.5 ** np.arange(12)
        self.bK = 3.0 ** np.arange(12)
        self.f0 = np.sum(self.aK * np.cos(np.pi * self.bK))
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Weierstrass'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = np.matmul(osc_transform(z), self.linearTF.T)
        return 10 * np.power(np.mean(np.sum(self.aK * np.cos(np.matmul(2 * np.pi * (z[:, :, None] + 0.5), self.bK[None, :])), axis=-1), axis=-1) - self.f0, 3) + \
               10 / self.dim * pen_func(x, self.ub) + self.bias

class _Scaffer(BBOB_Numpy_Problem):
    """
    # Introduction
    Abstract base class for Scaffer functions in the BBOB (Black-Box Optimization Benchmarking) suite, implemented using NumPy. This class provides the structure and common transformations for Scaffer-type problems, including conditioning, shifting, rotation, and boundary handling.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias term added to the objective value.
    - lb (float or np.ndarray): Lower bound(s) for the input variables.
    - ub (float or np.ndarray): Upper bound(s) for the input variables.
    # Attributes:
    - condition (float or None): Conditioning parameter, must be defined in subclasses.
    - linearTF (np.ndarray): Linear transformation matrix combining conditioning and rotation.
    - FES (int): Function evaluation count, incremented on each call to `func`.
    """
    condition = None  # need to be defined in subclass

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (self.condition ** 0.5) ** np.linspace(0, 1, dim)
        self.linearTF = np.matmul(np.diag(scales), rotate_gen(dim))
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        z = np.matmul(asy_transform(z, beta=0.5), self.linearTF.T)
        s = np.sqrt(z[:, :-1] ** 2 + z[:, 1:] ** 2)
        return np.power(1 / (self.dim - 1) * np.sum(np.sqrt(s) * (np.power(np.sin(50 * np.power(s, 0.2)), 2) + 1), axis=-1), 2) + \
               self.boundaryHandling(x) + self.bias

class F17(_Scaffer):
    """
    # Introduction
    Represents the Schaffer's F17 function from the BBOB COCO benchmark suite, used for single-objective optimization problems.
    # Attributes:
    - condition (float): Conditioning parameter for the function, default is 10.
    """
    condition = 10.
    def boundaryHandling(self, x):
        return 10 * pen_func(x, self.ub)

    def __str__(self):
        return 'Schaffers'

class F18(_Scaffer):
    """
    # Introduction
    Represents the Schaffer's high condition function (F18) from the COCO BBOB benchmark suite, used for single-objective optimization benchmarking.
    # Attributes:
    - condition (float): Conditioning factor for the function, set to 1000.
    """
    condition = 1000.
    def boundaryHandling(self, x):
        return 10 * pen_func(x, self.ub)

    def __str__(self):
        return 'Schaffers_high_cond'

class F122(GaussNoisyProblem, _Scaffer):
    """
    # Introduction
    Represents the Schaffer's function with Gaussian noise as a single-objective optimization problem, 
    inheriting from `GaussNoisyProblem` and `_Scaffer`. This class is part of the COCO BBOB benchmark suite.
    # Class Attributes:
    - condition (float): Conditioning parameter for the problem, set to 10.
    - gauss_beta (float): Standard deviation parameter for the Gaussian noise, set to 1.
    """
    condition = 10.
    gauss_beta = 1.
    def __str__(self):
        return 'Schaffers_gauss'

class F123(UniformNoisyProblem, _Scaffer):
    """
    # Introduction
    Represents the Schaffer's F6 function with uniform noise, as part of the COCO BBOB noisy single-objective optimization problems.
    # Methods:
    - __str__(): Returns the string representation 'Schaffers_uniform'.
    # Usage:
    Typically used as a benchmark function for evaluating optimization algorithms under uniform noise conditions.
    """
    condition = 10.
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Schaffers_uniform'

class F124(CauchyNoisyProblem, _Scaffer):
    """
    # Introduction
    Represents the Schaffer's function with Cauchy noise as defined in the BBOB COCO benchmark suite. Inherits from `CauchyNoisyProblem` and `_Scaffer` to provide a noisy single-objective optimization problem.
    # Class Attributes:
    - condition (float): Conditioning parameter for the problem (default: 10.0).
    - cauchy_alpha (float): Scale parameter for the Cauchy noise (default: 1.0).
    - cauchy_p (float): Probability of applying Cauchy noise (default: 0.2).
    """
    condition = 10.
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Schaffers_cauchy'

class _Composite_Grie_rosen(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents an abstract composite benchmark problem combining Griewank and Rosenbrock functions, used in the COCO BBOB benchmarking suite. This class applies a linear transformation and shift to the input, and computes a composite objective value with boundary handling.
    # Attributes:
    - factor (float or None): Scaling factor for the composite function.
    - linearTF (np.ndarray): Linear transformation matrix derived from rotation and scaling.
    """
    
    """
    Abstract Composite_Grie_rosen
    """
    factor = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scale = max(1., dim ** 0.5 / 8.)
        self.linearTF = scale * rotate
        shift = np.matmul(0.5 * np.ones(dim) / (scale ** 2.), self.linearTF)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(x, self.linearTF.T) + 0.5
        s = 100. * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (1. - z[:, :-1]) ** 2
        return self.factor + self.factor * np.sum(s / 4000. - np.cos(s), axis=-1) / (self.dim - 1.) + self.bias + self.boundaryHandling(x)

class F19(_Composite_Grie_rosen):
    """
    # Introduction
    Represents the F19 function, a composite benchmark function combining Griewank and Rosenbrock functions, used in black-box optimization benchmarking (BBOB).
    # Attributes:
    - factor (float): Scaling factor applied to the function, set to 10.0.
    """
    factor = 10.
    def boundaryHandling(self, x):
        return 0.

    def __str__(self):
        return 'Composite_Grie_rosen'

class F125(GaussNoisyProblem, _Composite_Grie_rosen):
    """
    # Introduction
    Represents the Composite Griewank-Rosenbrock function with Gaussian noise, as part of the COCO BBOB single-objective optimization problems.
    # Inheritance:
    - Inherits from `GaussNoisyProblem` and `_Composite_Grie_rosen`.
    # Class Attributes:
    - factor (float): Scaling factor for the function. Default is 1.0.
    - gauss_beta (float): Standard deviation parameter for the Gaussian noise. Default is 1.0.
    """
    factor = 1.
    gauss_beta = 1.
    def __str__(self):
        return 'Composite_Grie_rosen_gauss'

class F126(UniformNoisyProblem, _Composite_Grie_rosen):
    """
    # Introduction
    Represents a composite optimization problem combining Griewank and Rosenbrock functions with uniform noise, as part of the COCO BBOB suite.
    # Inherits:
    - UniformNoisyProblem: Adds uniform noise to the objective function.
    - _Composite_Grie_rosen: Composite function combining Griewank and Rosenbrock landscapes.
    # Class Attributes:
    - factor (float): Scaling factor for the objective function (default: 1.0).
    - uniform_alpha (float): Alpha parameter for the uniform noise (default: 1.0).
    - uniform_beta (float): Beta parameter for the uniform noise (default: 1.0).
    """

    factor = 1.
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Composite_Grie_rosen_uniform'

class F127(CauchyNoisyProblem, _Composite_Grie_rosen):
    """
    # Introduction
    Represents the Composite Griewank-Rosenbrock function with Cauchy noise, as part of the COCO BBOB noisy single-objective optimization problems.
    # Attributes:
    - factor (float): Scaling factor for the function output.
    - cauchy_alpha (float): Alpha parameter for the Cauchy noise.
    - cauchy_p (float): Probability parameter for applying Cauchy noise.
    """
    factor = 1.
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Composite_Grie_rosen_cauchy'

class F20(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Schwefel function (BBOB F20) for benchmarking single-objective optimization algorithms, implemented using NumPy.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector for the problem (overwritten in constructor).
    - rotate (np.ndarray): Rotation matrix for the problem.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        shift = 0.5 * 4.2096874633 * np.random.choice([-1., 1.], size=dim)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Schwefel'

    def func(self, x):
        self.FES += x.shape[0]
        tmp = 2 * np.abs(self.shift)
        scales = (10 ** 0.5) ** np.linspace(0, 1, self.dim)
        z = 2 * np.sign(self.shift) * x
        z[:, 1:] += 0.25 * (z[:, :-1] - tmp[:-1])
        z = 100. * (scales * (z - tmp) + tmp)
        b = 4.189828872724339
        return b - 0.01 * np.mean(z * np.sin(np.sqrt(np.abs(z))), axis=-1) + 100 * pen_func(z / 100, self.ub) + self.bias

class _Gallagher(BBOB_Numpy_Problem):
    """
    # Introduction
    Abstract base class for the Gallagher benchmark functions (F21 and F22) from the BBOB (Black-Box Optimization Benchmarking) suite, implemented using NumPy. The Gallagher functions are multimodal optimization problems with a configurable number of peaks (local optima), designed to test the performance of optimization algorithms in complex landscapes.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector for the global optimum.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    # Built-in Attribute:
    - n_peaks (int): Number of peaks (local optima) in the function. Must be set to 21 (F22) or 101 (F21) in subclasses.
    - y (np.ndarray): Locations of the global and local optima, shape [n_peaks, dim].
    - C (np.ndarray): Scaling matrices for each peak, shape [n_peaks, dim].
    - w (np.ndarray): Weights for each peak, shape [n_peaks].
    - FES (int): Function evaluation count (inherited from BBOB_Numpy_Problem).
    # Returns:
    - float or np.ndarray: The evaluated function value(s) for the given input(s).
    # Raises:
    - ValueError: If `n_peaks` is not set to a supported value (21 or 101).
    """
    n_peaks = None

    def __init__(self, dim, shift, rotate, bias, lb, ub):
        # problem param config
        if self.n_peaks == 101:   # F21
            opt_shrink = 1.       # shrink of global & local optima
            global_opt_alpha = 1e3
        elif self.n_peaks == 21:  # F22
            opt_shrink = 0.98     # shrink of global & local optima
            global_opt_alpha = 1e6
        else:
            raise ValueError(f'{self.n_peaks} peaks Gallagher is not supported yet.')

        # generate global & local optima y[i]
        self.y = opt_shrink * (np.random.rand(self.n_peaks, dim) * (ub - lb) + lb)  # [n_peaks, dim]
        self.y[0] = shift * opt_shrink  # the global optimum
        shift = self.y[0]

        # generate the matrix C[i]
        sqrt_alpha = 1000 ** np.random.permutation(np.linspace(0, 1, self.n_peaks - 1))
        sqrt_alpha = np.insert(sqrt_alpha, obj=0, values=np.sqrt(global_opt_alpha))
        self.C = [np.random.permutation(sqrt_alpha[i] ** np.linspace(-0.5, 0.5, dim)) for i in range(self.n_peaks)]
        self.C = np.vstack(self.C)  # [n_peaks, dim]

        # generate the weight w[i]
        self.w = np.insert(np.linspace(1.1, 9.1, self.n_peaks - 1), 0, 10.)  # [n_peaks]

        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def func(self, x):
        self.FES += x.shape[0]
        z = np.matmul(np.expand_dims(x, axis=1).repeat(self.n_peaks, axis=1) - self.y, self.rotate.T)  # [NP, n_peaks, dim]
        z = np.max(self.w * np.exp((-0.5 / self.dim) * np.sum(self.C * (z ** 2), axis=-1)), axis=-1)  # [NP]
        return osc_transform(10 - z) ** 2 + self.bias + self.boundaryHandling(x)

class F21(_Gallagher):
    """
    # Introduction
    Represents the Gallagher's 101 Peaks function (BBOB F21) for single-objective optimization benchmarking.
    # Attributes:
    - n_peaks (int): Number of peaks in the Gallagher function (fixed at 101).
    # Notes:
    This class is typically used within the COCO BBOB benchmarking suite for evaluating optimization algorithms on multimodal landscapes.
    """
    n_peaks = 101
    def boundaryHandling(self, x):
        return pen_func(x, self.ub)

    def __str__(self):
        return 'Gallagher_101Peaks'

class F22(_Gallagher):
    """
    # Introduction
    Represents the Gallagher's 21 Peaks function (BBOB F22) for single-objective optimization benchmarking.
    # Attributes:
    - n_peaks (int): Number of peaks in the Gallagher function (fixed at 21).
    # Notes:
    This class inherits from `_Gallagher` and is intended for use within the COCO BBOB benchmarking suite.
    """
    n_peaks = 21
    def boundaryHandling(self, x):
        return pen_func(x, self.ub)

    def __str__(self):
        return 'Gallagher_21Peaks'

class F128(GaussNoisyProblem, _Gallagher):
    """
    # Introduction
    Represents the noisy Gallagher 101 Peaks function with Gaussian noise, as defined in the COCO BBOB benchmark suite.
    # Inheritance:
    - Inherits from `GaussNoisyProblem` and `_Gallagher`.
    """
    n_peaks = 101
    gauss_beta = 1.
    def __str__(self):
        return 'Gallagher_101Peaks_gauss'


class F129(UniformNoisyProblem, _Gallagher):
    """
    # Introduction
    Represents the Gallagher 101 Peaks function with uniform noise, as part of the COCO BBOB single-objective optimization benchmark suite.
    # Inherits From:
    - UniformNoisyProblem: Adds uniform noise to the objective function.
    - _Gallagher: Provides the base implementation for Gallagher's multimodal functions.
    # Attributes:
    - n_peaks (int): Number of peaks in the Gallagher function (default: 101).
    - uniform_alpha (float): Alpha parameter for the uniform noise (default: 1.0).
    - uniform_beta (float): Beta parameter for the uniform noise (default: 1.0).
    """
    
    n_peaks = 101
    uniform_alpha = 1.
    uniform_beta = 1.
    def __str__(self):
        return 'Gallagher_101Peaks_uniform'


class F130(CauchyNoisyProblem, _Gallagher):
    """
    # Introduction
    Represents the Cauchy-noisy variant of the Gallagher 101 Peaks function (BBOB F130) for single-objective optimization benchmarking.
    # Attributes:
    - n_peaks (int): Number of peaks in the Gallagher function (default: 101).
    - cauchy_alpha (float): Scale parameter for the Cauchy noise (default: 1.0).
    - cauchy_p (float): Probability of applying Cauchy noise (default: 0.2).
    # Inheritance:
    Inherits from:
    - CauchyNoisyProblem: Adds Cauchy noise to the objective function.
    - _Gallagher: Implements the Gallagher 101 Peaks landscape.
    """
    n_peaks = 101
    cauchy_alpha = 1.
    cauchy_p = 0.2
    def __str__(self):
        return 'Gallagher_101Peaks_cauchy'


class F23(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Katsuura function (BBOB F23) as a single-objective optimization problem for benchmarking. 
    This class is part of the COCO BBOB suite and implements the function using numpy for efficient computation.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector applied to the input.
    - rotate (np.ndarray): Rotation matrix applied to the input.
    - bias (float): Bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        scales = (100. ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
        BBOB_Numpy_Problem.__init__(self, dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Katsuura'

    def func(self, x):
        self.FES += x.shape[0]
        z = sr_func(x, self.shift, self.rotate)
        tmp3 = np.power(self.dim, 1.2)
        tmp1 = np.repeat(np.power(np.ones((1, 32)) * 2, np.arange(1, 33)), x.shape[0], 0)
        res = np.ones(x.shape[0])
        for i in range(self.dim):
            tmp2 = tmp1 * np.repeat(z[:, i, None], 32, 1)
            temp = np.sum(np.fabs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1, -1)
            res *= np.power(1 + (i + 1) * temp, 10 / tmp3)
        tmp = 10 / self.dim / self.dim
        return res * tmp - tmp + pen_func(x, self.ub) + self.bias


class F24(BBOB_Numpy_Problem):
    """
    # Introduction
    Represents the Lunacek bi-Rastrigin function (BBOB F24) as a single-objective optimization problem, implemented using NumPy. This class is designed for benchmarking optimization algorithms and is part of the COCO BBOB test suite.
    # Args:
    - dim (int): Dimensionality of the problem.
    - shift (np.ndarray): Shift vector for the problem landscape.
    - rotate (np.ndarray): Rotation matrix for the problem landscape.
    - bias (float): Constant bias added to the function value.
    - lb (float or np.ndarray): Lower bound(s) of the search space.
    - ub (float or np.ndarray): Upper bound(s) of the search space.
    # Methods:
    - __str__(): Returns the name of the function ("Lunacek_bi_Rastrigin").
    - func(x): Evaluates the Lunacek bi-Rastrigin function at the given input(s) `x`.
    # Returns:
    - func(x): Returns the computed function value(s) as a float or np.ndarray, depending on the input shape.
    # Raises:
    - ValueError: If input dimensions do not match the problem's dimensionality.
    """
    def __init__(self, dim, shift, rotate, bias, lb, ub):
        self.mu0 = 2.5 / 5 * ub
        shift = np.random.choice([-1., 1.], size=dim) * self.mu0 / 2
        scales = (100 ** 0.5) ** np.linspace(0, 1, dim)
        rotate = np.matmul(np.matmul(rotate_gen(dim), np.diag(scales)), rotate)
        super().__init__(dim, shift, rotate, bias, lb, ub)

    def __str__(self):
        return 'Lunacek_bi_Rastrigin'

    def func(self, x):
        self.FES += x.shape[0]
        x_hat = 2. * np.sign(self.shift) * x
        z = np.matmul(x_hat - self.mu0, self.rotate.T)
        s = 1. - 1. / (2. * np.sqrt(self.dim + 20.) - 8.2)
        mu1 = -np.sqrt((self.mu0 ** 2 - 1) / s)
        return np.minimum(np.sum((x_hat - self.mu0) ** 2., axis=-1), self.dim + s * np.sum((x_hat - mu1) ** 2., axis=-1)) + \
               10. * (self.dim - np.sum(np.cos(2. * np.pi * z), axis=-1)) + 1e4 * pen_func(x, self.ub) + self.bias

