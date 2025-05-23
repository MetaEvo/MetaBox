from ....problem.basic_problem import Basic_Problem_Torch
import numpy as np
import math
import torch
import time
from os import path
import os
import importlib.util
import importlib.resources as pkg_resources
MINMAX = -1

class CEC2013MMO_Torch_Problem(Basic_Problem_Torch):
    """
    # CEC2013MMO_Torch_Problem
    A pyTorch-based implementation of the CEC 2013 Multi-Modal Optimization (MMO) benchmark suite.
    """
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.FES = 0
        self.optimum = fopt
        self.rho = rho
        self.nopt = nopt
        self.maxfes = maxfes

    def eval(self, x):
        start=time.perf_counter()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if x.dtype != torch.float64:
            x = x.type(torch.float64)
        if x.ndim == 1:  # x is a single individual
            y=self.func(x.reshape(1, -1))[0]
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        elif x.ndim == 2:  # x is a whole population
            y=self.func(x)
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
        else:
            y=self.func(x.reshape(-1, x.shape[-1]))
            end=time.perf_counter()
            self.T1+=(end-start)*1000
            return y
    
    def func(self, x):
        raise NotImplementedError

    def how_many_goptima_torch(self, pop, accuracy):
        NP, D = pop.shape[0], pop.shape[1]

        fits = self.eval(torch.tensor(pop)).data.numpy()

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

class CFunction(CEC2013MMO_Torch_Problem):
    # Abstract composition function
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
        super(CFunction, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)
        self.__nofunc_ = nofunc

    def func(self, x):
        raise NotImplementedError

    def __evaluate_inner_(self, x):
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")
        self.__fi_ = torch.tensor([], dtype = torch.float64, device = x.device)

        self.__calculate_weights(x)
        for i in range(self.__nofunc_):
            self.__transform_to_z(x, i)
            self.__fi_ = torch.cat((self.__fi_, self.__function_[i](self.__z_).reshape(x.shape[0], 1)), dim = 1)

        tmpsum = self.__weight_ * (
            self.__C_ * self.__fi_ / self.__fmaxi_[None, :].to(x.device) + torch.tensor(self.__bias_, device = x.device)[None, :]
        )
        return torch.sum(tmpsum, dim = 1) * MINMAX + self.__f_bias_

    def __calculate_weights(self, x):
        self.__weight_ = torch.tensor([], dtype = torch.float64, device = x.device)
        for i in range(self.__nofunc_):
            mysum = torch.sum((x - torch.tensor(self.__O_[i], device = x.device)) ** 2, -1)
            self.__weight_ = torch.cat((self.__weight_, torch.exp(-mysum / (2.0 * self.dim * self.__sigma_[i] * self.__sigma_[i])).reshape(x.shape[0], 1)), dim = 1)
        maxw = torch.max(self.__weight_, -1).values

        maxw10 = maxw ** 10
        mask = (self.__weight_ != maxw[:, None])
        mask_trans = (~mask)
        mask_content = (self.__weight_ * (1.0 - maxw10[:, None]))
        self.__weight_ = mask * mask_content + mask_trans * self.__weight_

        mysum = torch.sum(self.__weight_, -1)
        mask1 =  (mysum == 0.0)
        mask2 = (mysum != 0.0)
        sum_content1 = torch.ones_like(mysum, dtype = torch.float64, device = x.device)
        mysum = mask1 * sum_content1 + mask2 * mysum
        content_mask1 = torch.ones_like(self.__weight_, device = x.device, dtype = torch.float64) * (1.0 / (1.0 * self.__nofunc_))
        content_mask2 = self.__weight_ / mysum[:, None]
        self.__weight_ = mask1[:, None] * content_mask1 + mask2[:, None] * content_mask2

    def __calculate_fmaxi(self):
        self.__fmaxi_ = torch.zeros(self.__nofunc_, dtype=torch.float64)
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")

        x5 = 5 * torch.ones(self.dim, dtype = torch.float64)

        for i in range(self.__nofunc_):
            self.__transform_to_z_noshift(x5[None, :], i)
            self.__fmaxi_[i] = self.__function_[i](self.__z_)[0]

    def __transform_to_z_noshift(self, x, index):
        tmpx = torch.divide(x, torch.tensor(self.__lambda_[index], device = x.device))
        self.__z_ = torch.matmul(tmpx, torch.tensor(self.__M_[index], device = x.device))

    def __transform_to_z(self, x, index):
        tmpx = torch.divide((x - torch.tensor(self.__O_[index], device = x.device)), torch.tensor(self.__lambda_[index], device = x.device))
        self.__z_ = torch.matmul(tmpx, torch.tensor(self.__M_[index], device = x.device))

    def __load_rotmat(self, file_obj):
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
    return (x ** 2).sum(dim = 1)

# Rastrigin's function
def FRastrigin(x):
    return torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * np.pi * x) + 10, dim=1)

# Griewank's function
def FGrienwank(x):
    i = torch.sqrt(torch.arange(x.shape[1], dtype=torch.float64, device = x.device) + 1.0)
    return torch.sum(x ** 2, dim = 1) / 4000.0 - torch.prod(torch.cos(x / i[None, :]), dim = 1) + 1.0

# Weierstrass's function
def FWeierstrass(x):
    alpha = 0.5
    beta = 3.0
    kmax = 20
    D = x.shape[1]
    exprf = 0.0

    c1 = alpha ** torch.arange(kmax + 1, dtype=torch.float64, device = x.device)
    c2 = 2.0 * np.pi * beta ** np.arange(kmax + 1)
    c2 = torch.tensor(c2, device = x.device)
    f = torch.zeros(x.shape[0], dtype=torch.float64,device = x.device)
    c = -D * torch.sum(c1 * torch.cos(c2 * 0.5))

    for i in range(D):
        f = f + torch.sum(c1[None, :] * torch.cos(c2[None, :] * (x[:, i:i+1] + 0.5)), dim = 1)
    return f + c

def F8F2(x):
    f2 = 100.0 * (x[:, 0] ** 2 - x[:, 1]) ** 2 + (1.0 - x[:, 0]) ** 2
    return 1.0 + (f2 ** 2) / 4000.0 - torch.cos(f2)

# FEF8F2 function
def FEF8F2(x):
    D = x.shape[1]
    f = torch.zeros(x.shape[0], device = x.device, dtype=torch.float64)
    for i in range(D - 1):
        f = f + F8F2(x[:, [i, i + 1]] + 1)
    f = f + F8F2(x[:, [D - 1, 0]] + 1)
    return f

class F1_Torch(CEC2013MMO_Torch_Problem): # five_uneven_peak_trap
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F1_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'five_uneven_peak_trap'+'_D'+str(self.dim)

    def func(self, x):
        
        if x is None:
            return None
        
        assert x.shape[1] == self.dim

        self.FES += x.shape[0]
        x = x[:, 0]
        result = torch.zeros_like(x, device = x.device, dtype=torch.float64)
        mask1 = (x >= 0) & (x < 2.50)
        mask2 = (x >= 2.5) & (x < 5)
        mask3 = (x >= 5.0) & (x < 7.5)
        mask4 = (x >= 7.5) & (x < 12.5)
        mask5 = (x >= 12.5) & (x < 17.5)
        mask6 = (x >= 17.5) & (x < 22.5)
        mask7 = (x >= 22.5) & (x < 27.5)
        mask8 = (x >= 27.5) & (x <= 30)
        
        result = mask1 * (80 * (2.5 - x)) + (~mask1) * result
        result = mask2 * (64 * (x - 2.5)) + (~mask2) * result
        result = mask3 * (64 * (7.5 - x)) + (~ mask3) * result
        result = mask4 * (28 * (x - 7.5)) + (~ mask4) * result
        result = mask5 * (28 * (17.5 - x)) + (~ mask5) * result
        result = mask6 * (32 * (x - 17.5)) + (~ mask6) * result
        result = mask7 * (32 * (27.5 - x)) + (~ mask7) * result
        result = mask8 * (80 * (x - 27.5)) + (~ mask8) * result
        
        return -result

class F2_Torch(CEC2013MMO_Torch_Problem): # equal_maxima
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F2_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'equal_maxima'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -torch.sin(5.0 * np.pi * x[:, 0]) ** 6

class F3_Torch(CEC2013MMO_Torch_Problem): # uneven_decreasing_maxima
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F3_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'uneven_decreasing_maxima'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -(
            torch.exp(-2.0 * torch.log(torch.tensor(2, device = x.device)) * ((x[:, 0] - 0.08) / 0.854) ** 2)
            * (torch.sin(torch.tensor(5 * np.pi, device = x.device) * (x[:, 0] ** 0.75 - 0.05))) ** 6
        )
        
class F4_Torch(CEC2013MMO_Torch_Problem): # himmelblau
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F4_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'himmelblau'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        result = 200 - (x[:, 0] ** 2 + x[:, 1] - 11) ** 2 - (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
        return -result

class F5_Torch(CEC2013MMO_Torch_Problem): # six_hump_camel_back
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F5_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'six_hump_camel_back'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        x2 = x[:, 0] ** 2
        x4 = x[:, 0] ** 4
        y2 = x[:, 1] ** 2
        expr1 = (4.0 - 2.1 * x2 + x4 / 3.0) * x2
        expr2 = x[:, 0] * x[:, 1]
        expr3 = (4.0 * y2 - 4.0) * y2
        return -(-1.0 * (expr1 + expr2 + expr3))

class F6_Torch(CEC2013MMO_Torch_Problem): # shubert
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F6_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'shubert'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
        soma = torch.zeros((n, D), device = x.device, dtype=torch.float64)
        
        for j in range(1, 6):
            soma = soma + (j * torch.cos((j + 1) * x + j))
        result = torch.prod(soma, dim = 1)


        return -(-result)

class F7_Torch(CEC2013MMO_Torch_Problem): # vincent
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F7_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'vincent'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape

        result = torch.sum((torch.sin(10 * torch.log(x))) / D, dim = 1)
        return -result

class F8_Torch(CEC2013MMO_Torch_Problem): # modified_rastrigin_all
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F8_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes)

    def __str__(self):
        return 'modified_rastrigin_all'+'_D'+str(self.dim)

    def func(self, x):
        if x is None:
            return None

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        n, D = x.shape
    
        if D == 2:
            k = [3, 4]
        elif D == 8:
            k = [1, 2, 1, 2, 1, 3, 1, 4]
        elif D == 16:
            k = [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]

        result = torch.sum(10 + 9 * torch.cos(torch.tensor(2 * np.pi * np.array(k), device = x.device)[None, :] * x), dim=1)
        return -(-result)

class F9_Torch(CFunction): # CF1
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F9_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

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
        return 'CF1'+'_D'+str(self.dim)

    def func(self, x):

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

class F10_Torch(CFunction): # CF2
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F10_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

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
        return 'CF2'+'_D'+str(self.dim)

    def func(self, x):
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

class F11_Torch(CFunction): # CF3
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F11_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 6)

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
            # fname = path.join(path.dirname(__file__), 'datafile') + "/CF3_M_D" + str(dim) + ".dat"
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
        return 'CF3'+'_D'+str(self.dim)

    def func(self, x):

        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)

class F12_Torch(CFunction): # CF4
    def __init__(self, dim, lb, ub, fopt, rho, nopt, maxfes):
        super(F12_Torch, self).__init__(dim, lb, ub, fopt, rho, nopt, maxfes, 8)

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
            local_file_path = os.path.join(base_path, "", 'optima.dat')
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
        return 'CF4'+'_D'+str(self.dim)

    def func(self, x):
        assert x.shape[1] == self.dim
        self.FES += x.shape[0]
        return -self._CFunction__evaluate_inner_(x)




