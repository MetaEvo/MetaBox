from ....problem.basic_problem import Basic_Problem_Torch
import torch
import importlib.util
import importlib.resources as pkg_resources
import os
class CEC2013LSGO_Torch_Problem(Basic_Problem_Torch):
    """
    # CEC2013LSGO_Torch_Problem
    A PyTorch-based implementation of the CEC2013 Large-Scale Global Optimization (LSGO) benchmark problem set.  
    This class provides methods for loading problem-specific data, performing vector transformations, and evaluating standard benchmark functions.
    # Introduction
      CEC2013LSGO proposes 15 large-scale benchmark problems to represent a wider range of realworld large-scale optimization problems.
    # Original paper
      "[Benchmark functions for the CEC 2013 special session and competition on large-scale global optimization](https://al-roomi.org/multimedia/CEC_Database/CEC2015/LargeScaleGlobalOptimization/CEC2015_LargeScaleGO_TechnicalReport.pdf)." gene 7.33 (2013): 8.
    # Official Implementation
    [CEC2013LSGO](https://github.com/dmolina/cec2013lsgo)
    # License
    GPL-3.0
    # Problem Suite Composition
      CEC2013LSGO contains four major categories of large-scale problems:
      1. Fully-separable functions (F1-F3) 
      2. Two types of partially separable functions: 
          1. Partially separable functions with a set of non-separable subcomponents and one fully-separable subcomponents (F4-F7) 
          2. Partially separable functions with only a set of non-separable subcomponents and no fullyseparable subcomponent (F8-F11) 
      3. Two types of overlapping functions: 
          1. Overlapping functions with conforming subcomponents (F12-F13)
          2. Overlapping functions with conflicting subcomponents (F14)
      4. Fully-nonseparable functions (F15)
    # Attributes:
    - data_dir (str): Directory containing problem data files.
    - min_dim (int): Minimum subspace dimension.
    - med_dim (int): Medium subspace dimension.
    - max_dim (int): Maximum subspace dimension.
    - dim (int): Problem dimension (default: 1000).
    - ID (int or None): Problem identifier.
    - s_size (int): Number of subspaces.
    - overlap (int or None): Overlap size between subspaces.
    - lb (float or None): Lower bound of the search space.
    - ub (float or None): Upper bound of the search space.
    - Ovector (torch.Tensor or None): Optimum vector.
    - OvectorVec (list or None): List of optimum vectors for subspaces.
    - Pvector (torch.Tensor or None): Permutation vector.
    - r_min_dim, r_med_dim, r_max_dim (int or None): Rotation matrix dimensions.
    - anotherz (torch.Tensor): Auxiliary tensor for transformations.
    - anotherz1 (torch.Tensor or None): Auxiliary tensor for transformations.
    - numevals (int): Number of function evaluations.
    - opt (float or None): Optimal value.
    - optimum (float): Known optimum value.
    # Methods:
    - get_optimal(): Returns the optimal value.
    - func(x): Abstract method for evaluating the objective function (must be implemented in subclasses).
    - readOvector(): Loads the optimum vector from file.
    - readOvectorVec(): Loads and splits the optimum vector into subspace vectors.
    - readPermVector(): Loads the permutation vector from file.
    - readR(sub_dim): Loads the rotation matrix for a given subspace dimension.
    - readS(num): Loads the subspace dimensions from file.
    - readW(num): Loads the weights for subspaces from file.
    - multiply(vector, matrix): Multiplies a vector by a matrix.
    - rotateVector(i, c): Rotates a subspace vector using the appropriate rotation matrix.
    - rotateVectorConform(i, c): Rotates a subspace vector with overlap consideration.
    - rotateVectorConflict(i, c, x): Rotates a subspace vector with conflict consideration.
    - sphere(x): Sphere benchmark function.
    - elliptic(x): Elliptic benchmark function.
    - rastrigin(x): Rastrigin benchmark function.
    - ackley(x): Ackley benchmark function.
    - schwefel(x): Schwefel benchmark function.
    - rosenbrock(x): Rosenbrock benchmark function.
    - transform_osz(z): Applies the OSZ transformation to a vector.
    - transform_asy(z, beta=0.2): Applies the asymmetric transformation to a vector.
    - Lambda(z, alpha=10): Applies the Lambda transformation to a vector.
    # Notes:
    - This class is designed to be subclassed for specific CEC2013 LSGO functions.
    - Data files must be present in the specified `data_dir`.
    - All computations are performed using double-precision (`torch.float64`).
    """
    
    def __init__(self):
        
        # 设置默认的数据类型
        torch.set_default_dtype(torch.float64)
        self.data_dir = "metaevobox.environment.problem.SOO.CEC2013LSGO.datafile" # 数据文件夹

        # 子空间的维度大小, 先提供了三种子空间的维度大小
        self.min_dim = 25
        self.med_dim = 50
        self.max_dim = 100  

        # 基本量的设置, 不是准确的值，准确的值会在function中设置
        self.dim = 1000
        self.ID = None
        self.s_size = 20
        self.overlap = None
        self.lb = None
        self.ub = None
        self.Ovector = None
        self.OvectorVec = None
        self.Pvector = None
        self.r_min_dim = None
        self.r_med_dim = None
        self.r_max_dim = None
        self.anotherz = torch.zeros(self.dim)
        self.anotherz1 = None
        self.numevals = 0

        self.opt = None
        self.optimum = 0.0

    def get_optimal(self):
        return self.opt

    def func(self, x):
        raise NotImplementedError
        
    # 读取Ovector
    def readOvector(self):
        d = torch.zeros(self.dim)
        file_name = f"F{self.ID}-xopt.txt"

        try:
            data_dir = "metaevobox.environment.problem.SOO.CEC2013LSGO.datafile"
            if importlib.util.find_spec(data_dir) is not None:
                file_path = pkg_resources.files(data_dir).joinpath(file_name)
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_path, 'datafile', file_name)
            file_obj = open(local_path, 'r')

        try:
            with file_obj as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dim:
                            d[c] = float(value)
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_name}'")

        return d

    # 读取OvectorVec，根据子空间的大小分割，得到一个向量数组
    def readOvectorVec(self):
        d = [torch.zeros(self.s[i]) for i in range(self.s_size)]
        file_name = f"F{self.ID}-xopt.txt"

        try:
            data_dir = "metaevobox.environment.problem.SOO.CEC2013LSGO.datafile"
            if importlib.util.find_spec(data_dir) is not None:
                file_path = pkg_resources.files(data_dir).joinpath(file_name)
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_path, 'datafile', file_name)
            file_obj = open(local_path, 'r')

        try:
            with file_obj as file:
                c = 0  # index over 1 to dim
                i = -1  # index over 1 to s_size
                up = 0  # current upper bound for one group

                for line in file:
                    if c == up:  # out (start) of one group
                        i += 1
                        d[i] = torch.zeros(self.s[i])
                        up += self.s[i]

                    values = line.strip().split(',')
                    for value in values:
                        d[i][c - (up - self.s[i])] = float(value)
                        c += 1
        except FileNotFoundError:
            print(f"Cannot open the OvectorVec datafiles '{file_name}'")

        return d

    # 读取PermVector
    def readPermVector(self):
        d = torch.zeros(self.dim, dtype=int)
        file_name = f"F{self.ID}-p.txt"

        try:
            data_dir = "metaevobox.environment.problem.SOO.CEC2013LSGO.datafile"
            if importlib.util.find_spec(data_dir) is not None:
                file_path = pkg_resources.files(data_dir).joinpath(file_name)
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_path, 'datafile', file_name)
            file_obj = open(local_path, 'r')

        try:
            with file_obj as file:
                c = 0
                for line in file:
                    values = line.strip().split(',')
                    for value in values:
                        if c < self.dim:
                            d[c] = int(float(value)) - 1
                            c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_name}'")

        return d

    # 读取R，即为各个子空间的向量
    def readR(self, sub_dim):
        m = torch.zeros((sub_dim, sub_dim))
        file_name = f"F{self.ID}-R{sub_dim}.txt"

        try:
            data_dir = "metaevobox.environment.problem.SOO.CEC2013LSGO.datafile"
            if importlib.util.find_spec(data_dir) is not None:
                file_path = pkg_resources.files(data_dir).joinpath(file_name)
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_path, 'datafile', file_name)
            file_obj = open(local_path, 'r')

        try:
            with file_obj as file:
                i = 0
                for line in file:
                    values = line.strip().split(',')
                    for j, value in enumerate(values):
                        m[i, j] = float(value)
                    i += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_name}'")

        return m

    # 读取S，即为各个子问题的维度
    def readS(self, num):
        self.s = torch.zeros(num, dtype=int)
        file_name = f"F{self.ID}-s.txt"

        try:
            data_dir = "metaevobox.environment.problem.SOO.CEC2013LSGO.datafile"
            if importlib.util.find_spec(data_dir) is not None:
                file_path = pkg_resources.files(data_dir).joinpath(file_name)
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_path, 'datafile', file_name)
            file_obj = open(local_path, 'r')
        try:
            with file_obj as file:
                c = 0
                for line in file:
                    self.s[c] = int(float(line.strip()))
                    c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_name}'")
        return self.s

    # 读取W
    def readW(self, num):
        self.w = torch.zeros(num)
        file_name = f"F{self.ID}-w.txt"

        try:
            data_dir = "metaevobox.environment.problem.SOO.CEC2013LSGO.datafile"
            if importlib.util.find_spec(data_dir) is not None:
                file_path = pkg_resources.files(data_dir).joinpath(file_name)
                file_obj = file_path.open('r')
            else:
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            base_path = os.path.dirname(os.path.abspath(__file__))
            local_path = os.path.join(base_path, 'datafile', file_name)
            file_obj = open(local_path, 'r')

        try:
            with file_obj as file:
                c = 0
                for line in file:
                    self.w[c] = float(line.strip())
                    c += 1
        except FileNotFoundError:
            print(f"Cannot open the datafile '{file_name}'")

        return self.w

    # 向量乘矩阵
    def multiply(self, vector, matrix):
        return torch.matmul(matrix, vector.T).T

    # 旋转向量
    def rotateVector(self, i, c): 
        # 获取子问题的维度
        sub_dim = self.s[i]
        # 将值复制到新向量中
        indices = self.Pvector[c:c + sub_dim]
        z = self.anotherz[:,indices]
        # 选择正确的旋转矩阵并进行乘法运算
        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            print("Size of rotation matrix out of range")
            self.anotherz1 = None

        return self.anotherz1
    
    def rotateVectorConform(self, i, c):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap
        # 将值复制到新向量中
        indices = self.Pvector[start_index:end_index]
        z = self.anotherz[:, indices]
        # 选择正确的旋转矩阵并进行乘法运算
        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            print("Size of rotation matrix out of range")
            self.anotherz1 = None
    
        return self.anotherz1

    def rotateVectorConflict(self, i, c, x):
        sub_dim = self.s[i]
        start_index = c - i * self.overlap
        end_index = c + sub_dim - i * self.overlap

        # 将值复制到新向量中并进行减法运算
        indices = self.Pvector[start_index:end_index]
        z = x[:,indices] - self.OvectorVec[i]

        # 选择正确的旋转矩阵并进行乘法运算
        if sub_dim == self.r_min_dim:
            self.anotherz1 = self.multiply(z, self.r25)
        elif sub_dim == self.r_med_dim:
            self.anotherz1 = self.multiply(z, self.r50)
        elif sub_dim == self.r_max_dim:
            self.anotherz1 = self.multiply(z, self.r100)
        else:
            print("Size of rotation matrix out of range")
            self.anotherz1 = None

        return self.anotherz1
    
    # basic function
    def sphere(self,x):
        s2 = torch.sum(x ** 2,axis=-1)
        return s2

    def elliptic(self,x):
        nx = x.shape[-1]
        i = torch.arange(nx)
        return torch.sum(10 ** (6 * i / (nx - 1)) * (x ** 2), -1)

    def rastrigin(self,x):
        return torch.sum(x**2 - 10 * torch.cos(2*torch.pi*x) + 10, -1)

    def ackley(self,x):
        nx = x.shape[-1]
        sum1 = -0.2 * torch.sqrt(torch.sum(x ** 2, -1) / nx)
        sum2 = torch.sum(torch.cos(2 * torch.pi * x), -1) / nx
        return - 20 * torch.exp(sum1) - torch.exp(sum2)+20 +torch.e 

    def schwefel(self,x):
        s1 = torch.cumsum(x,axis=-1)
        s2 = torch.sum(s1 ** 2,axis=-1)
        return s2

    def rosenbrock(self,x):
        x0 = x[:,:x.shape[1]-1]
        x1 = x[:,1:x.shape[1]]
        t = x0**2 - x1
        s = torch.sum(100.0 * t**2 + (x0 - 1.0)**2,-1)
        return s
    
    def transform_osz(self,z):
        sign_z = torch.sign(z)
        hat_z = torch.where(z == 0, 0, torch.log(torch.abs(z)))
        c1_z = torch.where(z > 0, 10, 5.5)
        c2_z = torch.where(z > 0, 7.9, 3.1)
        sin_term = torch.sin(c1_z * hat_z) + torch.sin(c2_z * hat_z)
        z_transformed = sign_z * torch.exp(hat_z + 0.049 * sin_term)
        return z_transformed

    def transform_asy(self,z, beta=0.2):
        indices = torch.arange(z.shape[-1])[None,:].repeat(z.shape[0], 1)
        positive_mask = z > 0
        z[positive_mask] = z[positive_mask] ** (1 + beta * indices[positive_mask] / (z.shape[-1] - 1) * torch.sqrt(z[positive_mask]))
        return z
    
    def Lambda(self, z, alpha=10):
        dim = z.shape[-1]
        # 创建指数数组
        exponents = 0.5 * torch.arange(dim) / (dim - 1)
        # 计算变换后的z
        z = z * (alpha ** exponents)
        return z
    
class F1_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()
        self.ID = 1
        self.Ovector = self.readOvector()
        self.lb = -100.0
        self.ub = 100.0
        self.anotherz = torch.zeros(self.dim)

        self.opt = self.Ovector
    
    def __str__(self):
        return 'Shifted Elliptic'
    
    def func(self, x):
        
        self.anotherz = x - self.Ovector
        self.anotherz = self.transform_osz(self.anotherz)
        
        result = self.elliptic(self.anotherz)

        return result

class F2_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()
        self.ID = 2   
        self.Ovector = self.readOvector()
        self.lb = -5.0
        self.ub = 5.0
        self.anotherz = torch.zeros(self.dim)

        self.opt = self.Ovector

    def __str__(self):
        return 'Shifted Rastrigin'
        
    def func(self, x):
        
        self.anotherz = x - self.Ovector

        self.anotherz = self.transform_osz(self.anotherz)
        self.anotherz = self.transform_asy(self.anotherz, 0.2)
        self.anotherz = self.Lambda(self.anotherz, 10)
        
        result = self.rastrigin(self.anotherz)

        return result
    
class F3_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()
        self.ID = 3
        self.Ovector = self.readOvector()
        self.lb = -32.0
        self.ub = 32.0
        self.anotherz = torch.zeros(self.dim)

        self.opt = self.Ovector

    def __str__(self):
        return 'Shifted Ackley'
    
    def func(self, x):
        
        self.anotherz = x - self.Ovector

        self.anotherz = self.transform_osz(self.anotherz)
        self.anotherz = self.transform_asy(self.anotherz, 0.2)
        self.anotherz = self.Lambda(self.anotherz, 10)

        result = self.ackley(self.anotherz)
        
        return result

class F4_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 4
        self.s_size = 7
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -100.0
        self.ub = 100.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim)

        self.opt = self.Ovector

    def __str__(self):
        return '7-nonseparable, 1-separable Shifted and Rotated Elliptic'
    
    def func(self, x):
  
        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            result += self.w[i] * self.elliptic(anotherz1)
            c += self.s[i]  # 更新c的值

        if c < self.dim:
            z = self.anotherz[:, self.Pvector[c:self.dim]]
            z = self.transform_osz(z)
            result += self.elliptic(z)

        return result

class F5_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()
        self.ID = 5
        self.s_size = 7
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -5.0
        self.ub = 5.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim)

        self.opt = self.Ovector

    def __str__(self):
        return '7-nonseparable, 1-separable Shifted and Rotated Rastrigin'

    def compute(self, x):
   
        result = torch.zeros(x.shape[0]).to(self.device)

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            anotherz1 = self.Lambda(anotherz1, 10)
            result += self.w[i] * self.rastrigin(anotherz1)
            c += self.s[i]  # 更新c的值

        if c < self.dim:
            z = self.anotherz[:, self.Pvector[c:self.dim]]
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            z = self.Lambda(z, 10)
            result += self.rastrigin(z)

        return result

class F6_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 6
        self.s_size = 7
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -32.0
        self.ub = 32.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim) 

    def __str__(self):
        return '7-nonseparable, 1-separable Shifted and Rotated Ackley'

    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            anotherz1 = self.Lambda(anotherz1, 10)
            result += self.w[i] * self.ackley(anotherz1)
            c += self.s[i]  # 更新c的值

        if c < self.dim:
            z = self.anotherz[:, self.Pvector[c:self.dim]]
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            z = self.Lambda(z, 10)
            result += self.ackley(z)

        return result

class F7_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 7
        self.s_size = 7
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -100.0
        self.ub = 100.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim)

    def __str__(self):
        return '7-nonseparable, 1-separable Shifted Schwefel'


    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.schwefel(anotherz1)
            c += self.s[i]  # 更新c的值

        if c < self.dim:
            z = self.anotherz[:, self.Pvector[c:self.dim]]
            z = self.transform_osz(z)
            z = self.transform_asy(z, 0.2)
            result += self.sphere(z)

        return result

class F8_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 8
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -100.0
        self.ub = 100.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim)


    def __str__(self):
        return '20-nonseparable Shifted and Rotated Elliptic'

    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            result += self.w[i] * self.elliptic(anotherz1)
            c += self.s[i]  # 更新c的值

        return result

class F9_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 9
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -5.0
        self.ub = 5.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim)

    def __str__(self):
        return '20-nonseparable Shifted and Rotated Rastrigin'
    
    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            anotherz1 = self.Lambda(anotherz1, 10)
            result += self.w[i] * self.rastrigin(anotherz1)
            c += self.s[i]  # 更新c的值

        return result
    
class F10_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 10
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -32.0
        self.ub = 32.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim)

    def __str__(self):
        return '20-nonseparable Shifted and Rotated Ackley'


    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            anotherz1 = self.Lambda(anotherz1, 10)
            result += self.w[i] * self.ackley(anotherz1)
            c += self.s[i]  # 更新c的值

        return result
    
class F11_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 11
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -100.0
        self.ub = 100.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dim) 

    def __str__(self):
        return '20-nonseparable Shifted Schwefel'

    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVector(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.schwefel(anotherz1)
            c += self.s[i]  # 更新c的值

        return result

class F12_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 12
        self.s_size = 20
        self.Ovector = self.readOvector()
        self.lb = -100.0
        self.ub = 100.0
        self.anotherz = torch.zeros(self.dim)
    
    def __str__(self):
        return '20-nonseparable Shifted Schwefel'

    def compute(self, x):

        result = torch.zeros(x.shape[0])

        self.anotherz = x - self.Ovector
        result = self.rosenbrock(self.anotherz)

        return result

class F13_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 13
        self.s_size = 20
        self.dimension = 905 #because of overlapping
        self.overlap = 5
        self.Ovector = self.readOvector()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.s = self.readS(self.s_size)
        self.w = self.readW(self.s_size)
        self.lb = -100.0
        self.ub = 100.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dimension)

    def __str__(self):
        return 'Shifted Schwefel’s Function with Conforming Overlapping Subcomponents'

    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c = 0

        self.anotherz = x - self.Ovector

        for i in range(self.s_size):
            anotherz1 = self.rotateVectorConform(i, c)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.schwefel(anotherz1)
            c += self.s[i]  # 更新c的值

        return result

class F14_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 14
        self.s_size = 20
        self.dimension = 905 #because of overlapping
        self.overlap = 5
        self.s = self.readS(self.s_size)
        self.OvectorVec = self.readOvectorVec()
        self.Pvector = self.readPermVector()
        self.r25 = self.readR(25)
        self.r50 = self.readR(50)
        self.r100 = self.readR(100)
        self.w = self.readW(self.s_size)
        self.lb = -100.0
        self.ub = 100.0
        self.r_min_dim = 25
        self.r_med_dim = 50
        self.r_max_dim = 100
        self.anotherz = torch.zeros(self.dimension)
 
    def __str__(self):
        return 'Shifted Schwefel’s Function with Conflicting Overlapping Subcomponents'
        
    def compute(self, x):

        result = torch.zeros(x.shape[0])

        c=0

        for i in range(self.s_size):
            anotherz1 = self.rotateVectorConflict(i, c, x)
            anotherz1 = self.transform_osz(anotherz1)
            anotherz1 = self.transform_asy(anotherz1, 0.2)
            result += self.w[i] * self.schwefel(anotherz1)
            c += self.s[i]  # 更新c的值

        return result
    
class F15_Torch(CEC2013LSGO_Torch_Problem):
    def __init__(self):
        super().__init__()

        self.ID = 15
        self.s_size = 20
        self.dimension = 1000
        self.Ovector = self.readOvector()
        self.lb = -100.0
        self.ub = 100.0
        self.anotherz = torch.zeros(self.dimension) 

    def __str__(self):
        return 'Shifted Schwefel’s Function with Conflicting Overlapping Subcomponents'
    
    def compute(self, x):      

        result = torch.zeros(x.shape[0])

        self.anotherz = x - self.Ovector
        self.anotherz = self.transform_osz(self.anotherz)
        self.anotherz = self.transform_asy(self.anotherz, 0.2)
        result = self.schwefel(self.anotherz)

        return result
