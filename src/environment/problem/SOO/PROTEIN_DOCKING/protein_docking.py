from os import path
import torch
import numpy as np
from torch.utils.data import Dataset
from ....problem.basic_problem import Basic_Problem, Basic_Problem_Torch
import time


class Protein_Docking_Numpy_Problem(Basic_Problem):
    """
    # Introduction
    
    Protein-Docking benchmark, where the objective is to minimize the Gibbs free energy resulting from protein-protein interaction between a given complex and any other conformation. We select 28 protein complexes and randomly initialize 10 starting points for each complex, resulting in 280 problem instances. To simplify the problem structure, we only optimize 12 interaction points in a complex instance (12D problem).

    # Original paper
    "[Protein–protein docking benchmark version 4.0.](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22830)" Proteins: Structure, Function, and Bioinformatics 78.15 (2010): 3111-3114.
    # Official Implementation
    [Protein-Docking](https://zlab.wenglab.org/benchmark/)
    # License
    None
    """
    
    n_atoms = 100  # number of interface atoms considered for computational concern
    dim = 12
    lb = -1.5
    ub = 1.5

    def __init__(self, coor_init, q, e, r, basis, eigval, problem_id):
        """
        # Introduction
        Initializes the protein docking problem instance with the provided parameters.
        # Args:
        - coor_init (np.ndarray): Initial coordinates of atoms, shape [n_atoms, 3].
        - q (np.ndarray): Charge matrix, shape [n_atoms, n_atoms].
        - e (np.ndarray): Epsilon matrix, shape [n_atoms, n_atoms].
        - r (np.ndarray): Distance matrix, shape [n_atoms, n_atoms].
        - basis (np.ndarray): Basis vectors, shape [dim, 3*n_atoms].
        - eigval (np.ndarray): Eigenvalues, shape [dim].
        - problem_id (Any): Identifier for the problem instance.
        # Attributes:
        - optimum (Any or None): The optimum value, initially set to None as it is unknown.
        """
        
        self.coor_init = coor_init  # [n_atoms, 3]
        self.q = q                  # [n_atoms, n_atoms]
        self.e = e                  # [n_atoms, n_atoms]
        self.r = r                  # [n_atoms, n_atoms]
        self.basis = basis          # [dim, 3*n_atoms]
        self.eigval = eigval        # [dim]
        self.problem_id = problem_id
        self.optimum = None      # unknown, set to None

    def __str__(self):
        """
        # Introduction
        Returns a string representation of the object, specifically its `problem_id`.
        # Returns:
        - str: The `problem_id` attribute of the object.
        """
        return self.problem_id

    def func(self, x):
        """
        # Introduction
        Computes the energy of a protein docking configuration based on atomic coordinates and pairwise interactions.
        This function transforms the input coordinates, computes pairwise distances between atoms, applies interaction coefficients, and calculates the mean energy for each configuration in the batch.
        # Args:
        - x (np.ndarray): Input array of shape [NP, 3 * n_atoms], representing the coordinates for NP configurations.
        # Built-in Attribute:
        - self.eigval (np.ndarray): Eigenvalues used for coordinate transformation.
        - self.basis (np.ndarray): Basis matrix for coordinate transformation.
        - self.n_atoms (int): Number of atoms in the protein.
        - self.coor_init (np.ndarray): Initial coordinates of the atoms.
        - self.q (float): Charge-related coefficient for interaction energy.
        - self.e (float): Energy scaling factor.
        - self.r (float): Distance scaling factor.
        # Returns:
        - np.ndarray: Array of shape [NP], containing the computed energy for each configuration.
        # Raises:
        - ValueError: If input shapes are incompatible or required attributes are missing.
        """
        eigval = 1.0 / np.sqrt(self.eigval)
        product = np.matmul(x * eigval, self.basis)  # [NP, 3*n_atoms]
        new_coor = product.reshape((-1, self.n_atoms, 3)) + self.coor_init  # [NP, n_atoms, 3]

        p2 = np.expand_dims(np.sum(new_coor * new_coor, axis=-1), axis=-1)  # sum of squares along last dim.  [NP, n_atoms, 1]
        p3 = np.matmul(new_coor, np.transpose(new_coor, (0, 2, 1)))  # inner products among row vectors. [NP, n_atoms, n_atoms]
        pair_dis = p2 - 2 * p3 + np.transpose(p2, (0, 2, 1))
        pair_dis = np.sqrt(pair_dis + 0.01)  # [NP, n_atoms, n_atoms]

        gt0_lt7 = (pair_dis > 0.11) & (pair_dis < 7.0)
        gt7_lt9 = (pair_dis > 7.0) & (pair_dis < 9.0)

        pair_dis += np.eye(self.n_atoms)  # [NP, n_atoms, n_atoms]
        coeff = self.q / (4. * pair_dis) + np.sqrt(self.e) * ((self.r / pair_dis) ** 12 - (self.r / pair_dis) ** 6)  # [NP, n_atoms, n_atoms]

        energy = np.mean(
            np.sum(10 * gt0_lt7 * coeff + 10 * gt7_lt9 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8),
                   axis=1), axis=-1)  # [NP]

        return energy


class Protein_Docking_Torch_Problem(Basic_Problem_Torch):
    """
    # Introduction
    Represents a protein docking optimization problem using PyTorch tensors, enabling the evaluation of protein conformations based on atomic coordinates, interaction matrices, and physical potentials. This class supports both single and batch evaluations, and is designed for use in single-objective optimization (SOO) settings.
    # Original paper

    # Official Implementation

    # License
    None
    """
    n_atoms = 100  # number of interface atoms considered for computational concern
    dim = 12
    lb = -1.5
    ub = 1.5

    def __init__(self, coor_init, q, e, r, basis, eigval, problem_id):
        """
        # Introduction
        Initializes the protein docking problem instance with atomic coordinates, interaction matrices, basis, eigenvalues, and problem identifier.
        # Attributes:
        - coor_init (torch.Tensor): Tensor of initial atomic coordinates.
        - q (torch.Tensor): Tensor of atomic charges or related property.
        - e (torch.Tensor): Tensor of interaction energies or related property.
        - r (torch.Tensor): Tensor of distances or related property.
        - basis (torch.Tensor): Tensor of basis vectors.
        - eigval (torch.Tensor): Tensor of eigenvalues.
        - problem_id (Any): Problem identifier.
        - optimum (None): Placeholder for the optimum value, initially set to None.
        """
        
        self.coor_init = torch.as_tensor(coor_init, dtype=torch.float64)  # [n_atoms, 3]
        self.q = torch.as_tensor(q, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.e = torch.as_tensor(e, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.r = torch.as_tensor(r, dtype=torch.float64)  # [n_atoms, n_atoms]
        self.basis = torch.as_tensor(basis, dtype=torch.float64)    # [dim, 3*n_atoms]
        self.eigval = torch.as_tensor(eigval, dtype=torch.float64)  # [dim]
        self.problem_id = problem_id
        self.optimum = None  # unknown, set to None

    def __str__(self):
        """
        Returns a string representation of the protein docking problem instance.
        # Returns:
            str: The unique identifier (`problem_id`) of the problem instance.
        """
        
        return self.problem_id

    def eval(self, x):
        """
        # Introduction
        Evaluates the objective function for a given individual or population, supporting both single and batch evaluations. Tracks and accumulates the evaluation time in milliseconds.
        # Args:
        - x (array-like or torch.Tensor): The input(s) to be evaluated. Can be a 1D array/tensor (single individual) or 2D array/tensor (population).
        # Built-in Attribute:
        - self.T1 (float): Accumulates the total evaluation time in milliseconds.
        # Returns:
        - torch.Tensor or float: The evaluation result(s) from the objective function. Returns a scalar for a single individual or a tensor for a population.
        # Raises:
        - None explicitly. Assumes `self.func` handles input shape and type errors.
        """
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
        """
        # Introduction
        Computes the energy of a protein conformation based on atomic coordinates, pairwise distances, and interaction coefficients. The function transforms the input coordinates, calculates pairwise distances, applies interaction masks, and computes the total energy using a combination of electrostatic and Lennard-Jones-like potentials.
        # Args:
        - x (torch.Tensor): Input tensor of shape [NP, 3 * n_atoms], representing the coordinates in the transformed basis.
        # Built-in Attribute:
        - self.eigval (torch.Tensor): Eigenvalues used for scaling the input coordinates.
        - self.basis (torch.Tensor): Basis matrix for coordinate transformation.
        - self.n_atoms (int): Number of atoms in the protein.
        - self.coor_init (torch.Tensor): Initial coordinates of the atoms.
        - self.q (float or torch.Tensor): Charge parameter for electrostatic interaction.
        - self.e (float or torch.Tensor): Epsilon parameter for Lennard-Jones potential.
        - self.r (float or torch.Tensor): Sigma parameter for Lennard-Jones potential.
        # Returns:
        - torch.Tensor: Tensor of shape [NP], containing the computed energy for each input conformation.
        # Raises:
        - None
        """
        
        eigval = 1.0 / torch.sqrt(self.eigval)
        product = torch.matmul(x * eigval, self.basis)  # [NP, 3*n_atoms]
        new_coor = product.reshape((-1, self.n_atoms, 3)) + self.coor_init  # [NP, n_atoms, 3]

        p2 = torch.sum(new_coor * new_coor, dim=-1, dtype=torch.float64)[:, :,
             None]  # sum of squares along last dim.  [NP, n_atoms, 1]
        p3 = torch.matmul(new_coor,
                          new_coor.permute(0, 2, 1))  # inner products among row vectors. [NP, n_atoms, n_atoms]
        pair_dis = p2 - 2 * p3 + p2.permute(0, 2, 1)
        pair_dis = torch.sqrt(pair_dis + 0.01)  # [NP, n_atoms, n_atoms]

        gt0_lt7 = (pair_dis > 0.11) & (pair_dis < 7.0)
        gt7_lt9 = (pair_dis > 7.0) & (pair_dis < 9.0)

        pair_dis = pair_dis + torch.eye(self.n_atoms, dtype=torch.float64)  # [NP, n_atoms, n_atoms]
        coeff = self.q / (4. * pair_dis) + torch.sqrt(self.e) * (
                    (self.r / pair_dis) ** 12 - (self.r / pair_dis) ** 6)  # [NP, n_atoms, n_atoms]

        energy = torch.mean(
            torch.sum(10 * gt0_lt7 * coeff + 10 * gt7_lt9 * coeff * ((9 - pair_dis) ** 2 * (-12 + 2 * pair_dis) / 8),
                      dim=1, dtype=torch.float64), dim=-1)  # [NP]

        return energy
