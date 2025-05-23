import numpy as np
import torch
from .learnable_optimizer import Learnable_Optimizer
import torch.nn as nn

def vector2nn(x,net,device):
    """
    # Introduction
    Maps a flat parameter vector to the parameters of a given neural network, updating the network's weights in-place.
    # Args:
    - x (list or numpy.ndarray): A 1D array or list containing all the parameters to be assigned to the network.
    - net (torch.nn.Module): The neural network whose parameters will be updated.
    - device (torch.device or str): The device on which the parameters should be allocated (e.g., 'cpu' or 'cuda').
    # Returns:
    - torch.nn.Module: The neural network with updated parameters.
    # Raises:
    - AssertionError: If the length of `x` does not match the total number of parameters in `net`.
    """
    
    assert len(x) == sum([param.nelement() for param in net.parameters()]), 'dim of x and net not match!'
    params = net.parameters()
    ptr = 0
    for v in params:
        num_of_params = v.nelement()
        temp = torch.Tensor(x[ptr: ptr+num_of_params]).to(device)
        v.data = temp.reshape(v.shape)
        ptr += num_of_params
    return net


class SelfAttn(nn.Module):
    
    def __init__(self):
        """
        # Introduction
        Initializes the optimizer class and sets up the linear transformation layers for query, key, and value projections.
        # Built-in Attribute:
        - Wq (nn.Linear): Linear layer mapping input of size 3 to output of size 8 for query projection.
        - Wk (nn.Linear): Linear layer mapping input of size 3 to output of size 8 for key projection.
        - Wv (nn.Linear): Linear layer mapping input of size 3 to output of size 1 for value projection.
        # Raises:
        - None
        """
        
        super().__init__()
        self.Wq = nn.Linear(3,8)
        self.Wk = nn.Linear(3,8)
        self.Wv = nn.Linear(3,1)
    
    def forward(self, X):
        """
        # Introduction
        Computes the output of a single-head self-attention mechanism using the input tensor `X`.
        Applies learned linear projections to obtain queries, keys, and values, computes attention scores,
        and returns the attended output.
        # Args:
        - X (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        # Returns:
        - torch.Tensor: The output tensor after applying self-attention and softmax, with the last singleton dimension removed.
        # Raises:
        - None
        """
        
        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)
        attn_score = torch.softmax(torch.matmul(Q, K.T)/np.sqrt(8), dim=-1)
        return torch.softmax(torch.matmul(attn_score, V), dim=0).squeeze()

class LrNet(nn.Module):
    def __init__(self):
        """
        # Introduction
        Initializes the optimizer neural network with two linear layers and a sigmoid activation function.
        # Built-in Attribute:
        - ln1 (nn.Linear): First linear layer transforming input of size 19 to 8.
        - ln2 (nn.Linear): Second linear layer transforming input of size 8 to 2.
        - sm (nn.Sigmoid): Sigmoid activation function applied to the output.
        # Returns:
        - None
        """
        
        super().__init__()
        self.ln1 = nn.Linear(19,8)
        self.ln2 = nn.Linear(8,2)
        self.sm = nn.Sigmoid()
    def forward(self, X):
        """
        # Introduction
        Applies two sequential linear transformations with normalization and a softmax activation to the input tensor.
        # Args:
        - X (torch.Tensor): The input tensor to be processed.
        # Returns:
        - torch.Tensor: The output tensor after normalization, linear transformation, and softmax activation.
        # Raises:
        - None
        """
        
        X = self.ln1(X)
        return self.sm(self.ln2(X))

class LES_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    **L**earned **E**volution **S**trategy (LES) is a novel self-attention-based evolution strategies parametrization, and discover effective update rules for ES via meta-learning.
    # Original paper
    "[**Discovering evolution strategies via meta-black-box optimization**](https://iclr.cc/virtual/2023/poster/11005)." The Eleventh International Conference on Learning Representations. (2023).
    # Official Implementation
    [LES](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/les.py)
    """
    def __init__(self, config):
        """
        # Introduction
        Initializes the optimizer with the given configuration, setting up neural network components, device, and various hyperparameters for the optimization process.
        # Args:
        - config (object): Config object containing optimizer settings.
            - Attributes needed for the LES_Optimizer are the following:
                - device (str or torch.device): Device on which computations will be performed.Default is 'cpu'.
                - maxFEs (int): Maximum number of function evaluations allowed.
                - log_interval (int): Interval for logging progress.Default is 100.
                - n_logpoint (int): Number of log points for logging.Default is 50.
        # Built-in Attribute:
        - self.device (torch.device): Device on which computations will be performed.
        - self.max_fes (int): Maximum number of function evaluations allowed.
        - self.attn (SelfAttn): Self-attention neural network module.
        - self.mlp (LrNet): Multi-layer perceptron neural network module.
        - self.alpha (list of float): List of alpha time-scale values.
        - self.timestamp (np.ndarray): Array of timestamps for tracking progress.
        - self.save_time (int): Counter for save operations.Default is 0.
        - self.NP (int): Population size or related parameter.Default is 16.
        - self.sigma_ratio (float): Ratio used for sigma calculation. Default is 0.2.
        - self.fes (int or None): Current function evaluation count. Default is None.
        - self.cost (any): Variable for recording cost or loss. Default is None.
        - self.log_index (any): Index for logging. Default is None.
        - self.log_interval (int): Interval for logging progress.
        # Returns:
        - None
        # Raises:
        - None
        """
        
        super().__init__(config)
        self.__config = config
        self.device = self.__config.device
        self.max_fes = config.maxFEs

        self.attn = SelfAttn().to(self.device)
        self.mlp = LrNet().to(self.device)
        self.alpha = [0.1,0.5,0.9] # alpha time-scale
        self.timestamp = np.array([1,3,10,30,50,100,250,500,750,1000,1250,1500,2000])
        self.save_time=0
        self.NP=16
        self.sigma_ratio=0.2

        self.fes = None

        # for record
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval
    
    def __str__(self):
        """
        # Introduction
        Returns a string representation of the LES_Optimizer object.
        # Returns:
        - str: The string "LES_Optimizer", representing the class name.
        """
        
        return "LES_Optimizer"
    
    def init_population(self, problem):
        """
        # Introduction
        Initializes the population for the optimizer using a normal distribution based on the problem's bounds and dimension. Sets up initial evolution information, including parent solutions, their costs, and statistical parameters for the optimization process.
        # Args:
        - problem (object): The optimization problem object, which has attributes `ub` (upper bounds), `lb` (lower bounds), `dim` (problem dimensionality), and a method `eval` for evaluating a population.
        # Built-in Attributes:
        - self.ub (np.ndarray): Upper bounds for the variables.
        - self.lb (np.ndarray): Lower bounds for the variables.
        - self.problem (object): The optimization problem instance.
        - self.evolution_info (dict): Dictionary to store evolution information, including parents, costs, and generation counter.
        - self.cost (list): List to store the best costs found during the optimization process.
        - self.log_index (int): Index for logging progress.Default is 1.
        - self.fes (int): Total number of function evaluations performed.Default is 0.
        - self.meta_X (list, optional): List to store population snapshots for meta data logging.Default is empty.
        - self.meta_Cost (list, optional): List to store cost snapshots for meta data logging.Default is empty.
        # Returns:
        - None
        # Side Effects:
        - Sets several instance attributes such as `ub`, `lb`, `problem`, `evolution_info`, `fes`, `cost`, `log_index`, and optionally `meta_X` and `meta_Cost` if full meta data logging is enabled.
        # Notes:
        - The initial population is generated using a normal distribution clipped to the problem's bounds.
        - The method assumes that `self.rng` is a random number generator and `self.sigma_ratio`, `self.NP`, and `self.__config.full_meta_data` are properly initialized instance attributes.
        """
        
        self.ub = problem.ub
        self.lb = problem.lb
        self.problem = problem

        mu = problem.lb + (problem.ub-problem.lb) * self.rng.rand(problem.dim)
        sigma = np.ones(problem.dim)*self.ub*self.sigma_ratio
        population = np.clip(self.rng.normal(mu,sigma,(self.NP,problem.dim)), self.lb, self.ub) # is it correct?
        costs = problem.eval(population)
        self.evolution_info = {'parents': population,
                'parents_cost':costs,
                'generation_counter': 0, 
                'gbest':np.min(costs),
                'Pc':np.zeros((3,problem.dim)),
                'Ps':np.zeros((3,problem.dim)),
                'mu':mu,
                'sigma':sigma}
        self.fes = self.NP

        self.cost = [np.min(costs)]
        self.log_index = 1

        if self.__config.full_meta_data:
            self.meta_X = [self.evolution_info['parents'].copy()]
            self.meta_Cost = [self.evolution_info['parents_cost'].copy()]

        return None

    def cal_attn_feature(self):
        """
        # Introduction
        Computes attention features for the current population in the evolutionary optimization process. The features include the z-score of population costs, shifted normalized ranking, and an improvement indicator, which are concatenated into a single tensor.
        # Returns:
        - torch.FloatTensor: A tensor of shape (N, 3), where N is the population size. Each row contains the z-score of the cost, the shifted normalized rank, and a boolean indicator of improvement for each individual.
        # Notes:
        - The z-score is calculated to standardize the population costs.
        - The shifted rank normalizes the ranking of costs and centers it around zero.
        - The improvement indicator is a boolean array indicating whether each individual's cost is better than the global best.
        """
        
        # z-score of population costs
        population_costs = self.evolution_info['parents_cost']
        z_score = (population_costs-np.mean(population_costs))/(np.std(population_costs)+1e-8) # avoid nan
        # shifted normalized ranking
        shifted_rank = np.argsort(population_costs)/self.NP - 0.5
        # improvement indicator
        improved = population_costs < self.evolution_info['gbest']
        # concat above three feature to N * 3 array
        return torch.from_numpy(np.vstack([z_score,shifted_rank,improved]).T).to(torch.float64)
    
    def cal_mlp_feature(self, W):
        """
        # Introduction
        Calculates multi-layer perceptron (MLP) features based on the current evolutionary state, including evolution paths and timestamp embeddings.
        # Args:
        - W (np.ndarray): Weight vector or matrix used to compute weighted sums of evolutionary information.
        # Returns:
        - A torch tensor containing the concatenated feature vector (shape: [dim, 19]).
        - Numpy array of updated evolution paths for the mean (`c`, shape: [3, dim]).
        - Numpy array of updated evolution paths for the standard deviation (`s`, shape: [3, dim]).
        # Notes:
        The function computes updated evolution paths (`Pc` and `Ps`) for each alpha value, generates a timestamp embedding, and concatenates these features for use in an MLP. The output tensor is suitable for input into a neural network.
        """
        
        # P_c_t P_sigma_t
        Pc = []
        Ps = []
        for i,alpha in enumerate(self.alpha):
            temp1 = (1-alpha) * self.evolution_info['Pc'][i] + \
                    alpha * (np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])*W[:,None],axis=0) - self.evolution_info['Pc'][i]) # need to be checked!
            temp2 = (1-alpha) * self.evolution_info['Ps'][i] + \
                    alpha * (np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])/self.evolution_info['sigma']*W[:,None],axis=0) - self.evolution_info['Ps'][i]) # need to be checked!
            Pc.append(temp1)
            Ps.append(temp2)
        
        # timestamp embedding
        rho = np.tanh(self.evolution_info['generation_counter'] / self.timestamp  - 1)[None,:].repeat(self.problem.dim,axis=0) #  dim * 13
        c = np.vstack(Pc) # dim * 3
        s = np.vstack(Ps) # dim * 3
        # concat to 19dim feature
        return torch.from_numpy(np.hstack([c.T,s.T,rho])).to(torch.float64), c, s
    
    def update(self,action, problem):
        """
        # Introduction
        Updates the optimizer's internal state by performing one or more evolutionary optimization steps using the provided action and problem. The method adapts model parameters, generates new populations, evaluates them, and logs progress until a stopping criterion is met.
        # Args:
        - action (dict): Dictionary containing new model parameters for attention and MLP networks, and optionally a 'skip_step' key to limit the number of steps.
        - problem (object): The optimization problem object, which has a `dim` attribute and an `eval` method for evaluating populations.
        # Returns:
        - float: The best cost (fitness) found in the current optimization run.
        - float: The normalized improvement from the initial to the best cost.
        - bool: Whether the stopping criterion was met.
        - dict: Additional information (currently empty).
        # Raises:
        - None explicitly, but may raise exceptions if the action or problem objects are malformed or if numerical errors occur during optimization.
        """

        # get new model parameters 
        self.attn=vector2nn(action['attn'],self.attn,self.device)
        self.mlp=vector2nn(action['mlp'],self.mlp,self.device)
        skip_step = None
        if action.get('skip_step') is not None:
            skip_step = action['skip_step']
        
        step = 0
        is_end = False
        init_y = None
        while not is_end:
            # get features of present population
            fitness_feature = self.cal_attn_feature()
            # get w_{i} for each individual
            W = self.attn(fitness_feature.to(self.device)).detach().cpu().numpy() 
            # get features for mlp
            alpha_feature, Pc, Ps = self.cal_mlp_feature(W)
            # get learning rates
            alpha = self.mlp(alpha_feature.to(self.device)).detach().cpu().numpy() # self.dim * 2
            alpha_mu = alpha[:,0]
            alpha_sigma = alpha[:,1]
            # update mu and sigma for next generation
            mu = (1 - alpha_mu) * self.evolution_info['mu'] + \
                alpha_mu * np.sum((self.evolution_info['parents'] - self.evolution_info['mu'])*W[:,None],axis=0)
            sigma = (1 - alpha_sigma) * self.evolution_info['sigma'] + \
                alpha_sigma * np.sqrt(np.sum((self.evolution_info['parents'] - self.evolution_info['mu']) ** 2 *W[:,None],axis=0)) # need to be checked!
            # sample childs according new mu and sigma
            population = np.clip(self.rng.normal(mu,sigma,(self.NP,self.problem.dim)), self.lb, self.ub)
            # evaluate the childs
            costs = self.problem.eval(population)
            self.fes += self.NP
            gbest = np.min([np.min(costs),self.evolution_info['gbest']])
            if step == 0:
                init_y = gbest
            t = self.evolution_info['generation_counter'] + 1
            # update evolution information
            self.evolution_info = {'parents': population,
                    'parents_cost':costs,
                    'generation_counter': t, 
                    'gbest':gbest,
                    'Pc':Pc,
                    'Ps':Ps,
                    'mu':mu,
                    'sigma':sigma}

            if self.__config.full_meta_data:
                self.meta_X.append(self.evolution_info['parents'].copy())
                self.meta_Cost.append(self.evolution_info['parents_cost'].copy())

            is_end = (self.fes >= self.max_fes)

            step += 1
            if skip_step is not None:
                is_end = (step >= skip_step)

            if self.fes >= self.log_index * self.log_interval:
                self.log_index += 1
                self.cost.append(gbest)
            
            if is_end:
                if len(self.cost) >= self.__config.n_logpoint + 1:
                    self.cost[-1] = gbest
                else:
                    while len(self.cost) < self.__config.n_logpoint + 1:
                        self.cost.append(gbest)
        
        info = {}
        return self.evolution_info['gbest'],(init_y - gbest) / init_y,is_end,info
    


    
