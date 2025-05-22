import numpy as np
import torch
from .learnable_optimizer import Learnable_Optimizer

def scale(x,lb,ub):
    """
    # Introduction
    Scales the input tensor `x` to the range [`lb`, `ub`] using a sigmoid transformation.
    # Args:
    - x (torch.Tensor): The input tensor to be scaled.
    - lb (float or torch.Tensor): The lower bound of the target range.
    - ub (float or torch.Tensor): The upper bound of the target range.
    # Returns:
    - torch.Tensor: The scaled tensor with values in the range [`lb`, `ub`].
    """
    
    x=torch.sigmoid(x)
    x=lb+(ub-lb)*x
    return x

def np_scale(x,lb,ub):
    """
    # Introduction
    Scales the input value(s) `x` to a specified range [`lb`, `ub`] using a sigmoid transformation.
    # Args:
    - x (float or np.ndarray): The input value or array of values to be scaled.
    - lb (float): The lower bound of the target range.
    - ub (float): The upper bound of the target range.
    # Returns:
    - float or np.ndarray: The scaled value(s) within the range [`lb`, `ub`].
    """
    
    x=1/(1 + np.exp(-x))
    x=lb+(ub-lb)*x
    return x

class RNNOPT_Optimizer(Learnable_Optimizer):
    """
    # Introduction
    L2L_Optimizer is a learnable optimizer class that manages the optimization process for a given problem, tracking the best solution found and the cost history. It supports both rollout and standard evaluation modes, and terminates after a fixed number of function evaluations or when the optimum is known.
    """
    
    def __init__(self, config):
        """
        Initializes the optimizer with the given configuration.
        # Args:
        - config (dict): Configuration parameters for the optimizer.
        # Returns:
        - None
        """
        super().__init__(config)
        self.__config = config

    def __str__(self):
        """
        Returns a string representation of the RNN optimizer.
        # Returns:
            str: The name of the optimizer, "RNN_Optimizer".
        """
        return "RNN_Optimizer"

    def init_population(self, problem):
        """
        # Introduction
        Initializes the population and internal state variables for the optimizer.
        # Args:
        - problem: The optimization problem instance for which the population is to be initialized.
        # Built-in Attributes:
        - self.__fes (int): Function evaluation counter, initialized to zero.
        - self.cost (list): List to track the cost history, initialized as empty.
        - self.__best (Any): Tracker for the best solution found, initialized to None.
        - self.meta_X (list): List to store the input data for meta-learning, initialized as empty.
        - self.meta_Cost (list): List to store the cost data for meta-learning, initialized as empty.
        
        # Effects:
        - Resets the function evaluation counter (`__fes`) to zero.
        - Initializes the cost list (`cost`) as empty.
        - Sets the best solution tracker (`__best`) to None.
        """
        
        self.__fes=0
        self.cost=[]
        self.__best=None

        if self.__config.full_meta_data:
            self.meta_X = []
            self.meta_Cost = []
        

    def update(self,action,problem):
        """
        # Introduction
        Updates the optimizer state based on the provided action and problem instance. Scales the action, evaluates it on the problem, tracks the best result, and determines if the optimization process is done.
        # Args:
        - action (Any): The action to be evaluated, can be a numpy array or other type.
        - problem (object): The problem instance, expected to have `lb`, `ub`, `eval()`, and `optimum` attributes.
        # Returns:
        - y (float or np.ndarray): The evaluated result (possibly shifted by the optimum).
        - int: Always 0 (placeholder for compatibility).
        - bool: Whether the optimization process is done.
        # Notes:
        - Updates internal state variables such as the best result found, evaluation count, and cost history.
        - The process is considered done if the optimum is known or if the number of function evaluations reaches 100.
        """
        
        x=action
        pre_best = self.__best
        is_rollout=False
        if type(x) is np.ndarray:
            x=np_scale(x,problem.lb,problem.ub)
            is_rollout=True
        else:
            x=scale(x,problem.lb,problem.ub)
        
        # evaluate x
        if problem.optimum is None:
            y=problem.eval(x)
        else:
            y=problem.eval(x)-problem.optimum
        if self.__best is None:
            self.__best=y.item()
            self.__init_best = self.__best
            pre_best = self.__best
        elif y<self.__best:
            self.__best=y.item()
        self.cost.append(self.__best)
        self.__fes+=1

        reward = (pre_best - self.__best) / self.__init_best

        if self.__config.full_meta_data:
            if type(x) is np.ndarray:
                self.meta_X.append(x.copy())
            else:
                self.meta_X.append([x.clone().detach().cpu().data.numpy()])
            self.meta_Cost.append([y.item()])

        is_done=False
        if self.__fes>=100:
            is_done=True
        info = {}
        return y,reward,is_done, info
    
