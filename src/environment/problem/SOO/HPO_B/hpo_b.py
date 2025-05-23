import numpy as np
import xgboost as xgb
import pickle
import os, time
from ....problem.basic_problem import Basic_Problem


class HPOB_Problem(Basic_Problem):
    """
    # Introduction
    HPO-B is an autoML hyper-parameter optimization benchmark which includes a wide range of hyperparameter optimization tasks for 16 different model types (e.g., SVM, XGBoost, etc.), resulting in a total of 935 problem instances. The dimension of these problem instances range from 2 to 16. We also note that HPO-B represents problems with ill-conditioned landscape such as huge flattern.
    # Original paper
    "[Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml.](https://arxiv.org/pdf/2106.06257)" arXiv preprint arXiv:2106.06257 (2021).
    # Official Implementation
    [HPO-B](https://github.com/machinelearningnuremberg/HPO-B)
    # License
    None
    """
    def __init__(self,bst_surrogate,dim,y_min,y_max,lb,ub,normalized=False,name=None) -> None:
        """
        Initializes the class with the provided surrogate model, dimensionality, bounds, and other configuration parameters.
        # Args:
        - bst_surrogate: The surrogate model to be used for optimization.
        - dim (int): The dimensionality of the problem.
        - y_min (float): The minimum value of the objective function.
        - y_max (float): The maximum value of the objective function.
        - lb (array-like): The lower bounds for each dimension.
        - ub (array-like): The upper bounds for each dimension.
        - normalized (bool, optional): Whether to normalize the input data. Defaults to False.
        - name (str, optional): The name of the problem instance. Defaults to None.
        # Built-in Attributes:
        - bst_surrogate: Stores the surrogate model.
        - y_min: Stores the minimum objective value.
        - y_max: Stores the maximum objective value.
        - dim: Stores the problem dimensionality.
        - gbest: Stores the global best value found so far.Defaults to 1e+10.
        - normalized: Indicates if normalization is applied.
        - collect_gbest: List to collect global best values during optimization.
        - lb: Stores the lower bounds.
        - ub: Stores the upper bounds.
        - optimum: Stores the optimum solution (if found).
        - name: Stores the name of the problem instance.
        """
        self.bst_surrogate=bst_surrogate
        self.y_min=y_min
        self.y_max=y_max
        self.dim=dim
        self.gbest=1e+10
        self.normalized = normalized
        self.collect_gbest=[]
        self.lb = lb
        self.ub = ub
        self.optimum = None
        self.name = name
    def func(self,position):
        """
        # Introduction
        Evaluates the surrogate model prediction for a given position, normalizes the result, and updates the global best collection.
        # Args:
        - position (np.ndarray): The input position vector to be evaluated, expected to have shape compatible with (self.dim,).
        # Returns:
        - float: The negative normalized prediction from the surrogate model for the given position.
        # Side Effects:
        - Appends the current global best (`self.gbest`) to the `self.collect_gbest` list.
        # Notes:
        - Uses an XGBoost surrogate model (`self.bst_surrogate`) for prediction.
        - Normalization is performed using the `self.normalize` method.
        """
        x_q = xgb.DMatrix(position.reshape(-1,self.dim))
        new_y = self.bst_surrogate.predict(x_q)
        cost=-self.normalize(new_y)
        self.collect_gbest.append(self.gbest)
        return cost

    def normalize(self, y):
        """
        # Introduction
        Normalizes the input array `y` to the range [0, 1] based on the object's normalization settings.
        # Args:
        - y (np.ndarray or float): The input value(s) to be normalized.
        # Returns:
        - np.ndarray or float: The normalized value(s) if normalization is enabled; otherwise, returns the input as is.
        # Notes:
        - If `self.normalized` is True and `self.y_min` is None, normalization is performed using the min and max of `y`.
        - If `self.normalized` is True and `self.y_min` is set, normalization uses `self.y_min` and `self.y_max` and clips the result to [0, 1].
        - If `self.normalized` is False, the input `y` is returned unchanged.
        """
        if self.normalized:
            if self.y_min is None:
                return (y-np.min(y))/(np.max(y)-np.min(y))
            else:
                return np.clip((y-self.y_min)/(self.y_max-self.y_min),0,1)
        return y

    def __str__(self):
        """
        # Introduction
        Returns the string representation of the object, typically its name.
        # Returns:
        - str: The name attribute of the object.
        """
        return self.name