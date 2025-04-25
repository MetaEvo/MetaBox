import copy
import functools
import numpy as np
import math
import sys

from operator import itemgetter
from scipy.spatial.distance import cdist
from environment.optimizer.basic_optimizer import Basic_Optimizer
# from baseline.bbo.moo_utils import *
from environment.problem.MOO.MOO_synthetic.dtlz_numpy import *



POSITIVE_INFINITY = float("inf")
EPSILON = sys.float_info.epsilon


class PlatypusError(Exception):
    pass



class MOEAD(Basic_Optimizer):
    
    def __init__(self, config):
        super().__init__(config)
        # Problem Related
        self.n_ref_points = 1000
        # # MOEA/D Algorithm Related
        self.population_size = 100
        self.moead_neighborhood_size = 8
        self.moead_neighborhood_maxsize = 30
        self.moead_delta = 0.8
        self.moead_eta = 2
        # self.max_fes=config.maxFEs
        self.max_fes = config.maxFEs
    def __str__(self):
        return 'MOEAD'

    def init_population(self, problem):
        # problem
        self.problem = problem
        self.n_obj = problem.n_obj
        self.n_var = problem.n_var
        # population
        self.weights = self.random_weights(self.n_obj,self.population_size)
        if self.population_size!=len(self.weights):
            self.population_size = len(self.weights)
        self.population = self.rng.uniform(low=problem.lb, high=problem.ub, size=(self.population_size, problem.n_var))
        self.population_obj = problem.eval(self.population)
        self.neighborhoods = self.get_neighborhoods()
        # budget
        self.done = False
        self.fes = len(self.population)
        self.episode_limit = self.max_fes // self.population_size
        self.moead_generation = 0 
        # reference
        self.archive_maximum = np.max(self.population_obj, axis=0)
        self.archive_minimum = np.min(self.population_obj, axis=0)
        self.ideal_point = copy.deepcopy(self.archive_minimum)
        self.problem_ref_points = self.problem.get_ref_set(
            n_ref_points=self.n_ref_points)
        # indicators
        self.igd_his = []
        self.initial_igd = self.get_igd()
        self.last_igd = self.initial_igd
        self.best_igd = self.initial_igd
        self.hv_his = []
        self.initial_hv = self.get_hv()
        self.last_hv = self.initial_hv
        self.best_hv = self.initial_hv
        self.metadata = {'X':[],'cost':[]}
        self.update_information()

    def get_neighborhoods(self):
        neighborhoods = []  # the i-th element save the index of the neighborhoods of it
        for i in range(len(self.weights)):
            sorted_weights = self.moead_sort_weights(
                self.weights[i], self.weights)
            neighborhoods.append(
                sorted_weights[:self.moead_neighborhood_maxsize])
        return neighborhoods

    def get_weights(self, n_obj):
        weights = None
        if n_obj == 2:
            weights = self.normal_boundary_weights(n_obj,99, 0)
        elif n_obj == 3:
            weights = self.normal_boundary_weights(n_obj, 13, 0)
        elif n_obj == 5:
            weights = self.normal_boundary_weights(n_obj, 5, 0)
        elif n_obj == 7:
            weights = self.normal_boundary_weights(n_obj, 3, 2)
        elif n_obj == 8:
            weights = self.normal_boundary_weights(n_obj, 3, 1)
        elif n_obj == 10:
            weights = self.normal_boundary_weights(n_obj,2, 2)
        else:
            weights = self.random_weights(n_obj, self.population_size)
        return weights

    def moead_update_ideal(self, solution_obj):
        for i in range(solution_obj.shape[-1]):
            self.ideal_point[i] = min(
                self.ideal_point[i], solution_obj[i])

    def run_episode(self, problem):
        self.init_population(problem)
        
        while not self.done:
            subproblems = self.moead_get_subproblems()
            self.offspring_list = []
            self.offspring_obj_list = []
            for index in subproblems:
                mating_indices = self.moead_get_mating_indices(index)
                mating_population = [self.population[i] for i in mating_indices]
                if index in mating_indices:
                    mating_indices.remove(index)

                parents = [self.population[index]] + \
                            [self.population[i] for i in
                            self.rng.choice(mating_indices, 2, replace=False)]
                offspring = self.sbx(problem, parents)[0]
                offspring = self.pm(problem, offspring)
                offspring_obj = problem.eval(offspring)
                self.fes += 1
                safe_extend(self.offspring_list,offspring)
                safe_extend(self.offspring_obj_list,offspring_obj)
                # if offspring.ndim == 1:  # Check if offspring is a 1D array
                for child, child_obj in zip([offspring], [offspring_obj]):
                    self.moead_update_ideal(child_obj)
                    self.moead_update_solution(child, child_obj, mating_indices)
                # else:
                #     for child, child_obj in zip(offspring, offspring_obj):
                #         self.moead_update_ideal(child_obj)
                #         self.moead_update_solution(child, child_obj, mating_indices)
            self.moead_generation += 1
            self.last_igd = self.get_igd()
            self.best_igd = min(self.best_igd, self.last_igd)
            self.last_hv = self.get_hv()
            self.best_hv = max(self.best_hv, self.last_hv)
            self.update_information()
            print("igd:{},hv:{}".format(self.last_igd,self.last_hv))
            if self.fes >= self.max_fes:
                self.done = True
                print("fes:{},last_igd:{},last_hv:{}".format(self.fes,self.last_igd,self.last_hv))
            else:
                self.done = False
            
    def update_information(self):
        index =  self.find_non_dominated_indices(self.population_obj)
        self.cost = [copy.deepcopy(self.population_obj[i]) for i in index] # parato front
        self.metadata['X'].append(copy.deepcopy(self.population))
        self.metadata['cost'].append(copy.deepcopy(self.population_obj))
        
    def find_non_dominated_indices(self, population_list):
        """
        此函数用于找出种群中的支配解
        :param population_list: 种群的目标值的列表，列表中的每个元素是一个代表单个解目标值的列表
        :return: 支配解的列表
        """
        # 将列表转换为 numpy 数组
        population = np.array(population_list)
        n_solutions = population.shape[0]
        is_dominated = np.zeros(n_solutions, dtype=bool)

        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # 检查是否存在解 j 支配解 i
                    if np.all(population[j] <= population[i]) and np.any(population[j] < population[i]):
                        is_dominated[i] = True
                        break

        # 找出非支配解的索引
        non_dominated_indices = np.where(~is_dominated)[0]
        return non_dominated_indices

    def moead_calculate_fitness(self, solution_obj, weights):
        return chebyshev(solution_obj, self.ideal_point, weights)

    def moead_update_solution(self, solution, solution_obj, mating_indices):
        """
        repair solution, make constraint satisfiable
        :param solution:
        :param mating_indices:
        :return:
        """

        c = 0
        self.rng.shuffle(mating_indices)

        for i in mating_indices:
            candidate = self.population[i]
            candidate_obj = self.population_obj[i]
            weights = self.weights[i]
            replace = False
            if self.moead_calculate_fitness(solution_obj, weights) < self.moead_calculate_fitness(candidate_obj,
                                                                                                  weights):
                replace = True

            if replace:
                self.population[i] = copy.deepcopy(solution)
                self.population_obj[i] = copy.deepcopy(solution_obj)
                c = c + 1

            if c >= self.moead_eta:
                break

    @staticmethod
    def moead_sort_weights(base, weights):
        """Returns the index of weights nearest to the base weight."""

        def compare(weight1, weight2):
            dist1 = math.sqrt(
                sum([math.pow(base[i] - weight1[1][i], 2.0) for i in range(len(base))]))
            dist2 = math.sqrt(
                sum([math.pow(base[i] - weight2[1][i], 2.0) for i in range(len(base))]))

            if dist1 < dist2:
                return -1
            elif dist1 > dist2:
                return 1
            else:
                return 0

        sorted_weights = sorted(
            enumerate(weights), key=functools.cmp_to_key(compare))
        return [i[0] for i in sorted_weights]

    def moead_get_subproblems(self):
        """
        Determines the subproblems to search.
        If :code:`utility_update` has been set, then this method follows the
        utility-based moea/D search.
        Otherwise, it follows the original moea/D specification.
        """
        indices = list(range(self.population_size))
        self.rng.shuffle(indices)
        return indices

    def moead_get_mating_indices(self, index):
        """Determines the mating indices.

        Returns the population members that are considered during mating.  With
        probability :code:`delta`, the neighborhood is returned.  Otherwise,
        the entire population is returned.
        """
        if self.rng.uniform(0.0, 1.0) <= self.moead_delta:
            return self.neighborhoods[index][:self.moead_neighborhood_size]
        else:
            return list(range(self.population_size))

    def get_hv(self,n_samples=1e5):
        if self.problem.n_obj <= 3 or self.population_size <= 50:
            hv_fast = False
        else:
            hv_fast = True
        if not hv_fast:
            # Calculate the exact hv value
            hyp = Hypervolume(minimum=[0 for _ in range(
                self.n_obj)], maximum=self.archive_maximum)
            hv_value = hyp.calculate(np.array(self.population_obj))
        else:
            # Estimate the hv value by Monte Carlo
            popobj = copy.deepcopy(self.population_obj)
            optimum = self.problem_ref_points
            fmin = np.clip(np.min(popobj, axis=0), np.min(popobj), 0)
            fmax = np.max(optimum, axis=0)

            popobj = (popobj - np.tile(fmin, (self.population_size, 1))) / (
                np.tile(1.1 * (fmax - fmin), (self.population_size, 1)))
            index = np.all(popobj < 1, 1).tolist()
            popobj = popobj[index]
            if popobj.shape[0] <= 1:
                hv_value = 0
                self.hv_his.append(hv_value)
                self.hv_last5.add(hv_value)
                self.hv_running.update(np.array([hv_value]))
                return hv_value
            assert np.max(popobj) < 1
            hv_maximum = np.ones([self.n_obj])
            hv_minimum = np.min(popobj, axis=0)
            n_samples_hv = int(n_samples)
            samples = np.zeros([n_samples_hv, self.n_obj])
            for i in range(self.n_obj):
                samples[:, i] = self.rng.uniform(
                    hv_minimum[i], hv_maximum[i], n_samples_hv)
            for i in range(popobj.shape[0]):
                domi = np.ones([samples.shape[0]], dtype=bool)
                m = 0
                while m < self.n_obj and any(domi):
                    domi = np.logical_and(domi, popobj[i, m] <= samples[:, m])
                    m += 1
                save_id = np.logical_not(domi)
                samples = samples[save_id, :]
            hv_value = np.prod(hv_maximum - hv_minimum) * (
                    1 - samples.shape[0] / n_samples_hv)
        self.hv_his.append(hv_value)
        return hv_value

    def get_igd(self):
        igd_calculator = InvertedGenerationalDistance(reference_set=self.problem_ref_points)
        igd_value = igd_calculator.calculate(self.population_obj)
        self.igd_his.append(igd_value)

        return igd_value

    def close(self):
        self.reset()
    
    def normal_boundary_weights(self,nobjs, divisions_outer, divisions_inner=0):
        """Returns weights generated by the normal boundary method.

        The weights produced by this method are uniformly distributed on the
        hyperplane intersecting

            [(1, 0, ..., 0), (0, 1, ..., 0), ..., (0, 0, ..., 1)].

        Parameters
        ----------
        nobjs : int
            The number of objectives.
        divisions_outer : int
            The number of divisions along the outer set of weights.
        divisions_inner : int (optional)
            The number of divisions along the inner set of weights.
        """

        def generate_recursive(weights, weight, left, total, index):
            if index == nobjs - 1:
                weight[index] = float(left) / float(total)
                weights.append(copy.copy(weight))
            else:
                for i in range(left + 1):
                    weight[index] = float(i) / float(total)
                    generate_recursive(weights, weight, left - i, total, index + 1)

        def generate_weights(divisions):
            weights = []
            generate_recursive(weights, [0.0] * nobjs, divisions, divisions, 0)
            return weights

        weights = generate_weights(divisions_outer)

        if divisions_inner > 0:
            inner_weights = generate_weights(divisions_inner)

            for i in range(len(inner_weights)):
                weight = inner_weights[i]

                for j in range(len(weight)):
                    weight[j] = (1.0 / nobjs + weight[j]) / 2.0

                weights.append(weight)

        return weights

    def random_weights(self,nobjs, population_size):
        """Returns a set of randomly-generated but uniformly distributed weights.
        
        Simply producing N randomly-generated weights does not necessarily produce
        uniformly-distributed weights.  To help produce more uniformly-distributed
        weights, this method picks weights from a large collection of randomly-
        generated weights such that the distances between weights is maximized.
        
        Parameters
        ----------
        nobjs : int
            The number of objectives.
        population_size : int
            The number of weights to generate.
        """
        
        weights = []
        
        if nobjs == 2:
            weights = [[1, 0], [0, 1]]
            weights.extend([(i/(population_size-1.0), 1.0-i/(population_size-1.0)) for i in range(1, population_size-1)])
        else:
            # generate candidate weights
            candidate_weights = []
            
            for i in range(population_size*50):
                random_values = [np.random.uniform(0.0, 1.0) for _ in range(nobjs)]
                candidate_weights.append([x/sum(random_values) for x in random_values])
            
            # add weights for the corners
            for i in range(nobjs):
                weights.append([0]*i + [1] + [0]*(nobjs-i-1))
                
            # iteratively fill in the remaining weights by finding the candidate
            # weight with the largest distance from the assigned weights
            while len(weights) < population_size:
                max_index = -1
                max_distance = -POSITIVE_INFINITY
                
                for i in range(len(candidate_weights)):
                    distance = POSITIVE_INFINITY
                    
                    for j in range(len(weights)):
                        temp = math.sqrt(sum([math.pow(candidate_weights[i][k]-weights[j][k], 2.0) for k in range(nobjs)]))
                        distance = min(distance, temp)
                        
                    if distance > max_distance:
                        max_index = i
                        max_distance = distance
                        
                weights.append(candidate_weights[max_index])
                del candidate_weights[max_index]
                
        return weights

    def sbx(self, problem,parents,probability=1.0, distribution_index=20.0):
        def sbx_crossover(x1, x2, lb, ub, distribution_index):
            dx = x2 - x1

            if dx > EPSILON:
                if x2 > x1:
                    y2 = x2
                    y1 = x1
                else:
                    y2 = x1
                    y1 = x2

                beta = 1.0 / (1.0 + (2.0 * (y1 - lb) / (y2 - y1)))
                alpha = 2.0 - pow(beta, distribution_index + 1.0)
                rand = self.rng.uniform(0.0, 1.0)

                if rand <= 1.0 / alpha:
                    alpha = alpha * rand
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0))
                else:
                    alpha = alpha * rand;
                    alpha = 1.0 / (2.0 - alpha)
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0))

                x1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                beta = 1.0 / (1.0 + (2.0 * (ub - y2) / (y2 - y1)));
                alpha = 2.0 - pow(beta, distribution_index + 1.0);

                if rand <= 1.0 / alpha:
                    alpha = alpha * rand
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0));
                else:
                    alpha = alpha * rand
                    alpha = 1.0 / (2.0 - alpha)
                    betaq = pow(alpha, 1.0 / (distribution_index + 1.0));

                x2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1));

                # randomly swap the values
                if bool(self.rng.randint(0, 2)):
                    x1, x2 = x2, x1

                x1 = np.clip(x1, lb, ub)
                x2 = np.clip(x2, lb, ub)

            return x1, x2
        
        child1 = copy.deepcopy(parents[0])
        child2 = copy.deepcopy(parents[1])
        if self.rng.uniform(0.0, 1.0) <= probability:
            nvars = problem.n_var

            for i in range(nvars):
                if self.rng.uniform(0.0, 1.0) <= 0.5:
                    x1 = float(child1[i])
                    x2 = float(child2[i])
                    lb = problem.lb[i]
                    ub = problem.ub[i]

                    x1, x2 = sbx_crossover(x1, x2, lb, ub,distribution_index=distribution_index)
                    child1[i] = x1
                    child2[i]= x2

        return [child1, child2]
 
    def pm(self, problem,parent, probability=1.0, distribution_index=20.0):
        def pm_mutation(x, lb, ub,distribution_index):
            u = self.rng.uniform(0, 1)
            dx = ub - lb

            if u < 0.5:
                bl = (x - lb) / dx
                b = 2.0 * u + (1.0 - 2.0 * u) * pow(1.0 - bl, distribution_index + 1.0)
                delta = pow(b, 1.0 / (distribution_index + 1.0)) - 1.0
            else:
                bu = (ub - x) / dx
                b = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * pow(1.0 - bu, distribution_index + 1.0)
                delta = 1.0 - pow(b, 1.0 / (distribution_index + 1.0))

            x = x + delta * dx
            x = np.clip(x, lb, ub)

            return x
        child = copy.deepcopy(parent)
        probability /= float(problem.n_var)

        for i in range(problem.n_var):
            if self.rng.uniform(0.0, 1.0) <= probability:
                child[i] = pm_mutation(float(child[i]),
                                        problem.lb[i],
                                        problem.ub[i],
                                        distribution_index=distribution_index)
        return child

    

## Aggregate functions
def chebyshev(solution_obj, ideal_point, weights, min_weight=0.0001):
    """Chebyshev (Tchebycheff) fitness of a solution with multiple objectives.

    This function is designed to only work with minimized objectives.

    Parameters
    ----------
    solution : Solution
        The solution.
    ideal_point : list of float
        The ideal point.
    weights : list of float
        The weights.
    min_weight : float
        The minimum weight allowed.
    """
    objs = solution_obj
    n_obj = objs.shape[-1]
    return max([max(weights[i], min_weight) * (objs[i] - ideal_point[i]) for i in range(n_obj)])

def pbi(solution_obj, ideal_point, weights, theta=5):
    """Penalty-based boundary intersection fitness of a solution with multiple objectives.
    
    Requires numpy.  This function is designed to only work with minimized
    objectives.
    
    Callers need to set the theta value by using
        functools.partial(pbi, theta=0.5)
    
    Parameters
    ----------
    solution : Solution
        The solution.
    ideal_point: list of float
        The ideal point.
    weights : list of float
        The weights.
    theta : float
        The theta value.
    """
    try:
        import numpy as np
    except:
        print("The pbi function requires numpy.", file=sys.stderr)
        raise

    w = np.array(weights)
    z_star = np.array(ideal_point)
    F = np.array(solution_obj)

    d1 = np.linalg.norm(np.dot((F - z_star), w)) / np.linalg.norm(w)
    d2 = np.linalg.norm(F - (z_star + d1 * w))

    return (d1 + theta * d2).tolist()

## indicators
POSITIVE_INFINITY = float("inf")
EPSILON = sys.float_info.epsilon
class Indicator(object):
    #__metaclass = ABCMeta

    def __init__(self):
        super(Indicator, self).__init__()

    def __call__(self, set):
        return self.calculate(set)

    def calculate(self, set):
        raise NotImplementedError("method not implemented")


class Hypervolume(Indicator):
    # 只适用于最小化问题

    def __init__(self, reference_set=None, minimum=None, maximum=None):
        super(Hypervolume, self).__init__()
        if reference_set is not None:
            if minimum is not None or maximum is not None:
                raise ValueError("minimum and maximum must not be specified if reference_set is defined")
            self.minimum, self.maximum = normalize(reference_set)
        else:
            if minimum is None or maximum is None:
                raise ValueError("minimum and maximum must be specified when no reference_set is defined")
            self.minimum, self.maximum = minimum, maximum

    def invert(self, solution_normalized_obj: np.ndarray):
        for i in range(solution_normalized_obj.shape[1]):
            solution_normalized_obj[:, i] = 1.0 - np.clip(solution_normalized_obj[:, i], 0.0, 1.0)
        return solution_normalized_obj

    def dominates(self, solution1_obj, solution2_obj, nobjs):
        better = False
        worse = False

        for i in range(nobjs):
            if solution1_obj[i] > solution2_obj[i]:
                better = True
            else:
                worse = True
                break
        return not worse and better

    def swap(self, solutions_obj, i, j):
        solutions_obj[[i, j]] = solutions_obj[[j, i]]
        return solutions_obj

    def filter_nondominated(self, solutions_obj, nsols, nobjs):
        i = 0
        n = nsols
        while i < n:
            j = i + 1
            while j < n:
                if self.dominates(solutions_obj[i], solutions_obj[j], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, j, n)
                elif self.dominates(solutions_obj[j], solutions_obj[i], nobjs):
                    n -= 1
                    solutions_obj = self.swap(solutions_obj, i, n)
                    i -= 1
                    break
                else:
                    j += 1
            i += 1
        return n

    def surface_unchanged_to(self, solutions_normalized_obj, nsols, obj):
        return np.min(solutions_normalized_obj[:nsols, obj])

    def reduce_set(self, solutions, nsols, obj, threshold):
        i = 0
        n = nsols
        while i < n:
            if solutions[i, obj] <= threshold:
                n -= 1
                solutions = self.swap(solutions, i, n)
            else:
                i += 1
        return n

    def calc_internal(self, solutions_obj: np.ndarray, nsols, nobjs):
        volume = 0.0
        distance = 0.0
        n = nsols

        while n > 0:
            nnondom = self.filter_nondominated(solutions_obj, n, nobjs - 1)

            if nobjs < 3:
                temp_volume = solutions_obj[0][0]
            else:
                temp_volume = self.calc_internal(solutions_obj, nnondom, nobjs - 1)

            temp_distance = self.surface_unchanged_to(solutions_obj, n, nobjs - 1)
            volume += temp_volume * (temp_distance - distance)
            distance = temp_distance
            n = self.reduce_set(solutions_obj, n, nobjs - 1, distance)

        return volume

    def calculate(self, solutions_obj: np.ndarray):

        # 对可行解进行归一化
        solutions_normalized_obj = normalize(solutions_obj, self.minimum, self.maximum)

        # 筛选出所有目标值都小于等于 1.0 的解
        valid_mask = np.all(solutions_normalized_obj <= 1.0, axis=1)
        valid_feasible = solutions_normalized_obj[valid_mask]

        if valid_feasible.size == 0:
            return 0.0

        # 对可行解进行反转操作
        inverted_feasible = self.invert(valid_feasible)

        # 计算超体积
        nobjs = inverted_feasible.shape[1]
        return self.calc_internal(inverted_feasible, len(inverted_feasible), nobjs)


class InvertedGenerationalDistance(Indicator):
    def __init__(self, reference_set):
        super(InvertedGenerationalDistance, self).__init__()
        self.reference_set = reference_set


    def calculate(self, set):
        return sum([distance_to_nearest(s, set) for s in self.reference_set])/ len(self.reference_set)
                        

def distance_to_nearest(solution_obj, set):
    if len(set) == 0:
        return POSITIVE_INFINITY

    return min([euclidean_dist(solution_obj, s) for s in set])


def euclidean_dist(x, y):
    if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
        print("x:", x)
        print("y:", y)
        raise TypeError("x and y must be lists or tuples.")

    return math.sqrt(sum([math.pow(x[i] - y[i], 2.0) for i in range(len(x))]))


def normalize(solutions_obj: np.ndarray, minimum: np.ndarray = None, maximum: np.ndarray = None) -> np.ndarray:
    """Normalizes the solution objectives.

    Normalizes the objectives of each solution within the minimum and maximum
    bounds.  If the minimum and maximum bounds are not provided, then the
    bounds are computed based on the bounds of the solutions.

    Parameters
    ----------
    solutions_obj : numpy.ndarray
        The solutions to be normalized. It should be a 2D numpy array.
    minimum : numpy.ndarray
        The minimum values used to normalize the objectives.
    maximum : numpy.ndarray
        The maximum values used to normalize the objectives.

    Returns
    -------
    numpy.ndarray
        The normalized solutions.
    """
    # 如果输入数组为空，直接返回空数组
    if len(solutions_obj) == 0:
        return solutions_obj

    # 获取目标的数量
    n_obj = solutions_obj.shape[1]

    # 如果 minimum 或 maximum 未提供，则计算它们
    if minimum is None or maximum is None:
        if minimum is None:
            minimum = np.min(solutions_obj, axis=0)
        if maximum is None:
            maximum = np.max(solutions_obj, axis=0)

    # 检查是否有目标的范围为空
    if np.any(maximum - minimum < EPSILON):
        raise ValueError("objective with empty range")

    # 进行归一化操作
    solutions_normalized_obj = (solutions_obj - minimum) / (maximum - minimum)

    return solutions_normalized_obj
    
def safe_extend(lst, items):
    # 如果 items 是 None，跳过处理
    if items is None:
        return
    # 判断是否是 list 或 ndarray
    if isinstance(items, (list, np.ndarray)):
        # 获取维度
        shape = np.shape(items)
        if len(shape) > 1:  # 多维的：用 extend
            lst.extend(items)
        else:               # 一维的：用 append
            lst.append(items)
    else:
        # 不是列表/数组，默认 append
        lst.append(items)
