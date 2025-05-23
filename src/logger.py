import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Union, Callable
import argparse
params = {
    'axes.labelsize': '25',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'lines.linewidth': '3',
    'legend.fontsize': '24',
    'figure.figsize': '20,11',
}
plt.rcParams.update(params)

markers = ['o', '^', '*', 'O', 'v', 'x', 'X', 'd', 'D', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H']
colors = ['b', 'g', 'orange', 'r', 'purple', 'brown', 'grey', 'limegreen', 'turquoise', 'olivedrab', 'royalblue', 'darkviolet', 
          'chocolate', 'crimson', 'teal','seagreen', 'navy', 'deeppink', 'maroon', 'goldnrod', 
          ]


# def data_wrapper_prsr(data, ):
#     res = []
#     for key in data.keys():
#         res.append(np.array(data[key][:, -1, 3]))
#     return np.array(res)


def data_wrapper_cost(data, ):
    return np.array(data)[:, :, -1]


# def data_wrapper_prsr_test(data, ):
#     return np.array(data)[:,-1, 3]


def to_label(agent_name: str) -> str:
    """
    # Introduction
    Converts an agent's name to a simplified label for display or logging purposes.
    # Args:
    - agent_name (str): The name of the agent to be converted.
    # Returns:
    - str: The simplified label. If the agent name is 'L2L_Agent', returns 'RNN-OI'. 
        If the agent name ends with '_Agent' or '_agent', removes this suffix. Otherwise, returns the original name.
    """
    
    label = agent_name
    if label == 'L2L_Agent':
        return 'RNN-OI'
    if len(label) > 6 and (label[-6:] == '_Agent' or label[-6:] == '_agent'):
        label = label[:-6]
    return label


class Basic_Logger:
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0

    def get_average_data(self, results: dict, norm: bool=False, data_wrapper: Callable = None):
        """
        # Introduction
        Computes the average and standard deviation of each agent's results across multiple problems, with optional normalization and data preprocessing.
        # Args:
        - results (dict): Nested dictionary containing results structured as `results[problem][agent] = values`.They are the data to be process.
        - norm (bool, optional): If True, applies min-max normalization to the values for each agent and problem. Defaults to False.
        - data_wrapper (Callable, optional): A function to preprocess each data item for each agent under each problem. Defaults to None.
        # Returns:
        - tuple: A tuple `(avg_data, std_data)` where:
            - `avg_data` (dict): Dictionary mapping each agent to their average value across all problems.
            - `std_data` (dict): Dictionary mapping each agent to their standard deviation across all problems.
        # Raises:
        - KeyError: If the structure of `results` does not match the expected format.
        - ValueError: If normalization is requested but the data has zero range (max equals min).
        """
        """
        Get the average and standard deviation of each agent from the results
        :param results  dict: The data to be process
        :param norm     bool: Whether to min-max normalize data
        :param data_wrapper callable: A data pre-processing function wrapper applied to each data item of each agent under each problem
        """
        problems=[]
        agents=[]

        for problem in results.keys():
            problems.append(problem)
        for agent in results[problems[0]].keys():
            agents.append(agent)
        avg_data={}
        std_data={}
        for agent in agents:
            avg_data[agent]=[]
            std_data[agent]=[]
            for problem in problems:
                values = results[problem][agent]
                if data_wrapper is not None:
                    values = data_wrapper(values)
                if norm:
                    values = (values - np.min((values))) / (np.max(values) - np.min(values))
                std_data[agent].append(np.std(values, -1))
                avg_data[agent].append(np.mean(values, -1))
            avg_data[agent] = np.mean(avg_data[agent], 0)
            std_data[agent] = np.mean(std_data[agent], 0)
        return avg_data, std_data

    @staticmethod
    def data_wrapper_cost_rollout(data):
        res = np.array(data)
        return res[:, -1]
    
    def cal_scores1(self, D: dict, maxf: float):
        """
        # Introduction
        Calculates a custom score for each agent based on the provided dictionary of values and a normalization factor, intended as a tool function for the CEC metric.
        # Args:
        - D (dict): A dictionary where each key represents an agent and the value is an array-like structure representing the results in all test problems.
        - maxf (float): A normalization factor used to scale the minimum values for each agent.
        # Returns:
        - np.ndarray: An array of computed scores for each agent.
        # Notes:
        The function computes a score for each agent by first calculating a scaled sum of the minimum values in their associated array, then normalizing and scaling the result to produce the final score.
        """
        """
        Tool function for CEC metric
        """
        SNE = []
        for agent in D.keys():
            values = D[agent]
            sne = 0.5 * np.sum(np.min(values, -1) / maxf)
            SNE.append(sne)
        SNE = np.array(SNE)
        score1 = (1 - (SNE - np.min(SNE)) / SNE) * 50
        return score1

    def get_random_baseline(self, results: dict, fes: Optional[Union[int, float]]):
        """
        # Introduction
        Calculates baseline statistics from Random Search results for normalization and comparison purposes in optimization experiments.
        # Args:
        - results (dict): The results data. Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - fes (Optional[Union[int, float]]): The maximum number of function evaluations used for normalization in the 'fes' baseline calculation.
        # Returns:
        - dict: A dictionary containing the following baseline statistics:
            - 'complexity_avg': Mean log-complexity across problems.
            - 'complexity_std': Standard deviation of log-complexity.
            - 'fes_avg': Mean normalized log function evaluation score.
            - 'fes_std': Mean standard deviation of normalized log function evaluation score.
            - 'cost_avg': Mean log-inverse final cost.
            - 'cost_std': Mean standard deviation of log-inverse final cost.
        # Notes:
        - Assumes that the input `results` dictionary is structured with keys for 'T1', 'T2', 'T0', 'fes', and 'cost', each containing per-problem results for 'Random_search'.
        - Uses numpy for array operations and statistical calculations.
        """
        """
        Get the results of Random Search for further usage, i.e., for normalization
        """
        baseline = {}
        T1 = []
        T2 = []
        for pname in results['T1'].keys():
            T1.append(results['T1'][pname]['Random_search'])
            T2.append(results['T2'][pname]['Random_search'])
        baseline['complexity_avg'] = np.mean(np.log10(1. / (np.array(T2) - np.array(T1)) / results['T0']))
        baseline['complexity_std'] = np.std(np.log10(1. / (np.array(T2) - np.array(T1)) / results['T0']))
        avg = []
        std = []
        for problem in results['fes'].keys():
            g = np.log10(fes/np.array(results['fes'][problem]['Random_search']))
            avg.append(g.mean())
            std.append(g.std())
        baseline['fes_avg'] = np.mean(avg)
        baseline['fes_std'] = np.mean(std)
        avg = []
        std = []
        for problem in results['cost'].keys():
            g = np.log10(1/(np.array(results['cost'][problem]['Random_search'])[:, -1]+1))
            avg.append(g.mean())
            std.append(g.std()) 
        baseline['cost_avg'] = np.mean(avg)
        baseline['cost_std'] = np.mean(std)
        return baseline

    def gen_algorithm_complexity_table(self, results: dict, out_dir: str) -> None:
        """
        # Introduction
        Generates and saves an Excel table summarizing algorithm complexity metrics for different agents.
        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - out_dir (str): The output directory where the Excel file will be saved.
        # Returns:
        - None
        # Details:
        For each agent, the function computes the mean values of T1 and T2 across all problem names, calculates the ratio (T2-T1)/T0, and stores these metrics in a table. The resulting table is saved as 'algorithm_complexity.xlsx' in the specified output directory.
        """
        """
        Store algorithm complexity data as excel table 
        """
        save_list=[]
        t0=results['T0']
        ratios=[]
        t1_list = {}
        t2_list = {}
        indexs=list(results['T1'][list(results['T1'].keys())[0]].keys())
        columns=['T0','T1','T2','(T2-T1)/T0']
        for agent in indexs:
            t1_list[agent] = []
            t2_list[agent] = []
            for pname in results['T1'].keys():
                t1_list[agent].append(results['T1'][pname][agent])
                t2_list[agent].append(results['T2'][pname][agent])
            t1_list[agent] = np.mean(t1_list[agent])
            t2_list[agent] = np.mean(t2_list[agent])
            ratios.append((t2_list[agent] - t1_list[agent])/t0)

        n=len(indexs)
        data=np.zeros((n,4))
        data[:,0]=t0
        data[:,1]=list(t1_list.values())
        data[:,2]=list(t2_list.values())
        data[:,3]=ratios
        table=pd.DataFrame(data=np.round(data,2),index=indexs,columns=columns)
        table.to_excel(os.path.join(out_dir,'algorithm_complexity.xlsx'))

    def gen_agent_performance_table(self, results: dict, out_dir: str) -> None:
        """
        # Introduction
        Generates and saves Excel tables summarizing the performance statistics of different agents on various problems.
        For each agent and problem, the function computes the Worst, Best, Median, Mean, and Standard Deviation (Std) of the final cost values across multiple runs, and stores these statistics in an Excel file.
        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - out_dir (str): The directory path where the generated Excel files will be saved.
        # Returns:
        - None
        # Side Effects:
        - Writes one Excel file per agent to the specified output directory, each containing a table of performance statistics for all problems.
        # Raises:
        - KeyError: If the expected keys are missing in the `results` dictionary.
        - OSError: If there is an issue saving the Excel files to the specified directory.
        """
        """
        Store the `Worst`, `Best`, `Median`, `Mean` and `Std` of cost results of each agent as excel
        """
        total_cost=results['cost']
        table_data={}
        indexs=[]
        columns=['Worst','Best','Median','Mean','Std']
        for problem,value in total_cost.items():
            indexs.append(problem)
            problem_cost=value
            for alg,alg_cost in problem_cost.items():
                n_cost=[]
                for run in alg_cost:
                    n_cost.append(run[-1])
                # if alg == 'MadDE' and problem == 'F5':
                #     for run in alg_cost:
                #         print(len(run))
                #     print(len(n_cost))
                best=np.min(n_cost)
                best=np.format_float_scientific(best,precision=3,exp_digits=3)
                worst=np.max(n_cost)
                worst=np.format_float_scientific(worst,precision=3,exp_digits=3)
                median=np.median(n_cost)
                median=np.format_float_scientific(median,precision=3,exp_digits=3)
                mean=np.mean(n_cost)
                mean=np.format_float_scientific(mean,precision=3,exp_digits=3)
                std=np.std(n_cost)
                std=np.format_float_scientific(std,precision=3,exp_digits=3)

                if not alg in table_data:
                    table_data[alg]=[]
                table_data[alg].append([worst,best,median,mean,std])
        for alg,data in table_data.items():
            dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
            #print(dataframe)
            dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_table.xlsx'))

    def gen_overall_tab(self, results: dict, out_dir: str) -> None:
        """
        # Introduction
        Generates and stores an Excel table summarizing the overall results of optimization experiments, including objective values (costs), performance gap with CMAES, and consumed function evaluations (FEs) for each optimizer and problem.
        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - out_dir (str): Directory path where the resulting Excel file ('overall_table.xlsx') will be saved.
        # Returns:
        - None
        # Raises:
        - KeyError: If expected keys are missing in the `results` dictionary.
        - AttributeError: If `self.config.test_run` is not defined.
        - ValueError: If the data shapes in `results` do not match expectations.
        # Notes:
        - The resulting Excel file contains a multi-indexed table with optimizers as rows and (problem, metric) pairs as columns.
        - Metrics include average and standard deviation of objective values, gap with CMAES, and function evaluations, all formatted for readability.
        """
        """
        Store the overall results inculding `objective values` (costs), `gap` with CMAES and the consumed `FEs` as excel
        """
        # get multi-indexes first
        problems = []
        statics = ['Obj','Gap','FEs']
        optimizers = []
        for problem in results['cost'].keys():
            problems.append(problem)
        for optimizer in results['cost'][problems[0]].keys():
            optimizers.append(optimizer)
        multi_columns = pd.MultiIndex.from_product(
            [problems,statics], names=('Problem', 'metric')
        )
        df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                                index=optimizers,
                                columns=multi_columns)

        # calculate baseline1 cmaes
        cmaes_obj = {}
        for problem in problems:
            blobj_problem = results['cost'][problem]['CMAES']  # 51 * record_length
            objs = []
            for run in range(len(blobj_problem)):
                objs.append(blobj_problem[run][-1])
            cmaes_obj[problem] = sum(objs) / len(blobj_problem)

        # calculate baseline2 random_search
        rs_obj = {}
        for problem in problems:
            blobj_problem = results['cost'][problem]['Random_search']  # 51 * record_length
            objs = []
            for run in range(len(blobj_problem)):
                objs.append(blobj_problem[run][-1])
            rs_obj[problem] = sum(objs) / len(blobj_problem)

        # calculate each Obj
        for problem in problems:
            for optimizer in optimizers:
                obj_problem_optimizer = results['cost'][problem][optimizer]
                objs_ = []
                for run in range(len(obj_problem_optimizer)):
                    objs_.append(obj_problem_optimizer[run][-1])
                avg_obj = sum(objs_) / len(obj_problem_optimizer)
                std_obj = np.std(objs_)
                df_results.loc[optimizer, (problem, 'Obj')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_obj, precision=3, exp_digits=1) + ")"
                # calculate each Gap
                df_results.loc[optimizer, (problem, 'Gap')] = "%.3f" % (1-(rs_obj[problem]-avg_obj) / (rs_obj[problem]-cmaes_obj[problem]+1e-10))
                fes_problem_optimizer = np.array(results['fes'][problem][optimizer])
                avg_fes = np.mean(fes_problem_optimizer)
                std_fes = np.std(fes_problem_optimizer)
                df_results.loc[optimizer, (problem, 'FEs')] = np.format_float_scientific(avg_fes, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_fes, precision=3, exp_digits=1) + ")"
        df_results.to_excel(out_dir+'overall_table.xlsx')

    def aei_cost(self, cost_data: dict, baseline: dict, ignore: Optional[list]=None):
        """
        # Introduction
        Calculates the Aggregated Evaluation Indicator (AEI) cost for different agents based on provided cost data and a baseline. Optionally ignores specified agents.
        # Args:
        - cost_data (dict): Part of the result data,the results['cost']. Also a nested dictionary containing experimental results structured as `dict[problem][algorithm][run]`.
        - baseline (dict): A dictionary containing baseline statistics, structured as 'dict[metric]'.The metric includes 'complexity_avg', 'complexity_std', 'fes_avg', 'fes_std', 'cost_avg', and 'cost_std'.
        - dict: A dictionary containing the following baseline statistics:
        - ignore (Optional[list]): A list of agent names to ignore during calculation. Defaults to None.
        # Returns:
        - results_cost (dict): A dictionary mapping each agent to their computed AEI cost values.
        - aei_mean (float): The mean AEI value across agents (excluding ignored ones).
        - aei_std (float): The standard deviation of AEI values across agents (excluding ignored ones).
        """
        
        avg = baseline['cost_avg']
        problems = cost_data.keys()
        agents = cost_data[list(problems)[0]].keys()
        results_cost = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            costs_problem = []
            for problem in problems:
                cost_ = np.log10(1/(np.array(cost_data[problem][agent])[:, -1]+1))
                costs_problem.append(cost_.mean())
            results_cost[agent] = np.exp((costs_problem - avg) * 1)
        aei_mean, aei_std = self.cal_aei(results_cost, agents, ignore)
        return results_cost, aei_mean, aei_std
    
    def aei_fes(self, fes_data: dict, baseline: dict, maxFEs: Optional[Union[int, float]]=20000, ignore: Optional[list]=None):
        """
        # Introduction
        Computes the Aggregated Evaluation Indicator (AEI) for function evaluation steps (FEs) across multiple agents and problems, comparing them to a baseline. The method processes FEs data, applies logarithmic scaling, and calculates AEI statistics, optionally ignoring specified agents.
        # Args:
        - fes_data (dict):Part of the result data,the results['fes'].Also a nested dictionary containing experimental results structured as `dict[problem][algorithm][run]`.
        - baseline (dict): A dictionary containing baseline statistics, structured as 'dict[metric]'.The metric includes 'complexity_avg', 'complexity_std', 'fes_avg', 'fes_std', 'cost_avg', and 'cost_std'.
        - maxFEs (Optional[Union[int, float]], default=20000): The maximum number of function evaluations allowed, used for normalization.
        - ignore (Optional[list], default=None): List of agent names to ignore during computation.
        # Returns:
        - results_fes (dict): Dictionary mapping each agent to their computed AEI values across problems.
        - aei_mean (float): Mean AEI value across all considered agents.
        - aei_std (float): Standard deviation of AEI values across all considered agents.
        """
        
        avg = baseline['fes_avg']
        problems = fes_data.keys()
        agents = fes_data[list(problems)[0]].keys()
        results_fes = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            fes_problem = []
            for problem in problems:
                if agent == 'L2L':
                    fes_ = np.log10(100/np.array(fes_data[problem][agent]))
                else:
                    fes_ = np.log10(maxFEs/np.array(fes_data[problem][agent]))
                fes_problem.append(fes_.mean())
            results_fes[agent] = np.exp((fes_problem - avg) * 1)
        aei_mean, aei_std = self.cal_aei(results_fes, agents, ignore)
        return results_fes, aei_mean, aei_std
    
    def aei_complexity(self, complexity_data: dict, baseline: dict, ignore: Optional[list]=None):
        """
        # Introduction
        Calculates the AEI (Aggregated Evaluation Indicator) complexity for a set of agents based on provided complexity data and a baseline. The function computes a normalized complexity score for each agent, optionally ignoring specified agents, and returns the results along with the mean and standard deviation of the AEI.
        # Args:
        - complexity_data (dict):The result data,the results['fes'].Also a nested dictionary containing experimental results structured as `dict[problem][algorithm][run]`.
        - ignore (Optional[list]): A list of agent keys to ignore during computation. Defaults to None.
        # Returns:
        - results_complex (dict): A dictionary mapping each agent to its computed complexity score.
        - aei_mean (float): The mean AEI value across all considered agents.
        - aei_std (float): The standard deviation of the AEI values.
        # Raises:
        - KeyError: If required keys are missing in `complexity_data` or `baseline`.
        - ValueError: If the input data shapes are inconsistent or invalid.
        """
        
        avg = baseline['complexity_avg']
        std = baseline['complexity_std']
        problems = complexity_data['T1'].keys()
        agents = complexity_data['T1'][list(problems)[0]].keys()
        results_complex = {}
        complexity_data['complexity'] = {}
        for key in agents:
            if (ignore is not None) and (key in ignore):
                continue
            if key not in complexity_data['complexity'].keys():
                t0 = complexity_data['T0']
                t1 = np.array([complexity_data['T1'][pname][key] for pname in problems])
                t2 = np.array([complexity_data['T2'][pname][key] for pname in problems])
                complexity_data['complexity'][key] = np.mean((t2 - t1) / t0)
            results_complex[key] = np.exp((np.log10(1/complexity_data['complexity'][key]) - avg)/std/1000 * 1)
        aei_mean, aei_std = self.cal_aei(results_complex, agents, ignore)
        return results_complex, aei_mean, aei_std

    def cal_aei(self, results: dict, agents: dict, ignore: Optional[list]=None):
        """
        # Introduction
        Calculates the mean and standard deviation of AEI (Aggregated Evaluation Indicator) values for a set of agents, with options to ignore certain agents and apply problem-specific scaling to the standard deviation.
        # Args:
        - results (dict): A dictionary mapping agent names to their corresponding AEI values (iterable or array-like), the value here is generated in aei_fes, aei_cost, or aei_complexity.
        - agents (dict): A dictionary of algorithm agent names.
        - ignore (Optional[list]): A list of agent names to be excluded from the calculation. Defaults to None.
        # Returns:
        - tuple: A tuple containing two dictionaries:
            - mean (dict): The mean AEI value for each agent.
            - std (dict): The standard deviation of AEI values for each agent, scaled according to the test problem.
        # Notes:
        - Agents named 'Random_search' are always ignored.
        - For test problems 'protein' or 'protein-torch', the standard deviation is multiplied by 5; otherwise, it is divided by 5.
        """
        
        mean = {}
        std = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            if agent == 'Random_search':
                continue
            aei_k = results[agent]
            mean[agent] = np.mean(aei_k)
            if self.config.test_problem in ['protein', 'protein-torch']:
                std[agent] = np.std(aei_k) * 5.
            else:
                std[agent] = np.std(aei_k) / 5.
        return mean, std

    def aei_metric(self, data: dict, maxFEs: Optional[Union[int, float]]=20000, ignore: Optional[list]=None):
        """
        # Introduction
        Calculates the AEI (Aggregated Evaluation Indicator) metric for a set of agents across multiple problems, based on cost, function evaluations, and complexity. The AEI metric is computed by combining normalized cost, function evaluation, and complexity metrics for each agent, excluding specified agents if required.
        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - maxFEs (Optional[Union[int, float]], default=20000): The maximum number of function evaluations to consider for normalization.
        - ignore (Optional[list], default=None): A list of agent names to ignore in the AEI calculation.
        # Returns:
        - dict: A dictionary with two keys:
            - 'mean': A dictionary mapping each agent to its mean AEI value.
            - 'std': A dictionary mapping each agent to the standard deviation of its AEI value.
        # Notes:
        - The 'Random_search' agent and any agents listed in `ignore` are excluded from the results.
        - For certain test problems (e.g., 'protein', 'protein-torch'), the standard deviation is scaled differently.
        """
        
        """
        Calculate the AEI metric
        """
        
        baseline = self.get_random_baseline(data, maxFEs)
        problems = data['cost'].keys()
        agents = data['cost'][list(problems)[0]].keys()
        
        results_cost, aei_cost_mean, aei_cost_std = self.aei_cost(data['cost'], baseline, ignore)
        results_fes, aei_fes_mean, aei_fes_std = self.aei_fes(data['fes'], baseline, maxFEs, ignore)
        results_complex, aei_clx_mean, aei_clx_std = self.aei_complexity(data, baseline, ignore)
        
        mean = {}
        std = {}
        for agent in agents:
            if ignore is not None and agent in ignore:
                continue
            if agent == 'Random_search':
                continue
            aei_k = results_complex[agent] * results_cost[agent] * results_fes[agent]
            mean[agent] = np.mean(aei_k)
            if self.config.test_problem in ['protein', 'protein-torch']:
                std[agent] = np.std(aei_k) * 5.
            else:
                std[agent] = np.std(aei_k) / 5.
        return {'mean': mean, 'std': std}

    def cec_metric(self, data: dict, ignore: Optional[list]=None):
        """
        # Introduction
        Calculates the CEC metric for a set of optimization results, aggregating scores for different agents across multiple problems. The metric combines ranking and performance-based scores, optionally ignoring specified agents.
        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - ignore (Optional[list]): A list of agent names to exclude from the metric calculation. Defaults to None.
        # Returns:
        - dict: A dictionary mapping agent labels to their computed CEC metric scores.
        # Notes:
        - The function expects the input `data` to be structured with problems as keys, each containing agent results.
        - Uses helper function `to_label` to standardize agent names and `self.cal_scores1` for part of the score calculation.
        """
        """
        Calculate the metric adopted in CEC
        """
        
        score = {}
        M = []
        X = []
        Y = []
        R = []
        data, fes = data['cost'], data['fes']
        for problem in list(data.keys()):
            maxf = 0
            avg_cost = []
            avg_fes = []
            for agent in list(data[problem].keys()):
                if ignore is not None and agent in ignore:
                    continue
                key = to_label(agent)
                if key not in score.keys():
                    score[key] = []
                values = np.array(data[problem][agent])[:, -1]
                score[key].append(values)
                maxf = max(maxf, np.max(values))
                avg_cost.append(np.mean(values))
                avg_fes.append(np.mean(fes[problem][agent]))

            M.append(maxf)
            order = np.lexsort((avg_fes, avg_cost))
            rank = np.zeros(len(avg_cost))
            rank[order] = np.arange(len(avg_cost)) + 1
            R.append(rank)
        sr = 0.5 * np.sum(R, 0)
        score2 = (1 - (sr - np.min(sr)) / sr) * 50
        score1 = self.cal_scores1(score, M)
        for i, key in enumerate(score.keys()):
            score[key] = score1[i] + score2[i]
        return score

    def draw_ECDF(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True):
        """
        # Introduction
        Plots Empirical Cumulative Distribution Functions (ECDF) for cost data of different agents across problems, and saves the figures to the specified output directory.
        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - output_dir (str): The directory path where the ECDF plots will be saved.
        - Name (Optional[Union[str, list]], optional): The name(s) of the problem(s) to plot. If `None`, plots for all problems. Defaults to `None`.
        - pdf_fig (bool, optional): If `True`, saves figures as PDF; otherwise, saves as PNG. Defaults to `True`.
        # Returns:
        - None
        # Notes:
        - The method uses `self.color_arrangement` to assign colors to agents and updates it if new agents are encountered.
        - The method expects `to_label`, `colors`, `np`, and `plt` to be available in the scope.
        """
        
        data = data['cost']
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            plt.figure()
            for agent in list(data[name].keys()):
                if agent not in self.color_arrangement.keys():
                    self.color_arrangement[agent] = colors[self.arrange_index]
                    self.arrange_index += 1
                values = np.array(data[name][agent])[:, -1]
                plt.ecdf(values, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
            plt.grid()
            plt.xlabel('costs')
            plt.legend()
            fig_type = 'pdf' if pdf_fig else 'png'
            plt.savefig(output_dir + f'ECDF_{problem}.{fig_type}', bbox_inches='tight')

    def draw_covergence_curve(self, agent: str, problem: str, metadata_dir: str, output_dir: str, pdf_fig: bool = True):
        """
        # Introduction
        Plots the convergence curve of population diameter over optimization generations for a given agent and problem, using metadata from previous runs. The curve shows the mean and standard deviation of the population diameter across multiple test runs.
        # Args:
        - agent (str): The name of the agent whose convergence curve is to be plotted.
        - problem (str): The name of the optimization problem.
        - metadata_dir (str): Directory path where the metadata pickle file is stored.
        - output_dir (str): Directory path where the output plot will be saved.
        - pdf_fig (bool, optional): If True, saves the figure as a PDF; otherwise, saves as a PNG. Defaults to True.
        # Returns:
        - None
        # Raises:
        - FileNotFoundError: If the metadata file for the specified problem does not exist.
        - KeyError: If the specified agent is not found in the metadata.
        - Exception: For errors during file reading or plotting.
        """
        
        def cal_max_distance(X):
            X = np.array(X)
            return np.max(np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, -1)))
        with open(metadata_dir + f'/{problem}.pkl', 'rb') as f:
            metadata = pickle.load(f)[agent]
        plt.figure()
        Xs = []
        n_generations = int(1e9)
        for item in metadata:
            Xs.append(item['X'])
            n_generations = min(n_generations, len(item['X']))
        diameter = np.zeros(n_generations)
        std = np.zeros(n_generations)
        x_axis = np.arange(n_generations)
        for i in range(n_generations):  # episode length
            d = []
            for j in range(len(Xs)):  # test_run
                d.append(cal_max_distance(Xs[j][i]))
            diameter[i] = np.mean(d)
            std[i] = np.std(d)
        plt.plot(x_axis, diameter, marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
        plt.fill_between(x_axis, (diameter - std), (diameter + std), alpha=0.2, facecolor=self.color_arrangement[agent])
        plt.grid()
        plt.xlabel('Optimization Generations')    
        plt.ylabel('Population Diameter')
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'convergence_curve_{agent}_{problem}.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_test_data(self, data: dict, data_type: str, output_dir: str, Name: Optional[Union[str, list]]=None, logged: bool=False, categorized: bool=False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        """
        # Introduction
        Plots and saves performance curves for test data of different agents on various problems, supporting both categorized and uncategorized visualizations, with options for logarithmic scaling and output format.
        # Args:
        - results (dict): Part of the result data,the result[data_type]. Also a nested dictionary containing experimental results structured as `dict[problem][algo][run]`.
        - data_type (str): Label for the type of data being plotted (e.g., 'cost', 'accuracy').
        - output_dir (str): Directory path where the generated figures will be saved.
        - Name (Optional[Union[str, list]], optional): Specific problem name(s) to plot. If None, plots all problems. Defaults to None.
        - logged (bool, optional): If True, applies logarithmic scaling to the data before plotting. Defaults to False.
        - categorized (bool, optional): If True, separates plots into categories based on agent types (e.g., learnable vs. classic). Defaults to False.
        - pdf_fig (bool, optional): If True, saves figures as PDF; otherwise, saves as PNG. Defaults to True.
        - data_wrapper (Callable, optional): Optional function to preprocess or transform the data before plotting. Defaults to None.
        # Returns:
        - None
        # Notes:
        - The function uses matplotlib for plotting and saves figures to the specified output directory.
        - Color arrangements for agents are managed to ensure consistent coloring across plots.
        - When `categorized` is True, separate plots are generated for learnable and classic agent types.
        - The function assumes the existence of `self.color_arrangement`, `self.arrange_index`, `self.config`, and external variables `colors` and `to_label`.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            # if logged:
            #     plt.title('log cost curve ' + name)
            # else:
            #     plt.title('cost curve ' + name)
            if not categorized:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = data[name][agent]
                    if data_wrapper is not None:
                        values = data_wrapper(values)
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel(f'log {data_type}')
                    plt.savefig(output_dir + f'{name}_log_{data_type}_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel(data_type)
                    plt.savefig(output_dir + f'{name}_{data_type}_curve.{fig_type}', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    # if agent not in self.config.agent:
                    #     continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = data[name][agent]
                    if data_wrapper is not None:
                        values = data_wrapper(values)
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                if logged:
                    plt.ylabel(f'log {data_type}')
                    plt.savefig(output_dir + f'learnable_{name}_log_{data_type}_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel(data_type)
                    plt.savefig(output_dir + f'learnable_{name}_{data_type}_curve.{fig_type}', bbox_inches='tight')
                plt.close()

                plt.figure()
                for agent in list(data[name].keys()):
                    # if agent not in self.config.t_optimizer:
                    #     continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = data[name][agent]
                    if data_wrapper is not None:
                        values = data_wrapper(values)
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                
                plt.legend()
                if logged:
                    plt.ylabel(f'log {data_type}')
                    plt.savefig(output_dir + f'classic_{name}_log_{data_type}_curve.{fig_type}', bbox_inches='tight')
                else:
                    plt.ylabel(data_type)
                    plt.savefig(output_dir + f'classic_{name}_{data_type}_curve.{fig_type}', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_costs(self, data: dict, output_dir: str, named_agents: dict, logged: bool=False, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Plots and saves the average normalized test costs for multiple named agent groups across different problems. Each subplot corresponds to a group of agents, showing the mean and standard deviation of their normalized costs over function evaluations.
        # Args:
        - results (dict): Part of the result data.Also a nested dictionary containing experimental results structured as `dict[problem][algo][run]`.
        - output_dir (str): Directory path where the resulting plot will be saved.
        - named_agents (dict): Dictionary mapping subplot titles to lists of agent names to be plotted in each subplot.
        - logged (bool, optional): If True, applies logarithmic scaling to the normalized costs. Defaults to False.
        - pdf_fig (bool, optional): If True, saves the figure as a PDF; otherwise, saves as PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The function normalizes costs by the initial value for each run.
        - Each agent's mean and standard deviation curves are plotted with shaded error bands.
        - The function uses `self.color_arrangement` and `self.config.maxFEs` for color assignment and x-axis scaling, respectively.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            
            Y = {}
            for problem in list(data.keys()):
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent])
                    values /= values[:, 0].repeat(values.shape[-1]).reshape(values.shape)
                    if logged:
                        values = np.log(np.maximum(values, 1e-8))
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    Y[agent]['mean'].append(mean)
                    Y[agent]['std'].append(std)

            for id, agent in enumerate(list(Y.keys())):
                mean = np.mean(Y[agent]['mean'], 0)
                std = np.mean(Y[agent]['std'], 0)

                X = np.arange(mean.shape[-1])
                X = np.array(X, dtype=np.float64)
                X *= (self.config.maxFEs / X[-1])
                # X = np.log10(X)
                # X[0] = 0

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            # plt.xlabel('log10 FEs')
            plt.xlabel('FEs')
            plt.ylabel('Normalized Costs')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_cost_curve_logX.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves bar plots representing the normalized performance (final cost divided by initial cost) of different agents on various problems. The plots are saved as either PDF or PNG files in the specified output directory.
        # Args:
        - results (dict): Part of the result data.Also a nested dictionary containing experimental results structured as `dict[problem][algo][run]`.
        - output_dir (str): The directory path where the generated plots will be saved.
        - Name (Optional[Union[str, list]], optional): Specific problem name(s) to include in the plots. If None, all problems are included. Defaults to None.
        - pdf_fig (bool, optional): If True, saves plots as PDF files; otherwise, saves as PNG files. Defaults to True.
        # Returns:
        - None
        # Notes:
        - Each agent gets a separate bar plot.
        - The y-axis represents the mean normalized cost for each problem.
        - Plots are saved with filenames in the format '{agent}_concrete_performance_hist.{pdf|png}'.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        D = {}
        X = []
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D.keys():
                    D[agent] = []
                values = np.array(data[name][agent])
                D[agent].append(values[:, -1] / values[:, 0])

        for agent in D.keys():
            plt.figure()
            # plt.title(f'{agent} performance histgram')
            X = list(data.keys())
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])
            for a,b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)
            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            plt.ylabel('Normalized Costs')
            plt.savefig(output_dir + f'{agent}_concrete_performance_hist.{fig_type}', bbox_inches='tight')

    def draw_boxplot(self, data: dict, output_dir: str, Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction

        Generates and saves boxplot visualizations of cost data for different agents and problems.

        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - output_dir (str): The directory path where the generated boxplot figures will be saved.
        - Name (Optional[Union[str, list]], optional): Specific problem name(s) to plot. If `None`, all problems in the data are plotted. Defaults to `None`.
        - ignore (Optional[list], optional): List of agent names to ignore when plotting. If `None`, no agents are ignored. Defaults to `None`.
        - pdf_fig (bool, optional): If `True`, saves figures as PDF files; otherwise, saves as PNG files. Defaults to `True`.

        # Returns:

        - None

        # Notes:

        - The function creates one boxplot per problem, with each agent represented on the x-axis.
        - The last column of each agent's cost data is used for the boxplot.
        - Boxplots are saved to `output_dir` with filenames in the format `{problem}_boxplot.{pdf|png}`.
        """

        fig_type = 'pdf' if pdf_fig else 'png'
        data = data['cost']
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                values = np.array(data[name][agent])
                Y.append(values[:, -1])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} Cost Boxplots')
            plt.savefig(output_dir + f'{name}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()

    def draw_overall_boxplot(self, data: dict, output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves a boxplot comparing the performance of different agents across multiple problems, using the final cost values from the provided data. The boxplot is normalized per problem and can be saved as either a PDF or PNG file.
        # Args:
        - results (dict): Part of the result data.Also a nested dictionary containing experimental results structured as `dict[problem][algo][run]`.
        - output_dir (str): Directory path where the resulting boxplot image will be saved.
        - ignore (Optional[list], optional): List of agent names to exclude from the plot. Defaults to None.
        - pdf_fig (bool, optional): If True, saves the figure as a PDF; otherwise, saves as a PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The function normalizes the cost values for each problem before plotting.
        - The resulting boxplot displays agents on the x-axis and their normalized costs on the y-axis.
        - The plot includes mean and median markers, and outliers are not shown.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1]
            values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel('Cost Boxplots')
        plt.savefig(output_dir + f'overall_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_rank_hist(self, data: dict, random: dict, output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Plots a bar chart with error bars representing the AEI (Aggregated Evaluation Indicator) metric for different agents, and saves the figure to the specified output directory.
        # Args:
        - results (dict): The result data.Also a nested dictionary containing experimental results structured as `dict[metric][problem][algo][run]`.
        - random (dict): Dictionary containing the random baseline data for comparison.
        - output_dir (str): Path to the directory where the output figure will be saved.
        - ignore (Optional[list], optional): List of agent names to ignore in the plot. Defaults to None.
        - pdf_fig (bool, optional): If True, saves the figure as a PDF; otherwise, saves as a PNG. Defaults to True.
        # Returns:
        - None: This method saves the plot to a file and does not return any value.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        metric, metric_std = self.aei_metric(data, random, maxFEs=self.config.maxFEs, ignore=ignore)
        X, Y = list(metric.keys()), list(metric.values())
        _, S = list(metric_std.keys()), list(metric_std.values())
        n_agents = len(X)
        for i in range(n_agents):
            X[i] = to_label(X[i])

        plt.figure(figsize=(4*n_agents,15))
        plt.bar(X, Y)
        plt.errorbar(X, Y, S, fmt='s', ecolor='dimgray', ms=1, color='dimgray', elinewidth=5, capsize=30, capthick=5)
        for a,b in zip(X, Y):
            plt.text(a, b+0.05, '%.2f' % b, ha='center', fontsize=55)
        plt.xticks(rotation=45, fontsize=60)
        plt.yticks(fontsize=60)
        plt.ylim(0, np.max(np.array(Y) + np.array(S)) * 1.1)
        plt.title(f'The AEI for {self.config.dim}D {self.config.problem}-{self.config.difficulty}', fontsize=70)
        plt.ylabel('AEI', fontsize=60)
        plt.savefig(output_dir + f'rank_hist.{fig_type}', bbox_inches='tight')
        
    def draw_train_logger(self, data_type: str, steps: list, data: dict, agent_for_rollout: str, output_dir: str, ylabel: str = None, norm: bool = False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        """
        # Introduction
        Plots and saves the training curve for a given data type, applying smoothing and displaying mean and standard deviation shading. Supports normalization and custom data processing.
        # Args:
        - data_type (str): The type of data being plotted. e.g. cost
        - steps (list): List of step values (x-axis) corresponding to the data points.
        - results (dict): Part of the result data,the result[data_type]. Also a nested dictionary containing experimental results structured as `dict[problem][algo][run]`.
        - output_dir (str): Directory path where the output figure will be saved.
        - ylabel (str, optional): Label for the y-axis. If None, uses `data_type` as the label. Defaults to None.
        - norm (bool, optional): Whether to normalize the data before plotting. Defaults to False.
        - pdf_fig (bool, optional): Whether to save the figure as a PDF (if True) or PNG (if False). Defaults to True.
        - data_wrapper (Callable, optional): Optional function to preprocess or wrap the data before averaging. Defaults to None.
        # Returns:
        - None
        # Notes:
        - The function applies a smoothing operation to the plotted curve based on the configuration.
        - The mean and standard deviation are visualized, with the standard deviation shown as a shaded region.
        - The color arrangement for each agent is managed to ensure consistent coloring across plots.
        """
        
        means, stds = self.get_average_data(data, norm=norm, data_wrapper=data_wrapper)
        plt.figure()

        y = np.array([means[k] for k in means])
        y_std = np.array([stds[k] for k in stds])
        x = np.array(steps, dtype = np.float64)

        s = np.zeros(y.shape[0])
        a = s[0] = y[0]

        norm = 0.8 + 1
        for i in range(1, y.shape[0]):
            a = a * 0.8 + y[i]
            s[i] = a / norm if norm > 0 else a
            norm *= 0.8
            norm += 1
        if agent_for_rollout not in self.color_arrangement.keys():
            self.color_arrangement[agent_for_rollout] = colors[self.arrange_index]
            self.arrange_index += 1

        plt.plot(x, s, label = to_label(agent_for_rollout), marker = '*', markersize = 12, markevery = 2, c = self.color_arrangement[agent_for_rollout])
        plt.fill_between(x, (s - y_std), (s + y_std), alpha = 0.2, facecolor = self.color_arrangement[agent_for_rollout])

        plt.legend()
        plt.grid()
        plt.xlabel('Learning Steps')
        if ylabel is None:
            ylabel = data_type
        plt.ylabel(ylabel)
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'avg_{data_type}_curve.{fig_type}', bbox_inches='tight')
        plt.close()


        # if agent not in self.color_arrangement.keys():
        #     self.color_arrangement[agent] = colors[self.arrange_index]
        #     self.arrange_index += 1
        # plt.plot(x, s, label = to_label(agent), marker = '*', markersize = 12, markevery = 2, c = self.color_arrangement[agent])
        # plt.fill_between(x, (s - stds[agent]), (s + stds[agent]), alpha = 0.2, facecolor = self.color_arrangement[agent])


        # for agent in means.keys():
        #     x = np.arange(len(means[agent]), dtype=np.float64)
        #     x = (self.config.maxFEs / x[-1]) * x
        #     y = means[agent]
        #     s = np.zeros(y.shape[0])
        #     a = s[0] = y[0]
        #     norm = 0.8 + 1
        #     for i in range(1, y.shape[0]):
        #         a = a * 0.8 + y[i]
        #         s[i] = a / norm if norm > 0 else a
        #         norm *= 0.8
        #         norm += 1
        #     if agent not in self.color_arrangement.keys():
        #         self.color_arrangement[agent] = colors[self.arrange_index]
        #         self.arrange_index += 1
        #     plt.plot(x, s, label=to_label(agent), marker='*', markersize=12, markevery=2, c=self.color_arrangement[agent])
        #     plt.fill_between(x, (s - stds[agent]), (s + stds[agent]), alpha=0.2, facecolor=self.color_arrangement[agent])
        #     # plt.plot(x, returns[agent], label=to_label(agent))
    def post_processing_test_statics(self, log_dir: str, include_random_baseline: bool = True, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Post-processes test statistics by loading results, generating summary tables, and creating visualizations for algorithm performance evaluation.
        # Args:
        - log_dir (str): Directory path where test result files are stored and output files will be saved.
        - include_random_baseline (bool, optional): Whether to include a random baseline in the analysis. Defaults to True.
        - pdf_fig (bool, optional): Whether to save generated figures in PDF format. Defaults to True.
        # Returns:
        - None
        # Side Effects:
        - Reads test results from a pickle file in `log_dir`.
        - Creates directories for tables and figures if they do not exist.
        - Saves generated tables and figures to the corresponding directories.
        - Dumps additional metrics to a pickle file.
        # Raises:
        - FileNotFoundError: If the required test results file does not exist in `log_dir`.
        - Any exceptions raised by file I/O or pickle operations.
        """
        
        print('Post processing & drawing')
        with open(log_dir + 'test_results.pkl', 'rb') as f:
            results = pickle.load(f)

        metabbo = results['config'].baselines['metabbo']
        bbo = results['config'].baselines['bbo']
        
        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        self.gen_overall_tab(results, log_dir + 'tables/')
        self.gen_algorithm_complexity_table(results, log_dir + 'tables/')

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        # 如果需要，可以为不同的算法绘制图形（例如 cost 图）
        if 'cost' in results:
            self.draw_test_data(results['cost'], 'cost', log_dir + 'pics/', logged=True, categorized=True, pdf_fig=pdf_fig, data_wrapper=np.array)
            self.draw_named_average_test_costs(results['cost'], log_dir + 'pics/',
                                                {'MetaBBO-RL': metabbo,
                                                'Classic Optimizer': bbo},
                                                logged=False, pdf_fig=pdf_fig)
            self.draw_ECDF(results, log_dir + 'pics/', pdf_fig=pdf_fig)
            self.draw_boxplot(results, log_dir + 'pics/', pdf_fig=pdf_fig)
            with open(log_dir + 'aei.pkl', 'wb') as f:
                pickle.dump(self.aei_metric(results, self.config.maxFEs), f)

    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Processes rollout statistics after training, generates plots for return and cost, and saves them to the specified directory.
        # Args:
        - log_dir (str): The directory path where the rollout statistics file ('rollout.pkl') is located and where the output plots will be saved.
        - pdf_fig (bool, optional): Whether to save the generated plots as PDF files. Defaults to True.
        # Returns:
        - None
        # Raises:
        - FileNotFoundError: If the 'rollout.pkl' file does not exist in the specified directory.
        - KeyError: If expected keys ('steps', 'return', 'cost') are missing in the loaded results.
        - Exception: Propagates any exceptions raised during file I/O or plotting.
        """
        
        print('Post processing & drawing')
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        agent_for_rollout = results['agent_for_rollout']
        self.draw_train_logger('return', results['steps'], results['return'], agent_for_rollout, log_dir + 'pics/', pdf_fig=pdf_fig)
        self.draw_train_logger('cost', results['steps'], results['cost'], agent_for_rollout, log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper = Basic_Logger.data_wrapper_cost_rollout)

    
class MOO_Logger(Basic_Logger):
    """
    # Introduction
    Custormized logger for Moo scenary.
    # Attributes
    - config (argparse.Namespace): Configuration namespace with parameters like maxFEs, indicators, agents.
    - color_arrangement (dict): Mapping of agents to colors for consistent plotting.
    - arrange_index (int): Index to track color assignment order.
    - indicators (list): List of performance indicators to log and plot.

    # Methods
    - __init__(config): Initializes logger with configuration.
    - is_pareto_efficient(points): Computes Pareto-efficient points from a set.
    - draw_pareto_fronts(data, output_dir, Name): Plots Pareto fronts for given problems.
    - draw_test_indicator(data, output_dir, indicator, Name, categorized, pdf_fig): Plots test indicator curves.
    - draw_named_average_test_indicator(data, output_dir, named_agents, indicator, pdf_fig): Plots average indicator curves for named agent groups.
    - draw_concrete_performance_hist(data, output_dir, indicator, Name, pdf_fig): Draws bar charts for final performance values.
    - draw_boxplot(data, output_dir, indicator, Name, ignore, pdf_fig): Generates boxplots for agent performances on each problem.
    - draw_overall_boxplot(data, output_dir, indicator, ignore, pdf_fig): Generates combined boxplot across all problems.
    - draw_train_logger(data_type, steps, data, agent_for_rollout, output_dir, ylabel, norm, pdf_fig, data_wrapper): Plots training metric curves with smoothing.
    - post_processing_test_statics(log_dir, include_random_baseline, pdf_fig): Processes test results and generates summary tables and figures.
    - post_processing_rollout_statics(log_dir, pdf_fig): Processes rollout statistics and generates plots.
    """
    def __init__(self, config: argparse.Namespace) -> None:
        """
        # Introduction
        Initializes the logger with the given configuration.
        # Args:
        - config (argparse.Namespace): Configuration namespace containing parameters like maxFEs, indicators, and agents.
        # Returns:
        - None
        """
        super().__init__(config)
        self.config = config
        self.color_arrangement = {}
        self.arrange_index = 0
        if hasattr(config, 'indicators') and config.indicators is not None:
            self.indicators = config.indicators
        else:
            self.indicators = ['hv_his', 'igd_his']
    
    def is_pareto_efficient(self,points):
        """
        # Introduction
        Determines the Pareto-efficient points from a given set of points. A point is considered Pareto-efficient if no other point is strictly better in all dimensions.
        # Args:
        - points (array-like): A 2D array or list of shape (n_points, n_dimensions) representing the set of points to evaluate.
        # Returns:
        - numpy.ndarray: An array containing the Pareto-efficient points.
        # Raises:
        - None
        """
        
        points = np.array(points)
        pareto_mask = np.ones(points.shape[0], dtype=bool)
        for i, p in enumerate(points):
            if pareto_mask[i]:
                pareto_mask[pareto_mask] = np.any(points[pareto_mask] < p, axis=1)
                pareto_mask[i] = True
        return points[pareto_mask]
    
    def draw_pareto_fronts(self,data: dict, output_dir: str, Name: Optional[Union[str, list]] = None):
        """
        # Introduction
        Plots and saves the Pareto fronts for multiple algorithms on one or more optimization problems, supporting both 2D and 3D objective spaces. The function visualizes the final generation's Pareto-efficient solutions for each algorithm and problem, and saves the resulting plots as PNG files.
        # Args:
        - data (dict): Nested dictionary containing optimization results structured as `dict[problem][algorithm][run]`.
        - output_dir (str): Directory path where the generated Pareto front plots will be saved.
        - Name (Optional[Union[str, list]]): Specific problem name or list of problem names to plot. If `None`, all problems in `data` are plotted.
        # Returns:
        - None
        # Notes:
        - The function expects each algorithm's runs to be a list of generations, where each generation contains objective values.
        - The function uses `self.is_pareto_efficient` to extract Pareto-efficient solutions.
        - Plots are saved as PNG files named `{problem}_pareto_fronts.png` in the specified output directory.
        """
        
        for problem in list(data.keys()):
            if Name is not None and ((isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name)):
                continue
            else:
                name = problem

            fig = plt.figure(figsize=(8, 6))  
            is_3d = False
            algo_obj_dict = {}

            for algo, runs in data[problem].items():
                all_obj_values = []
                for generations in runs:
                    last_gen = np.array(generations[-1])
                    obj_values = last_gen.reshape(-1, last_gen.shape[-1])
                    if obj_values.shape[1] == 3:
                        is_3d = True
                    all_obj_values.append(obj_values)
                algo_obj_dict[algo] = np.vstack(all_obj_values)


            if is_3d:
                ax = fig.add_subplot(111, projection='3d')
                ax.view_init(elev=40, azim=135)  
                ax.set_proj_type('persp')
            else:
                ax = fig.add_subplot(111)

            colors = ['r', 'g', 'b', 'c', 'm', 'y']

            for algo_idx, (algo, obj_values) in enumerate(algo_obj_dict.items()):
                pareto_front = self.is_pareto_efficient(obj_values)
                color = colors[algo_idx % len(colors)]
                label = f"{algo}"

                if obj_values.shape[1] == 2:
                    ax.scatter(pareto_front[:, 0], pareto_front[:, 1],
                                label=label, color=color, edgecolors='k')
                elif obj_values.shape[1] == 3:
                    ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                                label=label, color=color, edgecolors='k')

            if is_3d:

                ax.set_xlabel('X', fontsize=12, labelpad=10)
                ax.set_ylabel('Y', fontsize=12, labelpad=10)
                ax.set_zlabel('Z', fontsize=12, labelpad=-0.5,color='black')  
                

                ax.zaxis.set_label_coords(1.05, 0.5)  


                ax.set_box_aspect([1.2, 1.1, 0.9])

            else:
                ax.set_xlabel('X', fontsize=14, labelpad=20)
                ax.set_ylabel('Y', fontsize=14, labelpad=20)

            plt.subplots_adjust(right=0.85)

            plt.legend()
            plt.grid(True)
            plt.title(f'Pareto Fronts of Algorithms on {problem}', fontsize=14)

            plt.savefig(output_dir + f'{name}_pareto_fronts.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.show()
    
    def draw_test_indicator(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, categorized: bool=False, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Plots and saves performance indicator curves for different agents based on the provided experimental data. Supports both categorized and non-categorized plotting, and can output figures in PDF or PNG format.
        # Args:
        - data (dict): Part of the result dictionary,mapping the indicator. Also a nested dictionary containing experimental results structured as `dict[problem][algorithm][run][generation][objective]`.
        - output_dir (str): Directory path where the generated plots will be saved.
        - indicator (str): Name of the performance indicator to be plotted (e.g., accuracy, loss).
        - Name (Optional[Union[str, list]], optional): Specific problem name(s) to plot. If None, plots for all problems in `data`. Defaults to None.
        - categorized (bool, optional): If True, separates plots into learnable and classic agent categories. Defaults to False.
        - pdf_fig (bool, optional): If True, saves plots as PDF files; otherwise, saves as PNG files. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The method uses `self.color_arrangement` and `self.arrange_index` to assign colors to agents.
        - Requires `self.config.maxFEs`, `self.config.agent`, and `self.config.t_optimizer` to be defined.
        - Uses matplotlib for plotting and numpy for numerical operations.
        - The function saves plots to disk and does not display them interactively.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            if not categorized:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])

                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'{name}_{indicator}_curve.png', bbox_inches='tight')
                plt.close()
            else:
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.agent:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'learnable_{name}_{indicator}_curve.png', bbox_inches='tight')
                plt.close()
                plt.figure()
                for agent in list(data[name].keys()):
                    if agent not in self.config.t_optimizer:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    values = np.array(data[name][agent])
                    x = np.arange(values.shape[-1])
                    x = np.array(x, dtype=np.float64)
                    x *= (self.config.maxFEs / x[-1])
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    plt.plot(x, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                    plt.fill_between(x, mean - std, mean + std, alpha=0.2, facecolor=self.color_arrangement[agent])
                plt.grid()
                plt.xlabel('FEs')
                plt.legend()
                plt.ylabel(str(indicator))
                plt.savefig(output_dir + f'classic_{name}_{indicator}_curve.{fig_type}', bbox_inches='tight')
                plt.close()
    
    def draw_named_average_test_indicator(self, data: dict, output_dir: str, named_agents: dict, indicator:str,pdf_fig: bool = True) -> None:
        """
        # Introduction
        Plots the normalized average and standard deviation curves for a specified indicator across multiple agents and problems, grouping agents by provided names, and saves the resulting figure.
        # Args:
        - data (dict): Part of the result dictionary,mapping the indicator. Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - output_dir (str): Directory path where the output figure will be saved.
        - named_agents (dict): Dictionary mapping group names (titles) to lists of agent names to be plotted together.
        - indicator (str): The key for the indicator to be plotted (e.g., 'reward', 'cost').
        - pdf_fig (bool, optional): If True, saves the figure as a PDF; otherwise, saves as a PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - Normalizes indicator values for each problem across all agents to [0, 1] before plotting.
        - Plots mean and standard deviation curves for each agent, grouped by `named_agents`.
        - Uses internal color arrangement for agent curves.
        - Saves the figure as 'all_problem_{indicator}_curve.{pdf|png}' in the specified output directory.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        fig = plt.figure(figsize=(50, 10))
        # plt.title('all problem cost curve')
        plots = len(named_agents.keys())
        for id, title in enumerate(named_agents.keys()):
            ax = plt.subplot(1, plots+1, id+1)
            ax.set_title(title, fontsize=25)
            Y = {}
            for problem in list(data.keys()):
               
                all_values = []
                for agent in data[problem].keys():
                    all_values.append(np.array(data[problem][agent]))
                all_values = np.concatenate(all_values, axis=0)  
                global_min = np.min(all_values)  
                global_max = np.max(all_values)  
                
                for agent in list(data[problem].keys()):
                    if agent not in named_agents[title]:
                        continue
                    if agent not in self.color_arrangement.keys():
                        self.color_arrangement[agent] = colors[self.arrange_index]
                        self.arrange_index += 1
                    if agent not in Y.keys():
                        Y[agent] = {'mean': [], 'std': []}
                    values = np.array(data[problem][agent])
                    values = (values - global_min) / (global_max - global_min + 1e-8)  
                
                    std = np.std(values, 0)
                    mean = np.mean(values, 0)
                    Y[agent]['mean'].append(mean)
                    Y[agent]['std'].append(std)

            for id, agent in enumerate(list(Y.keys())):
                mean = np.mean(Y[agent]['mean'], 0)
                std = np.mean(Y[agent]['std'], 0)

                X = np.arange(mean.shape[-1])
                X = np.array(X, dtype=np.float64)
                X *= (self.config.maxFEs / X[-1])

                ax.plot(X, mean, label=to_label(agent), marker='*', markevery=8, markersize=13, c=self.color_arrangement[agent])
                ax.fill_between(X, (mean - std), (mean + std), alpha=0.2, facecolor=self.color_arrangement[agent])
            plt.grid()
            plt.xlabel('FEs')
            plt.ylabel('Normalized {indicator}')
            plt.legend()
        # lines, labels = fig.axes[-1].get_legend_handles_labels()
        # fig.legend(lines, labels, bbox_to_anchor=(plots/(plots+1)-0.02, 0.5), borderaxespad=0., loc=6, facecolor='whitesmoke')
        
        plt.subplots_adjust(left=0.05, right=0.95, wspace=0.1)
        plt.savefig(output_dir + f'all_problem_{indicator}_curve.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_concrete_performance_hist(self, data: dict, output_dir: str, indicator: Optional[str] = None, Name: Optional[Union[str, list]] = None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves bar charts visualizing the concrete performance of different agents on various problems, based on the provided data. Each chart represents the mean performance of an agent across selected problems, with the option to filter by specific problem names and customize the output format.
        # Args:
        - data (dict) Part of the result dictionary,the results[indicator]. Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - output_dir (str): The directory path where the generated figures will be saved.
        - indicator (Optional[str], default=None): The label for the y-axis, typically representing the performance metric being visualized.
        - Name (Optional[Union[str, list]], default=None): Specific problem name(s) to include in the visualization. If None, all problems are included.
        - pdf_fig (bool, default=True): If True, saves the figures as PDF files; otherwise, saves them as PNG files.
        # Returns:
        - None
        # Raises:
        - KeyError: If the specified problem or agent names are not found in the data dictionary.
        - IndexError: If the data arrays do not have the expected shape.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        D = {}
        X = []
        
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D:
                    D[agent] = []
                values = np.array(data[name][agent])
                D[agent].append(values[:, -1])

        for agent in D.keys():
            plt.figure()
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])

            for a, b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)

            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            
            ylabel = indicator
            plt.ylabel(ylabel)

            plt.savefig(output_dir + f'{agent}_concrete_{indicator}_performance_hist.{fig_type}', bbox_inches='tight')
    
    def draw_boxplot(self, data: dict, output_dir: str, indicator:str,Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves boxplot visualizations for the provided data, comparing the performance of different agents on specified problems.
        # Args:
        - data (dict):Part of the test result,that is, result[indicator].Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - output_dir (str): The directory path where the generated boxplot figures will be saved.
        - indicator (str): The name of the indicator or metric to be displayed in the boxplot's ylabel and filename.
        - Name (Optional[Union[str, list]], optional): Specific problem name(s) to plot. If None, plots all problems in `data`. Defaults to None.
        - ignore (Optional[list], optional): List of agent names to exclude from the plots. Defaults to None.
        - pdf_fig (bool, optional): If True, saves figures as PDF; otherwise, saves as PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - Each boxplot compares the final column of results (assumed to be the last metric) for each agent on a given problem.
        - The function saves each plot to the specified output directory with a filename pattern: `{problem}_{indicator}_boxplot.{pdf/png}`.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                values = np.array(data[name][agent])
                Y.append(values[:, -1])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} {indicator} Boxplots')
            plt.savefig(output_dir + f'{name}_{indicator}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()
    
    def draw_overall_boxplot(self, data: dict, output_dir: str, indicator:str,ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves a normalized boxplot comparing the performance of different agents across multiple problems for a specified indicator.
        # Args:
        - data (dict): Part of the test result,that is, result[indicator].Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - output_dir (str): Directory path where the resulting boxplot image will be saved.
        - indicator (str): Name of the performance indicator to display on the plot's y-axis and in the filename.
        - ignore (Optional[list], optional): List of agent names to exclude from the plot. Defaults to None.
        - pdf_fig (bool, optional): If True, saves the plot as a PDF; otherwise, saves as a PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The function normalizes the results for each problem before plotting.
        - The resulting boxplot visualizes the distribution of the final indicator values for each agent across all problems.
        - The plot is saved to the specified output directory with a filename indicating the indicator and file type.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1]
            values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel(f'{indicator} Boxplots')
        plt.savefig(output_dir + f'overall_{indicator}_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def draw_train_logger(self, data_type: str, steps: list, data: dict, agent_for_rollout: str, output_dir: str, ylabel: str = None, norm: bool = False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        """
        # Introduction
        Plots and saves the training curve for a given data type, applying smoothing and displaying mean and standard deviation shading. Supports normalization and custom data processing.
        # Args:
        - data_type (str): The type of data being plotted. e.g. cost
        - steps (list): List of step values (x-axis) corresponding to the data points.
        - results (dict): Part of the result data,the result[data_type]. Also a nested dictionary containing experimental results structured as `dict[problem][algo][run]`.
        - output_dir (str): Directory path where the output figure will be saved.
        - ylabel (str, optional): Label for the y-axis. If None, uses `data_type` as the label. Defaults to None.
        - norm (bool, optional): Whether to normalize the data before plotting. Defaults to False.
        - pdf_fig (bool, optional): Whether to save the figure as a PDF (if True) or PNG (if False). Defaults to True.
        - data_wrapper (Callable, optional): Optional function to preprocess or wrap the data before averaging. Defaults to None.
        # Returns:
        - None
        # Notes:
        - The function applies a smoothing operation to the plotted curve based on the configuration.
        - The mean and standard deviation are visualized, with the standard deviation shown as a shaded region.
        - The color arrangement for each agent is managed to ensure consistent coloring across plots.
        """
        
        means, stds = self.get_average_data(data, norm=norm, data_wrapper=data_wrapper)
        plt.figure()

        y = np.array([means[k] for k in means])
        y_std = np.array([stds[k] for k in stds])
        x = np.array(steps, dtype = np.float64)

        s = np.zeros(y.shape[0])
        a = s[0] = y[0]

        norm = 0.8 + 1
        for i in range(1, y.shape[0]):
            a = a * 0.8 + y[i]
            s[i] = a / norm if norm > 0 else a
            norm *= 0.8
            norm += 1
        if agent_for_rollout not in self.color_arrangement.keys():
            self.color_arrangement[agent_for_rollout] = colors[self.arrange_index]
            self.arrange_index += 1

        plt.plot(x, s, label = to_label(agent_for_rollout), marker = '*', markersize = 12, markevery = 2, c = self.color_arrangement[agent_for_rollout])
        plt.fill_between(x, (s - y_std), (s + y_std), alpha = 0.2, facecolor = self.color_arrangement[agent_for_rollout])

        plt.legend()
        plt.grid()
        plt.xlabel('Learning Steps')
        if ylabel is None:
            ylabel = data_type
        plt.ylabel(ylabel)
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'avg_{data_type}_curve.{fig_type}', bbox_inches='tight')
        plt.close()
    
    def post_processing_test_statics(self, log_dir: str, include_random_baseline: bool = False, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Processes and visualizes test statistics from experiment logs, optionally including a random search baseline. Generates tables and figures summarizing algorithm performance.
        # Args:
        - log_dir (str): Directory path where the test results and output files are stored.
        - include_random_baseline (bool, optional): Whether to include results from a random search baseline. Defaults to False.
        - pdf_fig (bool, optional): Whether to save generated figures in PDF format. Defaults to True.
        # Returns:
        - None
        # Raises:
        - FileNotFoundError: If required result files (e.g., 'test.pkl') are not found in the specified directory.
        - Exception: For errors encountered during file reading, directory creation, or result processing.
        """
        
        with open(log_dir + 'test_results.pkl', 'rb') as f:
            results = pickle.load(f)

        metabbo = results['config'].baselines['metabbo']
        bbo = results['config'].baselines['bbo']
        
        if include_random_baseline:
            with open(log_dir + 'random_search_baseline.pkl', 'rb') as f:
                random = pickle.load(f)

        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        self.gen_algorithm_complexity_table(results, log_dir + 'tables/')

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        for indicator in self.indicators:
            self.draw_test_indicator(results[indicator], log_dir + 'pics/', indicator, pdf_fig=pdf_fig)
            self.draw_named_average_test_indicator(results[indicator], log_dir + 'pics/', \
                {'MetaBBO-RL': metabbo, 'Classic Optimizer': bbo}, indicator, pdf_fig=pdf_fig)
    
    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Processes and visualizes the rollout statistics after training or evaluation.Loads the `rollout.pkl` log file and generates plots for return and specified indicators.

        # Args
        - log_dir (str): Directory path where the `rollout.pkl` file is stored.
        - pdf_fig (bool): Whether to save plots as PDF (True) or PNG (False). Default is True.

        # Returns
        - None

        # Notes
        - Saves all generated plots into a `pics/` subdirectory inside the given `log_dir`.
        - Uses `Basic_Logger.data_wrapper_cost_rollout` to wrap non-return indicators.
        """
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        agent_for_rollout = results['agent_for_rollout']
        
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        self.draw_train_logger('return', results['steps'], results['return'], agent_for_rollout, log_dir + 'pics/', pdf_fig=pdf_fig)
        for indicator in self.indicators:
            self.draw_train_logger(indicator, results['steps'], results[indicator], agent_for_rollout, log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper = Basic_Logger.data_wrapper_cost_rollout)



class MMO_Logger(Basic_Logger):
    """
    #Introduction:
    The customized logger for multi-modal optimization(MMO) scenario.
    """
    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__(config)

    def data_wrapper_prsr_rollout(self, data, ):
        """
        #Introduction:
        Wrapper function to extract pr/sr data for logging rollout results.
        """
        res = np.array(data)
        return res[:, -1, 3]

    def data_wrapper_prsr_hist(self,data,):
        """
        #Introduction:
        Wrapper function to extract pr/sr historical information.
        """
        return np.array(data)[:, :, 3]

    def data_wrapper_cost_rollout(self,data, ):
        """
        #Introduction:
        Wrapper function to extract cost data for logging rollout results.
        """
        res = np.array(data)
        return res[:, -1]

    def gen_agent_performance_prsr_table(self, results: dict, data_type: str, out_dir: str) -> None:
        """
        # Introduction
        Generates and saves Excel tables summarizing the performance statistics (`Worst`, `Best`, `Median`, `Mean`, `Std`) of different agents on various problems, based on the provided results.
        # Args:
        - results (dict): Part of the result data,the result[data_type]. Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - data_type (str): A string indicating the type of data being processed (used in the output filename).
        - out_dir (str): The directory path where the resulting Excel files will be saved.
        # Returns:
        - None
        # Notes:
        - For each agent, an Excel file is generated with a table of performance statistics for each problem.
        - The statistics are computed from the last pr/sr value of each run.
        """
        """
        Store the `Worst`, `Best`, `Median`, `Mean` and `Std` of pr/sr results of each agent as excel
        """
        
        total_data=results
        table_data={}
        indexs=[]
        columns=['Worst','Best','Median','Mean','Std']
        for problem,value in total_data.items():
            indexs.append(problem)
            problem_data=value
            for alg,alg_data in problem_data.items():
                n_data=np.array(alg_data)[:, -1, 3]
                # if alg == 'MadDE' and problem == 'F5':
                #     for run in alg_data:
                #         print(len(run))
                #     print(len(n_data))
                best=np.min(n_data)
                best=np.format_float_scientific(best,precision=3,exp_digits=3)
                worst=np.max(n_data)
                worst=np.format_float_scientific(worst,precision=3,exp_digits=3)
                median=np.median(n_data)
                median=np.format_float_scientific(median,precision=3,exp_digits=3)
                mean=np.mean(n_data)
                mean=np.format_float_scientific(mean,precision=3,exp_digits=3)
                std=np.std(n_data)
                std=np.format_float_scientific(std,precision=3,exp_digits=3)

                if not alg in table_data:
                    table_data[alg]=[]
                table_data[alg].append([worst,best,median,mean,std])
        for alg,data in table_data.items():
            dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
            #print(dataframe)
            dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_{data_type}_table.xlsx'))

    def gen_overall_tab(self, results: dict, out_dir: str) -> None:
        """
        # Introduction
        Generates and saves an Excel table summarizing the overall results of optimization experiments, including objective values (costs), precision (pr), and success rate (sr) for each optimizer and problem.
        # Args:
        - results (dict): The result data. Also a nested dictionary,structured as `dict[metric][problem][algo][run]`,stores the test result data.
        - out_dir (str): The output directory path where the resulting Excel file ('overall_table.xlsx') will be saved.
        # Returns:
        - None: This method saves the results to an Excel file and does not return a value.
        # Raises:
        - KeyError: If the expected keys ('cost', 'pr', 'sr') or structure are missing in the `results` dictionary.
        - AttributeError: If `self.config.test_run` is not defined in the class instance.
        - ValueError: If the data shapes in `results` do not match the expected format for processing.
        """
        """
        Store the overall results inculding `objective values` (costs), `pr` and `sr` as excel
        """
        # get multi-indexes first
        problems = []
        statics = ['Obj','Pr', 'Sr']
        optimizers = []
        for problem in results['cost'].keys():
            problems.append(problem)
        for optimizer in results['cost'][problems[0]].keys():
            optimizers.append(optimizer)
        multi_columns = pd.MultiIndex.from_product(
            [problems,statics], names=('Problem', 'metric')
        )
        df_results = pd.DataFrame(np.ones(shape=(len(optimizers),len(problems)*len(statics))),
                                index=optimizers,
                                columns=multi_columns)

        # # calculate baseline1 cmaes
        # cmaes_obj = {}
        # for problem in problems:
        #     blobj_problem = results['cost'][problem]['CMAES']  # 51 * record_length
        #     objs = []
        #     for run in range(self.config.test_run):
        #         objs.append(blobj_problem[run][-1])
        #     cmaes_obj[problem] = sum(objs) / self.config.test_run

        # # calculate baseline2 random_search
        # rs_obj = {}
        # for problem in problems:
        #     blobj_problem = results['cost'][problem]['Random_search']  # 51 * record_length
        #     objs = []
        #     for run in range(self.config.test_run):
        #         objs.append(blobj_problem[run][-1])
        #     rs_obj[problem] = sum(objs) / self.config.test_run

        # calculate each Obj
        for problem in problems:
            for optimizer in optimizers:
                obj_problem_optimizer = results['cost'][problem][optimizer]
                objs_ = np.array(obj_problem_optimizer)[:, -1]
                avg_obj = np.mean(objs_)
                std_obj = np.std(objs_)
                df_results.loc[optimizer, (problem, 'Obj')] = np.format_float_scientific(avg_obj, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_obj, precision=3, exp_digits=1) + ")"

                pr_problem_optimizer = results['pr'][problem][optimizer]
                prs_ = np.array(pr_problem_optimizer)[:, -1, 3]
                avg_pr = np.mean(prs_)
                std_pr = np.std(prs_)
                df_results.loc[optimizer, (problem, 'Pr')] = np.format_float_scientific(avg_pr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_pr, precision=3, exp_digits=1) + ")"

                sr_problem_optimizer = results['sr'][problem][optimizer]
                srs_ = np.array(sr_problem_optimizer)[:, -1, 3]
                avg_sr = np.mean(srs_)
                std_sr = np.std(srs_)
                df_results.loc[optimizer, (problem, 'Sr')] = np.format_float_scientific(avg_sr, precision=3, exp_digits=1) + "(" + np.format_float_scientific(std_sr, precision=3, exp_digits=1) + ")"

        df_results.to_excel(out_dir+'overall_table.xlsx')

    def draw_concrete_performance_prsr_hist(self, data: dict, data_type: str,output_dir: str, Name: Optional[Union[str, list]]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves bar plots representing the normalized performance of different agents on various problems, based on the provided data. The function supports filtering by specific problem names and allows saving the figures in either PDF or PNG format.
        # Args:
        - data (dict):Part of the nested dictionary,the results[data_type].Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - data_type (str): A string indicating the type of data being visualized (used for labeling the y-axis).
        - output_dir (str): Directory path where the generated figures will be saved.
        - Name (Optional[Union[str, list]], optional): Specific problem name or list of problem names to include in the plot. If None, all problems are included. Defaults to None.
        - pdf_fig (bool, optional): If True, saves figures as PDF; otherwise, saves as PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The function computes the mean of the last column of the performance data for each agent and problem.
        - Each agent's performance is plotted as a separate bar chart, with values annotated on the bars.
        - The generated figures are saved to the specified output directory with filenames indicating the agent and data type.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        D = {}
        X = []
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            X.append(name)
            for agent in list(data[name].keys()):
                if agent not in D.keys():
                    D[agent] = []
                values = np.array(data[name][agent])[:, :, 3]
                D[agent].append(values[:, -1])

        for agent in D.keys():
            plt.figure()
            # plt.title(f'{agent} performance histgram')
            X = list(data.keys())
            D[agent] = np.mean(np.array(D[agent]), -1)
            plt.bar(X, D[agent])
            for a,b in zip(X, D[agent]):
                plt.text(a, b, '%.2f' % b, ha='center', fontsize=15)
            plt.xticks(rotation=30, fontsize=13)
            plt.xlabel('Problems')
            plt.ylabel(f'Normalized {data_type}')
            plt.savefig(output_dir + f'{agent}_concrete_performance_{data_type}_hist.{fig_type}', bbox_inches='tight')

    def draw_boxplot_prsr(self, data: dict, data_type: str,output_dir: str, Name: Optional[Union[str, list]]=None, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves boxplot visualizations for parsed result data of multiple agents on different problems. The function supports filtering by problem name, ignoring specific agents, and saving figures in PDF or PNG format.
        # Args:
        - data (dict): Part of the result data, the results[data_type].A nested dictionary where the first-level keys are problem names, and the second-level keys are agent names. The values are numpy arrays containing result data.
        - data_type (str): A string indicating the type of data being visualized (used in plot labels and filenames).
        - output_dir (str): The directory path where the generated boxplot figures will be saved.
        - Name (Optional[Union[str, list]], optional): A specific problem name or a list of problem names to plot. If None, all problems in `data` are plotted. Defaults to None.
        - ignore (Optional[list], optional): A list of agent names to exclude from the plots. If None, no agents are ignored. Defaults to None.
        - pdf_fig (bool, optional): If True, saves figures as PDF files; otherwise, saves as PNG files. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The function expects each value in `data[name][agent]` to be a numpy array with at least four columns, as it accesses `[:, -1, 3]`.
        - Boxplots are saved with filenames formatted as `{problem_name}_{data_type}_boxplot.{pdf|png}` in the specified `output_dir`.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        for problem in list(data.keys()):
            if Name is not None and (isinstance(Name, str) and problem != Name) or (isinstance(Name, list) and problem not in Name):
                continue
            else:
                name = problem
            Y = []
            X = []
            plt.figure(figsize=(30, 15))
            for agent in list(data[name].keys()):
                if ignore is not None and agent in ignore:
                    continue
                X.append(agent)
                # values = np.array(data[name][agent])
                # Y.append(values[:, -1])
                Y.append(np.array(data[name][agent])[:,-1,3])
            Y = np.transpose(Y)
            plt.boxplot(Y, labels=X, showmeans=True, patch_artist=True, showfliers=False,
                        medianprops={'color': 'green', 'linewidth': 3}, 
                        meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                        boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                        capprops={'linewidth': 2},
                        whiskerprops={'linewidth': 2},
                        )
            plt.xticks(rotation=30, fontsize=18)
            plt.xlabel('Agents')
            plt.ylabel(f'{name} {data_type} Boxplots')
            plt.savefig(output_dir + f'{name}_{data_type}_boxplot.{fig_type}', bbox_inches='tight')
            plt.close()

    def draw_overall_boxplot_prsr(self, data: dict, data_type: str,output_dir: str, ignore: Optional[list]=None, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves a boxplot comparing the performance of different agents across multiple problems using the provided data. The boxplot visualizes the distribution of the last metric (index 3) for each agent and problem, normalized per problem.
        # Args:
        - data (dict): Part of the result data, the results[data_type].Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - data_type (str): A string indicating the type of data being visualized (used in plot labels and filenames).
        - output_dir (str): Directory path where the resulting boxplot figure will be saved.
        - ignore (Optional[list]): List of agent names to exclude from the plot. Defaults to None.
        - pdf_fig (bool): If True, saves the figure as a PDF; otherwise, saves as a PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The function normalizes the data for each problem before plotting.
        - The resulting boxplot shows the distribution for each agent, aggregated over all problems and runs.
        - The plot is saved as 'overall_{data_type}_boxplot.{pdf|png}' in the specified output directory.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        problems=[]
        agents=[]
        for problem in data.keys():
            problems.append(problem)
        for agent in data[problems[0]].keys():
            if ignore is not None and agent in ignore:
                continue
            agents.append(agent)
        run = len(data[problems[0]][agents[0]])
        values = np.zeros((len(agents), len(problems), run))
        plt.figure(figsize=(30, 15))
        for ip, problem in enumerate(problems):
            for ia, agent in enumerate(agents):
                values[ia][ip] = np.array(data[problem][agent])[:, -1, 3]
            # values[:, ip, :] = (values[:, ip, :] - np.min(values[:, ip, :])) / (np.max(values[:, ip, :]) - np.min(values[:, ip, :]))
        values = values.reshape(len(agents), -1).transpose()
        
        plt.boxplot(values, labels=agents, showmeans=True, patch_artist=True, showfliers=False,
                    medianprops={'color': 'green', 'linewidth': 3}, 
                    meanprops={'markeredgecolor': 'red', 'markerfacecolor': 'red', 'markersize': 10, 'marker': 'D'}, 
                    boxprops={'color': 'black', 'facecolor': 'lightskyblue'},
                    capprops={'linewidth': 2},
                    whiskerprops={'linewidth': 2},
                    )
        plt.xticks(rotation=30, fontsize=18)
        plt.xlabel('Agents')
        plt.ylabel(f'{data_type} Boxplots')
        plt.savefig(output_dir + f'overall_{data_type}_boxplot.{fig_type}', bbox_inches='tight')
        plt.close()

    def get_average_prsr_rank(self, results: dict):
        """
        # Introduction
        Computes the average and standard deviation of the PRSR rank for each agent across multiple problems.
        # Args:
        - data (dict): Part of the result data, the results[data_type].Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        # Returns:
        - tuple: A tuple containing two dictionaries:
            - avg_data (dict): Maps each agent to the mean PRSR rank averaged across all problems.
            - std_data (dict): Maps each agent to the mean standard deviation of PRSR rank across all problems.
        # Notes:
        - Assumes that all problems have the same set of agents and that the data structure for each agent is consistent across problems.
        """
        
        problems=[]
        agents=[]
        for problem in results.keys():
            problems.append(problem)
        for agent in results[problems[0]].keys():
            agents.append(agent)
        avg_data={}
        std_data={}
        for agent in agents:
            avg_data[agent]=[]
            std_data[agent]=[]
            for problem in problems:
                values = results[problem][agent]
                values = np.array(values)[:, -1, 3]
                std_data[agent].append(np.std(values, -1))
                avg_data[agent].append(np.mean(values, -1))
            avg_data[agent] = np.mean(avg_data[agent], 0)
            std_data[agent] = np.mean(std_data[agent], 0)
        return avg_data, std_data

    def draw_rank_hist_prsr(self, data: dict, data_type: str,output_dir: str, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Generates and saves a bar plot with error bars representing the average PRSR (or similar metric) ranks for different agents, based on the provided data. The plot includes metric values, standard deviations, and agent labels, and is saved as either a PDF or PNG file.
        # Args:
        - data (dict): Part of the result data, the results[data_type].Also a nested dictionary,structured as `dict[problem][algorithm][run][generation][objective]`,stores the test result data.
        - data_type (str): The type or name of the metric being visualized (e.g., 'PRSR').
        - output_dir (str): Directory path where the generated plot will be saved.
        - pdf_fig (bool, optional): If True, saves the figure as a PDF; otherwise, saves as a PNG. Defaults to True.
        # Returns:
        - None
        # Notes:
        - The method uses `self.get_average_prsr_rank` to compute average metrics and standard deviations.
        - The plot is customized with agent labels, error bars, and formatted text for clarity.
        - The output file is named using the `data_type` and saved in the specified `output_dir`.
        """
        
        fig_type = 'pdf' if pdf_fig else 'png'
        metric, metric_std = self.get_average_prsr_rank(data)
        X, Y = list(metric.keys()), list(metric.values())
        _, S = list(metric_std.keys()), list(metric_std.values())
        n_agents = len(X)
        for i in range(n_agents):
            X[i] = to_label(X[i])

        plt.figure(figsize=(4*n_agents,15))
        plt.bar(X, Y)
        plt.errorbar(X, Y, S, fmt='s', ecolor='dimgray', ms=1, color='dimgray', elinewidth=5, capsize=30, capthick=5)
        for a,b in zip(X, Y):
            plt.text(a, b+0.05, '%.2f' % b, ha='center', fontsize=55)
        plt.xticks(rotation=45, fontsize=60)
        plt.yticks(fontsize=60)
        plt.ylim(0, np.max(np.array(Y) + np.array(S)) * 1.1)
        plt.title(f'The {data_type} for {self.config.test_problem}-{self.config.test_difficulty}', fontsize=70)
        plt.ylabel(f'{data_type}', fontsize=60)
        plt.savefig(output_dir + f'{data_type}_rank_hist.{fig_type}', bbox_inches='tight')

    def post_processing_test_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Processes and visualizes test statistics from a results file, generating tables and plots for performance analysis.
        # Args:
        - log_dir (str): The directory path where the test results (`test.pkl`) are stored and where output tables and figures will be saved.
        - pdf_fig (bool, optional): Whether to save generated figures in PDF format. Defaults to True.
        # Returns:
        - None
        # Description:
        This method loads test results from a pickle file, creates output directories if they do not exist, and generates a variety of tables and plots summarizing algorithm and agent performance. Outputs include overall statistics, algorithm complexity, agent performance tables, histograms, and boxplots for different metrics (cost, 'pr', 'sr'). The method supports saving figures in PDF format if specified.
        # Raises:
        - FileNotFoundError: If the specified `test.pkl` file does not exist in `log_dir`.
        - Any exceptions raised by the called table and plotting methods.
        """
        
        print('Post processing & drawing')
        with open(log_dir + 'test_results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metabbo = results['config'].baselines['metabbo']
        

        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        self.gen_overall_tab(results, log_dir + 'tables/')
        self.gen_algorithm_complexity_table(results, log_dir + 'tables/')
        # self.gen_agent_performance_table(results, log_dir + 'tables/')
        # self.gen_agent_performance_prsr_table(results['pr'],'pr', log_dir+'tables/') 
        # self.gen_agent_performance_prsr_table(results['sr'], 'sr',log_dir + 'tables/')
        

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        # self.draw_concrete_performance_hist(results['cost'], log_dir+'pics/',pdf_fig=pdf_fig)
        # self.draw_concrete_performance_prsr_hist(results['pr'], 'pr', log_dir+'pics/', pdf_fig = pdf_fig)
        # self.draw_concrete_performance_prsr_hist(results['sr'], 'sr', log_dir+'pics/', pdf_fig = pdf_fig)
        self.draw_boxplot(results, log_dir+'pics/', pdf_fig=pdf_fig)
        self.draw_boxplot_prsr(results['pr'], 'pr', log_dir+'pics/', pdf_fig=pdf_fig)
        self.draw_boxplot_prsr(results['sr'], 'sr', log_dir+'pics/', pdf_fig=pdf_fig)

        self.draw_test_data(results['cost'], 'cost', log_dir + 'pics/', logged=True, categorized=False, pdf_fig=pdf_fig, data_wrapper=np.array)
        self.draw_test_data(results['pr'],'pr', log_dir + 'pics/', logged=False, categorized=False, pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_hist)
        self.draw_test_data(results['sr'],'sr', log_dir + 'pics/', logged=False, categorized=False, pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_hist)
        # self.draw_rank_hist_prsr(results['pr'], 'pr',log_dir + 'pics/', pdf_fig=pdf_fig) 
        # self.draw_rank_hist_prsr(results['sr'], 'sr', log_dir + 'pics/',pdf_fig=pdf_fig)


    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        print('Post processing & drawing')
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        agent_for_rollout = results['agent_for_rollout']
        self.draw_train_logger('return', results['steps'], results['return'],agent_for_rollout, log_dir + 'pics/', pdf_fig=pdf_fig)
        self.draw_train_logger('cost', results['steps'],results['cost'], agent_for_rollout,log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_cost_rollout)
        self.draw_train_logger('pr', results['steps'],results['pr'], agent_for_rollout,log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_rollout)
        self.draw_train_logger('sr', results['steps'],results['sr'], agent_for_rollout,log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper=self.data_wrapper_prsr_rollout)

#logger
# class basic_Logger:
class MTO_Logger(Basic_Logger):
    """
    # Introduction
    The customized logger for multi-task optimization(MTO) scenario.
    """
    def __init__(self, config):
        super().__init__(config)

    def draw_avg_train_return(self, data: list, output_dir: str) -> None: 
        """
        # Introduction
        Plots and saves the average training return over learning steps using the provided data.
        # Args:
        - data (list): A list of lists or arrays containing return values for each epoch and environment.
        - output_dir (str): The directory path where the output plot image will be saved.
        # Returns:
        - None
        # Notes:
        - The plot is saved as 'avg_mto_return.png' in the specified output directory.
        """
        plt.figure()
        return_data = np.array(data,dtype=np.float32) #[epochs, env_cnt]
        x = np.arange(return_data.shape[0])
        y = np.mean(return_data, axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Return')
        plt.grid()
        plt.savefig(output_dir + f'avg_mto_return.png', bbox_inches='tight')
        plt.close()

    def draw_avg_train_cost(self, data:list, output_dir: str) -> None:
        """
        # Introduction
        Plots and saves the average training cost over learning steps using the provided cost data.
        # Args:
        - data (list): A list representing cost data with shape [epochs, env_cnt, task_cnt].
        - output_dir (str): The directory path where the output plot image will be saved.
        # Returns:
        - None
        # Notes:
        The function computes the mean cost across environments and tasks for each epoch, plots the result, and saves the figure as 'avg_mto_cost.png' in the specified output directory.
        """
        
        plt.figure()
        cost_data = np.array(data,dtype=np.float32) #[epochs, env_cnt, task_cnt]
        x = np.arange(cost_data.shape[0])
        y = np.mean(np.mean(cost_data, axis=-1), axis=-1)
        plt.plot(x, y, 
         color='blue',       
         marker='o',         
         linestyle='-',     
         linewidth=2,        
         markersize=8)       
        plt.xlabel('Learning Steps')
        plt.ylabel('Avg Cost')
        plt.grid()
        plt.savefig(output_dir + f'avg_mto_cost.png', bbox_inches='tight')
        plt.close()

    def draw_per_task_cost(self, data:list, output_dir: str) -> None:
        """
        # Introduction
        Plots and saves the cost (or value) curves for each task over epochs, given a dataset of per-task values.
        # Args:
        - data (list): A list or nested list containing per-task values for each epoch. Can be a 2D or 3D structure.
        - output_dir (str): The directory path where the output plot image will be saved.
        # Returns:
        - None
        # Notes:
        - If `data` is 3D, it is averaged along the second axis before plotting.
        - The function generates one subplot per task, showing the value progression over epochs.
        - The output image is saved as 'mto_each_task_cost.png' in the specified directory.
        """
        
        data = np.array(data, dtype=np.float32)
        if data.ndim == 3:  
            data = np.mean(data, axis=1)

        epochs, task_cnt = data.shape
        fig, axes = plt.subplots(task_cnt, 1, figsize=(10, 2 * task_cnt))  
        if task_cnt == 1:
            axes = [axes]

        for task_idx in range(task_cnt):
            ax = axes[task_idx]
            ax.plot(range(epochs), data[:, task_idx], color='blue', label=f'Task {task_idx+1}')
            ax.set_title(f'Task {task_idx+1}')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_dir + f'mto_each_task_cost.png', bbox_inches='tight')
        plt.close()

    def save_mto_cost_to_csv(self, data:list, output_dir: str) -> None:
        """
        # Introduction
        Saves multi-task optimization (MTO) cost data to a CSV file, optionally averaging over a specific axis if the input data is 3-dimensional.
        # Args:
        - data (list): A list (or nested list) containing cost values for each task and epoch. Can be 2D or 3D (in which case it is averaged over axis 1).
        - output_dir (str): The directory path where the CSV file will be saved.
        # Returns:
        - None
        # Notes:
        - The output CSV file will be named 'mto_each_task_cost.csv' and will contain columns for each task and an 'Epoch' column.
        """
        
        data = np.array(data, dtype=np.float32)
        if data.ndim == 3:  
            data = np.mean(data, axis=1)

        epochs, task_cnt = data.shape
        df = pd.DataFrame(data, columns=[f'Task_{i+1}' for i in range(task_cnt)])
        df.insert(0, 'Epoch', np.arange(epochs))
        output_path = output_dir + f'mto_each_task_cost.csv'
        df.to_csv(output_path, index=False)

    def save_mto_reward_to_csv(self, data:list, output_dir: str) -> None:
        """
        # Introduction
        Saves the mean of multi-task optimization (MTO) reward data to a CSV file. The function processes the input data, computes the mean across the last axis if the data is 2-dimensional, and writes the results to a CSV file with epoch indices.
        # Args:
        - data (list): A list (or nested list) of reward values, where each element represents reward data for an epoch or task.
        - output_dir (str): The directory path where the output CSV file will be saved.
        # Returns:
        - None
        # Notes:
        - The output CSV file will be named 'mto_return.csv' and will contain two columns: 'Epoch' and 'Value'.
        - If the input data is 2D, the mean is computed along the last axis before saving.
        """
        
        data = np.array(data, dtype=np.float32)
        if data.ndim == 2:  
            data = np.mean(data, axis=-1)
        epochs = data.shape[0]
        df = pd.DataFrame({
            "Epoch": np.arange(epochs),  
            "Value": data              
        })
        output_path = output_dir + f'mto_return.csv'
        df.to_csv(output_path, index=False)

    def draw_env_task_cost(self, data:list, output_dir:str) -> None:
        """
        # Introduction
        Plots and saves the performance metrics of multiple tasks across different environments over epochs.
        # Args:
        - data (list): A 3D list or array-like structure with shape (epochs, env_cnt, task_cnt), containing metric values.
        - output_dir (str): The directory path where the generated plot images will be saved.
        # Returns:
        - None
        # Notes:
        - For each task, a separate plot is generated showing the metric values for all environments across epochs.
        - Plots are saved as PNG files in the specified output directory, with filenames indicating the corresponding task.
        - If the input data has fewer than 3 dimensions, the function returns without plotting.
        """
        
        data = np.array(data, dtype=np.float32)
        if data.ndim < 3:
            return 
        epochs, env_cnt, task_cnt = data.shape

        for task in range(task_cnt):
            plt.figure(figsize=(10, 5))
            for env in range(env_cnt):
                plt.plot(data[:, env, task], label=f'Env {env+1}')
            plt.title(f'Task {task+1} Performance Across Environments')
            plt.xlabel('Epochs')
            plt.ylabel('Metric Value')
            plt.legend()
            plt.grid()
        plt.savefig(output_dir + f'mto_env_task_{task+1}_cost.png', bbox_inches='tight')
        plt.close()
    
    def draw_test_cost(self, data: dict, output_dir: str):
        """
        # Introduction
        Plots and saves the performance metrics of different algorithms and problems.
        # Args:
        - data (dict): A dict contains a 3D list or an array-like structure for each algorithm and problem with shape (test_epoch, log_point, task), containing metric values.
        - output_dir (str): The directory path where the generated plot images will be saved.
        # Returns:
        - None
        # Notes:
        - For each algorithm and problem, a separate plot is generated showing the metric values.
        - Plots are saved as PNG files in the specified output directory, with filenames indicating the corresponding task.
        """
        for problem_name in data.keys():
            for algorithm_name in data[problem_name].keys():
                arr = np.array(data[problem_name][algorithm_name])          #(2,63,2) [test_run, log_points,tasks]
                mean_result = np.mean(arr, axis=(0, 2))  # 形状 (63,)
                plt.figure(figsize=(10, 5))
                plt.plot(mean_result, marker='o', linestyle='-', color='b', label=f'{algorithm_name}')
                plt.xlabel('log_points')
                plt.ylabel('Averaged Value')
                plt.title(f'{problem_name}_{algorithm_name}_Cost_Test_Curves')
                plt.grid(True)
                plt.legend()
                plt.savefig(output_dir + f'mto_test_cost_{problem_name}_{algorithm_name}.png', bbox_inches='tight')

    def post_processing_test_statics(self, log_dir: str) -> None:
        """
        # Introduction
        Post-processes test statistics by loading results, generating summary tables, and creating visualizations for algorithm performance evaluation.
        # Args:
        - log_dir (str): The directory path where the plot images generated from test datas will be saved.
        # Returns:
        - None
        """

        print('Post processing & drawing')
        with open(log_dir + 'test_results.pkl', 'rb') as f:
            results = pickle.load(f)
            
        metabbo = results['config'].baselines['metabbo']
        bbo = results['config'].baselines['bbo']
        
        if not os.path.exists(log_dir + 'tables/'):
            os.makedirs(log_dir + 'tables/')

        self.gen_algorithm_complexity_table(results, log_dir + 'tables/')

        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')

        # 如果需要，可以为不同的算法绘制图形（例如 cost 图）
        if 'cost' in results:
            self.draw_test_cost(results['cost'], log_dir + 'pics/')
    
    @staticmethod
    def data_wrapper_mto_cost_rollout(data):
        """
        # Introduction
        Reshape the MTO rollout datas from 3D to 2D. 
        # Args:
        - data (list): A 3D list or array-like structure containing rollout datas to be reshaped.
        # Returns:
        - numpy.ndarray: A 2D reshaped rollout datas.
        """
        res = np.array(data)
        res = np.mean(res, axis=-1)
        return res[:, -1]

    def draw_train_logger(self, data_type: str, steps: list, data: dict, agent_for_rollout: str, output_dir: str, ylabel: str = None, norm: bool = False, pdf_fig: bool = True, data_wrapper: Callable = None) -> None:
        """
        # Introduction
        Plots and saves the training curve for a given data type, applying smoothing and displaying mean and standard deviation shading. Supports normalization and custom data processing.
        # Args:
        - data_type (str): The type of data being plotted. e.g. cost
        - steps (list): List of step values (x-axis) corresponding to the data points.
        - agent_for_rollout (str): A string represents the agent during the rollout process. 
        - data (dict): Part of the result data,the result[data_type]. Also a nested dictionary containing experimental datas structured as `dict[problem][algo][run]`.
        - output_dir (str): Directory path where the output figure will be saved.
        - ylabel (str, optional): Label for the y-axis. If None, uses `data_type` as the label. Defaults to None.
        - norm (bool, optional): Whether to normalize the data before plotting. Defaults to False.
        - pdf_fig (bool, optional): Whether to save the figure as a PDF (if True) or PNG (if False). Defaults to True.
        - data_wrapper (Callable, optional): Optional function to preprocess or wrap the data before averaging. Defaults to None.
        # Returns:
        - None
        # Notes:
        - The function applies a smoothing operation to the plotted curve based on the configuration.
        - The mean and standard deviation are visualized, with the standard deviation shown as a shaded region.
        - The color arrangement for each agent is managed to ensure consistent coloring across plots.
        """
        means, stds = self.get_average_data(data, norm=norm, data_wrapper=data_wrapper)
        plt.figure()

        y = np.array([means[k] for k in means])
        y_std = np.array([stds[k] for k in stds])
        x = np.array(steps, dtype = np.float64)

        s = np.zeros(y.shape[0])
        a = s[0] = y[0]

        norm = 0.8 + 1
        for i in range(1, y.shape[0]):
            a = a * 0.8 + y[i]
            s[i] = a / norm if norm > 0 else a
            norm *= 0.8
            norm += 1
        if agent_for_rollout not in self.color_arrangement.keys():
            self.color_arrangement[agent_for_rollout] = colors[self.arrange_index]
            self.arrange_index += 1

        plt.plot(x, s, label = to_label(agent_for_rollout), marker = '*', markersize = 12, markevery = 2, c = self.color_arrangement[agent_for_rollout])
        plt.fill_between(x, (s - y_std), (s + y_std), alpha = 0.2, facecolor = self.color_arrangement[agent_for_rollout])

        plt.legend()
        plt.grid()
        plt.xlabel('Learning Steps')
        if ylabel is None:
            ylabel = data_type
        plt.ylabel(ylabel)
        fig_type = 'pdf' if pdf_fig else 'png'
        plt.savefig(output_dir + f'avg_{data_type}_curve.{fig_type}', bbox_inches='tight')
        plt.close()

    def post_processing_rollout_statics(self, log_dir: str, pdf_fig: bool = True) -> None:
        """
        # Introduction
        Processes rollout statistics after the rollout process, generates plots for return and cost, and saves them to the specified directory.
        # Args:
        - log_dir (str): The directory path where the rollout statistics file ('rollout.pkl') is located and where the output plots will be saved.
        - pdf_fig (bool, optional): Whether to save the generated plots as PDF files. Defaults to True.
        # Returns:
        - None
        """
        print('Post processing & drawing')
        with open(log_dir+'rollout.pkl', 'rb') as f:
            results = pickle.load(f)
        if not os.path.exists(log_dir + 'pics/'):
            os.makedirs(log_dir + 'pics/')
        agent_for_rollout = results['agent_for_rollout']
        self.draw_train_logger('return', results['steps'], results['return'], agent_for_rollout, log_dir + 'pics/', pdf_fig=pdf_fig)
        self.draw_train_logger('cost', results['steps'], results['cost'], agent_for_rollout, log_dir + 'pics/', pdf_fig=pdf_fig, data_wrapper = MTO_Logger.data_wrapper_mto_cost_rollout)

