import copy
import os
import torch
from torch import nn
from ...rl.basic_agent import Basic_Agent
from ...environment.parallelenv.parallelenv import ParallelEnv
from typing import Optional, Union, Literal, List
import numpy as np
from ...rl.utils import clip_grad_norms, save_class
from cmaes import CMA
from dill import loads, dumps

class LGA(Basic_Agent):
    """
    # Introduction
    **L**earned **G**enetic **A**lgorithm parametrizes selection and mutation rate adaptation as cross- and self-attention modules and use MetaBBO to evolve their parameters on a set of diverse optimization tasks.
    # Original paper
    "[**Discovering attention-based genetic algorithms via meta-black-box optimization**](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)." Proceedings of the Genetic and Evolutionary Computation Conference. (2023).
    # Official Implementation
    [LGA](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/lga.py)
    # Args:
    - config (object): Configuration object containing agent and training parameters, such as device, save directories, and training intervals.
    # Attributes:
    - M (int): Population size for the optimizer.
    - T (int): Number of steps to skip in each action.
    - J (int): Number of evaluations per candidate.
    - optimizer (CMA): CMA-ES optimizer instance.
    - x_population (np.ndarray): Current population of candidate solutions.
    - meta_performances (List[List[float]]): Performance history for each candidate.
    - best_x (np.ndarray): Best candidate solution found so far.
    - costs (Any): Placeholder for cost tracking.
    - best_lga (int): Index of the best candidate in the population.
    - gbest (float): Best fitness score achieved.
    - learning_step (int): Number of learning steps completed.
    - cur_checkpoint (int): Current checkpoint index for saving.
    - task_step (int): Number of tasks (episodes) completed.
    # Methods:
    - __str__(): Returns the string identifier for the agent.
    - get_step(): Returns the current learning step.
    - optimizer_step(): Samples a new population from the optimizer.
    - train_episode(...): Runs a training episode, evaluates the population, and updates the optimizer.
    - rollout_episode(...): Evaluates the best candidate in a single environment.
    - update(): Updates the optimizer with new fitness scores and refreshes the population.
    - log_to_tb_train(...): Logs training metrics and extra information to TensorBoard.
    # Returns:
    Methods return various outputs, such as training progress, evaluation results, and logging status. See individual method docstrings for details.
    # Raises:
    - Any exceptions raised by underlying environment, optimizer, or file I/O operations may propagate.
    # Notes:
    - This class assumes the existence of supporting classes and functions such as `Basic_Agent`, `CMA`, `ParallelEnv`, `save_class`, and appropriate environment interfaces.
    - The agent is designed for meta-optimization tasks and may require adaptation for specific problem domains.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.M = 256
        self.T = 50
        self.J = 256
        self.optimizer = CMA(mean = np.zeros(673),
                             sigma = 0.1,
                             population_size = self.M)

        self.x_population = None
        self.meta_performances = None
        self.optimizer_step()
        self.best_x = self.x_population[0]

        self.costs = None
        self.best_lga = None
        self.gbest = 1e-10

        self.learning_step = 0
        self.cur_checkpoint = 0

        self.task_step = 0
        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1

    def __str__(self):
        return "LGA"

    def get_step(self):
        return self.learning_step

    def optimizer_step(self):
        # inital sampling
        samples = []
        for _ in range(self.M):
            samples.append(self.optimizer.ask())
        self.x_population = np.vstack(samples)
        self.meta_performances = [[] for _ in range(self.M)]

    def train_episode(self,
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      # todo: asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      # num_cpus: Optional[Union[int, None]] = 1,
                      # num_gpus: int = 0,
                      compute_resource = {},
                      tb_logger = None,
                      required_info = {}):
        num_cpus = None
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus = num_cpus, num_gpus = num_gpus)
        env.seed(seeds)

        env.set_env_attr("rng_cpu", "None")
        if self.config.device != 'cpu':
            env.set_env_attr("rng_gpu", "None")
        env_population = [loads(dumps(env)) for _ in range(self.M)]

        for i, e in enumerate(env_population):
            e.reset()
            action = {'net': self.x_population[i],
                      'skip_step': self.T}

            action = [copy.deepcopy(action) for _ in range(len(env))]
            sub_best, _, _, _ = e.step(action)

            self.meta_performances[i].append(sub_best)

        self.task_step += len(env)
        # Task 256
        if self.task_step % 256 == 0:
            self.update()
            self.learning_step += 1
            if not self.config.no_tb:
                self.log_to_tb_train(tb_logger, self.learning_step, self.gbest)

        if self.learning_step >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
            save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
            self.cur_checkpoint += 1

        return_info = {'return': 0, 'loss':0 ,'learn_steps': self.learning_step, }
        return_info['gbest'] = env_population[0].get_env_attr('cost')[-1],
        for key in required_info.keys():
            return_info[key] = env_population[0].get_env_attr(required_info[key])
        for i, e in enumerate(env_population):
            e.close()
        # return exceed_max_ls
        return self.learning_step >= self.config.max_learning_step, return_info

    def rollout_episode(self, env, seed = None, required_info = {}):
        env.seed(seed)
        R = 0
        # use best_x to rollout
        env.reset()
        action = {'net': self.best_x,
                  'skip_step': None}
        gbest, r, _, _ = env.step(action)
        R += r

        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        results = {'cost': env_cost, 'fes': env_fes, 'return': R}

        if self.config.full_meta_data:
            meta_X = env.get_env_attr('meta_X')
            meta_Cost = env.get_env_attr('meta_Cost')
            metadata = {'X': meta_X, 'Cost': meta_Cost}
            results['metadata'] = metadata

        for key in required_info.keys():
            results[key] = getattr(env, required_info[key])

        return results

    def update(self):
        scores = np.stack(self.meta_performances).reshape(self.M, -1) # [M, J]
        M_mean = np.mean(scores, axis = 1, keepdims = True)
        M_std = np.std(scores, axis = 1, keepdims = True)

        scores = (scores - M_mean) / (M_std + 1e-8)

        self.fitness = np.median(scores, axis = 1)

        self.meta_performances = [[] for _ in range(self.M)]

        if np.min(self.fitness) > self.gbest:
            self.gbest = np.max(self.fitness)
            self.best_lga = np.argmax(self.fitness)
            self.best_x = self.x_population[self.best_lga]

        self.optimizer.tell(list(zip(self.x_population, self.fitness)))
        self.optimizer_step()

    def log_to_tb_train(self, tb_logger, mini_step,gbest,
                        extra_info = {}):
        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # train metric
        tb_logger.add_scalar('train/gbest', self.gbest, mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)





