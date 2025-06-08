import numpy as np
import gym, os, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal
from .vectorenvs import DummyVectorEnv, SubprocVectorEnv, RayVectorEnv, RaySubprocVectorEnv
import warnings
import psutil, GPUtil, torch

class ParallelEnv():
    __VectorEnvOption = {'dummy': DummyVectorEnv,
                       'subproc': SubprocVectorEnv,
                       'ray': RayVectorEnv,
                       'ray-subproc': RaySubprocVectorEnv,
                       }
    def __init__(self,
                 envs: List[gym.Env], 
                 para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
                 asynchronous: Literal[None, 'idle', 'restart', 'continue']=None,
                 num_cpus: Optional[Union[int, None]]=None,
                 num_gpus: int=0,
                 no_warning=True,
                 ) -> None:
        """An integrated parallel Environment.
        :param envs:           The list of the GYM style Envs to be processed in parallel.
        :param para_mode:      The mode for parallel, can be:
            * ``dummy`` (sequential processing)
            * ``subproc`` (multi-processing parallel)
            * ``ray`` (parallel with Ray)
            * ``ray-subproc`` (hybrid parallel which uses subproc envs under each ray worker)
        :param asynchronous:   Whether to use asynchronous processing for sub envs with different life length.
            * ``None`` means terminating all sub envs when any one of them is done, after that any actions will get None state, None reward, True done and empty info;
            * ``idle`` means the living envs will return step results normally while the results of the done sub envs will be replaced with None results above;
            * ``restart`` means when an env is done, it will be reset immediately, its returned state will be the first state after reset and the last state before reset can be found in info['ended_state'] and the flag info['reset'] is True;
            * ``continue`` means all envs will reveive actions and return results normally even they are done, the processing logits are determined by users.
        :param num_cpus:       The number of cpu cores assigned for this parallel environment, default to be all cores.
        :param no_warning:     Whether to show warnings from the envs, default to True (not show warnings).
        """
        self.envs = envs
        self.para_mode = para_mode
        self.asynchronous = asynchronous
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.length = len(envs)
        
        # resource allocation
        if self.num_cpus is None:
            self.num_cpus = int(os.cpu_count() * (1 - psutil.cpu_percent() / 100))
            if self.num_cpus < 1:
                self.num_cpus = os.cpu_count()  # anyway, take all available
        # if self.num_gpus is None:
        #     self.num_gpus = torch.cuda.device_count()
        num_cpu_per_worker = 1
        num_gpu_per_worker = self.num_gpus
        if 'ray' in self.para_mode:
            num_cpu_per_worker = max(1, self.num_cpus // self.length)  # use multiple cpu cores for one ray task
            if num_gpus > 0:
                num_gpu_per_worker = max(1, self.num_gpus // self.length)
        # construct vector env
        self.VectorEnv = self.__VectorEnvOption[self.para_mode]
        if self.para_mode != 'ray-subproc' or self.num_cpus >= self.length:  # if not use ray-subproc vector env or cpu resource is enough, no need to use subproc in ray workers
            if self.para_mode == 'ray-subproc':
                self.para_mode = 'ray'
                warnings.warn('The number of allocated cpu is enough to assign each env a ray worker, downgraded to Ray')
            env_list = [lambda e=p: e for p in self.envs]
            self.workers = self.VectorEnv(env_list, num_cpus=self.num_cpus, num_gpus=self.num_gpus, num_cpu_per_worker=num_cpu_per_worker, num_gpu_per_worker=num_gpu_per_worker, no_warning=no_warning)
        else:
            # if use ray-subproc vector env, which employs several subproc envs in each ray worker,
            # first determine the number of workers
            num_workers = min(self.num_cpus, self.length)
            # and the number of subproc envs for each ray worker
            num_env_per_worker = int(np.ceil(self.length / num_workers))
            # then allocate the envs to each ray worker
            sub_envs_list = []
            for w in range(num_workers):
                sub_envs_list.append([lambda e=p: e for p in self.envs[w*num_env_per_worker:(w+1)*num_env_per_worker]])
            self.workers = self.VectorEnv(sub_envs_list, num_cpus=self.num_cpus, num_gpus=self.num_gpus, num_cpu_per_worker=num_cpu_per_worker, num_gpu_per_worker=num_gpu_per_worker, no_warning=no_warning)
        # record the done situation of the sub envs
        self.done = np.ones(self.length, dtype=bool)

    def __len__(self):
        return self.length

    def get_env_attr(
        self,
        key: str,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ):
        """get a attribute value for all or spercific (through id) envs, if the attribute doesn't exist in the env(s), it will return None(s)"""
        return self.workers.get_env_attr(key, id)
    
    def set_env_attr(
        self,
        key: str,
        value: Any,
        id: Optional[Union[int, List[int], np.ndarray]] = None
    ):
        """set the value of a attribute in all or spercific (through id) envs"""
        self.workers.set_env_attr(key, value, id)
        
    def has_done(self):
        """if there is an env done"""
        return self.done.any()
    
    def all_done(self):
        """if all envs done"""
        return self.done.all()
    
    def customized_method(self, env_method: str, data = None, id: Optional[Union[int, List[int], np.ndarray]] = None):
        """if user declares a method in the env named [env_method] and requiring arguments in a dictionary [data], this method can call the function in parallel.
        ```python
        # For instance, to run a ``func`` method on 8 of a batch of 16 envs which requires
        # an argument named ``x``, then construct a list of 8 argument dictionaries:
        data = [{'x': ...}, {...}, ...]
        # and the id of the envs to run the func:
        id = [0, 1, 2, ...]
        # call the customized_method
        results = VectorEnv.customized_method("func", data, id)
        ```
        """
        if data is not None:
            return self.workers.customized_method(env_method, data, id)
        else:
            return self.workers.customized_method(env_method, id)
        
    def reset(self, id: Optional[Union[int, List[int], np.ndarray]] = None):
        """reset the envs with index in [id], default to reset all envs"""
        states = self.workers.reset(id)
        if id is None:
            # applied to all sub envs
            self.done = np.zeros(self.length, dtype=bool)
        else:
            # applied to spercific sub envs
            self.done[id] = False
        return states
    
    def seed(self, seed: Optional[Union[int, List[int], np.ndarray]]):
        """set the seed for all envs"""
        self.workers.seed(seed)
    
    def step(self,
             action,
             id: Optional[Union[int, List[int], np.ndarray]] = None,  # 
             align: Literal['batch', 'item'] = 'item'
             ):
        """take a step in envs.
        :param action: the actions to be take.
        :param id:     the index of the envs to take the action, corresponding one-to-one with action, default to take steps in all envs.
        :param align:  the alignment mode of the output, ``batch`` means output as a batch of 4-item tuples [<state, reward, done, info>, <...>, ...], ``item`` means output as 4 batched data <states, rewards, dones, infos>
        
        """
        if id is not None:
            obs_list, rew_list, done_list, info_list = self.workers.step(action ,id)
            self.__update_done(done_list)
            return self.__process_align(obs_list, rew_list, done_list, info_list, align)
        idle_template = [None] * self.length
        if self.asynchronous is None:  # synchronous
            if not self.has_done():
                obs_list, rew_list, done_list, info_list = self.workers.step(action ,id)
                self.__update_done(done_list)
                if self.has_done():
                    self.done = np.ones(self.length, dtype=bool)
                return self.__process_align(obs_list, rew_list, done_list, info_list, align)
            else:
                return self.__process_align(idle_template, idle_template, [True] * self.length, [{}] * self.length, align)
        elif self.asynchronous == 'idle':
            if self.all_done():
                return self.__process_align(idle_template, idle_template, [True] * self.length, [{}] * self.length, align)
            obs = np.array([None for _ in range(self.length)])
            rew = np.array([None for _ in range(self.length)])
            done = np.ones(self.length, dtype=bool)
            info = np.array([{} for _ in range(self.length)])
            
            active_id = np.arange(self.length)[~self.done]
            act_obs, act_rew, act_done, act_info = self.workers.step(np.array(action)[active_id], active_id)
            for i, j in enumerate(active_id):
                obs[j] = act_obs[i]
                info[j] = act_info[i]
                rew[j] = act_rew[i]
            done[active_id] = act_done
            
            if not self.has_done():
                rew = np.stack(rew)
                try:
                    obs = np.stack(obs.tolist())
                except ValueError:  # different len(obs)
                    pass
            self.__update_done(np.array(done, dtype=bool))
            return self.__process_align(obs, rew, done, info, align)
        elif self.asynchronous == 'restart':
            assert not self.has_done()
            obs_list, rew_list, done_list, info_list = self.workers.step(action)
            
            if done_list.any():
                ended_id = np.arange(self.length)[done_list]
                next_states = self.reset(ended_id)
                i = 0
                for id in range(self.length):
                    if done_list[id]:
                        info_list[id]['ended_state'] = copy.deepcopy(obs_list[id])
                        info_list[id]['reset'] = True
                        obs_list[id] = copy.deepcopy(next_states[i])
                    else:
                        info_list[id]['reset'] = False
            return self.__process_align(obs_list, rew_list, self.done, info_list, align)
        elif self.asynchronous == 'continue':
            obs_list, rew_list, done_list, info_list = self.workers.step(action)
            self.__update_done(done_list)
            return self.__process_align(obs_list, rew_list, done_list, info_list, align)
        else:
            self.close()
            raise NotImplementedError
    
    def __process_align(self, obs_list, rew_list, done_list, info_list, align: Literal['batch', 'item'] = 'item'):
        if align == 'item':
            return obs_list, rew_list, done_list, info_list
        else:
            return np.array([list(item) for item in zip(*(obs_list, rew_list, done_list, info_list))], dtype=object)
        
    def __update_done(self, done_list):
        self.done = done_list
    
    def rollout(self, ):
        assert self.para_mode == 'ray'
        return self.workers.rollout()
    
    def close(self):
        """close the envs to avoid memory or process leak"""
        self.workers.close()
    
    # def __del__(self):
    #     """if the environment is not closed before being deleted, close it"""
    #     if not self.workers.is_closed:
    #         self.close()
            
