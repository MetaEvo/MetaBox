from typing import Tuple
# from src.agent.basic_agent import Basic_Agent
import torch
import math, copy
from typing import Any, Callable, List, Optional, Tuple, Union, Literal

from torch import nn
import torch
from torch.distributions import Normal
import torch.nn.functional as F
from rl.utils import *
from rl.basic_agent import Basic_Agent


from environment.parallelenv.parallelenv import ParallelEnv


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class VDN_Agent(Basic_Agent):
    def __init__(self, config,network: dict, learning_rates: float):
        super().__init__(config)
        self.config = config

        # define parameters
        self.n_agent = self.config.n_agent
        self.n_act = self.config.n_act
        self.available_action = self.config.available_action
        self.memory_size = self.config.memory_size
        self.warm_up_size = self.config.warm_up_size
        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon
        self.max_grad_norm = self.config.max_grad_norm
        self.batch_size = self.config.batch_size
        self.chunk_size = self.config.chunk_size
        self.update_iter = self.config.update_iter
        self.device = self.config.device
        
        self.replay_buffer = MultiAgent_ReplayBuffer(self.memory_size)
        self.set_network(network, learning_rates)

        # figure out the actor network
        # self.model = None
        # assert hasattr(self, 'model')

        # # figure out the optimizer
        # assert hasattr(torch.optim, self.config.optimizer)
        # self.optimizer = eval('torch.optim.' + self.config.optimizer)(
        #     [{'params': self.model.parameters(), 'lr': self.config.lr_model}])
        # # figure out the lr schedule
        # assert hasattr(torch.optim.lr_scheduler, self.config.lr_scheduler)
        # self.lr_scheduler = eval('torch.optim.lr_scheduler.' + self.config.lr_scheduler)(self.optimizer,
        #                                                                                  self.config.lr_decay,
        #                                                                                  last_epoch=-1, )

        # assert hasattr(torch.nn, self.config.criterion)
        # self.criterion = eval('torch.nn.' + self.config.criterion)()

        # self.replay_buffer = MultiAgent_ReplayBuffer(self.memory_size)

        # # move to device
        # self.model.to(self.device)

        # init learning time
        self.learning_time = 0
        self.cur_checkpoint = 0

        # save init agent
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint += 1
    
    def set_network(self, networks: dict, learning_rates: float):
        Network_name = []
        if networks:
            for name, network in networks.items():
                Network_name.append(name)
                setattr(self, name, network)  # Assign each network in the dictionary to the class instance
        self.network = Network_name

        assert hasattr(self, 'model')  # Ensure that 'model' is set as an attribute of the class
        self.target_model = copy.deepcopy(self.model)
        if isinstance(learning_rates, (int, float)):
            learning_rates = [learning_rates] * len(networks)
        elif len(learning_rates) != len(networks):
            raise ValueError("The length of the learning rates list must match the number of networks!")

        all_params = []
        for id, network_name in enumerate(networks):
            network = getattr(self, network_name)
            all_params.append({'params': network.parameters(), 'lr': learning_rates[id]})

        assert hasattr(torch.optim, self.config.optimizer)
        self.optimizer = eval('torch.optim.' + self.config.optimizer)(all_params)

        assert hasattr(torch.nn, self.config.criterion)
        self.criterion = eval('torch.nn.' + self.config.criterion)()

        for network_name in networks:
            getattr(self, network_name).to(self.device)

    def get_step(self):
        return self.learning_time

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_time = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

    def get_action(self, state, epsilon_greedy=False):
        state = torch.Tensor(state).to(self.device)
        with torch.no_grad():
            Q_list = self.model(state)
        if epsilon_greedy and np.random.rand() < self.epsilon:
            action = np.zeros((len(state), self.n_agent), dtype=int)
            for i in range(self.n_agent):
                action[:, i] = np.random.randint(low=0, high=self.available_action[i], size=len(state))
        else:
            for i in range(self.n_agent):
                action[:, i] = torch.argmax(Q_list[:, i, :self.available_action[i]], -1).detach().cpu().numpy().astype(int)
        return action

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
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
        env.seed(seeds)
        # params for training
        gamma = self.gamma

        state = env.reset()
        try:
            state = torch.FloatTensor(state)
        except:
            pass

        _R = torch.zeros(len(env))
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            action = self.get_action(state=state, epsilon_greedy=True)

            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward[:, 0]
            _reward.append(torch.FloatTensor(reward[:, 0]))
            # store info
            # convert next_state into tensor
            for s, a, r, ns, d in zip(state.numpy(), action, reward, next_state, is_end):
                self.replay_buffer.append((s, a, r, ns, d))
            try:
                state = torch.FloatTensor(next_state).to(self.device)
            except:
                state = copy.deepcopy(next_state)
            # begin update
            if len(self.replay_buffer) >= self.warm_up_size:
                for _ in range(self.update_iter):
                    batch_obs, batch_action, batch_reward, batch_next_obs, batch_done \
                        = self.replay_buffer.sample_chunk(self.batch_size,self.chunk_size)
                    loss = 0
                    for step_i in range(self.chunk_size):
                        q_out = self.model(batch_obs[:,step_i,:,:].to(self.device))
                        q_a = q_out.gather(2, batch_action[:, step_i, :].unsqueeze(-1).long()).squeeze(-1)
                        sum_q = q_a.sum(dim=1, keepdims=True)
                        max_q_prime = self.target_model(batch_next_obs[:, step_i, :, :])
                        max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)
                        target_q = batch_reward[:, step_i, :].sum(dim=1, keepdims=True)
                        target_q += self.gamma * max_q_prime.sum(dim=1, keepdims=True) * (1 - batch_done[:, step_i])

                        loss += self.criterion(sum_q, target_q.detach())
                    loss = loss / self.chunk_size
                    _loss.append(loss.item())
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                    self.optimizer.step()
                    self.learning_time += 1
                    
                    if self.config.target_update_interval is not None and self.learning_time % self.config.target_update_interval == 0:
                        self.target_model.load_state_dict(self.model.state_dict())
                    
                    if not self.config.no_tb:
                        self.log_to_tb_train(tb_logger, self.learning_time,
                                            grad_norms,
                                            loss,
                                            _R, _reward,
                                            q_out, max_q_prime)
                    if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                        self.cur_checkpoint += 1
                    
                    if self.learning_time >= self.config.max_learning_step:
                        return_info = {'return': _R, 'learn_steps': self.learning_time, }
                        env_cost = env.get_env_attr('cost')
                        return_info['gbest'] = env_cost[-1]
                        for key in required_info.keys():
                            return_info[key] = env.get_env_attr(required_info[key])
                        env.close()
                        return self.learning_time >= self.config.max_learning_step, return_info

        is_train_ended = self.learning_time >= self.config.max_learning_step
        return_info = {'return': _R, 'learn_steps': self.learning_time, }
        env_cost = env.get_env_attr('cost')
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()

        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed = None,
                        required_info = {}):
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                except:
                    state = [state]
                action = self.get_action(state)[0]
                state, reward, is_done = env.step(action)
                R += reward[:,0]
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            env_metadata = env.get_env_attr('metadata') 
            results = {'cost': env_cost, 'fes': env_fes, 'return': R, 'metadata': env_metadata}
            for key in required_info.keys():
                results[key] = env.get_env_attr(env, required_info[key])
            return results

    def rollout_batch_episode(self,
                              envs,
                              seeds=None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                              # todo: asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                              # num_cpus: Optional[Union[int, None]] = 1,
                              # num_gpus: int = 0,
                              compute_resource = {},
                              required_info = {}):
        num_cpus = None
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
        if seeds is not None:
            env.seed(seeds)
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass

        R = torch.zeros(len(env))
        # sample trajectory
        while not env.all_done():
            with torch.no_grad():
                action = self.get_action(state)

            # state transient
            state, rewards, is_end, info = env.step(action)
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards[:,0]).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
        env_cost = env.get_env_attr('cost')
        env_fes = env.get_env_attr('fes')
        env_metadata = env.get_env_attr('metadata') 
        results = {'cost': env_cost, 'fes': env_fes, 'return': R, 'metadata': env_metadata}
        '''
        cost: 每log_interval(config中设置)的最优评估值 : config.log_interval = config.maxFEs // config.n_logpoint(记录次数)
        fes: 评估次数
        return: 奖励
        metadata: 
            meta_X: 所有评估值
            meta_Cost: 所有评估点

        针对非并行环境,若并行环境则为np.array(len(envs)): 如'fes': np.array(['fes' for env in envs])
        results = {'cost': env_cost -> list, 'fes': env_fes ->float, 'return': R -> float, 'metadata': env_metadata -> dict}
        env_metadata: {'X': meta_X -> list, 'Cost': meta_Cost -> list(list)}
        '''
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results
    
    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms,
                        loss,
                        Return, Reward,
                        predict_Q, target_Q,
                        extra_info = {}):
        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # learning rate
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'learnrate/{network_name}', self.optimizer.param_groups[id]['lr'], mini_step)
        #
        # # grad and clipped grad
        grad_norms, grad_norms_clipped = grad_norms
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'grad/{network_name}', grad_norms[id], mini_step)
            tb_logger.add_scalar(f'grad_clipped/{network_name}', grad_norms_clipped[id], mini_step)

        # loss
        tb_logger.add_scalar('loss', loss.item(), mini_step)

        # Q
        for i in range(self.n_agent):
            tb_logger.add_scalar(f"Q/action_{i}", predict_Q[:, i].mean().item(), mini_step)
            tb_logger.add_scalar(f"Q/action_{i}_target", target_Q[:, i].mean().item(), mini_step)

        # train metric
        avg_reward = torch.stack(Reward).mean().item()
        max_reward = torch.stack(Reward).max().item()
        tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)
        tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
        tb_logger.add_scalar('train/max_reward', max_reward, mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)
