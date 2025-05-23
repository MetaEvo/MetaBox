import copy
import os
import torch
from torch import nn, optim
from ...rl.basic_agent import Basic_Agent
from ...environment.parallelenv.parallelenv import ParallelEnv
from typing import Optional, Union, Literal, List
import numpy as np
from ...rl.utils import clip_grad_norms, save_class
import math


class AttnWithFit(nn.Module):
    def __init__(self, popSize = 100, hiddenDim = 100):
        super().__init__()
        self.popSize = popSize
        self.attn = nn.Parameter(torch.randn((1, self.popSize, self.popSize)), requires_grad = True)
        self.q = nn.Sequential(
            nn.Linear(1, hiddenDim),
        )
        self.k = nn.Sequential(
            nn.Linear(1, hiddenDim),
        )
        self.num_heads = 1
        self.F = nn.Parameter(torch.randn((2,)), requires_grad = True)

    def forward(self, x, fitx):
        B, N, C = fitx.shape
        q = self.q(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B，H，N，SEQ
        k = self.k(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        fitattn = q @ k.transpose(2, 3) * (x.shape[-1] ** -0.5)  # b,h,n,n   (a11,a12,aij,...,ann)
        fitattn = torch.squeeze(fitattn.softmax(dim = -1), dim = 1)
        y1 = self.attn.softmax(dim = -1) @ x
        y2 = fitattn @ x
        y = y1 * self.F.softmax(-1)[0] + y2 * self.F.softmax(-1)[1]
        return y

    def getStrategy(self, fitx, dim):
        B, N, C = fitx.shape
        q = self.q(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  # B，H，N，SEQ
        k = self.k(fitx).view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        fitattn = q @ k.transpose(2, 3) * (dim ** -0.5)
        fitattn = torch.squeeze(fitattn.softmax(dim = -1), dim = 1)
        return self.F.softmax(-1)[0] * self.attn.softmax(dim = -1) + self.F.softmax(-1)[1] * fitattn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def sortpop(self, x, fitness):
        '''
        说明：
        输入：x(n,dim),f(x)(n)
        输出：排序后的x和f(x)
        '''
        fitness, fitindex = torch.sort(fitness, dim = -1)
        return x[fitindex], fitness


class OB(BaseModel):
    def __init__(self, dim = 64, hidden_dim = 100, popSize = 10, temid = 0):
        super().__init__()
        self.dim = dim
        self.trm = AttnWithFit(popSize = popSize, hiddenDim = hidden_dim)
        self.mut = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.id = temid
        self.vis = False
        self.f1 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad = True)
        self.f2 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad = True)
        self.f3 = nn.Parameter(torch.randn((1, popSize, 1)), requires_grad = True)

    def forward(self, x, xfit):
        b, n, d = x.shape
        fitx = xfit.softmax(dim = -1)
        fitx = fitx.view(b, n, 1)
        crosspop = self.trm(x, fitx)  ##A & AF
        offpop = self.mut(crosspop)  ##NN   MUT
        off = self.f1 * x + self.f2 * crosspop + self.f3 * offpop

        return off
        # childfit = func(off)
        # nextpop = self.sm(x, off, fatherfit, childfit)
        # return nextpop


class Policy(BaseModel):
    def __init__(self, popSize = 10, dim = 64, hidden_dim = 100, ems = 10, ws = False):
        super().__init__()
        self.ems = ems
        self.ws = ws
        if self.ws:
            self.ob = OB(dim, hidden_dim, popSize)
        else:
            self.ob = torch.nn.ModuleList([OB(dim, hidden_dim, popSize, i) for i in range(ems)])

    def forward(self, x, cost, pointer):
        # x [BS, NP, DIM]
        # cost [BS, NP]
        if self.ws:
            new_x = self.ob(x, cost)
        else:
            ob = self.ob[pointer]
            new_x = ob(x, cost)
        return new_x


    # def forward(self, x, func):
    #     self.trail = None
    #     self.evalnum = []
    #
    #     if self.ws is True:
    #         for i in range(self.ems):
    #             fatherfit = func(x)  # b,n
    #             x, fatherfit = self.sortpop(x, fatherfit)
    #             trail = torch.min(fatherfit, dim = -1)[0].view(-1, 1)
    #             if self.trail is None:
    #                 self.trail = trail
    #                 self.evalnum.append(x.shape[1])
    #             else:
    #                 self.trail = torch.cat((self.trail, trail), dim = -1)
    #             x = self.ob(x, func, fatherfit)
    #             self.evalnum.append(self.evalnum[-1] + x.shape[1])
    #     else:
    #         for ob in (self.ob):
    #             fatherfit = func(x)  # b,n  # todo 一开始X0
    #             x, fatherfit = self.sortpop(x, fatherfit)
    #             trail = torch.min(fatherfit, dim = -1)[0].view(-1, 1)
    #             if self.trail is None:
    #                 self.trail = trail
    #                 self.evalnum.append(x.shape[1])
    #             else:
    #                 self.trail = torch.cat((self.trail, trail), dim = -1)
    #             x = ob(x, func, fatherfit)
    #             self.evalnum.append(self.evalnum[-1] + x.shape[1])
    #
    #     return x, self.trail, self.evalnum


class B2OPT(Basic_Agent):
    """
    # Introduction
    B2Opt: Learning to Optimize Black-box Optimization with Little Budget.
    # Original paper
    "[**B2Opt: Learning to Optimize Black-box Optimization with Little Budget**](https://arxiv.org/abs/2304.11787)". arXiv preprint arXiv:2304.11787, (2023).
    # Official Implementation
    [B2Opt](https://github.com/ninja-wm/B2Opt)
    # Raises:
    - None explicitly, but underlying methods may raise exceptions related to environment interaction, tensor operations, or file I/O.
    """
    
    def __init__(self, config):
        """
        # Args:
        - config (object): Configuration object containing hyperparameters and settings for the agent, such as optimizer type, learning rate, device, save directory, and environment dimensions.
        # Built-in Attributes:
        - Opt: The policy network.
        - optimizer: The optimizer instance (Adam).
        - scheduler: Learning rate scheduler.
        - learning_time (int): Number of training steps completed.
        - cur_checkpoint (int): Current checkpoint index.
        - lr: learning rate is set as 1e-2 in B2Opt.
        - lr_step_size: learning rate decay periord, 100 steps as default.
        - lr_decay: the decay rate of lr is set as 0.9.
        """
        self.config = config

        self.config.optimizer = 'Adam'
        self.config.lr = 1e-2
        self.config.lr_step_size = 100
        self.config.lr_decay = 0.9

        self.config.lamda = 0.005
        self.config.max_grad_norm = 10

        self.config.hidden_dim = 100
        self.config.ws = False

        self.Opt = None
        self.optimizer = None
        self.scheduler = None

        self.learning_time = 0
        self.cur_checkpoint = 0

        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config)

    def __str__(self):
        return 'B2OPT'

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
        """
        Trains the agent for one episode across parallel environments.
        - envs: List of environments.
        - seeds (Optional[int, List[int], np.ndarray]): Random seeds for reproducibility.
        - para_mode (str): Parallelization mode ('dummy', 'subproc', 'ray', 'ray-subproc').
        - compute_resource (dict): Resource allocation for CPUs/GPUs.
        - tb_logger: TensorBoard logger for training metrics.
        - required_info (dict): Additional environment attributes to log.
        - Returns: (is_train_ended (bool), return_info (dict))
        """

        num_cpus = None
        num_gpus = 0 if self.config.device == 'cpu' else torch.cuda.device_count()
        if 'num_cpus' in compute_resource.keys():
            num_cpus = compute_resource['num_cpus']
        if 'num_gpus' in compute_resource.keys():
            num_gpus = compute_resource['num_gpus']
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)
        env.seed(seeds)

        if self.Opt is None:
            # beggining
            ps_list = env.get_env_attr('NP')
            ems_list = env.get_env_attr('ems')
            NP = ps_list[0]
            ems = ems_list[0]
            self.Opt = Policy(popSize = NP,
                              dim = self.config.dim,
                              hidden_dim = self.config.hidden_dim,
                              ems = ems,
                              ws = self.config.ws).to(self.config.device)
            self.optimizer = optim.Adam(self.Opt.parameters(), lr = self.config.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = self.config.lr_step_size, gamma = self.config.lr_decay)
            save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
            self.cur_checkpoint += 1

        state = env.reset()
        memory = []
        # state 2D [BS, NP] Y

        _loss = []
        _R = torch.zeros(len(env))
        t = 0
        memory.append(state)
        while not env.all_done():
            action = [self.Opt for _ in range(len(env))]
            next_state, rewards, done, info = env.step(action)

            _R += rewards
            state = next_state
            memory.append(state)
            t += 1

        # memory.append(state) # 3D [T, BS, NP]
        memory_tensor = torch.stack(memory, dim = 0)
        init_ = torch.mean(memory_tensor[0, :, :], dim = 1).detach()
        loss = (init_ - torch.mean(memory_tensor[1:, :, :], dim = (0, 2))) / init_ # bs

        loss = -torch.mean(loss)
        _loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.learning_time += 1

        if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
            save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
            self.cur_checkpoint += 1

        if not self.config.no_tb:
            self.log_to_tb_train(tb_logger, self.learning_time,
                                 grad_norms,
                                 loss,
                                 _R,
                                 )

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time}
        env_cost = env.get_env_attr('cost')
        return_info['normalizer'] = env_cost[0]
        return_info['gbest'] = env_cost[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()
        return is_train_ended, return_info

    def rollout_episode(self,
                        env,
                        seed = None,
                        required_info = {}):
        """
        Evaluates the agent in a single/multiple environment without training.
        - env: Environment instance.
        - seed (Optional[int]): Random seed.
        - required_info (dict): Additional environment attributes to log.
        - Returns: results (dict) with evaluation metrics.
        """
        with torch.no_grad():
            env.seed(seed)
            is_done = False
            env.reset()
            R = 0
            while not is_done:
                action = copy.deepcopy(self.Opt)
                _, reward, is_done, info = env.step(action)
                R += reward.item()
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

    def log_to_tb_train(self, tb_logger, mini_step,
                        grad_norms,
                        loss,
                        Return,
                        extra_info = {}):
        """
        Logs training metrics to TensorBoard.
        - tb_logger: TensorBoard logger.
        - mini_step (int): Current training step.
        - grad_norms (tuple): Gradient norms before and after clipping.
        - loss (torch.Tensor): Training loss.
        - Return (torch.Tensor): Episode returns.
        - extra_info (dict): Additional metrics to log.
        """
        # learning rate
        tb_logger.add_scalar('learnrate/OPT', self.optimizer.param_groups[0]['lr'], mini_step)

        # grad and clipped grad
        grad_norms, grad_norms_clipped = grad_norms
        tb_logger.add_scalar('grad/OPT', grad_norms[0], mini_step)
        tb_logger.add_scalar('grad_clipped/OPT', grad_norms_clipped[0], mini_step)

        # loss
        tb_logger.add_scalar('loss/Loss', loss.item(), mini_step)

        # train metric

        tb_logger.add_scalar('train/episode_avg_return', Return.mean().item(), mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)
