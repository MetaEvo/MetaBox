import copy
import os
import torch
from torch import nn
from ...rl.basic_agent import Basic_Agent
from ...environment.parallelenv.parallelenv import ParallelEnv
from typing import Optional, Union, Literal, List
import numpy as np
from ...rl.utils import clip_grad_norms, save_class
import math


class SMBND(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, batchpop1, batchpop2, minimize = True):
        '''
        实现选择操作,默认是最小化函数，若minimize=False,则为最大化目标值问题
        batchpop1 是offpop
        '''

        b, n, d = batchpop1.shape
        fit1 = batchpop1[..., 0]  #
        fit2 = batchpop2[..., 0]
        batchMask = fit1 - fit2  # b,n
        if minimize:
            batchMask[batchMask >= 0] = 0
            batchMask[batchMask < 0] = 1

        else:
            batchMask[batchMask <= 0] = 0
            batchMask[batchMask > 0] = 1

        # print('\n选择了',torch.sum(batchMask).item(),'/',b*n,'\nbatchmask:\n',batchMask)
        batchMask = torch.unsqueeze(batchMask, -1)  # b,n,1
        batchMask = batchMask.repeat(1, 1, d)
        batchMask1 = torch.ones_like(batchMask).to(self.device) - batchMask
        nextPop = batchpop1 * batchMask + batchpop2 * batchMask1

        return nextPop


class GBMutModel(nn.Module):
    def __init__(self, device, hdim = 1000):
        super().__init__()
        fdim = 2
        hdim2 = 100
        qkdim = hdim
        self.fq1 = nn.Sequential(
            nn.Linear(hdim2, qkdim),
            nn.Tanh(),
        )

        self.fk1 = nn.Sequential(
            nn.Linear(hdim2, qkdim),
            nn.Tanh(),
        )

        self.w = nn.Sequential(
            nn.Linear(fdim, hdim2),
            nn.Tanh(),
            # nn.LayerNorm(hdim2)
        )
        self.device = device

    def forward(self, x):
        # b,n,2
        x = self.w(x)
        q1 = self.fq1(x)
        k1 = self.fk1(x)
        A = torch.matmul(q1, torch.transpose(k1, -1, -2)) / torch.sqrt(torch.tensor(k1.shape[-1]).to(self.device))
        A = torch.tanh(A)
        mask = torch.rand_like(A).to(self.device)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        y = torch.eye(mask.shape[-2], mask.shape[-1]).to(self.device)
        y = torch.unsqueeze(y, 0)
        mask = mask + y
        mask[mask == 2] = 1
        A = A * mask
        return A


class GBLearnCrRate(nn.Module):
    def __init__(self, hdim = 100):
        super().__init__()
        inputdim = 3
        outputdim = 1
        self.net = nn.Sequential(
            nn.Linear(inputdim, hdim),
            nn.ReLU(),
            nn.LayerNorm(hdim),
            nn.Linear(hdim, outputdim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)  # b,n,dim
        return x


class Policy(nn.Module):
    def __init__(self, popsize = 100, selmod = '1-to-1', cr_policy = 'learned', muthdim = 1000, crhdim = 4, device = "cpu"):
        super().__init__()
        self.popsize = popsize
        self.cr_policy = cr_policy
        self.ranks = (torch.arange(0, popsize, requires_grad = False).to(device)).float().to(device)
        self.ranks = (self.ranks - torch.mean(self.ranks, dim = -1, keepdim = True)) / torch.std(self.ranks, dim = -1, keepdim = True)
        self.ranks = self.ranks.view(1, -1, 1)
        self.ranks2 = (torch.arange(0, popsize * 2 - 1, requires_grad = False).to(device)).float().to(device)
        self.ranks2 = (self.ranks2 - torch.mean(self.ranks2, dim = -1, keepdim = True)) / torch.std(self.ranks2, dim = -1, keepdim = True)
        self.ranks2 = self.ranks2.view(1, -1, 1)
        self.sm = SMBND(device)
        self.agen = GBMutModel(device, hdim = muthdim)
        self.crgen = GBLearnCrRate(hdim = crhdim)
        self.selmod = selmod
        self.adapter = None
        self.device = device

    def setAdapter(self, adapter):
        self.adapter = adapter

    def RoulleteSelectWithElite(self, pop, popsize):
        '''
        保留精英的轮盘赌选择
        pop -> b,n,d+1
        '''
        b, n, d = pop.shape
        fitness = pop[..., 0]  # b,n
        p = 1 - torch.softmax(fitness, dim = -1)
        selected_index = torch.multinomial(p, popsize, replacement = False)
        offs = []
        for idx, batchpop in enumerate(pop):
            index = selected_index[idx]
            tmp = batchpop[index]
            tmp = torch.unsqueeze(tmp, 0)
            offs.append(tmp)
        offs = torch.cat(offs)
        return offs

    def genGBMutToken(self, x, ranks):
        b, n = x.shape  # b,n
        # x=self.fln(x).view(b,-1,1)
        miu = torch.mean(x, dim = -1, keepdim = True)
        std = torch.std(x, dim = -1, keepdim = True)
        x = (x - miu) / (std + 1e-20)
        x = x.view(b, -1, 1)
        ranks = ranks.repeat(b, 1, 1)
        x = torch.cat((x, ranks[..., :n, :]), dim = -1)  # (b,w*h,2)
        return x

    def genGBMutToken2(self, x, ranks):
        b, _ = x.shape  # b,n
        x = self.fln2(x).view(b, -1, 1)
        ranks = ranks.repeat(b, 1, 1)
        x = torch.cat((x, ranks), dim = -1)  # (b,w*h,2)
        return x

    def genCrRankToken(self, fitness):
        '''
        input:b,n
        '''
        b, n = fitness.shape
        _, indexs = torch.sort(fitness, dim = -1)  # b,n
        ranks = self.ranks.repeat(b, 1, 1)
        # fitness=self.fln(fitness)
        miu = torch.mean(fitness, dim = -1, keepdim = True)
        std = torch.std(fitness, dim = -1, keepdim = True)
        fitness = (fitness - miu) / std
        fitness = fitness.view(b, -1, 1)
        newRanks = []
        for bid in range(b):
            index = indexs[bid, ...]
            index = torch.unsqueeze(index, -1)
            rank = ranks[bid, ...]
            tmp = torch.cat((index, rank[:n, ...]), dim = -1)
            _, tmp_index = torch.sort(tmp[:, 0], dim = 0)
            tmp = torch.index_select(tmp, 0, tmp_index)
            token = torch.cat((fitness[bid], torch.unsqueeze(tmp[:, 1], -1)), -1)
            newRanks.append(token)
        token = torch.stack(newRanks, 0)
        return token

    def genCrRankTokenWithoutFit(self, father, off):
        '''
        input:b,n,改为不用适应度的版本
        '''
        b, n, d = father.shape
        # 计算father和offer之间的余弦相似度
        tmp = torch.cat((father, off), dim = -1)  # b,n,2d
        ave = torch.mean(tmp, dim = -1, keepdim = True)  # b,n,1
        std = torch.std(tmp, dim = -1, keepdim = True)  # b,n,1
        tmp = (tmp - ave) / (std + 1e-8)
        fpop = tmp[:, :, :d]
        opop = tmp[:, :, d:]
        fMod = torch.sqrt(torch.sum(fpop ** 2, dim = -1, keepdim = True))  # b,n,1
        oMod = torch.sqrt(torch.sum(opop ** 2, dim = -1, keepdim = True))  # b,n,1
        item = fMod * oMod
        item = torch.clamp(item, min = 1e-8)
        sim = torch.sum(fpop * opop, dim = -1, keepdim = True) / (item)  # b,n,1
        # sim=torch.sum(father*off,dim=-1,keepdim=True)/(item) #b,n,1
        sim = sim.view(b, n)
        # sim=torch.softmax(sim,dim=-1)
        sim = (sim - torch.mean(sim, dim = -1, keepdim = True)) / torch.std(sim, dim = -1, keepdim = True)
        sim = torch.unsqueeze(sim, -1)

        return sim

    def clearMutstate(self):
        self.gbmut.resetSigma()
        self.improvedFlag = None

    def forward(self, batchPop = None):
        '''
        输入：
        已经有适应度的种群（batch,n,d）
        '''
        paramcr = None
        b, n, d = batchPop.shape
        batchPop = sortIndivBND(batchPop)
        batchfitness = batchPop[:, :, 0]
        fitnesstoken = self.genGBMutToken(batchfitness, self.ranks)  # b,n,2
        Atoken = fitnesstoken
        params = self.agen(Atoken)
        batchChrom = batchPop[:, :, 1:]
        # params=torch.zeros((1,n,n),device=DEVICE)
        # for i in range(n):
        #     idx=np.random.choice(n,3,False)
        #     params[0,i,idx[0]]=1
        #     params[0,i,idx[1]]=-0.5
        #     params[0,i,idx[2]]=0.5
        vchrom = torch.matmul(params, batchChrom)
        # 交叉
        if self.cr_policy == 'learned':
            # vpop,r1=problem.calfitness(vchrom)
            # r1=torch.squeeze(r1,-1)
            # vpopToken=self.genCrRankToken(r1)
            vpopToken = self.genCrRankTokenWithoutFit(batchChrom, vchrom)
            token = torch.cat((fitnesstoken, vpopToken), dim = -1)  # b,n,4
            cr = self.crgen(token)  # b,n,1
            cr = torch.unsqueeze(cr, -2)  # b,n,1,1
            paramcr = cr.view(b, n, )
            cr = cr.repeat(1, 1, batchChrom.shape[-1], 1)  # b,n,d,1
            r = torch.rand_like(cr).to(self.device)
            select_mask = torch.cat((cr, r), dim = -1)  # b,n,d,2
            select_mask = torch.nn.functional.gumbel_softmax(select_mask, tau = 1, hard = True, eps = 1e-10, dim = - 1)
            offpopChrom = select_mask[..., 0] * batchChrom + select_mask[..., 1] * vchrom

        else:
            mask = torch.rand_like(batchChrom).to(self.device)
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            offpopChrom = mask * batchChrom + (1 - mask) * vchrom

        if self.adapter:
            offpopChrom = self.adapter(offpopChrom)

        return offpopChrom

        # _, offpop = func(offpopChrom[0])  # b,n,d
        # offpop = offpop[None, :]
        # # 选择
        # mixpop = torch.cat((batchPop, offpop), dim = 1)
        # if self.selmod == 'learned':
        #     pass
        #
        # if self.selmod == '1-to-1':
        #     nextPop = self.sm(offpop, batchPop)
        #
        # if self.selmod == '轮盘赌':
        #     mixpop = sortIndivBND(mixpop)
        #     elitePop = mixpop[:, :1, :]
        #     mixpop = mixpop[:, 1:, :]
        #     selectednextPop = self.RoulleteSelectWithElite(mixpop, batchPop.shape[1] - 1)
        #     nextPop = torch.cat((elitePop, selectednextPop), dim = 1)
        #
        # return nextPop, params, paramcr


class GLHF(Basic_Agent):
    """
    # Introduction
    GLHF: General Learned Evolutionary Algorithm Via Hyper Functions
    # Original paper
    "[**GLHF: General Learned Evolutionary Algorithm Via Hyper Functions**](https://arxiv.org/abs/2405.03728)." arXiv preprint arXiv:2405.03728 (2024).
    # Official Implementation
    [GLHF](https://github.com/ninja-wm/POM/)
    # Args:
    - config (object): Configuration object containing hyperparameters and settings for the agent, such as optimizer type, learning rate, device, and save directories.
    # Attributes:
    - Pom (Policy): The policy model used by the agent.
    - optimizer (torch.optim.Optimizer): The optimizer for training the policy.
    - learning_time (int): Counter for the number of training steps taken.
    - cur_checkpoint (int): Counter for the current checkpoint index.
    - config (object): The configuration object with agent settings.
    # Methods:
    - __str__(): Returns the string representation of the agent.
    - train_episode(...): Trains the agent for one episode in parallel environments.
    - rollout_episode(...): Evaluates the agent in a single environment without training.
    - log_to_tb_train(...): Logs training metrics to TensorBoard.
    # train_episode
    Trains the agent for one episode using parallel environments. Handles environment setup, policy optimization, checkpointing, and logging.
    ## Args:
    - envs: List of environments to train on.
    - seeds (Optional[Union[int, List[int], np.ndarray]]): Seeds for environment reproducibility.
    - para_mode (str): Parallelization mode ('dummy', 'subproc', 'ray', 'ray-subproc').
    - compute_resource (dict): Resources for parallelization (e.g., number of CPUs/GPUs).
    - tb_logger: TensorBoard logger for recording metrics.
    - required_info (dict): Additional environment attributes to record.
    ## Returns:
    - is_train_ended (bool): Whether the training has reached the maximum step.
    - return_info (dict): Dictionary containing returns, losses, learning steps, and additional info.
    # rollout_episode
    Evaluates the agent in a single environment without updating the policy.
    ## Args:
    - env: The environment to evaluate in.
    - seed: Seed for reproducibility.
    - required_info (dict): Additional environment attributes to record.
    ## Returns:
    - results (dict): Dictionary containing cost, function evaluations, return, and optional metadata.
    # log_to_tb_train
    Logs training statistics and metrics to TensorBoard.
    ## Args:
    - tb_logger: TensorBoard logger.
    - mini_step (int): Current training step.
    - grad_norms: Gradient norms before and after clipping.
    - loss_1, loss_2, loss: Loss components.
    - Return: Episode returns.
    - reward: Rewards for the current step.
    - extra_info (dict): Additional metrics to log.
    # Raises:
    - ValueError: If invalid configuration or environment state is encountered during training.
    """
    
    def __init__(self, config):
        self.config = config

        self.config.optimizer = 'Adam'
        self.config.lr = 1e-4

        self.config.lamda = 0.005
        self.config.max_grad_norm = math.inf

        self.config.muthdim = 1000
        self.config.crhdim = 4
        self.config.selmod = '1-to-1'
        self.config.cr_policy = 'learned'

        self.Pom = None
        self.optimizer = None

        self.learning_time = 0
        self.cur_checkpoint = 0
        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config)

    def __str__(self):
        return 'GLHF'

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

        if self.Pom is None:
            # beginning
            ps_list = env.get_env_attr('NP')
            NP = ps_list[0]
            self.Pom = Policy(popsize = NP,
                              selmod = self.config.selmod,
                              cr_policy = self.config.cr_policy,
                              muthdim = self.config.muthdim,
                              crhdim = self.config.crhdim,
                              device = self.config.device).to(self.config.device)
            self.optimizer = torch.optim.Adam(self.Pom.parameters(), lr = self.config.lr)

            save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
            self.cur_checkpoint += 1

        state = env.reset()  # 给 set 初始化

        # state 3D [BS, NP, 1 + DIM] [:, :, 0] ---> Y
        lamda = self.config.lamda

        _loss = []
        TS = state.shape[0]

        _R = torch.zeros(TS)
        t = 0
        while not env.all_done():
            state = state.detach()

            action = [self.Pom for _ in range(TS)]
            next_state, rewards, is_end, info = env.step(action)

            _R += rewards

            loss_1 = (torch.mean(next_state[:, :, 0], dim = 1) - torch.mean(state[:, :, 0], dim = 1)) / (torch.mean(state[:, :, 0], dim = 1) + 1e-20)  # bs

            if torch.isnan(loss_1).any() or torch.isinf(loss_1).any():
                break

            loss_2 = torch.mean((torch.std(next_state[:, :, 1:], dim = 1)), dim = 1)  # bs

            loss = loss_1 - lamda * loss_2  # bs
            loss = torch.mean(loss)

            _loss.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()

            grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

            self.optimizer.step()
            self.learning_time += 1

            t += 1

            state = next_state.clone().detach()

            if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if not self.config.no_tb:
                self.log_to_tb_train(tb_logger, self.learning_time,
                                     grad_norms,
                                     loss_1, loss_2, loss,
                                     _R, rewards,
                                     )

            if self.learning_time >= self.config.max_learning_step:
                break

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
        # Don't need state
        with torch.no_grad():
            env.seed(seed)
            is_done = False
            env.reset()
            R = 0
            while not is_done:
                action = copy.deepcopy(self.Pom)
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
                        loss_1, loss_2, loss,
                        Return, reward,
                        extra_info = {}):
        # learning rate
        tb_logger.add_scalar('learnrate/POM', self.optimizer.param_groups[0]['lr'], mini_step)

        # grad and clipped grad
        grad_norms, grad_norms_clipped = grad_norms
        tb_logger.add_scalar('grad/POM', grad_norms[0], mini_step)
        tb_logger.add_scalar('grad_clipped/POM', grad_norms_clipped[0], mini_step)

        # loss
        tb_logger.add_scalar('loss/loss_1_avg', loss_1.mean().item(), mini_step)
        tb_logger.add_scalar('loss/loss_2_avg', loss_2.mean().item(), mini_step)
        tb_logger.add_scalar('loss/Loss', loss.item(), mini_step)

        # train metric
        avg_reward = reward.mean().item()
        max_reward = reward.max().item()

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


def sortIndiv(batchPop):
    '''
    作用：
    将一批种群中的个体按照 fitness维度的值来排序号

    输入：
    batchPop:一批种群，维度为（batchSize,dim+1,L*L）
    返回:
    排好序的（batch,dim+1,w,h的矩阵）
    '''
    b, d, w, h = batchPop.shape
    fitness = batchPop[:, 0, :, :]
    fitness = fitness.view(b, w * h)
    _, fit = torch.sort(fitness, dim = 1)  # b,n
    batchPop = batchPop.view(b, d, -1).permute(0, 2, 1)  # b,n,dim
    y = torch.zeros_like(batchPop)
    for index, pop in enumerate(batchPop):
        pop = batchPop[index]  # n,dim
        y[index] = torch.index_select(pop, 0, fit[index])
    y = y.permute(0, 2, 1).view(b, d, w, h)
    batchPop = y
    return batchPop


def sortIndivBND(batchPop):
    b, n, d = batchPop.shape
    fitness = batchPop[..., 0]
    _, fit = torch.sort(fitness, dim = 1)  # b,n
    y = torch.zeros_like(batchPop)
    for index, pop in enumerate(batchPop):
        pop = batchPop[index]  # n,dim
        y[index] = torch.index_select(pop, 0, fit[index])
    batchPop = y
    return batchPop









