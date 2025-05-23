from ...rl.ppo import *
from .networks import MLP
from torch import nn
import numpy as np
from typing import List
from sympy import lambdify
import math
from torch.distributions import Categorical

class Critic(nn.Module):
    def __init__(self, fea_dim, value_dim) -> None:
        super().__init__()
        self.input_dim = fea_dim
        self.output_dim = value_dim

        self.value_net = nn.Linear(self.input_dim, self.output_dim)

    # return baseline value detach & baseling value
    def forward(self, x):
        baseline_val = self.value_net(x)

        return baseline_val.detach().squeeze(), baseline_val.squeeze()

# memory for recording transition during training process
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.gap_rewards = []
        self.b_rewards = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.gap_rewards[:]
        del self.b_rewards[:]

class SYMBOL(PPO_Agent):
    """
    # Introduction
    The paper "SYMBOL: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning" introduces a novel framework, SYMBOL, designed to automate the discovery of advanced black-box optimizers using symbolic equation learning. Unlike traditional Meta-Black-Box Optimization (MetaBBO) methods that rely on predefined, hand-crafted optimizers, SYMBOL employs a Symbolic Equation Generator (SEG) to dynamically produce closed-form optimization rules tailored to specific tasks and optimization steps. The framework incorporates reinforcement learning-based strategies to efficiently meta-learn these symbolic rules. 
    # Original Paper
    "[**Symbol: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning**](https://openreview.net/forum?id=vLJcd43U7a)." The Twelfth International Conference on Learning Representations. (2024)
    # Official Implementation
    [SYMBOL](https://github.com/MetaEvo/Symbol)
    # Application Scenario
    single-object optimization problems(SOOP)
    # Args:
        `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    # Attributes:
        config (object): Configuration object with various hyperparameters and settings.
        tokenizer (MyTokenizer): Tokenizer instance used for processing sequences.
        actor (LSTM): Actor network for generating actions.
        critic (Critic): Critic network for evaluating states.
        optimizer (torch.optim.Optimizer): Optimizer for training the actor and critic networks.
        learning_time (int): Counter for the number of learning steps performed.
        cur_checkpoint (int): Counter for the current checkpoint during training.
    # Methods:
        __str__():
            Returns the string representation of the class.
        train_episode(envs, seeds, para_mode='dummy', compute_resource={}, tb_logger=None, required_info={}):
            Trains the agent for one episode using the provided environments.
            Args:
                envs (list): List of environments for training.
                seeds (Optional[Union[int, List[int], np.ndarray]]): Seeds for environment initialization.
                para_mode (Literal['dummy', 'subproc', 'ray', 'ray-subproc']): Parallelization mode.
                compute_resource (dict): Resources for computation (e.g., CPUs, GPUs).
                tb_logger (object): TensorBoard logger for logging training metrics.
                required_info (dict): Additional information required from the environment.
            Returns:
                Tuple[bool, dict]: A tuple containing a boolean indicating if training has ended and a dictionary with training information.
        rollout_episode(env, seed=None, required_info={}):
            Executes a single rollout episode in the given environment.
            Args:
                env (object): Environment for the rollout.
                seed (Optional[int]): Seed for environment initialization.
                required_info (dict): Additional information required from the environment.
            Returns:
                dict: A dictionary containing rollout results.
    # Returns:
        None
    # Raises:
        AssertionError: If a NaN value is found in the loss during training.
    """
    def __init__(self, config):
        self.config = config

        self.config.optimizer = 'Adam'
        self.config.init_pop = 'random'
        self.config.teacher = 'MadDE'
        self.config.population_size = 100
        self.config.boarder_method = 'clipping'
        self.config.skip_step = 5
        self.config.test_skip_step = 5
        self.config.max_c = 1.
        self.config.min_c = -1.
        self.config.c_interval = 0.4
        self.config.max_layer = 6
        self.config.value_dim = 1
        self.config.hidden_dim = 16
        self.config.num_layers = 1
        self.config.lr = 1e-3
        self.config.lr_critic = 1e-3
        self.config.max_grad_norm = math.inf

        self.config.encoder_head_num = 4
        self.config.decoder_head_num = 4
        self.config.critic_head_num = 4
        self.config.embedding_dim = 16
        self.config.n_encode_layers = 1
        self.config.normalization = 'layer'
        self.config.hidden_dim1_critic = 32
        self.config.hidden_dim2_critic = 16
        self.config.hidden_dim1_actor = 32
        self.config.hidden_dim2_actor = 8
        self.config.output_dim_actor = 1
        # self.config.lr_decay = 0.9862327
        self.config.gamma = 0.99
        self.config.K_epochs = 3
        self.config.eps_clip = 0.1
        self.config.n_step = 10

        self.config.fea_dim = 9

        self.tokenizer = MyTokenizer()

        actor = LSTM(max_layer = self.config.max_layer,
                     hidden_dim = self.config.hidden_dim,
                     num_layers = self.config.num_layers,
                     max_c = self.config.max_c,
                     min_c = self.config.min_c,
                     fea_dim = self.config.fea_dim,
                     c_interval = self.config.c_interval,
                     tokenizer = self.tokenizer)
        critic = Critic(fea_dim = self.config.fea_dim,
                        value_dim = self.config.value_dim)

        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config, {'actor': actor, 'critic': critic}, [self.config.lr, self.config.lr_critic])

    def __str__(self):
        return "SYMBOL"

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
        for env in envs:
            env.optimizer.is_train = True
        env = ParallelEnv(envs, para_mode, num_cpus=num_cpus, num_gpus=num_gpus)

        # set env.optimizer.is_train = True
        # set env.optimizer.is_train = True
        env.seed(seeds)
        memory = Memory()

        # params for training
        gamma = self.gamma
        n_step = self.n_step

        K_epochs = self.K_epochs
        eps_clip = self.eps_clip

        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device)
        except:
            pass

        t = 0
        # initial_cost = obj
        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step and not env.all_done():

                memory.states.append(state.clone())

                self.config.require_baseline = False
                seq, const_seq, log_prob, action_dict = self.actor(state, save_data = True)

                # critic network
                baseline_val_detached, baseline_val = self.critic(state)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # store reward for ppo
                memory.actions.append(action_dict)
                memory.logprobs.append(log_prob)

                action = []
                for s, cs in zip(seq, const_seq):
                    expr = construct_action(seq = s, const_seq = cs, tokenizer = self.tokenizer)
                    action.append({'expr': expr, 'skip_step': self.config.skip_step})

                # expr = construct_action(seq = seq, const_seq = const_seq, tokenizer = self.tokenizer)
                # action = {'expr': expr, 'skip_step': self.config.skip_step}
                state, reward, is_done, info = env.step(action)

                reward = torch.Tensor(reward).to(self.device)
                _R += reward.clone().detach().cpu()
                memory.rewards.append(reward)

                t = t + 1

                try:
                    state = torch.Tensor(state).to(self.device)
                except:
                    pass

            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update
            old_actions = memory.actions
            try:
                old_states = torch.stack(memory.states).detach()  # .view(t_time, bs, ps, dim_f)
            except:
                pass
            # old_actions = all_actions.view(t_time, bs, ps, -1)
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            # Optimize PPO policy for K mini-epochs:
            old_value = None
            for _k in range(K_epochs):
                if _k == 0:
                    logprobs = memory.logprobs

                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    entropy = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):
                        # get new action_prob
                        log_p = self.actor(old_states[tt],fix_action = old_actions[tt])

                        logprobs.append(log_p)

                        baseline_val_detached, baseline_val = self.critic(old_states[tt])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)
                logprobs = torch.stack(logprobs).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(state)[0]
                critic_output = R.clone()
                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()

                # check K-L divergence (for logging only)
                approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
                # calculate loss
                loss = baseline_loss + reinforce_loss

                if torch.isnan(loss):
                    print(f'baseline_loss:{baseline_loss}')
                    print(f'reinforce_loss:{reinforce_loss}')
                    assert True, 'nan found in loss!!'

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()
                _loss.append(loss.item())

                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                # grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                    save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if not self.config.no_tb:
                    self.log_to_tb_train(tb_logger, self.learning_time,
                                         grad_norms,
                                         reinforce_loss, baseline_loss,
                                         _R, Reward, memory.rewards,
                                         critic_output, logprobs, torch.Tensor([0.]), approx_kl_divergence)

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['normalizer'] = env_cost[0]
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()

        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
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
        with torch.no_grad():
            if seed is not None:
                env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.Tensor(state).to(self.device)
                except:
                    state = [state]
                seq, const_seq, log_prob = self.actor(state[None, :], save_data = False)
                action = []
                for s, cs in zip(seq, const_seq):
                    expr = construct_action(seq = s, const_seq = cs, tokenizer = self.tokenizer)
                    action.append({'expr': expr, 'skip_step': self.config.skip_step})

                state, reward, is_done, info = env.step(action[0])
                R += reward
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

class LSTM(nn.Module):
    def __init__(self, max_layer, hidden_dim, num_layers, max_c, min_c, fea_dim, c_interval, tokenizer) -> None:
        super().__init__()

        self.max_layer = max_layer
        self.output_size = tokenizer.vocab_size
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.max_c = max_c
        self.min_c = min_c
        self.fea_size = fea_dim
        self.tokenizer = tokenizer
        self.interval = c_interval
        self.binary_code_len = 4
        self.lstm = nn.LSTM(int(2 ** self.max_layer - 1) * self.binary_code_len, self.hidden_size, self.num_layers, batch_first = True)
        self.output_net = nn.Linear(self.hidden_size, self.output_size)

        self.x_to_c = nn.Linear(self.fea_size, self.hidden_size)
        self.constval_net = nn.Linear(self.hidden_size, int((self.max_c - self.min_c) // self.interval))
        self.num_c = int((self.max_c - self.min_c) // self.interval)

    def forward(self, x, save_data = False, fix_action = None):

        bs = x.shape[0]
        device = x.device

        log_prob_whole = torch.zeros(bs).to(device)
        pre_seq = []

        # initial input,h,c for lstm
        h_0 = torch.zeros((self.num_layers, bs, self.hidden_size)).to(device)

        c_0 = self.x_to_c(x)

        h = h_0
        c = c_0.unsqueeze(dim = 0).repeat(self.num_layers, 1, 1)

        if save_data:
            memory = LSTM_Memory()

        # generate seqence
        if not fix_action:
            len_seq = int(2 ** self.max_layer - 1)
            seq = (torch.ones((bs, len_seq), dtype = torch.long) * -1)
            const_vals = torch.zeros((bs, len_seq))

            x_in = torch.zeros((bs, 1, len_seq * self.binary_code_len)).to(device)

            # the generating position of the seq
            position = torch.zeros((bs,), dtype = torch.long)
            working_index = torch.arange(bs)
            # generate sequence
            while working_index.shape[0] > 0:
                # print(f'h.shape:{h.shape},c.shape:{c.shape}, x_in.shape:{x_in.shape}, working_index: {working_index}')
                output, (h, c) = self.lstm(x_in, (h, c))
                # output,(h,c)=self.lstm(x_in)
                out = self.output_net(output)

                # 如果position为全-1，则mask为全0
                mask = get_mask(seq[working_index], self.tokenizer, position, self.max_layer)

                # mask=get_mask(pre_seq,self.tokenizer,position)
                log_prob, choice, binary_code = get_choice(out, mask, self.binary_code_len)
                # prefix_seq.append(choice)

                # get c
                c_index = self.tokenizer.is_consts(choice)
                if np.any(c_index):
                    out_c = self.constval_net(output[c_index])
                    log_prob_c, c_val = get_c(out_c, self.min_c, self.interval)
                    log_prob_whole[working_index[c_index]] += log_prob_c
                    const_vals[working_index[c_index], position[c_index]] = c_val.cpu()

                # store if needed
                if save_data:
                    memory.c_index.append(c_index)
                    memory.position.append(position)
                    memory.working_index.append(working_index)
                    memory.mask.append(mask)
                    memory.x_in.append(x_in.clone().detach())

                # udpate
                # need to test!!!!
                x_in = x_in.clone().detach()
                binary_code = binary_code.to(device)
                for i in range(self.binary_code_len):
                    x_in[range(len(working_index)), 0, position * self.binary_code_len + i] = binary_code[:, i]

                log_prob_whole[working_index] += log_prob

                seq[working_index, position] = choice.cpu()

                position = get_next_position(seq[working_index], choice, position, self.tokenizer)

                # update working index when position is -1
                filter_index = (position != -1)
                working_index = working_index[filter_index]
                # print(f'filter_index: {filter_index}, working_index: {working_index.shape[0]}')
                position = position[filter_index]
                x_in = x_in[filter_index]
                h = h[:, filter_index]
                c = c[:, filter_index]
                if save_data:
                    memory.filter_index.append(filter_index)

            if not save_data:
                # 返回等长的序列，数组表示的二叉树
                return seq.numpy(), const_vals.numpy(), log_prob_whole
            else:
                memory.seq = seq
                memory.c_seq = const_vals

                return seq.numpy(), const_vals.numpy(), log_prob_whole, memory.get_dict()
        else:
            # fix_action get the new log_prob
            x_in = fix_action['x_in']  # x_in shape: (len, [bs,1,31*4])
            mask = fix_action['mask']  # mask shape: (len, [bs,vocab_size])
            working_index = fix_action['working_index']  # working_index
            # seq=torch.Tensor(fix_action['seq']).to(device)
            seq = fix_action['seq']
            c_seq = fix_action['c_seq']
            # c_seq=torch.Tensor(fix_action['c_seq']).to(device)
            position = fix_action['position']
            c_indexs = fix_action['c_index']
            filter_index = fix_action['filter_index']

            for i in range(len(x_in)):
                output, (h, c) = self.lstm(x_in[i], (h, c))
                # output,(h,c)=self.lstm(x_in[i])
                out = self.output_net(output)

                w_index = working_index[i]
                pos = position[i]
                log_prob = get_choice(out, mask[i], self.binary_code_len, fix_choice = seq[w_index, pos].to(device))
                log_prob_whole[w_index] += log_prob

                c_index = c_indexs[i]
                if np.any(c_index):
                    out_c = self.constval_net(output[c_index])
                    log_prob_c = get_c(out_c, self.min_c, self.interval, fix_c = c_seq[w_index[c_index], pos[c_index]])
                    log_prob_whole[w_index[c_index]] += log_prob_c

                # update h & c
                h = h[:, filter_index[i]]
                c = c[:, filter_index[i]]

            return log_prob_whole

    def get_random_seq(self, bs):
        len_seq = int(2 ** self.max_layer - 1)
        seq = (torch.ones((bs, len_seq), dtype = torch.long) * -1)
        const_vals = torch.zeros((bs, len_seq))
        position = torch.zeros((bs,), dtype = torch.long)

        working_index = torch.arange(bs)
        # generate sequence
        while working_index.shape[0] > 0:

            output = torch.rand((working_index.shape[0], 1, self.output_size))

            mask = get_mask(seq[working_index], self.tokenizer, position, self.max_layer)

            _, choice, _ = get_choice(output, mask, self.binary_code_len)

            c_index = self.tokenizer.is_consts(choice)

            if np.any(c_index):
                bs = output[c_index].shape[0]
                out_c = torch.rand((bs, 1, self.num_c))
                _, c_val = get_c(out_c, self.min_c, self.interval)
                const_vals[working_index[c_index], position[c_index]] = c_val

            seq[working_index, position] = choice

            position = get_next_position(seq[working_index], choice, position, self.tokenizer)

            # update working index when position is -1
            filter_index = (position != -1)
            working_index = working_index[filter_index]
            position = position[filter_index]

        return seq.numpy(), const_vals.numpy()


class LSTM_Memory():
    def __init__(self) -> None:
        self.position = []
        self.x_in = []
        self.mask = []
        self.working_index = []
        self.seq = None
        self.c_seq = None
        self.c_index = []
        self.filter_index = []

    def get_dict(self):
        return {
            'position': self.position,
            'x_in': self.x_in,
            'mask': self.mask,
            'working_index': self.working_index,
            'seq': self.seq,
            'c_seq': self.c_seq,
            'c_index': self.c_index,
            'filter_index': self.filter_index
        }


def get_binary(action, code_len):
    bs = action.shape[0]
    binary_code = torch.zeros((bs, code_len))
    for i in range(bs):
        binary = bin(int(action[i] + 1))[2:]
        l = list(map(int, str(binary)))
        while len(l) < code_len:
            l.insert(0, 0)
        binary_code[i] = torch.tensor(l)
    return binary_code


def get_choice(output, mask, code_len, fix_choice = None):
    # output:(bs,1,output_size)
    bs, _, output_size = output.size()
    output = output.squeeze(1)

    # apply mask
    output[mask == 0] = -math.inf
    # print(f'pre:{prob}')
    prob = torch.softmax(output, dim = -1)

    policy = Categorical(prob)
    if fix_choice is not None:
        action = fix_choice
        log_prob = policy.log_prob(action)
        return log_prob
    else:
        action = policy.sample()
    log_prob = policy.log_prob(action)

    # get binary code
    binary_code = get_binary(action, code_len)
    return log_prob, action, binary_code


def get_c(output, min_c, interval, fix_c = None):
    # output:(bs,1,output_size)
    output = output.squeeze(1)
    device = output.device

    bs, output_size = output.size()
    # print(f'output.size:{output.shape}')
    prob = torch.softmax(output, dim = -1)

    policy = Categorical(prob)
    if fix_c is not None:
        choice = (fix_c - min_c) // interval
        log_prob = policy.log_prob(choice.to(device))
        return log_prob
    else:
        choice = policy.sample()
    log_prob = policy.log_prob(choice)

    choice = min_c + choice * interval

    return log_prob, choice


class Tokenizer:
    SPECIAL_SYMBOLS = {}

    SPECIAL_FLOAT_SYMBOLS = {}

    SPECIAL_OPERATORS = {}

    SPECIAL_INTEGERS = {}

    def __init__(self):
        self.start = "<START>"
        self.start_id = 1
        self.end = "<END>"
        self.end_id = 2
        self.pad = "<PAD>"
        self.pad_id = 0
        self.vocab = [self.pad, self.start, self.end]

    def encode(self, expr):
        raise NotImplementedError()

    def decode(self, expr):
        raise NotImplementedError()

    def is_unary(self, token):
        raise NotImplementedError()

    def is_binary(self, token):
        raise NotImplementedError()

    def is_leaf(self, token):
        raise NotImplementedError()

    def get_constant_ids(self):
        pass


class MyTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()
        self.variables = [
            "x",
            "gb",
            "gw",
            "dx",
            "randx",
            "pb"
        ]
        self.binary_ops = ["+", "*"]
        self.unary_ops = [
            # "sin",
            # "cos",
            "-",
            # "sign"
        ]
        self.constants = [f"C{i}" for i in range(-1, 1)]
        self.leafs = self.constants + self.variables
        self.vocab = list(self.binary_ops) + list(self.unary_ops) + self.leafs
        self.lookup_table = dict(zip(self.vocab, range(len(self.vocab))))
        self.leaf_index = np.arange(len(self.vocab))[len(self.vocab) - len(self.leafs):]
        self.operator_index = np.arange(len(self.vocab) - len(self.leafs))
        self.binary_index = np.arange(len(self.binary_ops))
        self.unary_index = np.arange(len(self.unary_ops)) + len(self.binary_ops)
        self.vocab_size = len(self.vocab)
        self.constants_index = self.leaf_index[:len(self.constants)]
        self.non_const_index = list(set(range(self.vocab_size)) - set(self.constants_index))
        self.var_index = self.leaf_index[len(self.constants):]

    def decode(self, expr):
        return self.vocab[expr]

    def encode(self, expr):
        return self.lookup_table[expr]

    def is_consts(self, id):
        if torch.is_tensor(id):
            id = id.cpu()
        return np.isin(id, self.constants_index)
        # return id in self.constants_index

    def is_binary(self, token):
        return token in self.binary_ops

    def is_unary(self, token):
        return token in self.unary_ops

    def is_leaf(self, token):
        return token in self.leafs

    def is_var(self, token):
        return token in self.variables

def construct_action(seq, const_seq, tokenizer):
    pre,c_pre = get_prefix_with_consts(seq, const_seq, 0)
    str_expr = [tokenizer.decode(pre[i]) for i in range(len(pre))]
    success,infix = prefix_to_infix(str_expr, c_pre, tokenizer)
    assert success, 'fail to construct the update function'

    return infix


def get_mask(pre_seq, tokenizer, position, max_layer):
    if len(pre_seq.shape) == 1:
        pre_seq = [pre_seq]
    bs, _ = pre_seq.size()
    old_device = pre_seq.device
    pre_seq = pre_seq.cpu().numpy()
    position = position.cpu().numpy()
    masks = []
    for sub_seq, pos in zip(pre_seq, position):
        # if position==-1: set mask all to be zero
        if pos == -1:
            mask = np.zeros(tokenizer.vocab_size)
            masks.append(mask)
            continue
        # init mask
        mask = np.ones(tokenizer.vocab_size)
        # rule: token in the root should not be operands
        if pos == 0:
            mask[tokenizer.leaf_index] = 0
            # mask[tokenizer.encode('sign')]=0
            # mask[tokenizer.encode('sin')]=0
            # mask[tokenizer.encode('cos')]=0

            # mask[tokenizer.encode('*')]=0
            mask[tokenizer.encode('-')] = 0
        else:
            # rule: Avoid invalid operations of + -
            father_token = tokenizer.decode(sub_seq[(pos - 1) // 2])

            if (tokenizer.is_binary(father_token) and pos % 2 == 0) or tokenizer.is_unary(father_token):
                neg_ancestor, target_vocab = find_prefix_of_token_ancestor(tokenizer, sub_seq, pos, '-')
                # rule: direct child of - should not be - or +
                if neg_ancestor == (pos - 1) // 2:
                    mask[tokenizer.encode('+')] = 0
                    mask[tokenizer.encode('-')] = 0
                    # rule: direct child of - located in root should not be x
                    if neg_ancestor == 0:
                        mask[tokenizer.encode('x')] = 0

                if target_vocab is not None:
                    pre_vocab = along_continuous_plus(tokenizer, sub_seq, neg_ancestor)

                    if pre_vocab is not None:
                        mask_index = test_pre(target_vocab[1:], pre_vocab, tokenizer)
                        mask[mask_index] = 0

            if father_token == '+' or (tokenizer.is_binary(father_token) and pos % 2 == 0) or tokenizer.is_unary(father_token):
                plus_ancestor, target_vocab = find_prefix_of_token_ancestor(tokenizer, sub_seq, pos, '+')
                # print(f'plus_ancestor:{plus_ancestor}')
                if target_vocab is not None:
                    visited = np.zeros_like(sub_seq)
                    if father_token == '+' and left_or_right(pos, plus_ancestor) == 'l':
                        visited[2 * plus_ancestor + 1] = 1
                        target_vocab = get_prefix(sub_seq, 2 * plus_ancestor + 1)
                    else:
                        visited[2 * plus_ancestor + 2] = 1
                        target_vocab = get_prefix(sub_seq, 2 * plus_ancestor + 2)

                    sub_root_list = get_along_continuous_plus_with_minus(tokenizer, sub_seq, plus_ancestor, visited)

                    pre_vocab = [get_prefix(sub_seq, sub_root) for sub_root in sub_root_list]
                    if pre_vocab is not None:
                        mask_index = test_pre(target_vocab, pre_vocab, tokenizer)
                        mask[mask_index] = 0
            # rule: pure calculation between constant values is not allowed
            if have_continous_const(sub_seq, pos, tokenizer):
                mask[tokenizer.constants_index] = 0

            # rule: [sin cos sign] cannot directly nest with each other (if they are in the basis symbol set)
            # if father_token in ['sin','cos']:
            #     mask[tokenizer.encode('sign')]=0
            #     mask[tokenizer.encode('sin')]=0
            #     # mask[tokenizer.encode('cos')]=0
            # if father_token == 'sign':
            #     mask[tokenizer.encode('sign')]=0

            # rule: the direct children of + should not be constant values
            if father_token == '+' or father_token == '-':
                mask[tokenizer.constants_index] = 0

            if father_token == '+':
                # children of sign should not be sign (if sign is in the basis symbol set)
                # mask[tokenizer.encode('sign')]=0

                # rule: x+x, gbest+gbest ... is not allowed
                if pos % 2 == 0:
                    left_token = tokenizer.decode(sub_seq[pos - 1])
                    if tokenizer.is_leaf(left_token) and left_token != 'randx':
                        mask[sub_seq[pos - 1]] = 0

            # rule: children of * should not be the same
            if father_token == '*':
                mask[tokenizer.encode('*')] = 0
                mask[tokenizer.encode('-')] = 0
                if pos % 2 == 0:
                    left_id = sub_seq[pos - 1]
                    # 左孩子不是常数，则右孩子必须为常数
                    if not tokenizer.is_consts(left_id):
                        mask[tokenizer.non_const_index] = 0
                    else:
                        mask[tokenizer.constants_index] = 0

            # ! optional: set the minimum layer of the equation tree (you can uncomment the following code if needed)
            if which_layer(position = pos) <= 3:
                if father_token == '*':
                    mask[tokenizer.var_index] = 0
                elif (tokenizer.is_binary(father_token) and pos % 2 == 0 and tokenizer.is_leaf(tokenizer.decode(sub_seq[pos - 1]))) or tokenizer.is_unary(father_token):
                    mask[tokenizer.leaf_index] = 0

            # rule: the leaves should not be operators
            if pos >= int(2 ** (max_layer - 1) - 1):
                mask[tokenizer.operator_index] = 0
        # if np.all(mask<=0.2):
        #     # mask[tokenizer.leaf_index]=1
        #     print(f'mask:{mask}, pos:{pos}, seq:{sub_seq}')
        masks.append(mask)

    return torch.Tensor(masks).to(old_device)


def which_layer(position):
    level = math.floor(math.log2(position + 1))
    return level + 1


def left_or_right(position, root):
    tmp = position
    while tmp != root:
        position = (position - 1) // 2
        if position == root:
            if 2 * root + 1 == tmp:
                return 'l'
            else:
                return 'r'
        tmp = position


def have_continous_const(seq, position, tokenizer):
    father_index = (position - 1) // 2
    father_token = tokenizer.decode(seq[father_index])
    if tokenizer.is_unary(father_token):
        return True
    if tokenizer.is_binary(father_token):
        if position == father_index * 2 + 1:
            return False
        elif tokenizer.is_consts(seq[father_index * 2 + 1]):
            return True


def continus_mul_c(seq, position, tokenizer):
    list = []
    sub_root = (position - 1) // 2
    if tokenizer.decode(seq[sub_root]) == '*':
        visited = np.zeros_like(seq)
        visited[position] = 1

        return get_along_continuous_mul(tokenizer, seq, sub_root, visited)
    else:
        return False


def get_along_continuous_mul(tokenizer, seq, begin, visited):
    # list.append(begin)
    visited[begin] = 1

    if begin != 0 and visited[(begin - 1) // 2] != 1:
        father_token = tokenizer.decode(seq[(begin - 1) // 2])
        if father_token == '*':
            if get_along_continuous_mul(tokenizer, seq, (begin - 1) // 2, visited):
                return True

    if visited[begin * 2 + 1] == 0 and seq[begin * 2 + 1] != -1:
        left_child_token = tokenizer.decode(seq[begin * 2 + 1])
        if left_child_token == '*':
            if get_along_continuous_mul(tokenizer, seq, begin * 2 + 1, visited):
                return True
        elif left_child_token[0] == 'C':
            return True

    if visited[begin * 2 + 2] == 0 and seq[begin * 2 + 2] != -1:
        right_child_token = tokenizer.decode(seq[begin * 2 + 2])
        if right_child_token == '*':
            if get_along_continuous_mul(tokenizer, seq, begin * 2 + 2, visited):
                return True
        elif right_child_token[0] == 'C':
            return True

    return False


def test_pre(target_vocab, pre_vocab, tokenizer):
    target_len = len(target_vocab)
    mask_index = []
    for pre_prefix in pre_vocab:
        if len(pre_prefix) == target_len + 1 and np.all(pre_prefix[:-1] == target_vocab):
            last_token = tokenizer.decode(pre_prefix[-1])
            if last_token != 'randx' and last_token[0] != 'C':
                mask_index.append(pre_prefix[-1])

    return mask_index


def get_along_continuous_plus_with_minus(tokenizer, seq, begin, visited):
    list = []

    # list.append(begin)
    visited[begin] = 1

    if begin != 0 and visited[(begin - 1) // 2] == 0:
        father_token = tokenizer.decode(seq[(begin - 1) // 2])
        if father_token == '+':
            l = get_along_continuous_plus_with_minus(tokenizer, seq, (begin - 1) // 2, visited)
            list.extend(l)

    if visited[begin * 2 + 1] == 0 and seq[begin * 2 + 1] != -1:
        left_child_token = tokenizer.decode(seq[begin * 2 + 1])
        if left_child_token == '+':
            l = get_along_continuous_plus_with_minus(tokenizer, seq, begin * 2 + 1, visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2 * (begin * 2 + 1) + 1)

    if visited[begin * 2 + 2] == 0 and seq[begin * 2 + 2] != -1:
        right_child_token = tokenizer.decode(seq[begin * 2 + 2])
        if right_child_token == '+':
            l = get_along_continuous_plus_with_minus(tokenizer, seq, begin * 2 + 2, visited)
            list.extend(l)
        elif left_child_token == '-':
            list.append(2 * (begin * 2 + 2) + 1)

    return list


def get_along_continuous_plus(tokenizer, seq, begin, visited):
    list = []
    # list.append(begin)
    along_root = False
    visited[begin] = 1
    if begin == 0 and seq[begin] == tokenizer.encode('+'):
        along_root = True

    if begin != 0 and visited[(begin - 1) // 2] == 0:
        father_token = tokenizer.decode(seq[(begin - 1) // 2])
        if father_token == '+':
            l, flag = get_along_continuous_plus(tokenizer, seq, (begin - 1) // 2, visited)
            list.extend(l)
            if flag:
                along_root = True

    if visited[begin * 2 + 1] == 0 and seq[begin * 2 + 1] != -1:
        left_child_token = tokenizer.decode(seq[begin * 2 + 1])
        if left_child_token == '+':
            l, flag = get_along_continuous_plus(tokenizer, seq, begin * 2 + 1, visited)
            list.extend(l)
            if flag:
                along_root = True
        else:
            list.append(begin * 2 + 1)

    if visited[begin * 2 + 2] == 0 and seq[begin * 2 + 2] != -1:
        right_child_token = tokenizer.decode(seq[begin * 2 + 2])
        if right_child_token == '+':
            l, flag = get_along_continuous_plus(tokenizer, seq, begin * 2 + 2, visited)
            list.extend(l)
            if flag:
                along_root = True
        else:
            list.append(begin * 2 + 2)

    return list, along_root


def along_continuous_plus(tokenizer, seq, neg_ancestor):
    list = []
    sub_root = (neg_ancestor - 1) // 2
    if tokenizer.decode(seq[sub_root]) == '+':
        visited = np.zeros_like(seq)
        visited[neg_ancestor] = 1
        continuous_plus_token_list, along_root = get_along_continuous_plus(tokenizer, seq, sub_root, visited)

        pre_vocab = [get_prefix(seq, sub_root) for sub_root in continuous_plus_token_list]

        if along_root:
            pre_vocab.append([tokenizer.encode('x')])
        return pre_vocab
    else:
        return None


def find_prefix_of_token_ancestor(tokenizer, seq, position, token):
    while True:
        father_index = (position - 1) // 2
        father_token = tokenizer.decode(seq[father_index])
        if father_token != token:
            position = father_index
            if position == 0:
                break
        else:
            return father_index, get_prefix(seq, father_index)
    return -1, None


def get_prefix(seq, sub_root):
    if sub_root >= len(seq) or seq[sub_root] == -1:
        return []
    list = []
    list.append(seq[sub_root])
    list.extend(get_prefix(seq, 2 * sub_root + 1))
    list.extend(get_prefix(seq, 2 * sub_root + 2))
    return list


def get_prefix_with_consts(seq, consts, sub_root):
    if sub_root >= len(seq) or seq[sub_root] == -1:
        return [], []
    list_expr = []
    list_c = []
    list_expr.append(seq[sub_root])
    list_c.append(consts[sub_root])
    left_output = get_prefix_with_consts(seq, consts, 2 * sub_root + 1)
    list_expr.extend(left_output[0])
    list_c.extend(left_output[1])
    right_output = get_prefix_with_consts(seq, consts, 2 * sub_root + 2)

    list_expr.extend(right_output[0])
    list_c.extend(right_output[1])
    return list_expr, list_c


def get_next_position(seq, choice, position, tokenizer):
    old_device = position.device
    position = position.cpu().numpy()
    choice = choice.cpu().numpy()
    seq = seq.cpu().numpy()
    next_position = []
    for i in range(len(position)):
        c = choice[i]
        pos = position[i]
        sub_seq = seq[i]
        if c in tokenizer.operator_index:
            next_position.append(2 * pos + 1)
        else:
            append_index = -1
            while True:
                father_index = (pos - 1) // 2
                if father_index < 0:
                    break
                if sub_seq[father_index] in tokenizer.binary_index and sub_seq[2 * father_index + 2] == -1:
                    append_index = father_index * 2 + 2
                    break
                pos = father_index
            next_position.append(append_index)

    return torch.tensor(next_position, dtype = torch.long).to(old_device)


#
def get_str_prefix(seq, const_vals, tokenizer):
    str_expr = []
    c = []
    for i, token_id in enumerate(seq):
        if token_id != -1:
            str_expr.append(tokenizer.decode(token_id))
            c.append(const_vals[i])
    return str_expr, c


def prefix_to_infix(
        expr, constants, tokenizer: Tokenizer
):
    stack = []
    for i, symbol in reversed(list(enumerate(expr))):
        if tokenizer.is_binary(symbol):
            if len(stack) < 2:
                return False, None
            tmp_str = "(" + stack.pop() + symbol + stack.pop() + ")"
            stack.append(tmp_str)
        elif tokenizer.is_unary(symbol) or symbol == "abs":
            if len(stack) < 1:
                return False, None
            if symbol in tokenizer.SPECIAL_SYMBOLS:
                stack.append(tokenizer.SPECIAL_SYMBOLS[symbol].format(stack.pop()))
            else:
                stack.append(symbol + "(" + stack.pop() + ")")
        elif tokenizer.is_leaf(symbol):
            if symbol == "C":
                stack.append(str(constants[i]))
            elif "C" in symbol:
                exponent = int(symbol[1:])
                stack.append(str(constants[i] * 10 ** exponent))
            else:
                stack.append(symbol)

    if len(stack) != 1:
        return False, None

    return True, stack.pop()

def expr_to_func(sympy_expr, variables: List[str]):
    return lambdify(
        variables,
        sympy_expr,
        modules = ["numpy"],
    )
