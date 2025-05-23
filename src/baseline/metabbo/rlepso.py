from torch import nn
from torch.distributions import Normal

from .networks import MLP
from ...rl.ppo import *

class Actor(nn.Module):
    def __init__(self,
                 config,
                 ):
        super(Actor, self).__init__()
        net_config = [{'in': config.feature_dim, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 32, 'out': config.action_dim, 'drop_out': 0, 'activation': 'None'}]
        self.__mu_net = MLP(net_config)
        self.__sigma_net = MLP(net_config)
        self.__max_sigma = config.max_sigma
        self.__min_sigma = config.min_sigma

    def forward(self, x_in, fixed_action=None, require_entropy=False):  # x-in: bs*gs*9

        mu = (torch.tanh(self.__mu_net(x_in)) + 1.) / 2.
        sigma = (torch.tanh(self.__sigma_net(x_in)) + 1.) / 2. * (self.__max_sigma - self.__min_sigma) + self.__min_sigma

        policy = Normal(mu, sigma)

        if fixed_action is not None:
            action = fixed_action
        else:
            action = torch.clamp(policy.sample(), min=0, max=1)
        log_prob = policy.log_prob(action)

        log_prob = torch.sum(log_prob, dim = 1)

        if require_entropy:
            entropy = policy.entropy()  # for logging only bs,ps,2

            out = (action,
                   log_prob,
                   entropy)
        else:
            out = (action,
                   log_prob,
                   )
        return out


class Critic(nn.Module):
    def __init__(self,
                 config
                 ):
        super(Critic, self).__init__()
        self.__value_head = MLP([{'in': config.feature_dim, 'out': 16, 'drop_out': 0, 'activation': 'ReLU'},
                                 {'in': 16, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                                 {'in': 8, 'out': 1, 'drop_out': 0, 'activation': 'None'}])

    def forward(self, h_features):
        baseline_value = self.__value_head(h_features)
        # baseline_value = baseline_value[0]
        # print(type(baseline_value.detach()))
        return baseline_value.detach().squeeze(), baseline_value.squeeze()


class RLEPSO(PPO_Agent):
    """
    # Introduction
    This paper proposes a new reinforcement learning driven ensemble particle swarm optimization algorithm (RLEPSO). The algorithm uses reinforcement learning technology to adaptively select different PSO variants, thereby improving the algorithm's exploration ability and convergence. Specifically, the RLEPSO algorithm uses reinforcement learning to dynamically adjust the use probability of different PSO variants to better balance exploration and utilization. At the same time, it uses an ensemble learning method to combine multiple PSO variants together to fully utilize the advantages of different variants.
    # Original Paper
    "[**RLEPSO: Reinforcement learning based Ensemble particle swarm optimizer**](https://dl.acm.org/doi/abs/10.1145/3508546.3508599)." Proceedings of the 2021 4th International Conference on Algorithms, Computing and Artificial Intelligence. (2021)
    # Official Implementation
    None
    # Application Scenario
    single-object optimization problems(SOOP)
    # Args:
        `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    # Attributes:
        config (object): Configuration object containing hyperparameters and settings.
        actor (Actor): The actor network responsible for policy generation.
        critic (Critic): The critic network responsible for value estimation.
        optimizer (torch.optim.Optimizer): Optimizer used for training the networks.
        learning_time (int): Counter for the number of learning steps performed.
        device (str): Device used for computation ('cpu' or 'cuda').
    # Methods:
        __str__():
            Returns the string representation of the class.
        train_episode(envs, seeds, para_mode='dummy', compute_resource={}, tb_logger=None, required_info={}):
            Trains the agent for one episode using the PPO algorithm.
            Args:
                envs (list): List of environments for training.
                seeds (Optional[Union[int, List[int], np.ndarray]]): Seeds for environment initialization.
                para_mode (Literal['dummy', 'subproc', 'ray', 'ray-subproc']): Parallelization mode for environments.
                compute_resource (dict): Resources for computation (e.g., CPUs, GPUs).
                tb_logger (object): TensorBoard logger for logging training metrics.
                required_info (dict): Additional information required from the environment.
            Returns:
                tuple: A tuple containing a boolean indicating if training has ended and a dictionary with training information.
            Raises:
                None.
        rollout_episode(env, seed=None, required_info={}):
            Executes a single rollout episode in a given environment.
            Args:
                env (object): The environment for the rollout.
                seed (Optional[int]): Seed for environment initialization.
                required_info (dict): Additional information required from the environment.
            Returns:
                dict: A dictionary containing rollout results, including costs, function evaluations, and returns.
            Raises:
                None.
    # Returns:
        None.
    # Raises:
        None.
    """
    def __init__(self, config):

        self.config = config
        # add specified config
        self.config.feature_dim=1
        self.config.action_dim=35
        self.config.action_shape=(35,)
        self.config.n_step=10
        self.config.K_epochs=3
        self.config.eps_clip=0.1
        self.config.gamma=0.999
        self.config.max_sigma=0.7
        self.config.min_sigma=0.01
        self.config.lr=1e-5
        self.config.optimizer = 'Adam'
        self.config.max_grad_norm = math.inf
        # config.lr_decay=0.99


        # figure out the actor
        actor = Actor(config)

        # figure out the critic
        critic = Critic(config)
        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config, {'actor': actor, 'critic': critic}, self.config.lr)

    def __str__(self):
        return "RLEPSO"

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
            bl_val_detached = []
            bl_val = []
            entropy = []

            # accumulate transition
            while t - t_s < n_step and not env.all_done():

                memory.states.append(state.clone())
                action, log_lh, entro_p = self.actor(state,require_entropy=True)

                # action = action.reshape(self.config.action_shape)

                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))


                memory.logprobs.append(log_lh)
                entropy.append(entro_p.detach().cpu())

                baseline_val_detached, baseline_val = self.critic(state)
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action.detach().cpu().numpy())
                memory.rewards.append(torch.Tensor(rewards).to(self.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards
                # store info

                # next
                t = t + 1

                try:
                    state = torch.Tensor(state).to(self.device)
                except:
                    pass

            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update
            old_actions = torch.stack(memory.actions)
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
                    bl_val_detached = []
                    bl_val = []
                    entropy = []

                    for tt in range(t_time):
                        # get new action_prob
                        _, log_p, entro_p = self.actor(state, fixed_action = old_actions[tt], require_entropy=True)

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val_detached, baseline_val = self.critic(state)

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(state)[0]
                critic_output = R.clone()
                # print(R.shape)
                for r in range(len(reward_reversed)):
                    # print(reward_reversed[r].shape)
                    R = R * gamma + reward_reversed[r]
                    # print(R)
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)

                Reward = Reward.view(-1)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())
                # print(Reward.shape,bl_val_detached.shape)
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

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()
                _loss.append(loss.item())
                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

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
                                         critic_output, logprobs, entropy, approx_kl_divergence)

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': _loss,'learn_steps': self.learning_time, }
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
        return_info = {'return': _Rs, 'loss': _loss,'learn_steps': self.learning_time,}
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
                action = self.actor(state.unsqueeze(0))[0]
                action = action.detach().cpu().numpy()
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
