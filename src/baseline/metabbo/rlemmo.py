from torch import nn
from torch.distributions import Normal

from .networks import MultiHeadEncoder, MLP, EmbeddingNet
from ...rl.ppo import *

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Actor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 hidden_dim1,
                 hidden_dim2,
                 output_dim,
                 global_dim,
                 local_dim,
                 ind_dim
                 ):
        super(Actor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor     
        self.n_layers = n_layers
        self.normalization = normalization
        self.output_dim=output_dim
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.ind_dim = ind_dim

        # figure out the Actor network
        # figure out the embedder for feature embedding
        self.embedder_1 = EmbeddingNet(
                            int(self.global_dim + self.local_dim),
                            int(self.embedding_dim / 2))
        self.embedder_2 = EmbeddingNet(
                            self.ind_dim,
                            int(self.embedding_dim / 2))
        # figure out the fully informed encoder
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization, )
                for _ in range(self.n_layers)))  # stack L layers
        
        # figure out the mu_net and sigma_net
        mlp_config = [{'in': self.embedding_dim,'out': hidden_dim1,'drop_out': 0,'activation':'LeakyReLU'},
                  {'in': hidden_dim1,'out': hidden_dim2,'drop_out':0,'activation':'LeakyReLU'},
                  {'in': hidden_dim2,'out': self.output_dim,'drop_out':0,'activation':'None'}]
        self.decoder = MLP(mlp_config) 
        self.softmax = nn.Softmax(dim = -1)
        
        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in, fixed_action = None, require_entropy = False, to_critic = False,only_critic=False, sampling = True):
        
        population_feature=x_in[:,:,:(self.global_dim + self.local_dim)]
        ind_feature=x_in[:,:,(self.global_dim + self.local_dim):]
        

        # pass through embedder
        h_em_1 = self.embedder_1(population_feature)
        h_em_2 = self.embedder_2(ind_feature)

        h_em = torch.cat((h_em_1, h_em_2), dim = -1)
        assert h_em.shape == (x_in.shape[0], x_in.shape[1], self.embedding_dim)

        # pass through encoder
        logits = self.encoder(h_em)
            
        # share logits to critic net, where logits is from the decoder output 
        if only_critic:
            return logits  # .view(bs, dim, ps, -1)
        
        probs = self.softmax(self.decoder(logits))

        # don't share the network between actor and critic if there is no attention mechanism
        _to_critic= logits

        policy = torch.distributions.Categorical(probs)
        

        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            if sampling:
                action = policy.sample()
            else:
                action = torch.argmax(probs, dim = -1)
        assert action.shape == (x_in.shape[0], x_in.shape[1])
        # get log probability
        log_prob=policy.log_prob(action)

        # The log_prob of each instance is summed up, since it is a joint action for a population
        log_prob=torch.sum(log_prob,dim=1)

        
        if require_entropy:
            entropy = policy.entropy()  # for logging only 
            
            out = (action,
                   log_prob,
                   entropy,
                   _to_critic if to_critic else None,
                   )
        else:
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   )
        return out

class Critic(nn.Module):
    def __init__(self,
             input_dim,
             hidden_dim1,
             hidden_dim2
             ):
        
        super(Critic, self).__init__()
        self.input_dim = input_dim
        # for GLEET, hidden_dim1 = 32, hidden_dim2 = 16
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        mlp_config = [{'in': self.input_dim,'out': hidden_dim1, 'drop_out': 0,'activation':'LeakyReLU'},
                  {'in': hidden_dim1,'out': hidden_dim2,'drop_out':0,'activation':'LeakyReLU'},
                  {'in': hidden_dim2,'out': 1,'drop_out':0,'activation':'None'}]
        self.value_head=MLP(config=mlp_config)

    def forward(self, h_features):
        # since it's joint actions, the input should be meaned at population-dimention
        h_features=torch.mean(h_features,dim=-2)
        # pass through value_head to get baseline_value
        baseline_value = self.value_head(h_features)
        
        return baseline_value.squeeze()


class RLEMMO(PPO_Agent):
    """
    # Introduction
    RLEMMO: Evolutionary Multimodal Optimization Assisted By Deep Reinforcement Learning.
    RLEMMO adopts a meta-black-box optimization framework, maintains a population of solutions, and combines a reinforcement learning agent to flexibly adjust individual-level search strategies to match the current optimization state, thereby improving the search performance for MMOPs. Specifically, RLEMMO encodes terrain characteristics and evolution path information into each individual, and then uses an attention network to promote group information sharing. Through a new reward mechanism that encourages quality and diversity, RLEMMO can be effectively trained using a policy gradient algorithm.
    In the CEC2013 MMOP benchmark, RLEMMO's optimization performance outperforms several strong baseline algorithms, demonstrating its competitiveness in solving multimodal optimization problems.
    # Original Paper
    "[**RLEMMO: Evolutionary Multimodal Optimization Assisted By Deep Reinforcement Learning.**](https://dl.acm.org/doi/abs/10.1145/3638529.3653995)." 
    # Official Implementation
    None
    # Application Scenario
    multi-modal optimization problems(MMOP)
    # Args:
        `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    # Attributes:
        config (object): Stores the configuration object with hyperparameters and settings.
        output_dim (int): Dimension of the output layer for the actor network.
        global_dim (int): Dimension of the global input features.
        local_dim (int): Dimension of the local input features.
        ind_dim (int): Dimension of the individual input features.
        learning_time (int): Tracks the number of learning steps performed.
        cur_checkpoint (int): Tracks the current checkpoint for saving the model.
    # Methods:
        __str__():
            Returns the string representation of the class.
        train_episode(envs, seeds, para_mode='dummy', compute_resource={}, tb_logger=None, required_info={}):
            Trains the agent for one episode using the PPO algorithm.
            Args:
                envs (list): List of environments for training.
                seeds (Optional[Union[int, List[int], np.ndarray]]): Seeds for environment initialization.
                para_mode (Literal['dummy', 'subproc', 'ray', 'ray-subproc']): Parallelization mode for environments.
                compute_resource (dict): Dictionary specifying compute resources (e.g., num_cpus, num_gpus).
                tb_logger (object): TensorBoard logger for logging training metrics.
                required_info (dict): Additional information required from the environment.
            Returns:
                is_train_ended (bool): Indicates whether training has reached the maximum learning step.
                return_info (dict): Dictionary containing training metrics and results.
        rollout_episode(env, seed=None, required_info={}):
            Executes a single rollout episode without training.
            Args:
                env (object): Environment for the rollout.
                seed (Optional[int]): Seed for environment initialization.
                required_info (dict): Additional information required from the environment.
            Returns:
                results (dict): Dictionary containing rollout metrics and results.
    # Returns:
        None
    # Raises:
        None
    """

    def __init__(self, config):
        self.config = config
        
        self.config.optimizer = 'Adam'
        self.config.lr_actor = 5e-4
        self.config.lr_critic = 5e-4
        self.config.lr_scheduler = 'ExponentialLR'
        
        # define parameters
        self.config.embedding_dim = 64
        self.config.encoder_head_num = 4
        self.config.n_encode_layers = 1
        self.config.normalization = 'layer'
        
        self.config.hidden_dim = 64
        self.config.hidden_dim1_actor = 16
        self.config.hidden_dim2_actor = 8
        self.config.hidden_dim1_critic = 16
        self.config.hidden_dim2_critic = 8
        self.output_dim = 5
        self.config.gamma = 0.99
        self.config.n_step = 10
        self.config.K_epochs = 3
        self.config.eps_clip = 0.1 
        self.config.lr_decay = 0.9862327
        self.config.max_grad_norm = 0.1
        
        self.global_dim = 5
        self.local_dim = 5
        self.ind_dim = 12

        
        # figure out the actor network
        actor = Actor(
            embedding_dim = self.config.embedding_dim,
            hidden_dim = self.config.hidden_dim,
            n_heads_actor = self.config.encoder_head_num,
            n_layers = self.config.n_encode_layers,
            normalization = self.config.normalization,
            hidden_dim1=self.config.hidden_dim1_actor,
            hidden_dim2=self.config.hidden_dim2_actor,
            output_dim = self.output_dim,
            global_dim = self.global_dim,
            local_dim = self.local_dim,
            ind_dim = self.ind_dim,
        )

        # figure out the critic network
        critic = Critic(
            input_dim = self.config.embedding_dim,
            hidden_dim1 = self.config.hidden_dim1_critic,
            hidden_dim2 = self.config.hidden_dim2_critic,
        )
        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config, {'actor': actor, 'critic': critic}, [self.config.lr_actor, self.config.lr_critic])
        
    def __str__(self):
        return "RLEMMO"

    def train_episode(self, 
                      envs, 
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                    #   asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                    #   num_cpus: Optional[Union[int, None]] = 1,
                    #   num_gpus: int = 0,
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
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step and not env.all_done():  
                
                memory.states.append(state.clone())
                action, log_lh, entro_p, _to_critic = self.actor(state, require_entropy = True, to_critic=True, sampling = True)
                
                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))
                memory.logprobs.append(log_lh)
                
                entropy.append(entro_p.detach().cpu())

                baseline_val = self.critic(_to_critic)
                baseline_val_detached = baseline_val.detach()
                
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action.cpu().numpy().squeeze())
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
                    entropy = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):
                        # get new action_prob
                        _, log_p, entro_p, _to_critic = self.actor(old_states[tt], fixed_action = old_actions[tt],
                                                        require_entropy = True,# take same action
                                                        to_critic = True,
                                                        sampling = True)

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val = self.critic(_to_critic)
                        baseline_val_detached = baseline_val.detach()
                        
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
                R = self.critic(self.actor(state, only_critic = True, sampling = True)).detach()
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
                    return_info = {'return': _Rs,'loss': _loss, 'learn_steps': self.learning_time, }
                    env_cost = env.get_env_attr('cost')
                    return_info['gbest'] = env_cost[-1]
                    for key in required_info.keys():
                        return_info[key] = env.get_env_attr(required_info[key])
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()
        
        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs,'loss': _loss, 'learn_steps': self.learning_time,}
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
            env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.Tensor(state).double().unsqueeze(0).to(self.device)
                except:
                    state = [state]
                action = self.actor(state, sampling = False)[0]
                action = action.cpu().numpy().squeeze()
                state, reward, is_done, info = env.step(action)
                R += reward
            env_cost = env.get_env_attr('cost')
            env_fes = env.get_env_attr('fes')
            env_pr = env.get_env_attr('pr')
            env_sr = env.get_env_attr('sr')
            results = {'cost': env_cost, 'fes': env_fes, 'return': R, 'pr': env_pr, 'sr':env_sr}

            if self.config.full_meta_data:
                meta_X = env.get_env_attr('meta_X')
                meta_Cost = env.get_env_attr('meta_Cost')
                meta_Pr = env.get_env_attr('meta_Pr')
                meta_Sr = env.get_env_attr('meta_Sr')
                metadata = {'X': meta_X, 'Cost': meta_Cost, 'Pr': meta_Pr, 'Sr': meta_Sr}
                results['metadata'] = metadata
    
            for key in required_info.keys():
                results[key] = getattr(env, required_info[key])
            return results

