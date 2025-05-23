import torch.nn as nn
from torch.distributions import Normal

from .networks import MLP, MultiHeadEncoder, EmbeddingNet, PositionalEncoding
from ...rl.ppo import *

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input

class Feature_Extractor(nn.Module):
    def __init__(self, node_dim = 3, hidden_dim = 16, n_heads = 1, ffh = 16, n_layers = 1,
                 use_positional_encoding = True,
                 is_train = False,
                 device = None
                 ):
        super(Feature_Extractor, self).__init__()
        # bs * dim * pop_size * 2 -> bs * dim * pop_size * hidden_dim
        # node_dim = 2 if is_mlp else 3
        self.device = device
        self.embedder = EmbeddingNet(node_dim = node_dim, embedding_dim = hidden_dim)

        self.fes_embedder = EmbeddingNet(node_dim = 1, embedding_dim = 16)
        self.use_PE = use_positional_encoding
        if self.use_PE:
            self.position_encoder = PositionalEncoding(hidden_dim, 512)

        # (bs, dim, pop_size, hidden_dim) -> (bs, dim, pop_size, hidden_dim)
        self.dimension_encoder = mySequential(*(MultiHeadEncoder(n_heads = n_heads,
                                                                 embed_dim = hidden_dim,
                                                                 feed_forward_hidden = ffh,
                                                                 normalization = 'layer')
                                                for _ in range(n_layers)))
        # (bs, pop_size, dim, hidden_dim) -> (bs, pop_size, dim, hidden_dim)
        self.individual_encoder = mySequential(*(MultiHeadEncoder(n_heads = n_heads,
                                                                  embed_dim = hidden_dim,
                                                                  feed_forward_hidden = ffh,
                                                                  normalization = 'layer')
                                                 for _ in range(n_layers)))
        self.is_train = is_train
        self.to(self.device)

    def set_on_train(self):
        self.is_train = True

    def set_off_train(self):
        self.is_train = False

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Number of trainable parameters
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, state):
        if self.is_train:
            return self._run(state)
        else:
            with torch.no_grad():
                return self._run(state)

    def encode_y(self, ys, constant = 32.0):
        """
            Encodes the input tensor `ys` by separating it into mantissa and exponent parts using PyTorch.

            Parameters:
            - ys: Input tensor of shape (bs, n), where bs is the batch size and n is the population size.
            - constant: A scaling constant for the exponent part, default is 32.0.

            Returns:
            - encoded_y: The mantissa part of the encoded `ys`, scaled by 1/10, shape (bs, n).
            - encoded_e: The exponent part of the encoded `ys`, scaled by the given constant, shape (bs, n).
        """
        # Get the batch size (bs) and population size (n)
        bs, n = ys.shape

        # Create a mask for ys being zero
        mask = ys == 0

        # Replace zeros in ys with 1 to avoid log10 issues
        ys[mask] = 1

        # Calculate the exponent and mantissa
        exponent = torch.floor(torch.log10(torch.abs(ys)))
        mantissa = ys / (10 ** exponent)

        # Reshape and scale mantissa and exponent
        encoded_y = mantissa.reshape(bs, n) / 10
        encoded_e = exponent.reshape(bs, n) + 1
        encoded_e = encoded_e / constant

        # Apply the mask to set encoded_y and encoded_e to 0 where ys was 0
        encoded_y[mask] = 0
        encoded_e[mask] = 0

        return encoded_y, encoded_e

    def _run(self, state):
        """
            Processes the input tensors and applies attention mechanisms based on the selected order.

            Parameters:
            - xs: Input numpy_array of shape (bs, n, d),
                where bs is the batch size, n is the population size, and d is the dimension.
            - ys: Input numpy_array of shape (bs, n), representing target values.

            Returns:
            - out: Processed output tensor, depending on the attention mechanism used,
                with shape (bs, pop_size, hidden_dim).
        """
        # xs : bs * n * d
        # ys : bs * n
        # fes : bs * 1
        xs = state[:, :, :-2]
        ys = state[:, :, -2]
        fes = state[:, 0, -1][:, None]
        # xs, ys, fes = state['x'], state['y'], state['fes']
        bs, pop_size, dim = xs.shape

        y_, e_ = self.encode_y(ys)  # 2D : bs * n and bs * n
        ys = y_[:, :, None]  # 3D : bs * n * 1
        es = e_[:, :, None]  # 3D : bs * n * 1

        a_x = xs[:, :, :, None]  # 4D : bs * n * d * 1
        a_y = ys.repeat((1, 1, dim))[:, :, :, None]
        a_e = es.repeat((1, 1, dim))[:, :, :, None]


        raw_feature = torch.cat([a_x, a_y, a_e], dim = -1)
        raw_feature = raw_feature.permute(0, 2, 1, 3) # 4D : bs * d * n * 3
        # raw_feature = np.concatenate([a_x, a_y, a_e], axis = -1).transpose((0, 2, 1, 3))  # 4D : bs * d * n * 3

        h_ij = self.embedder(raw_feature.to(self.device))  # 4D : bs * d * n * hidden_dim

        node_dim = h_ij.shape[-1]

        h_ij = h_ij.view(-1, pop_size, node_dim)  # resize h_ij 3D : (bs * dim, pop_size, hidden_dim)

        # (bs * dim, pop_size, hidden_dim) -> (bs, dim, pop_size, hidden_dim)
        o_ij = self.dimension_encoder(h_ij).view(bs, dim, pop_size, node_dim).to(self.device)
        # (bs, pop_size, dim, hidden_dim) -> (bs * pop_size, dim, hidden_dim)
        o_i = o_ij.permute(0, 2, 1, 3).contiguous().view(-1, dim, node_dim).to(self.device)

        if self.use_PE:
            o_i = o_i + self.position_encoder.get_PE(dim).to(self.device) * 0.5

        # mean
        # o_i = torch.mean(o_i, 1).view(bs, pop_size, node_dim).to(self.device) # (bs, pop_size, hidden_dim)

        tensor_fes = torch.tensor(fes).to(self.device)
        embed_fes = self.fes_embedder(tensor_fes).to(self.device)  # [bs, 16]
        embed_fes = embed_fes.unsqueeze(1)  # [bs, 1, 16]
        embed_fes = embed_fes.expand(-1, pop_size, -1)  # [bs, pop_size, 16]

        # o_i = self.individual_encoder(o_i).to(self.device) # (bs, pop_size, hidden_dim)
        o_i = self.individual_encoder(o_i).view(bs, pop_size, dim, node_dim).to(self.device)

        o_i = torch.mean(o_i, 2).to(self.device)
        out = torch.cat((o_i, embed_fes), -1).to(self.device)  # (bs, pop_size, hidden_dim + 16)

        return out

class Actor(nn.Module):
    '''
        Actor is a neural network module designed to select mutation and crossover parameters
            for a differential evolution algorithm.

        This network consists of:
        - An operator selection network to choose a mutation operator.
        - Networks to predict mutation and crossover parameters (means and standard deviations).

        Attributes:
        - input_dim: The dimension of the input state.
        - n_operator: The number of mutation operators to choose from.
        - n_mutation: The number of mutation parameters to output.
        - n_crossover: The number of crossover parameters to output.
        - output_dim: The total number of outputs, including the selected operator and mutation/crossover parameters.
        - max_sigma: The maximum value for the standard deviation of mutation/crossover parameters.
        - min_sigma: The minimum value for the standard deviation of mutation/crossover parameters.
        '''
    def __init__(self, input_dim, mu_operator, cr_operator, n_mutation, n_crossover):
        super(Actor, self).__init__()
        self.output_dim = 2 + n_mutation + n_crossover

        # Configuration for the mutation operator selection network
        mutation_operator_net_config = [{'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                            {'in': 32, 'out': mu_operator, 'drop_out': 0, 'activation': 'None'}]

        # Configuration for the crossover operator selection network
        crossover_operator_net_config = [{'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                            {'in': 32, 'out': cr_operator, 'drop_out': 0, 'activation': 'None'}]

        # Configuration for the mutation parameters
        mutation_param_net_config = [{'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                        {'in': 32, 'out': n_mutation, 'drop_out': 0, 'activation': 'None'}]

        # Configuration for the crossover parameters
        crossover_param_net_config = [{'in': input_dim, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
                                        {'in': 32, 'out': n_crossover, 'drop_out': 0, 'activation': 'None'}]

        # Initialize networks for operator selection and parameters
        self.mutation_selector_net = MLP(mutation_operator_net_config)
        self.crossover_selector_net = MLP(crossover_operator_net_config)

        self.mutation_param_sigma_net = MLP(mutation_param_net_config)
        self.mutation_param_mu_net = MLP(mutation_param_net_config)
        self.crossover_param_sigma_net = MLP(crossover_param_net_config)
        self.crossover_param_mu_net = MLP(crossover_param_net_config)

        # Define maximum and minimum sigma values for parameter scaling
        self.max_sigma = 0.7
        self.min_sigma = 0.1

        self.n_mutation = n_mutation
        self.n_crossover = n_crossover

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Number of trainable parameters
        return {'Total': total_num, 'Trainable': trainable_num}

    def get_action(self, x):
        mutation_p = self.mutation_selector_net(x)  # 3D : bs * n *mu_operator
        crossover_p = self.crossover_selector_net(x)  # 3D : bs * n * cr_operator
        # Apply softmax to get probabilities for operator selection

        mutation_p = torch.softmax(mutation_p, dim = 2)
        crossover_p = torch.softmax(crossover_p, dim = 2)
        # Create categorical distribution for operator selection
        mutation_operator_distribution = torch.distributions.Categorical(mutation_p)
        crossover_operator_distribution = torch.distributions.Categorical(crossover_p)

        # Calculate mutation parameters using tanh activation and normalization
        m_mu = (torch.tanh(self.mutation_param_mu_net(x)) + 1.) / 2.
        m_sigma = (torch.tanh(self.mutation_param_sigma_net(x)) + 1.) / 2. * (
                self.max_sigma - self.min_sigma) + self.min_sigma

        # Calculate crossover parameters using tanh activation and normalization
        c_mu = (torch.tanh(self.crossover_param_mu_net(x)) + 1.) / 2.
        c_sigma = (torch.tanh(self.crossover_param_sigma_net(x)) + 1.) / 2. * (
                self.max_sigma - self.min_sigma) + self.min_sigma

        # Create Normal distributions for mutation and crossover parameters
        m_policy = Normal(m_mu, m_sigma)
        c_policy = Normal(c_mu, c_sigma)

        mutation_operator_action = mutation_operator_distribution.sample()  # 2D : bs * n
        crossover_operator_action = crossover_operator_distribution.sample()  # 2D : bs * n
        mutation_action = torch.clamp(m_policy.sample(), min = 0, max = 1)  # 3D : bs * n * n_mutation
        crossover_action = torch.clamp(c_policy.sample(), min = 0, max = 1)  # 3D : bs * n * n_crossover

        # Concatenate all actions into a single tensor
        action = torch.cat([mutation_operator_action[:, :, None],
                            crossover_operator_action[:, :, None],
                            mutation_action,
                            crossover_action],
                            dim = 2)  # 3D : bs * n * (2 + n_mutation + n_crossover)

        return (action, m_mu, m_sigma, c_mu, c_sigma)

    def forward(self, x, fixed_action = None, require_entropy = False):
        mutation_p = self.mutation_selector_net(x)  # 3D : bs * n * mu_operator
        crossover_p = self.crossover_selector_net(x) # 3D : bs * n * cr_operator
        # Apply softmax to get probabilities for operator selection

        mutation_p = torch.softmax(mutation_p, dim = 2)
        crossover_p = torch.softmax(crossover_p, dim = 2)

        mutation_operator_distribution = torch.distributions.Categorical(mutation_p)
        crossover_operator_distribution = torch.distributions.Categorical(crossover_p)

        # Calculate mutation parameters using tanh activation and normalization
        m_mu = (torch.tanh(self.mutation_param_mu_net(x)) + 1.) / 2.
        m_sigma = (torch.tanh(self.mutation_param_sigma_net(x)) + 1.) / 2. * (
                    self.max_sigma - self.min_sigma) + self.min_sigma

        # Calculate crossover parameters using tanh activation and normalization
        c_mu = (torch.tanh(self.crossover_param_mu_net(x)) + 1.) / 2.
        c_sigma = (torch.tanh(self.crossover_param_sigma_net(x)) + 1.) / 2. * (
                    self.max_sigma - self.min_sigma) + self.min_sigma

        # Create Normal distributions for mutation and crossover parameters
        m_policy = Normal(m_mu, m_sigma)
        c_policy = Normal(c_mu, c_sigma)

        if fixed_action is not None:
            # If fixed actions are provided, use them directly
            mutation_operator_action = fixed_action[:, :, 0]
            crossover_operator_action = fixed_action[:, :, 1]
            mutation_action = fixed_action[:, :, 2: 2 + self.n_mutation]
            crossover_action = fixed_action[:, :, -self.n_crossover:]
        else:
            # Sample actions from the distributions
            mutation_operator_action = mutation_operator_distribution.sample()  # 2D : bs * n
            crossover_operator_action = crossover_operator_distribution.sample() # 2D : bs * n
            mutation_action = torch.clamp(m_policy.sample(), min = 0, max = 1)  # 3D : bs * n * n_mutation
            crossover_action = torch.clamp(c_policy.sample(), min = 0, max = 1)  # 3D : bs * n * n_crossover

        # Concatenate all actions into a single tensor
        action = torch.cat([mutation_operator_action[:, :, None],
                            crossover_operator_action[:, :, None],
                            mutation_action,
                            crossover_action],
                            dim = 2)  # 3D : bs * n * (2 + n_mutation + n_crossover)

        # Calculate log probabilities for each action
        mutation_operator_log_prob = mutation_operator_distribution.log_prob(mutation_operator_action)  # 2D : bs * n
        crossover_operator_log_prob = crossover_operator_distribution.log_prob(crossover_operator_action)  # 2D : bs * b
        mutation_log_prob = m_policy.log_prob(mutation_action)  # 3D : bs * n * n_mutation
        crossover_log_prob = c_policy.log_prob(crossover_action)  # 3D : bs * n * n_crossover

        mutation_log_prob = torch.sum(mutation_log_prob, dim = 2)  # 2D : bs * n
        crossover_log_prob = torch.sum(crossover_log_prob, dim = 2)  # 2D : bs * n
        log_prob = mutation_operator_log_prob + \
                    crossover_operator_log_prob + \
                    mutation_log_prob + \
                    crossover_log_prob  # Total log probability
        log_prob = torch.sum(log_prob, dim = 1)
        if require_entropy:
            # Calculate entropy for each distribution if required
            mutation_operator_entropy = mutation_operator_distribution.entropy()
            crossover_operator_entropy = crossover_operator_distribution.entropy()
            mutation_entropy = m_policy.entropy()
            crossover_entropy = c_policy.entropy()
            # Concatenate all entropy values into a single tensor
            entropy = torch.cat([mutation_operator_entropy[:, :, None],
                                crossover_operator_entropy[:, :, None],
                                mutation_entropy,
                                crossover_entropy],
                                dim = 2)  # 3D : bs * n * (2 + n_mutation + n_crossover)
            out = (action, log_prob, entropy)
        else:
            out = (action, log_prob,)

        return out

class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        net_config = [{'in': input_dim, 'out': 16, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 16, 'out': 8, 'drop_out': 0, 'activation': 'ReLU'},
                      {'in': 8, 'out': 1, 'drop_out': 0, 'activation': 'None'}]
        self.__value_head = MLP(net_config)
    def forward(self, state):
        baseline_value = self.__value_head(state).mean(1) # [bs, ps, 1] -> [bs, 1]
        baseline_value = baseline_value.squeeze()
        return baseline_value.detach(), baseline_value

    def get_parameter_number(self):
        total_num = sum(p.numel() for p in self.parameters())  # Total number of parameters
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)  # Number of trainable parameters
        return {'Total': total_num, 'Trainable': trainable_num}

class RLDEAFL(PPO_Agent):
    """
    # Introduction
    This paper proposes a new reinforcement learning-based adaptive differential evolution algorithm, called RLDE-AFL. The algorithm achieves automatic feature learning for optimization problems by integrating a learnable feature extraction module into the differential evolution algorithm. At the same time, it also adopts a reinforcement learning-driven adaptive algorithm configuration strategy to better adapt to the characteristics of different optimization problems.
    Specifically, RLDE-AFL uses an attention mechanism neural network and a mantissa-based embedding method to transform the solution population and its target value into expressive optimization problem features. In addition, it introduces a comprehensive algorithm configuration space containing multiple differential evolution operators, and uses reinforcement learning to adaptively adjust these operators and their parameters, thereby improving the algorithm's behavioral diversity and optimization performance.
    # Original Paper
    "[**Reinforcement Learning-based Self-adaptive Differential Evolution through Automated Landscape Feature Learning**](https://arxiv.org/abs/2503.18061)." 
    # Official Implementation
    [RLDE-AFL](https://github.com/GMC-DRL/RLDE-AFL)
    # Application Scenario
    single-object optimization problems(SOOP)
    # Args:
        `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    # Attributes:
        config (object): Stores the configuration object with hyperparameters and settings.
        fe (Feature_Extractor): Feature extractor for processing input states.
        actor (Actor): Actor network for generating actions.
        critic (Critic): Critic network for estimating value functions.
        optimizer (torch.optim.Optimizer): Optimizer for training the agent.
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
                compute_resource (dict): Resources for computation (e.g., CPUs, GPUs).
                tb_logger (object): TensorBoard logger for tracking metrics.
                required_info (dict): Additional information required from the environment.
            Returns:
                Tuple[bool, dict]: A tuple containing a boolean indicating if training has ended and a dictionary with training information.
        rollout_episode(env, seed=None, required_info={}):
            Executes a single rollout episode in a given environment.
            Args:
                env (object): The environment for the rollout.
                seed (Optional[int]): Seed for environment initialization.
                required_info (dict): Additional information required from the environment.
            Returns:
                dict: A dictionary containing rollout results such as costs, returns, and additional information.
    # Returns:
        None
    # Raises:
        None
    """
    def __init__(self, config):
        self.config = config

        self.config.optimizer = 'Adam'
        self.config.lr = 1e-4

        self.config.fe_hidden_dim = 64
        self.config.fe_n_layers = 1

        self.config.gamma = 0.99
        self.config.n_step = 10
        self.config.K_epochs = 3
        self.config.eps_clip = 0.2
        self.config.max_grad_norm = 1

        self.config.mu_operator = 14
        # [best/1] [best/2] [rand/1] [rand/2] [current-to-best/1] [rand-to-best/1] [current-to-rand/1] [current-to-pbest/1] [ProDE-rand/1]
        # [TopoDE-rand/1] [current-to-pbest/1+archive] [HARDDE-current-to-pbest/2] [current-to-rand/1+archive] [weighted-rand-to-pbest/1]
        self.config.cr_operator = 3
        # [binomial] [exponential] [p-binomial]
        self.config.n_mutation = 3
        self.config.n_crossover = 2

        fe = Feature_Extractor(hidden_dim = self.config.fe_hidden_dim,
                               n_layers = self.config.fe_n_layers,
                               device = self.config.device)

        actor = Actor(input_dim = self.config.fe_hidden_dim + 16,
                      mu_operator = self.config.mu_operator,
                      cr_operator = self.config.cr_operator,
                      n_mutation = self.config.n_mutation,
                      n_crossover = self.config.n_crossover)

        critic = Critic(input_dim = self.config.fe_hidden_dim + 16)

        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config, {'actor': actor, 'critic': critic, 'fe': fe}, self.config.lr)

    def __str__(self):
        return "RLDEAFL"

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

        torch.set_grad_enabled(True)
        self.fe.set_on_train()
        self.actor.train()
        self.critic.train()
        env.seed(seeds)
        memory = Memory()

        gamma = self.gamma
        n_step = self.n_step

        K_epochs = self.K_epochs
        eps_clip = self.eps_clip
        # 3D [bs, NP, DIM + 2]
        state = env.reset()
        try:
            state = torch.Tensor(state).to(self.device)
        except:
            pass

        t = 0
        _R = torch.zeros(len(env))
        _loss = []

        while not env.all_done():
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step and not env.all_done():
                memory.states.append(state.clone())

                feature = self.fe(state).to(self.config.device) # 3D: [bs, pop_size, hidden_dim + 16]

                action, log_lh, entro_p = self.actor(feature, require_entropy = True)

                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))
                memory.logprobs.append(log_lh)

                entropy.append(entro_p.detach().cpu())

                baseline_val_detached, baseline_val = self.critic(feature)

                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                action = action.cpu().numpy()
                # state transient
                state, rewards, is_end, info = env.step(action)
                memory.rewards.append(torch.Tensor(rewards).to(self.device))

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
                        old_features = self.fe(old_states[tt])

                        _, log_p, entro_p = self.actor(old_features,
                                                       fixed_action = old_actions[tt],
                                                       require_entropy = True  # take same action
                                                       )

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val_detached, baseline_val = self.critic(old_features)

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
                feature = self.fe(state) # 3D [bs, ps, hidden_dim + 16]
                R = self.critic(feature)[0]
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
        self.fe.set_off_train()
        self.actor.eval()
        self.critic.eval()
        with torch.no_grad():
            env.seed(seed)
            is_done = False
            state = env.reset()
            R = 0
            while not is_done:
                try:
                    state = torch.Tensor(state).to(self.device)
                except:
                    pass
                feature = self.fe(state[None, :]).to(self.config.device)
                action = self.actor.get_action(feature)[0].detach().cpu().numpy()
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
                results[key] = env.get_env_attr(required_info[key])
            return results
