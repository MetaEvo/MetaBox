import torch
from torch import nn
from ...rl.basic_agent import Basic_Agent
from ...rl.utils import *
from ...environment.parallelenv.parallelenv import ParallelEnv
from dill import loads, dumps
from typing import Optional, Union, Literal, List


def scale(x, lb, ub):
    x = torch.sigmoid(x)
    x = lb + (ub - lb) * x
    return x


class RNNOPT(Basic_Agent):
    """
    # Introduction
    The paper "Meta-Learning for Black-Box Optimization" explores the use of meta-learning techniques to address black-box optimization problems, where the objective function is unknown and derivative-free methods are required. The authors propose RNN-Opt, a recurrent neural network-based optimizer trained under the meta-learning framework to optimize real-parameter single-objective continuous functions within constrained budgets. Unlike traditional approaches, this method employs a regret-based loss function during training, which better aligns with real-world testing scenarios. Additionally, the paper introduces enhancements to handle challenges such as unknown function ranges and domain-specific constraints. 
    # Original Paper
    "[**Meta-learning for black-box optimization**](https://link.springer.com/chapter/10.1007/978-3-030-46147-8_22)." Joint European Conference on Machine Learning and Knowledge Discovery in Databases. (2019)
    # Official Implementation
    None
    # Application Scenario
    single-object optimization problems(SOOP)
    # Args:
        `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    # Attributes:
        config (object): Configuration object containing hyperparameters and settings.
        device (str): Device to be used for computation ('cpu' or 'cuda').
        hidden_size (int): Size of the hidden layer in the LSTM network.
        proj_size (int): Size of the projection layer in the LSTM network.
        optimizer (torch.optim.Optimizer): Optimizer used for training the network.
        network (list): List of network names used in the agent.
        learning_step (int): Counter for the number of learning steps performed.
        cur_checkpoint (int): Counter for the current checkpoint saved.
    # Methods:
        __str__():
            Returns the string representation of the class.
        set_network(networks: dict, learning_rates: float):
            Sets up the neural networks and their corresponding optimizers.
            Args:
                networks (dict): Dictionary of network names and their instances.
                learning_rates (float): Learning rate(s) for the networks.
            Raises:
                ValueError: If the length of learning rates does not match the number of networks.
        get_step():
            Returns the current learning step.
            Returns:
                int: The current learning step.
        update_setting(config):
            Updates the agent's settings and resets the learning step.
            Args:
                config (object): Configuration object with updated settings.
        train_episode(envs, seeds, para_mode='dummy', compute_resource={}, tb_logger=None, required_info={}):
            Trains the agent for one episode.
            Args:
                envs (list): List of environments for training.
                seeds (Optional[Union[int, List[int], np.ndarray]]): Seeds for environment initialization.
                para_mode (str): Parallelization mode for environments.
                compute_resource (dict): Resources for computation (e.g., CPUs, GPUs).
                tb_logger (object): TensorBoard logger for logging training metrics.
                required_info (dict): Additional information required from the environment.
            Returns:
                tuple: A boolean indicating if the maximum learning step was exceeded and a dictionary with training information.
        rollout_episode(env, seed=None, required_info={}):
            Performs a rollout episode in the environment.
            Args:
                env (object): Environment for the rollout.
                seed (Optional[int]): Seed for environment initialization.
                required_info (dict): Additional information required from the environment.
            Returns:
                dict: Results of the rollout, including cost, function evaluations, and metadata.
        log_to_tb_train(tb_logger, mini_step, grad_norms, loss, extra_info={}):
            Logs training metrics to TensorBoard.
            Args:
                tb_logger (object): TensorBoard logger.
                mini_step (int): Current mini-step in training.
                grad_norms (tuple): Gradient norms before and after clipping.
                loss (torch.Tensor): Loss value.
                extra_info (dict): Additional information to log.
    # Returns:
        None
    # Raises:
        ValueError: If the length of the learning rates list does not match the number of networks.
        AssertionError: If the optimizer specified in the configuration is not available in PyTorch.
    """

    def __init__(self, config):
        super().__init__(config)
        config.lr = 1e-5
        self.config = config
        self.device = self.config.device
        self.config.optimizer = 'Adam'
        self.config.max_grad_norm = np.inf
        self.hidden_size = 32
        self.proj_size = config.dim
        torch.set_default_dtype(torch.float64)

        # self.net=nn.LSTM(input_size=config.dim+2,hidden_size=self.hidden_size,proj_size=config.dim)
        # self.optimizer = torch.optim.Adam([{'params': self.net.parameters(), 'lr': config.lr}])
        net = nn.LSTM(input_size = config.dim + 2, hidden_size = self.hidden_size, proj_size = config.dim)
        self.set_network({'net': net}, [config.lr])
        self.learning_step = 0
        self.cur_checkpoint = 0

        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
        self.cur_checkpoint = +1

    def __str__(self):
        return "RNNOPT"

    def set_network(self, networks: dict, learning_rates: float):
        Network_name = []
        if networks:
            for name, network in networks.items():
                Network_name.append(name)
                setattr(self, name, network)  # Assign each network in the dictionary to the class instance
        self.network = Network_name

        # make sure actor and critic network
        assert hasattr(self, 'net')

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

        for network_name in networks:
            getattr(self, network_name).to(self.device)

    def get_step(self):
        return self.learning_step

    def update_setting(self, config):
        self.config.max_learning_step = config.max_learning_step
        self.config.agent_save_dir = config.agent_save_dir
        self.learning_step = 0
        save_class(self.config.agent_save_dir, 'checkpoint0', self)
        self.config.save_interval = config.save_interval
        self.cur_checkpoint = 1

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

        T = 100
        train_interval = 10
        t = 0
        dim = self.config.dim

        # init input to zeros
        input = torch.zeros((1, len(env), self.config.dim + 2), dtype = torch.float64).to(self.device)
        # input=input[None,None,:]
        y_sum = torch.zeros(len(env), dtype = torch.float64).to(self.device)
        # init h & c to zeros
        h = torch.zeros((1, len(env), self.proj_size), dtype = torch.float64).to(self.device)
        c = torch.zeros((1, len(env), self.hidden_size), dtype = torch.float64).to(self.device)
        exceed_max_ls = False
        env.reset()
        _loss = []
        while t < T:
            out, (h, c) = self.net(input, (h, c))
            # get new x
            x = out[0, :]
            # print(x)
            y, _, _, _ = env.step(x)
            y_sum += y

            # update input
            input = torch.cat((x[None, :, :], y[None, :, None], torch.ones(1, len(env), 1).to(self.device)), dim = -1).to(self.device)

            t += 1

            # update network
            if t % train_interval == 0:
                loss = torch.mean(y_sum)
                self.optimizer.zero_grad()
                loss.mean().backward()
                _loss.append(loss.mean().item())
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

                self.optimizer.step()
                y_sum = y_sum.detach()
                h = h.detach()
                c = c.detach()
                input = input.detach()
                self.learning_step += 1
                if not self.config.no_tb:
                    self.log_to_tb_train(tb_logger, self.learning_step, grad_norms, loss.mean())

                if self.learning_step >= (self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == 'step':
                    save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1
                if self.learning_step >= self.config.max_learning_step:
                    exceed_max_ls = True
                    break
                # loss.detach()
        # return exceed_max_ls
        return_info = {'return': [0] * len(env), 'loss': _loss, 'learn_steps': self.learning_step, }
        return_info['gbest'] = env.get_env_attr('cost')[-1]
        for key in required_info.keys():
            return_info[key] = env.get_env_attr(required_info[key])
        env.close()
        return exceed_max_ls, return_info

    # rollout_episode need transform
    def rollout_episode(self, env, seed = None, required_info = {}):
        env.seed(seed)

        torch.set_grad_enabled(False)
        T = 100

        dim = self.config.dim

        fes = 0

        best = None
        cost = []

        t = 0
        input = torch.zeros((self.config.dim + 2), dtype = torch.float64)
        input = input[None, None, :]

        h = torch.zeros((self.proj_size), dtype = torch.float64)[None, None, :]
        c = torch.zeros((self.hidden_size), dtype = torch.float64)[None, None, :]
        env.reset()

        y_sum = 0
        init_y = None

        while t < T:
            out, (h, c) = self.net(input, (h, c))
            x = out[0, 0]
            y, _, is_done, _ = env.step(x.detach().numpy())

            y_sum += y
            if t == 0:
                init_y = y
            fes += 1
            # print(y)
            if best is None:
                best = y
            elif y < best:
                best = y

            input = torch.cat((x, torch.tensor([y]), torch.tensor([1])))[None, None, :]
            if is_done:
                break
            t += 1

        torch.set_grad_enabled(True)

        env_cost = env.get_env_attr('cost')[::2]
        results = {'cost': env_cost, 'fes': fes, 'return': y_sum / init_y}

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
                        extra_info = {}):
        # Iterate over the extra_info dictionary and log data to tb_logger
        # extra_info: Dict[str, Dict[str, Union[List[str], List[Union[int, float]]]]] = {
        #     "loss": {"name": [], "data": [0.5]},  # No "name", logs under "loss"
        #     "accuracy": {"name": ["top1", "top5"], "data": [85.2, 92.5]},  # Logs as "accuracy/top1" and "accuracy/top5"
        #     "learning_rate": {"name": ["adam", "sgd"], "data": [0.001, 0.01]}  # Logs as "learning_rate/adam" and "learning_rate/sgd"
        # }
        #
        # train metric
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'learnrate/{network_name}', self.optimizer.param_groups[id]['lr'], mini_step)
        grad_norms, grad_norms_clipped = grad_norms
        for id, network_name in enumerate(self.network):
            tb_logger.add_scalar(f'grad/{network_name}', grad_norms[id], mini_step)
            tb_logger.add_scalar(f'grad_clipped/{network_name}', grad_norms_clipped[id], mini_step)

        tb_logger.add_scalar('loss/loss', loss.item(), mini_step)

        # extra info
        for key, value in extra_info.items():
            if not value['name']:
                tb_logger.add_scalar(f'{key}', value['data'][0], mini_step)
            else:
                name_list = value['name']
                data_list = value['data']
                for name, data in zip(name_list, data_list):
                    tb_logger.add_scalar(f'{key}/{name}', data, mini_step)