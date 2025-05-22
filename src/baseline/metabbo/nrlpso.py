from scipy.special import softmax
from typing import Optional, Union, Literal, List
from ...rl.qlearning import *
from ...rl.utils import save_class


class NRLPSO(QLearning_Agent):
    """
    # Introduction
    This paper proposes a new reinforcement learning driven particle swarm optimization algorithm, which enhances the algorithm's exploration ability and convergence by introducing a neighborhood differential mutation strategy into the PSO algorithm.
    Specifically, the algorithm uses reinforcement learning to adaptively adjust the mutation probability and mutation amplitude of particles to better balance exploration and utilization. At the same time, it adopts a neighborhood differential mutation strategy, using information within the particle neighborhood to guide the search direction of the particles, further improving the algorithm's convergence speed and solution quality.
    # Original Paper
    "[**Reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy**](https://www.sciencedirect.com/science/article/pii/S2210650223000482)." Swarm and Evolutionary Computation (2023)
    # Official Implementation
    None
    # Application Scenario
    single-object optimization problems(SOOP)
    # Args:
        `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    # Attributes:
        config (object): Stores the configuration object passed during initialization.
        device (str): Specifies the device to be used for computation ('cpu' or 'gpu').
        __alpha_max (float): Maximum learning rate for the agent.
        __max_learning_step (int): Maximum number of learning steps allowed.
        q_table (torch.Tensor): Q-table used for storing state-action values.
        learning_time (int): Counter for the number of learning steps completed.
        cur_checkpoint (int): Counter for the current checkpoint during training.
    # Methods:
        __str__():
            Returns the string representation of the class ("NRLPSO").
        __get_action(state):
            Determines the action to take based on the given state using the Q-table and softmax probabilities.
            Args:
                state (int): The current state of the environment.
            Returns:
                numpy.ndarray: The selected action(s) as a numpy array.
        train_episode(envs, seeds, para_mode='dummy', asynchronous=None, num_cpus=1, num_gpus=0, tb_logger=None, required_info={}):
            Trains the agent for one episode using the provided environment(s).
            Args:
                envs (list): List of environments for training.
                seeds (int, list, or np.ndarray): Seed(s) for environment randomization.
                para_mode (str): Parallelization mode for environments ('dummy', 'subproc', 'ray', 'ray-subproc').
                asynchronous (str or None): Asynchronous mode for environment execution ('idle', 'restart', 'continue').
                num_cpus (int): Number of CPUs to use for parallelization.
                num_gpus (int): Number of GPUs to use for computation.
                tb_logger (object): TensorBoard logger for logging training metrics.
                required_info (dict): Additional information to retrieve from the environment.
            Returns:
                tuple: A tuple containing:
                    - is_train_ended (bool): Whether the training has reached the maximum learning steps.
                    - return_info (dict): Dictionary containing training metrics such as 'return', 'loss',
                      'learn_steps', 'normalizer', 'gbest', and any additional required information.
            Raises:
                ValueError: If the environment configuration or parameters are invalid.
    """
    def __init__(self, config):

        self.config = config
        self.config.n_state = 4
        self.config.n_act = 4
        # self.__n_actions = 4
        self.config.gamma = 0.8
        self.config.lr_model = 1
        self.__alpha_max = self.config.lr_model
        # self.__q_table = np.zeros((config.n_states, config.n_actions))
        self.config.epsilon = None
        self.__max_learning_step = config.max_learning_step
        self.device = self.config.device
        # self.__global_ls = 0  # a counter of accumulated learned steps

        # self.__cur_checkpoint = 0
        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config)

    def __str__(self):
        return "NRLPSO"

    def __get_action(self, state):  # Make action decision according to the given state
        # Get the corresponding rows from the Q-table and compute the softmax
        q_values = self.q_table[state]  # shape: (bs, n_actions)

        # Compute the action probabilities for each state
        prob = torch.softmax(q_values, dim=0)  # shape: (bs, n_actions)
        # Choose an action based on the probabilities
        action = torch.multinomial(prob, 1)  # shape: (bs, 1)

        # Return the action
        return action.view(-1).detach().cpu().numpy()  # Return the action and remove unnecessary dimensions

    def train_episode(self,
                      envs,
                      seeds: Optional[Union[int, List[int], np.ndarray]],
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      num_cpus: Optional[Union[int, None]] = 1,
                      num_gpus: int = 0,
                      tb_logger=None,
                      required_info={}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        env.seed(seeds)
        # params for training
        gamma = self.gamma

        state = env.reset()
        state = torch.tensor(state, dtype=torch.int64)

        _R = torch.zeros(len(env))
        _loss = []
        _reward = []
        # sample trajectory
        while not env.all_done():
            action = self.__get_action(state)
            # state transient
            next_state, reward, is_end, info = env.step(action)
            _R += reward
            reward = torch.Tensor(reward).to(self.device)
            _reward.append(reward)
            # update Q-table
            TD_error = reward + gamma * torch.max(self.q_table[next_state], dim=1)[0].to(self.device) - self.q_table[
                state, action]

            _loss.append(TD_error.mean().item())
            self.q_table[state, action] += self.lr_model * TD_error

            self.learning_time += 1

            if self.learning_time >= (
                    self.config.save_interval * self.cur_checkpoint) and self.config.end_mode == "step":
                save_class(self.config.agent_save_dir, 'checkpoint-' + str(self.cur_checkpoint), self)
                self.cur_checkpoint += 1

            if not self.config.no_tb:
                self.log_to_tb_train(tb_logger, self.learning_time,
                                     TD_error.mean(),
                                     _R, _reward,
                                     )

            if self.learning_time >= self.config.max_learning_step:
                _Rs = _R.detach().numpy().tolist()
                return_info = {'return': _Rs, 'loss': _loss, 'learn_steps': self.learning_time, }
                env_cost = env.get_env_attr('cost')
                return_info['normalizer'] = env_cost[0]
                return_info['gbest'] = env_cost[-1]
                for key in required_info.keys():
                    return_info[key] = env.get_env_attr(required_info[key])
                env.close()
                return self.learning_time >= self.config.max_learning_step, return_info

            self.lr_model = self.__alpha_max - (self.__alpha_max - 0.1) * self.learning_time / self.__max_learning_step

            # store info
            state = torch.tensor(next_state, dtype=torch.int64)

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

