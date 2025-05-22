from scipy.special import softmax
from typing import Optional, Union, Literal, List
from ...rl.qlearning import *
from ...rl.utils import save_class


class RLHPSDE(QLearning_Agent):
    """
    # Introduction
    The paper "Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning" explores an innovative approach to enhancing the performance of Differential Evolution (DE), a widely-used optimization algorithm. The authors propose a hybrid framework that integrates reinforcement learning (RL) to dynamically adjust DE's key parameters and mutation strategies. This adaptive mechanism enables the algorithm to balance exploration and exploitation more effectively, improving its ability to solve complex optimization problems. The study demonstrates the efficacy of the proposed method through extensive experiments on benchmark functions and real-world applications, highlighting its potential to achieve superior performance compared to traditional DE variants.
    # Original Paper
    "[**Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning**](https://www.sciencedirect.com/science/article/pii/S2210650222001602)." Swarm and Evolutionary Computation (2022): 101194
    # Official Implementation
    None
    # Application Scenario
    single-object optimization problems(SOOP)
    # Args:
        `config`: Configuration object containing all necessary parameters for experiment.For details you can visit config.py.
    # Attributes:
        config (object): Configuration object passed during initialization.
        device (str): Device to be used for computation ('cpu' or 'gpu').
        __alpha_max (float): Maximum learning rate for the agent.
        __alpha_decay (bool): Flag indicating whether learning rate decay is enabled.
        __max_learning_step (int): Maximum number of learning steps allowed.
        q_table (torch.Tensor): Q-table used for storing state-action values.
        learning_time (int): Counter for the number of learning steps completed.
        cur_checkpoint (int): Current checkpoint index for saving the agent's state.
    # Methods:
        __str__():
            Returns the string representation of the class.
        __get_action(state):
            Determines the action to take based on the current state using the Q-table and softmax probabilities.
            Args:
                state (torch.Tensor): Current state of the environment.
            Returns:
                numpy.ndarray: Selected action(s) for the given state.
        train_episode(envs, seeds, para_mode='dummy', asynchronous=None, num_cpus=1, num_gpus=0, tb_logger=None, required_info={}):
            Trains the agent for one episode in the given environment(s).
            Args:
                envs (list): List of environments for training.
                seeds (int, list, or np.ndarray): Seed(s) for environment randomization.
                para_mode (str): Parallelization mode for environments ('dummy', 'subproc', 'ray', 'ray-subproc').
                asynchronous (str or None): Asynchronous mode for environment execution.
                num_cpus (int or None): Number of CPUs to use for parallelization.
                num_gpus (int): Number of GPUs to use for computation.
                tb_logger (object): TensorBoard logger for logging training metrics.
                required_info (dict): Additional information to retrieve from the environment.
            Returns:
                tuple: A tuple containing:
                    - bool: Whether the training has ended.
                    - dict: Information about the training episode, including returns, losses, and environment attributes.
    # Returns:
        __str__(): str: The string representation of the class.
        __get_action(state): numpy.ndarray: Selected action(s) for the given state.
        train_episode(): tuple: A tuple containing training status and episode information.
    # Raises:
        None
    """
    def __init__(self, config):

        self.config = config
        # define hyperparameters that agent needs
        self.config.n_state = 4
        self.config.n_act = 4
        self.config.lr_model = 0.8
        self.config.alpha_decay = False
        self.config.gamma = 0.5

        self.config.epsilon = None

        self.__alpha_max = self.config.lr_model
        self.__alpha_decay = self.config.alpha_decay
        self.__max_learning_step = self.config.max_learning_step
        self.device = self.config.device

        self.config.agent_save_dir = os.path.join(
            self.config.agent_save_dir,
            self.__str__(),
            self.config.train_name
        )
        super().__init__(self.config)

    def __str__(self):
        return "RLHPSDE"

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
            # update Q-table
            reward = torch.Tensor(reward).to(self.device)

            _reward.append(reward)

            TD_error = reward + gamma * torch.max(self.q_table[next_state], dim=1)[0] - self.q_table[state, action]

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

            if self.__alpha_decay:
                self.lr_model = self.__alpha_max - (
                            self.__alpha_max - 0.1) * self.learning_time / self.__max_learning_step

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
