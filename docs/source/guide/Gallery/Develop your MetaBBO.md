# Develop your MetaBBO

## MetaBBO

### 1. Create your own Agent

```{tip}
MetaBOX not only supports RL-based MetaBBO methods, \
but also supports MetaBBO methods based on SL, NE, and ICL. \
Below, we use the RL-based method as an example; \
for other methods, the overall development logic remains the same, except that you should define and develop your own internal learning logic.
```

#### 1.1. Create your own Meta-level Policy

```{important}
The Meta-level policy is a learning agent that could be either RL or other learning paradigm. \
MetaBOX has pre-implemented various RL methods — refer to [here](https://github.com/MetaEvo/MetaBox/tree/v2.0.0/src/rl) for details. \
You just need to inherit a RL class and design your own Agent — Jump directly to [Create your own Agent](#create-your-own-optimizer) ！
```


1️⃣ Import Required Packages

```python
import torch
from metaevobox.rl import basic_agent
```

2️⃣ Initialize the RL Class

```python
class MyRL(basic_agent):
     
     def __init__(self, config):
    
    # If rl contains the network
    # def __init__(self, config, networks: dict, learning_rates: float):
          super().__init___(config)
          self.config = config
          # Init parameters
          # xxx
          
          # If rl contains the network
          # self.set_network(networks, learning_rates)
          
          # Init learning time
          self.learning_time = 0
          self.cur_checkpoint = 0
          
          # Save init agent
          save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
          self.cur_checkpoint += 1
```

3️⃣ Initialize the Network (Optional)

```{note}
This function is designed for rl methods that require networks and is not necessary.
```

```python
    def set_network(self, networks: dict, learning_rates: float):
        pass
```

4️⃣ Set update rules

```{code}
    def update_setting(self, config):
        pass
```

5️⃣ The Main Function for Training 

```python
def train_episode(self, 
                 envs,
                 seeds: Optional[Union[int, List[int], np.ndarray]],
                 para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
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

   state = env.reset()
   state = torch.FloatTensor(state)

   R = torch.zeros(len(env))

   while not env.all_done():
       # Get actions based on specific methods
       # action = ....

       # State transient
       next_state, reward, is_end, info = env.step(action)
       R += reward

       # Specific operations
       # xxxx

       # Store info
       self.learning_time += 1
       if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
           save_class(self.config.agent_save_dir, 'checkpoint-'+str(self.cur_checkpoint), self)
           self.cur_checkpoint += 1

       return self.learning_time >= self.config.max_learning_step, return_info

   # Return the necessary training data
   is_train_ended = self.learning_time >= self.config.max_learning_step
   return_info = {'return': R, 'learn_steps': self.learning_time, }
   env_cost = env.get_env_attr('cost')
   return_info['gbest'] = env_cost[-1]
   for key in required_info.keys():
       return_info[key] = env.get_env_attr(required_info[key])
   env.close()
   return is_train_ended, return_info
```

6️⃣ The Main Function for Testing

```python
def rollout_episode(self, env, seed=None, required_info = {}):
   with torch.no_grad():
       if seed is not None:
           env.seed(seed)
       is_done = False
       state = env.reset()
       R = 0
       
       while not is_done:
           # Get actions based on specific methods
           # action = ....
           
           # State transient
           next_state, reward, is_end, info = env.step(action)
           R += reward
          
       # Return the necessary test data
       env_cost = env.get_env_attr('cost')
       env_fes = env.get_env_attr('fes')
       env_metadata = env.get_env_attr('metadata') 
       results = {'cost': env_cost, 'fes': env_fes, 'return': R, 'metadata': env_metadata}
       for key in required_info.keys():
           results[key] = getattr(env, required_info[key])
       return results
```
       
6️⃣ The Main Function to Record Data for Analysis

```python
def log_to_tb_train(self, tb_logger):
   # Record the training data to tensorboard
   # Exp：tb_logger.add_scalar('loss', loss.item(), mini_step)
   pass
```

```{tip}
Not familiar with tensorboard? Click this [link](https://www.tensorflow.org/tensorboard/get_started).
```

#### 1.2. Create your own Agent

```{important}
MetaBOX has pre-implemented various RL methods — refer to **Gallery > Config** for details. \
You just need to inherit it and design your own Agent ！\
Here we take the rl method inherited from MetaBOX as an example.
```

1️⃣ Inheritance and Initialization

```python
from metaevobox.rl import xxx

class MyAgent(xxx):
     def __init__(self, config):
         super().__init__(self.config):
         self.config = config

         # Init parameters
         # XXX

     def __str__(self):
         return "MyAgent"
```

2️⃣ Modify train_episode according to specific work (Optional)

```{note}
This is designed for those with special training needs, not necessary
```

```python
def train_episode(self, 
                 envs,
                 seeds: Optional[Union[int, List[int], np.ndarray]],
                 para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc']='dummy',
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

   state = env.reset()
   state = torch.FloatTensor(state)

   R = torch.zeros(len(env))

   while not env.all_done():
       # Get actions based on specific methods
       # action = ....

       # State transient
       next_state, reward, is_end, info = env.step(action)
       R += reward

       # Specific operations
       # xxxx

       # Store info
       self.learning_time += 1
       if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
           save_class(self.config.agent_save_dir, 'checkpoint-'+str(self.cur_checkpoint), self)
           self.cur_checkpoint += 1

       return self.learning_time >= self.config.max_learning_step

   # Return the necessary training data
   is_train_ended = self.learning_time >= self.config.max_learning_step
   return_info = {'return': R, 'learn_steps': self.learning_time, }
   env_cost = env.get_env_attr('cost')
   return_info['gbest'] = env_cost[-1]
   for key in required_info.keys():
       return_info[key] = env.get_env_attr(required_info[key])
   env.close()
   return is_train_ended, return_info
```

2️⃣ Modify rollout_episode according to specific work (Optional)

```{note}
This is designed for those with special test needs, not necessary
```

```python
def rollout_episode(self, env, seed=None, required_info = {}):
   with torch.no_grad():
       if seed is not None:
           env.seed(seed)
       is_done = False
       state = env.reset()
       R = 0
       
       while not is_done:
           # Get actions based on specific methods
           # action = ....
           
           # State transient
           next_state, reward, is_end, info = env.step(action)
           R += reward
          
       # Return the necessary test data
       env_cost = env.get_env_attr('cost')
       env_fes = env.get_env_attr('fes')
       env_metadata = env.get_env_attr('metadata') 
       results = {'cost': env_cost, 'fes': env_fes, 'return': R, 'metadata': env_metadata}
       for key in required_info.keys():
           results[key] = getattr(env, required_info[key])
       return results
```

### 2. Create your own Optimizer

1️⃣ Inheritance and Initialization

```python
from metaevobox.environment.optimizer import Learnale_Optimizer

class MyOptimizer(Learnale_Optimizer):
     def __init__(self, config):
         super().__init__(config)
         
         self.config = config
         self.max_fes = config.maxFEs
         self.fes = None
         self.cost = None
         self.log_index = None
         self.log_interval = config.log_interval

         # Init parameters
         # XXX

     def __str__(self):
         return "MyOptimizer"
```     

2️⃣ Initialize the population

```python
def init_population(self, problem):

    # Specific operations
    # xxxx

    if self.config.full_meta_data:
        self.meta_X = [population.copy()]
        # population is all individuals in each generation
        self.meta_Cost = [all_cost.copy()]
        # all_cost is all evaluation values in each generation

    return # According to your specific needs

```
3️⃣ The Main function for updating

```python
def update(self, action, problem):

    # Specific operations
    # xxxx

    # Record all individuals in each generation
    # and their corresponding evaluation values
    if self.full_meta_data:
        self.meta_X.append(population.copy())
        # population is all individuals in each generation
        self.meta_Cost.append(all_cost.copy())
        # all_cost is all evaluation values in each generation

    # In order to ensure that the logger data format is correct,
    # there is only one stop mechanism.
    is_end = self.fes >= self.max_fes

    if self.fes >= self.log_interval * self.log_index:
        self.log_index += 1
        self.cost.append(self.gbest_cost)
        # gbest_cost is the optimal evaluation value

    if is_end:
        if len(self.cost) >= self.config.n_logpoint + 1:
            self.cost[-1] = self.gbest_cost
        else:
            while len(self.cost) < self.config.n_logpoint + 1:
                self.cost.append(self.gbest_cost)

    info = {}
    return next_state, reward, is_end, info
```

```{important}
Since optimizer is extremely flexible, the above functions are only necessary \
and need to be adjusted appropriately **according to specific tasks**.
```

## BBO

```{tip}
Considering that you may need to compare with other bbo,\
we also open the bbo design interface to you! 😉
```

1️⃣ Inheritance and Initialization

```python
from metaevobox.environment.optimizer.basic_optimizer import Basic_Optimizer

class MyBBO(Basic_Optimizer):
    def __init__(self, config):
        super(self).__init__(config)
         
        self.config = config
        self.max_fes = config.maxFEs
        self.fes = None
        self.cost = None
        self.log_index = None
        self.log_interval = config.log_interval

        # Init parameters
        # XXX

    def __str__(self):
        return "MyBBO"
```

2️⃣ The Main Function for Testing

```python
def run_episode(self, problem):

    # Update
    is_end = False
    while not is_end:

        if self.full_meta_data:
            self.meta_X.append(population.copy())
            # population is all individuals in each generation
            self.meta_Cost.append(all_cost.copy())
            # all_cost is all evaluation values in each generation

        if self.fes >= log_index * self.log_interval:
            log_index += 1
            self.cost.append(self.gbest_cost)
            # gbest_cost is the optimal evaluation value

        is_end = self.fes >= self.config.maxFEs

     # Record
     if len(self.cost) >= self.__n_logpoint + 1:
         self.cost[-1] = self.gbest
     else:
         self.cost.append(self.gbest)
     results = {'cost': self.cost, 'fes': self.fes}

     if self.full_meta_data:
         metadata = {'X':self.meta_X, 'Cost':self.meta_Cost}
         results['metadata'] = metadata
      return results
```

```{tip}
MetaBOX not only supports RL-based MetaBBO methods, but also supports MetaBBO methods based on SL, NE, and ICL.
```

Exp:
- **MetaBBO-RL**：GLEET
- **MetaBBO-SL**：GLHF
- **MetaBBO-NE**：LES
- **MetaBBO-ICL**：OPRO

Compared to RL, other methods differ in that they do not require an `rl` class and are instead built directly by inheriting from `basic_agent`.

1️⃣ Create the Agent

```{python}
from metaevobox.rl import basic_agent
class MyAgent(basic_agent)
    # Specific Operation
```

The procedure can be referred to in [Create your own agent](#12-create-your-own-agent).

2️⃣ Create the Optimizer

The procedure can be referred to in [Create your own optimizer](#2-create-your-own-optimizer).

