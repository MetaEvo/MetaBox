# Train

```{note}
**The following code demonstrates the core training logic.**
Numerous configurable options are available — refer to **Quickstart > [Config](https://metaboxdoc.readthedocs.io/en/stable/guide/QuickStart/Config.html#config)** for details.
```

```{important}
The following is the code specific to Linux.
If you are using Windows, please add:  \
```if __name__ == "__main__":```
```


🧪 General Training Code

```python
from metaevobox import Config, Trainer
# import meta-level agent of MetaBBO you want to meta-train
from metaevobox.baseline.metabbo import XXX
# import low-level BBO optimizer of MetaBBO you want to meta-train
from metaevobox.environment.optimizer import XXX_Optimizer
from metaevobox.environment.problem.utils import construct_problem_set

# put user-specific configuration
user_config = {'train_problem': "xxx", # specify the problem set you want to train your MetaBBO 
          'train_batch_size': 16,
          'train_parallel_mode':'subproc', # choose parallel training mode
          }
config = Config(user_config)

# construct dataset
config, datasets = construct_problem_set(config)

# initialize your MetaBBO's meta-level agent & low-level optimizer
agent = XXX(config)
optimizer = XXX_Optimizer(config)

trainer = Trainer(config, agent, optimizer, dataset)
trainer.train()
```

If you want to check out the visualized information of the training progress, run following code to start training logger.
```bash
cd your_dir/output/tensorboard
tensorboard --logdir=./
```

🎯 Example: Train GLEET on COCO's BBOB (10D, easy)

```python
from metaevobox import Config, Trainer
# import meta-level agent of MetaBBO you want to meta-train
from metaevobox.baseline.metabbo import GLEET
# import low-level BBO optimizer of MetaBBO you want to meta-train
from metaevobox.environment.optimizer import GLEET_Optimizer
from metaevobox.environment.problem.utils import construct_problem_set

# put user-specific configuration
config = {'train_problem': 'bbob-10D', # specify the problem set you want to train your MetaBBO 
          'train_batch_size': 16,
          'train_parallel_mode':'subproc', # choose parallel training mode
          }
config = Config(config)
# construct dataset
config, datasets = construct_problem_set(config)
# initialize your MetaBBO's meta-level agent & low-level optimizer
gleet = GLEET(config)
gleet_opt = GLEET_Optimizer(config)
trainer = Trainer(config, gleet, gleet_opt, datasets)
trainer.train()
```

```{tip}
**Train your algorithm on MetaBox** — refer to  **Developer Guide > [Develop your MetaBBO](https://metaboxdoc.readthedocs.io/en/stable/guide/Gallery/Develop%20your%20MetaBBO.html#develop-your-metabbo)** for details.
```
