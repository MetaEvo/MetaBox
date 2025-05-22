# Common Usage
```{toctree}

:maxdepth: 1
Train one MetaBox's algorithm on MetaBox's one Dataset
Test one MetaBox's algorithm on MetaBox's one Dataset

```

## 1. Train one MetaBox's algorithm on MetaBox's one Dataset

```{note}
**The following code demonstrates the core training logic.**
Numerous configurable options are available — refer to **Gallery > Config** for details.
```

<!-- ```{note} Notes require **no** arguments, so content can start here.
```
```{tip} Notes require **no** arguments, so content can start here.
```
```{warning} Notes require **no** arguments, so content can start here.
```
:::{note}
This text is **standard** _Markdown_
:::
:::{warning}
This text is **standard** _Markdown_
:::
```{admonition} Here's my title
:class: note

Here's my admonition content

``` -->

🧪 General Training Code

```python
from metaevobox import Trainer, Config
from metaevobox.baseline.metabbo import XXX
from metaevobox.baseline.metabbo import XXX_Optimizer
from metaevobox.environment.problem.utils import construct_problem_set

user_config = {"train_problem": "xxx",
                   "train_difficulty": "xxx"
                   }
config = Config(user_config)
config, dataset = construct_problem_set(config)

agent = XXX(config)
optimizer = XXX_Optimizer(config)

trainer = Trainer(config, agent, optimizer, dataset)
trainer.train()
```

🎯 Example: Train GLEET on COCO's BBOB (10D, easy)

```python
from metaevobox import Trainer, Config
from metaevobox.baseline.metabbo import GLEET
from metaevobox.baseline.metabbo import GLEET_Optimizer
from metaevobox.environment.problem.utils import construct_problem_set

user_config = {"train_problem": "bbob-10D",
               "train_difficulty": "easy"
               }
config = Config(user_config)
config, dataset = construct_problem_set(config)

agent = GLEET(config)
optimizer = GLEET_Optimizer(config)

trainer = Trainer(user_config, agent, optimizer, dataset)
trainer.train()
```

```{tip}
**Train your algorithm on MetaBox** — refer to  **Gallery > Config** for details.
```

## 2. Test one MetaBox's algorithm on MetaBox's one Dataset

<!-- > [!NOTE]
> **The following code demonstrates the core test logic.**
> Numerous configurable options are available — refer to **Gallery > Config** for details. -->

```{note} **The following code demonstrates the core test logic.**
Numerous configurable options are available — refer to **Gallery > Config** for details.
```

🧪 General Tester Code

```python
import pickle
from metaevobox import Tester, Config
from metaevobox.environment.problem.utils import construct_problem_set
user_config = {"test_problem": "xxx",
               "test_difficulty": "xxx"
               }
config = Config(user_config)
config, dataset = construct_problem_set(config)

dir = "xxx"
with open(dir, 'rb') as f:
     agent = pickle.load(f)
opt = XXX_Optimizer(config)

tester = Tester(config, user_agents: [agent], user_loptimizers: [opt], user_datasets = dataset)
tester.test()
```

🎯 Example: Test GLEET and CMAES on COCO's BBOB (10D, easy)

Assume the GLEET agent is saved in "agent_model/train/GLEET/20250426T113530_bbob-10D_easy/checkpoint-0.pk1"

```python
from metaevobox import Tester, Config
from metaevobox.environment.problem.utils import construct_problem_set
from metaevobox.environment.optimizer import GLEET_Optimizer
from metaevobox.bbo import CMAES

user_config = {"train_problem": "bbob-10D",
                "train_difficulty": "easy",
                }
config = Config(user_config)
config, dataset = construct_problem_set(config)

dir = "agent_model/train/GLEET/20250426T113530_bbob-10D_easy/checkpoint-0.pk1"
with open(dir, 'rb') as f:
     agent = pickle.load(f)
opt = GLEET_Optimizer(config)

tester = Tester(config, user_agents: [agent], user_loptimizers: [opt], user_toprimizers：[CMAES], user_datasets = dataset)
tester.test()
```

```{tip} **Test your algorithm on MetaBox** — refer to  **Gallery > Config** for details.\
**Test two or more algorithms** — refer to  **Gallery > Config** for details.
```
