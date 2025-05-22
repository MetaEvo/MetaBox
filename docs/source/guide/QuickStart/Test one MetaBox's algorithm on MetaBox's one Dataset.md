# Test one MetaBox's algorithm on MetaBox's one Dataset

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
