# Train

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
**Train your algorithm on MetaBox** — refer to  **QuickStart > Config** for details.
```
