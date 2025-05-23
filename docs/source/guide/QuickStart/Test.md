# Test

```{note} **The following code demonstrates the core test logic.**
Numerous configurable options are available — refer to **Quickstart > [Config](https://metaboxdoc.readthedocs.io/en/stable/guide/QuickStart/Config.html)** for details.
```

```{important}
The following is the code specific to Linux.
If you are using Windows, please add: \
```if __name__ == "__main__":```
```

🧪 General Tester Code

```python
from metaevobox import Config, Tester, get_baseline
# import meta-level agent of MetaBBO you want to test
from metaevobox.baseline.metabbo import XXX
# import low-level BBO optimizer of MetaBBO you want to test
from metaevobox.environment.optimizer import XXX_Optimizer
# import other baselines you want to compare with your MetaBBO
from metaevobox.baseline.bbo import YYY, ZZZ
from metaevobox.environment.problem.utils import construct_problem_set

# specify your configuration
config = {
    'test_problem':'xxx', # specify the problem set you want to benchmark
    'test_batch_size':16,
    'test_difficulty':'difficult', # this is a train-test split mode
    'baselines':{
        # your MetaBBO
        'XXX':{
            'agent': 'XXX',
            'optimizer': XXX_Optimizer,
            'model_load_path': None, # by default is None, we will load a built-in pre-trained checkpoint for you.
        },

        # Other baselines to compare              
        'YYY':{'optimizer': YYY},
        'ZZZ':{'optimizer': ZZZ},
    },
}

config = Config(config)
# load test dataset
config, datasets = construct_problem_set(config)
# initialize all baselines to compare (yours + others)
baselines, config = get_baseline(config)
# initialize tester
tester = Tester(config, baselines, datasets)
# test
tester.test()
```

🎯 Example: Test GLEET and CMAES on COCO's BBOB (10D, easy)

```python
from metaevobox import Config, Tester, get_baseline
# import meta-level agent of MetaBBO you want to test
from metaevobox.baseline.metabbo import GLEET
# import low-level BBO optimizer of MetaBBO you want to test
from metaevobox.environment.optimizer import GLEET_Optimizer
# import other baselines you want to compare with your MetaBBO
from metaevobox.baseline.bbo import CMAES, SHADE
from metaevobox.environment.problem.utils import construct_problem_set

# specify your configuration
config = {
    'test_problem':'bbob-10D', # specify the problem set you want to benchmark
    'test_batch_size':16,
    'test_difficulty':'difficult', # this is a train-test split mode
    'baselines':{
        # your MetaBBO
        'GLEET':{
            'agent': 'GLEET',
            'optimizer': GLEET_Optimizer,
            'model_load_path': None, # by default is None, we will load a built-in pre-trained checkpoint for you.
        },

        # Other baselines to compare              
        'SHADE':{'optimizer': SHADE},
        'CMAES':{'optimizer': CMAES},
    },
}

config = Config(config)
# load test dataset
config, datasets = construct_problem_set(config)
# initialize all baselines to compare (yours + others)
baselines, config = get_baseline(config)
# initialize tester
tester = Tester(config, baselines, datasets)
# test
tester.test()
```

```{tip} **Test your algorithm on MetaBox** — refer to  **Developer Guide > [Develop your MetaBBO](https://metaboxdoc.readthedocs.io/en/stable/guide/Gallery/Develop%20your%20MetaBBO.html)** for details.
```
