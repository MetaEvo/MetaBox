<div align="center">
<img src="https://github.com/MetaEvo/MetaBox/blob/v2.0.0/docs/source/_static/MetaBOX-title.png" width="50%">
</div>
<h2 align="center">
  <div style="font-size: 0.9em; margin-top: 12px">
    Benchmarking Meta-Black-Box Optimization under<br/>
    Diverse Optimization Scenarios with Efficiency and Flexibility
  </div>
</h2>

[![NeurIPS](https://img.shields.io/badge/NeurIPS-2023-b31b1b.svg)]([https://proceedings.neurips.cc/paper_files/paper/2023/hash/232eee8ef411a0a316efa298d7be3c2b-Abstract-Datasets_and_Benchmarks.html]) **MetaBox-v1 has been accepted as an oral presentation at NeurIPS 2023!**

😀[Online Documentation](https://metaboxdoc.readthedocs.io/en/latest/index.html) is here, you can get started quickly！😀

<div align="center">
<img src="https://github.com/MetaEvo/MetaBox/blob/v2.0.0/docs/source/_static/MetaBOX-features.png" width="99%">
</div>

we propose MetaBox 2.0 version (MetaBox-v2) as a major upgradation of [MetaBox-v1](https://github.com/MetaEvo/MetaBox/tree/v1.0.0). MetaBox-v2 now supports plentiful optimization scenarios to embrace users from single-objective optimization, multi-objective optimization, multi-modal optimization, multi-task optimization and etc. Correspondingly, **18 optimization problem sets** (synthetic + realistic), **1900+ problem instances** and **36 baseline methods** (traditional optimizers + up-to-date MetaBBOs) are reproduced within MetaBox-v2 to assist various research ideas and comprehensive comparison. To address MetaBBO's inherent efficiency issue, we have optimized low-level implementation of MetaBox-v2 to support parallel meta-training and evaluation, which reduces the running cost from days to hours. More importantly, we have optimized MetaBox-v2's sourcecode to support **sufficient development flexbility**, with clear and sound tutotials correspondingly. Enjoy your journey of learning and using MetaBBO from here!   


## Quick Start
### Installation

> [!Important]
> Below we install a cpu-version torch for you, if you need install any other versions, \
> see [torch](https://pytorch.org/get-started) and replace the corresponding installation instruction below.

```bash
## create a venv
conda create -n metaevobox_env python=3.11.5 -y
conda activate metaevobox_env
## install pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
## install metabox
pip install metaevobox
```
### Common Usage

> [!Important]
> The following is the code specific to Linux.
> If you are using Windows, please add: ```if __name__ == "__main__":```

#### Train a MetaBBO baseline
create your_dir, then create a your_train.py file in your_dir, write following codes into your_train.py.
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
If you want to check out the visualized information of the training progress, run following code to start training logger.
```bash
cd your_dir/output/tensorboard
tensorboard --logdir=./
```

#### Test BBO/MetaBBO baselines
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
        'CMAES':{'optimizer':CMAES},
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
By default, MetaBox would automatically generate various visualized experimental results in your_dir/output/test/, enjoy these useful analysis results!

### High-level Development Usage
We sincerely suggest researchers with interests to check out **[Online Documentation](https://metaboxdoc.readthedocs.io/en/latest/index.html)** for further flexible usege of MetaBox-v2, such as implementing your own MetaBBO, customized experimental design & analysis, using pre-collected metadata and seamless API calling with other famous optimization repos.


## Available Optimization Problem Set in MetaBox

<table>
  <thead>
    <tr>
      <th rowspan="2" align="center">Type</th> <!-- Center the Type column -->
      <th colspan="3" align="center">Problem Set</th> <!-- Center the Problem Set columns -->
      <th rowspan="2" align="center">Description</th> <!-- Center the Description column -->
    </tr>
    <tr>
      <th align="center">Name</th>
      <th align="center">Paper</th>
      <th align="center">Code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7" align="center">Single-Objective Optimization</td> <!-- Center the Type column -->
      <td align="center">bbob</td>
      <td><a href="https://arxiv.org/pdf/1603.08785">Paper</a></td>
      <td><a href="https://github.com/numbbo/coco">Code</a></td>
      <td>bbob is based on CoCo platform, which includes 96 representative single-objective synthetic problem instances. These instances all originate from the same group of 24 objective functions (CoCo-BBOB), which have been used in many papers and widely accepted as golden standard for evaluating the robustbess of an optimizer. In MetaBox-v2, bbob includes 4 subsets: bbob-10D, bbob-30D, bbob-noisy-10D and bbob-noisy-30D, each of them contains the 24 functions. "noisy" here indicates that the function's objective value is added with a gaussian noise before it is output, which significantly increase the solving difficulty. </td>
    </tr>
    <tr>
      <td align="center">bbob-surrogate</td>
      <td><a href="https://arxiv.org/abs/2503.18060">Paper</a></td>
      <td><a href="https://github.com/GMC-DRL/Surr-RLDE">Code</a></td>
      <td>bbob-surrogate includes 72 problem instances, each of which is a surrogate model. In specific, it can be divided into 3 subsets: bbob-surrogate-2D, bbob-surrogate-5D and bbob-surrogate-10D, each of which corresponds to 24 bbob problems. We first train KAN or MLP networks to fit 24 black box functions from bbob, then use the one with more accuracy as the surrogate model. This set is mainly developed for users who aims at exploring the potential of surrogate model in MetaBBO.</td>
    </tr>
    <td align="center">hpo-b</td>
      <td><a href="https://arxiv.org/pdf/2106.06257">Paper</a></td>
      <td><a href="https://github.com/machinelearningnuremberg/HPO-B">Code</a></td>
      <td>hpo-b is an autoML hyper-parameter optimization benchmark which includes a wide range of hyperparameter optimization tasks for 16 different model types (e.g., SVM, XGBoost, etc.), resulting in a total of 935 problem instances. The dimension of these problem instances range from 2 to 16. We also note that HPO-B represents problems with ill-conditioned landscape such as huge flattern.</td>
    </tr>
    <tr>
      <td align="center">uav</td>
      <td><a href="https://arxiv.org/abs/2501.14503">Paper</a></td>
      <td><a href="https://zenodo.org/records/12793991">Code</a></td>
      <td> uav provides 56 terrain-based landscapes as realistic Unmanned Aerial Vehicle(UAV) path planning problems, each of which is 30D. The objective is to select given number of path nodes (x,y,z coordinates) from the 3D space, so the the UAV could fly as shortly as possible in a collision-free way.  </td>
    </tr>
    <tr>
      <td align="center">ne<br>(large-scale)</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10499977">Paper</a></td>
      <td><a href="https://github.com/EMI-Group/evox">Code</a></td>
      <td>This problem set is based on the neuroevolution interfaces in <a href="https://evox.readthedocs.io/en/latest/examples/brax.html">EvoX</a>. The goal is to optimize the parameters of neural network-based RL agents for a series of Robotic Control tasks. We pre-define 11 control tasks (e.g., swimmer, ant, walker2D etc.), and 6 MLP structures with 0~5 hidden layers. The combinations of task & network structure result in 66 problem instances, which feature extremely high-dimensional problems (>=1000D).</td>
    </tr>
    <tr>
      <td align="center">protein</td>
      <td><a href="https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22830">Paper</a></td>
      <td><a href="https://zlab.wenglab.org/benchmark/">Code</a></td>
      <td>protein-docking benchmark, where the objective is to minimize the Gibbs free energy resulting from protein-protein interaction between a given complex and any other conformation. We select 28 protein complexes and randomly initialize 10 starting points for each complex, resulting in 280 problem instances. To simplify the problem structure, we only optimize 12 interaction points in a complex instance (12D problem).</td>
    </tr>
    <tr>
      <td align="center">lsgo<br>(large-scale)</td>
      <td><a href="https://al-roomi.org/multimedia/CEC_Database/CEC2015/LargeScaleGlobalOptimization/CEC2015_LargeScaleGO_TechnicalReport.pdf">Paper</a></td>
      <td><a href="https://github.com/dmolina/cec2013lsgo">Code</a></td>
      <td>
        lsgo contains 20 large-scale problems instances (>=905D. <=1000D):
        <br>
        <ol>
          <li>Fully-separable functions (F1-F3)</li>
          <li>Two types of partially separable functions:
            <ol>
              <li>Partially separable functions with a set of non-separable subcomponents and one fully-separable subcomponents (F4-F7)</li>
              <li>Partially separable functions with only a set of non-separable subcomponents and no fully-separable subcomponent (F8-F11)</li>
            </ol>
          </li>
          <li>Two types of overlapping functions:
            <ol>
              <li>Overlapping functions with conforming subcomponents (F12-F13)</li>
              <li>Overlapping functions with conflicting subcomponents (F14)</li>
            </ol>
          </li>
          <li>Fully-nonseparable functions (F15)</li>
        </ol>
      </td>
    </tr>
    <tr>
      <td rowspan="2" align="center">Multi-Objective Optimization</td> <!-- Center the Type column -->
      <td align="center">moo-synthetic</td>
      <td>
        <a href="https://ieeexplore.ieee.org/abstract/document/6787994">ZDT</a><br>
        <a href="https://www.al-roomi.org/multimedia/CEC_Database/CEC2009/MultiObjectiveEA/CEC2009_MultiObjectiveEA_TechnicalReport.pdf">UF</a><br>
        <a href="https://ieeexplore.ieee.org/abstract/document/1007032">DTLZ</a><br>
        <a href="https://ieeexplore.ieee.org/abstract/document/1705400">WFG</a>
      </td>
      <td><a href="https://github.com/anyoptimization/pymoo">Code</a></td>
      <td> moo-synthetic is constructed by mixing 4 well-known multi-objective problem sets: ZDT, UF, DTLZ and WFG. In total, we have constructed 187 problem instances. Their objective numbers range from 2~10, dimensions range from 6D~38D. </td>
   </tr> 
   <tr>
      <td align="center">moo-uav</td>
      <td>
        <a href="https://ieeexplore.ieee.org/abstract/document/6787994">paper</a><br>
      </td>
      <td><a href="https://github.com/anyoptimization/pymoo">Code</a></td>
      <td> We decompose the objective value of instances in uav into 5 separate objectives, which results in 56 30D realistic 5-objective problem instances. </td>
    </tr>
    <tr>
      <td rowspan="1" align="center">Multi-Model Optimization</td> <!-- Center the Type column -->
      <td align="center">mmo</td>
      <td><a href="https://web.xidian.edu.cn/xlwang/files/20150312_175833.pdf">Paper</a></td>
      <td><a href="https://github.com/mikeagn/CEC2013">Code</a></td>
      <td> mmo is based on CEC2013LSGO benchmark and specially crafeted for multi-modal optimization, which includes 20 synthetic problem instances covering various dimensions (1D~20D), each with varied number of (1 ~ 216) global optima. Among them, F1 to F5 are simple uni-modal functions, F6 to F10 are dimension-scalable functions with multiple global optima, and F11 to F20 are complex composition functions with challenging landscapes.</td>
    </tr>
    <tr>
      <td rowspan="3" align="center">Multi-Task Optimization</td> <!-- Center the Type column -->
      <td align="center">cec2017mto</td>
      <td><a href="https://arxiv.org/abs/1706.03470">Paper</a></td>
      <td><a href="http://www.bdsc.site/websites/MTO/index.html">Code</a></td>
      <td> cec2017mto comprises 9 multi-task problem instances, each of which contains two basic problems. Optional basic problems include Shpere, Rosenbrock, Ackley, Rastrigin, Griewank, Weierstrass and Schwefel, with dimension ranging from 25D~50D. </td>
    </tr>
    <tr>
      <td align="center">wcci2020</td>
      <td><a href="http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html">Paper</a></td>
      <td><a href="http://www.bdsc.site/websites/MTO/index.html">Code</a></td>
      <td> wcci2020 comprises 10 multi-task problem instances, each of which contains 50 basic problems. Optional basic problems include Shpere, Rosenbrock, Ackley, Rastrigin, Griewank, Weierstrass and Schwefel, which are all 50D. </td>
    </tr>
    <tr>
      <td align="center">augmented-wcci2020</td>
      <td><a href="http://www.bdsc.site/websites/MTO_competition_2020/MTO_Competition_WCCI_2020.html">Paper</a></td>
      <td><a href="http://www.bdsc.site/websites/MTO/index.html">Code</a></td>
      <td> augmented-wcci2020 comprises 127 multi-task problems, each of which optinally contains 1~7 basic problems. Optional basic problems include Shpere, Rosenbrock, Ackley, Rastrigin, Griewank, Weierstrass and Schwefel, which are all 50D. </td>
    </tr>
  </tbody>
</table>


## Available BBO/MetaBBO Baselines in MetaBox

|Baseline Name|Target Optimization Scenario|Type|Paper|Year|
|---|---|---|---|---|
|[Random_search](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/random_search.py)|||||
|[PSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/pso.py)|Single-Objective Optimization|BBO|[Particle swarm optimization](https://ieeexplore.ieee.org/abstract/document/488968)|1995|
|[DE](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/de.py)|Single-Objective Optimization|BBO|[Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces](https://dl.acm.org/doi/abs/10.1023/A%3A1008202821328)|1997|
|[CMAES](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/cmaes.py)|Single-Objective Optimization|BBO|[Completely Derandomized Self-Adaptation in Evolution Strategies](https://ieeexplore.ieee.org/document/6790628)|2001|
|[SHADE](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/shade.py)|Single-Objective Optimization|BBO|[Success-history based parameter adaptation for differential evolution](https://ieeexplore.ieee.org/document/6557555)|2013|
|[GLPSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/glpso.py)|Single-Objective Optimization|BBO|[Genetic Learning Particle Swarm Optimization](https://ieeexplore.ieee.org/abstract/document/7271066/)|2015|
|[SDMSPSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/sdmspso.py)|Single-Objective Optimization|BBO|[A Self-adaptive Dynamic Particle Swarm Optimizer](https://ieeexplore.ieee.org/document/7257290)|2015|
|[SAHLPSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/sahlpso.py)|Single-Objective Optimization|BBO|[Self-Adaptive two roles hybrid learning strategies-based particle swarm optimization](https://www.sciencedirect.com/science/article/pii/S0020025521006988)|2021|
|[JDE21](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/sahlpso.py)|Single-Objective Optimization|BBO|[Self-adaptive Differential Evolution Algorithm with Population Size Reduction for Single Objective Bound-Constrained Optimization: Algorithm j21](https://ieeexplore.ieee.org/document/9504782)|2021|
|[MADDE](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/madde.py)|Single-Objective Optimization|BBO|[Improving Differential Evolution through Bayesian Hyperparameter Optimization](https://ieeexplore.ieee.org/document/9504792)|2021|
|[NLSHADELBC](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/nlshadelbc.py)|Single-Objective Optimization|BBO|[NL-SHADE-LBC algorithm with linear parameter adaptation bias change for CEC 2022 Numerical Optimization](https://ieeexplore.ieee.org/abstract/document/9870295)|2022|
|[MOEAD](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/moead.py)|Multi-Objective Optimization|BBO|[MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition](https://ieeexplore.ieee.org/document/4358754)|2007|
|[MFEA](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/mfea.py)|Multi-Task Optimization|BBO|[Multifactorial Evolution: Toward Evolutionary Multitasking](https://ieeexplore.ieee.org/abstract/document/7161358)|2016|
|[RNNOPT](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/rnnopt.py)|Single-Objective Optimization|MetaBBO|[Learning to learn without gradient descent by gradient descent](https://dl.acm.org/doi/10.5555/3305381.3305459)|2017|
|[QLPSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/qlpso.py)|Single-Objective Optimization|MetaBBO|[A reinforcement learning-based communication topology in particle swarm optimization](https://link.springer.com/article/10.1007/s00521-019-04527-9)|2019|
|[DEDDQN](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/deddqn.py)|Single-Objective Optimization|MetaBBO|[Deep reinforcement learning based parameter control in differential evolution](https://dl.acm.org/doi/10.1145/3321707.3321813)|2019|
|[DEDQN](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/dedqn.py)|Single-Objective Optimization|MetaBBO|[Differential evolution with mixed mutation strategy based on deep reinforcement learning](https://www.sciencedirect.com/science/article/pii/S1568494621005998)|2021|
|[LDE](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/lde.py)|Single-Objective Optimization|MetaBBO|[Learning Adaptive Differential Evolution Algorithm From Optimization Experiences by Policy Gradient](https://ieeexplore.ieee.org/document/9359652)|2021|
|[RLPSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/rlpso.py)|Single-Objective Optimization|MetaBBO|[Employing reinforcement learning to enhance particle swarm optimization methods](https://www.tandfonline.com/doi/full/10.1080/0305215X.2020.1867120)|2021|
|[RLEPSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/rlepso.py)|Single-Objective Optimization|MetaBBO|[RLEPSO:Reinforcement learning based Ensemble particle swarm optimizer](https://dl.acm.org/doi/abs/10.1145/3508546.3508599)|2022|
|[RLHPSDE](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/rlhpsde.py)|Single-Objective Optimization|MetaBBO|[Differential evolution with hybrid parameters and mutation strategies based on reinforcement learning](https://www.sciencedirect.com/science/article/pii/S2210650222001602)|2022|
|[NRLPSO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/nrlpso.py)|Single-Objective Optimization|MetaBBO|[Reinforcement learning-based particle swarm optimization with neighborhood differential mutation strategy](https://www.sciencedirect.com/science/article/abs/pii/S2210650223000482)|2023|
|[OPRO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/opro.py)|Single-Objective Optimization|MetaBBO|[Large Language Models as Optimizers](https://arxiv.org/abs/2309.03409)|2024|
|[RLDAS](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/rldas.py)|Single-Objective Optimization|MetaBBO|[Deep Reinforcement Learning for Dynamic Algorithm Selection: A Proof-of-Principle Study on Differential Evolution](https://ieeexplore.ieee.org/abstract/document/10496708)|2024|
|[SYMBOL](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/symbol.py)|Single-Objective Optimization|MetaBBO|[SYMBOL: Generating Flexible Black-Box Optimizers through Symbolic Equation Learning](https://openreview.net/forum?id=vLJcd43U7a)|2024|
|[GLEET](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/gleet.py)|Single-Objective Optimization|MetaBBO|[Auto-configuring Exploration-Exploitation Tradeoff in Evolutionary Computation via Deep Reinforcement Learning](https://arxiv.org/abs/2404.08239)|2024|
|[RLDEAFL](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/rldeafl.py)|Single-Objective Optimization|MetaBBO|[Reinforcement Learning-based Self-adaptive Differential Evolution through Automated Landscape Feature Learning](https://arxiv.org/abs/2503.18061)|2025|
|[Surr_RLDE](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/surrrlde.py)|Single-Objective Optimization|MetaBBO|[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)|2025|
|[MADAC](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/madac.py)|Multi-Objective Optimization|MetaBBO|[Multi-agent Dynamic Algorithm Configuration](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7f02b39c0424cc4a422994289ca03e46-Abstract-Conference.html)|2022|
|[LGA](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/lga.py)|Large Scale Global Optimization|MetaBBO|[Discovering Attention-Based Genetic Algorithms via Meta-Black-Box Optimization](https://dl.acm.org/doi/abs/10.1145/3583131.3590496)|2023|
|[LES](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/les.py)|Large Scale Global Optimization|MetaBBO|[Discovering evolution strategies via meta-black-box optimization](https://iclr.cc/virtual/2023/poster/11005)|2023|
|[GLHF](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/glhf.py)|Large Scale Global Optimization|MetaBBO|[Pretrained Optimization Model for Zero-Shot Black Box Optimization](https://link.springer.com/article/10.1007/s00521-019-04527-9)|2024|
|[B2OPT](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/b2opt.py)|Large Scale Global Optimization|MetaBBO|[B2Opt: Learning to Optimize Black-box Optimization with Little Budget](https://ojs.aaai.org/index.php/AAAI/article/view/34036)|2025|
|[PSORLNS](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/psorlns.py)|Multi-Modal Optimization|MetaBBO|[A reinforcement learning-based neighborhood search operator for multi-modal optimization and its applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417424000150)|2024|
|[RLEMMO](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/rlemmo.py)|Multi-Modal Optimization|MetaBBO|[RLEMMO: Evolutionary Multimodal Optimization Assisted By Deep Reinforcement Learning](https://dl.acm.org/doi/abs/10.1145/3638529.3653995)|2024|
|[L2T](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/metabbo/l2t.py)|Multi-Task Optimization|MetaBBO|[Learning to Transfer for Evolutionary Multitasking](https://arxiv.org/abs/2406.14359)|2024|


## Citing MetaBox

The PDF version of the paper is available [here](https://arxiv.org/abs/2310.08252). If you find our MetaBox useful, please cite it in your publications or projects.

```latex
@inproceedings{metabox,
author={Ma, Zeyuan and Guo, Hongshu and Chen, Jiacheng and Li, Zhenrui and Peng, Guojun and Gong, Yue-Jiao and Ma, Yining and Cao, Zhiguang},
title={MetaBox: A Benchmark Platform for Meta-Black-Box Optimization with Reinforcement Learning},
booktitle = {Advances in Neural Information Processing Systems},
year={2023},
volume = {36}
}
```

## 😁Contact Us
<div align="center">
<img src="https://github.com/MetaEvo/.github/blob/main/profile/logo.png" width="20%">
</div>
👨‍💻👩‍💻We are a research team mainly focus on Meta-Black-Box-Optimization (MetaBBO)
     which assists automated algorithm design for Evolutionary Computation. 

Here is our [homepage](https://metaevo.github.io/) and [github](https://github.com/MetaEvo). **🥰🥰🥰Please feel free to contact us—any suggestions are welcome!**

If you have any question or want to contact us：
- 🌱Fork, Add, and Merge
- ❓️Report an [issue](https://github.com/MetaEvo/MetaBox/issues)
- 📧Contact WenJie Qiu ([wukongqwj@gmail.com](mailto:wukongqwj@gmail.com))
- 🚨**We warmly invite you to join our QQ group for further communication (Group Number: 952185139).**



