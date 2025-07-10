# Flexibly Integrate other EC Libraries

```{tip}
There are many excellent libraries in our community that provide either testsuites or optimizer implementations. \
MetaBOX supports their flexible integration to promote the open-source ecosystem in the community. \
We provide following two examples to showcase such flexible usage, and hope you users come up with more novel integration ideas.
```
## EvoX
```{note}
[EvoX](https://github.com/EMI-Group/evox) is a distributed GPU-accelerated evolutionary computation framework compatible with PyTorch. 
```
See [here](https://github.com/MetaEvo/MetaBox/tree/v2.0.0/src/environment/problem/SOO/NE) to check how we wrap BraxProblem from EvoX into the neuroevolution problem instances. Based on the warpped problem instances, users can easily make their own testsuites in MetaBox-v2 and use the testsuites for training and testing MetaBBO baselines.

## PyPop7
```{note}
[PyPop7](https://github.com/Evolutionary-Intelligence/pypop) is a Python library of POPulation-based OPtimization for single-objective, real-parameter, unconstrained black-box problems. It maintains many BBO baselines that could be used and integrated into MetaBox-v2 for comparison and analysis usages.
```
See [here](https://github.com/MetaEvo/MetaBox/blob/v2.0.0/src/baseline/bbo/shade.py) to check how we warp a PyPop7 optimizer into the BBBO optimizer in MetaBox-v2.
