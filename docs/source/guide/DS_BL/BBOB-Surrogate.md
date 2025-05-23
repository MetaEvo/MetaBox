# BBOB-Surrogate

bbob-surrogate includes 72 problem instances, each of which is a surrogate model. In specific, it can be divided into 3 subsets: bbob-surrogate-2D, bbob-surrogate-5D and bbob-surrogate-10D, each of which corresponds to 24 bbob problems. We first train KAN or MLP networks to fit 24 black box functions from bbob, then use the one with more accuracy as the surrogate model. This set is mainly developed for users who aims at exploring the potential of surrogate model in MetaBBO.

- **Paper**：\
  "[Surrogate Learning in Meta-Black-Box Optimization: A Preliminary Study](https://arxiv.org/abs/2503.18060)." arXiv preprint arXiv:2503.18060 (2025).
- **Code Resource**： [BBOB-Surrogate](https://github.com/GMC-DRL/Surr-RLDE)

