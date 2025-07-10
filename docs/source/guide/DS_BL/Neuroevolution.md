# Neuroevolution
Problem Difficulty Classification
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy** | Deep networks (depth > 2) | Shallow networks (depth ≤ 2) |
| **difficult** | Shallow networks (depth ≤ 2) | Deep networks (depth > 2) |

*Note: Total 66 networks available. When `difficulty` is 'all', both sets contain all networks.*

---

This problem set is based on the neuroevolution interfaces in <a href="https://evox.readthedocs.io/en/latest/examples/brax.html">EvoX</a>. The goal is to optimize the parameters of neural network-based RL agents for a series of Robotic Control tasks. We pre-define 11 control tasks (e.g., swimmer, ant, walker2D etc.), and 6 MLP structures with 0~5 hidden layers. The combinations of task & network structure result in 66 problem instances, which feature extremely high-dimensional problems (>=1000D).

- **Paper**："[EvoX: A distributed GPU-accelerated framework for scalable evolutionary computation.](https://ieeexplore.ieee.org/abstract/document/10499977)" IEEE Transactions on Evolutionary Computation (2024).
- **Code Resource**： [NE](https://github.com/EMI-Group/evox)
