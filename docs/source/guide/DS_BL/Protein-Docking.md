# Protein-Docking
Problem Difficulty Classification

The dataset is deterministically split based on a fixed random seed (1035).
| Difficulty Mode | Training Set Ratio | Testing Set Ratio |
|-----------------|--------------------|-------------------|
| **easy** | 75% | 25% |
| **difficult** | 25% | 75% |

*Note: The split is applied to each protein category ('rigid', 'medium', 'difficult') separately. When `difficulty` is 'all', both sets contain all 280 problems.*

---

Protein-Docking benchmark, where the objective is to minimize the Gibbs free energy resulting from protein-protein interaction between a given complex and any other conformation. We select 28 protein complexes and randomly initialize 10 starting points for each complex, resulting in 280 problem instances. To simplify the problem structure, we only optimize 12 interaction points in a complex instance (12D problem).

- **Paper**："[Protein–protein docking benchmark version 4.0.](https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.22830)" Proteins: Structure, Function, and Bioinformatics 78.15 (2010): 3111-3114.
- **Code Resource**： [Protein-Docking](https://zlab.wenglab.org/benchmark/)
