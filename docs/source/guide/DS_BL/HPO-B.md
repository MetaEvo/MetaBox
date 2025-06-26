# HPO-B
Problem Difficulty Classification

By default, the dataset is split into a fixed training set and a testing set.
| Set Type | Source Data | Number of Problems |
|--------------|----------|---------------|
| **Training Set** | `meta_train_data` | 758 |
| **Testing Set** | `meta_vali_data` + `meta_test_data` | 177 |

*Note: If `difficulty` is set to 'all', the training and testing sets are merged, containing all 935 problems.*

---

HPO-B is an autoML hyper-parameter optimization benchmark which includes a wide range of hyperparameter optimization tasks for 16 different model types (e.g., SVM, XGBoost, etc.), resulting in a total of 935 problem instances. The dimension of these problem instances range from 2 to 16. We also note that HPO-B represents problems with ill-conditioned landscape such as huge flattern.

- **Paper**："[Hpo-b: A large-scale reproducible benchmark for black-box hpo based on openml.](https://arxiv.org/pdf/2106.06257)" arXiv preprint arXiv:2106.06257 (2021).
- **Code Resource**： [HPO-B](https://github.com/machinelearningnuremberg/HPO-B)
