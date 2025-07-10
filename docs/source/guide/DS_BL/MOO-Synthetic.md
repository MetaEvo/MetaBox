# MOO-Synthetic
Problem Difficulty Classification
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy** | First 80% of problems sorted by complexity | Last 20% of problems sorted by complexity |
| **difficult** | First 20% of problems sorted by complexity | Last 80% of problems sorted by complexity |

*Note: Problems are sorted by complexity (n_obj × n_var). When `difficulty` is 'all', both sets contain all 187 problems.*

---

MOO-Synthetic is constructed by mixing 4 well-known multi-objective problem sets: ZDT, UF, DTLZ and WFG. In total, we have constructed 187 problem instances. Their objective numbers range from 2 \~ 10, dimensions range from 6D \~ 38D.

- **Paper**：
  - [ZDT](https://ieeexplore.ieee.org/abstract/document/6787994)
  - [UF](https://www.al-roomi.org/multimedia/CEC_Database/CEC2009/MultiObjectiveEA/CEC2009_MultiObjectiveEA_TechnicalReport.pdf)
  - [DTLZ](https://ieeexplore.ieee.org/abstract/document/1007032)
  - [WFG](https://ieeexplore.ieee.org/abstract/document/1705400)   
- **Code Resource**： [MOO-Synthetic](https://github.com/anyoptimization/pymoo)

## Overview

| Problem Set | Number of Problems | Objectives | Dimension | Source |
|-------------|--------------------|------------|-----------|--------|
| ZDT         | 5                  | 2          | 10 or 30  | *Comparison of Multiobjective Evolutionary Algorithms: Empirical Results* |
| UF          | 10                 | 2 or 3     | 30        | *CEC’2009 Test Instances* |
| DTLZ        | 46                 | 2–10       | 6–29      | *Scalable MOO Test Problems* |
| WFG         | 117                | 2–10       | 12–38     | *A Review of Multiobjective Test Problems* |

## Detailed Configurations

### ZDT Problems

- ZDT1–ZDT3: 30 dimensions, 2 objectives  
- ZDT4, ZDT6: 10 dimensions, 2 objectives  
- ZDT5 (binary problem) is excluded.

### UF Problems

| Problem | Objectives | Dimension |
|---------|------------|-----------|
| UF1–UF7 | 2          | 30        |
| UF8–UF10 | 3         | 30        |

### DTLZ Problems

| Problem Type | Configurations (Objectives, Dimensions) |
|--------------|------------------------------------------|
| DTLZ1        | (2,6), (3,7), (5,9), (7,11), (8,12), (10,14) |
| DTLZ2/4/6    | (2,11), (3,11), (3,12), (5,14), (7,16), (8,17), (10,19) |
| DTLZ3/5      | (2,11), (3,12), (5,14), (7,16), (8,17), (10,19) |
| DTLZ7        | (2,21), (3,22), (5,24), (7,16), (7,26), (8,27), (10,29) |

### WFG Problems

| Objectives | Dimensions |
|------------|------------|
| 2          | 12, 22     |
| 3          | 12, 14, 24 |
| 5          | 14, 18, 28 |
| 7          | 16         |
| 8          | 24, 34     |
| 10         | 28, 38     |

## Dataset Split Strategy

To assess the generalization capabilities of algorithms, the dataset can be split based on **problem difficulty**, defined as:

```
difficulty = number_of_objectives × dimension
```

Then, sort all 178 problems by this value:

- **Easy Mode**:
  - Training set: first 80% (142 problems)
  - Test set: remaining 20% (36 problems)

- **Difficult Mode**:
  - Training set: first 20% (36 problems)
  - Test set: remaining 80% (142 problems)

## Notes

If you would like, we can also provide:
- Sample loading scripts (e.g., Python)
- Visualization tools for Pareto fronts
- JSON/CSV versions of the benchmark metadata
