# CEC2013MMO
Problem Difficulty Classification
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy** | 8, 9, 13-20 | 1-7, 10-12 |
| **difficult** | 1-7, 10-12 | 8, 9, 13-20 |

*Note: When `difficulty` is 'all', both training and testing sets contain all problems (1-20).*

---

CEC2013MMO is based on CEC2013LSGO benchmark and specially crafeted for multi-modal optimization, which includes 20 synthetic problem instances covering various dimensions (1D~20D), each with varied number of (1 ~ 216) global optima. Among them, F1 to F5 are simple uni-modal functions, F6 to F10 are dimension-scalable functions with multiple global optima, and F11 to F20 are complex composition functions with challenging landscapes.

- Paper："[Benchmark functions for CEC’2013 special session and competition on niching methods for multimodal function optimization.](https://web.xidian.edu.cn/xlwang/files/20150312_175833.pdf)" RMIT University, evolutionary computation and machine learning Group, Australia, Tech. Rep (2013).
- Code Resource： [CEC2013MMO](https://github.com/mikeagn/CEC2013)

## Characteristics

- Original tasks are maximization; **converted to minimization** via sign flip.
- Covers 1D to 20D settings with varying numbers of optima.

## Function Summary

| ID | Function Name               | Dim | Global Optima | Local Optima | Range               |
|----|-----------------------------|-----|----------------|--------------|---------------------|
| F1 | Five-Uneven-Peak Trap       | 1   | 2              | 3            | [0, 30]             |
| F2 | Equal Maxima                | 1   | 5              | 0            | [0, 1]              |
| F3 | Uneven Decreasing Maxima   | 1   | 1              | 4            | [0, 1]              |
| F4 | Himmelblau                  | 2   | 4              | 0            | [-6, 6]             |
| F5 | Six-Hump Camel Back         | 2   | 2              | 2            | [-1.1, 1.1]         |
| F6 | Shubert                     | D   | 3^D            | many         | [-10, 10]^D         |
| F7 | Vincent                     | D   | 6^D            | 0            | [0.25, 10]^D       |
| F8 | Modified Rastrigin          | D   | $$\prod_{i = 1}^{D} k_i$$         | 0            | [0, 1]^D            |
| F9–F12 | Composition Functions   | 2–20| 6–8            | complex      | [-5, 5]^D          |

## Dataset Setup

The parameter settings used for each problem are as follows. **Note again** that here we reformulate the origin problems as minimization problems in the dataset setting by applying negative signs to the evaluation results of the original functions.

|Problem id | Function      | r | Peaek height | No. global optima|
|----------- | ----------- |---| ------------------- | ----------- |
|P1      | F1 (1D)  | 0.01 |-200.0 |2|
|P2      | F2 (1D) | 0.01| -1.0 |5|
|P3      | F3 (1D)| 0.01 |-1.0 |1|
|P4      | F4 (2D) | 0.01| -200.0| 4|
|P5      | F5 (2D) | 0.5 |-1.031628453| 2|
|P6      | F6 (2D) | 0.5| -186.7309088 |18|
|P7      | F7 (2D)| 0.2| -1.0| 36|
|P8      | F6 (3D) | 0.5 |-2709.093505| 81|
|P9      | F7 (3D)| 0.2 |-1.0| 216|
|P10     | F8 (2D) | 0.01| 2.0| 12|
|P11     | F9 (2D)| 0.01| 0| 6|
|P12     | F10 (2D)| 0.01 |0| 8|
|P13    | F11 (2D) | 0.01| 0| 6|
|P14    | F11 (3D)| 0.01| 0| 6|
|P15    | F12 (3D)| 0.01| 0| 8|
|P16    | F11 (5D)| 0.01| 0 |6|
|P17    | F12 (5D) | 0.01| 0| 8|
|P18    | F11 (10D) | 0.01 |0| 6|
|P19    | F12 (10D)|0.01| 0 |8|
|P20    | F12 (20D) |0.01 |0| 8|

## Evaluation Metrics

- **Peak Ratio (PR)**: Measures average % of known optima found.
- **Success Rate (SR)**: % of runs that find *all* global optima.
- Both metrics use ε = 1e-4 as the primary accuracy threshold.

## Max Function Evaluations (MaxFEs)

| Function Range        | MaxFEs  |
|-----------------------|---------|
| F1–F5 (1D/2D)         | 5e4     |
| F6–F11 (2D)           | 2e5     |
| F6–F12 (≥3D)          | 4e5     |

## Train-Test Split

Based on difficulty from empirical studies:

- **Easy Problems**: P1–P7, P10–P12
- **Difficult Problems**: P8–P9, P13–P20

| Mode     | Train Set     | Test Set      |
|----------|---------------|---------------|
| Easy     | Difficult Set | Easy Set      |
| Difficult| Easy Set      | Difficult Set |
