# COCO-BBOB
Problem Difficulty Classification

**BBOB (F1-F24)**
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy**        | 4, 6-14, 18-20, 22-24 | 1, 2, 3, 5, 15, 16, 17, 21 |
| **difficult**   | 1, 2, 3, 5, 15, 16, 17, 21 | 4, 6-14, 18-20, 22-24 |

**BBOB-Noisy (F101-F130)**
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy**        | 102-104, 106-114, 118, 121-124, 126-130 | 101, 105, 115-117, 119, 120, 125 |
| **difficult**   | 101, 105, 115-117, 119, 120, 125 | 102-104, 106-114, 118, 121-124, 126-130 |

*Note: When `difficulty` is 'all', both training and testing sets contain all problems in the suite.*

---

The blackbox optimization benchmarking (bbob) test suite is COCO's standard test suite with 24 noiseless, single-objective and scalable test functions. Each function is provided in dimensions (2, 3, 5, 10, 20, 40) and in 15 instances, however also available for arbitrary dimensions and number of instances. Links to their definition with visualizations are provided in the table.

- Paper:
  - [N. Hansen et al (2010)](https://dl.acm.org/doi/abs/10.1145/1830761.1830790).
  - [Comparing Results of 31 Algorithms from the Black-Box Optimization Benchmarking BBOB-2009.](https://dl.acm.org/doi/abs/10.1145/1830761.1830790)
  - [Workshop Proceedings of the GECCO Genetic and Evolutionary Computation Conference 2010, ACM.](https://dl.acm.org/doi/abs/10.1145/1830761.1830790)
- Code Resource: [COCO](https://github.com/numbbo/coco)
  
### 1. Separable Functions

| Indax | Name |
|------|----------|
| f1   | [Sphere Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=5) |
| f2   | [Separable Ellipsoidal Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=10) |
| f3   | [Rastrigin Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=15) |
| f4   | [Büche-Rastrigin Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=20) |
| f5   | [Linear Slope](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=25) |

### 2. Functions with low or moderate conditioning

| Index | Name |
|------|----------|
| f6   | [Attractive Sector Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=30) |
| f7   | [Step Ellipsoidal Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=35) |
| f8   | [Rosenbrock Function, original](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=40) |
| f9   | [Rosenbrock Function, rotated](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=45) |

### 3. Functions with high conditioning and unimodal

| Index | Name |
|------|----------|
| f10  | [Ellipsoidal Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=50) |
| f11  | [Discus Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=55) |
| f12  | [Bent Cigar Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=60) |
| f13  | [Sharp Ridge Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=65) |
| f14  | [Different Powers Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=70) |

### 4. Multi-modal functions with adequate global structure

| Index | Name |
|------|----------|
| f15  | [Rastrigin Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=75) |
| f16  | [Weierstrass Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=80) |
| f17  | [Schaffer's F7 Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=85) |
| f18  | [Schaffer's F7 Function, moderately ill-conditioned](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=90) |
| f19  | [Composite Griewank-Rosenbrock Function F8F2](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=95) |

### 5. Multi-modal functions with weak global structure

| Index | Name |
|------|----------|
| f20  | [Schwefel Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=100) |
| f21  | [Gallagher's Gaussian 101-me Peaks Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=105) |
| f22  | [Gallagher's Gaussian 21-hi Peaks Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=110) |
| f23  | [Katsuura Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=115) |
| f24  | [Lunacek bi-Rastrigin Function](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf#page=120) |

Only f1 and f5 are purely convex quadratic (f1) or purely linear in the domain of interest [-5,5]D (f5).



