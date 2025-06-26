# CEC2013LSGO
Problem Difficulty Classification
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy** | 8, 9, 13-20 | 1-7, 10-12 |
| **difficult** | 1-7, 10-12 | 8, 9, 13-20 |

*Note: When `difficulty` is 'all', both training and testing sets contain all problems (1-20).*

---

CEC2013LSGO proposes 15 large-scale benchmark problems to represent a wider range of realworld large-scale optimization problems.
- **paper**：\
  "[Benchmark functions for the CEC 2013 special session and competition on large-scale global optimization](https://al-roomi.org/multimedia/CEC_Database/CEC2015/LargeScaleGlobalOptimization/CEC2015_LargeScaleGO_TechnicalReport.pdf)." gene 7.33 (2013): 8.
- **Code Resource**： [CEC2013LSGO](https://github.com/dmolina/cec2013lsgo)
- **Problem Suite Composition**：\
  CEC2013LSGO contains four major categories of large-scale problems:
  1. Fully-separable functions (F1-F3) 
  2. Two types of partially separable functions:
     1. Partially separable functions with a set of non-separable subcomponents and one fully-separable subcomponents (F4-F7) 
     2. Partially separable functions with only a set of non-separable subcomponents and no fullyseparable subcomponent (F8-F11) 
  3. Two types of overlapping functions:
     1. Overlapping functions with conforming subcomponents (F12-F13) 
     2. Overlapping functions with conflicting subcomponents (F14)
  4. Fully-nonseparable functions (F15)

Except for F13 and F14, which are 905-dimensional, the dimensions of other problems are 1000.
