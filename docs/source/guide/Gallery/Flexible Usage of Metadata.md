# Flexible Usage of Metadata

## 1. the concept of the Metadata
<p align="center">
  <img src="../../_static/metadata.png" alt="Metabox" width="800"/>
</p>

## 2. save/load your metadata for a testing procedure
Suppose you want to  test a checkpoint of GLEET on all test problem isntances in bbob-10D-difficult, MetaBox-v2 will automatically record and organize the process data as the metadata and saves it to a file you indicate.
```python

```
Once the metada is saved, you can load it back anytime you want to do some analysis works.
```python

```

## 3. usage cases
### 3.1 draw optimization curve from metadata for "one baseline on one testsuites"
Since metadata has record all optimization episodes (per-step solution positions and objective values) for all tested problem instances and all test runs, you can draw optimization curves of the baseline under different granularities. For example:

draw optimization curve of a specific test run of a specific problem instance:
```python

```

draw normalized optimization curve across all problem instances and test runs:
```python

```


### 3.2 summarize performance comparison table from metadata for "multiple baselines on one testsuites"
suppose you have saved multiple metadata files for multiple baselines, you could load them back and summarize a performance table, where each row is a baseline's performances on each problem instance.
```python

```

### 3.3 summarize Anti-NFL performance comparison from metadata for "multiple baselines on multiple testsuites"
suppose you have saved multiple metadata files from combinations of multiple baselines and multiple testsuites, calculate high-level metrics such Anti-NFL indicator in the paper could be very easy.
```python

```

## 4. Other usages
We provide metadata to maximize the freedom of users. Based on these very basic data items from the testing procedure, we believe users could determine their customized analysis metrics. 
