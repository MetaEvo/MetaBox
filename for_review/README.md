## We provide this for_review directory for the reviewers and future readers to reproduce the experimental procedure in [MetaBox-v2's paper]().

## 0.1 install MetaBox-v2 through the instructions below.
```shell
## create a venv
conda create -n metaevobox_env python=3.11.5 -y
conda activate metaevobox_env
## install pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
## install metabox
pip install metaevobox
```

## 0.2 download this for_review directory.

## 1. train baselines on bbob-10D testsuites, train-test split is set as "difficult".
```shell
cd for_review
python train.py
```
This might costs you 1 day to train all MetaBBO baselines. After the training, all checkpoints of all baselines are saved for subsequent experiments.

## 2. test all baselines on all selected testsuites in the paper.
```shell
python test.py
```
This might costs you 2 days to test all baselines (MetaBBO + BBO) on all tested problem instances across 51 independent runs. After testing, all "metadata" is well organized as elaborated in our paper and saved.

## 3. in-distribution test in Table 3 of the paper.
```shell
python table_3.py
```

## 4. out-of-distribution test in Figure 6 of the paper.
```shell
python figure_6.py
```

## 5. learning efficiency and Anti-NFL indicator test in Figure 7 of the paper.
```shell
python figure_7.py
```
