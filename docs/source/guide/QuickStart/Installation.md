# Installation

```{important}
Below we install a cpu-version torch for you, if you need install any other versions, \
see [torch](https://pytorch.org/get-started) and replace the corresponding installation instruction below.
```

```bash
## create a venv
conda create -n metaevobox_env python=3.11.5 -y
conda activate metaevobox_env
## install pytorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
## install metabox
pip install metaevobox
```
