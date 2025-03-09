# arith25-stochastic-rounding
Support materials for "On Stochastic Rounding with Few Random Bits", Fitzgibbon and Felix, ARITH 2025

This repository is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) with changes in order to enable quantization-aware training with various float formats and definitions of stochastic rounding from the [gfloat](https://github.com/graphcore-research/gfloat) library.

The original nanoGPT readme is in [README-nanoGPT](README-nanoGPT.md).

# Installation

First set up by following README-nanoGPT.

Then install the gfloat and awfutils packages:
```
pip install git+https://github.com/awf/awfutils@7e99007
pip install git+https://github.com/graphcore-research/gfloat@c332c01
```

At this point, a basic command such as
```
python train.py config/train_shakespeare_char.py
```
should run.

Then, to train with quantization to binary8p4, using stochastic rounding:
```
python train.py config/train_shakespeare_char.py --qat=b8p4 --qat_rnd=sr
```

# Running the paper's experiments

To regenerate figure 3, run
```
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=float16 --qat_rnd=tne
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_rnd=tne
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=sr  # Called "SRC" in the paper
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=srf
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=srff
```

To regenerate figure 4, run
```
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=float16 --qat_rnd=tne
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_rnd=tne
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=sr  # Called "SRC" in the paper
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=srf
python train.py config/train_shakespeare_char.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=srff
```

