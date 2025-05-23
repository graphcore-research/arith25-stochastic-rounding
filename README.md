# arith25-stochastic-rounding
Support materials for "On Stochastic Rounding with Few Random Bits", Fitzgibbon and Felix, ARITH 2025

This repository is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) with changes in order to enable quantization-aware training with various float formats and definitions of stochastic rounding from the [gfloat](https://github.com/graphcore-research/gfloat) library.

The original nanoGPT readme is in [README-nanoGPT](README-nanoGPT.md).

# Installation

First set up by following README-nanoGPT, up to the point where training fails because you haven't installed `awfutils`.

Then install the gfloat and awfutils packages:
```
pip install git+https://github.com/awf/awfutils@7e99007
pip install git+https://github.com/graphcore-research/gfloat@9c31e1d
```
There is also a "pip frozen" ``requirements-frozen.txt``, which is supplied for reference, for anyone trying to exactly duplicate the conditions of the paper, but for the most part, an up-to-date pytorch and gfloat should work.

At this point, a basic command such as
```
python train.py config/train_shakespeare_char.py
```
should run.

Then, to train with quantization to binary8p4, using stochastic rounding with three random bits (one fewer than the difference between bfloat16's precision of 8 and p4):
```
python train.py config/train_shakespeare_char.py --qat=b8p4 --qat_rnd=sr --qat_srn=3
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

And to do so including uncertainties, and bfloat16, use MkSweep:
```
python sweep.py sweeps
make -f sweeps/Makefile
```

To regenerate figure 4, run
```
python train.py config/train_gpt2.py --dtype=bfloat16 --qat=float16 --qat_rnd=tne
python train.py config/train_gpt2.py --dtype=bfloat16 --qat=b8p4 --qat_rnd=tne
python train.py config/train_gpt2.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=sr --qat_scale=0.0 --qat_start_iter=
python train.py config/train_gpt2.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=srf
python train.py config/train_gpt2.py --dtype=bfloat16 --qat=b8p4 --qat_srn=3 --qat_rnd=srff
```

