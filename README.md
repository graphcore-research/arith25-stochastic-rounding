# arith25-stochastic-rounding
Support materials for "On Stochastic Rounding with Few Random Bits", Fitzgibbon and Felix, ARITH 2025

This repository is a fork of [nanoGPT](https://github.com/karpathy/nanoGPT) with changes in order to enable quantization-aware training with various float formats and definitions of stochastic rounding from the [gfloat](https://github.com/graphcore-research/gfloat) library.

The original nanoGPT readme is in [README-nanoGPT](README-nanoGPT.md).

# Running

Set up following README-nanoGPT, so that a basic command such as
```
python train.py config/train_shakespeare_char.py
```
runs, and reduces loss from around 4.3 to say 2.

Then, to train with quantization to binary8p4, using stochastic rounding:
```
python train.py config/train_shakespeare_char.py --qat=b8p4 --qat_rnd=sr
```
