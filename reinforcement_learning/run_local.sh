# Copyright (c) Facebook, Inc. and its affiliates.

#!/bin/bash

DOMAIN=cartpole
TASK=swingup

SAVEDIR=./save

CUDA_VISIBLE_DEVICES=1 python train.py \
    env=${DOMAIN}_${TASK} \
    experiment=${DOMAIN}_${TASK} \
    agent=causal \
    seed=1
