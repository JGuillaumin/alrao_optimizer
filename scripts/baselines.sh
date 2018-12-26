#!/usr/bin/env bash

source activate dl-1.10-gpu
cd ..
devices="1"

CUDA_VISIBLE_DEVICES=$devices python train_baseline.py \
--lr 0.1 \
--momentum 0. \
--weight_decay 0. \
--output_dir "logs/baselines/sgd0.1_no_wd/"

CUDA_VISIBLE_DEVICES=$devices python train_baseline.py \
--lr 0.1 \
--momentum 0. \
--weight_decay 0.0001 \
--output_dir "logs/baselines/sgd0.1_wd0.0001/"


CUDA_VISIBLE_DEVICES=$devices python train_baseline.py \
--lr 0.1 \
--momentum 0.9 \
--weight_decay 0. \
--output_dir "logs/baselines/momentum0.1_no_wd/"

CUDA_VISIBLE_DEVICES=$devices python train_baseline.py \
--lr 0.1 \
--momentum 0.9 \
--weight_decay 0.0001 \
--output_dir "logs/baselines/momentum0.1_wd0.0001/"