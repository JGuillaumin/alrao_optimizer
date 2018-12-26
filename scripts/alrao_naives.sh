#!/usr/bin/env bash

source activate dl-1.10-gpu
cd ..
devices="0"

CUDA_VISIBLE_DEVICES=$devices python train_alrao_naive.py \
--momentum 0. \
--weight_decay 0. \
--output_dir "logs/alrao_naives/sgd-alrao_no_wd/"

CUDA_VISIBLE_DEVICES=$devices python train_alrao_naive.py \
--momentum 0. \
--weight_decay 0.0001 \
--output_dir "logs/alrao_naives/sgd-alrao_wd0.0001/"


CUDA_VISIBLE_DEVICES=$devices python train_alrao_naive.py \
--momentum 0.9 \
--weight_decay 0. \
--output_dir "logs/alrao_naives/momentum-alrao_no_wd/"

CUDA_VISIBLE_DEVICES=$devices python train_alrao_naive.py \
--momentum 0.9 \
--weight_decay 0.0001 \
--output_dir "logs/alrao_naives/momentum-alrao_wd0.0001/"

