#!/usr/bin/env bash

DATASET_NAME="MRE"
BERT_NAME="bert-base-uncased"

CUDA_VISIBLE_DEVICES=0 python -u  run.py \
        --dataset_name=${DATASET_NAME} \
        --bert_name=${BERT_NAME} \
        --num_epochs=15 \
        --batch_size=16 \
        --lr=1e-2 \
        --warmup_ratio=0.06 \
        --eval_begin_epoch=1 \
        --seed=3407 \
        --do_train \
        --max_seq=80 \
        --use_prompt \
        --prompt_len=4 \
        --sample_ratio=1.0 \
        --save_path='ckpt/re/' \
        --img_dim  1024 \
        --embed_size  1024 \
        --cnn_type  'resnet50'