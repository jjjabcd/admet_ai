#!/bin/bash

DATASET=HIA_Hou
DATA_DIR=/HDD1/rlawlsgurjh/work/QSAR/admet_ai/data/processed/${DATASET}/${DATASET}
OUT_DIR=/HDD1/rlawlsgurjh/work/QSAR/admet_ai/results

python train.py \
    --dataset_dir ${DATA_DIR} \
    --output_dir ${OUT_DIR} \
    --fold_num 5 \
    --dataset_name ${DATASET} \
    --dataset_type classification \
    --compound_col smiles \
    --label_col HIA_Hou \
    --ensemble_size 5 \
    --batch_size 64 \
    --gpu 0 \
    --metric auc \
    --extra_metrics prc-auc \
    --save_preds \
    --quiet