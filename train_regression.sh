#!/bin/bash

DATASET=Half_Life_Obach
DATA_DIR=/HDD1/rlawlsgurjh/work/QSAR/admet_ai/data/processed/${DATASET}/${DATASET}
OUT_DIR=/HDD1/rlawlsgurjh/work/QSAR/admet_ai/results

python train.py \
    --dataset_dir ${DATA_DIR} \
    --output_dir ${OUT_DIR} \
    --fold_num 5 \
    --dataset_name ${DATASET} \
    --dataset_type regression \
    --compound_col smiles \
    --label_col Half_Life_Obach \
    --ensemble_size 5 \
    --batch_size 50 \
    --gpu 0 \
    --metric mae \
    --extra_metrics r2 \
    --save_preds \
    --quiet
