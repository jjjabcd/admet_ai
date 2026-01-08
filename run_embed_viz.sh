#!/bin/bash

DATASET_NAME="Ki"
DATA_DIR="/home/rlawlsgurjh/hdd/work/QSAR/gitlab/LT-MMFE/admet_ai/data/processed/${DATASET_NAME}"
MODEL_DIR="/home/rlawlsgurjh/hdd/work/QSAR/gitlab/LT-MMFE/admet_ai/results/chemprop_rdkit/${DATASET_NAME}"

FOLD_NUM=5
SMILES_COL="SMILES"
LABEL_COL="Ssel"

python embed_viz.py \
    --data_dir "${DATA_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --fold_num "${FOLD_NUM}" \
    --smiles_col "${SMILES_COL}" \
    --label_col "${LABEL_COL}" \
    --model_index 0 \
    --batch_size 256 \
    --num_workers 8 \
    --gpu 0 \
    --seed 0 \
    --do_umap \
    --umap_neighbors 15 \
    --umap_min_dist 0.1
