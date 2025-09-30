#!/bin/bash

HOME_LOC=/path/to/your/home
ENV_PATH=/path/to/your/env
SCRIPT_LOC=$HOME_LOC/TR2-D2/tr2d2-pep
LOG_LOC=$HOME_LOC/TR2-D2/tr2d2-pep/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='tr2d2-finetune-tfr'
# set 3 have skip connection
PYTHON_EXECUTABLE=$ENV_PATH/bin/python

# ===================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_PATH

# ===================================================================

$PYTHON_EXECUTABLE $SCRIPT_LOC/finetune.py \
    --base_path $HOME_LOC \
    --device "cuda:6" \
    --noise_removal \
    --wdce_num_replicates 16 \
    --buffer_size 20 \
    --seq_length 200 \
    --num_children 50 \
    --total_num_steps 128 \
    --num_iter 10 \
    --resample_every_n_step 10 \
    --num_epochs 1000 \
    --exploration 0.1 \
    --save_every_n_epoch 50 \
    --reset_every_n_step 1 \
    --alpha 0.1 \
    --grad_clip > "${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}.log" 2>&1

conda deactivate