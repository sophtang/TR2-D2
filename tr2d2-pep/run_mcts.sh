#!/bin/bash

HOME_LOC=/path/to/your/home
ENV_PATH=/path/to/your/env
SCRIPT_LOC=$HOME_LOC/tr2d2/peptides
LOG_LOC=$HOME_LOC/tr2d2/peptides/logs
DATE=$(date +%m_%d)
SPECIAL_PREFIX='tfr-peptune-baseline'
# set 3 have skip connection
PYTHON_EXECUTABLE=$ENV_PATH/bin/python

# ===================================================================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_PATH

$PYTHON_EXECUTABLE $SCRIPT_LOC/generate_mcts.py \
    --base_path $HOME_LOC \
    --device "cuda:0" \
    --noise_removal \
    --run_name "tfr-peptune-baseline" \
    --num_children 50 \
    --num_iter 100 \
    --buffer_size 100 \
    --seq_length 200 \
    --total_num_steps 128 \
    --exploration 0.1 > "${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}.log" 2>&1

conda deactivate