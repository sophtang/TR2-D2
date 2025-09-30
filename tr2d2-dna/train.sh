#!/bin/bash

#SBATCH --job-name=dna_mdns
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --time=16:00:00
# max 16 GPU hours, i.e., time <= 16h / num of GPUs
#SBATCH --mem-per-gpu=60G
# maximum GPU RAM, 141G for H200, 94G for H100
# in the current setting, 40G is enough for num_replicates=2 and 80G is enough for num_replicates=4
#SBATCH --cpus-per-task=2
#SBATCH --wait-all-nodes=1
#SBATCH --output=../outputs/%j.%x/.log


HOME_LOC= "" # Fill in directory of the repo
SAVE_PATH = "" # Fill in directory to save the checkpoints
BASE_PATH = "" # Fill in directory of the pretrained checkpoints, e.g., "...../data_and_model/"
SCRIPT_LOC=$HOME_LOC/tr2d2/dna
LOG_LOC=$HOME_LOC/tr2d2/dna/logs
DATE=$(date +%m_%d)



mkdir -p "$LOG_LOC"

# set 3 have skip connection
# ===================================================================
python $SCRIPT_LOC/finetune.py \
    --base_path $BASE_PATH \
    --device "cuda:0" \
    --noise_removal \
    --save_path_dir $SAVE_PATH \
    --wdce_num_replicates 16 \
    --buffer_size 160 \
    --batch_size 160 \
    --seq_length 200 \
    --num_children 32 \
    --total_num_steps 128 \
    --num_iter 5 \
    --resample_every_n_step 5 \
    --eval_every_n_epochs 10 \
    --num_epochs 60000 \
    --exploration 0.1 \
    --save_every_n_epoch 2000 \
    --alpha 0.1 \
    --centering \
    --grad_clip \
    --reward_clip \
    --reward_clip_value 15.0 \
    --reset_tree