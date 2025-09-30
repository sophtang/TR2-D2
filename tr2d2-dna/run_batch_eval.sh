#!/bin/bash
#SBATCH --job-name=dna
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:H200:1
#SBATCH --time=16:00:00
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-task=2
#SBATCH --wait-all-nodes=1
#SBATCH --output=../outputs/%j.%x/.log


# Set the path to your runs directory
RUNS_DIR="" # Fill in directory of which to eval the checkpoints

# Set output file name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="batch_eval_results_${TIMESTAMP}.txt"

# Run the batch evaluation
python eval_runs_batch.py \
    --runs_dir "$RUNS_DIR" \
    --output_file "$OUTPUT_FILE" \
    --device "cuda:0" \
    --total_num_steps 128 \
    --batch_size 128 \
    --num_seeds 3 \
    --total_samples 640 \
    --seq_length 200

echo "Batch evaluation completed. Results saved to: $OUTPUT_FILE"
