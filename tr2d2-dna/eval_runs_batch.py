#!/usr/bin/env python3
"""
Batch evaluation script for multiple runs with checkpoints.

This script:
1. Scans a folder containing different runs
2. For each run, finds checkpoints and selects the one with largest epoch number
3. Evaluates that checkpoint and saves results indexed by run folder name
"""

import os
import re
import glob
import argparse
from pathlib import Path
from diffusion import Diffusion
import dataloader_gosai
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import oracle
from scipy.stats import pearsonr
import torch
from tqdm import tqdm
from eval_utils import get_eval_matrics
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class Args:
    total_num_steps: int
    batch_size: int
    num_seeds: int
    total_samples: int
    seq_length: int


def find_latest_checkpoint(run_dir):
    """
    Find the checkpoint with the largest epoch/step number in a run directory.
    
    Args:
        run_dir (str): Path to the run directory
        
    Returns:
        str or None: Path to the latest checkpoint, or None if no checkpoints found
    """
    ckpt_pattern = os.path.join(run_dir, "model_*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern)
    
    if not ckpt_files:
        return None
    
    # Extract step numbers from checkpoint filenames
    step_numbers = []
    for ckpt_file in ckpt_files:
        filename = os.path.basename(ckpt_file)
        match = re.search(r'model_(\d+)\.ckpt', filename)
        if match:
            step_numbers.append((int(match.group(1)), ckpt_file))
    
    if not step_numbers:
        return None
    
    # Return checkpoint with largest step number
    step_numbers.sort(key=lambda x: x[0], reverse=True)
    return step_numbers[0][1]


def evaluate_checkpoint(checkpoint_path, args, cfg, pretrained_model, gosai_oracle, 
                       cal_atac_pred_new_mdl, highexp_kmers_999, n_highexp_kmers_999, device):
    """
    Evaluate a single checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        args: Evaluation arguments
        cfg: Configuration object
        pretrained_model: Pretrained reference model
        gosai_oracle: GOSAI oracle model
        cal_atac_pred_new_mdl: ATAC prediction model
        highexp_kmers_999: High expression k-mers
        n_highexp_kmers_999: Number of high expression k-mers
        device: Device to run evaluation on
        
    Returns:
        tuple: (eval_metrics_agg, total_rewards_agg) containing aggregated results
    """
    # Load the policy model from checkpoint
    policy_model = Diffusion(cfg).to(device)
    policy_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy_model.eval()
    
    total_rewards_all = []
    eval_metrics_all = []
    
    print(f"Evaluating checkpoint: {os.path.basename(checkpoint_path)}")
    
    for i in range(args.num_seeds):
        iter_times = args.total_samples // args.batch_size
        total_samples = []
        total_rewards = []
        range_bar = tqdm(range(iter_times), desc=f"Seed {i+1}", leave=False)
        
        for j in range_bar:
            x_eval, mean_reward_eval = policy_model.sample_finetuned(args, gosai_oracle)
            total_samples.append(x_eval)
            total_rewards.append(mean_reward_eval.item() * args.batch_size)
            
        total_samples = torch.concat(total_samples)
        eval_metrics = get_eval_matrics(samples=total_samples, ref_model=pretrained_model, 
                                        gosai_oracle=gosai_oracle, cal_atac_pred_new_mdl=cal_atac_pred_new_mdl, 
                                        highexp_kmers_999=highexp_kmers_999, n_highexp_kmers_999=n_highexp_kmers_999)

        eval_metrics_all.append(eval_metrics)
        total_rewards_all.append(np.sum(total_rewards) / args.total_samples)
        
    # Aggregate results
    eval_metrics_agg = {k: (np.mean([eval_metrics[k] for eval_metrics in eval_metrics_all]), 
                            np.std([eval_metrics[k] for eval_metrics in eval_metrics_all])) 
                       for k in eval_metrics_all[0].keys()}
    total_rewards_agg = (np.mean(total_rewards_all), np.std(total_rewards_all))
    
    return eval_metrics_agg, total_rewards_agg


def save_results(results, output_file):
    """
    Save evaluation results to a text file.
    
    Args:
        results (dict): Dictionary containing results for each run
        output_file (str): Path to output file
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BATCH EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total runs evaluated: {len(results)}\n\n")
        
        for run_name, result in results.items():
            if result is None:
                f.write(f"RUN: {run_name}\n")
                f.write("-" * 60 + "\n")
                f.write("Status: No checkpoints found or evaluation failed\n\n")
                continue
                
            eval_metrics_agg, total_rewards_agg, checkpoint_path = result
            
            f.write(f"RUN: {run_name}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
            f.write(f"Full path: {checkpoint_path}\n\n")
            
            f.write("📊 EVALUATION METRICS:\n")
            for metric_name in eval_metrics_agg.keys():
                mean_val = eval_metrics_agg[metric_name][0]
                std_val = eval_metrics_agg[metric_name][1]
                f.write(f"  {metric_name:<20}: {mean_val:8.4f} ± {std_val:6.4f}\n")
            
            f.write(f"\n🎯 TOTAL REWARDS:\n")
            f.write(f"  {'Mean':<20}: {total_rewards_agg[0]:8.4f}\n")
            f.write(f"  {'Std':<20}: {total_rewards_agg[1]:8.4f}\n")
            f.write("\n")
    
    print(f"Results saved to: {output_file}")


def append_single_result(run_name, result, output_file, is_first_run=False):
    """
    Append a single successful run result to the output file.
    
    Args:
        run_name (str): Name of the run
        result: Result tuple (eval_metrics_agg, total_rewards_agg, checkpoint_path)
        output_file (str): Path to output file
        is_first_run (bool): Whether this is the first successful run (write header)
    """
    mode = 'w' if is_first_run else 'a'
    
    with open(output_file, mode) as f:
        if is_first_run:
            f.write("="*80 + "\n")
            f.write("BATCH EVALUATION RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Started on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("Results are saved incrementally as each run completes.\n")
            f.write("Only successful evaluations are included.\n\n")
            
        eval_metrics_agg, total_rewards_agg, checkpoint_path = result
        
        f.write(f"RUN: {run_name}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Checkpoint: {os.path.basename(checkpoint_path)}\n")
        f.write(f"Full path: {checkpoint_path}\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("📊 EVALUATION METRICS:\n")
        for metric_name in eval_metrics_agg.keys():
            mean_val = eval_metrics_agg[metric_name][0]
            std_val = eval_metrics_agg[metric_name][1]
            f.write(f"  {metric_name:<20}: {mean_val:8.4f} ± {std_val:6.4f}\n")
        
        f.write(f"\n🎯 TOTAL REWARDS:\n")
        f.write(f"  {'Mean':<20}: {total_rewards_agg[0]:8.4f}\n")
        f.write(f"  {'Std':<20}: {total_rewards_agg[1]:8.4f}\n")
        f.write("\n" + "="*80 + "\n\n")  # Add separator line and extra spacing


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation of multiple runs")
    parser.add_argument("--runs_dir", type=str, required=True, 
                       help="Directory containing run folders with checkpoints")
    parser.add_argument("--output_file", type=str, default="batch_eval_results.txt",
                       help="Output file to save results")
    parser.add_argument("--device", type=str, default="cuda:0", 
                       help="Device to run evaluation on")
    parser.add_argument("--total_num_steps", type=int, default=128,
                       help="Total number of diffusion steps")
    parser.add_argument("--batch_size", type=int, default=128,
                       help="Batch size for evaluation")
    parser.add_argument("--num_seeds", type=int, default=3,
                       help="Number of random seeds for evaluation")
    parser.add_argument("--total_samples", type=int, default=640,
                       help="Total number of samples to generate")
    parser.add_argument("--seq_length", type=int, default=200,
                       help="Sequence length")
    parser.add_argument("--pretrained_path", type=str, 
                       default=None,
                       help="Path to pretrained model checkpoint")
    
    args = parser.parse_args()
    
    # Setup evaluation arguments
    eval_args = Args(
        total_num_steps=args.total_num_steps,
        batch_size=args.batch_size,
        num_seeds=args.num_seeds,
        total_samples=args.total_samples,
        seq_length=args.seq_length
    )
    
    device = args.device
    
    # Initialize Hydra configuration
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize(config_path="configs_gosai", job_name="batch_eval")
    cfg = compose(config_name="config_gosai.yaml")
    
    print("Loading pretrained model and oracles...")
    # Load pretrained model
    pretrained_model = Diffusion.load_from_checkpoint(args.pretrained_path, config=cfg, map_location=device)
    pretrained_model.eval()
    
    # Load oracles
    _, _, highexp_kmers_999, n_highexp_kmers_999, _, _, _ = oracle.cal_highexp_kmers(return_clss=True)
    cal_atac_pred_new_mdl = oracle.get_cal_atac_orale(device=device)
    cal_atac_pred_new_mdl.eval()
    gosai_oracle = oracle.get_gosai_oracle(mode='eval', device=device)
    gosai_oracle.eval()
    
    print("Scanning for runs...")
    # Find all run directories
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"Error: Directory {args.runs_dir} does not exist")
        return
    
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    run_dirs.sort()  # Sort for consistent ordering
    
    print(f"Found {len(run_dirs)} run directories")
    
    results = {}
    successful_runs = 0
    failed_runs = 0
    
    # Process each run
    for i, run_dir in enumerate(tqdm(run_dirs, desc="Processing runs")):
        run_name = run_dir.name
        print(f"\nProcessing run {i+1}/{len(run_dirs)}: {run_name}")
        
        # Find latest checkpoint
        latest_ckpt = find_latest_checkpoint(str(run_dir))
        
        if latest_ckpt is None:
            print(f"  No checkpoints found in {run_name} - skipping")
            failed_runs += 1
            continue  # Skip this run entirely, don't save anything to file
        
        print(f"  Found latest checkpoint: {os.path.basename(latest_ckpt)}")
        
        try:
            # Evaluate checkpoint
            eval_metrics_agg, total_rewards_agg = evaluate_checkpoint(
                latest_ckpt, eval_args, cfg, pretrained_model, gosai_oracle,
                cal_atac_pred_new_mdl, highexp_kmers_999, n_highexp_kmers_999, device
            )
            
            result = (eval_metrics_agg, total_rewards_agg, latest_ckpt)
            results[run_name] = result
            successful_runs += 1
            print(f"  ✓ Evaluation completed successfully")
            
            # Save result incrementally (only for successful evaluations)
            is_first_run = (len(results) == 1)  # First successful run
            append_single_result(run_name, result, args.output_file, is_first_run=is_first_run)
            print(f"  Result saved to {args.output_file}")
            
        except Exception as e:
            print(f"  ✗ Evaluation failed: {str(e)}")
            failed_runs += 1
            # Don't save failed evaluations to file either
    
    # Add final summary to the file (only if there were successful runs)
    if successful_runs > 0:
        with open(args.output_file, 'a') as f:
            f.write("="*80 + "\n")
            f.write("FINAL SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total runs processed: {len(run_dirs)}\n")
            f.write(f"Successful evaluations: {successful_runs}\n")
            f.write(f"Failed/skipped runs: {failed_runs}\n")
    else:
        print(f"No successful evaluations - output file {args.output_file} not created")
    
    # Print summary
    print(f"\nFinal Summary:")
    print(f"  Total runs processed: {len(run_dirs)}")
    print(f"  Successful evaluations: {successful_runs}")
    print(f"  Failed/skipped runs: {failed_runs}")
    if successful_runs > 0:
        print(f"  Results saved to: {args.output_file}")
    else:
        print(f"  No output file created (no successful evaluations)")


if __name__ == "__main__":
    main()
