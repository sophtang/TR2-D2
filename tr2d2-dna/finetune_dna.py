# direct reward backpropagation
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import oracle
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import argparse
import wandb
import os
import datetime
from utils import str2bool, set_seed
# imports
from finetune_utils import loss_wdce
from tqdm import tqdm

def finetune(args, cfg, policy_model, reward_model, mcts = None, pretrained_model = None, eps=1e-5):
    """
    Finetuning with WDCE loss
    """
    dt = (1 - eps) / args.total_num_steps
    
    if args.no_mcts:
        assert pretrained_model is not None, "pretrained model is required for no mcts"
    else:
        assert mcts is not None, "mcts is required for mcts"
    
    # set model to train mode
    policy_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    # record metrics
    batch_losses = []
    batch_rewards = []
    
    # initialize the final seqs and log_rnd of the trajectories that generated those seqs
    x_saved, log_rnd_saved, final_rewards_saved = None, None, None

    # finetuning loop
    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:
        # store metrics
        rewards = []
        losses = []
        
        policy_model.train()
        
        with torch.no_grad():
            if x_saved is None or epoch % args.resample_every_n_step == 0:
                # compute final sequences and trajectory log_rnd
                if args.no_mcts:
                    x_final, log_rnd, final_rewards = policy_model.sample_finetuned_with_rnd(args, reward_model, pretrained_model)
                else:
                    x_final, log_rnd, final_rewards = mcts.forward(args.reset_tree)
                
                
                # save for next iteration
                x_saved, log_rnd_saved, final_rewards_saved = x_final, log_rnd, final_rewards
            else:
                x_final, log_rnd, final_rewards = x_saved, log_rnd_saved, final_rewards_saved
                
        # compute wdce loss
        loss = loss_wdce(policy_model, log_rnd, x_final, num_replicates=args.wdce_num_replicates)
        
        # gradient descent
        loss.backward()
        
        # optimizer
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.gradnorm_clip)
        optim.step()
        optim.zero_grad()
        
        pbar.set_postfix(loss=loss.item())
        
        losses.append(loss.item())
        
        # sample a eval batch with updated policy to evaluate rewards
        x_eval, mean_reward_eval = policy_model.sample_finetuned(args, reward_model)

        batch_losses.append(loss.cpu().detach().numpy())
        batch_rewards.append(mean_reward_eval.cpu().detach().item())
        losses.append(loss.cpu().detach().numpy())
        
        rewards = np.array(mean_reward_eval.detach().cpu().numpy())
        losses = np.array(losses)
        
        mean_reward_search = final_rewards.mean().item()
        min_reward_search = final_rewards.min().item()
        max_reward_search = final_rewards.max().item()
        median_reward_search = final_rewards.median().item()
        
        
        #reward_losses = np.array(reward_losses)

        print("epoch %d"%epoch, "mean reward %f"%mean_reward_eval, "mean loss %f"%np.mean(losses))

        wandb.log({"epoch": epoch, "mean_reward": mean_reward_eval, "mean_loss": np.mean(losses),
                   "mean_reward_search": mean_reward_search, "min_reward_search": min_reward_search,
                   "max_reward_search": max_reward_search, "median_reward_search": median_reward_search})
        
        
        if (epoch+1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(args.save_path, f'model_{epoch}.ckpt')
            torch.save(policy_model.state_dict(), model_path)
            print(f"model saved at epoch {epoch}")
    
    
    wandb.finish()

    return batch_losses