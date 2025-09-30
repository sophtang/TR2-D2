# direct reward backpropagation
import numpy as np
import torch
import wandb
import os
from finetune_utils import loss_wdce
from tqdm import tqdm
import pandas as pd
from plotting import plot_data_with_distribution_seaborn, plot_data

def finetune(args, cfg, policy_model, reward_model, mcts=None, pretrained=None, filename=None, prot_name=None, eps=1e-5):
    """
    Finetuning with WDCE loss
    """
    base_path = args.base_path
    dt = (1 - eps) / args.total_num_steps
    
    if args.no_mcts:
        assert pretrained is not None, "pretrained model is required for no mcts"
    else:
        assert mcts is not None, "mcts is required for mcts"
        
    # set model to train mode
    policy_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    # record metrics
    batch_losses = []
    #batch_rewards = []
    
    # initialize the final seqs and log_rnd of the trajectories that generated those seqs
    x_saved, log_rnd_saved, final_rewards_saved = None, None, None
    
    valid_fraction_log = []
    affinity_log = []
    sol_log = []
    hemo_log = []
    nf_log = []
    permeability_log = []

     ### End of Fine-Tuning Loop ###
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
                    x_final, log_rnd, final_rewards = policy_model.sample_finetuned_with_rnd(args, reward_model, pretrained)
                else:
                    # decides whether to reset tree                
                    if (epoch) % args.reset_every_n_step == 0:
                        x_final, log_rnd, final_rewards, _, _ = mcts.forward(resetTree=True)
                    else:
                        x_final, log_rnd, final_rewards, _, _ = mcts.forward(resetTree=False)
                
                # save for next iteration
                x_saved, log_rnd_saved, final_rewards_saved = x_final, log_rnd, final_rewards
            else:
                x_final, log_rnd, final_rewards = x_saved, log_rnd_saved, final_rewards_saved
                
        # compute wdce loss
        loss = loss_wdce(policy_model, log_rnd, x_final, num_replicates=args.wdce_num_replicates, centering=args.centering)
        
        # gradient descent
        loss.backward()
        
        # optimizer
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.gradnorm_clip)
        
        optim.step()
        optim.zero_grad()
        
        pbar.set_postfix(loss=loss.item())
                
        # sample a eval batch with updated policy to evaluate rewards
        x_eval, affinity, sol, hemo, nf, permeability, valid_fraction = policy_model.sample_finetuned(args, reward_model, batch_size=50, dataframe=False)

        # append to log
        affinity_log.append(affinity)
        sol_log.append(sol)
        hemo_log.append(hemo)
        nf_log.append(nf)
        permeability_log.append(permeability)
        valid_fraction_log.append(valid_fraction)
        
        batch_losses.append(loss.cpu().detach().numpy())
        
        losses.append(loss.cpu().detach().numpy())
        losses = np.array(losses)
        
        if args.no_mcts:
            mean_reward_search = final_rewards.mean().item()
            min_reward_search = final_rewards.min().item()
            max_reward_search = final_rewards.max().item()
            median_reward_search = final_rewards.median().item()
        else:
            mean_reward_search = np.mean(final_rewards)
            min_reward_search = np.min(final_rewards)
            max_reward_search = np.max(final_rewards)
            median_reward_search = np.median(final_rewards)
        
        print("epoch %d"%epoch, "affinity %f"%np.mean(affinity), "sol %f"%np.mean(sol), "hemo %f"%np.mean(hemo), "nf %f"%np.mean(nf), "permeability %f"%np.mean(permeability), "mean loss %f"%np.mean(losses))
        
        wandb.log({"epoch": epoch, "affinity": np.mean(affinity), "sol": np.mean(sol), "hemo": np.mean(hemo), "nf": np.mean(nf), "permeability": np.mean(permeability),
                   "mean_loss": np.mean(losses),
                   "mean_reward_search": mean_reward_search, "min_reward_search": min_reward_search,
                   "max_reward_search": max_reward_search, "median_reward_search": median_reward_search})
        
        if (epoch+1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(args.save_path, f'model_{epoch}.ckpt')
            torch.save(policy_model.state_dict(), model_path)
            print(f"model saved at epoch {epoch}")
    
    ### End of Fine-Tuning Loop ###
    
    wandb.finish()
    
    # save logs and plot
    plot_path = f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}'
    os.makedirs(plot_path, exist_ok=True)
    output_log_path = f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/log_{filename}.csv'
    save_logs_to_file(valid_fraction_log, affinity_log, 
                      sol_log, hemo_log, nf_log, 
                      permeability_log, output_log_path)
    
    plot_data(valid_fraction_log, 
            save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/valid_{filename}.png')
    
    plot_data_with_distribution_seaborn(log1=affinity_log,
            save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/binding_{filename}.png',
            label1=f"Average Binding Affinity to {prot_name}",
            title=f"Average Binding Affinity to {prot_name} Over Iterations")
    
    plot_data_with_distribution_seaborn(log1=sol_log,
            save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/sol_{filename}.png',
            label1="Average Solubility Score",
            title="Average Solubility Score Over Iterations")
    plot_data_with_distribution_seaborn(log1=hemo_log,
            save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/hemo_{filename}.png',
            label1="Average Hemolysis Score",
            title="Average Hemolysis Score Over Iterations")
    plot_data_with_distribution_seaborn(log1=nf_log,
            save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/nf_{filename}.png',
            label1="Average Nonfouling Score",
            title="Average Nonfouling Score Over Iterations")
    plot_data_with_distribution_seaborn(log1=permeability_log,
            save_path=f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/perm_{filename}.png',
            label1="Average Permeability Score",
            title="Average Permeability Score Over Iterations")
    
    x_eval, affinity, sol, hemo, nf, permeability, valid_fraction, df = policy_model.sample_finetuned(args, reward_model, batch_size=200, dataframe=True)
    df.to_csv(f'{base_path}/TR2-D2/tr2d2-pep/results/{args.run_name}/{prot_name}_generation_results.csv', index=False)

    return batch_losses

def save_logs_to_file(valid_fraction_log, affinity_log, 
                      sol_log, hemo_log, nf_log, 
                      permeability_log, output_path):
    """
    Saves the logs (valid_fraction_log, affinity1_log, and permeability_log) to a CSV file.
    
    Parameters:
        valid_fraction_log (list): Log of valid fractions over iterations.
        affinity1_log (list): Log of binding affinity over iterations.
        permeability_log (list): Log of membrane permeability over iterations.
        output_path (str): Path to save the log CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine logs into a DataFrame
    log_data = {
        "Iteration": list(range(1, len(valid_fraction_log) + 1)),
        "Valid Fraction": valid_fraction_log,
        "Binding Affinity": affinity_log,
        "Solubility": sol_log,
        "Hemolysis": hemo_log, 
        "Nonfouling": nf_log,
        "Permeability": permeability_log
    }
        
    df = pd.DataFrame(log_data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)