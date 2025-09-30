# direct reward backpropagation
from diffusion import Diffusion
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
from finetune_dna import finetune
from mcts import MCTS

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default="")
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=100)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
argparser.add_argument('--gradnorm_clip', type=float, default=1.0)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--name', type=str, default='debug')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=10)
argparser.add_argument('--eval_every_n_epochs', type=int, default=200)
argparser.add_argument('--alpha', type=float, default=0.001)
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
# new
argparser.add_argument('--run_name', type=str, default='drakes')
argparser.add_argument("--device", default="cuda:0", type=str)
argparser.add_argument("--save_path_dir", default=None, type=str)
argparser.add_argument("--no_mcts", action='store_true', default=False)
argparser.add_argument("--centering", action='store_true', default=False)
argparser.add_argument("--reward_clip", action='store_true', default=False)
argparser.add_argument("--reward_clip_value", type=float, default=15.0)
argparser.add_argument("--select_topk", action='store_true', default=False)
argparser.add_argument('--select_topk_value', type=int, default=10)
argparser.add_argument("--restart_ckpt_path", type=str, default=None)


# mcts
argparser.add_argument('--num_sequences', type=int, default=10)
argparser.add_argument('--num_children', type=int, default=50)
argparser.add_argument('--num_iter', type=int, default=30) # iterations of mcts
argparser.add_argument('--seq_length', type=int, default=200)
argparser.add_argument('--time_conditioning', action='store_true', default=False)
argparser.add_argument('--mcts_sampling', type=int, default=0) # for batched categorical sampling: '0' means gumbel noise
argparser.add_argument('--buffer_size', type=int, default=100)
argparser.add_argument('--wdce_num_replicates', type=int, default=16)
argparser.add_argument('--noise_removal', action='store_true', default=False)
argparser.add_argument('--grad_clip', action='store_true', default=False)
argparser.add_argument('--resample_every_n_step', type=int, default=10)
argparser.add_argument('--exploration', type=float, default=0.1)
argparser.add_argument('--reset_tree', action='store_true', default=False)

# eval

args = argparser.parse_args()
print(args)

# pretrained model path
CKPT_PATH = os.path.join(args.base_path, 'mdlm/outputs_gosai/pretrained.ckpt')
log_base_dir = os.path.join(args.save_path_dir, 'mdlm/reward_bp_results_final')

# reinitialize Hydra
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration
initialize(config_path="configs_gosai", job_name="load_model")
cfg = compose(config_name="config_gosai.yaml")
cfg.eval.checkpoint_path = CKPT_PATH
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

if args.no_mcts:
    run_name = f'MDNS_buffer{args.buffer_size}_alpha{args.alpha}_resample{args.resample_every_n_step}_centering{args.centering}_{curr_time}'
else:
    run_name = f'MCTS_buffer{args.buffer_size}_alpha{args.alpha}_resample{args.resample_every_n_step}_num_iter{args.num_iter}_centering{args.centering}_select_topk{args.select_topk}_select_topk_value{args.select_topk_value}_{curr_time}'
    
args.save_path = os.path.join(args.save_path_dir, run_name)
os.makedirs(args.save_path, exist_ok=True)
# wandb init
wandb.init(project='search-rl', name=run_name, config=args, dir=args.save_path)

log_path = os.path.join(args.save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

# Initialize the model
if args.restart_ckpt_path is not None:
    # Resume from saved ckpt
    restart_ckpt_path = os.path.join(args.base_path, args.restart_ckpt_path)
    restart_epoch = restart_ckpt_path.split('_')[-1].split('.')[0]
    args.restart_epoch = restart_epoch
    policy_model = Diffusion(cfg).to(args.device)
    policy_model.load_state_dict(torch.load(restart_ckpt_path, map_location=args.device))
else:
    # Start from pretrained model
    policy_model = Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg, map_location=args.device)
pretrained = Diffusion.load_from_checkpoint(cfg.eval.checkpoint_path, config=cfg, map_location=args.device)
reward_model = oracle.get_gosai_oracle(mode='train', device=args.device)

#reward_model_eval = oracle.get_gosai_oracle(mode='eval').to(args.device)

reward_model.eval()
pretrained.eval()
#reward_model_eval.eval()

# define mcts
mcts = MCTS(args, cfg, policy_model, pretrained, reward_model)


_, _, highexp_kmers_999, n_highexp_kmers_999, _, _, _ = oracle.cal_highexp_kmers(return_clss=True)

cal_atac_pred_new_mdl = oracle.get_cal_atac_orale(device=args.device)
cal_atac_pred_new_mdl.eval()

gosai_oracle = oracle.get_gosai_oracle(mode='eval', device=args.device)
gosai_oracle.eval()


print("args.device:", args.device)
print("policy_model device:", policy_model.device)
print("pretrained device:", pretrained.device)
print("reward_model device:", reward_model.device)
print("mcts device:", mcts.device)
print("gosai_oracle device:", gosai_oracle.device)
print("cal_atac_pred_new_mdl device:", cal_atac_pred_new_mdl.device)

eval_model_dict = {
    "gosai_oracle": gosai_oracle,
    "highexp_kmers_999": highexp_kmers_999, 
    "n_highexp_kmers_999": n_highexp_kmers_999, 
    "cal_atac_pred_new_mdl": cal_atac_pred_new_mdl,
    "gosai_oracle": gosai_oracle
}


finetune(args = args, cfg = cfg, policy_model = policy_model, 
        reward_model = reward_model, mcts = mcts, 
        pretrained_model = pretrained, 
        eval_model_dict = eval_model_dict)