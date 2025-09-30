import random
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import sample_categorical_logits
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F

def compute_ess(log_rnd, normalize=True): 
    """
    log_rnd: [B]
    Compute effective sample size:
        If normalize: divide ESS by batch size, so range is [0, 1]; 
        otherwise, range is [0, B]
    """
    weights = log_rnd.detach().softmax(dim=-1)
    ess = 1 / (weights ** 2).sum().item()
    return ess / log_rnd.shape[0] if normalize else ess

def to_one_hot(x_idx, num_classes=4):
    oh = F.one_hot(x_idx.long(), num_classes=num_classes)
    return oh.float()

def rnd(model, reward_model, batch_size, scale=1, device='cuda:0'):
    r"""
    Run random order sampling and compute the RND $\log\frac{dP^*}{dP^u}$ along the trajectory
    reward_model: r(X)

    return:
    - x: the final samples, [B, D]
    - log_rnd: the log RND along this trajectory, [B]
    """
    if hasattr(model, 'module'):
        model = model.module
    
    x = torch.full((batch_size, model.length), model.vocab_size-1).to(device=device, dtype=torch.int64)
    batch_arange = torch.arange(batch_size, device=device)
    jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
    # jump_times, jump_pos = torch.rand(x.shape, device=device).sort(dim=-1)
    # jump_times: Unif[0,1] in increasing order
    # jump_pos: random permutation of range(D)
    log_rnd = torch.zeros(batch_size, device=device) # [B]
    for d in range(model.length-1, -1, -1):
        # jump at time jump_times[:, d] at position jump_pos[:, d]
        logits = model(x)[:, :, :-1] # [B, D, N-1]
        update = sample_categorical_logits(
            logits[batch_arange, jump_pos[:, d]]) # [B]
        if torch.is_grad_enabled(): # avoid issues with in-place operations
            x = x.clone()
        x[batch_arange, jump_pos[:, d]] = update
        log_rnd += -np.log(model.vocab_size-1) - logits[batch_arange, jump_pos[:, d], update]
    log_rnd += scale * reward_model(x) # [B]
    return x, log_rnd


@torch.no_grad()
def sampling(model, batch_size, rounds=1, device='cuda:0'):
    """Any order autoregressive sampling"""
    if hasattr(model, 'module'):
        model = model.module
    batch_arange = torch.arange(batch_size, device=device)
    all_samples = []
    for _ in tqdm(range(rounds), leave=False):
        x = torch.full((batch_size, model.length), model.vocab_size-1).to(device=device, dtype=torch.int64)
        jump_pos = torch.rand(x.shape, device=device).argsort(dim=-1)
        # jump_times, jump_pos = torch.rand(x.shape, device=device).sort(dim=-1)
        # jump_times: Unif[0,1] in increasing order
        # jump_pos: random permutation of range(D)
        for d in tqdm(range(model.length-1, -1, -1), leave=False):
            # jump at time jump_times[:, d] at position jump_pos[:, d]
            logits = model.logits(x)[:, :, :-1] # [B, D, N-1], not log-softmaxed but fine
            update = sample_categorical_logits(
                logits[batch_arange, jump_pos[:, d]]) # [B]
            x[batch_arange, jump_pos[:, d]] = update
        all_samples.append(x)
    return torch.cat(all_samples) # (rounds * B, L)


def loss_ce(log_rnd):
    """Cross entropy loss KL(P^*||P^u)"""
    weights = log_rnd.detach().softmax(dim=-1)
    return (log_rnd * weights).sum()


def loss_lv(log_rnd):
    r"""Log variance loss Var_{P^\bar{u}}\log\frac{dP^*}{dP^u}"""
    return log_rnd.var()


def loss_re_rf(log_rnd, const=0):
    r"""Relative entropy loss KL(P^u||P^*) with REINFORCE trick"""
    return (-log_rnd * (-log_rnd.detach() + const)).mean()


def loss_wdce(policy_model, log_rnd, x, num_replicates=16, weight_func=lambda l: 1/l, eps=1e-3, centering = False):
    r"""
    Weighted denoising cross entropy loss
    X_T ~ P^u_T and weights \log\frac{dP^*}{dP^u}(X)
    
    log_rnd: [B]; x: [B, L] (no mask)
    num_replicates: R, number of replicates of each row in x
    weight_func: w(lambda) for each sample, 1/lambda by default
    """
    mask_index = policy_model.mask_index
    if hasattr(policy_model, 'module'):
        policy_model = policy_model.module
    
    batch = x.repeat_interleave(num_replicates, dim=0) # [B*R, L]
    batch_weights = log_rnd.detach_().softmax(dim=-1)
    if centering:
        batch_weights = batch_weights - batch_weights.mean(dim=-1, keepdim=True)
    
    batch_weights = batch_weights.repeat_interleave(num_replicates, dim=0) # [B*R]
    
    lamda = torch.rand(batch.shape[0], device=batch.device) # [B*R]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B*R]
    
    masked_index = torch.rand(*batch.shape, device=batch.device) < lamda[..., None] # [B*R, D]
    perturbed_batch = torch.where(masked_index, mask_index, batch)
    
    # add time conditioning
    t = lamda 
    sigma_t = -torch.log1p(-(1 - eps) * t)
    
    # compute logits
    logits = policy_model(perturbed_batch, sigma_t)
    losses = torch.zeros(*batch.shape, device=batch.device, dtype=logits.dtype) # [B*R, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=batch[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights * batch_weights).mean()


def loss_dce(model, x, weight_func=lambda l: 1/l):
    r"""
    Denoising cross entropy loss, x [B, D] are ground truth samples
    weight_func: w(lambda) for each sample, 1/lambda by default
    """
    lamda = torch.rand(x.shape[0], device=x.device) # [B]
    lamda_weights = weight_func(lamda).clamp(max=1e5) # [B]
    masked_index = torch.rand(*x.shape, device=x.device) < lamda[..., None] # [B, D]
    perturbed_batch = torch.where(masked_index, model.vocab_size-1, x)
    logits = model(perturbed_batch)
    losses = torch.zeros(*x.shape, device=x.device, dtype=logits.dtype) # [B, D]
    losses[masked_index] = torch.gather(input=logits[masked_index], dim=-1,
                                        index=x[masked_index][..., None]).squeeze(-1)
    return - (losses.sum(dim=-1) * lamda_weights).mean()