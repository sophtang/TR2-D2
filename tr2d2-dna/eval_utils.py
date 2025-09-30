import numpy as np
import torch
from scipy.stats import pearsonr
import dataloader_gosai
import oracle


def compare_kmer(kmer1, kmer2, n_sp1, n_sp2):
    kmer_set = set(kmer1.keys()) | set(kmer2.keys())
    counts = np.zeros((len(kmer_set), 2))
    for i, kmer in enumerate(kmer_set):
        if kmer in kmer1: counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
        if kmer in kmer2: counts[i][0] = kmer2[kmer]
    return pearsonr(counts[:, 0], counts[:, 1])[0]


def get_eval_matrics(samples, ref_model, gosai_oracle, cal_atac_pred_new_mdl, highexp_kmers_999, n_highexp_kmers_999):
    """samples: [B, 200]"""
    info = {}
    detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples.detach().cpu().numpy()) # [B], strings with length 200
    ref_log_lik = ref_model.get_likelihood(samples, num_steps=128, n_samples=1) # [B]
    info['[log-lik-med]'] = torch.median(ref_log_lik).item() 
    preds = oracle.cal_gosai_pred_new(detokenized_samples, gosai_oracle, mode='eval')[:, 0]
    info['[pred-activity-med]'] = np.median(preds).item()
    atac = oracle.cal_atac_pred_new(detokenized_samples, cal_atac_pred_new_mdl)[:, 1]
    info['[atac-acc%]'] = (atac > 0.5).sum().item() / len(samples) * 100
    kmer = oracle.count_kmers(detokenized_samples)
    info['[3-mer-corr]'] = compare_kmer(highexp_kmers_999, kmer, n_highexp_kmers_999, len(detokenized_samples)).item()
    return info