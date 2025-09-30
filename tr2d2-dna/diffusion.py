import itertools
import math
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

import dataloader_gosai
import models
import noise_schedule
import utils
import oracle
from scipy.stats import wasserstein_distance, pearsonr
from finetune_utils import to_one_hot

LOG2 = math.log(2)
LOGGER = utils.get_logger(__name__)


def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1).to(dtype=torch.long)

def _sample_categorical_gradient(categorical_probs, temp = 1.0):
    gumbel_norm = (
        1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log())
    output = torch.nn.functional.softmax((torch.log(categorical_probs)-torch.log(gumbel_norm))/temp, 2)
    return output

def _unsqueeze(x, reference):
    return x.view(
        * x.shape,
        * ((1,) * (len(reference.shape) - len(x.shape))))

def sample_batched_categorical(categorical_probs, batch_size):
    """
    Generates `m` distinct sequences sampled from categorical probabilities 
    using the Gumbel distribution to ensure randomness while following probabilities
    
    Args:
        categorical_probs (torch.Tensor): tensor of shape (sequence_length, vocab_length)
                                          representing categorical probabilities
        m (int): number of distinct sequences to sample
    
    Returns:
        torch.Tensor: tensor of shape (m, sequence_length), where each row is a 
                      distinct sequence of sampled category indices.
    """
    _, sequence_length, vocab_size = categorical_probs.shape

    # add Gumbel noise and sample m sequences
    gumbel_noise = (-torch.log(-torch.log(torch.rand(batch_size, sequence_length, vocab_size) + 1e-10) + 1e-10)).to(categorical_probs.device)
    noisy_scores = torch.log(categorical_probs) + gumbel_noise  # add Gumbel noise to log probabilities
    
    # select the highest score (most likely category after Gumbel noise)
    sampled_sequences = noisy_scores.argmax(dim=-1).to(dtype=torch.long)  # shape: (m, sequence_length)

    return sampled_sequences

def sample_batched_top_k(categorical_probs, batch_size, k):
    """
    Generates `m` sequences sampled from the top-k probabilities of each token
    using Gumbel noise to ensure randomness and reduce bias towards the most likely options.

    Args:
        categorical_probs (torch.Tensor): A tensor of shape (sequence_length, vocab_length)
                                          representing categorical probabilities.
        m (int): Number of sequences to sample.
        k (int): Number of top probabilities to consider for sampling.

    Returns:
        torch.Tensor: A tensor of shape (m, sequence_length), where each row is a 
                      sampled sequence of category indices.
    """
    _, sequence_length, vocab_length = categorical_probs.shape

    # Add Gumbel noise to the log probabilities
    gumbel_noise = -torch.log(-torch.log(torch.rand(batch_size, sequence_length, vocab_length) + 1e-10) + 1e-10).to(categorical_probs.device)
    noisy_scores = torch.log(categorical_probs[None, :, :]) + gumbel_noise  # Shape: (m, sequence_length, vocab_length)

    # Get the top-k categories based on noisy scores
    top_k_scores, top_k_indices = torch.topk(noisy_scores, k, dim=-1)  # Shape: (m, sequence_length, k)

    # Convert top-k scores back to probabilities and normalize
    top_k_probs = torch.softmax(top_k_scores, dim=-1).to(categorical_probs.device)  # Shape: (m, sequence_length, k)

    # Sample randomly from the top-k probabilities
    sampled_indices_in_top_k = torch.multinomial(top_k_probs.reshape(-1, k), num_samples=1).squeeze(-1).to(categorical_probs.device)
    sampled_indices_in_top_k = sampled_indices_in_top_k.view(batch_size, sequence_length).to(categorical_probs.device)  # Shape: (batch_size, sequence_length)

    # Map sampled indices back to the original vocabulary indices
    sampled_sequences = torch.gather(top_k_indices, -1, sampled_indices_in_top_k.unsqueeze(-1)).squeeze(-1).to(categorical_probs.device).to(dtype=torch.long)

    return sampled_sequences

@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
  def compute(self) -> Tensor:
    """Computes the bits per dimension.

    Returns:
      bpd
    """
    return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
        Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
    def __init__(
        self,
        config,
        eval=False):
        
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.vocab_size = 4
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        # add mask token
        self.mask_index = self.vocab_size
        self.vocab_size += 1
        self.parameterization = self.config.parameterization
    
        # dna backbone model
        if self.config.backbone == 'cnn':
            self.backbone = models.dnaconv.CNNModel(
                self.config.model, alphabet_size=self.vocab_size, num_cls=3) # num_cls is not used since classifier is always set to False
        else:
            raise ValueError(f'Unknown backbone: {self.config.backbone}')

        self.T = self.config.T
        self.subs_masking = self.config.subs_masking

        self.softplus = torch.nn.Softplus()
        # metrics are automatically reset at end of epoch
        metrics = torchmetrics.MetricCollection({
            'nll': NLL(),
            'bpd': BPD(),
            'ppl': Perplexity(),
        })
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix='train/')
        self.valid_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')

        # generative perplexity
        self.gen_ppl_metric = Perplexity()
        self.noise = noise_schedule.get_noise(self.config,
                                            dtype=self.dtype)
        
        # ema
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(),
                                self.noise.parameters()),
                decay=self.config.training.ema)
        else:
            self.ema = None
    
        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._validate_configuration()

        # subset of data for evaluation
        if eval:
            self.eval_sets_sp = oracle.subset_for_eval(n=config.eval.subset_size) 
            self.eval_sets_sp_clss = oracle.subset_eval_groundtruth(self.eval_sets_sp)
            self.eval_sets_sp_preds = oracle.subset_eval_preds(self.eval_sets_sp) 
            self.eval_sets_sp_kmers = oracle.subset_eval_kmers(self.eval_sets_sp) 
            self.emb_pca = oracle.cal_emb_pca(oracle.subset_for_eval(n=40000), n_components=50)
            self.eval_sets_sp_embs_pca = oracle.subset_eval_embs_pca(self.eval_sets_sp, self.emb_pca) 
  
    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        assert self.parameterization == 'subs'


    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
                'current']['completed']

    
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
                'completed'] = checkpoint['loops']['fit_loop'][
                'epoch_loop.automatic_optimization.optim_progress'][
                    'optimizer']['step']['total'][
                    'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
                'completed'] = checkpoint['loops']['fit_loop'][
                'epoch_loop.automatic_optimization.optim_progress'][
                    'optimizer']['step']['current'][
                    'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
                '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
                'epoch_loop.automatic_optimization.optim_progress'][
                    'optimizer']['step']['total']['completed']
        if 'sampler' not in checkpoint.keys():
            checkpoint['sampler'] = {}
        if hasattr(self.trainer.train_dataloader.sampler, 'state_dict'):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint['sampler']['random_state'] = sampler_state_dict.get('random_state', None)
        else:
            checkpoint['sampler']['random_state'] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed)
        
        print('distributed:', distributed)

        if distributed:
            sampler_cls = dataloader_gosai.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader_gosai.RandomFaultTolerantSampler
        
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
                if (distributed and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                    
                    dl_sampler.load_state_dict({
                        'epoch': self.fast_forward_epochs,
                        'counter': (self.fast_forward_batches
                                    * self.config.loader.batch_size)})
                updated_dls.append(
                    torch.utils.data.DataLoader(
                        dl.dataset,
                        batch_size=self.config.loader.batch_size,
                        num_workers=self.config.loader.num_workers,
                        pin_memory=self.config.loader.pin_memory,
                        sampler=dl_sampler,
                        shuffle=False,
                        persistent_workers=True))
                
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(
                self.backbone.parameters(),
                self.noise.parameters()))

    # subs parameterization from MDLM
    def _subs_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        if xt.ndim > 2 and xt.shape[-1] == self.vocab_size:
            # this is for finetuning setting when the input is one-hot encoded or probs
            xt = xt.argmax(dim=-1)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == 'ar'
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma):
        """Returns log score."""
        sigma = self._process_sigma(sigma)
        
        x = x.to(dtype=torch.long)
        
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.backbone(x, sigma)
            
        if self.parameterization == 'subs':
            return self._subs_parameterization(logits=logits, xt=x)
        
        return logits

    # might need changing to match wdce loss
    def _compute_loss(self, batch, prefix):
        
        if 'attention_mask' in batch:
            attention_mask = batch['attention_mask']
        else:
            attention_mask = None
        losses = self._loss(batch['seqs'], attention_mask)
        loss = losses.loss

        
        if prefix == 'train':
            self.train_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.train_metrics
        elif prefix == 'val':
            self.valid_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.valid_metrics
        elif prefix == 'test':
            self.test_metrics.update(losses.nlls, losses.token_mask)
            metrics = self.test_metrics
        else:
            raise ValueError(f'Invalid prefix: {prefix}')

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, prefix='train')
        self.log(name='trainer/loss',
                value=loss.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(itertools.chain(
                    self.backbone.parameters(),
                    self.noise.parameters()))
            self.ema.copy_to(itertools.chain(
                    self.backbone.parameters(),
                    self.noise.parameters()))
        self.backbone.eval()
        self.noise.eval()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, prefix='val')

    def on_validation_epoch_end(self):
        if ((self.config.eval.compute_perplexity_on_sanity
                or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples
                and not self.parameterization == 'ar'):
            all_samples, all_detoeknized_samples = [], []
            
            for _ in range(self.config.sampling.num_sample_batches):
                
                samples = self._sample().detach().cpu().numpy()
                detokenized_samples = dataloader_gosai.batch_dna_detokenize(samples)
                all_samples.append(samples)
                all_detoeknized_samples.extend(detokenized_samples)
                
            all_samples = np.concatenate(all_samples, axis=0)
            ws_distance_dict = self.cal_wasserstein_distance(all_detoeknized_samples)
            pearsonr_list = self.cal_kmer_pearsonr(all_detoeknized_samples)
            ws_embpca_list = self.cal_ws_distance_embpca(all_detoeknized_samples)
      
        current_step = self.trainer.global_step
        LOGGER.info(f'Current step: {current_step}')
        LOGGER.info(f'Wasserstein distance: {ws_distance_dict}')
        LOGGER.info(f'3mer Pearsonr: {pearsonr_list}')
        LOGGER.info(f'Wasserstein distance embpca: {ws_embpca_list}')
        self.log('val/3mer_pearsonr', pearsonr_list, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/ws_embpca', ws_embpca_list, on_step=False, on_epoch=True, sync_dist=True)

        for key in ws_distance_dict:
            for cell_type in ws_distance_dict[key]:
                metric_values = ws_distance_dict[key][cell_type]
                if metric_values:  # Check if the list is not empty
                    # Assuming metric_values contains [train_metric, valid_metric, test_metric]
                    self.log(f'val/{key}_{cell_type}', metric_values[0], on_step=False, on_epoch=True, sync_dist=True)

        if self.ema:
            self.ema.restore(itertools.chain(self.backbone.parameters(),
                                self.noise.parameters()))
            
    ### VALIDATION METRICS ###
    def cal_wasserstein_distance(self, seqs):
        generated_preds = oracle.cal_gosai_pred_new(seqs)
        ws_distance_dict = {'truth': {'hepg2': [], 'k562': [], 'sknsh': []}, 
                            'preds': {'hepg2': [], 'k562': [], 'sknsh': []}} 
        ws_distance_dict['truth']['hepg2'].append(wasserstein_distance(generated_preds[:, 0], self.eval_sets_sp_clss[:, 0]))
        ws_distance_dict['truth']['k562'].append(wasserstein_distance(generated_preds[:, 1], self.eval_sets_sp_clss[:, 1]))
        ws_distance_dict['truth']['sknsh'].append(wasserstein_distance(generated_preds[:, 2], self.eval_sets_sp_clss[:, 2]))   
        ws_distance_dict['preds']['hepg2'].append(wasserstein_distance(generated_preds[:, 0], self.eval_sets_sp_preds[:, 0]))
        ws_distance_dict['preds']['k562'].append(wasserstein_distance(generated_preds[:, 1], self.eval_sets_sp_preds[:, 1]))
        ws_distance_dict['preds']['sknsh'].append(wasserstein_distance(generated_preds[:, 2], self.eval_sets_sp_preds[:, 2])) 
        return ws_distance_dict

    def cal_ws_distance_embpca(self, seqs):
        generated_embs = oracle.cal_gosai_emb(seqs)
        generated_embs_pca = self.emb_pca.transform(generated_embs.reshape(generated_embs.shape[0], -1))
        return oracle.get_wasserstein_dist(generated_embs_pca, self.eval_sets_sp_embs_pca)
  
    def compare_kmer(self, kmer1, kmer2, n_sp1, n_sp2):
        kmer_set = set(kmer1.keys()) | set(kmer2.keys())
        counts = np.zeros((len(kmer_set), 2))
        for i, kmer in enumerate(kmer_set):
            if kmer in kmer1:
                counts[i][1] = kmer1[kmer] * n_sp2 / n_sp1
            if kmer in kmer2:
                counts[i][0] = kmer2[kmer]
        return pearsonr(counts[:, 0], counts[:, 1])[0]

    def cal_kmer_pearsonr(self, seqs):
        generated_kmer = oracle.count_kmers(seqs)
        return self.compare_kmer(self.eval_sets_sp_kmers, generated_kmer, self.config.eval.subset_size, len(seqs))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(),
                            self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(* x.shape, device=x.device) < move_chance
        
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _sample_prior(self, *batch_dims):
        """
            Returns array of fully masked sequences with same shape as input
        """
        return self.mask_index * torch.ones(* batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()
        
        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        
        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt, return_process=False):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t) # t
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        if return_process:
            return copy_flag * x + (1 - copy_flag) * _x, x, unet_conditioning, move_chance_t, copy_flag
        else:
            return copy_flag * x + (1 - copy_flag) * _x
  
    def _ar_sampler(self, bsz):
        # precompute token buffer
        num_pred_tokens = self.config.model.length - 1
        x = torch.zeros(
            (bsz, num_pred_tokens + 1),
            dtype=torch.long,
            device=self.device)
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (torch.distributions.Gumbel(0, 1)
                .sample((bsz, num_pred_tokens, self.vocab_size))
                .to(self.device))
        for i in range(num_pred_tokens):
            next_logits = self.forward(x[:, :i + 1], None)[:, -1]
            y = (next_logits + noise[:, i]).argmax(-1)
            x[:, i + 1] = y
        return x

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5, eval_sp_size=None):
        """Generate samples from the model."""
        if eval_sp_size is None:
            batch_size_per_gpu = self.config.loader.eval_batch_size
        else:
            batch_size_per_gpu = eval_sp_size
        if self.parameterization == 'ar':
            return self._ar_sampler(batch_size_per_gpu)
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length).to(self.device)
        
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            
            if self.sampler == 'ddpm':
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == 'ddpm_cache':
                p_x0_cache, x_next = self._ddpm_caching_update(x, t, dt, p_x0=p_x0_cache)
                if (not torch.allclose(x_next, x) or self.time_conditioning):
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                        device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                logits = self.forward(x, unet_conditioning)
                x = logits[:, :, :-1].argmax(dim=-1)
        return x
    
    ### FOR THE EXPANSION AND ROLLOUT STEP ###
    def sample_finetuned_with_rnd(self, args, reward_model,pretrained, eps=1e-5):
        num_steps = args.total_num_steps
        x_rollout = self._sample_prior(
            args.batch_size,
            args.seq_length).to(self.device)
        
        log_rnd = torch.zeros(args.batch_size, device=self.device)
        
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x_rollout.shape[0], 1, device=self.device)
            
            log_p, x_next, log_policy_step, log_pretrained_step = self.mcts_reverse_step(x_rollout, t=t, dt=dt, pretrained=pretrained)            
            log_rnd += log_pretrained_step - log_policy_step
            
            x_rollout = x_next
            
        # if mask token remains, fully unmask
        mask_positions = (x_rollout == self.mask_index)        # (B, L) bool

        # does **any** mask remain in any sequence
        any_mask_global = mask_positions.any().item()  # true if mask remains
        if any_mask_global:
            log_p, x_next = self.single_noise_removal(x_rollout, t=t, dt=dt)
            
            x_rollout = x_next
        
        x_final = x_rollout
        
        x_one_hot = to_one_hot(x_final)
        x_one_hot_reward = torch.transpose(x_one_hot, 1, 2)
        reward_preds = reward_model(x_one_hot_reward).squeeze(-1) # (num_children, 4)
        rewards = reward_preds[:, 0] # (num_children, 1)
        log_rnd = log_rnd + rewards / args.alpha
        mean_reward = rewards.mean()
        
        return x_final, log_rnd, rewards
    
    def sample_finetuned(self, args, reward_model, eps=1e-5):
        num_steps = args.total_num_steps
        x_rollout = self._sample_prior(
            args.batch_size,
            args.seq_length).to(self.device)
        
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        
        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x_rollout.shape[0], 1, device=self.device)
            
            log_p, x_next = self.single_reverse_step(x_rollout, t=t, dt=dt)
                        
            x_rollout = x_next
            
        # if mask token remains, fully unmask
        mask_positions = (x_rollout == self.mask_index)        # (B, L) bool

        # does **any** mask remain in any sequence
        any_mask_global = mask_positions.any().item()  # true if mask remains
        if any_mask_global:
            log_p, x_next = self.single_noise_removal(x_rollout, t=t, dt=dt)
            
            x_rollout = x_next
        
        x_final = x_rollout
        
        x_one_hot = to_one_hot(x_final)
        x_one_hot_reward = torch.transpose(x_one_hot, 1, 2)
        reward_preds = reward_model(x_one_hot_reward).squeeze(-1) # (num_children, 4)
        rewards = reward_preds[:, 0] # (num_children, 1)
                
        mean_reward = rewards.mean()
        
        return x_final, mean_reward
    
    def compute_log_policy(self, token_array, x_next, t, dt):
        sigma_t, _ = self.noise(t)
        
        if token_array.ndim == 1:
            token_array = token_array.unsqueeze(0)
            
        if x_next.ndim == 1:
            x_next = x_next.unsqueeze(0)
                    
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        log_p = self.forward(token_array, sigma=sigma_t)
        p_x0 = log_p.exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_index] = change_prob_s[:, :, 0]
        
        copy_flag = (token_array != self.mask_index)
        
        assert copy_flag.dtype == torch.bool, "copy_flag must be bool"
        changed_mask = (~copy_flag)
        
        # compute the per-sequence log-probability under the pretrained model 
        log_policy_token = log_p.gather(-1, x_next.unsqueeze(-1)).squeeze(-1)
        
        unmasked_this_step = (changed_mask & (x_next != self.mask_index)).to(log_policy_token.dtype)
        log_policy_step = (log_policy_token * unmasked_this_step).sum(dim=-1) 
        
        # returns:
        # log_policy_step (B, ) log probability x_next tokens under policy
        if log_policy_step.ndim == 1:
            log_policy_step = log_policy_step.squeeze(0)
            
        return log_policy_step
    
    
    def single_reverse_step(self, token_array, t, dt, p_x0=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if p_x0 is None:
            log_p = self.forward(token_array, sigma=sigma_t)
            p_x0 = log_p.exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_index] = change_prob_s[:, :, 0]
        
        x_changed = _sample_categorical(q_xs)
        
        copy_flag = (token_array != self.mask_index)
        
        int_copy_flag = copy_flag.to(token_array.dtype)
        x_next = int_copy_flag * token_array + (1 - int_copy_flag) * x_changed
        
        # returns:
        # log_p (B, L, D) log probabilties of each token under the policy model
        # x_next (B, L) next sequences
        return log_p, x_next
    
    
    def single_noise_removal(self, token_array, t, dt, p_x0=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if p_x0 is None:
            log_p = self.forward(token_array, sigma=sigma_t)
            p_x0 = log_p.exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        # changed for noise removal
        p_x0 = p_x0.clone()
        p_x0[:, :, self.mask_index] = 0.0 # prevent remaining a mask
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # renorm over non-MASK
        q_xs = p_x0 * (change_prob_t - change_prob_s)   
        
        x_changed = _sample_categorical(q_xs)
        
        copy_flag = (token_array != self.mask_index)
        
        int_copy_flag = copy_flag.to(token_array.dtype)
        x_next = int_copy_flag * token_array + (1 - int_copy_flag) * x_changed

        # returns:
        # log_p (B, L, D) log probabilties of each token under the policy model
        # x_next (B, L) next sequences
        return log_p, x_next
    
    def mcts_reverse_step(self, token_array, t, dt, pretrained, p_x0=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if p_x0 is None:
            log_p = self.forward(token_array, sigma=sigma_t)
            p_x0 = log_p.exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_index] = change_prob_s[:, :, 0]
        
        x_changed = _sample_categorical(q_xs)
        
        copy_flag = (token_array != self.mask_index)
        
        int_copy_flag = copy_flag.to(token_array.dtype)
        x_next = int_copy_flag * token_array + (1 - int_copy_flag) * x_changed

        # compute the log-probability under pretrained model at each step
        with torch.no_grad():
            # pretrained should output log-probs over vocab at each position given the *parent* (masked) input
            log_pre = pretrained.forward(token_array, sigma=sigma_t)

            # log-prob of the *sampled token* at each position
            log_pre_token = log_pre.gather(-1, x_next.unsqueeze(-1)).squeeze(-1)  # [B*batch,L]

            # sum only over the sites actually sampled this step (i.e., where parent was mask)
            
            assert copy_flag.dtype == torch.bool, "copy_flag must be bool"
            changed_mask = (~copy_flag)
            # mask of tokens that were unmasked in this step
            unmasked_this_step = (changed_mask & (x_next != self.mask_index)).to(log_pre_token.dtype)
            
            log_pretrained_step = (log_pre_token * unmasked_this_step).sum(dim=-1)
        
        # compute the per-sequence log-probability under the pretrained model 
        log_policy_token = log_p.gather(-1, x_next.unsqueeze(-1)).squeeze(-1)      # [B*batch,L]
        log_policy_step = (log_policy_token * unmasked_this_step).sum(dim=-1) 
        
        # returns:
        # log_p (B, L, D) log probabilties of each token under the policy model
        # x_next (B, L) next sequences
        # log_policy_step (B, ) log probability of all unmasked tokens under policy
        # log_pretrained_step (B, ) log probabiltiy of all unmasked tokens under pretrained model
        return log_p, x_next, log_policy_step, log_pretrained_step
    
    def mcts_noise_removal(self, token_array, t, dt, pretrained, p_x0=None):
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if p_x0 is None:
            log_p = self.forward(token_array, sigma=sigma_t)
            p_x0 = log_p.exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        # changed for noise removal
        p_x0 = p_x0.clone()
        p_x0[:, :, self.mask_index] = 0.0 # prevent remaining a mask
        p_x0 = p_x0 / p_x0.sum(dim=-1, keepdim=True).clamp_min(1e-12)  # renorm over non-MASK
        q_xs = p_x0 * (change_prob_t - change_prob_s)   
        
        x_changed = _sample_categorical(q_xs)
        
        copy_flag = (token_array != self.mask_index)
        
        int_copy_flag = copy_flag.to(token_array.dtype)
        x_next = int_copy_flag * token_array + (1 - int_copy_flag) * x_changed

        # compute the log-probability under pretrained model at each step
        with torch.no_grad():
            # pretrained should output log-probs over vocab at each position given the *parent* (masked) input
            log_pre = pretrained.forward(token_array, sigma=sigma_t)

            # log-prob of the *sampled token* at each position
            log_pre_token = log_pre.gather(-1, x_next.unsqueeze(-1)).squeeze(-1)  # [B*batch,L]

            # sum only over the sites actually sampled this step (i.e., where parent was mask)
            
            assert copy_flag.dtype == torch.bool, "copy_flag must be bool"
            changed_mask = (~copy_flag)
            # mask of tokens that were unmasked in this step
            unmasked_this_step = (changed_mask & (x_next != self.mask_index)).to(log_pre_token.dtype)
            
            log_pretrained_step = (log_pre_token * unmasked_this_step).sum(dim=-1)
        
        # compute the per-sequence log-probability under the pretrained model 
        log_policy_token = log_p.gather(-1, x_next.unsqueeze(-1)).squeeze(-1)      # [B*batch,L]
        log_policy_step = (log_policy_token * unmasked_this_step).sum(dim=-1) 
        
        # returns:
        # log_p (B, L, D) log probabilties of each token under the policy model
        # x_next (B, L) next sequences
        # log_policy_step (B, ) log probability of all unmasked tokens under policy
        # log_pretrained_step (B, ) log probabiltiy of all unmasked tokens under pretrained model
        return log_p, x_next, log_policy_step, log_pretrained_step
    
    # first step in expansion
    def batch_mcts_reverse_step(self, token_array, t, dt, batch_size, pretrained, p_x0=None):
        
        assert self.config.noise.type == 'loglinear'
        sigma_t, _ = self.noise(t)
        
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        
        change_prob_t = t[:, None, None]
        change_prob_s = (t - dt)[:, None, None]
        
        assert change_prob_t.ndim == 3, change_prob_t.shape
        
        if token_array.dim() == 1:
            token_array = token_array.unsqueeze(0)
        
        # expand to match (num_children, L)
        
        if p_x0 is None:
            log_p = self.forward(token_array, sigma=sigma_t)
            p_x0 = log_p.exp()
        
        assert change_prob_t.ndim == p_x0.ndim
        
        q_xs = p_x0 * (change_prob_t - change_prob_s)
        
        # zero-masking probability
        q_xs[:, :, self.mask_index] = change_prob_s[:, :, 0]
        
        # repeat the parent token along the first dimension which will be unmasked into distinct sequences
        token_array_expanded = token_array.repeat(batch_size, 1)
        
        if self.config.mcts.sampling == 0:
            x_changed = sample_batched_categorical(q_xs.to(self.device), batch_size)
        else:
            x_changed = sample_batched_top_k(q_xs.to(self.device), batch_size, self.config.mcts.sampling)
        
        copy_flag = (token_array_expanded != self.mask_index)
        
        int_copy_flag = copy_flag.to(token_array.dtype)
        x_children = int_copy_flag * token_array_expanded + (1 - int_copy_flag) * x_changed

        
        # compute the log-probability under pretrained model at each step
        with torch.no_grad():
            # pretrained should output log-probs over vocab at each position given the *parent* (masked) input
            log_pre = pretrained.forward(token_array, sigma=sigma_t)
            
            # expand to match the shape of x_children
            log_pre = log_pre.repeat(batch_size, 1, 1)

            # log-prob of the *sampled token* at each position
            log_pre_token = log_pre.gather(-1, x_children.unsqueeze(-1)).squeeze(-1)  # [B*batch,L]

            # sum only over the sites actually sampled this step (i.e., where parent was mask)
            
            assert copy_flag.dtype == torch.bool, "copy_flag must be bool"
            changed_mask = (~copy_flag)
            # mask of tokens that were unmasked in this step
            unmasked_this_step = (changed_mask & (x_children != self.mask_index)).to(log_pre_token.dtype)
            
            log_pretrained_step = (log_pre_token * unmasked_this_step).sum(dim=-1)
        
        # compute the per-child log-probability under the pretrained model 
        log_p = log_p.repeat(batch_size, 1, 1)
        log_policy_token = log_p.gather(-1, x_children.unsqueeze(-1)).squeeze(-1)  # (B, L) probability of each chosen token
        #print(log_policy_token)
        log_policy_step = (log_policy_token * unmasked_this_step).sum(dim=-1) 
        
        # returns:
        # log_p (B, L, D) log probabilties of each token under the policy model
        # x_children (B, L) child sequences
        # log_policy_step (B, ) log probability of all unmasked tokens under policy
        # log_pretrained_step (B, ) log probabiltiy of all unmasked tokens under pretrained model
        return log_p, x_children, log_policy_step, log_pretrained_step
    
    ### SPECIFIC TO DRAKES? ###
    def _ddpm_update_finetune_gradient(self, x, t, dt, copy_flag_temp, return_process=False):
    
        if x.ndim == 2 or x.shape[-1] != self.vocab_size:
            x = F.one_hot(x, num_classes=self.vocab_size).to(torch.float32)

        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t) # (1-eps)*t
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical_gradient(q_xs, temp=self.config.finetuning.gumbel_softmax_temp)
        
        if copy_flag_temp is not None:
            copy_flag_prob = 1 - x[:, :, self.mask_index].unsqueeze(-1)
            soft_copy_flag = torch.nn.functional.sigmoid(copy_flag_prob/copy_flag_temp)
        else:
            soft_copy_flag = 1 - x[:, :, self.mask_index].unsqueeze(-1)

        if return_process:
            return soft_copy_flag * x + (1 - soft_copy_flag) * _x, x, unet_conditioning, move_chance_t, soft_copy_flag
        else:
            return soft_copy_flag * x + (1 - soft_copy_flag) * _x
    
   
    def _sample_finetune_gradient(self, num_steps=None, eps=1e-5, eval_sp_size=None, copy_flag_temp=None):
        """Generate samples from the model."""
        assert self.parameterization == 'subs' and self.sampler == 'ddpm'
        if eval_sp_size is None:
            batch_size_per_gpu = self.config.loader.eval_batch_size
        else:
            batch_size_per_gpu = eval_sp_size
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length).to(self.device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        last_x_list = []
        condt_list = []
        move_chance_t_list = []
        copy_flag_list = []

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                    if i < num_steps - self.config.finetuning.truncate_steps:
                        x, last_x, condt, move_chance_t, copy_flag = self._ddpm_update(x, t, dt, return_process=True)
                        x = x.detach()
                        copy_flag = copy_flag.unsqueeze(-1)
                        last_x = F.one_hot(last_x, num_classes=self.vocab_size).to(torch.float32).detach()
                    else: 
                        x, last_x, condt, move_chance_t, copy_flag = self._ddpm_update_finetune_gradient(x, t, dt, copy_flag_temp, return_process=True)
                
            last_x_list.append(last_x)
            condt_list.append(condt)
            move_chance_t_list.append(move_chance_t)
            copy_flag_list.append(copy_flag)

        x_argmax = x[:, :, :-1].argmax(dim=-1)
        x_argmax = torch.nn.functional.one_hot(x_argmax, num_classes=self.vocab_size-1).to(torch.float32)
        return x[:, :, :-1] + (x_argmax - x[:, :, :-1]).detach(), last_x_list, condt_list, move_chance_t_list, copy_flag_list
  
    @torch.no_grad()
    def _ddpm_update_finetune_controlled_SMC(self, x, t, dt, reward_model, alpha = 1.0):

        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        copy_flag = (x != self.mask_index).to(x.dtype)
        sample = copy_flag * x + (1 - copy_flag) * _sample_categorical(q_xs)
        '''
        Calcualte exp(v_{t-1}(x_{t-1})/alpha)
        '''
        expected_x0 = self.forward(sample, sigma_s) # Calcualte E[x_0|x_{t-1}]
        expected_x0_arg = torch.argmax(expected_x0,dim=2)
        expected_x0_onehot = torch.nn.functional.one_hot(expected_x0_arg)
        reward_num = reward_model(expected_x0_onehot.float().transpose(1, 2)).detach()[:, 0][:, 0]
        '''
        Calcualte exp(v_{t}(x_{t})/alpha)
        '''
        expected_x0 = self.forward(x, sigma_s) # Calcualte E[x_0|x_t]
        expected_x0_arg = torch.argmax(expected_x0,dim=2) 
        expected_x0_onehot = torch.nn.functional.one_hot(expected_x0_arg)
        reward_den = reward_model(expected_x0_onehot.float().transpose(1, 2)).detach()[:, 0][:, 0]
    
        ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) # Now calculate exp( (v_{t-1}(x_{t-1) -v_{t}(x_{t}) /alpha) 
        ratio = ratio.detach().cpu().numpy()
        final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
    
        return sample[final_sample_indices]
  
    def _ddpm_update_finetune_controlled_CG(self, x, t, dt, reward_model,  guidance_scale):

        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        x_onehot = F.one_hot(x, num_classes=5).float()

        x_grad = self.compute_gradient_CG(x_onehot, x, reward_model, sigma_s )
        guidance = guidance_scale * (x_grad - x_grad[:, :, self.mask_index][:, :, None])
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        q_xs = q_xs * guidance.exp()

        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x 

    def compute_gradient_CG(self, x_onehot, x, reward_model, sigma_s):
        x_onehot.requires_grad_(True)
        expected_x0 = self.forward(x_onehot, sigma_s) # Calcualte E[x_0|x_t]
        scores = reward_model(expected_x0.transpose(1, 2)[:,0:4,:])[:, 0]
        scores = scores.mean()
        scores.backward()
        x_grad = x_onehot.grad.clone()
        return x_grad

    def _ddpm_update_finetune_controlled_TDS(self, x, t, dt, reward_model, alpha = 1.0, guidance_scale=1000):
        # SMC with the twisted proposal

        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        x_onehot = F.one_hot(x, num_classes=5).float()

        x_grad = self.compute_gradient_CG(x_onehot, x, reward_model, sigma_s )
        guidance = guidance_scale * (x_grad - x_grad[:, :, self.mask_index][:, :, None])
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        # print(q_xs.sum(-1))
        q_xs = q_xs * guidance.exp()

        _x = _sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        sample = copy_flag * x + (1 - copy_flag) * _x
        prob_multiplier = (1 - copy_flag) * torch.gather(guidance.exp(), 2, _x.unsqueeze(-1)).squeeze(-1) + copy_flag * torch.ones_like(_x)
        '''
        Calcualte exp(v_{t-1}(x_{t-1})/alpha)
        '''
        expected_x0 = self.forward(sample, sigma_s) # Calcualte E[x_0|x_{t-1}]
        expected_x0_arg = torch.argmax(expected_x0,dim=2)
        expected_x0_onehot = torch.nn.functional.one_hot(expected_x0_arg)
        reward_num = reward_model(expected_x0_onehot.float().transpose(1, 2)).detach()[:, 0][:, 0]
        '''
        Calcualte exp(v_{t}(x_{t})/alpha)
        '''
        expected_x0 = self.forward(x, sigma_s) # Calcualte E[x_0|x_t]
        expected_x0_arg = torch.argmax(expected_x0,dim=2) 
        expected_x0_onehot = torch.nn.functional.one_hot(expected_x0_arg)
        reward_den = reward_model(expected_x0_onehot.float().transpose(1, 2)).detach()[:, 0][:, 0]
        
        # set the nan values to 1
        prob_multiplier[torch.isnan(prob_multiplier)] = 1
        ratio = torch.exp(1.0/alpha * (reward_num - reward_den)) / prob_multiplier.prod(dim=-1)
        ratio = ratio.detach().cpu().numpy()
        final_sample_indices = np.random.choice(reward_num.shape[0], reward_num.shape[0], p =  ratio/ratio.sum() ) 
    
        return sample[final_sample_indices]
  
    @torch.no_grad()
    def controlled_sample_SMC(self, reward_model, alpha, num_steps=None, eps=1e-5, eval_sp_size=None):
        """Generate samples from the model."""
        if eval_sp_size is None:
            batch_size_per_gpu = self.config.loader.eval_batch_size
        else:
            batch_size_per_gpu = eval_sp_size
        if self.parameterization == 'ar':
            return self._ar_sampler(batch_size_per_gpu)
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length).to(self.device)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                    x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                    x  = self._ddpm_update_finetune_controlled_SMC(x, t, dt, reward_model, alpha)
            else:
                    x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == 'analytic':
                    x = self._denoiser_update(x, t)
            else:
                    unet_conditioning = self.noise(t)[0]
                    logits = self.forward(x, unet_conditioning)
                    x = logits[:, :, :-1].argmax(dim=-1)
        return x

    def controlled_sample_CG(self, reward_model, guidance_scale, num_steps=None, eps=1e-5, eval_sp_size=None):
        """Generate samples from the model."""
        if eval_sp_size is None:
            batch_size_per_gpu = self.config.loader.eval_batch_size
        else:
            batch_size_per_gpu = eval_sp_size
        if self.parameterization == 'ar':
            return self._ar_sampler(batch_size_per_gpu)
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length).to(self.device)
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x  = self._ddpm_update_finetune_controlled_CG(x, t, dt, reward_model, guidance_scale)
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                        device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                logits = self.forward(x, unet_conditioning)
                x = logits[:, :, :-1].argmax(dim=-1)
        return x

    def controlled_sample_TDS(self, reward_model, alpha, guidance_scale, num_steps=None, eps=1e-5, eval_sp_size=None):
        """Generate samples from the model."""
        if eval_sp_size is None:
            batch_size_per_gpu = self.config.loader.eval_batch_size
        else:
            batch_size_per_gpu = eval_sp_size
            
        if self.parameterization == 'ar':
            return self._ar_sampler(batch_size_per_gpu)

        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self._sample_prior(
            batch_size_per_gpu,
            self.config.model.length).to(self.device)
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=self.device)
            if self.sampler == 'ddpm':
                x  = self._ddpm_update_finetune_controlled_TDS(x, t, dt, reward_model,alpha, guidance_scale)
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1,
                                        device=self.device)
            if self.sampler == 'analytic':
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                logits = self.forward(x, unet_conditioning)
                x = logits[:, :, :-1].argmax(dim=-1)
        return x

    @torch.no_grad()
    def get_likelihood(self, x0, num_steps=None, eps=1e-5, n_samples=1):
        """Compute the likelihood of a sequence under the model.
        x0: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length)
        """
        if num_steps is None:
            num_steps = self.config.sampling.steps
        timesteps = torch.linspace(
            1, eps, num_steps + 1, device=self.device) # t=0 is clean data
        dt = (1 - eps) / num_steps
        log_p_sample_list = []
        for _ in range(n_samples):
            log_p_at_time_list = []
            for i in range(num_steps):
                t = timesteps[i] * torch.ones(
                x0.shape[0], 1, device=self.device)
                sigma_t, _ = self.noise(t)
                sigma_s, _ = self.noise(t - dt)
                if sigma_t.ndim > 1:
                    sigma_t = sigma_t.squeeze(-1)
                if sigma_s.ndim > 1:
                    sigma_s = sigma_s.squeeze(-1)
                assert sigma_t.ndim == 1, sigma_t.shape
                assert sigma_s.ndim == 1, sigma_s.shape
                move_chance_t = 1 - torch.exp(-sigma_t) # (1-eps)*t
                move_chance_s = 1 - torch.exp(-sigma_s)
                move_chance_t = move_chance_t[:, None] # [bsz, 1]
                move_chance_s = move_chance_s[:, None]
                unet_conditioning = sigma_t # [bsz]
                multiplier = (move_chance_t - move_chance_s)/move_chance_t # [bsz, 1]
                xt = self.q_xt(x0, move_chance_t) # [bsz, seq_len]
                # log prob, already apply subs parametrization (unmasked token remains unchanged)
                model_output = self.forward(xt, unet_conditioning) # [bsz, seq_len, vocab_size]
                # take the log prob of the token that corresponds to x0
                log_p_x0 = model_output.gather(-1, x0[..., None]).squeeze(-1) # [bsz, seq_len]
                log_p_x0 = log_p_x0 * multiplier
                log_p_at_time_list.append(log_p_x0)
            log_p_x0 = torch.stack(log_p_at_time_list, dim=0).sum(dim=0) # [bsz, seq_len]
            log_p_sample_list.append(log_p_x0.sum(dim=-1))
        log_p_sample = torch.stack(log_p_sample_list, dim=0).mean(dim=0)
        return log_p_sample

    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        if self.parameterization == 'subs':
            # score(x, t) = p_t(y) / p_t(x)
            # => log score(x, t) = log p_t(y) - log p_t(x)
            
            # case 1: x = masked
            #   (i) y = unmasked
            #     log score(x, t) = log p_\theta(x)|_y + log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))
            #   (ii) y = masked
            #     log score(x, t) = 0

            # case 2: x = unmasked
            #   (i) y != masked, y != x
            #     log score(x_i, t) = - inf
            #   (ii) y = x 
            #     log score(x_i, t) = 0
            #   (iii) y = masked token
            #     log score(x_i, t) = - log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))
            
            log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1
        
            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, self.mask_index] = 0

            unmasked_score = self.neg_infinity * torch.ones_like(
                model_output)
            unmasked_score = torch.scatter(
                unmasked_score,
                -1,
                x[..., None],
                torch.zeros_like(unmasked_score[..., :1]))
            unmasked_score[:, :, self.mask_index] = - (
                log_k[:, None] * torch.ones_like(x))
        
            masked_indices = (x == self.mask_index).to(
                model_output.dtype)[:, :, None]
            model_output = (
                masked_score * masked_indices
                + unmasked_score * (1 - masked_indices))
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, step_size):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return _sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = _sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(
            i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index,
                            1 - torch.exp(-sigma).squeeze(-1),
                            0)[..., None]
        return edge

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            # for variance reduction
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        if seqlen > self.config.model.length:
            raise NotImplementedError('Sub-sampling not implemented')
        elif self.parameterization == 'ar':
            input_tokens = x0[:, :-1]
            output_tokens = x0[:, 1:]
            new_attention_mask = attention_mask[:, 1:]
        else:
            input_tokens = x0
            output_tokens = None
            new_attention_mask = attention_mask
        return input_tokens, output_tokens, new_attention_mask

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(x0.shape[0], dtype=self.dtype,
                        device=self.device)
        assert self.config.noise.type == 'loglinear'
        # The above assert is for d3pm parameterization
        unet_conditioning = self.noise(t0)[0][:, None]
        model_output_t0 = self.forward(x0, unet_conditioning)
        return - torch.gather(input=model_output_t0,
                            dim=-1,
                            index=x0[:, :, None]).squeeze(-1)

    def _forward_pass_diffusion(self, x0):
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            # else ts are between 0 and 1
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)

        if self.change_of_variables: # False
            unet_conditioning = t[:, None]
            f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
            f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t) # total noise, rate noise
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance) # q(xt|x0)
        model_output = self.forward(xt, unet_conditioning)
        utils.print_nans(model_output, 'model_output')

        if self.parameterization == 'sedd':
            return dsigma[:, None] * self._score_entropy(
                model_output, sigma[:, None], xt, x0)
        
        if self.T > 0:
            diffusion_loss = self._d3pm_loss(
                    model_output=model_output, xt=xt, x0=x0, t=t)
            if self.parameterization == 'd3pm':
                    reconstruction_loss = self._reconstruction_loss(x0)
            elif self.parameterization == 'subs':
                    reconstruction_loss = 0
            return reconstruction_loss + diffusion_loss
        
        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        
        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(
                - torch.exp(- self.noise.sigma_min))
        
        return - log_p_theta * (
            dsigma / torch.expm1(sigma))[:, None]

    def _loss(self, x0, attention_mask):
        (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(
                x0, attention_mask)

        if self.parameterization == 'ar':
            logprobs = self.backbone(input_tokens, None)
            loss = - logprobs.gather(
                -1, output_tokens[:, :, None])[:, :, 0]
        else:
            loss = self._forward_pass_diffusion(input_tokens)
    
        nlls = loss * attention_mask
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        return Loss(loss=token_nll,
                    nlls=nlls,
                    token_mask=attention_mask)

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
        log_score: float torch.Tensor with shape (batch_size,
            diffusion_model_input_length, vocab_size),
            log score, output of the denoising network.
        xt: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        x0: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input.
        sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
        loss with shape (batch_size, diffusion_model_input_length)
        """
        # seems that it takes y=x0,xt=M case
        # what is the const term for, seems to be y=M,xt=x0 case and x0 is known so score estimation is precise
        masked_indices = xt == self.mask_index

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_score[masked_indices],
            -1,
            words_that_were_masked[..., None]).squeeze(-1)
        score = log_score[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(
                dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(* xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy

