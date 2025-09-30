from transformers import RoFormerConfig, RoFormerForMaskedLM
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

class Roformer(nn.Module):
    def __init__(self, config, tokenizer, device=None):
        super(Roformer, self).__init__()
        
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device

        
        roformer_config = RoFormerConfig(
            vocab_size=self.tokenizer.vocab_size,
            embedding_size=config.roformer.hidden_size,
            hidden_size=config.roformer.hidden_size,
            num_hidden_layers=config.roformer.n_layers,
            num_attention_heads=config.roformer.n_heads,
            intermediate_size=config.roformer.hidden_size * 4,
            max_position_embeddings=config.roformer.max_position_embeddings,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=0,
            rotary_value=False
        )
        
        self.model = RoFormerForMaskedLM(roformer_config).to(self.device)
                
    def freeze_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True
    
    def unfreeze_n_layers(self, n):
        num_layers = 8
        
        for i, layer in enumerate(self.model.roformer.encoder.layer):
            # finetune final n layers
            if i >= num_layers - n:
                # unfreeze query weights
                for module in layer.attention.self.query.modules():
                    for param in module.parameters():
                         param.requires_grad = True
                # unfreeze key weights
                for module in layer.attention.self.key.modules():
                    for param in module.parameters():
                        param.requires_grad = True
            
    def forward(self, input_ids, attn_mask):
        
        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        # get logits embeddings
        logits = self.model(input_ids=input_ids, attention_mask=attn_mask)
        # return logits
        #print(logits.logits)
        return logits.logits
    
    def save_model(self, save_dir):
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
    
    @classmethod
    def load_model(cls, save_dir, config, tokenizer):
        roformer = cls(config, tokenizer)
        roformer.model = RoFormerForMaskedLM.from_pretrained(save_dir)
        return roformer