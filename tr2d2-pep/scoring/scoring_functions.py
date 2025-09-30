from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from transformers import AutoModelForMaskedLM
import numpy as np
from scoring.functions.binding import BindingAffinity
from scoring.functions.permeability import Permeability
from scoring.functions.solubility import Solubility
from scoring.functions.hemolysis import Hemolysis
from scoring.functions.nonfouling import Nonfouling

base_path = '/path/to/your/home'

class ScoringFunctions:
    def __init__(self, score_func_names=None, prot_seqs=None, device=None):
        """
        Class for generating score vectors given generated sequence

        Args:
            score_func_names: list of scoring function names to be evaluated
            score_weights: weights to scale scores (default: 1)
            target_protein: sequence of target protein binder
        """
        emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device).eval()
        tokenizer = SMILES_SPE_Tokenizer(f'{base_path}/TR2-D2/tr2d2-pep/tokenizer/new_vocab.txt',
                                        f'{base_path}/TR2-D2/tr2d2-pep/tokenizer/new_splits.txt')
        prot_seqs = prot_seqs if prot_seqs is not None else []
        
        if score_func_names is None:
            # just do unmasking based on validity of peptide bonds
            self.score_func_names = []
        else:
            self.score_func_names = score_func_names
                
        # self.weights = np.array([1] * len(self.score_func_names) if score_weights is None else score_weights)
        
        # binding affinities
        self.target_protein = prot_seqs
        print(len(prot_seqs))
        
        if ('binding_affinity1' in score_func_names) and (len(prot_seqs) == 1):
            binding_affinity1 = BindingAffinity(prot_seqs[0], tokenizer=tokenizer, base_path=base_path, device=device)
            binding_affinity2 = None
        elif ('binding_affinity1' in score_func_names) and ('binding_affinity2' in score_func_names) and (len(prot_seqs) == 2):
            binding_affinity1 = BindingAffinity(prot_seqs[0], tokenizer=tokenizer, base_path=base_path, device=device)
            binding_affinity2 = BindingAffinity(prot_seqs[1], tokenizer=tokenizer, base_path=base_path, device=device)
        else:
            print("here")
            binding_affinity1 = None
            binding_affinity2 = None

        permeability = Permeability(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)
        sol = Solubility(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)
        nonfouling = Nonfouling(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)
        hemo = Hemolysis(tokenizer=tokenizer, base_path=base_path, device=device, emb_model=emb_model)

        self.all_funcs = {'binding_affinity1': binding_affinity1,
                          'binding_affinity2': binding_affinity2,
                          'permeability': permeability,
                          'nonfouling': nonfouling, 
                          'solubility': sol, 
                          'hemolysis': hemo
                          } 
        
    def forward(self, input_seqs):
        scores = []
        
        for i, score_func in enumerate(self.score_func_names): 
            score = self.all_funcs[score_func](input_seqs = input_seqs)
        
            scores.append(score)
            
        # convert to numpy arrays with shape (num_sequences, num_functions)
        scores = np.float32(scores).T
        
        return scores
    
    def __call__(self, input_seqs: list):
        return self.forward(input_seqs)