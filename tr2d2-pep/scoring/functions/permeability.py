import sys
import os
import xgboost as xgb
import torch
import numpy as np
from transformers import AutoModelForMaskedLM
import warnings
import numpy as np
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import Chem, rdBase, DataStructs
from rdkit.Chem import AllChem
from typing import List
from transformers import AutoModelForMaskedLM


rdBase.DisableLog('rdApp.error')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def fingerprints_from_smiles(smiles: List, size=2048):
    """ Create ECFP fingerprints of smiles, with validity check """
    fps = []
    valid_mask = []
    for i, smile in enumerate(smiles):
        mol = Chem.MolFromSmiles(smile)
        valid_mask.append(int(mol is not None))
        fp = fingerprints_from_mol(mol, size=size) if mol else np.zeros((1, size))
        fps.append(fp)

    fps = np.concatenate(fps, axis=0)
    return fps, valid_mask


def fingerprints_from_mol(molecule, radius=3, size=2048, hashed=False):
    """ Create ECFP fingerprint of a molecule """
    if hashed:
        fp_bits = AllChem.GetHashedMorganFingerprint(molecule, radius, nBits=size)
    else:
        fp_bits = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=size)
    fp_np = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp_bits, fp_np)
    return fp_np.reshape(1, -1)

def getMolDescriptors(mol, missingVal=0):
    """ calculate the full list of descriptors for a molecule """

    values, names = [], []
    for nm, fn in Descriptors._descList:
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)

    custom_descriptors = {'hydrogen-bond donors': rdMolDescriptors.CalcNumLipinskiHBD,
                          'hydrogen-bond acceptors': rdMolDescriptors.CalcNumLipinskiHBA,
                          'rotatable bonds': rdMolDescriptors.CalcNumRotatableBonds,}
    
    for nm, fn in custom_descriptors.items():
        try:
            val = fn(mol)
        except:
            val = missingVal
        values.append(val)
        names.append(nm)
    return values, names

def get_pep_dps_from_smi(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        print(f"convert smi {smi} to molecule failed!")
        mol = None
    
    dps, _ = getMolDescriptors(mol)
    return np.array(dps)


def get_pep_dps(smi_list):
    if len(smi_list) == 0:
        return np.zeros((0, 213))
    return np.array([get_pep_dps_from_smi(smi) for smi in smi_list])

def check_smi_validity(smiles: list):
    valid_smi, valid_idx = [], []
    for idx, smi in enumerate(smiles):
        try:
            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol:
                valid_smi.append(smi)
                valid_idx.append(idx)
        except Exception as e:
            # logger.debug(f'Error: {e} in smiles {smi}')
            pass 
    return valid_smi, valid_idx

class Permeability:
    
    def __init__(self, tokenizer, base_path, device=None, emb_model=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.predictor = xgb.Booster(model_file=f'{base_path}/TR2-D2/tr2d2-pep/scoring/functions/classifiers/permeability-xgboost.json')
        if emb_model is not None:
            self.emb_model = emb_model.to(self.device).eval()
        else:
            self.emb_model = AutoModelForMaskedLM.from_pretrained('aaronfeller/PeptideCLM-23M-all').roformer.to(device).eval()
        
        self.tokenizer = tokenizer

    def generate_embeddings(self, sequences):
        embeddings = []
        for sequence in sequences:
            tokenized = self.tokenizer(sequence, return_tensors='pt')
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            with torch.no_grad():
                output = self.emb_model(**tokenized)
            # Mean pooling across sequence length
            embedding = output.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def get_features(self, input_seqs: list, dps=False, fps=False):
        #valid_smiles, valid_idxes = check_smi_validity(input_seqs)
    

        if fps: 
            fingerprints = fingerprints_from_smiles(input_seqs)[0]
        else:
            fingerprints = torch.empty((len(input_seqs), 0))
        
        if dps:
            descriptors = get_pep_dps(input_seqs)
        else: 
            descriptors = torch.empty((len(input_seqs), 0))
            
        embeddings = self.generate_embeddings(input_seqs)
        # logger.debug(f'X_fps.shape: {X_fps.shape}, X_dps.shape: {X_dps.shape}')

        features = np.concatenate([fingerprints, descriptors, embeddings], axis=1)
            
        return features
    
    def get_scores(self, input_seqs: list):
        scores = -10 * np.ones(len(input_seqs))
        features = self.get_features(input_seqs)
        
        if len(features) == 0:
            return scores
        
        features = np.nan_to_num(features, nan=0.)
        features = np.clip(features, np.finfo(np.float32).min, np.finfo(np.float32).max)
        
        features = xgb.DMatrix(features)
        
        scores = self.predictor.predict(features)
        return scores 
    
    def __call__(self, input_seqs: list):
        scores = self.get_scores(input_seqs)
        return scores
    
def unittest():
    permeability = Permeability()
    seq = ['N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1cNc2c1cc(O)cc2)C(=O)N[C@@H](CC1=CN=C-N1)C(=O)N[C@@H](CCCNC(=N)N)C(=O)N[C@@H](Cc1ccccc1)C(=O)N[C@@H](CCC(=O)O)C(=O)N[C@@H]([C@@H](O)C(C)C)C(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)N[C@H](CC(=CN2)C1=C2C=CC=C1)C(=O)O']
    scores = permeability(input_seqs=seq)
    print(scores)


if __name__ == '__main__':
    unittest()