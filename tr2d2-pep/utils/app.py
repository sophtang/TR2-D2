import os
import re
import pandas as pd
from io import StringIO
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
import tempfile
from rdkit import Chem

class PeptideAnalyzer:
    def __init__(self):
        self.bond_patterns = [
            (r'OC\(=O\)', 'ester'),  # Ester bond
            (r'N\(C\)C\(=O\)', 'n_methyl'),  # N-methylated peptide bond
            (r'N[0-9]C\(=O\)', 'proline'),  # Proline peptide bond
            (r'NC\(=O\)', 'peptide'),  # Standard peptide bond
            (r'C\(=O\)N\(C\)', 'n_methyl_reverse'),  # Reverse N-methylated
            (r'C\(=O\)N[12]?', 'peptide_reverse')  # Reverse peptide bond
        ]
        # Three to one letter code mapping
        self.three_to_one = {
            'Ala': 'A', 'Cys': 'C', 'Asp': 'D', 'Glu': 'E',
            'Phe': 'F', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
            'Lys': 'K', 'Leu': 'L', 'Met': 'M', 'Asn': 'N',
            'Pro': 'P', 'Gln': 'Q', 'Arg': 'R', 'Ser': 'S',
            'Thr': 'T', 'Val': 'V', 'Trp': 'W', 'Tyr': 'Y'
        }
    
    def is_peptide(self, smiles):
        """Check if the SMILES represents a peptide structure"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
            
        # Look for peptide bonds: NC(=O) pattern
        peptide_bond_pattern = Chem.MolFromSmarts('[NH][C](=O)')
        if mol.HasSubstructMatch(peptide_bond_pattern):
            return True
            
        # Look for N-methylated peptide bonds: N(C)C(=O) pattern
        n_methyl_pattern = Chem.MolFromSmarts('[N;H0;$(NC)](C)[C](=O)')
        if mol.HasSubstructMatch(n_methyl_pattern):
            return True
        
        return False

    def is_cyclic(self, smiles):
        """Improved cyclic peptide detection"""
        # Check for C-terminal carboxyl
        if smiles.endswith('C(=O)O'):
            return False, [], []
            
        # Find all numbers used in ring closures
        ring_numbers = re.findall(r'(?:^|[^c])[0-9](?=[A-Z@\(\)])', smiles)
        
        # Find aromatic ring numbers
        aromatic_matches = re.findall(r'c[0-9](?:ccccc|c\[nH\]c)[0-9]', smiles)
        aromatic_cycles = []
        for match in aromatic_matches:
            numbers = re.findall(r'[0-9]', match)
            aromatic_cycles.extend(numbers)
        
        # Numbers that aren't part of aromatic rings are peptide cycles
        peptide_cycles = [n for n in ring_numbers if n not in aromatic_cycles]
        
        is_cyclic = len(peptide_cycles) > 0 and not smiles.endswith('C(=O)O')
        return is_cyclic, peptide_cycles, aromatic_cycles
    
    def split_on_bonds(self, smiles):
        """Split SMILES into segments with simplified Pro handling"""
        positions = []
        used = set()
        
        # Find Gly pattern first
        gly_pattern = r'NCC\(=O\)'
        for match in re.finditer(gly_pattern, smiles):
            if not any(p in range(match.start(), match.end()) for p in used):
                positions.append({
                    'start': match.start(),
                    'end': match.end(),
                    'type': 'gly',
                    'pattern': match.group()
                })
                used.update(range(match.start(), match.end()))
        
        for pattern, bond_type in self.bond_patterns:
            for match in re.finditer(pattern, smiles):
                if not any(p in range(match.start(), match.end()) for p in used):
                    positions.append({
                        'start': match.start(),
                        'end': match.end(),
                        'type': bond_type,
                        'pattern': match.group()
                    })
                    used.update(range(match.start(), match.end()))

        # Sort by position
        positions.sort(key=lambda x: x['start'])
        
        # Create segments
        segments = []
        
        if positions:
            # First segment
            if positions[0]['start'] > 0:
                segments.append({
                    'content': smiles[0:positions[0]['start']],
                    'bond_after': positions[0]['pattern']
                })
            
            # Process segments
            for i in range(len(positions)-1):
                current = positions[i]
                next_pos = positions[i+1]
                
                if current['type'] == 'gly':
                    segments.append({
                        'content': 'NCC(=O)',
                        'bond_before': positions[i-1]['pattern'] if i > 0 else None,
                        'bond_after': next_pos['pattern']
                    })
                else:
                    content = smiles[current['end']:next_pos['start']]
                    if content:
                        segments.append({
                            'content': content,
                            'bond_before': current['pattern'],
                            'bond_after': next_pos['pattern']
                        })
            
            # Last segment
            if positions[-1]['end'] < len(smiles):
                segments.append({
                    'content': smiles[positions[-1]['end']:],
                    'bond_before': positions[-1]['pattern']
                })
        
        return segments

    def clean_terminal_carboxyl(self, segment):
        """Remove C-terminal carboxyl only if it's the true terminus"""
        content = segment['content']
        
        # Only clean if:
        # 1. Contains C(=O)O
        # 2. No bond_after exists (meaning it's the last segment)
        # 3. C(=O)O is at the end of the content
        if 'C(=O)O' in content and not segment.get('bond_after'):
            print('recognized?')
            # Remove C(=O)O pattern regardless of position
            cleaned = re.sub(r'\(C\(=O\)O\)', '', content)
            # Remove any leftover empty parentheses
            cleaned = re.sub(r'\(\)', '', cleaned)
            print(cleaned)
            return cleaned
        return content
    
    def identify_residue(self, segment):
        """Identify residue with Pro reconstruction"""
        # Only clean terminal carboxyl if this is the last segment
        content = self.clean_terminal_carboxyl(segment)
        mods = self.get_modifications(segment)
        
        # UAA pattern matching section - before regular residues
        # Phenylglycine and derivatives
        if 'c1ccccc1' in content:
            if '[C@@H](c1ccccc1)' in content or '[C@H](c1ccccc1)' in content:
                return '4', mods  # Base phenylglycine
                
        # 4-substituted phenylalanines
        if 'Cc1ccc' in content:
            if 'OMe' in content or 'OCc1ccc' in content:
                return '0A1', mods  # 4-methoxy-Phenylalanine
            elif 'Clc1ccc' in content:
                return '200', mods  # 4-chloro-Phenylalanine
            elif 'Brc1ccc' in content:
                return '4BF', mods  # 4-Bromo-phenylalanine
            elif 'C#Nc1ccc' in content:
                return '4CF', mods  # 4-cyano-phenylalanine
            elif 'Ic1ccc' in content:
                return 'PHI', mods  # 4-Iodo-phenylalanine
            elif 'Fc1ccc' in content:
                return 'PFF', mods  # 4-Fluoro-phenylalanine
                
        # Modified tryptophans
        if 'c[nH]c2' in content:
            if 'Oc2cccc2' in content:
                return '0AF', mods  # 7-hydroxy-tryptophan
            elif 'Fc2cccc2' in content:
                return '4FW', mods  # 4-fluoro-tryptophan
            elif 'Clc2cccc2' in content:
                return '6CW', mods  # 6-chloro-tryptophan
            elif 'Brc2cccc2' in content:
                return 'BTR', mods  # 6-bromo-tryptophan
            elif 'COc2cccc2' in content:
                return 'MOT5', mods  # 5-Methoxy-tryptophan
            elif 'Cc2cccc2' in content:
                return 'MTR5', mods  # 5-Methyl-tryptophan
                
        # Special amino acids
        if 'CC(C)(C)[C@@H]' in content or 'CC(C)(C)[C@H]' in content:
            return 'BUG', mods  # Tertleucine
            
        if 'CCCNC(=N)N' in content:
            return 'CIR', mods  # Citrulline
            
        if '[SeH]' in content:
            return 'CSE', mods  # Selenocysteine
            
        if '[NH3]CC[C@@H]' in content or '[NH3]CC[C@H]' in content:
            return 'DAB', mods  # Diaminobutyric acid
            
        if 'C1CCCCC1' in content:
            if 'C1CCCCC1[C@@H]' in content or 'C1CCCCC1[C@H]' in content:
                return 'CHG', mods  # Cyclohexylglycine
            elif 'C1CCCCC1C[C@@H]' in content or 'C1CCCCC1C[C@H]' in content:
                return 'ALC', mods  # 3-cyclohexyl-alanine
                
        # Naphthalene derivatives
        if 'c1cccc2c1cccc2' in content:
            if 'c1cccc2c1cccc2[C@@H]' in content or 'c1cccc2c1cccc2[C@H]' in content:
                return 'NAL', mods  # 2-Naphthyl-alanine
                
        # Heteroaromatic derivatives
        if 'c1cncc' in content:
            return 'PYR4', mods  # 3-(4-Pyridyl)-alanine
        if 'c1cscc' in content:
            return 'THA3', mods  # 3-(3-thienyl)-alanine
        if 'c1nnc' in content:
            return 'TRZ4', mods  # 3-(1,2,4-Triazol-1-yl)-alanine
            
        # Modified serines and threonines
        if 'OP(O)(O)O' in content:
            if '[C@@H](COP' in content or '[C@H](COP' in content:
                return 'SEP', mods  # phosphoserine
            elif '[C@@H](OP' in content or '[C@H](OP' in content:
                return 'TPO', mods  # phosphothreonine
            
        # Specialized ring systems
        if 'c1c2ccccc2cc2c1cccc2' in content:
            return 'ANTH', mods  # 3-(9-anthryl)-alanine
        if 'c1csc2c1cccc2' in content:
            return 'BTH3', mods  # 3-(3-benzothienyl)-alanine
        if '[C@]12C[C@H]3C[C@@H](C2)C[C@@H](C1)C3' in content:
            return 'ADAM', mods  # Adamanthane

        # Fluorinated derivatives
        if 'FC(F)(F)' in content:
            if 'CC(F)(F)F' in content:
                return 'FLA', mods  # Trifluoro-alanine
            if 'C(F)(F)F)c1' in content:
                if 'c1ccccc1C(F)(F)F' in content:
                    return 'TFG2', mods  # 2-(Trifluoromethyl)-phenylglycine
                if 'c1cccc(c1)C(F)(F)F' in content:
                    return 'TFG3', mods  # 3-(Trifluoromethyl)-phenylglycine
                if 'c1ccc(cc1)C(F)(F)F' in content:
                    return 'TFG4', mods  # 4-(Trifluoromethyl)-phenylglycine
        
        # Multiple halogen patterns
        if 'F' in content and 'c1' in content:
            if 'c1ccc(c(c1)F)F' in content:
                return 'F2F', mods  # 3,4-Difluoro-phenylalanine
            if 'cc(F)cc(c1)F' in content:
                return 'WFP', mods  # 3,5-Difluoro-phenylalanine
        if 'Cl' in content and 'c1' in content:
            if 'c1ccc(cc1Cl)Cl' in content:
                return 'CP24', mods  # 2,4-dichloro-phenylalanine
            if 'c1ccc(c(c1)Cl)Cl' in content:
                return 'CP34', mods  # 3,4-dichloro-phenylalanine

        # Hydroxy and amino derivatives
        if 'O' in content and 'c1' in content:
            if 'c1cc(O)cc(c1)O' in content:
                return '3FG', mods  # (2s)-amino(3,5-dihydroxyphenyl)-ethanoic acid
            if 'c1ccc(c(c1)O)O' in content:
                return 'DAH', mods  # 3,4-Dihydroxy-phenylalanine
        
        # Cyclic amino acids
        if 'C1CCCC1' in content:
            return 'CPA3', mods  # 3-Cyclopentyl-alanine
        if 'C1CCCCC1' in content:
            if 'CC1CCCCC1' in content:
                return 'ALC', mods  # 3-cyclohexyl-alanine
            else:
                return 'CHG', mods  # Cyclohexylglycine

        # Chain-length variants
        if 'CCC[C@@H]' in content or 'CCC[C@H]' in content:
            return 'NLE', mods  # Norleucine
        if 'CC[C@@H]' in content or 'CC[C@H]' in content:
            if not any(x in content for x in ['CC(C)', 'COC', 'CN(']):
                return 'ABA', mods  # 2-Aminobutyric acid
        
        # Modified histidines
        if 'c1cnc' in content:
            if '[C@@H]1CN[C@@H](N1)F' in content:
                return '2HF', mods  # 2-fluoro-l-histidine
            if 'c1cnc([nH]1)F' in content:
                return '2HF1', mods  # 2-fluoro-l-histidine variant
            if 'c1c[nH]c(n1)F' in content:
                return '2HF2', mods  # 2-fluoro-l-histidine variant

        # Sulfur and selenium containing
        if '[SeH]' in content:
            return 'CSE', mods  # Selenocysteine
        if 'S' in content:
            if 'CSCc1ccccc1' in content:
                return 'BCS', mods  # benzylcysteine
            if 'CCSC' in content:
                return 'ESC', mods  # Ethionine
            if 'CCS' in content:
                return 'HCS', mods  # homocysteine

        # Additional modifications
        if 'CN=[N]=N' in content:
            return 'AZDA', mods  # azido-alanine
        if '[NH]=[C](=[NH2])=[NH2]' in content:
            if 'CCC[NH]=' in content:
                return 'AGM', mods  # 5-methyl-arginine
            if 'CC[NH]=' in content:
                return 'GDPR', mods  # 2-Amino-3-guanidinopropionic acid

        if 'CCON' in content:
            return 'CAN', mods  # canaline
        if '[C@@H]1C=C[C@@H](C=C1)' in content:
            return 'ACZ', mods  # cis-amiclenomycin
        if 'CCC(=O)[NH3]' in content:
            return 'ONL', mods  # 5-oxo-l-norleucine
        if 'c1ccncc1' in content:
            return 'PYR4', mods  # 3-(4-Pyridyl)-alanine
        if 'c1ccco1' in content:
            return 'FUA2', mods  # (2-furyl)-alanine
            
        if 'c1ccc' in content:
            if 'c1ccc(cc1)c1ccccc1' in content:
                return 'BIF', mods  # 4,4-biphenylalanine
            if 'c1ccc(cc1)C(=O)c1ccccc1' in content:
                return 'PBF', mods  # 4-benzoyl-phenylalanine
            if 'c1ccc(cc1)C(C)(C)C' in content:
                return 'TBP4', mods  # 4-tert-butyl-phenylalanine
            if 'c1ccc(cc1)[C](=[NH2])=[NH2]' in content:
                return '0BN', mods  # 4-carbamimidoyl-l-phenylalanine
            if 'c1cccc(c1)[C](=[NH2])=[NH2]' in content:
                return 'APM', mods  # m-amidinophenyl-3-alanine

        # Multiple hydroxy patterns
        if 'O' in content:
            if '[C@H]([C@H](C)O)O' in content:
                return 'ILX', mods  # 4,5-dihydroxy-isoleucine
            if '[C@H]([C@@H](C)O)O' in content:
                return 'ALO', mods  # Allo-threonine
            if '[C@H](COP(O)(O)O)' in content:
                return 'SEP', mods  # phosphoserine
            if '[C@H]([C@@H](C)OP(O)(O)O)' in content:
                return 'TPO', mods  # phosphothreonine
            if '[C@H](c1ccc(O)cc1)O' in content:
                return 'OMX', mods  # (betar)-beta-hydroxy-l-tyrosine
            if '[C@H](c1ccc(c(Cl)c1)O)O' in content:
                return 'OMY', mods  # (betar)-3-chloro-beta-hydroxy-l-tyrosine

        # Heterocyclic patterns
        if 'n1' in content:
            if 'n1cccn1' in content:
                return 'PYZ1', mods  # 3-(1-Pyrazolyl)-alanine
            if 'n1nncn1' in content:
                return 'TEZA', mods  # 3-(2-Tetrazolyl)-alanine
            if 'c2c(n1)cccc2' in content:
                return 'QU32', mods  # 3-(2-Quinolyl)-alanine
            if 'c1cnc2c(c1)cccc2' in content:
                return 'QU33', mods  # 3-(3-quinolyl)-alanine
            if 'c1ccnc2c1cccc2' in content:
                return 'QU34', mods  # 3-(4-quinolyl)-alanine
            if 'c1ccc2c(c1)nccc2' in content:
                return 'QU35', mods  # 3-(5-Quinolyl)-alanine
            if 'c1ccc2c(c1)cncc2' in content:
                return 'QU36', mods  # 3-(6-Quinolyl)-alanine
            if 'c1cnc2c(n1)cccc2' in content:
                return 'QX32', mods  # 3-(2-quinoxalyl)-alanine

        # Multiple nitrogen patterns
        if 'N' in content:
            if '[NH3]CC[C@@H]' in content:
                return 'DAB', mods  # Diaminobutyric acid
            if '[NH3]C[C@@H]' in content:
                return 'DPP', mods  # 2,3-Diaminopropanoic acid
            if '[NH3]CCCCCC[C@@H]' in content:
                return 'HHK', mods  # (2s)-2,8-diaminooctanoic acid
            if 'CCC[NH]=[C](=[NH2])=[NH2]' in content:
                return 'GBUT', mods  # 2-Amino-4-guanidinobutryric acid
            if '[NH]=[C](=S)=[NH2]' in content:
                return 'THIC', mods  # Thio-citrulline

        # Chain modified amino acids
        if 'CC' in content:
            if 'CCCC[C@@H]' in content:
                return 'AHP', mods  # 2-Aminoheptanoic acid
            if 'CCC([C@@H])(C)C' in content:
                return 'I2M', mods  # 3-methyl-l-alloisoleucine
            if 'CC[C@H]([C@@H])C' in content:
                return 'IIL', mods  # Allo-Isoleucine
            if '[C@H](CCC(C)C)' in content:
                return 'HLEU', mods  # Homoleucine
            if '[C@@H]([C@@H](C)O)C' in content:
                return 'HLU', mods  # beta-hydroxyleucine

        # Modified glutamate/aspartate patterns
        if '[C@@H]' in content:
            if '[C@@H](C[C@@H](F))' in content:
                return 'FGA4', mods  # 4-Fluoro-glutamic acid
            if '[C@@H](C[C@@H](O))' in content:
                return '3GL', mods  # 4-hydroxy-glutamic-acid
            if '[C@@H](C[C@H](C))' in content:
                return 'LME', mods  # (3r)-3-methyl-l-glutamic acid
            if '[C@@H](CC[C@H](C))' in content:
                return 'MEG', mods  # (3s)-3-methyl-l-glutamic acid

        # Sulfur and selenium modifications
        if 'S' in content:
            if 'SCC[C@@H]' in content:
                return 'HSER', mods  # homoserine
            if 'SCCN' in content:
                return 'SLZ', mods  # thialysine
            if 'SC(=O)' in content:
                return 'CSA', mods  # s-acetonylcysteine
            if '[S@@](=O)' in content:
                return 'SME', mods  # Methionine sulfoxide
            if 'S(=O)(=O)' in content:
                return 'OMT', mods  # Methionine sulfone

        # Double bond containing
        if 'C=' in content:
            if 'C=C[C@@H]' in content:
                return '2AG', mods  # 2-Allyl-glycine
            if 'C=C[C@@H]' in content:
                return 'LVG', mods  # vinylglycine
            if 'C=Cc1ccccc1' in content:
                return 'STYA', mods  # Styrylalanine

        # Special cases
        if '[C@@H]1Cc2c(C1)cccc2' in content:
            return 'IGL', mods  # alpha-amino-2-indanacetic acid
        if '[C](=[C](=O)=O)=O' in content:
            return '26P', mods  # 2-amino-6-oxopimelic acid
        if '[C](=[C](=O)=O)=C' in content:
            return '2NP', mods  # l-2-amino-6-methylene-pimelic acid
        if 'c2cnc[nH]2' in content:
            return 'HIS', mods  # histidine core
        if 'c1cccc2c1cc(O)cc2' in content:
            return 'NAO1', mods  # 5-hydroxy-1-naphthalene
        if 'c1ccc2c(c1)cc(O)cc2' in content:
            return 'NAO2', mods  # 6-hydroxy-2-naphthalene
            
        # Proline (P) - flexible ring numbers
        if any([
            # Check for any ring number in bond patterns
            (segment.get('bond_after', '').startswith(f'N{n}C(=O)') and 'CCC' in content and 
            any(f'[C@@H]{n}' in content or f'[C@H]{n}' in content for n in '123456789'))
            for n in '123456789'
        ]) or any([
            # Check ending patterns with any ring number
            (f'CCCN{n}' in content and content.endswith('=O') and 
            any(f'[C@@H]{n}' in content or f'[C@H]{n}' in content for n in '123456789'))
            for n in '123456789'
        ]) or any([
            # Handle CCC[C@H]n patterns
            (content == f'CCC[C@H]{n}' and segment.get('bond_before', '').startswith(f'C(=O)N{n}')) or
            (content == f'CCC[C@@H]{n}' and segment.get('bond_before', '').startswith(f'C(=O)N{n}')) or
            # N-terminal Pro with any ring number
            (f'N{n}CCC[C@H]{n}' in content) or
            (f'N{n}CCC[C@@H]{n}' in content)
            for n in '123456789'
        ]):
            return 'Pro', mods
        
        # Tryptophan (W) - more specific indole pattern
        if re.search(r'c[0-9]c\[nH\]c[0-9]ccccc[0-9][0-9]', content) and \
        'c[nH]c' in content.replace(' ', ''):
            return 'Trp', mods
        
        # Lysine (K) - both patterns
        if '[C@@H](CCCCN)' in content or '[C@H](CCCCN)' in content:
            return 'Lys', mods
            
        # Arginine (R) - both patterns
        if '[C@@H](CCCNC(=N)N)' in content or '[C@H](CCCNC(=N)N)' in content:
            return 'Arg', mods
        
        if ('C[C@H](CCCC)' in content or 'C[C@@H](CCCC)' in content) and 'CC(C)' not in content:
            return 'Nle', mods
            
        # Ornithine (Orn) - 3-carbon chain with NH2
        if ('C[C@H](CCCN)' in content or 'C[C@@H](CCCN)' in content) and 'CC(C)' not in content:
            return 'Orn', mods
            
        # 2-Naphthylalanine (2Nal) - distinct from Phe pattern
        if ('Cc3cc2ccccc2c3' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return '2Nal', mods
            
        # Cyclohexylalanine (Cha) - already in your code but moved here for clarity
        if 'N2CCCCC2' in content or 'CCCCC2' in content:
            return 'Cha', mods
            
        # Aminobutyric acid (Abu) - 2-carbon chain
        if ('C[C@H](CC)' in content or 'C[C@@H](CC)' in content) and not any(p in content for p in ['CC(C)', 'CCCC', 'CCC(C)']):
            return 'Abu', mods
            
        # Pipecolic acid (Pip) - 6-membered ring like Pro
        if ('N3CCCCC3' in content or 'CCCCC3' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return 'Pip', mods

        # Cyclohexylglycine (Chg) - direct cyclohexyl without CH2
        if ('C[C@H](C1CCCCC1)' in content or 'C[C@@H](C1CCCCC1)' in content):
            return 'Chg', mods
            
        # 4-Fluorophenylalanine (4F-Phe)
        if ('Cc2ccc(F)cc2' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return '4F-Phe', mods
        
        # Regular residue identification
        if ('NCC(=O)' in content) or (content == 'C'):
            # Middle case - between bonds
            if segment.get('bond_before') and segment.get('bond_after'):
                if ('C(=O)N' in segment['bond_before'] or 'C(=O)N(C)' in segment['bond_before']):
                    return 'Gly', mods
            # Terminal case - at the end
            elif segment.get('bond_before') and segment.get('bond_before').startswith('C(=O)N'):
                return 'Gly', mods
            
        if 'CC(C)C[C@H]' in content or 'CC(C)C[C@@H]' in content:
            return 'Leu', mods
        if '[C@@H](CC(C)C)' in content or '[C@H](CC(C)C)' in content:
            return 'Leu', mods

        if '[C@@H]([C@@H](C)O)' in content or '[C@H]([C@H](C)O)' in content:
            return 'Thr', mods
        
        if '[C@H](Cc2ccccc2)' in content or '[C@@H](Cc2ccccc2)' in content:
            return 'Phe', mods
            
        if ('[C@H](C(C)C)' in content or       # With outer parentheses
            '[C@@H](C(C)C)' in content or      # With outer parentheses
            '[C@H]C(C)C' in content or         # Without outer parentheses 
            '[C@@H]C(C)C' in content):         # Without outer parentheses
            if not any(p in content for p in ['CC(C)C[C@H]', 'CC(C)C[C@@H]']):  # Still check not Leu
                return 'Val', mods
        
        if '[C@H](COC(C)(C)C)' in content or '[C@@H](COC(C)(C)C)' in content:
            return 'O-tBu', mods
        
        if any([
            'CC[C@H](C)' in content,
            'CC[C@@H](C)' in content,
            'C(C)C[C@H]' in content and 'CC(C)C' not in content,
            'C(C)C[C@@H]' in content and 'CC(C)C' not in content
        ]):
            return 'Ile', mods
        
        if ('[C@H](C)' in content or '[C@@H](C)' in content):
            if not any(p in content for p in ['C(C)C', 'COC', 'CN(', 'C(C)O', 'CC[C@H]', 'CC[C@@H]']):
                return 'Ala', mods
        
        # Tyrosine (Tyr) - 4-hydroxybenzyl side chain
        if re.search(r'Cc[0-9]ccc\(O\)cc[0-9]', content):
            return 'Tyr', mods

        
        # Serine (Ser) - Hydroxymethyl side chain
        if '[C@H](CO)' in content or '[C@@H](CO)' in content:
            if not ('C(C)O' in content or 'COC' in content):
                return 'Ser', mods
        
        # Threonine (Thr) - 1-hydroxyethyl side chain
        if '[C@@H]([C@@H](C)O)' in content or '[C@H]([C@H](C)O)' in content or '[C@@H](C)O' in content or '[C@H](C)O' in content:
            return 'Thr', mods
        
        # Cysteine (Cys) - Thiol side chain
        if '[C@H](CS)' in content or '[C@@H](CS)' in content:
            return 'Cys', mods
        
        # Methionine (Met) - Methylthioethyl side chain
        if ('C[C@H](CCSC)' in content or 'C[C@@H](CCSC)' in content):
            return 'Met', mods
        
        # Asparagine (Asn) - Carbamoylmethyl side chain
        if ('CC(=O)N' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return 'Asn', mods
        
        # Glutamine (Gln) - Carbamoylethyl side chain
        if ('CCC(=O)N' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return 'Gln', mods
        
        # Aspartic acid (Asp) - Carboxymethyl side chain
        if ('CC(=O)O' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return 'Asp', mods
        
        # Glutamic acid (Glu) - Carboxyethyl side chain
        if ('CCC(=O)O' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return 'Glu', mods
        
        # Arginine (Arg) - 3-guanidinopropyl side chain
        if ('CCCNC(=N)N' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return 'Arg', mods
        
        # Histidine (His) - Imidazole side chain
        if ('Cc2cnc[nH]2' in content) and ('C[C@H]' in content or 'C[C@@H]' in content):
            return 'His', mods
            
        return None, mods

    def get_modifications(self, segment):
        """Get modifications based on bond types"""
        mods = []
        if segment.get('bond_after'):
            if 'N(C)' in segment['bond_after'] or segment['bond_after'].startswith('C(=O)N(C)'):
                mods.append('N-Me')
            if 'OC(=O)' in segment['bond_after']:
                mods.append('O-linked')
        return mods

    def analyze_structure(self, smiles):
        """Main analysis function with debug output"""
        print("\nAnalyzing structure:", smiles)
        
        # Split into segments
        segments = self.split_on_bonds(smiles)
        
        print("\nSegment Analysis:")
        sequence = []
        for i, segment in enumerate(segments):
            print(f"\nSegment {i}:")
            print(f"Content: {segment['content']}")
            print(f"Bond before: {segment.get('bond_before', 'None')}")
            print(f"Bond after: {segment.get('bond_after', 'None')}")
            
            residue, mods = self.identify_residue(segment)
            if residue:
                if mods:
                    sequence.append(f"{residue}({','.join(mods)})")
                else:
                    sequence.append(residue)
                print(f"Identified as: {residue}")
                print(f"Modifications: {mods}")
            else:
                print(f"Warning: Could not identify residue in segment: {segment['content']}")
        
        # Check if cyclic
        is_cyclic, peptide_cycles, aromatic_cycles = self.is_cyclic(smiles)
        three_letter = '-'.join(sequence)
        one_letter = ''.join(self.three_to_one.get(aa.split('(')[0], 'X') for aa in sequence)
        
        if is_cyclic:
            three_letter = f"cyclo({three_letter})"
            one_letter = f"cyclo({one_letter})"
        
        print(f"\nFinal sequence: {three_letter}")
        print(f"One-letter code: {one_letter}")
        print(f"Is cyclic: {is_cyclic}")
        #print(f"Peptide cycles: {peptide_cycles}")
        #print(f"Aromatic cycles: {aromatic_cycles}")
        
        return three_letter, len(segments)
        """return {
            'three_letter': three_letter,
            #'one_letter': one_letter,
            'is_cyclic': is_cyclic
        }"""
        
    def return_sequence(self, smiles):
        """Main analysis function with debug output"""
        print("\nAnalyzing structure:", smiles)
        
        # Split into segments
        segments = self.split_on_bonds(smiles)
        
        print("\nSegment Analysis:")
        sequence = []
        for i, segment in enumerate(segments):
            print(f"\nSegment {i}:")
            print(f"Content: {segment['content']}")
            print(f"Bond before: {segment.get('bond_before', 'None')}")
            print(f"Bond after: {segment.get('bond_after', 'None')}")
            
            residue, mods = self.identify_residue(segment)
            if residue:
                if mods:
                    sequence.append(f"{residue}({','.join(mods)})")
                else:
                    sequence.append(residue)
                print(f"Identified as: {residue}")
                print(f"Modifications: {mods}")
            else:
                print(f"Warning: Could not identify residue in segment: {segment['content']}")
        
        return sequence

"""
def annotate_cyclic_structure(mol, sequence):
    '''Create annotated 2D structure with clear, non-overlapping residue labels'''
    # Generate 2D coordinates
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Create drawer with larger size for annotations
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(2000, 2000)  # Even larger size
    
    # Get residue list and reverse it to match structural representation
    if sequence.startswith('cyclo('):
        residues = sequence[6:-1].split('-')
    else:
        residues = sequence.split('-')
    residues = list(reversed(residues))  # Reverse the sequence
    
    # Draw molecule first to get its bounds
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert to PIL Image
    img = Image.open(BytesIO(drawer.GetDrawingText()))
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use DejaVuSans as it's commonly available on Linux systems
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
    except OSError:
        try:
            # Fallback to Arial if available (common on Windows)
            font = ImageFont.truetype("arial.ttf", 60)
            small_font = ImageFont.truetype("arial.ttf", 60)
        except OSError:
            # If no TrueType fonts are available, fall back to default
            print("Warning: TrueType fonts not available, using default font")
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()
    # Get molecule bounds
    conf = mol.GetConformer()
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append((pos.x, pos.y))
    
    x_coords = [p[0] for p in positions]
    y_coords = [p[1] for p in positions]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    # Calculate scaling factors
    scale = 150  # Increased scale factor
    center_x = 1000  # Image center
    center_y = 1000
    
    # Add residue labels in a circular arrangement around the structure
    n_residues = len(residues)
    radius = 700  # Distance of labels from center
    
    # Start from the rightmost point (3 o'clock position) and go counterclockwise
    # Offset by -3 positions to align with structure
    offset = 0  # Adjust this value to match the structure alignment
    for i, residue in enumerate(residues):
        # Calculate position in a circle around the structure
        # Start from 0 (3 o'clock) and go counterclockwise
        angle = -(2 * np.pi * ((i + offset) % n_residues) / n_residues)
        
        # Calculate label position
        label_x = center_x + radius * np.cos(angle)
        label_y = center_y + radius * np.sin(angle)
        
        # Draw residue label
        text = f"{i+1}. {residue}"
        bbox = draw.textbbox((label_x, label_y), text, font=font)
        padding = 10
        draw.rectangle([bbox[0]-padding, bbox[1]-padding, 
                       bbox[2]+padding, bbox[3]+padding], 
                      fill='white', outline='white')
        draw.text((label_x, label_y), text, 
                 font=font, fill='black', anchor="mm")
    
    # Add sequence at the top with white background
    seq_text = f"Sequence: {sequence}"
    bbox = draw.textbbox((center_x, 100), seq_text, font=small_font)
    padding = 10
    draw.rectangle([bbox[0]-padding, bbox[1]-padding, 
                   bbox[2]+padding, bbox[3]+padding], 
                  fill='white', outline='white')
    draw.text((center_x, 100), seq_text, 
             font=small_font, fill='black', anchor="mm")
    
    return img

"""
def annotate_cyclic_structure(mol, sequence):
    """Create structure visualization with just the sequence header"""
    # Generate 2D coordinates
    AllChem.Compute2DCoords(mol)
    
    # Create drawer with larger size for annotations
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(2000, 2000)
    
    # Draw molecule first
    drawer.drawOptions().addAtomIndices = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    
    # Convert to PIL Image
    img = Image.open(BytesIO(drawer.GetDrawingText()))
    draw = ImageDraw.Draw(img)
    try:
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
    except OSError:
        try:
            small_font = ImageFont.truetype("arial.ttf", 60)
        except OSError:
            print("Warning: TrueType fonts not available, using default font")
            small_font = ImageFont.load_default()
    
    # Add just the sequence header at the top
    seq_text = f"Sequence: {sequence}"
    bbox = draw.textbbox((1000, 100), seq_text, font=small_font)
    padding = 10
    draw.rectangle([bbox[0]-padding, bbox[1]-padding, 
                   bbox[2]+padding, bbox[3]+padding], 
                  fill='white', outline='white')
    draw.text((1000, 100), seq_text, 
             font=small_font, fill='black', anchor="mm")
    
    return img

def create_enhanced_linear_viz(sequence, smiles):
    """Create an enhanced linear representation using PeptideAnalyzer"""
    analyzer = PeptideAnalyzer()  # Create analyzer instance
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2])
    ax_struct = fig.add_subplot(gs[0])
    ax_detail = fig.add_subplot(gs[1])
    
    # Parse sequence and get residues
    if sequence.startswith('cyclo('):
        residues = sequence[6:-1].split('-')
    else:
        residues = sequence.split('-')
    
    # Get segments using analyzer
    segments = analyzer.split_on_bonds(smiles)
    
    # Debug print
    print(f"Number of residues: {len(residues)}")
    print(f"Number of segments: {len(segments)}")
    
    # Top subplot - Basic structure
    ax_struct.set_xlim(0, 10)
    ax_struct.set_ylim(0, 2)
    
    num_residues = len(residues)
    spacing = 9.0 / (num_residues - 1) if num_residues > 1 else 9.0
    
    # Draw basic structure
    y_pos = 1.5
    for i in range(num_residues):
        x_pos = 0.5 + i * spacing
        
        # Draw amino acid box
        rect = patches.Rectangle((x_pos-0.3, y_pos-0.2), 0.6, 0.4, 
                               facecolor='lightblue', edgecolor='black')
        ax_struct.add_patch(rect)
        
        # Draw connecting bonds if not the last residue
        if i < num_residues - 1:
            segment = segments[i] if i < len(segments) else None
            if segment:
                # Determine bond type from segment info
                bond_type = 'ester' if 'O-linked' in segment.get('bond_after', '') else 'peptide'
                is_n_methylated = 'N-Me' in segment.get('bond_after', '')
                
                bond_color = 'red' if bond_type == 'ester' else 'black'
                linestyle = '--' if bond_type == 'ester' else '-'
                
                # Draw bond line
                ax_struct.plot([x_pos+0.3, x_pos+spacing-0.3], [y_pos, y_pos], 
                             color=bond_color, linestyle=linestyle, linewidth=2)
                
                # Add bond type label
                mid_x = x_pos + spacing/2
                bond_label = f"{bond_type}"
                if is_n_methylated:
                    bond_label += "\n(N-Me)"
                ax_struct.text(mid_x, y_pos+0.1, bond_label, 
                             ha='center', va='bottom', fontsize=10, 
                             color=bond_color)
        
        # Add residue label
        ax_struct.text(x_pos, y_pos-0.5, residues[i], 
                      ha='center', va='top', fontsize=14)
    
    # Bottom subplot - Detailed breakdown
    ax_detail.set_ylim(0, len(segments)+1)
    ax_detail.set_xlim(0, 1)
    
    # Create detailed breakdown
    segment_y = len(segments)  # Start from top
    for i, segment in enumerate(segments):
        y = segment_y - i
        
        # Check if this is a bond or residue
        residue, mods = analyzer.identify_residue(segment)
        if residue:
            text = f"Residue {i+1}: {residue}"
            if mods:
                text += f" ({', '.join(mods)})"
            color = 'blue'
        else:
            # Must be a bond
            text = f"Bond {i}: "
            if 'O-linked' in segment.get('bond_after', ''):
                text += "ester"
            elif 'N-Me' in segment.get('bond_after', ''):
                text += "peptide (N-methylated)"
            else:
                text += "peptide"
            color = 'red'
        
        # Add segment analysis
        ax_detail.text(0.05, y, text, fontsize=12, color=color)
        ax_detail.text(0.5, y, f"SMILES: {segment.get('content', '')}", fontsize=10, color='gray')
    
    # If cyclic, add connection indicator
    if sequence.startswith('cyclo('):
        ax_struct.annotate('', xy=(9.5, y_pos), xytext=(0.5, y_pos),
                          arrowprops=dict(arrowstyle='<->', color='red', lw=2))
        ax_struct.text(5, y_pos+0.3, 'Cyclic Connection', 
                      ha='center', color='red', fontsize=14)
    
    # Add titles and adjust layout
    ax_struct.set_title("Peptide Structure Overview", pad=20)
    ax_detail.set_title("Segment Analysis Breakdown", pad=20)
    
    # Remove axes
    for ax in [ax_struct, ax_detail]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    
    plt.tight_layout()
    return fig

class PeptideStructureGenerator:
    """A class to generate 3D structures of peptides using different embedding methods"""
    
    @staticmethod
    def prepare_molecule(smiles):
        """Prepare molecule with proper hydrogen handling"""
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            raise ValueError("Failed to create molecule from SMILES")
        
        # Calculate valence for each atom
        for atom in mol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
        
        # Sanitize with reduced requirements
        Chem.SanitizeMol(mol, 
                        sanitizeOps=Chem.SANITIZE_FINDRADICALS|
                                  Chem.SANITIZE_KEKULIZE|
                                  Chem.SANITIZE_SETAROMATICITY|
                                  Chem.SANITIZE_SETCONJUGATION|
                                  Chem.SANITIZE_SETHYBRIDIZATION|
                                  Chem.SANITIZE_CLEANUPCHIRALITY)
        
        mol = Chem.AddHs(mol)
        return mol

    @staticmethod
    def get_etkdg_params(attempt=0):
        """Get ETKDG parameters with optional modifications based on attempt number"""
        params = AllChem.ETKDGv3()
        params.randomSeed = -1
        params.maxIterations = 200
        params.numThreads = 4  # Reduced for web interface
        params.useBasicKnowledge = True
        params.enforceChirality = True
        params.useExpTorsionAnglePrefs = True
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.ETversion = 2
        params.pruneRmsThresh = -1
        params.embedRmsThresh = 0.5
        
        if attempt > 10:
            params.bondLength = 1.5 + (attempt - 10) * 0.02
            params.useExpTorsionAnglePrefs = False
        
        return params

    def generate_structure_etkdg(self, smiles, max_attempts=20):
        """Generate 3D structure using ETKDG without UFF optimization"""
        success = False
        mol = None
        
        for attempt in range(max_attempts):
            try:
                mol = self.prepare_molecule(smiles)
                params = self.get_etkdg_params(attempt)
                
                if AllChem.EmbedMolecule(mol, params) == 0:
                    success = True
                    break
            except Exception as e:
                continue
        
        if not success:
            raise ValueError("Failed to generate structure with ETKDG")
        
        return mol

    def generate_structure_uff(self, smiles, max_attempts=20):
        """Generate 3D structure using ETKDG followed by UFF optimization"""
        best_mol = None
        lowest_energy = float('inf')
        
        for attempt in range(max_attempts):
            try:
                test_mol = self.prepare_molecule(smiles)
                params = self.get_etkdg_params(attempt)
                
                if AllChem.EmbedMolecule(test_mol, params) == 0:
                    res = AllChem.UFFOptimizeMolecule(test_mol, maxIters=2000, 
                                                     vdwThresh=10.0, confId=0,
                                                     ignoreInterfragInteractions=True)
                    
                    if res == 0:
                        ff = AllChem.UFFGetMoleculeForceField(test_mol)
                        if ff:
                            current_energy = ff.CalcEnergy()
                            if current_energy < lowest_energy:
                                lowest_energy = current_energy
                                best_mol = Chem.Mol(test_mol)
            except Exception:
                continue
        
        if best_mol is None:
            raise ValueError("Failed to generate optimized structure")
        
        return best_mol

    @staticmethod
    def mol_to_sdf_bytes(mol):
        """Convert RDKit molecule to SDF file bytes"""
        # First write to StringIO in text mode
        sio = StringIO()
        writer = Chem.SDWriter(sio)
        writer.write(mol)
        writer.close()
        
        # Convert the string to bytes
        return sio.getvalue().encode('utf-8')

def process_input(smiles_input=None, file_obj=None, show_linear=False, 
                 show_segment_details=False, generate_3d=False, use_uff=False):
    """Process input and create visualizations using PeptideAnalyzer"""
    analyzer = PeptideAnalyzer()
    temp_dir = tempfile.mkdtemp() if generate_3d else None
    structure_files = []
    
    # Handle direct SMILES input
    if smiles_input:
        smiles = smiles_input.strip()
        
        # First check if it's a peptide using analyzer's method
        if not analyzer.is_peptide(smiles):
            return "Error: Input SMILES does not appear to be a peptide structure.", None, None
                
        try:
            # Create molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return "Error: Invalid SMILES notation.", None, None
            
            # Generate 3D structures if requested
            if generate_3d:
                generator = PeptideStructureGenerator()
                
                try:
                    # Generate ETKDG structure
                    mol_etkdg = generator.generate_structure_etkdg(smiles)
                    etkdg_path = os.path.join(temp_dir, "structure_etkdg.sdf")
                    writer = Chem.SDWriter(etkdg_path)
                    writer.write(mol_etkdg)
                    writer.close()
                    structure_files.append(etkdg_path)
                    
                    # Generate UFF structure if requested
                    if use_uff:
                        mol_uff = generator.generate_structure_uff(smiles)
                        uff_path = os.path.join(temp_dir, "structure_uff.sdf")
                        writer = Chem.SDWriter(uff_path)
                        writer.write(mol_uff)
                        writer.close()
                        structure_files.append(uff_path)
                
                except Exception as e:
                    return f"Error generating 3D structures: {str(e)}", None, None, None
                
            # Use analyzer to get sequence
            segments = analyzer.split_on_bonds(smiles)
            
            # Process segments and build sequence
            sequence_parts = []
            output_text = ""
            
            # Only include segment analysis in output if requested
            if show_segment_details:
                output_text += "Segment Analysis:\n"
                for i, segment in enumerate(segments):
                    output_text += f"\nSegment {i}:\n"
                    output_text += f"Content: {segment['content']}\n"
                    output_text += f"Bond before: {segment.get('bond_before', 'None')}\n"
                    output_text += f"Bond after: {segment.get('bond_after', 'None')}\n"
                    
                    residue, mods = analyzer.identify_residue(segment)
                    if residue:
                        if mods:
                            sequence_parts.append(f"{residue}({','.join(mods)})")
                        else:
                            sequence_parts.append(residue)
                        output_text += f"Identified as: {residue}\n"
                        output_text += f"Modifications: {mods}\n"
                    else:
                        output_text += f"Warning: Could not identify residue in segment: {segment['content']}\n"
                output_text += "\n"
            else:
                # Just build sequence without detailed analysis in output
                for segment in segments:
                    residue, mods = analyzer.identify_residue(segment)
                    if residue:
                        if mods:
                            sequence_parts.append(f"{residue}({','.join(mods)})")
                        else:
                            sequence_parts.append(residue)
            
            # Check if cyclic using analyzer's method
            is_cyclic, peptide_cycles, aromatic_cycles = analyzer.is_cyclic(smiles)
            three_letter = '-'.join(sequence_parts)
            one_letter = ''.join(analyzer.three_to_one.get(aa.split('(')[0], 'X') for aa in sequence_parts)
        
            if is_cyclic:
                three_letter = f"cyclo({three_letter})"
                one_letter = f"cyclo({one_letter})"
            
            # Create cyclic structure visualization
            img_cyclic = annotate_cyclic_structure(mol, three_letter)
            
            # Create linear representation if requested
            img_linear = None
            if show_linear:
                fig_linear = create_enhanced_linear_viz(three_letter, smiles)
                buf = BytesIO()
                fig_linear.savefig(buf, format='png', bbox_inches='tight', dpi=300)
                buf.seek(0)
                img_linear = Image.open(buf)
                plt.close(fig_linear)

            # Add summary to output
            summary = "Summary:\n"
            summary += f"Sequence: {three_letter}\n"
            summary += f"One-letter code: {one_letter}\n"
            summary += f"Is Cyclic: {'Yes' if is_cyclic else 'No'}\n"
            #if is_cyclic:
                #summary += f"Peptide Cycles: {', '.join(peptide_cycles)}\n"
                #summary += f"Aromatic Cycles: {', '.join(aromatic_cycles)}\n"
            
            if structure_files:
                summary += "\n3D Structures Generated:\n"
                for filepath in structure_files:
                    summary += f"- {os.path.basename(filepath)}\n"
            
            return summary + output_text, img_cyclic, img_linear, structure_files if structure_files else None
            
        except Exception as e:
            return f"Error processing SMILES: {str(e)}", None, None, None
    
    # Handle file input
    if file_obj is not None:
        try:
            # Handle file content
            if hasattr(file_obj, 'name'):
                with open(file_obj.name, 'r') as f:
                    content = f.read()
            else:
                content = file_obj.decode('utf-8') if isinstance(file_obj, bytes) else str(file_obj)
            
            output_text = ""
            for line in content.splitlines():
                smiles = line.strip()
                if smiles:
                    # Check if it's a peptide
                    if not analyzer.is_peptide(smiles):
                        output_text += f"Skipping non-peptide SMILES: {smiles}\n"
                        continue
                    
                    # Process this SMILES
                    segments = analyzer.split_on_bonds(smiles)
                    sequence_parts = []
                    
                    # Add segment details if requested
                    if show_segment_details:
                        output_text += f"\nSegment Analysis for SMILES: {smiles}\n"
                        for i, segment in enumerate(segments):
                            output_text += f"\nSegment {i}:\n"
                            output_text += f"Content: {segment['content']}\n"
                            output_text += f"Bond before: {segment.get('bond_before', 'None')}\n"
                            output_text += f"Bond after: {segment.get('bond_after', 'None')}\n"
                            residue, mods = analyzer.identify_residue(segment)
                            if residue:
                                if mods:
                                    sequence_parts.append(f"{residue}({','.join(mods)})")
                                else:
                                    sequence_parts.append(residue)
                                output_text += f"Identified as: {residue}\n"
                                output_text += f"Modifications: {mods}\n"
                    else:
                        for segment in segments:
                            residue, mods = analyzer.identify_residue(segment)
                            if residue:
                                if mods:
                                    sequence_parts.append(f"{residue}({','.join(mods)})")
                                else:
                                    sequence_parts.append(residue)
                    
                    # Get cyclicity and create sequence
                    is_cyclic, peptide_cycles, aromatic_cycles = analyzer.is_cyclic(smiles)
                    sequence = f"cyclo({'-'.join(sequence_parts)})" if is_cyclic else '-'.join(sequence_parts)
                    
                    output_text += f"\nSummary for SMILES: {smiles}\n"
                    output_text += f"Sequence: {sequence}\n"
                    output_text += f"Is Cyclic: {'Yes' if is_cyclic else 'No'}\n"
                    if is_cyclic:
                        output_text += f"Peptide Cycles: {', '.join(peptide_cycles)}\n"
                        #output_text += f"Aromatic Cycles: {', '.join(aromatic_cycles)}\n"
                    output_text += "-" * 50 + "\n"
            
            return output_text, None, None
            
        except Exception as e:
            return f"Error processing file: {str(e)}", None, None
    
    return "No input provided.", None, None

