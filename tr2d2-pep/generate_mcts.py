#!/usr/bin/env
import torch
import torch.nn.functional as F
import math
import random
import sys
from diffusion import Diffusion
import hydra
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
# direct reward backpropagation
from diffusion import Diffusion
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
import torch
import argparse
import os
import datetime
from utils.utils import str2bool, set_seed

# for peptides
from utils.app import PeptideAnalyzer
from peptide_mcts import MCTS

@torch.no_grad()
def generate_mcts(args, cfg, policy_model, pretrained, prot=None, prot_name=None, filename=None):
    
    score_func_names = ['binding_affinity1', 'solubility', 'hemolysis', 'nonfouling', 'permeability']
    
    mcts = MCTS(args, cfg, policy_model, pretrained, score_func_names, prot_seqs=[prot])
    
    final_x, log_rnd, final_rewards, score_vectors, sequences = mcts.forward()
    
    return final_x, log_rnd, final_rewards, score_vectors, sequences

def save_logs_to_file(reward_log, logrnd_log, 
                      valid_fraction_log, affinity1_log, 
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
        "Reward": reward_log,
        "Log RND": logrnd_log,
        "Valid Fraction": valid_fraction_log,
        "Binding Affinity": affinity1_log,
        "Solubility": sol_log,
        "Hemolysis": hemo_log, 
        "Nonfouling": nf_log,
        "Permeability": permeability_log
    }
        
    df = pd.DataFrame(log_data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default='')
argparser.add_argument('--learning_rate', type=float, default=1e-4)
argparser.add_argument('--num_epochs', type=int, default=1000)
argparser.add_argument('--num_accum_steps', type=int, default=4)
argparser.add_argument('--truncate_steps', type=int, default=50)
argparser.add_argument("--truncate_kl", type=str2bool, default=False)
argparser.add_argument('--gumbel_temp', type=float, default=1.0)
argparser.add_argument('--gradnorm_clip', type=float, default=1.0)
argparser.add_argument('--batch_size', type=int, default=32)
argparser.add_argument('--name', type=str, default='debug')
argparser.add_argument('--total_num_steps', type=int, default=128)
argparser.add_argument('--copy_flag_temp', type=float, default=None)
argparser.add_argument('--save_every_n_epochs', type=int, default=50)
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
# new
argparser.add_argument('--run_name', type=str, default='drakes')
argparser.add_argument("--device", default="cuda", type=str)

# mcts
argparser.add_argument('--num_sequences', type=int, default=100)
argparser.add_argument('--num_children', type=int, default=20)
argparser.add_argument('--num_iter', type=int, default=100) # iterations of mcts
argparser.add_argument('--seq_length', type=int, default=200)
argparser.add_argument('--time_conditioning', action='store_true', default=False)
argparser.add_argument('--mcts_sampling', type=int, default=0) # for batched categorical sampling: '0' means gumbel noise
argparser.add_argument('--buffer_size', type=int, default=100)
argparser.add_argument('--wdce_num_replicates', type=int, default=16)
argparser.add_argument('--noise_removal', action='store_true', default=False)
argparser.add_argument('--exploration', type=float, default=0.1)
argparser.add_argument('--reset_every_n_step', type=int, default=100)
argparser.add_argument('--alpha', type=float, default=0.01)
argparser.add_argument('--scalarization', type=str, default='sum')
argparser.add_argument('--no_mcts', action='store_true', default=False)
argparser.add_argument("--centering", action='store_true', default=False)
argparser.add_argument('--num_obj', type=int, default=5)

argparser.add_argument('--prot_seq', type=str, default=None)
argparser.add_argument('--prot_name', type=str, default=None)

args = argparser.parse_args()
print(args)
    
# pretrained model path
ckpt_path = f'{args.base_path}/TR2-D2/tr2d2-pep/pretrained/peptune-pretrained.ckpt'

# reinitialize Hydra
GlobalHydra.instance().clear()

# Initialize Hydra and compose the configuration
initialize(config_path="configs", job_name="load_model")
cfg = compose(config_name="config.yaml")
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

set_seed(args.seed, use_cuda=True)

# proteins
amhr = 'MLGSLGLWALLPTAVEAPPNRRTCVFFEAPGVRGSTKTLGELLDTGTELPRAIRCLYSRCCFGIWNLTQDRAQVEMQGCRDSDEPGCESLHCDPSPRAHPSPGSTLFTCSCGTDFCNANYSHLPPPGSPGTPGSQGPQAAPGESIWMALVLLGLFLLLLLLLGSIILALLQRKNYRVRGEPVPEPRPDSGRDWSVELQELPELCFSQVIREGGHAVVWAGQLQGKLVAIKAFPPRSVAQFQAERALYELPGLQHDHIVRFITASRGGPGRLLSGPLLVLELHPKGSLCHYLTQYTSDWGSSLRMALSLAQGLAFLHEERWQNGQYKPGIAHRDLSSQNVLIREDGSCAIGDLGLALVLPGLTQPPAWTPTQPQGPAAIMEAGTQRYMAPELLDKTLDLQDWGMALRRADIYSLALLLWEILSRCPDLRPDSSPPPFQLAYEAELGNTPTSDELWALAVQERRRPYIPSTWRCFATDPDGLRELLEDCWDADPEARLTAECVQQRLAALAHPQESHPFPESCPRGCPPLCPEDCTSIPAPTILPCRPQRSACHFSVQQGPCSRNPQPACTLSPV'
tfr = 'MMDQARSAFSNLFGGEPLSYTRFSLARQVDGDNSHVEMKLAVDEEENADNNTKANVTKPKRCSGSICYGTIAVIVFFLIGFMIGYLGYCKGVEPKTECERLAGTESPVREEPGEDFPAARRLYWDDLKRKLSEKLDSTDFTGTIKLLNENSYVPREAGSQKDENLALYVENQFREFKLSKVWRDQHFVKIQVKDSAQNSVIIVDKNGRLVYLVENPGGYVAYSKAATVTGKLVHANFGTKKDFEDLYTPVNGSIVIVRAGKITFAEKVANAESLNAIGVLIYMDQTKFPIVNAELSFFGHAHLGTGDPYTPGFPSFNHTQFPPSRSSGLPNIPVQTISRAAAEKLFGNMEGDCPSDWKTDSTCRMVTSESKNVKLTVSNVLKEIKILNIFGVIKGFVEPDHYVVVGAQRDAWGPGAAKSGVGTALLLKLAQMFSDMVLKDGFQPSRSIIFASWSAGDFGSVGATEWLEGYLSSLHLKAFTYINLDKAVLGTSNFKVSASPLLYTLIEKTMQNVKHPVTGQFLYQDSNWASKVEKLTLDNAAFPFLAYSGIPAVSFCFCEDTDYPYLGTTMDTYKELIERIPELNKVARAAAEVAGQFVIKLTHDVELNLDYERYNSQLLSFVRDLNQYRADIKEMGLSLQWLYSARGDFFRATSRLTTDFGNAEKTDRFVMKKLNDRVMRVEYHFLSPYVSPKESPFRHVFWGSGSHTLPALLENLKLRKQNNGAFNETLFRNQLALATWTIQGAANALSGDVWDIDNEF'
gfap = 'MERRRITSAARRSYVSSGEMMVGGLAPGRRLGPGTRLSLARMPPPLPTRVDFSLAGALNAGFKETRASERAEMMELNDRFASYIEKVRFLEQQNKALAAELNQLRAKEPTKLADVYQAELRELRLRLDQLTANSARLEVERDNLAQDLATVRQKLQDETNLRLEAENNLAAYRQEADEATLARLDLERKIESLEEEIRFLRKIHEEEVRELQEQLARQQVHVELDVAKPDLTAALKEIRTQYEAMASSNMHEAEEWYRSKFADLTDAAARNAELLRQAKHEANDYRRQLQSLTCDLESLRGTNESLERQMREQEERHVREAASYQEALARLEEEGQSLKDEMARHLQEYQDLLNVKLALDIEIATYRKLLEGEENRITIPVQTFSNLQIRETSLDTKSVSEGHLKRNIVVKTVEMRDGEVIKESKQEHKDVM'
glp1 = 'MAGAPGPLRLALLLLGMVGRAGPRPQGATVSLWETVQKWREYRRQCQRSLTEDPPPATDLFCNRTFDEYACWPDGEPGSFVNVSCPWYLPWASSVPQGHVYRFCTAEGLWLQKDNSSLPWRDLSECEESKRGERSSPEEQLLFLYIIYTVGYALSFSALVIASAILLGFRHLHCTRNYIHLNLFASFILRALSVFIKDAALKWMYSTAAQQHQWDGLLSYQDSLSCRLVFLLMQYCVAANYYWLLVEGVYLYTLLAFSVLSEQWIFRLYVSIGWGVPLLFVVPWGIVKYLYEDEGCWTRNSNMNYWLIIRLPILFAIGVNFLIFVRVICIVVSKLKANLMCKTDIKCRLAKSTLTLIPLLGTHEVIFAFVMDEHARGTLRFIKLFTELSFTSFQGLMVAILYCFVNNEVQLEFRKSWERWRLEHLHIQRDSSMKPLKCPTSSLSSGATAGSSMYTATCQASCS'
glast = 'MTKSNGEEPKMGGRMERFQQGVRKRTLLAKKKVQNITKEDVKSYLFRNAFVLLTVTAVIVGTILGFTLRPYRMSYREVKYFSFPGELLMRMLQMLVLPLIISSLVTGMAALDSKASGKMGMRAVVYYMTTTIIAVVIGIIIVIIIHPGKGTKENMHREGKIVRVTAADAFLDLIRNMFPPNLVEACFKQFKTNYEKRSFKVPIQANETLVGAVINNVSEAMETLTRITEELVPVPGSVNGVNALGLVVFSMCFGFVIGNMKEQGQALREFFDSLNEAIMRLVAVIMWYAPVGILFLIAGKIVEMEDMGVIGGQLAMYTVTVIVGLLIHAVIVLPLLYFLVTRKNPWVFIGGLLQALITALGTSSSSATLPITFKCLEENNGVDKRVTRFVLPVGATINMDGTALYEALAAIFIAQVNNFELNFGQIITISITATAASIGAAGIPQAGLVTMVIVLTSVGLPTDDITLIIAVDWFLDRLRTTTNVLGDSLGAGIVEHLSRHELKNRDVEMGNSVIEENEMKKPYQLIAQDNETEKPIDSETKM'
ncam = 'LQTKDLIWTLFFLGTAVSLQVDIVPSQGEISVGESKFFLCQVAGDAKDKDISWFSPNGEKLTPNQQRISVVWNDDSSSTLTIYNANIDDAGIYKCVVTGEDGSESEATVNVKIFQKLMFKNAPTPQEFREGEDAVIVCDVVSSLPPTIIWKHKGRDVILKKDVRFIVLSNNYLQIRGIKKTDEGTYRCEGRILARGEINFKDIQVIVNVPPTIQARQNIVNATANLGQSVTLVCDAEGFPEPTMSWTKDGEQIEQEEDDEKYIFSDDSSQLTIKKVDKNDEAEYICIAENKAGEQDATIHLKVFAKPKITYVENQTAMELEEQVTLTCEASGDPIPSITWRTSTRNISSEEKASWTRPEKQETLDGHMVVRSHARVSSLTLKSIQYTDAGEYICTASNTIGQDSQSMYLEVQYAPKLQGPVAVYTWEGNQVNITCEVFAYPSATISWFRDGQLLPSSNYSNIKIYNTPSASYLEVTPDSENDFGNYNCTAVNRIGQESLEFILVQADTPSSPSIDQVEPYSSTAQVQFDEPEATGGVPILKYKAEWRAVGEEVWHSKWYDAKEASMEGIVTIVGLKPETTYAVRLAALNGKGLGEISAASEF'
cereblon = 'MAGEGDQQDAAHNMGNHLPLLPAESEEEDEMEVEDQDSKEAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSCQVIPVLPQVMMILIPGQTLPLQLFHPQEVSMVRNLIQKDRTFAVLAYSNVQEREAQFGTTAEIYAYREEQDFGIEIVKVKAIGRQRFKVLELRTQSDGIQQAKVQILPECVLPSTMSAVQLESLNKCQIFPSKPVSREDQCSYKWWQKYQKRKFHCANLTSWPRWLYSLYDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYRVAACLPIDDVLRIQLLKIGSAIQRLRCELDIMNKCTSLCCKQCQETEITTKNEIFSLSLCGPMAAYVNPHGYVHETLTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALLPTIPDTEDEISPDKVILCL'
ligase = 'MASQPPEDTAESQASDELECKICYNRYNLKQRKPKVLECCHRVCAKCLYKIIDFGDSPQGVIVCPFCRFETCLPDDEVSSLPDDNNILVNLTCGGKGKKCLPENPTELLLTPKRLASLVSPSHTSSNCLVITIMEVQRESSPSLSSTPVVEFYRPASFDSVTTVSHNWTVWNCTSLLFQTSIRVLVWLLGLLYFSSLPLGIYLLVSKKVTLGVVFVSLVPSSLVILMVYGFCQCVCHEFLDCMAPPS'
skp2 = 'MHRKHLQEIPDLSSNVATSFTWGWDSSKTSELLSGMGVSALEKEEPDSENIPQELLSNLGHPESPPRKRLKSKGSDKDFVIVRRPKLNRENFPGVSWDSLPDELLLGIFSCLCLPELLKVSGVCKRWYRLASDESLWQTLDLTGKNLHPDVTGRLLSQGVIAFRCPRSFMDQPLAEHFSPFRVQHMDLSNSVIEVSTLHGILSQCSKLQNLSLEGLRLSDPIVNTLAKNSNLVRLNLSGCSGFSEFALQTLLSSCSRLDELNLSWCFDFTEKHVQVAVAHVSETITQLNLSGYRKNLQKSDLSTLVRRCPNLVHLDLSDSVMLKNDCFQEFFQLNYLQHLSLSRCYDIIPETLLELGEIPTLKTLQVFGIVPDGTLQLLKEALPHLQINCSHFTTIARPTIGNKKNQEIWGIKCRLTLQKPSCL'

if args.prot_seq is not None:
    prot = args.prot_seq
    prot_name = args.prot_name
    filename = args.prot_name
else:
    prot = tfr
    prot_name = "tfr"
    filename = "tfr"

# Initialize the model
new_model = Diffusion.load_from_checkpoint(ckpt_path, config=cfg, strict=False, map_location=args.device)
old_model = Diffusion.load_from_checkpoint(ckpt_path, config=cfg, strict=False, map_location=args.device)

with torch.no_grad():
    final_x, log_rnd, final_rewards, score_vectors, sequences = generate_mcts(args, cfg, new_model, old_model, 
                                                                              prot=prot, prot_name=prot_name)
    
    final_x = final_x.detach().to('cpu')            # [B, L] integer tokens
    log_rnd = log_rnd.detach().to('cpu').float().view(-1)          # [B]
    #final_rewards = final_rewards.detach().to('cpu').float().view(-1)  # [B]

print("loaded models...")
analyzer = PeptideAnalyzer()

generation_results = []

for i in range(final_x.shape[0]):
    sequence = sequences[i]
    log_rnd_single = log_rnd[i]
    final_reward = final_rewards[i]
    
    aa_seq, seq_length = analyzer.analyze_structure(sequence)
    
    scores = score_vectors[i]
    
    binding1 = scores[0]
    solubility = scores[1]
    hemo = scores[2]
    nonfouling = scores[3]
    permeability = scores[4]
    
    generation_results.append([sequence, aa_seq, final_reward, log_rnd_single, binding1, solubility, hemo, nonfouling, permeability])
    print(f"length: {seq_length} | smiles sequence: {sequence} | amino acid sequence: {aa_seq} | Binding Affinity: {binding1} | Solubility: {solubility} | Hemolysis: {hemo} | Nonfouling: {nonfouling} | Permeability: {permeability}")

    sys.stdout.flush()

df = pd.DataFrame(generation_results, columns=['Generated SMILES', 'Peptide Sequence', 'Final Reward', 'Log RND', 'Binding Affinity', 'Solubility', 'Hemolysis', 'Nonfouling', 'Permeability'])


df.to_csv(f'{args.base_path}/TR2-D2/tr2d2-pep/plots/{prot_name}-peptune-baseline/generation_results.csv', index=False)
