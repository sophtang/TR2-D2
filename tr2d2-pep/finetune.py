# direct reward backpropagation
from diffusion import Diffusion
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import numpy as np
from scipy.stats import pearsonr
import torch
import torch.nn.functional as F
import argparse
import wandb
import os
import datetime
from finetune_peptides import finetune
from peptide_mcts import MCTS
from utils.utils import str2bool, set_seed
from scoring.scoring_functions import ScoringFunctions


argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--base_path', type=str, default='')
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
argparser.add_argument('--alpha_schedule_warmup', type=int, default=0)
argparser.add_argument("--seed", type=int, default=0)
# new
argparser.add_argument('--run_name', type=str, default='peptides')
argparser.add_argument("--device", default="cuda:0", type=str)
argparser.add_argument("--save_path_dir", default="/scratch/pranamlab/sophtang/home/tr2d2/peptides/checkpoints/", type=str)
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
argparser.add_argument('--reset_every_n_step', type=int, default=100)
argparser.add_argument('--alpha', type=float, default=0.01)
argparser.add_argument('--scalarization', type=str, default='sum')
argparser.add_argument('--no_mcts', action='store_true', default=False)
argparser.add_argument("--centering", action='store_true', default=False)

# objectives
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
cfg = compose(config_name="peptune_config.yaml")
curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

if args.no_mcts:
    args.run_name = f'{prot_name}_resample{args.resample_every_n_step}_no-mcts'
else:
    args.run_name = f'{prot_name}_resample{args.resample_every_n_step}_buffer{args.buffer_size}_numiter{args.num_iter}_children{args.num_children}_{curr_time}'

args.save_path = os.path.join(args.save_path_dir, args.run_name)
os.makedirs(args.save_path, exist_ok=True)
# wandb init
wandb.init(project='tree-multi', name=args.run_name, config=args, dir=args.save_path)

log_path = os.path.join(args.save_path, 'log.txt')

set_seed(args.seed, use_cuda=True)

# Initialize the model
policy_model = Diffusion.load_from_checkpoint(ckpt_path, 
                                              config=cfg, 
                                              mode="train",
                                              device=args.device,
                                              map_location=args.device)
pretrained = Diffusion.load_from_checkpoint(ckpt_path, 
                                            config=cfg, 
                                            mode="eval",
                                            device=args.device,
                                            map_location=args.device)

# define mcts
score_func_names = ['binding_affinity1', 'solubility', 'hemolysis', 'nonfouling', 'permeability']

mcts = MCTS(args, cfg, policy_model, pretrained, score_func_names, prot_seqs=[prot])

if args.no_mcts:
    reward_model = ScoringFunctions(score_func_names, prot_seqs=[prot], device=args.device)
    finetune(args, cfg, policy_model, reward_model=reward_model, mcts=None, pretrained=pretrained, filename=filename, prot_name=prot_name)
else:
    mcts = MCTS(args, cfg, policy_model, pretrained, score_func_names, prot_seqs=[prot])
    finetune(args, cfg, policy_model, reward_model=mcts.rewardFunc, mcts=mcts, pretrained=None, filename=filename, prot_name=prot_name)