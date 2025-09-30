import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import random as rd
from utils.app import PeptideAnalyzer
from utils.timer import StepTimer
from scoring.scoring_functions import ScoringFunctions

import noise_schedule

### for peptide multi-objective ###
def dominates(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return np.all(a >= b) and np.any(a > b)

def dominated_by(a, b):
    return dominates(b, a)


def updateParetoFront(paretoFront, node, scoreVector, totalSize=None, eps=1e-12):
    """
    Maintain a non-dominated set (Pareto front) of (node -> scoreVector).

    - Accept 'node' iff it is NOT dominated by any node in the set.
    - Remove any nodes that ARE dominated by 'node'.
    - Skip insertion if an equal point already exists (within eps).
    - If totalSize is given and the archive exceeds it, drop the item
      with the smallest sum(scoreVector) as a simple tie-breaker.

    Args:
        paretoFront (dict): {node: scoreVector}
        node: candidate node (used as dict key)
        scoreVector (array-like): candidate scores (to be maximized)
        totalSize (int|None): optional max size for the archive
        eps (float): tolerance for equality/inequality checks

    Returns:
        dict: updated paretoFront
    """
    s = np.asarray(scoreVector, dtype=float)

    def dominates(a, b):
        # a >= b in all coords and > in at least one (with tolerance)
        return np.all(a >= b - eps) and np.any(a > b + eps)

    def equal(a, b):
        return np.all(np.abs(a - b) <= eps)

    # reject if candidate is dominated by any node already in the set
    for v in paretoFront.values():
        v = np.asarray(v, dtype=float)
        if dominates(v, s):
            return paretoFront  # no change

    # remove any nodes dominated by candidate node
    survivors = {}
    #has_equal = False
    for k, v in paretoFront.items():
        v_arr = np.asarray(v, dtype=float)
        if dominates(s, v_arr):
            continue  # drop dominated incumbent
        """if equal(s, v_arr):
            has_equal = True  # skip duplicate insertion later"""
        survivors[k] = v_arr

    # if an equal point exists, keep survivors as-is (no duplicate)
    """if has_equal:
        return survivors"""

    # insert node
    survivors[node] = s

    # delete nodes if larger than total size
    if totalSize is not None and totalSize > 0 and len(survivors) > totalSize:
        # remove the item with the smallest sum(scoreVector)
        keys = list(survivors.keys())
        sums = np.array([np.sum(np.asarray(survivors[k], dtype=float)) for k in keys])
        drop_idx = int(np.argmin(sums))
        del survivors[keys[drop_idx]]

    return survivors

### BEGINNING OF NODE CLASS ###

class Node:
    """
    Node class: partially unmasked sequence
    - parentNode: Node object at previous time step
    - childNodes: set of M Node objects generated from sampling M distinct unmasking schemes
    - totalReward: vector of cumulative rewards for all K objectives
    - visits: number of times the node has been visited by an interation
    - path: array of partially unmasked SMILES strings leading to the node from the completely masked root node
    - timestep: the time step where the sequence was sampled
    """
    def __init__(self, args, tokens=None, log_rnd=None, log_policy_step=None, log_pretrained_step=None, parentNode=None, childNodes=None, totalReward=None, timestep=None):
        self.args = args 
        self.parentNode = parentNode
        # fixed child node list creation
        self.childNodes = [] if childNodes is None else childNodes
        
        self.log_rnd = log_rnd # stores the log_rnd up to that step
        
        #self.log_p0 = 0 # stores the log probabiltiy of the unmasking step from the previous iteration
        self.log_policy_step = log_policy_step # stores the log probability of the unmasking step under the current policy 
        self.log_pretrained_step = log_pretrained_step
        
        # initialize total rewards to the reward of the roll out unmasked sequence
        if totalReward is not None:
            self.totalReward = totalReward # potential reward of the node based on generated children 
        else:
            self.totalReward = np.zeros(self.args.num_obj)
            
        # set initial visits to 1
        self.visits = 1
        
        # set timestep (value between 0 and num_steps)
        self.timestep = timestep
        
        # dict with 'seqs' as token array and 'attention_mask' 
        self.tokens = tokens
            
    def selectNode(self):
        """
            Selects a node to move to among the children nodes based on select score
        """
        # extract the status of the current node
        nodeStatus = self.getExpandStatus()
        
        # if the node is a legal non-leaf node
        if (nodeStatus == 3):
            # initialize array that will store select score vectors of each child node
            
            paretoFront = {}
            
            for childNode in self.childNodes:
                childStatus = childNode.getExpandStatus()
                # only append child if it is legal leaf node (expandable) or legal non-leaf node
                if childStatus == 2 or childStatus == 3:
                    selectScore = childNode.calcSelectScore()
                    paretoFront = updateParetoFront(paretoFront, childNode, selectScore)
            
            selected = rd.choice(list(paretoFront.keys()))
            
            # return selected child node and status
            return selected, selected.getExpandStatus()
        
        # if node is not valid non-leaf node
        return self, nodeStatus

    def addChildNode(self, tokens, log_rnd, log_policy_step, log_pretrained_step, totalReward):
        """"
            Adds a child node:
            log_rnd: log_rnd of the path up to the added child node
            log_policy_step: scalar value of the log-prob of sampling the step under the policy
            log_pretrained_step: scalar value of the log-prob of sampling the step under the pretrained model
        """
        child = Node(args=self.args,
                     tokens=tokens, 
                     log_rnd = log_rnd,
                     log_policy_step=log_policy_step,
                     log_pretrained_step=log_pretrained_step,
                     parentNode=self, 
                     childNodes=[], 
                     totalReward=totalReward,
                     timestep=self.timestep+1)
        
        self.childNodes.append(child)
        return child
    
    def update_logrnd(self, log_policy_step, log_rnd):
        self.log_policy_step = log_policy_step
        self.log_rnd = log_rnd
        
    def updateNode(self, rewards):
        """
            Updates the cumulative rewards vector with the reward vector at a descendent leaf node. 
            Increments the number of visits to the node.
        """
        self.visits += 1
                
        self.totalReward += rewards # singleton tensor
    
    def calcSelectScore(self):
        """
            Calculates the select score for the node from the cumulative rewards vector and number of visits.
            - c: determines the degree of exploration
            - minSelectScore: determines the 
        """
        scaling = 0.1 # scaling of the second term in the select score
        
        # K-dimensional vector of normalized rewards for each objective 
        normRewards = self.totalReward / self.visits 
        
        # scales the cumulative reward by the sampling probability
        
        return normRewards + (scaling * self.log_policy_step.detach().cpu().item() * np.sqrt(self.parentNode.visits) / self.visits)
    
    def getExpandStatus(self):
        """
            Returns an integer indicating whether the node is a:
            1. terminal node (sequence is fully unmasked)
            2. legal leaf node (partially unmasked sequence that can be expanded)
            3. legal non-leaf node (already expanded sequence with M child nodes)
        """
        if self.timestep == self.args.total_num_steps:
            return 1
        elif (self.timestep < self.args.total_num_steps) and (len(self.childNodes) == 0):
            return 2
        return 3
    
### END OF NODE CLASS ###
    
### BEGINNING OF MCTS CLASS ###

class MCTS:
    def __init__(self, args, config, policy_model, pretrained, score_func_names=[], prot_seqs=None, rootNode=None):
        self.timer = StepTimer(policy_model.device)
        
        self.device = policy_model.device
        
        self.args = args
        self.config = config
        self.noise = noise_schedule.get_noise(config)
        self.time_conditioning = args.time_conditioning
        
        self.num_obj = len(score_func_names)
        
        self.mask_index = policy_model.mask_index
        masked_seq = torch.ones((self.args.seq_length), device = self.device) * self.mask_index
        masked_tokens = {'seqs': masked_seq.to(dtype=torch.long), 'attention_mask': torch.ones_like(masked_seq).to(self.device)}
        if rootNode is None:
            self.rootNode = Node(self.args, tokens = masked_tokens, 
                                 log_rnd=torch.zeros((), device=self.device), 
                                 log_policy_step=torch.zeros((), device=self.device), 
                                 log_pretrained_step=torch.zeros((), device=self.device), 
                                 totalReward=np.zeros(self.num_obj), timestep=0)
        else:
            self.rootNode = rootNode  # stores the root node of the tree
        
        # dictionary:
        # "seq": final unmasked sequence
        # "traj": list of (N_steps, L)
        # "reward": reward of the trajectory
        self.buffer = [] # List[Dict[str, Any]]
        
        self.buffer_size = args.buffer_size
        
        self.num_steps = args.total_num_steps
        #self.num_sequences = args.num_sequences
        
        # pretrained model
        self.pretrained = pretrained
        
        # the policy model that we want to finetune
        self.policy_model = policy_model
        #self.tokenizer = policy_model.tokenizer
        self.device = policy_model.device
        
        self.sequence_length = args.seq_length
            
        self.num_iter = args.num_iter
        
        self.num_children = args.num_children
        
        # score functions
        
        self.rewardFunc = ScoringFunctions(score_func_names, prot_seqs, device=args.device)

        self.iter_num = 0
        
        self.reward_log = [] # stores scalarized total rewards
        self.logrnd_log = []
        # stores each objective
        self.valid_fraction_log = []
        self.affinity1_log = []
        self.affinity2_log = []
        self.permeability_log = []
        self.sol_log = []
        self.hemo_log = []
        self.nf_log = []
        
        self.policy_model.eval()
        self.pretrained.eval()
        
        # for peptides
        self.analyzer = PeptideAnalyzer()
        self.tokenizer = policy_model.tokenizer
        
        
    def reset(self, resetTree):
        self.iter_num = 0
        self.buffer = []
        self.reward_log = []
        self.logrnd_log = []
        
        # reset logs for each objective
        self.valid_fraction_log = []
        self.affinity1_log = []
        self.affinity2_log = []
        self.permeability_log = []
        self.sol_log = []
        self.hemo_log = []
        self.nf_log = []
        
        # add option to continue with the same tree
        if resetTree:
            masked_seq = torch.ones((self.args.seq_length), device = self.device) * self.mask_index
            masked_tokens = {'seqs': masked_seq.to(dtype=torch.long), 'attention_mask': torch.ones_like(masked_seq).to(self.device)}
            self.rootNode = Node(self.args, tokens = masked_tokens, 
                                 log_rnd=torch.zeros((), device=self.device), 
                                 log_policy_step=torch.zeros((), device=self.device), 
                                 log_pretrained_step=torch.zeros((), device=self.device), 
                                 totalReward=np.zeros(self.num_obj), timestep=0)

    def forward(self, resetTree=False):
        
        self.reset(resetTree)
        
        while (self.iter_num < self.num_iter):
            self.iter_num += 1
            
            # traverse the tree form the root node until a leaf node
            with self.timer.section("select"):
                leafNode, _ = self.select(self.rootNode)
            
            # expand leaf node into num_children partially unmasked sequences at the next timestep
            with self.timer.section("expand"):
                self.expand(leafNode)
        
        final_x, log_rnd, final_rewards, score_vectors, sequences = self.consolidateBuffer()
        # return final_seqs (B, L), log_rnd (B, ), and final rewards (B, )
        
        rows = self.timer.summary()
        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")
            
        return final_x, log_rnd, final_rewards, score_vectors, sequences

    # new updateBuffer
    def _debug_buffer_decision(self, sv, reason, extra=None):
        if extra is None: extra = {}
        print(f"[BUFFER] reason={reason} sv={np.round(sv,4)} "
            f"buf_len={len(self.buffer)} extra={extra}")

    def updateBuffer(self, x_final, log_rnd, score_vectors, childSequences):
        B = x_final.shape[0]
        traj_log_rnds, scalar_rewards = [], []

        for i in range(B):
            sv = np.asarray(score_vectors[i], dtype=float)
            
            # determine how to scalarize the multi-objective rewards
            if self.args.scalarization == "normalized":
                pass
            elif self.args.scalarization == "weighted":
                pass
            else: 
                scalar_reward = float(np.sum(sv))
                
            traj_log_rnd = log_rnd[i] + (scalar_reward / self.args.alpha) # scale down by alpha

            item = {
                "x_final": x_final[i].clone(), # clone?
                "log_rnd": traj_log_rnd.clone(),
                "final_reward": scalar_reward,
                "score_vector": sv.copy(),
                "seq": childSequences[i],
            }

            # Drop if dominated by any existing
            if any(dominated_by(sv, bi["score_vector"]) for bi in self.buffer):
                # for debugging
                self._debug_buffer_decision(sv, "rejected_dominated")
                continue

            # Remove any existing that this candidate dominates
            keep = []
            for bi in self.buffer:
                if not dominates(sv, bi["score_vector"]):
                    keep.append(bi)
            self.buffer = keep

            # Insert with capacity rule
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(item)
            else:
                # tie-breaker: replace the worst by a simple heuristic (min sum)
                worst_i = int(np.argmin([np.sum(bi["score_vector"]) for bi in self.buffer]))
                self.buffer[worst_i] = item
            
            # for debugging
            self._debug_buffer_decision(sv, "inserted", {"new_len": len(self.buffer)})

            traj_log_rnds.append(traj_log_rnd)
            scalar_rewards.append(scalar_reward)

        traj_log_rnds = torch.stack(traj_log_rnds, dim=0) if traj_log_rnds else torch.empty(0)
        scalar_rewards = np.asarray(scalar_rewards, dtype=float)
        return traj_log_rnds, scalar_rewards
                
    def consolidateBuffer(self):
        """
        returns x_final, log_rnd, and final_rewards in tensors
        """
        x_final = []
        log_rnd = []
        final_rewards = []
        score_vectors = []
        sequences = []
        for item in self.buffer:
            x_final.append(item["x_final"])
            log_rnd.append(item["log_rnd"])
            final_rewards.append(item["final_reward"])
            score_vectors.append(item["score_vector"])
            sequences.append(item["seq"])
        
        x_final = torch.stack(x_final, dim=0) # (B, L)
        log_rnd = torch.stack(log_rnd, dim=0).to(dtype=torch.float32) # (B)
        final_rewards = np.stack(final_rewards, axis=0).astype(np.float32)
        score_vectors = np.stack(score_vectors, axis=0).astype(np.float32)
        
        return x_final, log_rnd, final_rewards, score_vectors, sequences
            

    def isPathEnd(self, path, maxDepth): 
        """
            Checks if the node is completely unmasked (ie. end of path)
            or if the path is at the max depth
        """
        if (path[-1] != self.mask_index).all():
            return True
        elif len(path) >= maxDepth: 
            return True
        return False
    
    def select(self, currNode, eps=1e-5):
        """
            Traverse the tree from the root node until reaching a legal leaf node
        """
        updated_log_rnd = torch.zeros((), device=self.device)
        while True: 
            currNode, nodeStatus = currNode.selectNode()
            
            if currNode.parentNode is not None:
                # compute new log_policy
                child_tokens = currNode.tokens['seqs'].to(self.device)
                attn_mask = currNode.tokens['attention_mask'].to(self.device)
                parent = currNode.parentNode
                parent_tokens = parent.tokens['seqs'].to(self.device)
                t = torch.ones(1, device = self.device)
                dt = (1 - eps) / self.num_steps
                with torch.no_grad():
                    with self.timer.section("select.compute_log_policy"):
                        updated_log_policy_step = self.policy_model.compute_log_policy(parent_tokens, 
                                                                                   child_tokens, 
                                                                                   t=t, dt=dt)
                updated_log_rnd += updated_log_policy_step
                
                currNode.update_logrnd(updated_log_policy_step, updated_log_rnd) # update log_rnd
                
            if nodeStatus != 3:
                return currNode, nodeStatus
            
    def expand(self, parentNode, eps=1e-5):
        """
        Sample unmasking steps from the pre-trained MDLM 
        adds num_children partially unmasked sequences to the children of the parentNode
        """
        
        num_children = self.num_children
        # initialize child rewards that will be added to total rewards
        
        
        # compute number of rollout steps
        # if parentNode.timestep = self.num_steps then num_rollout_steps = 1
        num_rollout_steps = self.num_steps - parentNode.timestep
        # array of rollout timesteps from the timestep of parent node to 0
        rollout_t = torch.linspace(1, eps, self.num_steps + 1, device=self.device)
        dt = (1 - eps) / self.num_steps
        
        # initialize x and attn_mask
        x = parentNode.tokens['seqs'].to(self.device)
        attn_mask = parentNode.tokens['attention_mask'].to(self.device)
        parent_log_rnd = parentNode.log_rnd # stores the log_rnd up to parent node
        
        t = rollout_t[parentNode.timestep] * torch.ones(1, 1, device = self.device)
        
        # sample M child sequences and compute their log probabilities
        with torch.no_grad():
            with self.timer.section("expand.batch_mcts_reverse_step"):
                _, x_children, child_log_policy_step, child_log_pretrained_step = \
                    self.policy_model.batch_mcts_reverse_step(token_array=x, 
                                                            t=t, dt=dt, 
                                                            batch_size=num_children, 
                                                            pretrained=self.pretrained)
        
        # compute weight of the step (num_children, 1)
    
        child_log_rnd = (parent_log_rnd + (child_log_pretrained_step - child_log_policy_step)).to(self.device)
        
        x_rollout = x_children
        
        traj_log_rnd = child_log_rnd # initialize log_rnd for entire rolled out trajectory
        
        # rollout under the policy and compute the log ratio at each step
        with self.timer.section("expand.rollout_total"):
            for i in range(1, num_rollout_steps):
                t = rollout_t[parentNode.timestep + i] * torch.ones(num_children, 1, device = self.device)
                
                with torch.no_grad():
                    _, x_next, log_policy_step, log_pretrained_step = \
                        self.policy_model.mcts_reverse_step(x_rollout, 
                                                            t=t, dt=dt, 
                                                            pretrained=self.pretrained)
                
                # add the rollout step
                traj_log_rnd += log_pretrained_step - log_policy_step
                
                x_rollout = x_next
        
        
        # if mask token remains, fully unmask
        mask_positions = (x_rollout == self.mask_index)        # (B, L) bool

        # does **any** mask remain in any sequence
        any_mask_global = mask_positions.any().item()  # true if mask remains
        if any_mask_global:
            with torch.no_grad():
                with self.timer.section("expand.noise_removal"):
                    log_p, x_next, log_policy_step, log_pretrained_step = \
                        self.policy_model.mcts_noise_removal(x_rollout, 
                                                            t=t, dt=dt, 
                                                            pretrained=self.pretrained)

            traj_log_rnd += log_pretrained_step - log_policy_step
            
            x_rollout = x_next
                
        # stores the string sequences for reward evaluation
        with self.timer.section("expand.decode"):
            childSequences = self.tokenizer.batch_decode(x_rollout)
        
        ## FOR PEPTIDES ONLY ##
        valid_x_children = []
        valid_x_final = []
        validSequences = []
        valid_traj_log_rnd = []
        
        with self.timer.section("expand.filter_is_peptide"):
            for i in range(num_children):
                # string sequence
                childSeq = childSequences[i]
                
                # check if the peptide is valid
                if self.analyzer.is_peptide(childSeq):
                    valid_x_children.append(x_children[i])
                    valid_x_final.append(x_rollout[i])
                    validSequences.append(childSeq)
                    valid_traj_log_rnd.append(traj_log_rnd[i])
                else:
                    childTokens = {'seqs': x_children[i].to(dtype=torch.long), 'attention_mask': attn_mask}
                    parentNode.addChildNode(tokens=childTokens, 
                                        log_rnd=child_log_rnd[i], 
                                        log_policy_step=child_log_policy_step[i], 
                                        log_pretrained_step=child_log_pretrained_step[i], 
                                        totalReward=np.zeros(self.num_obj))
        
        del traj_log_rnd
        
        if (len(validSequences) != 0):
            # add scores to log
            with self.timer.section("expand.scoring_functions"):
                score_vectors = self.rewardFunc(input_seqs=validSequences) # (num_children, num_objectives)
                
            average_scores = score_vectors.T
            
            self.affinity1_log.append(average_scores[0])
            self.sol_log.append(average_scores[1])
            self.hemo_log.append(average_scores[2])
            self.nf_log.append(average_scores[3])
            self.permeability_log.append(average_scores[4])

        else:
            # set the values added to log as 0s if there are no valid sequences
            self.affinity1_log.append(np.zeros((self.num_obj, self.num_children)))
            self.sol_log.append(np.zeros((self.num_obj, self.num_children)))
            self.hemo_log.append(np.zeros((self.num_obj, self.num_children)))
            self.nf_log.append(np.zeros((self.num_obj, self.num_children)))
            self.permeability_log.append(np.zeros((self.num_obj, self.num_children)))
        
        # convert to tensor
        if len(valid_x_final) == 0:
            # log and bail out gracefully for this expansion
            self.valid_fraction_log.append(0.0)
            return
        
        valid_x_final = torch.stack(valid_x_final, dim=0)
        valid_traj_log_rnd = torch.stack(valid_traj_log_rnd, dim=0)
        # update buffer and get rewards
        with self.timer.section("expand.update_buffer"):
            traj_log_rnds, scalar_rewards = self.updateBuffer(valid_x_final, valid_traj_log_rnd, score_vectors, childSequences)
        
        allChildReward = np.zeros_like(score_vectors[0])
        
        for i in range(len(score_vectors)):
            reward = score_vectors[i]
            
            # add to all child reward vector for backprop
            allChildReward += reward # (num_objectives,)
            
            # create node for sequence and add to the children node of parent
            childTokens = {'seqs': valid_x_children[i].to(dtype=torch.long), 'attention_mask': attn_mask}
            parentNode.addChildNode(tokens=childTokens, 
                                    log_rnd=child_log_rnd[i], 
                                    log_policy_step=child_log_policy_step[i], 
                                    log_pretrained_step=child_log_pretrained_step[i], 
                                    totalReward=reward)
        
        ### END OF FOR PEPTIDES ONLY ###
        
        valid_fraction = len(validSequences) / num_children
        self.valid_fraction_log.append(valid_fraction)
        
        # debugging
        print(f"[EXPAND] iter={self.iter_num} parent_t={parentNode.timestep} "
            f"num_children={num_children} valid={len(validSequences)} any_mask={any_mask_global}")
        if score_vectors is not None:
            print(f"[SCORES] min={np.min(score_vectors,0)} max={np.max(score_vectors,0)} "
                f"nan_any={np.isnan(score_vectors).any()}")
        # end debugging
        
        self.reward_log.append(scalar_rewards)
        self.logrnd_log.append(traj_log_rnds.detach().cpu().numpy())
        
        allChildReward = allChildReward / len(validSequences) # normalize by number of valid children
        # backpropogate all child rewards
        with self.timer.section("expand.backprop"):
            self.backprop(parentNode, allChildReward)
        

    def backprop(self, node, allChildReward):
        # backpropogate rewards through the path leading to the leaf node from the root
        while node:
            node.updateNode(allChildReward)
            node = node.parentNode