import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import random as rd
from finetune_utils import to_one_hot
from utils import StepTimer

import noise_schedule

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
        self.childNodes = [] if childNodes is None else childNodes
        
        self.log_rnd = log_rnd # stores the log_rnd up to that step
        
        #self.log_p0 = 0 # stores the log probabiltiy of the unmasking step from the previous iteration
        self.log_policy_step = log_policy_step # stores the log probability of the unmasking step under the current policy 
        self.log_pretrained_step = log_pretrained_step
        
        # initialize total rewards to the reward of the roll out unmasked sequence
        self.totalReward = totalReward # potential reward of the node based on generated children 
            
        # set initial visits to 1
        self.visits = 1

        #self.path = path 
        
        # set timestep (value between 0 and num_steps)
        self.timestep = timestep
        # set the sampling probabiltiy equal to the probability from the reverse posterior
        #self.sampleProb = sampleProb # stores the probability of the sampling step under the current policy
        
        # dict with 'seqs' as token array and 'attention_mask' 
        self.tokens = tokens
            
    def selectNode(self, rootNode):
        """
            Selects a node to move to among the children nodes based on select score
        """
        # extract the status of the current node
        nodeStatus = self.getExpandStatus()
        
        # if the node is a legal non-leaf node
        if (nodeStatus == 3):
            # initialize array that will store select score vectors of each child node
            selectScores = []
            selectable_children = [] # children nodes that can be selected
            
            for childNode in self.childNodes:
                childStatus = childNode.getExpandStatus()
                # only append child if it is legal leaf node (expandable) or legal non-leaf node
                if childStatus == 2 or childStatus == 3:
                    selectScore = childNode.calcSelectScore()
                    if torch.is_tensor(selectScore) and selectScore.numel() == 1:
                        selectScore = selectScore.item()
                    
                    selectable_children.append(childNode)
                    selectScores.append(float(selectScore))
            
            # no selectable children
            if len(selectable_children) == 0:
                return rootNode, 3
            
            selectScores = np.asarray(selectScores, dtype=np.float64)
            
            temp = 1.0
            # compute softmax probabiltiies
            m = np.max(selectScores)
            exps = np.exp((selectScores - m) / temp)
            tot = exps.sum()

            if not np.isfinite(tot) or tot <= 0.0:
                probs = np.full(len(selectable_children), 1.0 / len(selectable_children))
            else:
                probs = exps / tot
            
            # choose child index from categorical distribution
            idx = np.random.choice(len(selectable_children), p=probs)
            selected = selectable_children[idx]
            
            # return selected child node and status
            return selected, selected.getExpandStatus()
       
        elif (nodeStatus == 2):
            return self, nodeStatus
        
        # if node is not valid non-leaf node
        return rootNode, 3
    
    def selectNodeTopK(self, rootNode, k = 3, temp  = 1.0):
        """
        Pick from the top-k by select score.
        Returns: (selected_node, selected_status)
        """
        nodeStatus = self.getExpandStatus()

        # If expandable leaf, return it directly
        if nodeStatus == 2:
            return self, nodeStatus

        if nodeStatus == 3:
            selectable_children = []
            selectScores = []

            # collect candidates
            for ch in self.childNodes:
                s = ch.getExpandStatus()
                if s in (2, 3):
                    sc = ch.calcSelectScore()
                    if torch.is_tensor(sc):
                        sc = sc.item() if sc.numel() == 1 else float(sc.mean().item())
                    sc = float(sc) if np.isfinite(sc) else -np.inf  # push bad scores to -inf
                    selectable_children.append(ch)
                    selectScores.append(sc)

            if not selectable_children:
                return rootNode, 3

            scores = np.asarray(selectScores, dtype=np.float64)

            # top-k indices (largest scores)
            k_eff = min(k, len(scores))
            topk_idx = np.argpartition(-scores, kth=k_eff-1)[:k_eff]
            # sort the top-k by score desc for stability
            topk_idx = topk_idx[np.argsort(-scores[topk_idx])]

            # slice down to top-k pool
            pool_scores = scores[topk_idx]
            pool_children = [selectable_children[i] for i in topk_idx]

            # softmax over the top-k
            m = np.max(pool_scores)
            z = (pool_scores - m) / max(1e-8, temp)
            exps = np.exp(np.clip(z, -60, 60))
            tot = exps.sum()
            if not np.isfinite(tot) or tot <= 0.0:
                idx_local = 0  # best
            else:
                probs = exps / tot
                
                idx_local = int(np.random.choice(len(pool_children), p=probs))

            selected = pool_children[idx_local]
            return selected, selected.getExpandStatus()

        return rootNode, 3

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
        # K-dimensional vector of normalized rewards for each objective 
        normRewards = self.totalReward / self.visits 
        
        # scales the cumulative reward by the sampling probability
        
        return normRewards + (self.args.exploration * self.log_policy_step * np.sqrt(self.parentNode.visits) / self.visits)
    
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
    def __init__(self, args, config, policy_model, pretrained, rewardFunc, rootNode=None):
        self.timer = StepTimer(policy_model.device)
        
        # debugging
        self.buf_stats = {"insert":0, "replace":0, "reject_worse":0,
                          "reject_dup":0, "reject_nonfinite":0}
        self._seen_hashes = set()
        
        self.device = policy_model.device
        print(f"MCTS device: {self.device}")
        
        self.args = args
        self.config = config
        self.noise = noise_schedule.get_noise(config)
        self.time_conditioning = args.time_conditioning
        
        self.mask_index = policy_model.mask_index
        masked_seq = torch.ones((self.args.seq_length), device = self.device) * self.mask_index
        masked_tokens = {'seqs': masked_seq.to(dtype=torch.long), 'attention_mask': torch.ones_like(masked_seq).to(self.device)}
        if rootNode is None:
            self.rootNode = Node(self.args, tokens = masked_tokens, 
                                 log_rnd=torch.zeros((), device=self.device), 
                                 log_policy_step=torch.zeros((), device=self.device), 
                                 log_pretrained_step=torch.zeros((), device=self.device), 
                                 totalReward=torch.zeros((), device=self.device), timestep=0)
        else:
            self.rootNode = rootNode  # stores the root node of the tree
        
        # dictionary:
        # "seq": final unmasked sequence
        # "traj": list of (N_steps, L)
        # "reward": reward of the trajectory
        self.buffer = [] # List[Dict[str, Any]]
        
        self.buffer_size = args.buffer_size
        
        self.num_steps = args.total_num_steps
        self.num_sequences = args.num_sequences
        
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
        self.rewardFunc = rewardFunc

        self.iter_num = 0
        
        self.reward_log = []
        self.logrnd_log = []
        
        self.policy_model.eval()
        self.pretrained.eval()
        self.rewardFunc.eval()
    
    def _hash_tokens(self, t):
        # t: (L,) torch.long
        return tuple(t.detach().cpu().tolist())
        
    def reset(self, resetTree):
        self.iter_num = 0
        self.buffer = []
        self._seen_hashes = set()  # Clear the hash set too!
        self.reward_log = []
        self.logrnd_log = []
        
        # add option to continue with the same tree
        if resetTree:
            masked_seq = torch.ones((self.args.seq_length), device = self.device) * self.mask_index
            masked_tokens = {'seqs': masked_seq.to(dtype=torch.long), 'attention_mask': torch.ones_like(masked_seq).to(self.device)}
            self.rootNode = Node(self.args, tokens = masked_tokens, 
                                 log_rnd=torch.zeros((), device=self.device), 
                                 log_policy_step=torch.zeros((), device=self.device), 
                                 log_pretrained_step=torch.zeros((), device=self.device), 
                                 totalReward=torch.zeros((), device=self.device), timestep=0)

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
        
        final_x, log_rnd, final_rewards = self.consolidateBuffer()
        
        rows = self.timer.summary()
        print("\n=== Timing summary (by total time) ===")
        for name, cnt, total, mean, p50, p95 in rows:
            print(f"{name:30s}  n={cnt:5d}  total={total:8.3f}s  mean={mean*1e3:7.2f}ms  "
                f"p50={p50*1e3:7.2f}ms  p95={p95*1e3:7.2f}ms")
        
        # return final_seqs (B, L), log_rnd (B, ), and final rewards (B, )
        return final_x, log_rnd, final_rewards

    
    def updateBuffer(self, x_final, log_rnd, final_reward):
        B = x_final.shape[0]
        for i in range(B):
            # Finite check
            if not torch.isfinite(final_reward[i]) or not torch.isfinite(log_rnd[i]):
                self.buf_stats["reject_nonfinite"] += 1
                continue

            h = self._hash_tokens(x_final[i])
            if h in self._seen_hashes:
                self.buf_stats["reject_dup"] += 1
                continue

            item = {"x_final": x_final[i].clone(),
                    "log_rnd": log_rnd[i].clone(),
                    "final_reward": final_reward[i].clone()}

            if len(self.buffer) < self.buffer_size:
                self.buffer.append(item)
                self._seen_hashes.add(h)
                self.buf_stats["insert"] += 1
            else:
                # replace if strictly better, or tie-break with log_rnd
                min_idx, min_item = min(
                    enumerate(self.buffer),
                    key=lambda kv: (kv[1]["final_reward"].item(), kv[1]["log_rnd"].item())
                )
                cand_key = (final_reward[i].item(), log_rnd[i].item())
                min_key  = (min_item["final_reward"].item(), min_item["log_rnd"].item())

                if cand_key > min_key:  # allow ties via 2nd key
                    # update hash set
                    old_h = self._hash_tokens(self.buffer[min_idx]["x_final"])
                    if old_h in self._seen_hashes:
                        self._seen_hashes.remove(old_h)
                    self.buffer[min_idx] = item
                    self._seen_hashes.add(h)
                    self.buf_stats["replace"] += 1
                else:
                    self.buf_stats["reject_worse"] += 1
    
    def print_buffer_stats(self):
        print("[BUFFER] ",
              " ".join(f"{k}={v}" for k,v in self.buf_stats.items()),
              f" size={len(self.buffer)}/{self.buffer_size}")
        if len(self.buffer):
            vals = torch.stack([b["final_reward"] for b in self.buffer]).float()
            print(f"[BUFFER] reward min/mean/max: {vals.min():.4f} {vals.mean():.4f} {vals.max():.4f}")
    
    def consolidateBuffer(self):
        """
        returns x_final, log_rnd, and final_rewards in tensors
        """
        x_final = []
        log_rnd = []
        final_rewards = []
        for item in self.buffer:
            x_final.append(item["x_final"])
            log_rnd.append(item["log_rnd"])
            final_rewards.append(item["final_reward"])
        
        x_final = torch.stack(x_final, dim=0) # (B, L)
        log_rnd = torch.stack(log_rnd, dim=0).to(dtype=torch.float32) # (B)
        final_rewards = torch.stack(final_rewards, dim=0).to(dtype=torch.float32) # (B)
        
        return x_final, log_rnd, final_rewards
            

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
        #iter = 1
        updated_log_rnd = torch.zeros((), device=self.device)
        while True: 
            if self.args.select_topk:
                currNode, nodeStatus = currNode.selectNodeTopK(self.rootNode, k=self.args.select_topk_value, temp=1.0)
            else:
                currNode, nodeStatus = currNode.selectNode(self.rootNode)
            
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
                updated_log_rnd += (currNode.log_pretrained_step - updated_log_policy_step)
                
                currNode.update_logrnd(updated_log_policy_step, updated_log_rnd) # update log_rnd
            
            # node is terminal node or logal leaf node, return for expansion
            if nodeStatus == 2:
                return currNode, nodeStatus
            elif nodeStatus == 1:
                currNode = self.rootNode
            
    def expand(self, parentNode, eps=1e-5):
        """
            Sample unmasking steps from the pre-trained MDLM 
            adds num_children partially unmasked sequences to the children of the parentNode
        """
        
        num_children = self.num_children
        # initialize child rewards that will be added to total rewards
        
        allChildReward = torch.zeros((), device=self.device)
        
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
        
        # generate (n_children, seq_length) array of sampled children nodes
        
        # sample M child sequences and compute their log probabilities
        with torch.no_grad():
            with self.timer.section("expand.batch_mcts_reverse_step"):
                child_log_p, x_children, child_log_policy_step, child_log_pretrained_step = \
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
                    log_p, x_next, log_policy_step, log_pretrained_step = \
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
        
        x_final = x_rollout # final sequences (B, L)
        
        # edit? how is the reward model defined?
        #childSequences = self.tokenizer.batch_decode(x_rollout)
        
        #if self.args.data == "peptide":
            #validSequences = []
            
        
        # get final rewards
        x_one_hot = to_one_hot(x_final)
        x_one_hot_reward = torch.transpose(x_one_hot, 1, 2)
        reward_preds = self.rewardFunc(x_one_hot_reward).squeeze(-1) # (num_children, 4)
        rewards_value = reward_preds[:, 0] # (num_children, 1)
        
        if self.args.reward_clip:
            rewards = torch.clamp(rewards_value, max=self.args.reward_clip_value)
        else:
            rewards = rewards_value
        
        traj_log_rnd += rewards / self.args.alpha
        
        self.reward_log.append(rewards.detach().cpu().numpy())
        self.logrnd_log.append(traj_log_rnd.detach().cpu().numpy())
        
        # update buffer
        with self.timer.section("expand.update_buffer"):
            self.updateBuffer(x_final, traj_log_rnd, rewards)
            
        for i in range(num_children):
            
            # add to all child reward vector for backprop
            allChildReward += rewards[i]
            
            # create node for sequence and add to the children node of parent
            childTokens = {'seqs': x_children[i].to(dtype=torch.long), 'attention_mask': attn_mask}
            parentNode.addChildNode(tokens=childTokens, 
                                    log_rnd=child_log_rnd[i], 
                                    log_policy_step=child_log_policy_step[i], 
                                    log_pretrained_step=child_log_pretrained_step[i], 
                                    totalReward=rewards[i])
        
        # backpropogate all child rewards
        with self.timer.section("expand.backprop"):
            self.backprop(parentNode, allChildReward)
        

    def backprop(self, node, allChildReward):
        # backpropogate rewards through the path leading to the leaf node from the root
        while node:
            node.updateNode(allChildReward)
            node = node.parentNode