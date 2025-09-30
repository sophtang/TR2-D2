# [TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171) 🤖🌳



[**Sophia Tang**](https://sophtang.github.io/)\*, [**Yuchen Zhu**](https://yuchen-zhu-zyc.github.io/)\*, [**Molei Tao**](https://mtao8.math.gatech.edu/), and [**Pranam Chatterjee**](https://www.chatterjeelab.com/)

![TR2-D2](assets/tr2d2-anim.gif)

This is the repository for **[TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion](https://arxiv.org/abs/2509.25171)** 🤖🌳. It is partially built on the **[PepTune repo](https://github.com/programmablebio/peptune)** ([Tang et al. 2024](https://arxiv.org/abs/2412.17780)) and **MDNS** ([Zhu et al. 2025](https://arxiv.org/abs/2508.10684)).

Inspired by the incredible success of off-policy reinforcement learning (RL), **TR2-D2** introduces a general framework that enhances the performance of off-policy RL with tree search for discrete diffusion fine-tuning.

🤖 Off-policy RL enables learning from diffusion trajectories from the non-gradient tracking policy model by storing sampling trajectories in a replay buffer for repeated use. 

🌳 Tree search balances exploration and exploitation to generate optimal diffusion trajectories, and stores the optimal sequences in the buffer. 

We use this framework to develop an efficient discrete diffusion fine-tuning strategy that leverages **Monte-Carlo Tree Search (MCTS)** to curate a replay buffer of optimal trajectories combined with an **off-policy control-based RL algorithm grounded in stochastic optimal control theory**, yielding theoretically guaranteed convergence to the optimal distribution. 🌟

### Regulatory DNA Sequence Design 🧬

---

In this experiment, we fine-tune the pre-trained **DNA enhancer MDM from DRAKES** (Wang et al. 2025) trained on **~700k HepG2 sequences** to optimize the measured enhancer activity using the reward oracles from DRAKES. Code and instructions to reproduce our results are provided in `/tr2d2-dna`.

### Multi-Objective Therapeutic Peptide Design 🧫

---

In this experiment, we fine-tune the pre-trained **unconditional peptide SMILES MDM from PepTune** ([Tang et al. 2024](https://arxiv.org/abs/2412.17780)) to optimize **multiple therapeutic properties**, including target protein binding affinity, solubility, non-hemolysis, non-fouling, and permeability. We show that one-shot generation from the fine-tuned policy outperforms inference-time multi-objective guidance, marking a significant advance over prior fine-tuning methods. Code and instructions to reproduce our results are provided in `/tr2d2-pep`.

![TR2-D2 for Multi-Objective Peptide Design](assets/peptides.png)

## Citation

If you find this repository helpful for your publications, please consider citing our paper:

```python
@article{tang2024tr2d2,
  title={TR2-D2: Tree Search Guided Trajectory-Aware Fine-Tuning for Discrete Diffusion},
  author={Sophia Tang and Yuchen Zhu and Molei Tao and Pranam Chatterjee},
  journal={arXiv preprint arXiv:2509.25171},
  year={2025}
}
```

To use this repository, you agree to abide by the [PepTune License](https://drive.google.com/file/d/1Hsu91wTmxyoJLNJzfPDw5_nTbxVySP5x/view?usp=sharing).