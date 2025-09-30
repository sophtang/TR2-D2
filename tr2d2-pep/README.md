# TR2-D2 For Multi-Objective Therapeutic Peptide Design 🧫

This part of the code is for finetuning a peptide MDM to optimize multiple therapeutic properties, including binding affinity to a protein target, solubility, non-hemolysis, non-fouling, and cell membrane permeability, with TR2-D2. 

The codebase is built upon [PepTune (Tang et.al, 2024)](https://arxiv.org/abs/2412.17780), [MDLM (Sahoo et.al, 2023)](https://github.com/kuleshov-group/mdlm), [SEPO (Zekri et.al, 2025)](https://github.com/ozekri/SEPO/tree/main), and [MDNS (Zhu et.al, 2025)](https://arxiv.org/abs/2508.10684).

## Environment Installation
```
conda env create -f environment.yml

conda activate tr2d2-pep
```

## Model Pretrained Weights Download

Follow the steps below to download the model weights requried for this experiment, which is originally from [PepTune](https://arxiv.org/abs/2412.17780). 

1. Download the PepTune pre-trained MDLM and place in `/TR2-D2/peptides/pretrained/`: https://drive.google.com/file/d/1oXGDpKLNF0KX0ZdOcl1NZj5Czk2lSFUn/view?usp=sharing 
2. Download the pre-trained binding affinity Transformer model and place in `/TR2-D2/tr2d2-pep/scoring/functions/classifiers/`: https://drive.google.com/file/d/128shlEP_-rYAxPgZRCk_n0HBWVbOYSva/view?usp=sharing



## Finetune with TR2-D2
After downloading the pretrained checkpoints, follow the steps below to run fine-tuning:

1. Fill in the `base_path` in `scoring/scoring_functions.py` and `diffusion.py`. 
2. Fill in `HOME_LOC` to the base path where `TR2-D2` is located and `ENV_PATH` to the directory where your environment is downloaded in `finetune.sh`. 
3. Create a path `tr2d2-pep/results` where the fine-tuning curves and generation results will be saved and `tr2d2-pep/checkpoints` for checkpoint saving. Also, create `tr2d2-pep/logs` where the training logs will be saved.
3. To specify a target protein, set `--prot_seq <insert amino acid sequence>` and `--prot_name <insert protein name>`. Default protein is Transferrin receptor (TfR).

Run fine-tuning using `nohup` with the following commands:
```
chmod +x finetune.sh

nohup ./finetune.sh > finetune.log 2>&1 &
```
Evaluation will run automatically after the specified number of fine-tuning epochs `--num_epochs` is finished. To summarize metrics, fill in `path` and `prot_name` in `metrics.py` and run:

```
python metrics.py
```