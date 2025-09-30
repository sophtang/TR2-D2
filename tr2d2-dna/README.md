# TR2-D2 For Enhancer DNA Design

This part of the code is for finetuning DNA sequence models for optimizing DNA enhancer activity with TR2-D2.

The codebase is built upon [MDLM (Sahoo et.al, 2023)](https://github.com/kuleshov-group/mdlm), [Drakes (Wang et.al, 2024)](https://github.com/ChenyuWang-Monica/DRAKES), [SEPO (Zekri et.al, 2025)](https://github.com/ozekri/SEPO/tree/main), and [MDNS (Zhu et.al, 2025)](https://arxiv.org/abs/2508.10684).

## Environment Installation
```
conda create -n tr2d2-dna python=3.9.18

conda activate tr2d2-dna

bash env.sh
```

## Model Pretrained Weights Download

All data and model weights can be downloaded from the link below, which is provided by the [DRAKES](https://arxiv.org/abs/2410.13643) author. Save the downloaded file in `$BASE_PATH`.

https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=0

For downloading using terminal, use 

```
curl -L -o dna.zip "https://www.dropbox.com/scl/fi/zi6egfppp0o78gr0tmbb1/DRAKES_data.zip?rlkey=yf7w0pm64tlypwsewqc01wmfq&st=xe8dzn8k&dl=0"

unzip dna.zip
```

## Finetune with TR2-D2
After downloading the pretrained checkpoints, fill in the `base_path` in `dataloader_gosai.py`, `oracle.py`, and `finetune.sh`. Fill in `HOME_LOC` and `SAVE_PATH` in `finetune.sh` as well.

Reproduce the DNA experiments with $\alpha = 0.1$ using
```
sbatch train.sh
```

## Evaluate saved checkpoints
The checkpoints will be saved to `SAVE_PATH`.
Fill in `RUNS_DIR` in `run_batch_eval.sh` to be the same as `SAVE_PATH`. The checkpoints can be evaluated with
```
sbatch run_batch_eval.sh
```






