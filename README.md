<div align="center">

# Bridging Semantics and Geometry: A Decoupled LVLM–SAM Framework for Reasoning Segmentation in Remote Sensing

</div>

## 🎉 News

<!-- - **2025/10/23**: Our 3B model weight has been released! 🔥 [Hugging Face](https://huggingface.co/RicardoString/Think2Seg-RS-3B). -->

## Introduction

Code and detailed documentation will be released soon.

## 🛠️ Setup

**1. Clone the repository**

```bash
git clone https://github.com/Thunderstring/Think2Seg-RS
cd Think2Seg-RS
```

**2. Create conda environment and install dependencies for Think2Seg-RS**

```bash
# Create and activate the environment
conda create -n think2seg-rs python=3.10
conda activate think2seg-rs
# Install dependencies for Think2Seg-RS
bash setup.sh
```

**3. Install SAM2**

**Note**: You can clone the SAM2 repository into any directory. It does not have to be inside the Think2Seg-RS project folder.

```bash
# Clone and install SAM2
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
# Download SAM2 Checkpoints
cd checkpoints && \
./download_ckpts.sh && \
```

## 💪🏻 Training

Before training, set the required environment variables in `src/open-r1-multimodal/run_scripts/*.sh`:

```bash
export MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
export EARTHREASON_ROOT=your_earthreason_root_path_here
export SAM_SIZE=small
export SAM_ROOT=your_sam_root_path_here
export RUN_NAME=your_sam_run_name_here
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=your_wandb_api_key_here
# set --report_to to "none" to disable wandb logging
```

Then run the bash script:

```bash
cd src/open-r1-multimodal
bash run_scripts/run_grpo_geo_ultra-qwen-3B.sh
```


## 📊 Evaluation
