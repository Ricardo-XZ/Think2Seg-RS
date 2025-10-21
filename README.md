<div align="center">

## Bridging Semantics and Geometry: A Decoupled LVLM–SAM Framework for Reasoning Segmentation in Remote Sensing

</div>

## 📖 Overview

This is the official implementation of **Think2Seg-RS**, a decoupled framework for reasoning segmentation in remote sensing (RS) imagery.

Our core idea is to decouple high-level semantic reasoning from low-level geometric execution. Specifically, we train an LVLM prompter (e.g., Qwen-2.5-VL) to control a frozen Segment Anything Model (SAM2) via structured geometric prompts. Through a mask-only reinforcement learning objective, the LVLM learns to translate abstract semantic reasoning into spatially grounded actions, achieving state-of-the-art performance on the EarthReason dataset.  

Examples of Think2Seg-RS on the EarthReason dataset:

![result show](assets/show_results_appendix.svg)


<!-- Large Vision–Language Models (LVLMs) hold great promise for advancing remote sensing (RS) analysis, yet existing reasoning segmentation frameworks couple linguistic reasoning and pixel prediction through end-to-end supervised fine-tuning, leading to weak geometric grounding and limited generalization across tasks. To address this, we developed Think2Seg-RS, a decoupled framework that trains an LVLM prompter to control a frozen Segment Anything Model (SAM) via structured geometric prompts. Through a mask-only reinforcement learning objective, the LVLM learns to translate abstract semantic reasoning into spatially grounded actions, achieving state-of-the-art performance on the EarthReason dataset. Remarkably, the learned prompting policy generalizes zero-shot to multiple referring segmentation benchmarks, exposing a distinct divide between semantic-level and instance-level grounding. We further found that compact segmenters outperform larger ones under semantic-level supervision, and that negative prompts are ineffective in heterogeneous aerial backgrounds. Together, these findings establish semantic-level reasoning segmentation as a new paradigm for geospatial understanding, opening the way toward unified, interpretable LVLM-driven Earth observation. -->

## 🎉 News

<!-- - **2025/10/23**: Our 3B model weight has been released! 🔥 [Hugging Face](https://huggingface.co/RicardoString/Think2Seg-RS-3B). -->
<!-- - **[2025/10/21]** Our paper is currently under review. -->
<!-- - **[2025/10/21]** Our paper has been submitted. -->

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

## 🚀 Training

Before training, set the required environment variables in `src/open-r1-multimodal/run_scripts/*.sh`:

```bash
export MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
export EARTHREASON_ROOT=<your_earthreason_root_path_here>
export SAM_SIZE=small
export SAM_ROOT=<your_sam_root_path_here>
export RUN_NAME=<your_sam_run_name_here>
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY=<your_wandb_api_key_here>
# set --report_to to "none" to disable wandb logging
```

Then run the bash script:

```bash
cd src/open-r1-multimodal
bash run_scripts/run_grpo_geo_ultra-qwen-3B.sh
```


## 📊 Evaluation



## 🤝 Acknowledgement

Think2Seg-RS is built upon the open-source projects [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [SAM2](https://github.com/facebookresearch/sam2), and [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct). We extend our sincere gratitude to the original authors and contributors of these remarkable works.
