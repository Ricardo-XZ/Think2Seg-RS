# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from open_r1.trainer import GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.vlm_modules import *

# ----------------------- Fix the flash attention bug in the current version of transformers -----------------------
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
import numpy as np
from typing import Tuple
from transformers.utils import logging
from open_r1.dataset.combind_datasets import create_combined_dataset
from open_r1.trainer.geo_grpo_trainer_ultra import Geo_VLMGRPOTrainer_ultra
from open_r1.geo_reward_func_ultra import GEO_Reward_Func_Ultra

logger = logging.get_logger(__name__)

def custom_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        # print(111, 222, 333, 444, 555, 666, 777, 888, 999)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `rotary_pos_emb` (2D tensor of RoPE theta values), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.54 `rotary_pos_emb` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
            # Add this
            cos = cos.to(torch.float)
            sin = sin.to(torch.float)
        q, k = apply_rotary_pos_emb_flashatt(q.unsqueeze(0), k.unsqueeze(0), cos, sin)
        q = q.squeeze(0)
        k = k.squeeze(0)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen).reshape(
            seq_length, -1
        )
        attn_output = self.proj(attn_output)
        return attn_output

Qwen2_5_VLVisionFlashAttention2.forward = custom_forward

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    # reward_funcs: list[str] = field(
    #     default_factory=lambda: ["accuracy", "format"],
    #     metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    # )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    sam_model_size: Optional[str] = field(
        default='tiny',
        metadata={"help": "Size of SAM model to use"},
    )
    sam_root: Optional[str] = field(
        default='../../sam2',
        metadata={"help": "Root directory of SAM model"},
    )
    sam_device: Optional[str] = field(
        default='cpu',
        metadata={"help": "Device to use for SAM model"},
    )
    use_datasets: Optional[str] = field(
        default='earthreason',
        metadata={"help": "Comma-separated list of datasets to use."},
    )
    earthreason_root: Optional[str] = field(
         default='',
        metadata={"help": "Root directory for EarthReason dataset"},
    )
    earthreason_resize_size: Optional[int] = field(
        default=840,  # Default resize size for EarthReason dataset
        metadata={"help": "Resize size for EarthReason dataset images"},
    )



def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format_reward.txt"), "a", encoding='utf-8') as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")
                f.write(f"-----------------------------------\n\n")

    return [1.0 if match else 0.0 for match in matches]


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    elif "internvl" in model_name_or_path.lower():
        return InvernVLModule
    elif "llava" in model_name_or_path.lower():
        return LLaVAModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def main(script_args, training_args, model_args):

    # import random
    # random.seed(training_args.seed)
    # np.random.seed(training_args.seed)
    # torch.manual_seed(training_args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(training_args.seed)
    #     torch.cuda.manual_seed_all(training_args.seed)
    
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)

    reward_funcs = []

    ################ GEO_Reward_Func_Ultra #############
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" 

    geo_reward_func = GEO_Reward_Func_Ultra(script_args.sam_model_size, script_args.sam_root, device)
    reward_funcs.append(geo_reward_func.sam_reward_func_ultra)
    reward_funcs.append(geo_reward_func.sam_format_reward)
    reward_funcs.append(geo_reward_func.thk_ans_format_reward)


    print("reward_funcs:", reward_funcs)

    # Load the datasets based on user configuration
    datasets_to_use = script_args.use_datasets.split(',')
    available_datasets = []

    # Import required dataset classes
    # Initialize selected datasets
    if 'earthreason' in datasets_to_use:
        from open_r1.dataset.EarthReason_datasets import EarthReasonDataset
        print(f"Loading EarthReason dataset from {script_args.earthreason_root}...")
        earthreason_dataset = EarthReasonDataset(data_dir=script_args.earthreason_root, 
                                                 split=['train'],
                                                 resize_size=script_args.earthreason_resize_size)
        available_datasets.append(earthreason_dataset)
    # if 'refsegrs' in datasets_to_use:
    #     from open_r1.dataset.RefSegRS_datasets import RefSegRSDataset
    #     print(f"Loading RefSegRS dataset from {script_args.refsegrs_root}...")
    #     refsegrs_dataset = RefSegRSDataset(data_root=script_args.refsegrs_root, split='train')
    #     available_datasets.append(refsegrs_dataset)

    # if 'rrsisd' in datasets_to_use:
    #     from open_r1.dataset.rrsisd_datasets import RRSISD_Dataset
    #     print(f"Loading RRSIS-D dataset from {script_args.rrsisd_root}...")
    #     rrsisd_dataset = RRSISD_Dataset(data_dir=script_args.rrsisd_root, 
    #                                              split='train')
    #     available_datasets.append(rrsisd_dataset)
    # if 'risbench' in datasets_to_use:
    #     from open_r1.dataset.risbench_datasets import RisBenchDataset
    #     risbench_dataset = RisBenchDataset(
    #         data_dir=script_args.risbench_root,
    #         split='train',
    #         resize_size=None,
    #     )
    #     available_datasets.append(risbench_dataset)

    # Create combined dataset from all available datasets
    if len(available_datasets) > 1:
        print(f"Creating combined dataset from {len(available_datasets)} datasets...")
        dataset = create_combined_dataset(*available_datasets)
    elif len(available_datasets) == 1:
        print("Using single dataset...")
        dataset = available_datasets[0]
    else:
        raise ValueError("No datasets were loaded. Please check your dataset configuration.")

    from torch.utils.data import random_split
    splits = {'train': dataset}
    script_args.val_split_ratio = 0.0
    if script_args.val_split_ratio > 0:
        val_size = int(len(dataset) * script_args.val_split_ratio)
        train_size = len(dataset) - val_size

        # random_split 
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        splits['train'] = train_dataset
        splits['validation'] = val_dataset
        print(f"Dataset split into train ({len(train_dataset)} samples) and validation ({len(val_dataset)} samples)")
    else:
        print(f"No validation split requested. Using entire dataset ({len(dataset)} samples) for training.")



    # Select trainer class based on vlm_trainer argument
    trainer_cls = Geo_VLMGRPOTrainer_ultra
    print("using trainer:", trainer_cls.__name__)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        # reward_weights=reward_weights,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
