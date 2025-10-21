from transformers import (
    LlavaForConditionalGeneration, 
    LlavaProcessor, 
    AutoProcessor,
    AutoConfig
)
from transformers.models.llava_next import LlavaNextForConditionalGeneration
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from open_r1.vlm_modules.vlm_module import VLMBaseModule

class LLaVAModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "llava"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        # Check if it's a LLaVA model
        if "llava" not in model_id.lower():
            raise ValueError(f"Unsupported model: {model_id}")
        
        # Determine model class based on config
        try:
            config = AutoConfig.from_pretrained(model_id)
            model_type = getattr(config, 'model_type', '')
            
            if model_type == 'llava':
                # LLaVA-1.5 models
                model_cls = LlavaForConditionalGeneration
            elif model_type == 'llava_next':
                # LLaVA-Next (1.6+) models
                model_cls = LlavaNextForConditionalGeneration
            else:
                # Fallback: try to infer from model_id
                if any(version in model_id.lower() for version in ['v1.6', '1.6', 'next']):
                    model_cls = LlavaNextForConditionalGeneration
                else:
                    model_cls = LlavaForConditionalGeneration
                    
        except Exception as e:
            # Fallback to LlavaForConditionalGeneration if config loading fails
            print(f"Warning: Could not determine LLaVA model type from config: {e}")
            model_cls = LlavaForConditionalGeneration
        
        # "use_cache" should be removed as LLaVA models don't support it during training
        model_init_kwargs.pop("use_cache", None)
        # model_init_kwargs["use_cache"] = False
        
        # Handle flash attention implementation for LLaVA
        # Both LLaVA and LLaVA-Next support standard "attn_implementation" parameter
        attn_impl = model_init_kwargs.get("attn_implementation", "")
        if attn_impl == "flash_attention_2":
            # LLaVA supports flash_attention_2 natively, so keep it as is
            pass
        
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        # Use AutoProcessor for better compatibility with both LLaVA-1.5 and LLaVA-Next
        # AutoProcessor will automatically select the appropriate processor class
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['vision_tower', 'multi_modal_projector']
    
    def get_custom_multimodal_keywords(self):
        # pixel_values: common to both LLaVA-1.5 and LLaVA-Next
        # image_sizes: additional key in LLaVA-Next for dynamic resolution
        return ['pixel_values', 'image_sizes']

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        return []
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text

    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # LLaVA specific input processing
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens
            )
        return prompt_inputs

    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."

    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the LLaVA model output matches a specific format."""
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def format_reward(completions, **kwargs):
        import re
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from LLaVA model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        
        for content, sol in zip(contents, solution):
            reward = 0.0
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        reward = iou(bbox, sol)
            except Exception:
                pass
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards