from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
import json
import re
import os
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from qwen_vl_utils import process_vision_info
import sys
import cv2
from matplotlib.patches import Polygon
import PIL
# Add path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "open-r1-multimodal/src"))
# from open_r1.grpo_geo import GEO_Reward_Func

# Ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import datetime
import torch.distributed as dist
def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank) 
    
    dist.init_process_group(backend="nccl", 
                            timeout=datetime.timedelta(hours=1))
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    return local_rank, world_size, rank


def parse_args():
    parser = argparse.ArgumentParser(description='Test SAM segmentation model with GEOBench dataset')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='./geo_results', help='Output directory for visualizations')
    parser.add_argument('--sam_model_size', type=str, default='small', help='SAM model size (tiny, small, base_plus, large)')
    parser.add_argument('--sam_root', type=str, default='', help='Root directory for SAM model')
    parser.add_argument('--sam_device', type=str, default='cuda:1', help='Device to use (cuda or cpu) for sam')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to test')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_results', action='store_true', help='Save results to JSON file')
    parser.add_argument('--visualize_num', type=int, default=50, help='Number of samples to visualize')
    parser.add_argument('--dataset', type=str, default='earthreason', help='Dataset to use for evaluation.')
    parser.add_argument('--refsegrs_root', type=str, default='', help='Root directory for RefSegRS dataset')
    parser.add_argument('--earthreason_root', type=str, default='', help='Root directory for EarthReason dataset')
    parser.add_argument('--rrsisd_root', type=str, default='', help='Root directory for RRSIS-D dataset')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use (train, val, test)')
    parser.add_argument('--resize_size', type=int, default=896, help='Resize size for input images')

    return parser.parse_args()

def extract_sam_json(content):
    """Extract SAM JSON format prompts from model output"""
    # Extract content from <answer> tags
    answer_pattern = r'<answer>(.*?)</answer>'
    json_pattern = r'```json\s*(.*?)\s*```'
    
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        json_match = re.search(json_pattern, answer_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                # Try to parse JSON
                parsed_json = json.loads(json_str)
                return parsed_json
            except json.JSONDecodeError:
                # Try to clean and parse again
                if "[" in json_str and "]" in json_str:
                    start = json_str.find("[")
                    end = json_str.rfind("]") + 1
                    json_str = json_str[start:end]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        return None
    return None

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=1.5, linestyle='--'))    

def visualize_prediction(img, predicted_mask, gt_mask, extracted_answer, question_text, iou_reward, vis_path, bbox=None, gt_bbox=None):
    """
    Create visualization image for prediction results
    
    Args:
        img: PIL image object
        predicted_mask: Predicted mask
        gt_mask: Ground truth mask
        extracted_answer: JSON answer extracted from model output
        question_text: Question text
        iou_reward: IoU value
        vis_path: Path to save result image
        bbox: Predicted bounding box
        gt_bbox: Ground truth bounding box
    """
    img_np = np.array(img)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # First subplot: original image + prompts + predicted mask
    axes[0].imshow(img_np)
    show_mask(predicted_mask, axes[0])
    
    if bbox is not None:
        show_box(bbox, axes[0])

    if extracted_answer:
        for instance in extracted_answer:
            points_coords = []
            points_labels = []
            box_coords = None

            # Handle bounding box
            box_coords = instance.get("bbox_2d", None)
            if box_coords:
                box_coords = np.array(box_coords)
                show_box(box_coords, axes[0])
            
            # Handle points
            positive_points = instance.get("positive_points", [])
            negative_points = instance.get("negative_points", [])
            input_points = positive_points.copy()
            input_points.extend(negative_points)
            input_label = [1] * len(positive_points) + [0] * len(negative_points)
            if input_points:
                points_coords = np.array(input_points)
                points_labels = np.array(input_label)
                show_points(points_coords, points_labels, axes[0])
    
    axes[0].set_title("predicted mask")
    axes[0].axis('off')
    
    # Second subplot: original image + predicted mask
    axes[1].imshow(img_np)
    show_mask(gt_mask, axes[1], borders=False)
    # Show GT bbox (if exists)
    if gt_bbox is not None:
        for ins_gt in gt_bbox:
            ins_box = ins_gt.get("bbox", None)
            if ins_box:
                # ins_box = np.array(ins_box)
                show_box(ins_box, axes[1])
    axes[1].set_title("GT mask")
    axes[1].axis('off')
    
    # Add question text as overall title
    quoted_content = re.findall(r'"([^"]*)"', question_text)
    if quoted_content:
        text = quoted_content[0]  # Take the first matched content
    else:
        text = question_text  # If no quotes, use original text
    # Add question text as overall title
    plt.suptitle(f"question: {text}\nIoU: {iou_reward if iou_reward is not None else 'N/A'}", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(vis_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def parse_model_path(model_path):
    """Extract RUN_NAME and STEP from model path"""
    # Model path format: /path/to/output/RUN_NAME/checkpoint-STEP
    parts = model_path.split('/')
    
    # Get the last two parts: RUN_NAME and checkpoint-STEP
    if 'checkpoint-' in parts[-1]:
        run_name = parts[-2]
        step = parts[-1].split('-')[-1]
    else:
        # If model path doesn't contain checkpoint
        run_name = parts[-1]
        step = 'final'
        
    return run_name, step

def main(local_rank, world_size, rank):
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize vlm_module (consistent with training)
    from open_r1.vlm_modules.internvl_module import InvernVLModule
    vlm_module = InvernVLModule()
    # Load InternVL model
    print(f"Loading InternVL model from {args.model_path}...")
    from transformers import AutoModel
    model_cls = vlm_module.get_model_class(args.model_path, {})
    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map={"": local_rank},
    )
    
    # tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # model.generation_config.pad_token_id = tokenizer.pad_token_id
    # vlm_module.post_model_init(model, tokenizer)
    # tokenizer.max_anyres_num = 12

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    vlm_module.post_model_init(model, processor)
    # processor = tokenizer

    # Extract RUN_NAME and STEP from model path
    run_name, step = parse_model_path(args.model_path)
    
    # Create output directory
    # out_dir = os.path.join(args.output_dir, f"{run_name}_step{step}_{args.sam_model_size}")
    out_dir = os.path.join(args.output_dir, f"step{step}_{args.sam_model_size}_{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == 'earthreason':
        from open_r1.dataset.EarthReason_datasets import EarthReasonDataset_test
        dataset = EarthReasonDataset_test(data_dir=args.earthreason_root, split=args.split, resize_size=args.resize_size)
    elif args.dataset == 'refsegrs':
        from open_r1.dataset.RefSegRS_datasets import RefSegRSDataset
        dataset = RefSegRSDataset(data_root=args.refsegrs_root, split=args.split)
    elif args.dataset == 'rrsisd':
        from open_r1.dataset.rrsisd_datasets import RRSISD_Dataset
        dataset = RRSISD_Dataset(data_dir=args.rrsisd_root, split=args.split)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # If sample count is specified, randomly select the specified number of samples
    if (args.num_samples < len(dataset)) and (args.num_samples > 0):
        indices = random.sample(range(len(dataset)), args.num_samples)
        data = [dataset[i] for i in indices]
    else:
        data = [dataset[i] for i in range(len(dataset))]
    
    print(f"Testing on {len(data)} samples...")
    
    # Split data for distributed evaluation
    per_rank_data = len(data) // world_size
    start_idx = rank * per_rank_data
    end_idx = start_idx + per_rank_data if rank < world_size - 1 else len(data)
    rank_data = data[start_idx:end_idx]

    print(f"Rank {rank} has {len(rank_data)} samples, from {start_idx} to {end_idx}")

    messages = []
    for sample in rank_data:
        image_path = os.path.join(sample['image_path'])

        message = sample['prompt']
        message[1]['content'][0]['image'] = f'file://{image_path}'

        messages.append(message)

    rank_outputs = {
        'text': [],
        'reward': [],
        'masks': []
    }
    all_outputs = []  # List to store all answers

    # Process data
    for i in tqdm(range(0, len(messages), args.batch_size), desc=f"Rank {rank}"):
        batch_messages = messages[i:i + args.batch_size]
        batch_data = rank_data[i:i + args.batch_size]

        # Use the same data processing as training
        # First prepare prompt text
        prompts_text = vlm_module.prepare_prompt(processor, batch_data)
        
        # Process image data
        images = []
        for x in batch_data:
            if "image" in x:
                imgs = [x["image"]] if not isinstance(x["image"], list) else x["image"]
            elif "image_path" in x and x["image_path"] is not None:
                img_paths = [x["image_path"]] if not isinstance(x["image_path"], list) else x["image_path"]
                imgs = [PIL.Image.open(p) for p in img_paths]
            
            for img in imgs:
                try:
                    # Ensure minimum size of 28 pixels (consistent with training)
                    w, h = img.size
                    if w < 28 or h < 28:
                        if w < h:
                            new_w = 28
                            new_h = int(h * (28/w))
                        else:
                            new_h = 28
                            new_w = int(w * (28/h))
                        img = img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
                except:
                    pass
                images.append(img)

        # Use vlm_module to prepare model inputs
        inputs = vlm_module.prepare_model_inputs(
            processor,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        # Fix device issues
        inputs = {k: v.to(f"cuda:{local_rank}") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Use the same generation config as training
        generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,  # Usually use greedy decoding for inference
            temperature=1.0,
            # pad_token_id=tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else processor.pad_token_id,
            use_cache=True,
        )
        
        # If vlm_module has eos_token_id settings
        if hasattr(vlm_module, "get_eos_token_id"):
            generation_config.eos_token_id = vlm_module.get_eos_token_id(processor)

        # Inference: Generation of the output
        with torch.no_grad():
            # Use the same generation method as training
            generated_ids = model.generate(
                **{k: v for k, v in inputs.items() if k not in vlm_module.get_non_generate_params()},
                generation_config=generation_config
            )
            
            prompt_length = inputs["input_ids"].size(1)
            if not vlm_module.is_embeds_input():
                prompt_completion_ids = generated_ids
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # For embeds input case
                completion_ids = generated_ids
        
        # Decode the generated completion part
        batch_output_text = processor.batch_decode(
            completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        rank_outputs['text'].extend(batch_output_text)

    print(f"Rank {rank} has finished processing {len(rank_outputs['text'])} examples")

    all_outputs = [None] * len(data)
    rank_results = [(start_idx + i, output) for i, output in enumerate(rank_outputs['text'])]

    gathered_results = [None] * world_size
    dist.all_gather_object(gathered_results, rank_results)
    
    assert gathered_results[-1][-1][0] == len(data) - 1

    # Free cache
    torch.cuda.empty_cache()
    del model, processor

    # The main process will collect all results
    if rank == 0:
        for results in gathered_results:
            for idx, output in results:
                assert idx < len(all_outputs)
                all_outputs[idx] = output
        assert all_outputs[-1] is not None

        # torch.cuda.empty_cache()
        print(f"Initializing SAM model (size: {args.sam_model_size})...")

        from open_r1.geo_reward_func_ultra import GEO_Reward_Func_Ultra
        geo_reward_func_ultra = GEO_Reward_Func_Ultra(
            sam_model_size=args.sam_model_size,
            sam_root=args.sam_root,
            device=args.sam_device
        )

        final_output = []
        total_iou = 0.0
        
        # Variables for gIoU and cIoU calculation
        total_intersection = 0.0  # cumulative intersection for cIoU
        total_union = 0.0  # cumulative union for cIoU
        valid_samples = 0  # count of samples with valid predictions

        for i, (input_example, model_output) in tqdm(enumerate(zip(data, all_outputs)), total=len(data), desc="Processing samples"):
            original_output = model_output
            
            # Extract SAM JSON answer
            # extracted_answer = extract_sam_json(original_output)
            extracted_answer, error_msg = geo_reward_func_ultra.parse_sam_json(original_output)
            think_content = geo_reward_func_ultra.parse_think_content(original_output)

            # Calculate IoU
            gt_mask = input_example['GT_mask']
            img = input_example['image']

            rewards, all_details = geo_reward_func_ultra.sam_iou_reward_func_ultra_test(
                [[({"content": original_output})]],
                [gt_mask],
                [img],
                return_details=True
            )
            iou_reward = rewards[0]
            union = all_details['union'][0]
            intersection = all_details['intersection'][0]

            thk_ans_format_reward = geo_reward_func_ultra.thk_ans_format_reward([[({"content": original_output})]])
            sam_format_reward = geo_reward_func_ultra.sam_format_reward([[({"content": original_output})]])
            
            data_idx = input_example['data_idx']

            predicted_mask = all_details['pred_masks'][0] if all_details['pred_masks'][0] is not None else None
            
            # Update cumulative metrics only for valid predictions
            if iou_reward is not None:
                total_iou += iou_reward
            if predicted_mask is not None and intersection is not None and union is not None:
                total_intersection += intersection
                total_union += union
                valid_samples += 1
            elif predicted_mask is not None and not predicted_mask.sum():
                valid_samples += 1
            else:
                total_union += np.sum(gt_mask > 0)
            
            # visualization
            if args.save_results and predicted_mask is not None and i <= args.visualize_num:
                vis_dir = os.path.join(out_dir, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)

                vis_path = os.path.join(vis_dir, f"sample_{i}.png")
                # Get question text
                question_text = input_example['prompt'][1]['content'][1]['text']
                
                gt_bbox = None
                
                # Call visualization function
                visualize_prediction(
                    img=input_example['image'],
                    predicted_mask=predicted_mask,
                    gt_mask=gt_mask,
                    extracted_answer=extracted_answer,
                    question_text=question_text,
                    iou_reward=iou_reward,
                    vis_path=vis_path,
                    gt_bbox=gt_bbox
                )

            
            result = {
                'sample_id': i,
                'image_shape': gt_mask.shape,
                'image_path': input_example['image_path'],
                'GT_mask_path': input_example['GT_mask_path'],
                'question': input_example['prompt'][1]['content'][1]['text'],
                'model_output': original_output,
                'think_content': think_content,
                'extracted_answer': extracted_answer,
                'mask_iou_reward': float(iou_reward) if iou_reward is not None else 0.0,
                'thk_ans_format_reward': thk_ans_format_reward[0] if thk_ans_format_reward is not None else 0.0,
                'sam_format_reward': sam_format_reward[0] if sam_format_reward is not None else 0.0,
            }
            label = input_example.get('class_label_id', None)
            if label is not None:
                result['class_label_id'] = label
            final_output.append(result)
        
        # Calculate gIoU (global IoU) and cIoU (cumulative IoU)
        if valid_samples > 0:
            # gIoU: average of per-image IoUs across all samples
            giou = total_iou / len(data)
            
            # cIoU: cumulative intersection over cumulative union
            ciou = total_intersection / total_union if total_union > 0 else 0.0
        else:
            giou = 0.0
            ciou = 0.0
        
        # Calculate average IoU (same as gIoU)
        mean_iou = giou
        print(f"\nValid samples: {valid_samples}/{len(data)}")
        print(f"gIoU (Global IoU - average per-image IoU): {giou:.4f}")
        print(f"cIoU (Cumulative IoU - cumulative intersection/cumulative union): {ciou:.4f}")

        save_dict = {
                'giou': float(giou),
                'ciou': float(ciou),
                'valid_samples': valid_samples,
                'total_samples': len(data),
                'total_intersection': float(total_intersection),
                'total_union': float(total_union),
                'results': final_output
            }
        
        # Save results to JSON file
        output_json_path = os.path.join(out_dir, "geo_results.json")
        with open(output_json_path, "w") as f:
            json.dump(save_dict, f, indent=2)
        
        print(f"Results saved to {output_json_path}")


if __name__ == "__main__":

    local_rank, world_size, rank = setup_distributed()
    device = f"cuda:{local_rank}"
    print(f"Process {rank} using {device}")

    main(local_rank, world_size, rank)