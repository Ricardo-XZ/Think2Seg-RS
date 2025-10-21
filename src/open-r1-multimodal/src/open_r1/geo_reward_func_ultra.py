import os
import re
from datetime import datetime
import json
# from open_r1.vlm_modules import *
import torch
import numpy as np
# from json_repair import repair_json
import json_repair

class GEO_Reward_Func_Ultra():
    def __init__(self, 
                 sam_model_size='tiny', 
                 sam_root=None, 
                 device='cuda'):
        super().__init__()
    
        self.init_sam2(sam_model_size=sam_model_size, sam_root=sam_root, device=device)

        self.pos_point_num = 2  # 2 positive points
        self.neg_point_num = 0  # no negative points

    def init_sam2(self,
                 sam_model_size='tiny', 
                 sam_root='../../sam2', 
                 device='cuda'):
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if sam_model_size == 'large':
            sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif sam_model_size == 'base_plus':
            sam2_checkpoint = "checkpoints/sam2.1_hiera_base_plus.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif sam_model_size == 'small':
            sam2_checkpoint = "checkpoints/sam2.1_hiera_small.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        elif sam_model_size == 'tiny':
            sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
        else:
            raise ValueError('Invalid SAM2 model size')
                
        self.sam_model_size = sam_model_size
        self.sam2_checkpoint = os.path.join(sam_root, sam2_checkpoint)
        self.sam_device = device

        print(f'Loading SAM2 {self.sam_model_size} model from {self.sam2_checkpoint}...')
        self.sam2_model = build_sam2(model_cfg, self.sam2_checkpoint, device=self.sam_device)
        # freeze SAM2 model parameters
        for param in self.sam2_model.parameters():
            param.requires_grad = False

        self.sam2_model.eval()
        
        self.predictor = SAM2ImagePredictor(self.sam2_model)
    
    def sam_iou_reward_func_ultra_test(self, completions, GT_mask, image, return_details = False, **kwargs):
        '''
        Test function

        Args:
            completions: list of model outputs (each element is a completion structure containing "content")
            GT_mask: list/array of ground-truth segmentation masks (boolean or binary arrays)
            image: list of PIL.Image.Image objects (mode == "RGB")
            return_details: bool, whether to return prediction details (default False)
            **kwargs: additional unused keyword arguments

        Returns:
            If return_details is False:
            list of reward values (float in [0.0, 1.0]) for each completion.
            If return_details is True:
            tuple (rewards, details) where details is a dict containing:
                - pred_masks: predicted binary masks
                - pred_ins_masks: predicted instance masks
                - mask_iou: IoU values
                - intersection: intersection pixel counts
                - union: union pixel counts
        '''
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = [0.0] * len(contents)  # default all rewards to 0.0
        no_target_flags = [np.all(mask == 0) for mask in GT_mask]  # precompute no-target flags

        # preparse all JSON
        all_parsed_data = []
        valid_indices = []

        # If returning details, prepare storage
        if return_details:
            all_details = {
                'pred_masks': [None] * len(contents),  # predicted masks
                'pred_ins_masks': [None] * len(contents),  # predicted instance masks
                'mask_iou': [None] * len(contents),  # predicted IoU
                'intersection': [None] * len(contents),  # each instance's intersection
                'union': [None] * len(contents),  # each instance's union
            }
        
        for i, (content, no_target) in enumerate(zip(contents, no_target_flags)):
            parsed_data, error_msg = self.parse_sam_json_test(content)
            
            # Special handling for no-target scenarios
            if no_target:
                if parsed_data is not None:
                    # Correct case: empty array with no error
                    rewards[i] = 1.0 if len(parsed_data) == 0 else 0.0
                    if return_details:
                        all_details['pred_masks'][i] = np.zeros_like(GT_mask[i], dtype=bool)  # generate all-zero mask
                        all_details['pred_ins_masks'][i] = np.zeros_like(GT_mask[i], dtype=bool)  # generate all-zero instance mask
                else:
                    # Parsing failed or non-empty array
                    rewards[i] = 0.0
                continue  # Skip subsequent processing
            if parsed_data is None:
                # JSON format error, directly give 0 points
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                        f.write(f"\n{'='*30} JSON parse error {'='*30}\n")
                        f.write(f"Time: {current_time}\n")
                        f.write(f"Instance #{i+1} | Error: {error_msg}\n")
            else:
                all_parsed_data.append(parsed_data)
                valid_indices.append(i)
        
        if not valid_indices:
            return rewards if not return_details else (rewards, all_details)
        
        try:
            with torch.inference_mode(), torch.autocast(self.sam_device, dtype=torch.bfloat16):
                for idx, orig_idx in enumerate(valid_indices):
                    # Get image and instance list
                    img_np = np.array(image[orig_idx])
                    instances = all_parsed_data[idx]
                    
                    self.predictor.set_image(img_np)
                    
                    combined_mask = None

                    combined_ins_mask = None  # to store each instance's mask
                    
                    for ins_id, instance in enumerate(instances):
                        bbox = np.array(instance["bbox_2d"]) if "bbox_2d" in instance else None
                        
                        pos_points = np.array(instance["positive_points"]) if "positive_points" in instance and len(instance["positive_points"]) > 0 else None
                        pos_labels = np.ones(pos_points.shape[0]) if pos_points is not None else None
                        
                        neg_points = np.array(instance["negative_points"]) if "negative_points" in instance and len(instance["negative_points"]) > 0 else None
                        neg_labels = np.zeros(neg_points.shape[0]) if neg_points is not None else None
                        
                        # Ensure at least one prompt is provided
                        if bbox is None and pos_points is None and neg_points is None:
                            continue
                            
                        point_coords = None
                        point_labels = None
                        
                        if pos_points is not None or neg_points is not None:
                            if pos_points is not None and neg_points is not None:
                                point_coords = np.vstack([pos_points, neg_points])
                                point_labels = np.append(pos_labels, neg_labels)
                            elif pos_points is not None:
                                point_coords = pos_points
                                point_labels = pos_labels
                            else:
                                point_coords = neg_points
                                point_labels = neg_labels
                        
                        masks, _, _ = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=bbox,
                            multimask_output=False
                        )
                        
                        if masks.shape[0] > 0:
                            instance_mask = masks[0]
                            
                            if combined_mask is None:
                                combined_mask = instance_mask
                            else:
                                combined_mask = np.logical_or(combined_mask, instance_mask)

                            if combined_ins_mask is None:
                                combined_ins_mask = np.zeros_like(instance_mask, dtype=np.uint8)
                                combined_ins_mask[instance_mask.astype(bool)] = ins_id + 1  # use instance ID
                            else:
                                combined_ins_mask[instance_mask.astype(bool)] = ins_id + 1  # use instance ID

                    if combined_mask is not None:
                        gt_mask = GT_mask[orig_idx]
                        intersection = np.logical_and(combined_mask, gt_mask).sum()
                        union = np.logical_or(combined_mask, gt_mask).sum()
                        iou = intersection / union if union > 0 else 0.0
                        
                        # update reward value
                        rewards[orig_idx] = float(iou)
                        
                        if return_details:
                            all_details['pred_masks'][orig_idx] = combined_mask
                            all_details['pred_ins_masks'][orig_idx] = combined_ins_mask
                            all_details['mask_iou'][orig_idx] = iou
                            all_details['intersection'][orig_idx] = intersection
                            all_details['union'][orig_idx] = union
                        
                        # Debug output
                        if os.getenv("DEBUG_MODE") == "true":
                            log_path = os.getenv("LOG_PATH")
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                                f.write(f"\n{'='*30} Predict results {'='*30}\n")
                                f.write(f"Time: {current_time}\n")
                                f.write(f"Instance #{orig_idx+1} | IoU: {iou:.4f} | Intersection: {intersection} | Union: {union}\n")
                                f.write(f"Number of Instances: {len(instances)}\n")
                    else:
                        if os.getenv("DEBUG_MODE") == "true":
                            log_path = os.getenv("LOG_PATH")
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                                f.write(f"Time: {current_time}\n")
                                f.write(f"\n{'='*30} Prediction Failed {'='*30}\n")
                                f.write(f"Instance #{orig_idx+1} | Reason: No valid mask generated\n")
                                f.write(f"Number of Instances: {len(instances)}\n")

        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                import traceback
                with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                    f.write(f"\n{'='*30} Error Occurred {'='*30}\n")
                    f.write(f"Time: {current_time}\n")
                    f.write(f"Error: {type(e).__name__} - {str(e)}\n")
                    f.write(f"Stack Trace: {traceback.format_exc()}\n")

        return rewards if not return_details else (rewards, all_details)
    

    def sam_iou_reward_func_ultra_test_ori_res(self, completions, GT_mask, image, resize_size, return_details = False, **kwargs):
        '''
        test function for origional resolution
        '''

        def scale_coordinates(x_factor, y_factor, bbox=None, pos_points=None, neg_points=None):
            """
            将坐标从resize后的尺寸缩放到原始分辨率
            
            Args:
                x_factor (float): x scale factor (original width / resize_size)
                y_factor (float): y scale factor (original height / resize_size)
                bbox (list): bounding box coordinates [x1, y1, x2, y2]
                pos_points (list): positive point coordinates [[x1, y1], [x2, y2], ...]
                neg_points (list): negative point coordinates [[x1, y1], [x2, y2], ...]

            Returns:
                dict: dictionary with scaled 'bbox', 'pos_points', 'neg_points'
            """
            result = {}
            
            if bbox is not None:
                scaled_bbox = [
                    int(bbox[0] * x_factor + 0.5),  # x1
                    int(bbox[1] * y_factor + 0.5),  # y1  
                    int(bbox[2] * x_factor + 0.5),  # x2
                    int(bbox[3] * y_factor + 0.5)   # y2
                ]
                result['bbox'] = scaled_bbox
            else:
                result['bbox'] = None
                
            if pos_points is not None and len(pos_points) > 0:
                scaled_pos_points = []
                for point in pos_points:
                    scaled_point = [
                        int(point[0] * x_factor + 0.5),  # x
                        int(point[1] * y_factor + 0.5)   # y
                    ]
                    scaled_pos_points.append(scaled_point)
                result['pos_points'] = scaled_pos_points
            else:
                result['pos_points'] = None
                
            if neg_points is not None and len(neg_points) > 0:
                scaled_neg_points = []
                for point in neg_points:
                    scaled_point = [
                        int(point[0] * x_factor + 0.5),  # x
                        int(point[1] * y_factor + 0.5)   # y
                    ]
                    scaled_neg_points.append(scaled_point)
                result['neg_points'] = scaled_neg_points
            else:
                result['neg_points'] = None
                
            return result
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = [0.0] * len(contents)  
        no_target_flags = [np.all(mask == 0) for mask in GT_mask]  

        all_parsed_data = []
        valid_indices = []

        if return_details:
            all_details = {
                'pred_masks': [None] * len(contents),  
                'pred_ins_masks': [None] * len(contents),  
                'mask_iou': [None] * len(contents),  
                'intersection': [None] * len(contents),  
                'union': [None] * len(contents), 
            }
        
        for i, (content, no_target) in enumerate(zip(contents, no_target_flags)):
            parsed_data, error_msg = self.parse_sam_json_test(content)
            
            if no_target:
                if parsed_data is not None:
                    rewards[i] = 1.0 if len(parsed_data) == 0 else 0.0
                    if return_details:
                        all_details['pred_masks'][i] = np.zeros_like(GT_mask[i], dtype=bool) 
                        all_details['pred_ins_masks'][i] = np.zeros_like(GT_mask[i], dtype=bool)  
                else:
                    rewards[i] = 0.0
                continue           
            if parsed_data is None:
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                        f.write(f"\n{'='*30} JSON parse error {'='*30}\n")
                        f.write(f"Time: {current_time}\n")
                        f.write(f"Instance #{i+1} | Error: {error_msg}\n")
            else:
                all_parsed_data.append(parsed_data)
                valid_indices.append(i)
        
        if not valid_indices:
            return rewards if not return_details else (rewards, all_details)
        
        try:
            with torch.inference_mode(), torch.autocast(self.sam_device, dtype=torch.bfloat16):
                for idx, orig_idx in enumerate(valid_indices):
                    img_np = np.array(image[orig_idx])
                    instances = all_parsed_data[idx]

                    original_height, original_width = img_np.shape[:2]
                    x_factor = original_width / resize_size
                    y_factor = original_height / resize_size
                    
                    self.predictor.set_image(img_np)
                    
                    combined_mask = None

                    combined_ins_mask = None 
                    
                    for ins_id, instance in enumerate(instances):
                        bbox = np.array(instance["bbox_2d"]) if "bbox_2d" in instance else None
                        
                        pos_points = np.array(instance["positive_points"]) if "positive_points" in instance and len(instance["positive_points"]) > 0 else None
                        pos_labels = np.ones(pos_points.shape[0]) if pos_points is not None else None
                        
                        neg_points = np.array(instance["negative_points"]) if "negative_points" in instance and len(instance["negative_points"]) > 0 else None
                        neg_labels = np.zeros(neg_points.shape[0]) if neg_points is not None else None
                        
                        scaled_coords = scale_coordinates(
                            x_factor, y_factor,
                            bbox=bbox,
                            pos_points=pos_points,
                            neg_points=neg_points
                        )
                        bbox = np.array(scaled_coords['bbox']) if scaled_coords['bbox'] is not None else None
                        pos_points = np.array(scaled_coords['pos_points']) if scaled_coords['pos_points'] is not None else None
                        neg_points = np.array(scaled_coords['neg_points']) if scaled_coords['neg_points'] is not None else None

                        if bbox is None and pos_points is None and neg_points is None:
                            continue
                            
                        point_coords = None
                        point_labels = None
                        
                        if pos_points is not None or neg_points is not None:
                            if pos_points is not None and neg_points is not None:
                                point_coords = np.vstack([pos_points, neg_points])
                                point_labels = np.append(pos_labels, neg_labels)
                            elif pos_points is not None:
                                point_coords = pos_points
                                point_labels = pos_labels
                            else:
                                point_coords = neg_points
                                point_labels = neg_labels
                        
                        masks, _, _ = self.predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=bbox,
                            multimask_output=False
                        )
                        
                        if masks.shape[0] > 0:
                            instance_mask = masks[0]
                            
                            if combined_mask is None:
                                combined_mask = instance_mask
                            else:
                                combined_mask = np.logical_or(combined_mask, instance_mask)

                            if combined_ins_mask is None:
                                combined_ins_mask = np.zeros_like(instance_mask, dtype=np.uint8)
                                combined_ins_mask[instance_mask.astype(bool)] = ins_id + 1  
                            else:
                                combined_ins_mask[instance_mask.astype(bool)] = ins_id + 1  

                    if combined_mask is not None:
                        gt_mask = GT_mask[orig_idx]
                        intersection = np.logical_and(combined_mask, gt_mask).sum()
                        union = np.logical_or(combined_mask, gt_mask).sum()
                        iou = intersection / union if union > 0 else 0.0
                        
                        rewards[orig_idx] = float(iou)
                        
                        if return_details:
                            all_details['pred_masks'][orig_idx] = combined_mask
                            all_details['pred_ins_masks'][orig_idx] = combined_ins_mask
                            all_details['mask_iou'][orig_idx] = iou
                            all_details['intersection'][orig_idx] = intersection
                            all_details['union'][orig_idx] = union
                        
                        if os.getenv("DEBUG_MODE") == "true":
                            log_path = os.getenv("LOG_PATH")
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                                f.write(f"\n{'='*30} Predict results {'='*30}\n")
                                f.write(f"Time: {current_time}\n")
                                f.write(f"Instance #{orig_idx+1} | IoU: {iou:.4f} | Intersection: {intersection} | Union: {union}\n")
                                f.write(f"Number of instances: {len(instances)}\n")
                    else:
                        if os.getenv("DEBUG_MODE") == "true":
                            log_path = os.getenv("LOG_PATH")
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                            with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                                f.write(f"Time: {current_time}\n")
                                f.write(f"\n{'='*30} Predict failed {'='*30}\n")
                                f.write(f"Instance #{orig_idx+1} | Reason: No valid mask generated\n")
                                f.write(f"Number of instances: {len(instances)}\n")

        except Exception as e:
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                import traceback
                with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                    f.write(f"\n{'='*30} Handle error {'='*30}\n")
                    f.write(f"Time: {current_time}\n")
                    f.write(f"Error: {type(e).__name__} - {str(e)}\n")
                    f.write(f"Stack: {traceback.format_exc()}\n") 

        return rewards if not return_details else (rewards, all_details)

    def sam_reward_func_ultra(self, completions, GT_mask, image, return_details = False, **kwargs):
        '''
        Args:
            completions: list of model outputs (each element is a completion structure containing "content")
            GT_mask: list/array of ground-truth segmentation masks (boolean or binary arrays)
            image: list of PIL.Image.Image objects (mode == "RGB")
            return_details: bool, whether to return prediction details (default False)
            **kwargs: additional unused keyword arguments

        Returns:
            If return_details is False:
            list of reward values (float in [0.0, 1.0]) for each completion.
            If return_details is True:
            tuple (rewards, details) where details is a dict containing:
                - pred_masks: predicted binary masks
                - pred_ins_masks: predicted instance masks
                - mask_iou: IoU values
                - intersection: intersection pixel counts
                - union: union pixel counts
        '''
        
        contents = [completion[0]["content"] for completion in completions]
        rewards = {
                'final_reward': [0.0] * len(contents),
                'mask_iou_reward': [0.0] * len(contents),  
        }
        no_target_flags = [np.all(mask == 0) for mask in GT_mask] 

        all_parsed_data = []
        valid_indices = []

        # if returning details, prepare storage
        if return_details:
            all_details = {
                'pred_masks': [None] * len(contents),  
                'mask_iou': [None] * len(contents),  
                'intersection': [None] * len(contents),  
                'union': [None] * len(contents),  
            }
        
        for i, (content, no_target) in enumerate(zip(contents, no_target_flags)):
            parsed_data, error_msg = self.parse_sam_json(content)
            
            # no-target
            if no_target:
                if parsed_data is not None:
                    if len(parsed_data) == 0:
                        rewards['mask_iou_reward'][i] = 1.0
                    if return_details:
                        all_details['pred_masks'][i] = np.zeros_like(GT_mask[i], dtype=bool)  # 生成全零掩码
                continue         
            if parsed_data is None:
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                        f.write(f"\n{'='*30} JSON parse error {'='*30}\n")
                        f.write(f"Time: {current_time}\n")
                        f.write(f"Instance #{i+1} | Error: {error_msg}\n")
            else:
                if parsed_data: 
                    all_parsed_data.append(parsed_data)
                    valid_indices.append(i)
        
        if not valid_indices:
            rewards = self.calculate_final_reward(rewards)
            return rewards if not return_details else (rewards, all_details)
        
        try:
            with torch.inference_mode(), torch.autocast(self.sam_device, dtype=torch.bfloat16):
                for idx, orig_idx in enumerate(valid_indices):
                    img_np = np.array(image[orig_idx])
                    gt_mask = GT_mask[orig_idx]
                    instances = all_parsed_data[idx]

                    sam_dict = self.sam_forward(img_np, gt_mask, instances)

                    # update rewards
                    rewards['mask_iou_reward'][orig_idx] = float(sam_dict['iou'])

                    # if return details, save the details
                    if return_details:
                        all_details['pred_masks'][orig_idx] = sam_dict['combined_mask']
                        all_details['mask_iou'][orig_idx] = sam_dict['iou']
                        all_details['intersection'][orig_idx] = sam_dict['intersection']
                        all_details['union'][orig_idx] = sam_dict['union']
                        
                    # Debug output
                    if os.getenv("DEBUG_MODE") == "true":
                        log_path = os.getenv("LOG_PATH")
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                        with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                            f.write(f"\n{'='*30} Predict results {'='*30}\n")
                            f.write(f"Time: {current_time}\n")
                            f.write(f"Instance #{orig_idx+1} | IoU: {sam_dict['iou']:.4f} | Intersection: {sam_dict['intersection']} | Union: {sam_dict['union']}\n")
                            f.write(f"Number of instances: {len(instances)}\n")
                            f.write(f"Instances: {instances}")
            rewards = self.calculate_final_reward(rewards)
        except Exception as e:
            # If an error occurs during processing, keep default 0 scores
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                import traceback
                with open(log_path.replace(".txt", "_sam_iou_reward.txt"), "a", encoding='utf-8') as f:
                    f.write(f"\n{'='*30} error occurred {'='*30}\n")
                    f.write(f"Time: {current_time}\n")
                    f.write(f"Error: {type(e).__name__} - {str(e)}\n")
                    f.write(f"Stack trace: {traceback.format_exc()}\n")
        
        return rewards if not return_details else (rewards, all_details)

    def sam_forward(self, img_np, gt_mask, instances):

        self.predictor.set_image(img_np)
        
        combined_mask = None
        point_reward_list = []
        for ins_idx, instance in enumerate(instances):

            # BOX
            bbox = np.array(instance["bbox_2d"]) if "bbox_2d" in instance else None
            
            # POINT
            pos_points = np.array(instance["positive_points"]) if "positive_points" in instance and len(instance["positive_points"]) > 0 else None
            pos_labels = np.ones(pos_points.shape[0]) if pos_points is not None else None
            
            neg_points = np.array(instance["negative_points"]) if "negative_points" in instance and len(instance["negative_points"]) > 0 else None
            neg_labels = np.zeros(neg_points.shape[0]) if neg_points is not None else None

            # At least one prompt
            if bbox is None and pos_points is None and neg_points is None:
                continue
                
            point_coords = None
            point_labels = None
            
            if pos_points is not None or neg_points is not None:
                if pos_points is not None and neg_points is not None:
                    point_coords = np.vstack([pos_points, neg_points])
                    point_labels = np.append(pos_labels, neg_labels)
                elif pos_points is not None:
                    point_coords = pos_points
                    point_labels = pos_labels
                else:
                    point_coords = neg_points
                    point_labels = neg_labels
            
            masks, _, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox,
                multimask_output=False
            )
            
            if masks.shape[0] > 0:
                instance_mask = masks[0]
                
                # OR
                if combined_mask is None:
                    combined_mask = instance_mask
                else:
                    combined_mask = np.logical_or(combined_mask, instance_mask)
                
                # ins_point_reward = self.get_point_reward(instance_mask, ins_gt_mask, pos_points)
                # point_reward_list.append(ins_point_reward)

        point_reward = np.mean(point_reward_list) if point_reward_list else 0.0
        iou, intersection, union = self.get_mask_iou(combined_mask, gt_mask) if combined_mask is not None else (0.0, 0, 0)

        # return iou, intersection, union, combined_mask
        return {
            'iou': iou,
            'intersection': intersection,
            'union': union,
            'combined_mask': combined_mask,
        }
        
    def calculate_final_reward(self, rewards, weights=None):
        """
        if you want to add any other reward type, add it to the weights dict and rewards dict
        
        Args:
            rewards (dict): dict of different reward types
            weights (dict): dict of weights for each reward type
            
        Returns:
            rewards: dict with final_reward calculated and added
        """

        if weights is None:
            weights = {
                'mask_iou_reward': 2,     
            }
        
        total_weight = sum(weights.values())

        reward_arrays = []
        weight_values = []
        
        for reward_type, weight in weights.items():
            if reward_type in rewards:
                reward_arrays.append(np.array(rewards[reward_type]))
                weight_values.append(weight)
            else:
                print(f"Warning: Reward type '{reward_type}' not found in rewards dict")
        
        if reward_arrays:
            stacked_rewards = np.stack(reward_arrays, axis=0)  # shape: (num_reward_types, num_samples)
            weight_vector = np.array(weight_values).reshape(-1, 1)  # shape: (num_reward_types, 1)
            
            final_rewards = np.sum(stacked_rewards * weight_vector, axis=0)
            
            final_rewards = np.clip(final_rewards, 0.0, np.sum(weight_values))    # total_weight
            
            rewards['final_reward'] = final_rewards.tolist()

        return rewards

    def compute_bbox_iou(self, bboxes1, bboxes2):
        """
        Calculate IOU matrix between two sets of bounding boxes
        bboxes1: shape (N, 4) prediction boxes
        bboxes2: shape (M, 4) ground truth boxes
        Returns: shape (N, M) IOU matrix
        """
        # Expand dimensions to support broadcasting
        bboxes1 = np.array(bboxes1)[:, None, :]  # (N, 1, 4)
        bboxes2 = np.array(bboxes2)[None, :, :]  # (1, M, 4)
        
        # Calculate intersection area
        x1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
        y1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
        x2 = np.minimum(bboxes1[..., 2], bboxes2[..., 2])
        y2 = np.minimum(bboxes1[..., 3], bboxes2[..., 3])
        
        # Calculate intersection area
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Calculate the areas of the two sets of bboxes
        area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
        area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])
        
        # Calculate union area
        union = area1 + area2 - intersection
        
        # Avoid division by zero
        iou = np.where(union > 0, intersection / union, 0)
        
        return iou

    def get_mask_iou(self, combined_mask, gt_mask):
        intersection = np.logical_and(combined_mask, gt_mask).sum()
        union = np.logical_or(combined_mask, gt_mask).sum()
        iou = intersection / union if union > 0 else 0.0

        return iou, intersection, union

    def parse_sam_json(self, json_string):
        try:
            if "<answer>" in json_string and "</answer>" in json_string:
                pattern = r'<answer>(.*?)</answer>'
                match = re.search(pattern, json_string, re.DOTALL)
                if match:
                    json_string = match.group(1).strip()

            if "```json" in json_string:
                pattern = r'```json\s*(.*?)\s*```'
                match = re.search(pattern, json_string, re.DOTALL)
                if match:
                    json_string = match.group(1).strip()
                else:
                    return None, "Cannot find complete ```json ``` format"
            else:
                return None, "Output must contain ```json ``` format"

            if "[" in json_string and "]" in json_string:
                start = json_string.find("[")
                end = json_string.rfind("]") + 1
                json_string = json_string[start:end]
                
            try:
                parsed_data = json.loads(json_string)
            except json.JSONDecodeError:
                parsed_data = json_repair.loads(json_string)
            
            if not isinstance(parsed_data, list):
                return None, "Output must be a JSON array"

            for i, instance in enumerate(parsed_data):
                if not isinstance(instance, dict):
                    return None, f"Instance #{i+1} must be a JSON object"

                # at least one of bbox_2d and positive_points
                has_bbox = "bbox_2d" in instance
                has_positive_points = "positive_points" in instance
                
                if not (has_bbox or has_positive_points):
                    return None, f"Instance #{i+1} must contain at least one of bbox_2d or positive_points"
                if has_bbox and not self.has_valid_bbox(instance):
                    return None, f"Instance #{i+1} has invalid bbox_2d format"
                if not self.has_valid_points(instance, "positive_points"):
                    return None, f"Instance #{i+1} has invalid positive_points format"
                if not self.has_valid_points(instance, "negative_points"):
                    return None, f"Instance #{i+1} has invalid negative_points format"

            parsed_data = self.remove_repeated_boxes_nms(parsed_data)
            
            return parsed_data, None
            
        except json.JSONDecodeError as e:
            return None, f"JSON parsing error: {str(e)}"
        except Exception as e:
            return None, f"Unknown error: {str(e)}"

    def parse_sam_json_test(self, json_string):        
        try:
            if "<answer>" in json_string and "</answer>" in json_string:
                pattern = r'<answer>(.*?)</answer>'
                match = re.search(pattern, json_string, re.DOTALL)
                if match:
                    json_string = match.group(1).strip()
            
            if "[" in json_string and "]" in json_string:
                start = json_string.find("[")
                end = json_string.rfind("]") + 1
                json_string = json_string[start:end]
                
            # parse JSON
            try:
                parsed_data = json.loads(json_string)
            except json.JSONDecodeError:
                parsed_data = json_repair.loads(json_string)
            
            if not isinstance(parsed_data, list):
                return None, "Output must be a JSON array"
                
            for i, instance in enumerate(parsed_data):
                if not isinstance(instance, dict):
                    return None, f"Instance #{i} must be a JSON object"

                # at least one of bbox_2d and positive_points
                has_bbox = "bbox_2d" in instance
                has_positive_points = "positive_points" in instance
                
                if not (has_bbox or has_positive_points):
                    return None, f"Instance #{i} must contain at least one of bbox_2d or positive_points"
                if has_bbox and not self.has_valid_bbox(instance):
                    return None, f"Instance #{i} has invalid bbox_2d format"
                if has_positive_points and not self.has_valid_points(instance, "positive_points"):
                    return None, f"Instance #{i} has invalid positive_points format"
                if "negative_points" in instance and not self.has_valid_points(instance, "negative_points"):
                    return None, f"Instance #{i} has invalid negative_points format"

            parsed_data = self.remove_repeated_boxes_nms(parsed_data)

            return parsed_data, None
            
        except json.JSONDecodeError as e:
            return None, f"JSON parsing error: {str(e)}"
        except Exception as e:
            return None, f"Unknown error: {str(e)}"

    def parse_think_content(self, json_string):
        try:
            if "<think>" in json_string and "</think>" in json_string:
                pattern = r'<think>(.*?)</think>'
                match = re.search(pattern, json_string, re.DOTALL)
                if match:
                    json_string = match.group(1).strip()
                    return json_string
        except:
            return None
    
    def remove_repeated_boxes_nms(self, parsed_data, iou_threshold=0.5):
        """
        Remove repeated bounding boxes using non-maximum suppression (NMS).
        """
        if not parsed_data or len(parsed_data) <= 1:
            return parsed_data
            
        valid_instances = []
        for i, instance in enumerate(parsed_data):
            if self.has_valid_bbox(instance):
                bbox = instance['bbox_2d']
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                valid_instances.append({
                    'index': i,
                    'bbox': bbox,
                    'area': area,
                    'instance': instance
                })
        
        if len(valid_instances) <= 1:
            return parsed_data
        
        # sort by area descending
        valid_instances.sort(key=lambda x: x['area'], reverse=True)
        
        bboxes = [item['bbox'] for item in valid_instances]
        iou_matrix = self.compute_bbox_iou(bboxes, bboxes)
        
        to_remove = set()
        n = len(valid_instances)
        
        for i in range(n):
            if i in to_remove:
                continue
            for j in range(i + 1, n):
                if j in to_remove:
                    continue
                    
                if iou_matrix[i, j] > iou_threshold:
                    to_remove.add(j)
        
        removed_indices = {valid_instances[i]['index'] for i in to_remove}
        filtered_data = [instance for i, instance in enumerate(parsed_data) 
                        if i not in removed_indices]
        
        return filtered_data

    def has_valid_bbox(self, instance):
        try:
            if 'bbox_2d' in instance and isinstance(instance['bbox_2d'], list) and len(instance['bbox_2d']) == 4:
                return True
            else:
                return False
        except:
            return False
    
    def has_valid_points(self, instance, point_type):
        try:
            points = instance.get(point_type, None)
            if point_type == "positive_points":
                point_num = self.pos_point_num
            elif point_type == "negative_points":
                point_num = self.neg_point_num
            else:
                return False
            if points is None and point_num == 0:
                return True
            if not isinstance(points, list) or len(points) == 0:
                return False
            if len(points) != point_num:
                return False
            for point in points:
                if not isinstance(point, (list, tuple)):
                    return False
                if len(point) != 2:
                    return False
                try:
                    float(point[0])
                    float(point[1])
                except (TypeError, ValueError):
                    return False
            
            return True
        except:
            return False

    def sam_format_reward(self, completions, **kwargs):
        """
        Reward function that checks if the completion is valid SAM JSON format.
        
        Args:
            completions: the list of model completions
            
        Returns:
            list: a list of reward values for each output (1.0 indicates correct format, 0.0 indicates incorrect format)
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        
        for content in contents:
            parsed_data, error_msg = self.parse_sam_json(content)

            if parsed_data is not None:
                reward = 1.0
            else:
                reward = 0.0 
                
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                with open(log_path.replace(".txt", "_json_format_reward.txt"), "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} SAM Format reward: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    if error_msg:
                        f.write(f"Error: {error_msg}\n")
                    f.write(f"-----------------------------------\n\n")
                    
            rewards.append(reward)
        
        return rewards

    def thk_ans_format_reward(self, completions, **kwargs):
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
