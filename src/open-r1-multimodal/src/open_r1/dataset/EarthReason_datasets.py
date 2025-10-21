import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import copy

############################
# usual version, bbox + 2 pos_points
SYSTEM_PROMPT = "You are a remote sensing analysis assistant. Your task is to generate spatial prompts for the Segment Anything Model (SAM) based on a user's request."
USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide:
1.  `bbox_2d`: A tight bounding box.
2.  `positive_points`: Exactly two points, placed inside the target.
Output your thinking process in <think> </think> tags.
Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
i.e. <think> thinking process here </think> 
<answer>```json[{{"bbox_2d": [310,360,567,586], "positive_points": [[434, 474], [450, 460]]}}, {{"bbox_2d": [10,200,100,320], "positive_points": [[50, 250], [90, 300]]}}]```</answer>"""

###########################
# # with negative points
# SYSTEM_PROMPT = "You are a remote sensing analysis assistant. Your task is to generate spatial prompts for the Segment Anything Model (SAM) based on a user's request."
# USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide:
# 1.  `bbox_2d`: A tight bounding box.
# 2.  `positive_points`: Exactly two points, placed inside the target.
# 3.  `negative_points`: Exactly two points, placed outside the target's boundary to separate it from the background.
# Output your thinking process in <think> </think> tags.
# Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
# i.e. <think> thinking process here </think> 
# <answer>```json[{{"bbox_2d": [310,360,567,586], "positive_points": [[434, 474], [450, 460]], "negative_points":[[320, 400], [500, 480]]}}, {{"bbox_2d": [10,200,100,320], "positive_points": [[50, 250], [90, 300]], "negative_points":[[20, 240], [30, 250]]}}]```</answer>"""

# ############################
# # only bbox
# SYSTEM_PROMPT = "You are a remote sensing analysis assistant. Your task is to generate spatial prompts for the Segment Anything Model (SAM) based on a user's request."
# USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide a tight bounding box.
# Output your thinking process in <think> </think> tags.
# Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
# i.e. <think> thinking process here </think> 
# <answer>```json[{{"bbox_2d": [310,360,567,586]}}, {{"bbox_2d": [10,200,100,320]}}]```</answer>"""

# ############################
# # only points
# SYSTEM_PROMPT = "You are a remote sensing analysis assistant. Your task is to generate spatial prompts for the Segment Anything Model (SAM) based on a user's request."
# USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide two positive points, placed inside the target.
# Output your thinking process in <think> </think> tags.
# Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
# i.e. <think> thinking process here </think> 
# <answer>```json[{{"positive_points": [[434, 474], [450, 460]]}}, {{"positive_points": [[50, 250], [90, 300]]}}]```</answer>"""

# ############################
# # bbox + 4 pos_points
# SYSTEM_PROMPT = "You are a remote sensing analysis assistant. Your task is to generate spatial prompts for the Segment Anything Model (SAM) based on a user's request."
# USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide:
# 1.  `bbox_2d`: A tight bounding box.
# 2.  `positive_points`: Exactly four points, placed inside the target.
# Output your thinking process in <think> </think> tags.
# Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
# i.e. <think> thinking process here </think> 
# <answer>```json[{{"bbox_2d": [310,360,567,586], "positive_points": [[434, 474], [450, 460], [500, 480], [520, 500]]}}, {{"bbox_2d": [10,200,100,320], "positive_points": [[50, 250], [90, 300], [70, 280], [30, 240]]}}]```</answer>"""


class EarthReasonDataset(Dataset):
    def __init__(self, 
                 data_dir='/mnt/HDD1/Datasets/EarthReason/EarthReason',
                 split='train',
                 transform=None,
                 mask_transform=None,
                 max_retry=5,
                 resize_size=840,
                 vis_dir=None):
        """
        This version updates the label_instance and GTs.
        Args:
            data_dir (str): Root directory of the EarthReason dataset
            split (str or list): Data split ('train', 'val', 'test') or a list of splits ['train', 'val']
            transform (callable, optional): Transform to be applied to images
            mask_transform (callable, optional): Transform to be applied to masks
            max_retry (int): Maximum retry count for loading images
            vis_dir (str, optional): Directory to save visualizations
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.max_retry = max_retry
        self.resize_size = resize_size
        self.vis_dir = vis_dir
        
        # Convert string split to a list for uniform handling
        if isinstance(split, str):
            split = [split]
        self.splits = split
        
        # Load dataset
        self.metadata = self._load_metadata()
        
        # Initialize visualization directory if specified
        if vis_dir and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)


    def _load_metadata(self):
        """Load and validate metadata from image, label, and QA directories for all splits"""
        valid_metadata = []
        
        # Process each split
        for split in self.splits:
            image_dir = os.path.join(self.data_dir, split, 'images')
            label_dir = os.path.join(self.data_dir, split, 'labels')
            qa_dir = os.path.join(self.data_dir, split, 'QAs')
            
            if not os.path.exists(image_dir):
                print(f"Warning: Image directory {image_dir} not found for split '{split}'")
                raise FileNotFoundError(f"Image directory {image_dir} not found")
            
            # Get list of image files
            image_files = os.listdir(image_dir)
            print(f"Found {len(image_files)} images in {image_dir}")
            
            # Validate and create metadata
            for img_file in image_files:
                img_id = img_file.split('.')[0]
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, f"{img_id}.png")
                qa_path = os.path.join(qa_dir, f"{img_id}.json")
                
                # Check if corresponding files exist
                if os.path.exists(img_path) and os.path.exists(label_path) and os.path.exists(qa_path):
                    # Load QA data to get questions
                    try:
                        with open(qa_path, 'r') as f:
                            qa_data = json.load(f)
                        
                        # Extract questions for prompts
                        if 'questions' in qa_data and qa_data['questions']:
                            item = {
                                'data_id': img_id,
                                'image_path': img_path,
                                'mask_path': label_path,
                                'qa_path': qa_path,
                                # 'prompts': qa_data['questions']
                                'prompts': random.choice(qa_data['questions'])
                            }
                            valid_metadata.append(item)
                        else:
                            print(f"Warning: No questions found in {qa_path}")
                    except Exception as e:
                        print(f"Error loading QA data from {qa_path}: {str(e)}")
                else:
                    missing = []
                    if not os.path.exists(img_path): missing.append("image")
                    if not os.path.exists(label_path): missing.append("label")
                    if not os.path.exists(qa_path): missing.append("QA data")
                    
                    print(f"Warning: Missing {', '.join(missing)} for {img_id} in split '{split}'")
        
        splits_str = ', '.join(self.splits)
        print(f"Loaded {len(valid_metadata)} valid instances from {splits_str} split{'s' if len(self.splits) > 1 else ''}")
        return valid_metadata

    def _build_conversation(self, question):
        """Construct conversation format for multimodal input"""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT.format(Question=question)}
                ]
            }
        ]

    def scale_box_coordinates(self, bbox_2d, x_factor, y_factor):
        """
        对边界框坐标进行缩放
        
        bbox_2d: [x1, y1, x2, y2]
        """
        # 缩放边界框坐标
        scaled_bbox = [
            int(bbox_2d[0] * x_factor + 0.5),  # x1
            int(bbox_2d[1] * y_factor + 0.5),  # y1
            int(bbox_2d[2] * x_factor + 0.5),  # x2
            int(bbox_2d[3] * y_factor + 0.5)   # y2
        ]

        
        return scaled_bbox

    def scale_point_coordinates(self, point_2d, x_factor, y_factor):
        """
        对中心点坐标进行缩放
        point_2d: [x, y]
        """

        # 缩放中心点坐标
        scaled_point = [
            int(point_2d[0] * x_factor + 0.5),  # x
            int(point_2d[1] * y_factor + 0.5)   # y
        ]
        
        return scaled_point

    def smart_resize(self,
        height: int, width: int, factor: int = 28, min_pixels: int = 4*28*28, max_pixels: int = 37*37*28*28, max_ratio: int = 100
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor
        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor
        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def _visualize(self, image, mask_array, prompt, save_path):
        """Visualize and save image with mask overlay, instance masks, bboxes, and points."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plt.suptitle(prompt, fontsize=10, wrap=True)

        # Subplot 1: Original image with segmentation mask overlay
        axes.ravel()[0].imshow(image)
        mask_array = np.ma.masked_where(mask_array == 0, mask_array)
        axes.ravel()[0].imshow(mask_array, cmap='jet', alpha=0.6)
        # Add contours to the segmentation mask
        contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            axes.ravel()[0].plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=1, alpha=0.5)  # Draw contours in red

        axes.ravel()[0].set_title('Image + Segmentation Mask', fontsize=8)
        axes.ravel()[0].axis('off')

        # # Subplot 2: Instance masks, bounding boxes, and center points
        # axes.ravel()[1].imshow(ins_mask_array, cmap='nipy_spectral', alpha=1)
        # axes.ravel()[1].set_title('Instance Masks, BBoxes, Points', fontsize=10)
        # axes.ravel()[1].axis('off')

        # unique_instance_ids = np.unique(ins_mask_array)
        # unique_instance_ids = unique_instance_ids[(unique_instance_ids != 0)] # Exclude background

        # random colors for each instance
        # colors = {}
        # for instance_id in unique_instance_ids:
        #     colors.setdefault(instance_id, np.random.rand(3))

        # Predefined colors for instances
        predefined_colors = [
            [1.0, 0.0, 0.0],    # Red
            [0.0, 1.0, 0.0],    # Green
            [0.0, 0.0, 1.0],    # Blue
            [1.0, 1.0, 0.0],    # Yellow
            [1.0, 0.0, 1.0],    # Magenta
            [0.0, 1.0, 1.0],    # Cyan
            [1.0, 0.5, 0.0],    # Orange
            [0.5, 0.0, 1.0],    # Purple
            [0.0, 0.5, 0.0],    # Dark Green
            [0.5, 0.5, 0.5],    # Gray
        ]
        
        colors = {}
        for i, instance_id in enumerate(unique_instance_ids):
            # Use predefined colors, cycle through them if there are more instances
            color_idx = i % len(predefined_colors)
            colors[instance_id] = predefined_colors[color_idx]

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        plt.savefig(save_path)
        plt.close(fig)

    def __len__(self):
        """Return the total number of samples"""
        return len(self.metadata)

    def __getitem__(self, index):
        """Get a single dataset item"""
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                item = self.metadata[index]
                
                # Load image
                image = Image.open(item['image_path']).convert('RGB')
                
                # Load gt mask
                gt_mask = Image.open(item['mask_path'])

                # Select a random prompt/question
                # prompt = random.choice(item['prompts'])
                prompt = item['prompts']
                
                ## resize image to adapt to Qwen2.5-VL default patch size
                width, height = image.size
                resized_height, resized_width = self.resize_size, self.resize_size
                # resized_height, resized_width = self.smart_resize(
                #     height, 
                #     width,
                #     factor=28,  # Qwen2.5-VL default patch size
                #     min_pixels=4*28*28,  
                #     max_pixels=37*37*28*28,  # 1024*1024->1036*1036
                #     )
                x_factor = resized_width / width
                y_factor = resized_height / height
                image = image.resize((resized_width, resized_height), Image.LANCZOS)  
                gt_mask = gt_mask.resize((resized_width, resized_height), Image.NEAREST)
                mask_array = np.array(gt_mask) > 0
                
                # Create visualization if specified
                if self.vis_dir:
                    image_array = np.array(image)
                    vis_path = os.path.join(self.vis_dir, f"{item['data_id']}.png")
                    self._visualize(image_array, mask_array, prompt, vis_path)

                # Prepare the dataset item
                return {
                    'data_idx': item['data_id'],
                    'image_path': item['image_path'],
                    'image': image,
                    'GT_mask_path': item['mask_path'],
                    'GT_mask': mask_array,
                    'problem': prompt,
                    'prompt': self._build_conversation(prompt)
                }
            
            except Exception as e:
                print(f"Error loading data at index {index}: {str(e)}")
                retry_count += 1
                # Try a different random index
                index = random.randint(0, len(self) - 1)
        
        # If all retries failed
        raise RuntimeError(f"Failed to load data after {self.max_retry} retries")

class EarthReasonDataset_test(Dataset):
    def __init__(self, 
                 data_dir='/mnt/HDD1/Datasets/EarthReason/EarthReason',
                 split='train',
                 transform=None,
                 mask_transform=None,
                 max_retry=5,
                 resize_size=840,
                 vis_dir=None):
        """
        Args:
            data_dir (str): Root directory of the EarthReason dataset
            split (str or list): Data split ('train', 'val', 'test') or a list of splits ['train', 'val']
            transform (callable, optional): Transform to be applied to images
            mask_transform (callable, optional): Transform to be applied to masks
            max_retry (int): Maximum retry count for loading images
            vis_dir (str, optional): Directory to save visualizations
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.max_retry = max_retry
        self.vis_dir = vis_dir

        self.resize_size = resize_size
        
        # Convert string split to a list for uniform handling
        if isinstance(split, str):
            split = [split]
        self.splits = split
        
        # Load dataset
        self.metadata = self._load_metadata()
        
        # Initialize visualization directory if specified
        if vis_dir and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def _load_metadata(self):
        """Load and validate metadata from image, label, and QA directories for all splits"""
        valid_metadata = []
        
        # Process each split
        for split in self.splits:
            image_dir = os.path.join(self.data_dir, split, 'images')
            label_dir = os.path.join(self.data_dir, split, 'labels')
            qa_dir = os.path.join(self.data_dir, split, 'QAs')
            
            if not os.path.exists(image_dir):
                print(f"Warning: Image directory {image_dir} not found for split '{split}'")
                raise FileNotFoundError(f"Image directory {image_dir} not found")
            
            # Get list of image files
            image_files = os.listdir(image_dir)
            print(f"Found {len(image_files)} images in {image_dir}")
            
            # Validate and create metadata
            for img_file in image_files:
                img_id = img_file.split('.')[0]
                img_path = os.path.join(image_dir, img_file)
                label_path = os.path.join(label_dir, f"{img_id}.png")
                qa_path = os.path.join(qa_dir, f"{img_id}.json")
                
                # Check if corresponding files exist
                if os.path.exists(img_path) and os.path.exists(label_path) and os.path.exists(qa_path):
                    # Load QA data to get questions
                    try:
                        with open(qa_path, 'r') as f:
                            qa_data = json.load(f)
                        
                        # Extract questions for prompts
                        if 'questions' in qa_data and qa_data['questions']:
                            item = {
                                'data_id': img_id,
                                'image_path': img_path,
                                'mask_path': label_path,
                                'qa_path': qa_path,
                                'prompts': qa_data['questions']
                            }
                            valid_metadata.append(item)
                        else:
                            print(f"Warning: No questions found in {qa_path}")
                    except Exception as e:
                        print(f"Error loading QA data from {qa_path}: {str(e)}")
                else:
                    missing = []
                    if not os.path.exists(img_path): missing.append("image")
                    if not os.path.exists(label_path): missing.append("label")
                    if not os.path.exists(qa_path): missing.append("QA data")
                    
                    print(f"Warning: Missing {', '.join(missing)} for {img_id} in split '{split}'")
        
        splits_str = ', '.join(self.splits)
        print(f"Loaded {len(valid_metadata)} valid instances from {splits_str} split{'s' if len(self.splits) > 1 else ''}")
        return valid_metadata

    def _build_conversation(self, question):
        """Construct conversation format for multimodal input"""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT.format(Question=question)}
                ]
            }
        ]

    def _visualize(self, image, mask, save_path):
        """Visualize and save image with mask overlay"""
        # Convert mask to colored visualization
        mask_vis = np.zeros_like(image)
        # Assuming mask contains class labels (adapt this based on your actual mask format)
        unique_labels = np.unique(mask)
        
        # Skip background (typically 0)
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
                
            # Use a different color for each label (simple color mapping)
            color = [(label * 50) % 255, (label * 80) % 255, (label * 110) % 255]
            mask_vis[mask == label] = color
        
        # Create blended visualization
        blended = cv2.addWeighted(image, 0.7, mask_vis, 0.3, 0)
        cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    def smart_resize(self,
        height: int, width: int, factor: int = 28, min_pixels: int = 4*28*28, max_pixels: int = 37*37*28*28, max_ratio: int = 100
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        def round_by_factor(number: int, factor: int) -> int:
            """Returns the closest integer to 'number' that is divisible by 'factor'."""
            return round(number / factor) * factor
        def floor_by_factor(number: int, factor: int) -> int:
            """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
            return math.floor(number / factor) * factor
        def ceil_by_factor(number: int, factor: int) -> int:
            """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
            return math.ceil(number / factor) * factor

        if max(height, width) / min(height, width) > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = floor_by_factor(height / beta, factor)
            w_bar = floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def __len__(self):
        """Return the total number of samples"""
        return len(self.metadata)

    def __getitem__(self, index):
        """Get a single dataset item"""
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                item = self.metadata[index]
                
                # Load image
                image = Image.open(item['image_path']).convert('RGB')
                
                # Load gt mask
                gt_mask = Image.open(item['mask_path'])
                
                # Select a random prompt/question
                prompt = random.choice(item['prompts'])
                
                ## resize image to adapt to Qwen2.5-VL default patch size
                width, height = image.size
                resized_height, resized_width = self.resize_size, self.resize_size
                # resized_height, resized_width = self.smart_resize(
                #     height, 
                #     width,
                #     factor=28,  # Qwen2.5-VL default patch size
                #     min_pixels=4*28*28,  
                #     max_pixels=37*37*28*28,  # 1024*1024->1036*1036
                #     )
                image = image.resize((resized_width, resized_height), Image.LANCZOS)  
                gt_mask = gt_mask.resize((resized_width, resized_height), Image.NEAREST)
                mask_array = np.array(gt_mask) > 0

                print(f"Resizing image from {width}x{height} to {resized_width}x{resized_height}")


                # # Apply transforms if specified
                # if self.transform:
                #     image = self.transform(image)
                
                # if self.mask_transform:
                #     gt_mask = self.mask_transform(Image.fromarray(mask_array))
                
                # Create visualization if specified
                if self.vis_dir:
                    image_array = np.array(image)
                    vis_path = os.path.join(self.vis_dir, f"{item['data_id']}.png")
                    self._visualize(image_array, mask_array, vis_path)

                # Prepare the dataset item
                return {
                    'data_idx': item['data_id'],
                    'image_path': item['image_path'],
                    'image': image,
                    'GT_mask_path': item['mask_path'],
                    'GT_mask': mask_array,
                    'problem': prompt,
                    'prompt': self._build_conversation(prompt)
                }
            
            except Exception as e:
                print(f"Error loading data at index {index}: {str(e)}")
                retry_count += 1
                # Try a different random index
                index = random.randint(0, len(self) - 1)
        
        # If all retries failed
        raise RuntimeError(f"Failed to load data after {self.max_retry} retries")


if __name__ == '__main__':
    # Simple test for the dataset
    dataset = EarthReasonDataset_test(
        data_dir='your_earthreason_data_path_here',
        split=['val'],      # 'val', 'test'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # # Print information for a few samples
    # for i in tqdm(range(min(100, len(dataset)))):
    #     item = dataset[i]
    #     print(f"Sample {i}:")
    #     print(f"  Data ID: {item['data_idx']}")
    #     print(f"  Image path: {item['image_path']}")
    #     print(f"  Image size: {item['image'].size if isinstance(item['image'], Image.Image) else item['image'].shape}")
    #     print(f"  Mask shape: {item['GT_mask'].shape}")
    #     print(f"  Problem: {item['problem']}")
    #     print("-" * 50)