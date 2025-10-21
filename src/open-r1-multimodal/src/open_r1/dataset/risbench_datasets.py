import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from tqdm import tqdm
import math
import copy

# usual version, bbox + 2 pos_points
SYSTEM_PROMPT = "You are a remote sensing analysis assistant. Your task is to generate spatial prompts for the Segment Anything Model (SAM) based on a user's request."
USER_PROMPT = """Please find "{Question}", identify the target. For each target instance, provide:
1.  `bbox_2d`: A tight bounding box.
2.  `positive_points`: Exactly two points, placed inside the target.
Output your thinking process in <think> </think> tags.
Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
i.e. <think> thinking process here </think> 
<answer>```json[{{"bbox_2d": [310,360,567,586], "positive_points": [[434, 474], [450, 460]]}}, {{"bbox_2d": [10,200,100,320], "positive_points": [[50, 250], [90, 300]]}}]```</answer>"""

class RisBenchDataset(Dataset):
    def __init__(self, 
                 data_dir='/mnt/HDD1/Datasets/RISBench/RISBench_dataset',
                 split='train',
                 transform=None,
                 mask_transform=None,
                 max_retry=5,
                 resize_size=None,
                 vis_dir=None):
        """
        Args:
            data_dir (str): Root directory of the RISBench dataset
            split (str or list): Data split ('train', 'val', 'test') or a list of splits ['train', 'val']
            transform (callable, optional): Transform to be applied to images
            mask_transform (callable, optional): Transform to be applied to masks
            max_retry (int): Maximum retry count for loading images
            resize_size (int): Target size for image resizing
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
        """Load and validate metadata from txt files for all splits"""
        valid_metadata = []
        
        img_dir = os.path.join(self.data_dir, 'img_rgb')
        mask_dir = os.path.join(self.data_dir, 'mask')
        
        # Process each split
        for split in self.splits:
            txt_file = os.path.join(self.data_dir, f'output_phrase_{split}.txt')
            
            if not os.path.exists(txt_file):
                print(f"Warning: Text file {txt_file} not found for split '{split}'")
                continue
                
            # Parse lines from txt file
            filenames, phrases = self._parse_txt_file(txt_file)
            
            print(f"Found {len(filenames)} entries in {txt_file}")
            
            # Validate and create metadata
            for filename, phrase in zip(filenames, phrases):
                img_path = os.path.join(img_dir, filename)
                mask_path = os.path.join(mask_dir, filename)
                
                # Check if corresponding files exist
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    # Use filename without extension as data_id
                    data_id = filename.split('.')[0]
                    
                    item = {
                        'data_id': data_id,
                        'image_path': img_path,
                        'mask_path': mask_path,
                        'prompts': phrase
                    }
                    valid_metadata.append(item)
                else:
                    missing = []
                    if not os.path.exists(img_path): missing.append("image")
                    if not os.path.exists(mask_path): missing.append("mask")
                    
                    print(f"Warning: Missing {', '.join(missing)} for {filename} in split '{split}'")
        
        splits_str = ', '.join(self.splits)
        print(f"Loaded {len(valid_metadata)} valid instances from {splits_str} split{'s' if len(self.splits) > 1 else ''}")
        return valid_metadata

    def _parse_txt_file(self, txt_file):
        """Parse txt file to extract filenames and phrases"""
        filenames = []
        phrases = []
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)  # 只分割第一个空格
                if len(parts) == 2:
                    filename, phrase = parts
                    filenames.append(filename)
                    phrases.append(phrase)
        
        return filenames, phrases

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

        # Show the original image
        fig, ax = plt.subplots()
        ax.imshow(image)
        show_mask(mask, ax)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


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
                
                # Use the phrase as prompt
                prompt = item['prompts']
                
                ## resize image
                if self.resize_size:
                    width, height = image.size
                    resized_height, resized_width = self.resize_size, self.resize_size
                    
                    image = image.resize((resized_width, resized_height), Image.LANCZOS)  
                    gt_mask = gt_mask.resize((resized_width, resized_height), Image.NEAREST)
                    mask_array = np.array(gt_mask) > 0
                else:
                    mask_array = np.array(gt_mask) > 0

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
                    'prompt': self._build_conversation(prompt),
                }
            
            except Exception as e:
                print(f"Error loading data at index {index}: {str(e)}")
                retry_count += 1
                # Try a different random index
                index = random.randint(0, len(self) - 1)
        
        # If all retries failed
        raise RuntimeError(f"Failed to load data after {self.max_retry} retries")


if __name__ == '__main__':
    # Test the RisBenchDataset
    dataset = RisBenchDataset(
        data_dir='your_risbench_data_path_here',
        split='test',
        resize_size=None,
        vis_dir='./test/ris_vis_test'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a few samples
    for i in range(min(3, len(dataset))):
        try:
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Data ID: {sample['data_idx']}")
            print(f"  Image path: {sample['image_path']}")
            print(f"  Image size: {sample['image'].size if isinstance(sample['image'], Image.Image) else sample['image'].shape}")
            print(f"  Mask shape: {sample['GT_mask'].shape}")
            print(f"  Problem: {sample['problem']}")
            print(f"  Prompt length: {len(sample['prompt'])}")
            print(f"  Prompt content: {sample['prompt']}")
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            break