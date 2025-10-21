import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import numpy as np
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


class RefSegRSDataset(Dataset):
    def __init__(self, 
                 data_root,
                 split='test',
                 transform=None,
                 mask_transform=None,
                 max_retry=5,
                 resize_size=512):
        self.data_root = data_root
        # self.transform = transform
        # self.mask_transform = mask_transform
        self.max_retry = max_retry
        self.resize_size = resize_size
        
        self.images_dir = os.path.join(data_root, 'images')
        self.masks_dir = os.path.join(data_root, 'masks')
        self.phrase_file = os.path.join(data_root, f'output_phrase_{split}.txt')
        
        self.samples = self._load_and_validate_samples()
        print(f'Loaded {len(self)} valid samples')

    def _load_and_validate_samples(self):
        valid_samples = []
        
        with open(self.phrase_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 2:
                    continue  
                
                img_id = parts[0]
                phrase = ' '.join(parts[1:])
                
                img_path = os.path.join(self.images_dir, f'{img_id}.tif')
                mask_path = os.path.join(self.masks_dir, f'{img_id}.tif')
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    valid_samples.append({
                        'img_id': img_id,
                        'phrase': phrase,
                        'img_path': img_path,
                        'mask_path': mask_path
                    })
        
        return valid_samples

    def _build_conversation(self, question):

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                sample = self.samples[index]
                
                image = Image.open(sample['img_path']).convert('RGB')
                
                GT_mask_image = Image.open(sample['mask_path'])
                GT_mask = np.array(GT_mask_image)[:, :, 0]
                # [0,255] -> [0,1]
                GT_mask = GT_mask > 0
                
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
                GT_mask_image = GT_mask_image.resize((resized_width, resized_height), Image.NEAREST)
                GT_mask = np.array(GT_mask_image)[:, :, 0] > 0

                return {
                    'data_idx': index,
                    'image_path': sample['img_path'],
                    'image': image,
                    'GT_mask_path': sample['mask_path'],
                    'GT_mask': GT_mask,
                    'problem': sample['phrase'],
                    'prompt': self._build_conversation(sample['phrase'])
                }
            
            except Exception as e:
                print(f"Error loading {index}: {str(e)}")
                index = random.randint(0, len(self)-1)
                retry_count += 1
        
        raise RuntimeError(f"Failed to load data after {self.max_retry} retries")

    @staticmethod
    def collate_fn(batch):
        return {
            'image_path': [item['image_path'] for item in batch],
            'image': torch.stack([item['image'] for item in batch]),
            'mask': torch.stack([item['mask'] for item in batch]),
            'description': [item['description'] for item in batch],
            'conversation': [item['conversation'] for item in batch]
        }
    
if __name__ == '__main__':

    dataset = RefSegRSDataset(data_root='your_refsegrs_data_path_here', 
                            split='val',
                            transform=None, mask_transform=None)
    # availableSplits = ['train', 'test', 'val']
    print(len(dataset))


    for i in range(10):
        info = dataset[i]
        # print(info['problem']+'\n')
        print(np.unique(info['GT_mask']))
        print(info['GT_mask'].shape)
        print(info['prompt'])