import sys
import os.path as osp
import json
import pickle as pickle
import time
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pprint import pprint
import numpy as np 
from pycocotools import mask
import torch.utils.data as data
import os
import random
import cv2
from PIL import Image

SYSTEM_PROMPT = "You are a remote sensing analysis assistant. Your task is to generate spatial prompts for the Segment Anything Model (SAM) based on a user's request."
USER_PROMPT = """Please find "{ref_prompt}", identify the target. For each target instance, provide:
1.  `bbox_2d`: A tight bounding box.
2.  `positive_points`: Exactly two points, placed inside the target.
Output your thinking process in <think> </think> tags.
Output the final answer in <answer> </answer> tags with the specified JSON format. If no targets are found, output an empty list.
i.e. <think> thinking process here </think> 
<answer>```json[{{"bbox_2d": [310,360,567,586], "positive_points": [[434, 474], [450, 460]]}}, {{"bbox_2d": [10,200,100,320], "positive_points": [[50, 250], [90, 300]]}}]```</answer>"""


class RRSISD_REFER:

    def __init__(self, 
                 data_root='/mnt/SrvDataDisk/Datasets_RemoteSensing/RRSIS-D/RRSIS-D',
                 dataset='rrsisd', 
                 splitBy='unc'
                 ):
        # provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('loading dataset %s into memory...' % dataset)
        if dataset == 'refcocog':
            print('Split by {}!'.format(splitBy))
        self.DATA_DIR = osp.join(data_root, dataset)
        if dataset in ['refcoco', 'refcoco+', 'refcocog']:
            self.IMAGE_DIR = osp.join(data_root, 'images/mscoco/images/train2014')
        elif dataset == 'refclef':
            self.IMAGE_DIR = osp.join(data_root, 'images/saiapr_tc-12')
        elif dataset == 'rrsisd':
            self.IMAGE_DIR = osp.join(data_root, 'images/rrsisd/JPEGImages')
        else:
            print('No refer dataset is called [%s]' % dataset)
            sys.exit()

        # load refs from data/dataset/refs(dataset).json
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        self.data = {}
        self.data['dataset'] = dataset
        f = open(ref_file, 'r')
        self.data['refs'] = pickle.load(open(ref_file, 'rb'))

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_file, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # create index
        self.createIndex()
        print('DONE (t=%.2fs)' % (time.time() - tic))

    def createIndex(self):
        # create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('creating index...')
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data['annotations']:
            Anns[ann['id']] = ann
            imgToAnns[ann['image_id']] = imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            Imgs[img['id']] = img
        for cat in self.data['categories']:
            Cats[cat['id']] = cat['name']

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data['refs']:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            category_id = ref['category_id']
            image_id = ref['image_id']

            # add mapping related to ref
            Refs[ref_id] = ref
            imgToRefs[image_id] = imgToRefs.get(image_id, []) + [ref]
            catToRefs[category_id] = catToRefs.get(category_id, []) + [ref]
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref['sentences']:
                Sents[sent['sent_id']] = sent
                sentToRef[sent['sent_id']] = ref
                sentToTokens[sent['sent_id']] = sent['tokens']

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print('index created.')

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=''):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data['refs']
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.data['refs']
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # we also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann['id'] for ann in self.data['annotations']]
        else:
            if not len(image_ids) == 0:
                lists = [self.imgToAnns[image_id] for image_id in image_ids if
                         image_id in self.imgToAnns]  # list of [anns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data['annotations']
            if not len(cat_ids) == 0:
                anns = [ann for ann in anns if ann['category_id'] in cat_ids]
            ann_ids = [ann['id'] for ann in anns]
            if not len(ref_ids) == 0:
                ids = set(ann_ids).intersection(set([self.Refs[ref_id]['ann_id'] for ref_id in ref_ids]))
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=[]):
        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=[]):
        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == unicode:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=[]):
        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=[]):
        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # show image
        image = self.Imgs[ref['image_id']]
        # I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        I = Image.open(osp.join(self.IMAGE_DIR, image['file_name'])).convert('RGB')
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # show segmentations
        if seg_box == 'seg':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) / 2, 2))
                    polygons.append(Polygon(poly, True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == 'box':
            ann_id = ref['ann_id']
            ann = self.Anns[ann_id]
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]
        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']

        m = mask.decode(rle)
        m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
        m = m.astype(np.uint8)  # convert to np.uint8
        # compute area
        area = sum(mask.area(rle))  # should be close to ann['area']

        return {'mask': m, 'area': area}


    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)

class RRSISD_Dataset(data.Dataset):
    def __init__(self,
                 data_dir='/mnt/SrvDataDisk/Datasets_RemoteSensing/RRSIS-D/RRSIS-D',
                 dataset='rrsisd',
                 splitBy='unc',
                 split='train',
                 transform=None,
                 mask_transform=None,
                 max_retry=5,
                 vis_dir=None):
        """
        Args:
            data_dir (str): Root directory of the RRSISD dataset
            dataset (str): Dataset name
            splitBy (str): Split method
            split (str): Data split ('train', 'val', 'test')
            transform (callable, optional): Transform to be applied to images
            mask_transform (callable, optional): Transform to be applied to masks
            max_retry (int): Maximum retry count for loading images
            vis_dir (str, optional): Directory to save visualizations
        """
        self.data_dir = data_dir
        self.dataset = dataset
        self.splitBy = splitBy
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform
        self.max_retry = max_retry
        self.vis_dir = vis_dir
        # self.resize_size = 784
        self.resize_size = 840
        
        # Initialize RRSISD REFER API
        self.refer = RRSISD_REFER(data_root=data_dir, dataset=dataset, splitBy=splitBy)
        
        # Get reference IDs for the specified split
        self.ref_ids = self.refer.getRefIds(split=self.split)
        print(f"Loaded {len(self.ref_ids)} references from {self.split} split")
        
        # Initialize visualization directory if specified
        if vis_dir and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def _visualize(self, image, mask, save_path):
        """Visualize and save image with mask overlay"""
        # Convert mask to colored visualization
        mask_vis = np.zeros_like(image)
        # Assuming mask contains binary values (0 or 1)
        mask_vis[mask > 0] = [0, 255, 0]  # Green for the segmented region
        
        # Create blended visualization
        blended = cv2.addWeighted(image, 0.7, mask_vis, 0.3, 0)
        cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    def __len__(self):
        """Return the total number of samples"""
        return len(self.ref_ids)
    
    def __getitem__(self, index):
        """Get a single dataset item"""
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                # Get reference ID and corresponding image
                ref_id = self.ref_ids[index]
                ref = self.refer.loadRefs(ref_id)[0]
                img_id = ref['image_id']
                img_info = self.refer.Imgs[img_id]
                
                # Load image
                image_path = os.path.join(self.refer.IMAGE_DIR, img_info['file_name'])
                image = Image.open(image_path).convert('RGB')

                image = image.resize((self.resize_size, self.resize_size), Image.LANCZOS)
                
                # Load mask
                mask_dict = self.refer.getMask(ref)
                # mask_array = mask_dict['mask']
                # mask_array = mask_array > 0

                gt_mask = Image.fromarray(mask_dict['mask'])
                gt_mask = gt_mask.resize((self.resize_size, self.resize_size), Image.NEAREST)
                mask_array = np.array(gt_mask) > 0
                
                # # Get a random referring expression if multiple are available
                # sent_idx = random.randrange(len(ref['sentences']))
                # ref_prompt = ref['sentences'][sent_idx]['sent']

                ref_prompt = ref['sentences'][0]['sent'] + '.'
                
                # # Apply transforms if specified
                # if self.transform:
                #     image = self.transform(image)
                
                # if self.mask_transform and isinstance(mask_array, np.ndarray):
                #     mask = self.mask_transform(Image.fromarray(mask_array))
                
                # # Create visualization if specified
                # if self.vis_dir:
                #     image_array = np.array(image)
                #     vis_path = os.path.join(self.vis_dir, f"{ref_id}.jpg")
                #     self._visualize(image_array, mask_array, vis_path)
                
                # Prepare the dataset item
                return {
                    'data_idx': ref_id,
                    'image_path': image_path,
                    'image': image,
                    'GT_mask_path': None,  # Not directly available for RRSISD
                    'GT_mask': mask_array,
                    'ins_GT_mask': mask_array,
                    'problem': ref_prompt,
                    'prompt': self._build_conversation(ref_prompt),
                    'bbox_point_GT': None, # add later
                }
            
            except Exception as e:
                print(f"Error loading data at index {index}: {str(e)}")
                retry_count += 1
                # Try a different random index
                index = random.randint(0, len(self) - 1)
        
        # If all retries failed
        raise RuntimeError(f"Failed to load data after {self.max_retry} retries")

    def _build_conversation(self, ref_prompt):
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
                    {"type": "text", "text": USER_PROMPT.format(ref_prompt=ref_prompt)}
                ]
            }
        ]
        
if __name__ == '__main__':
    # Simple test for the dataset
    dataset = RRSISD_Dataset(
        data_dir='your_rrsisd_data_path_here',
        dataset='rrsisd',
        split='train',      # 'train', 'val', 'test'
    )
    
    print(f"Dataset size: {len(dataset)}")

    # def plot_img_mask(img, mask, problem=None, figsize=(8, 4)):
    #     # Create figure for visualization
    #     fig, axes = plt.subplots(1, 2, figsize=figsize)
        
    #     # Add problem text as suptitle if provided
    #     if problem:
    #         fig.suptitle(f"Problem: {problem}", fontsize=12)
    #         fig.subplots_adjust(top=0.85)  # Make room for the title
        
    #     # Plot original image
    #     axes[0].imshow(img)
    #     axes[0].set_title('Image')
    #     axes[0].axis('off')
        
    #     # Plot full mask with color-coding
    #     axes[1].imshow(mask)
    #     axes[1].set_title(f'Mask')
    #     axes[1].axis('off')

    #     plt.tight_layout()
    #     plt.show()
    
    # # Print information for a few samples
    # for i in range(min(20, len(dataset))):
    #     item = dataset[i]
    #     print(f"Sample {i}:")
    #     print(f"  Data ID: {item['data_idx']}")
    #     print(f"  Image path: {item['image_path']}")
    #     print(f"  Image size: {item['image'].size if isinstance(item['image'], Image.Image) else item['image'].shape}")
    #     print(f"  Mask shape: {item['GT_mask'].shape}")
    #     print(f"  Problem: {item['problem']}")
    #     print(f"  Prompt: {item['prompt']}")
    #     print("-" * 50)

    #     plot_img_mask(item['image'], item['GT_mask'], problem=item['problem'], figsize=(10, 5))
    #     plt.savefig(f"test/sample_{i}.png")
        