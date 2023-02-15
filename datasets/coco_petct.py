# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image

from panopticapi.utils import rgb2id
from util.box_ops import masks_to_boxes

from .coco import make_coco_transforms # TO DO: will want to use own PET specifc transforms


def random_shuffle_two_lists(list1, list2):
    random.seed(42)
    
    temp = list(zip(list1, list2))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    
    return res1, res2


def get_annotations(ann_file):
    with open(ann_file, 'r') as f:
        coco = json.load(f)
    
    # sort 'images' and 'annotations' so that they are aligned with 'annotations'
    # i.e., in alphabetical order
    coco['images'] = sorted(self.coco['images'], key=lambda x: x['image_id']) 
    coco['annotations'] = sorted(self.coco['annotations'], key=lambda x: x['image_id']) 
    
    return coco


def train_val_split(ann_file, train_split, cross_val, val_fold):
    coco = get_annotations(ann_file)
    
    train_dict = {'images':[],'annotations':[]}
    val_dict = {'images':[],'annotations':[]}
    num_train = ((train_split*len(coco['images']))//48)*48 #there are 48 slices per study
    
    # TO DO: If NOT cross validation, then split per train_split
    if not cross_val:
        train_dict['images'] = coco['images'][:num_train].copy()
        train_dict['annotations'] = coco['annotations'][:num_train].copy()
        val_dict['images'] = coco['images'][num_train:].copy()
        val_dict['annotations'] = coco['annotations'][num_train:].copy()
    # If cross validation, sort by image_id 
    else:
        num_folds = 1//(1-train_split)
        train_folds = list(range(0,num_folds)).remove(val_fold)
        num_val = len(coco['images'])-num_train
        coco['images'] = [coco['images'][i:i+num_val] for i in range(0, len(coco['images']), num_val)]
        coco['annotations'] = [coco['annotations'][i:i+num_val] for i in range(0, len(coco['annotations']), num_val)]
        val_dict['images'] = coco['images'][val_fold].copy()
        val_dict['annotations'] = coco['annotations'][val_fold].copy()
        for fold in train_folds:
            train_dict['images'] = train_dict['images'].extend(coco['images'][fold].copy())
            train_dict['annotations'] = train_dict['annotations'].extend(coco['annotations'][fold].copy())
    
    # shuffle the images -- otherwise 48 images in a row will be from same patient      
    train_dict['images'], train_dict['annotations'] = random_shuffle_two_lists(train_dict['images'], train_dict['annotations'])
    
    return train_dict, val_dict


class CocoPETCT:
    def __init__(self, img_folder, ann_folder, ann_dict, transforms=None, return_masks=True):
        self.coco = annot_dict
        
        # sanity check
        if "annotations" in self.coco:
            for img, ann in zip(self.coco['images'], self.coco['annotations']):
                #assert img['file_name'][:-4] == ann['file_name'][:-4]
                assert img['image_id'] == ann['image_id']

        self.img_folder = img_folder
        self.ann_folder = ann_folder # don't really need this for the petct dataset as they're the same
        #self.ann_file = ann_file # already taken care of by train_val_split
        self.transforms = transforms
        self.return_masks = return_masks

    def __getitem__(self, idx):
        #ann_info = self.coco['annotations'][idx] if "annotations" in self.coco else self.coco['images'][idx]
        #img_path = Path(self.img_folder) / ann_info['file_name'].replace('.png', '.jpg')
        #ann_path = Path(self.ann_folder) / ann_info['file_name']
        ann_info = self.coco['annotations'][idx] 
        img_info = self.coco['images'][idx]
        img_ann_path = Path(self.img_folder) / ann_info['file_name']

        #img = Image.open(img_path).convert('RGB')
        with open(img_ann_path, 'rb') as f:
            suv_img = np.load(f) # this is a gray image... need to make into rgb and also want to augment for different SUV max
            masks = np.load(f)
        w, h = suv_img.size
        
        # Presaved for petct dataset
        img = torch.as_tensor(img, dtype=torch.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)
        
#         if "segments_info" in ann_info:
#             masks = np.asarray(Image.open(ann_path), dtype=np.uint32)
#             masks = rgb2id(masks)

#             ids = np.array([ann['id'] for ann in ann_info['segments_info']])
#             masks = masks == ids[:, None, None]

#             masks = torch.as_tensor(masks, dtype=torch.uint8)
#             labels = torch.tensor([ann['category_id'] for ann in ann_info['segments_info']], dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([ann_info['image_id'] if "image_id" in ann_info else img_info["image_id"]])
        if self.return_masks:
            target['masks'] = masks
        target['labels'] = labels

        #target["boxes"] = masks_to_boxes(masks)
        target["boxes"] = torch.tensor([ann['bbox'] for ann in ann_info['segments_info']], dtype=torch.int64)

        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        
        if "segments_info" in ann_info:
            for name in ['iscrowd', 'area']:
                target[name] = torch.tensor([ann[name] for ann in ann_info['segments_info']])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.coco['images'])

    def get_height_and_width(self, idx):
        img_info = self.coco['images'][idx]
        height = img_info['height']
        width = img_info['width']
        return height, width
    
    def get_img_demo(self, idx):
        img_info = self.coco['images'][idx]
        age = img_info['age']
        sex = img_info['sex']
        diagnosis = img_info['diagnosis']
        return age, sex, diagnosis
    
    def get_img_suv(self, idx):
        stats = self.coco['images'][idx]['nii_stats']
        liver_suv = stats['liver']
        brain_suv = stats['brain']
        max_suv = stats['suv_max'] 
        return liver_suv, brain_suv, max_suv
    
    def get_dcm_spacing(self, idx):
        stats = self.coco['images'][idx]['nii_stats']
        w = stats['pixdim'][1]
        y = stats['pixdim'][3]
        spacing = (1, y/w)
        return spacing


def build(image_set, args):
    img_folder_root = Path(args.coco_path)
    ann_folder_root = Path(args.coco_petct_path)
    assert img_folder_root.exists(), f'provided COCO path {img_folder_root} does not exist'
    assert ann_folder_root.exists(), f'provided COCO path {ann_folder_root} does not exist'
    mode = 'detr'
    PATHS = {
        "train": ("images/train", Path("annotations") / f'{mode}_train.json'), # this will be used for train and val
        "test": ("images/test", Path("annotations") / f'{mode}_test.json'), # this will be set aside for final testing
    }

    img_folder, ann_file = PATHS[image_set]
    img_folder_path = img_folder_root / img_folder
    #ann_folder = ann_folder_root / f'{mode}_{img_folder}'
    ann_folder = image_folder_path # binary masks pre-saved in same .npy as the image
    ann_file = ann_folder_root / ann_file
    
    if image_set == "train":
        # customized train val split
        train_dict, val_dict = train_val_split(ann_file, args.train_split, args.cross_val, args.val_fold)

        train_dataset = CocoPETCT(img_folder_path, ann_folder, train_dict,
                               transforms=make_coco_transforms("train"), return_masks=args.masks)

        val_dataset = CocoPETCT(img_folder_path, ann_folder, val_dict,
                               transforms=make_coco_transforms("val"), return_masks=args.masks)
        
    elif image_set == "test":
        train_dataset = None
        
        val_dict = get_annotations(ann_file)
        val_dataset = CocoPETCT(img_folder_path, ann_folder, val_dict,
                               transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return train_dataset, val_dataset
