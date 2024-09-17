# -*- coding: utf-8 -*-
# @Author  : jasonaidm

import sys
import math
import numbers
import random
from typing import Any
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Union
from monai.config import IndexSelection, KeysCollection, SequenceStr
from monai.transforms import (
    Compose,
    AddChanneld,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
)
from utils.load_dcm import *


class BinarizeLabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            threshold: float = 0.5,
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d


class DCMAugmentForSeg(object):
    def __init__(self, **kwargs):
        self.win_width = kwargs.get('win_width', 1600)
        self.win_center = kwargs.get('win_center', -600)
        self.win_change_rate_scope = kwargs.get('win_change_rate_scope', [-0.1, 0.1])
        self.dcm_read_methods = kwargs.get('dcm_read_methods', ['sigmoid', 'linear'])
        self.gobal_aug_prob = kwargs.get('gobal_aug_prob', 0)
        self.resize_rate_scope = kwargs.get('resize_rate_scope', [-0.15, 0.15])
        self.crop_rate_scope = kwargs.get('crop_rate_scope', [0.05, 0.15])
        # self.rotate_scope = kwargs.get('rotate_scope')
        # self.add_noise_prob = kwargs.get('add_noise_prob', 0)
        self.brightness_rate_scope = kwargs.get('brightness_rate_scope') #, [-0.1, 0.1])
        self.blur_prob = kwargs.get('blur_prob', 0)
        # self.random_contrast_prob = kwargs.get('random_contrast', 0)
        self.horizontal_flip_prob = kwargs.get('horizontal_prob', 0)

    def __call__(self, data: dict):
        dcm_file_list = data['dcm_file_list']
        label_list = data['label_list']
        dcm_read_method = random.choice(self.dcm_read_methods)
        obj_img_size = data.get('img_size', 512)
        # 增强参数
        global_random_state = random.random()
        if global_random_state < self.gobal_aug_prob:
            if isinstance(self.win_change_rate_scope, list):
                win_change_rate = random.uniform(*self.win_change_rate_scope)
                new_win_width = self.win_width + int(self.win_width*win_change_rate)
                new_win_center = self.win_center + int(self.win_center*win_change_rate)
            else:
                new_win_width = self.win_width
                new_win_center = self.win_center
        img_list = []
        seg_list = []
        for idx, dcm_file_path in enumerate(dcm_file_list):
            if global_random_state < self.gobal_aug_prob:
                img_arr = dcm2img_by_method2(dcm_file_path, new_win_width, new_win_center, 
                                             proc_method=dcm_read_method)
                
                # 调整图片size到目标size
                current_h, current_w = img_arr.shape[:2]
                if obj_img_size < max(current_h, current_w):
                    resize_rate2 = obj_img_size / max(current_h, current_w)
                    img_arr = cv2.resize(img_arr, (int(current_w * resize_rate2), int(current_h * resize_rate2)))
                    current_h, current_w = img_arr.shape[:2]
                new_img = np.ones(shape=(obj_img_size, obj_img_size), dtype=np.uint8) * 114
                delta_h = (obj_img_size - current_h) // 2
                delta_w = (obj_img_size - current_w) // 2
                new_img[delta_h:delta_h + current_h, delta_w:delta_w + current_w] = img_arr
                img_list.append(new_img)
            else:
                img_arr = dcm2img_by_method2(dcm_file_path, self.win_width, self.win_center, 
                                             proc_method=dcm_read_method)
                img_list.append(cv2.resize(img_arr, dsize=(obj_img_size, obj_img_size)))
            
            # 处理标签，分割格式
            seg = np.zeros(shape=(obj_img_size, obj_img_size), dtype=np.float32)
            label_item = label_list[idx]
            if label_item is not None:
                for item in label_item:
                    bbox = item['bbox']
                    category_id = item['category_id']
                    seg[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[0])+int(bbox[2])] = category_id
            seg_list.append(seg)

        # stack to 3d
        img_vol = np.stack(img_list, axis=0)
        data['img'] = img_vol

        seg_vol = np.stack(seg_list, axis=0)
        data['seg'] = seg_vol
        return data


class DCMAugment(object):
    def __init__(self, **kwargs):
        self.win_width = kwargs.get('win_width', 1600)
        self.win_center = kwargs.get('win_center', -600)
        # self.win_change_rate_scope = kwargs.get('win_change_rate_scope', [-0.1, 0.1])
        self.dcm_read_methods = kwargs.get('dcm_read_methods', ['sigmoid', 'linear'])
        

    def __call__(self, data: dict):
        dcm_file_list = data['dcm_file_list']
        dcm_read_method = random.choice(self.dcm_read_methods)
        obj_img_size = data.get('img_size', 512)

        img_list = []
        for dcm_file_path in dcm_file_list:
            img_arr = dcm2img_by_method2(dcm_file_path, self.win_width, self.win_center, 
                                            proc_method=dcm_read_method)
            img_list.append(cv2.resize(img_arr, dsize=(obj_img_size, obj_img_size)))
        

        # stack to 3d
        img_vol = np.stack(img_list, axis=0)
        data['img'] = img_vol

        return data


class ImgAugment(object):
    def __init__(self, **kwargs):
        self.img_size = kwargs.get('img_size', 512)
        

    def __call__(self, data: dict):
        dcm_file_list = data['dcm_file_list']
        img_list = []
        for dcm_file_path in dcm_file_list:
            # print(dcm_file_path)
            img_arr = cv2.imread(dcm_file_path, cv2.IMREAD_GRAYSCALE)
            try:
                img_list.append(cv2.resize(img_arr, dsize=(self.img_size, self.img_size)))
            except Exception as e:
                print(dcm_file_path, e)
        
        # stack to 3d
        img_vol = np.stack(img_list, axis=0)
        data['img'] = img_vol

        return data
    

class VolumeCrop(object):
    def __init__(self, **kwargs):
        self.crop_size = kwargs.get('crop_size', 128)
        self.do_debug_just_one = False
    
    def __call__(self, data: dict) -> Any:
        img_vol = data['img']
        img_h, img_w = img_vol.shape[1:]
        # 制作segmentation mask
        seg_vol = np.zeros_like(img_vol)
        num_seg_mask = len(data['gt_label_list'])
        start_idx = seg_vol.shape[0]//2-num_seg_mask//2

        min_x = img_w
        min_y = img_h
        max_x = 0
        max_y = 0
        for cursor_idx in range(num_seg_mask):
            bbox = data['gt_label_list'][cursor_idx][0]['bbox']
            x, y, w, h = [int(x) for x in bbox]

            # 在img_vol上画框
            # img_vol2 = cv2.rectangle(img_vol[start_idx+cursor_idx], (x, y), (x+w, y+h), 0, 1)
            # cv2.imwrite(f'./img_vol2_{cursor_idx}.jpg', img_vol2)  

            min_x = min(x, min_x)
            max_x = max(x+w, max_x)
            min_y = min(y, min_y)
            max_y = max(y+h, max_y)

            category_id = data['gt_label_list'][cursor_idx][0]['category_id'] #;print(f"#### category_id: {category_id}")
            seg_vol[start_idx+cursor_idx, :][y:y+h, x:x+w] = category_id

        # 裁剪
        crop_x1_min = max(0, max_x - self.crop_size)
        crop_y1_min = max(0, max_y - self.crop_size)
        crop_x1_max = max(0, min(min_x+self.crop_size, img_w) - self.crop_size)
        crop_y1_max = max(0, min(min_y+self.crop_size, img_h) - self.crop_size)
        
        # TODO: bbox修正

        try:
            x1 = random.randint(crop_x1_min, crop_x1_max)
            x2 = x1 + self.crop_size
            y1 = random.randint(crop_y1_min, crop_y1_max)
            y2 = y1 + self.crop_size
            data['img'] = img_vol[:, y1:y2, x1:x2]
            data['seg'] = seg_vol[:, y1:y2, x1:x2]
        except Exception as err:
            print(err)

        # if data['seg'].max() > 1:
        #     pdb.set_trace()

        if os.getenv('debug') and self.do_debug_just_one:
            debug_dir = os.path.join(os.getenv('debug'), 'VolumeCrop')
            img_dir = os.path.join(debug_dir, 'img')
            seg_dir = os.path.join(debug_dir, 'seg')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(seg_dir, exist_ok=True)

            for idx in range(data['img'].shape[0]):
                cv2.imwrite(os.path.join(img_dir, f'img_{idx}.jpg'), data['img'][idx])
            for idx in range(data['seg'].shape[0]):
                cv2.imwrite(os.path.join(seg_dir, f'seg_{idx}.jpg'), np.clip(data['seg'][idx]*255, 0, 255))
                                                                       
            self.do_debug_just_one = True


        return data 


class VolumeAugByMonai(object):
    def __init__(
            self,
            split="train",
            rand_crop_spatial_size=(96, 96, 96),
            do_val_crop=True,
    ):
        super().__init__()
        self.split = split
        self.rand_crop_spatial_size = rand_crop_spatial_size
        self.do_val_crop = do_val_crop
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.target_class = 1
        self.transforms = self.get_transforms()

    def __call__(self, data: dict) -> Any:
        seg = np.expand_dims(data['seg'], 0)
        seg = (seg == self.target_class).astype(np.float32)

        img = np.expand_dims(data['img'], 0)

        if self.split == "train" or ((self.do_val_crop  and self.split=='val')):
            try:
                trans_dict = self.transforms({"image": img, "label": seg})[0]
            except:
                print(1)  # 需要解决无标签的问题
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

        else:
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

        data['img'] = img_aug.repeat(3, 1, 1, 1)
        data['seg'] = seg_aug.squeeze()
        return data

    def get_transforms(self):
        transforms = [
            # ScaleIntensityRanged(
            #     keys=["image"],
            #     a_min=self.intensity_range[0],
            #     a_max=self.intensity_range[1],
            #     b_min=self.intensity_range[0],
            #     b_max=self.intensity_range[1],
            #     clip=True,
            # ),
        ]

        if self.split == "train":
            transforms.extend( 
                [
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=20,
                        prob=0.1,
                    ),
                    CropForegroundd(  # zjx: 确定是否有用
                        keys=["image", "label"],
                        source_key="label",
                        select_fn=lambda x: x > 0,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                ]
            )

            transforms.extend(
                [
                    RandZoomd(
                        keys=["image", "label"],
                        prob=0.8,
                        min_zoom=0.85,
                        max_zoom=1.25,
                        mode=["trilinear", "trilinear"],
                    ),
                ]
            )

            transforms.extend(
                [
                    BinarizeLabeld(keys=["label"]),
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                        label_key="label",
                        pos=2,
                        neg=1,
                        num_samples=1,
                    ),
                    RandSpatialCropd(
                        keys=["image", "label"],
                        roi_size=self.rand_crop_spatial_size,
                        random_size=False,
                    ),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                ]
            )
        elif (not self.do_val_crop) and (self.split == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.split == "val"):
            transforms.extend(
                [
                    SpatialPadd(
                        keys=["image", "label"],
                        spatial_size=[i for i in self.rand_crop_spatial_size],
                    ),
                    RandCropByPosNegLabeld(
                        keys=["image", "label"],
                        spatial_size=self.rand_crop_spatial_size,
                        label_key="label",
                        pos=1,
                        neg=0,
                        num_samples=1,
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.split == "test":
            transforms.extend(
                [
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)

        return transforms

 
class Transform3D(object):
    def __init__(self, **kwargs):
        self.mean = kwargs.get('mean', [0.485, 0.456, 0.406])
        self.std = kwargs.get('std', [0.229, 0.224, 0.225])
    
    def __call__(self, data: dict):
        img = torch.from_numpy(data['img']).unsqueeze(0)
        img = img.float() / 255.0
        mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device)
        std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device)
        img.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        data['img'] = img
        return data