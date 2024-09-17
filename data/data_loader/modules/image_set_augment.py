# -*- coding: utf-8 -*-
# @Author  : jasonaidm

import sys
import math
import numbers
import random
import albumentations as A
import cv2
import numpy as np
import torch
from utils.load_dcm import *


class DCMSetAugment(object):
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
        dcm_read_method = random.choice(self.dcm_read_methods)
        obj_img_size = data.get('img_size', 512)
        global_random_state = random.random()
        default_random_state = random.random()
        blur_random_state = random.random()
        brightness_random_state = random.random()
        horizontal_flip_random_state = random.random()

        # 增强参数
        if global_random_state < self.gobal_aug_prob:
            if isinstance(self.win_change_rate_scope, list):
                win_change_rate = random.uniform(*self.win_change_rate_scope)
                new_win_width = self.win_width + int(self.win_width*win_change_rate)
                new_win_center = self.win_center + int(self.win_center*win_change_rate)
            else:
                new_win_width = self.win_width
                new_win_center = self.win_center
            
            
            resize_rate = random.uniform(*self.resize_rate_scope)
            crop_rate = random.uniform(*self.crop_rate_scope)
            blur_engine = A.OneOf([
                                A.Blur(blur_limit=5, p=1),
                                A.MotionBlur(blur_limit=7, p=1),
                                A.MedianBlur(blur_limit=5, p=1),
                                A.GaussianBlur(blur_limit=(3, 7), p=1),
                                A.ImageCompression(quality_lower=85, quality_upper=95, p=1),
                                ], p=1)
            brightness_rate = random.uniform(*self.brightness_rate_scope)
        
        img_list = []
        for dcm_file_path in dcm_file_list:
            if global_random_state < self.gobal_aug_prob:
                img_arr = dcm2img_by_method2(dcm_file_path, new_win_width, new_win_center, 
                                             proc_method=dcm_read_method)
                ori_h, ori_w = img_arr.shape[:2]
                if default_random_state < self.gobal_aug_prob:
                    ori_h += int(ori_h*resize_rate)
                    ori_w += int(ori_w*resize_rate)
                    img_arr = cv2.resize(img_arr, (ori_w, ori_h))
                if default_random_state < self.gobal_aug_prob:
                    crop_h = int(ori_h*crop_rate) // 2
                    crop_w = int(ori_w*crop_rate) // 2
                    img_arr = img_arr[crop_h: -crop_h, crop_w: -crop_w]
                
                if blur_random_state < self.blur_prob:
                    # A.Compose()
                    img_arr = blur_engine(image=img_arr)['image']
                
                if brightness_random_state < self.gobal_aug_prob:
                    brightness_val = img_arr.mean() * brightness_rate
                    img_arr = np.clip(img_arr.astype(np.int) + brightness_val, 0, 255).astype(np.uint8)
                
                if horizontal_flip_random_state < self.horizontal_flip_prob:
                    img_arr = np.fliplr(img_arr)
                
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
            
        # 转化为三维图片
        img = np.stack(img_list, axis=0)
        data['img'] = img
        return data


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