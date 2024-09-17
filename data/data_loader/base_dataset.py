# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from data.data_loader import *
from data.data_loader.modules import *
from loguru import logger
import pdb


class BaseDataset(Dataset):
    def __init__(self,
                 data_path,
                 img_mode,
                 pre_processes,
                 filter_keys=[],
                 transform=None,
                 sample_rate=1.0,
                 save_augimg={},
                 **kwargs):
        assert img_mode in ['RGB', 'BGR', 'GRAY']
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.save_augimg = save_augimg
        self.transform = transform
        self.sample_rate = sample_rate

        self.count = 0
        self.data_handle, self.num_samples = self.pre_load(data_path)
        self._init_pre_processes(pre_processes)

    def __len__(self):
        epoch_len = int(self.sample_rate * self.num_samples)
        return epoch_len

    def __getitem__(self, index):  #zjx(self, index): #
        data = self.load_data(index)
        data = self.apply_pre_processes(data)
        # pdb.set_trace()
        if self.transform is not None:
            data['img'] = self.transform(data['img'])
            
        if self.save_augimg.get('switch') and self.count < self.save_augimg.get('num', 10):
            file_name = os.path.basename(data['img_path']).split('.')[0]     
            self.count += 1
            augimg_dir = os.path.join(self.save_augimg.get('save_dir'), 'aug_imgs')
            os.makedirs(augimg_dir, exist_ok=True)
            augimg_path = os.path.join(augimg_dir, '{}_{}.jpg'.format(file_name, data['label']))
            aug_img = data['img'].permute(1, 2, 0).numpy()
            means = self.transform.__dict__['transforms'][-1].__dict__['mean']
            stds = self.transform.__dict__['transforms'][-1].__dict__['std']
            aug_img = ((aug_img*stds + means) * 255).astype(np.uint8)
            cv2.imwrite(augimg_path, aug_img)
        

        if len(self.filter_keys):
            data_dict = {}
            for k, v in data.items():
                if k not in self.filter_keys:
                    data_dict[k] = v
            return data_dict
        else:
            return data

    def pre_load(self, data_path: str):
        raise NotImplementedError

    def load_data(self, index):
        data = copy.deepcopy(self.data_handle[index])
        img_path = data['img_path']
        img = cv2.imread(data['img_path'], 1 if self.img_mode != 'GRAY' else 0)
        if img is None:
            logger.error(f"{img_path} not exists!")
            
        # if self.img_mode.lower() == 'gray':
        #     img = np.expand_dims(img, axis=2)
        
        elif self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        data['img'] = img
        data['shape'] = [img.shape[0], img.shape[1]]
        return data

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

