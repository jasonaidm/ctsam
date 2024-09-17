import random
import cv2
import numpy as np
import pdb
import os


class AssemblePatches(object):
    def __init__(self, resize=None, **kwargs):
        self.resize = resize
        self.dst_h, self.dst_w = kwargs.get('input_size')
        # self.interpolations

    def __call__(self, data: dict):
        img = data['img']
        img_h, img_w = img.shape[:2]
        ratio_h = img_h / self.dst_h
        ratio_w = img_w / self.dst_w
        # pdb.set_trace()
        if len(img.shape) == 3:
            shape = (self.dst_h, self.dst_w, img.shape[2])
        else:
            shape = (self.dst_h, self.dst_w)
        new_img = np.zeros(shape, dtype=img.dtype)
        if ratio_h >= ratio_w:
            tmp_h = self.dst_h*2
            if ratio_h > ratio_w*2:  # 如果宽度达不到高度的1/2，则需要形变resize
                tmp_w = tmp_h // 2
            else:
                tmp_w = int(tmp_h / img_h * img_w)
            img = cv2.resize(img, (tmp_w, tmp_h))
            # 取左上角区块和右下角区块
            new_img[:, :self.dst_w//2] = img[:self.dst_h, :self.dst_w//2]
            new_img[:, -self.dst_w//2:] = img[-self.dst_h:, -self.dst_w//2:]
        else:
            tmp_w = self.dst_w*2
            if ratio_w > ratio_h*2:  # 如果宽度达不到高度的1/2，则需要形变resize
                tmp_h = tmp_w // 2
            else:
                tmp_h = int(tmp_w / img_w * img_h)

            img = cv2.resize(img, (tmp_w, tmp_h))
            new_img[:self.dst_h//2, :] = img[:self.dst_h//2, :self.dst_w]
            new_img[-self.dst_h//2:, :] = img[-self.dst_h//2:, -self.dst_w:]

        data['img'] = new_img # data['category_id']
        data['shape'] = [new_img.shape[0], new_img.shape[1]]
        return data

