# -*- coding: utf-8 -*-
# @Author  : jasonaidm

import sys
import math
import numbers
import random

import cv2
import numpy as np
from skimage.util import random_noise


class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """
        对图片加噪声
        :param data: {'img':,'seg_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        data['img'] = (random_noise(data['img'], mode='gaussian', clip=True) * 255).astype(data['img'].dtype)
        return data


class RandomScale:
    def __init__(self, scales, random_rate):
        """
        :param scales: 尺度
        :param ramdon_rate: 随机系数
        :return:
        """
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'seg_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        seg_polys = data['seg_polys']

        tmp_seg_polys = seg_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_seg_polys *= rd_scale

        data['img'] = im
        data['seg_polys'] = tmp_seg_polys
        return data


class RandomRotateImgBox:
    def __init__(self, degrees, random_rate, same_size=False):
        """
        :param degrees: 角度，可以是一个数值或者list
        :param ramdon_rate: 随机系数
        :param same_size: 是否保持和原图一样大
        :return:
        """
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            degrees = (-degrees, degrees)
        elif isinstance(degrees, list) or isinstance(degrees, tuple) or isinstance(degrees, np.ndarray):
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            degrees = degrees
        else:
            raise Exception('degrees must in Number or list or tuple or np.ndarray')
        self.degrees = degrees
        self.same_size = same_size
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'seg_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']

        # ---------------------- 旋转图像 ----------------------
        w = im.shape[1]
        h = im.shape[0]
        angle = np.random.randint(self.degrees[0], self.degrees[1]+1)

        if self.same_size:
            nw = w
            nh = h
        else:
            # 角度变弧度
            rangle = np.deg2rad(angle)
            # 计算旋转之后图像的w, h
            nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
        # 构造仿射矩阵
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, 1)
        # 计算原图中心点到新图中心点的偏移量
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # 更新仿射矩阵
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        rot_img = cv2.warpAffine(im, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        data['img'] = rot_img

        if 'seg_polys' in data:
            seg_polys = data['seg_polys']
            # ---------------------- 矫正bbox坐标 ----------------------
            # rot_mat是最终的旋转矩阵
            # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
            rot_seg_polys = list()
            for bbox in seg_polys:
                point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
                point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
                point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
                point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
                rot_seg_polys.append([point1, point2, point3, point4])

            data['seg_polys'] = np.array(rot_seg_polys)
        return data


class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        """
        :param input_size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :param ramdon_rate: 随机系数
        :param keep_ratio: 是否保持长宽比
        :return:
        """
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, list) or isinstance(size, tuple) or isinstance(size, np.ndarray):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'seg_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        seg_polys = data['seg_polys']

        if self.keep_ratio:
            # 将图片短边pad到和长边一样
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        seg_polys = seg_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        seg_polys[:, :, 0] *= w_scale
        seg_polys[:, :, 1] *= h_scale

        data['img'] = im
        data['seg_polys'] = seg_polys
        return data


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img, (new_width / width, new_height / height)


class ResizeShortSize:
    def __init__(self, short_size, resize_seg_polys=True):
        """
        :param size: resize尺寸,数字或者list的形式，如果为list形式，就是[w,h]
        :return:
        """
        self.short_size = short_size
        self.resize_seg_polys = resize_seg_polys

    def __call__(self, data: dict) -> dict:
        """
        对图片和文本框进行缩放
        :param data: {'img':,'seg_polys':,'texts':,'ignore_tags':}
        :return:
        """
        im = data['img']
        seg_polys = data['seg_polys']

        h, w, _ = im.shape
        short_edge = min(h, w)
        if short_edge < self.short_size:
            # 保证短边 >= short_size
            scale = self.short_size / short_edge
            im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
            scale = (scale, scale)
            # im, scale = resize_image(im, self.short_size)
            if self.resize_seg_polys:
                # seg_polys *= scale
                seg_polys[:, 0] *= scale[0]
                seg_polys[:, 1] *= scale[1]

        data['img'] = im
        data['seg_polys'] = seg_polys
        return data


class HorizontalFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'seg_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        seg_polys = data['seg_polys']

        flip_seg_polys = seg_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_seg_polys[:, :, 0] = w - flip_seg_polys[:, :, 0]

        data['img'] = flip_im
        data['seg_polys'] = flip_seg_polys
        return data


class VerticallFlip:
    def __init__(self, random_rate):
        """

        :param random_rate: 随机系数
        """
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        """
        从scales中随机选择一个尺度，对图片和文本框进行缩放
        :param data: {'img':,'seg_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        im = data['img']
        seg_polys = data['seg_polys']

        flip_seg_polys = seg_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_seg_polys[:, :, 1] = h - flip_seg_polys[:, :, 1]
        data['img'] = flip_im
        data['seg_polys'] = flip_seg_polys
        return data


class HorizontalStretch(object):
    def __init__(self, min_scale, max_scale, random_rate):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.random_rate = random_rate

    def __call__(self, data):
        if random.uniform(0, 1) > self.random_rate:
            return data
        h, w = data['img'].shape[:2]
        scale = random.uniform(self.min_scale, self.max_scale)
        new_w = int(w * scale)
        data['img'] = cv2.resize(
            data['img'],
            dsize=(new_w, h),
            interpolation=cv2.INTER_LINEAR
        )
        return data


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['img']
        src_h, src_w, _ = img.shape

        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)
        data['img'] = img
        data['shape'] = [src_h, src_w, ratio_h, ratio_w]
        return data

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, _ = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        else:
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = int(round(resize_h / 32) * 32)
        resize_w = int(round(resize_w / 32) * 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        # return img, np.array([h, w])
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class RandomCrop(object):
    def __init__(self, crop_rate, random_rate):
        self.crop_rate = crop_rate
        self.random_rate = random_rate

    def __call__(self, data):
        if random.uniform(0, 1) > self.random_rate:
            return data
        h, w = data['img'].shape[:2]
        crop_h = int(h * self.crop_rate)
        crop_w = int(w * self.crop_rate)
        crop_h = np.random.randint(0, crop_h)
        crop_w = np.random.randint(0, crop_w)
        data['img'] = data['img'][crop_h:-crop_h, crop_w:-crop_w, :].copy()

        return data


class RandomRot90(object):
    def __init__(self, adjust_label=True, random_rate=1.0):
        self.adjust_label = adjust_label
        self.random_rate = random_rate
        self.tag_map = {0: 0, 1: 3, 2: 2, 3: 1}

    def __call__(self, data):
        if random.uniform(0, 1) > self.random_rate:
            return data
        randint = np.random.randint(0, 4)
        # pdb.set_trace()
        data['img'] = np.rot90(data['img'], randint)
        if self.adjust_label:
            new_label = data['label'] + self.tag_map[randint]
            if new_label > 3:
                new_label -= 4
            data['label'] = new_label

        return data
