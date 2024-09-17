import random
import cv2
import numpy as np
import pdb
import os


class BboxCropper(object):
    def __init__(self, 
                 expand_scales=[0.1, 0.4], 
                 shift_scales=[-0.2, 0.2],
                 max_expand=128,
                 max_shift=64,
                 keep_position=False, 
                 resize=None,
                 padded_value=114,
                 **kwargs
                 ):

        self.expand_scales = expand_scales
        self.max_expand = max_expand
        self.shift_scales = shift_scales
        self.max_shift = max_shift
        self.keep_position = keep_position
        self.resize = resize
        self.padded_value = padded_value
        self.dst_h, self.dst_w = kwargs.get('input_size')
        self.do_resize = kwargs.get('do_resize')
        self.narrow_edge_type = kwargs.get('narrow_edge_type')
        # self.interpolations

    def __call__(self, data: dict):
        img = data['img']
        img_h, img_w = img.shape[:2]
        x1, y1, w, h = [int(i) for i in data['bbox']]
        # 去除检测框在padding中的case
        
        if self.resize is not None:
            rate = 1.
            if isinstance(self.resize, int):
                resize = self.resize
                if random.random() < 0.7:
                    resize += random.randint(-32, 32)
                
                if img_h > img_w:
                    rate = resize / img_h
                    img_w = int(rate * img_w)
                    img_h = resize
                else:
                    rate = resize / img_w
                    img_h = int(rate * img_h)
                    img_w = resize
                  
            elif isinstance(self.resize, float) and self.resize < 1.0:
                rate = random.uniform(-self.resize, self.resize)
                rate += 1
                img_h = min(int(img_h * rate), img_h)
                img_w = min(int(img_w * rate), img_w)
            
            # interpolation = random.choice(self.interpolations)
            img = cv2.resize(img, dsize=(img_w, img_h)) #, interpolation=interpolation)
            
            # resize bbox
            x1 *= rate
            y1 *= rate
            w *= rate
            h *= rate
        
        # bbox中心点坐标偏移扰动
        shift_x = w*random.uniform(*self.shift_scales)
        shift_y = h*random.uniform(*self.shift_scales)
        shift_x = int(np.clip(shift_x, -1*self.max_shift, self.max_shift))
        shift_y = int(np.clip(shift_y, -1*self.max_shift, self.max_shift))
        x1 += shift_x
        y1 += shift_y

        x2 = x1 + w
        y2 = y1 + h
        
        if self.narrow_edge_type == 'jingxiang':
            if w <= self.dst_w:
                delta_x = random.randint(self.dst_w//2-1, self.dst_w//2 + 4)
            else:
                delta_x = min(w//2, random.randint(self.dst_w//2, self.dst_w))
                
            cx = (x1 + x2) // 2
            x1 = cx - delta_x
            x2 = cx + delta_x
        
        elif self.narrow_edge_type == 'weixiang':
            if h < self.dst_h:
                delta_y = random.randint(self.dst_h//2-1, self.dst_h + 4)
            else:
                delta_y = min(h//2, random.randint(self.dst_h//2, self.dst_h))
                
            cy = (y1 + y2) // 2
            y1 = cy - delta_y
            y2 = cy + delta_y
            
        else:
            expand_rate = random.uniform(*self.expand_scales)
            delta_x = int(min(w*expand_rate, self.max_expand))
            delta_y = int(min(h*expand_rate, self.max_expand))
            # print(delta_x, delta_x)
            # 对不同的类别进行定制化外扩
        
            # 保证经向/纬向的疵点宽度/高度不会过窄
            # if w < h:
            #     delta_x = max(delta_x, 40)
            # if h < 50:
            #     delta_y = max(delta_y, 10)
            x1 -= delta_x
            y1 -= delta_y
            x2 += delta_x
            y2 += delta_y

        # clip
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(img_w, x2))
        y2 = int(min(img_h, y2))
        new_w = x2 - x1
        new_h = y2 - y1
        cropped_img = img[y1:y1+new_h, x1:x1+new_w, :].copy()
        
        if self.do_resize:
            if (new_h / self.dst_h) > (new_w / self.dst_w):
                rate = self.dst_h / new_h
            else:
                rate = self.dst_w / new_w
            new_w = int(new_w * rate)
            new_h = int(new_h * rate)
            cropped_img = cv2.resize(cropped_img, dsize=(new_w, new_h))

        top = 0
        bottom = 0
        left = 0
        right = 0
        if new_h<self.dst_h:
            top = (self.dst_h-new_h) // 2
            bottom = self.dst_h - top - new_h
        if new_w<self.dst_w:
            left = (self.dst_w-new_w) // 2
            right = self.dst_w-left-new_w
        # pdb.set_trace()
        img2 = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114,114,114])

        src_h, src_w = img2.shape[:2]
        if (src_h>self.dst_h) or (src_w>self.dst_w):
            cropped_x1 = (src_w-self.dst_w) // 2
            cropped_y1 = (src_h-self.dst_h) // 2
            img2 = img2[cropped_y1:cropped_y1+self.dst_h, cropped_x1:cropped_x1+self.dst_w, :]

        data['img'] = img2.copy() # data['category_id']
        data['shape'] = [img2.shape[0], img2.shape[1]]
        return data

