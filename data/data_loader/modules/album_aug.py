import random
import albumentations as A


class ClsAlbum:
    def __init__(self, global_prob=0.5, blur_prob=0.2, hsv_prob=0.2, noise_prob=0.1, 
                 rotation_prob=0.1, **kwargs):
        self.global_prob = global_prob
        self.transform = A.Compose([
            # 模糊增强
            A.OneOf([
            A.Blur(blur_limit=5, p=1),
            A.MotionBlur(blur_limit=7, p=1),
            A.MedianBlur(blur_limit=5, p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.ImageCompression(quality_lower=85, quality_upper=95, p=1),
            ], p=blur_prob),
            
            # 噪声
            A.OneOf([
            A.RandomGamma(gamma_limit=(60, 120), p=1),
            A.GaussNoise(var_limit=(00, 50), mean=0, p=1),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1),
            A.ISONoise(color_shift=(0.01, 0.10), intensity=(0.1, 0.8), p=0.5),
            ], p=noise_prob),
            
            # 明暗度
            A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), p=0.5),
            A.HueSaturationValue(hue_shift_limit=20,
                                 sat_shift_limit=30,
                                 val_shift_limit=(-20, 10),
                                 p=0.5),
            ], p=hsv_prob),
            A.CoarseDropout(max_holes=48, fill_value=127, p=kwargs.get('dropout_prob', 0.)),
            A.GridDropout(ratio=0.1,shift_x=0, shift_y=0, fill_value=(0, 0, 0), p=kwargs.get('grid_dropout_prob', 0.05)),
            A.ShiftScaleRotate(shift_limit=0.03,rotate_limit=3, p=rotation_prob),
            A.HorizontalFlip(p=kwargs.get('horizontal_flip', 0.3)),
            A.VerticalFlip(p=kwargs.get('vertical_flip', 0.3)),
            ])


    def __call__(self, data: dict):
        if random.random() < self.global_prob:
            aug_res = self.transform(image=data['img'])
            data['img'] = aug_res['image']
        return data
