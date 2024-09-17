import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from data.data_loader.modules import Transform3D
from models.vpi_rec import VPIRecBySAM
from utils.load_dcm import dcm2img_by_method2


class Predictor(object):
    def __init__(self, ckpt_path, device='cpu', depth=32, patch_size=128, mean=[0.456], std=[0.224], **kwargs):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.vpi_rec_model = VPIRecBySAM(sam_type='vit_b', 
                                         rand_crop_size=[depth, patch_size, patch_size],
                                         num_classes=3,
                                         sam_ckpt='./model_files/sam_vit_b_01ec64.pth',
                                         )

        self.vpi_rec_model.load_state_dict(ckpt['state_dict'])
        self.vpi_rec_model.eval()
        self.vpi_rec_model = self.vpi_rec_model.to(device)
        self.device = device
        self.depth = depth
        self.patch_size = patch_size
        self.transform = Transform3D(mean=mean, std=std)

    def __call__(self, img_dir, points):
        img_patch, point_torch = self.preprocess(img_dir, points)
        batch = {'img': img_patch, 'points_torch': point_torch}
        with torch.no_grad():
            masks = self.vpi_rec_model(batch)
        return masks

    def preprocess(self, img_dir, points, file_type='dcm'):
        # get 3d coord
        x, y, z = points  # width, height, depth
        img_paths = os.listdir(img_dir)
        z = len(img_paths) // 2
        
        # conver to 3d image
        # 基于prompt point坐标，在原图上裁剪出patch_size大小的patch
        x_min = max(0, x - self.patch_size // 2)
        x_max = max(self.patch_size, x + self.patch_size // 2)
        y_min = max(0, y - self.patch_size // 2)
        y_max = max(self.patch_size, y + self.patch_size // 2)
        z_min = z - self.depth // 2
        z_max = z + self.depth // 2

        imgs = []
        for i in range(z_min, z_max):
            img_path = os.path.join(img_dir, img_paths[i])
            if file_type == 'dcm':
                img = dcm2img_by_method2(img_path, winwidth=1600, wincenter=-600)
            else:
                img = cv2.imread(img_path, flags=0)
            imgs.append(img[y_min:y_max, x_min:x_max])
        imgs = np.stack(imgs, axis=0)

        # 标准化处理
        img_patch = self.transform({'img': imgs})['img'].to(self.device)
        point_torch = torch.tensor([[z, x, y]]).unsqueeze(0).to(self.device).float()
        return img_patch, point_torch


if __name__ == '__main__':
    ckpt_path = 'all_outputs/lung_vpi_sam_cls3_cocojson_all/checkpoint/model_best.pth'
    predictor = Predictor(ckpt_path, device='cuda:0')
    img_dir = '/data1/zjx/medical/ct_model/data/dataset/dcm_stage1_30samples/10'
    points = [293, 299, 32]  # x, y, z
    masks = predictor(img_dir, points)

    # 输出类别
    masks_softmax = torch.softmax(masks, 1)
    cls_map = masks_softmax.argmax(1)
    roi_pix_num = len(cls_map[cls_map==1])
    neg_pix_num = len(cls_map[cls_map==2])
    if roi_pix_num > neg_pix_num:
        print('## positive')
    else:
        print('## negative')

    # 预测所有样本，并计算f1 score
