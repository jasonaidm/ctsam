import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from models.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
from models.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from models.prompt_encoder import PromptEncoder, TwoWayTransformer
from functools import partial


class VPIRecBySAM(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        sam = sam_model_registry[kwargs.get('sam_type', 'vit_b')](checkpoint=kwargs.get('sam_ckpt', 'ckpt/sam_vit_b_01ec64.pth'))
        self.mask_generator = SamAutomaticMaskGenerator(sam)
        rand_crop_size = kwargs.get('rand_crop_size', [64, 128, 128])
        self.D_patch_size, self.HW_patch_size = rand_crop_size[:2]
        self.img_encoder = ImageEncoderViT_3d(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            patch_depth=32*self.D_patch_size//self.HW_patch_size,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            cubic_window_size=8,
            out_chans=256,
            num_slice = 16)
        self.img_encoder.load_state_dict(self.mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
        del sam
        # self.img_encoder.to(self.device)

        for p in self.img_encoder.parameters():
            p.requires_grad = False
        self.img_encoder.depth_embed.requires_grad = True
        for p in self.img_encoder.slice_embed.parameters():
            p.requires_grad = True
        for i in self.img_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.adapter.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)
        for i in self.img_encoder.neck_3d:
            for p in i.parameters():
                p.requires_grad = True

        # self.prompt_encoder_list = []
        self.parameter_list = []

        self.prompt_encoder = PromptEncoder(transformer=TwoWayTransformer(depth=2,
                                                                    embedding_dim=256,
                                                                    mlp_dim=2048,
                                                                    num_heads=8)
                                                                    )
        self.parameter_list.extend([i for i in self.prompt_encoder.parameters() if i.requires_grad == True])

        self.mask_decoder = VIT_MLAHead(img_size=96, num_classes=kwargs.get('num_classes', 3)) # mask解码器
        # self.init_status = False
        

    def forward(self, batch):
        # input_data = F.interpolate(batch['img'], scale_factor=512 / self.HW_patch_size, mode='trilinear')
        input_data = batch['img']

        # 三维resize和二维resize
        resize_3d = True
        if resize_3d:
            # 单通道转换为三通道
            input_data = input_data.repeat(1, 3, 1, 1, 1)
            # 上采样HW 4倍
            input_data = F.interpolate(input_data, scale_factor=512/self.HW_patch_size, mode='trilinear')  # zjx: 后续提特征时，会进行多次下采样
            input_data = input_data[0].transpose(0, 1) # zjx: 将Depth维放在batch维度
        else:
             input_data = F.interpolate(input_data, scale_factor=512/self.HW_patch_size)
             input_data = input_data.repeat(3, 1, 1, 1)
             input_data = input_data.transpose(0, 1)
        
        
        batch_features, feature_list = self.img_encoder(input_data)
        feature_list.append(batch_features)
        points = self.get_points(batch['seg'])  # 缺失seg
        new_feature = []
        for i, feature in enumerate(feature_list):
            if i == 3:   # 位置prompt编码
                new_feature.append(
                    self.prompt_encoder(feature, points.clone(), [self.HW_patch_size, self.HW_patch_size, self.D_patch_size], list(feature.shape[2:]))  # zjx mod
                )
            else:
                new_feature.append(feature)
        img_resize = F.interpolate(batch['img'].permute(0, 2, 3, 1).unsqueeze(1), scale_factor=0.5, mode='trilinear')
        new_feature.append(img_resize)
        masks = self.mask_decoder(new_feature, 2, 2)
        masks = masks.permute(0, 1, 4, 2, 3)  # zjx: 调整空间排布顺序为[N, C, D, H, W]
        return masks

    def get_points(self, seg):
        # seg = seg.cpu()
        pos_seg_tuple = torch.where(seg > 0)
        l = len(pos_seg_tuple[0])
        points_torch = None
        if l > 0:  # TODO: 可视化
            sample = np.random.choice(np.arange(l), 10, replace=True) 
            x = pos_seg_tuple[1][sample].unsqueeze(1)  # 所随机选取的10个点的depth空间标
            y = pos_seg_tuple[3][sample].unsqueeze(1)  # 所随机选取的10个点的width空间标
            z = pos_seg_tuple[2][sample].unsqueeze(1)  # 所随机选取的10个点的height空间标
            points_torch = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
            # points_torch = points_torch.to(seg.device)
            points_torch = points_torch.transpose(0,1)
        neg_seg_tuple = torch.where(seg == 0)
        l = len(neg_seg_tuple[0])
        sample = np.random.choice(np.arange(l), 20, replace=True)
        x = neg_seg_tuple[1][sample].unsqueeze(1)
        y = neg_seg_tuple[3][sample].unsqueeze(1)
        z = neg_seg_tuple[2][sample].unsqueeze(1)
        points_torch_negative = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
        # points_torch_negative = points_torch_negative.to(seg.device)
        points_torch_negative = points_torch_negative.transpose(0, 1)
        if points_torch is not None:  # 考虑到完全没有mask的ct样本
            points_torch = torch.cat([points_torch, points_torch_negative], dim=1)
        else:
            points_torch = points_torch_negative
        return points_torch