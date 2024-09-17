import torch
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
# Step 1: Initialize model with the best available weights


def swin3d_tiny(in_channels=1, num_classes=2, **kwargs):
    weights = Swin3D_T_Weights.DEFAULT
    model = swin3d_t(in_channels=in_channels, num_classes=num_classes, **kwargs)
    if weights is not None:
        ckpt_weights = weights.get_state_dict(progress=True)
        if in_channels != ckpt_weights['patch_embed.proj.weight'].shape[1]:
            ckpt_weights['patch_embed.proj.weight'] = ckpt_weights['patch_embed.proj.weight'][:, 1:2]
        
        if num_classes != ckpt_weights['head.weight'].shape[0]:
            del ckpt_weights['head.weight']
            del ckpt_weights['head.bias']
        model.load_state_dict(ckpt_weights, strict=False)
    return model


if __name__ == '__main__':
    model = swin3d_tiny(in_channels=1, num_classes=2).to('cuda:0')
    model.eval()
    a = torch.rand(1, 1, 4, 512, 512).cuda()
    y = model(a)
    print(y)




