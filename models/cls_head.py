import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class ClsHead(nn.Module):
    """
    Class orientation
    Args:
        params(dict): super parameters for build Class network
    """

    def __init__(self, in_channels, num_classes, pool_method='AdaptiveAvgPool3d', fc_inner_dims=[], **kwargs):
        super(ClsHead, self).__init__()
        self.training = False
        self.pool_method = pool_method
        if pool_method == 'AdaptiveMaxPool3d':
            self.pool = nn.AdaptiveMaxPool3d(2)
        elif pool_method == 'AdaptiveAvgPool3d':
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif pool_method == 'MaxPool3d':
            self.pool = nn.MaxPool3d(kernel_size=(2, 2), stride=(2, 2))
        elif pool_method == 'AvgPool3d':
            self.pool = nn.AvgPool3d(kernel_size=(2, 2), stride=(2, 2))
        
        if not fc_inner_dims:
            self.fc = nn.Linear(in_channels, num_classes, bias=True)
        else:
            fc_dims = [in_channels] + fc_inner_dims + [num_classes]
            self.fc = nn.Sequential()

            for i in range(len(fc_dims) - 1):
                self.fc.add_module('fc_{}'.format(i), nn.Linear(fc_dims[i], fc_dims[i+1], bias=True))
                if i != len(fc_dims) - 2:
                    self.fc.add_module('fc_relu_{}'.format(i),nn.ReLU(inplace=True))
        self.get_prob = kwargs.get('get_prob', True)
        
        self.Sigmoid = nn.Sigmoid()
        self.Last_activation = kwargs.get('last_activation', 'Softmax')
        
    def forward(self, x):
        if isinstance(x, list):
            x = x[-1]
        if self.pool_method is not None:
            x = self.pool(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        
        x_vector = x.view(-1, int(x.numel()/x.size(0)))  # torch.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.fc(x_vector)

        if not self.training and self.get_prob:
            if self.Last_activation == 'Sigmoid':
                x = self.Sigmoid(x)
            else:
                x = F.softmax(x, dim=1)
        return x
    