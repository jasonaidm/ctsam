# -*- coding: utf-8 -*-
import copy
from .basic_loss import CTCLoss
from .seg_loss import *
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from monai.losses import DiceCELoss, DiceLoss

# __all__ = ['build_loss']
# support_loss = ['DBLoss', 'PSELoss', 'CTCLoss', 'ArcCTCLoss']


def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    # assert loss_type in support_loss, 'all support loss is {}'.format(support_loss)
    criterion = eval(loss_type)(**copy_config)
    return criterion
