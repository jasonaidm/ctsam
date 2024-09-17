# -*- coding: utf-8 -*-
import copy
from .resnet3d import *
# from .swin3d import *
from .vpi_rec import *
from .losses import build_loss


def build_model(config):
    """
    get model architecture
    """
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.pop('type')
    arch_args = copy_config.pop('args')
    arch_model = eval(arch_type)(**arch_args)
    return arch_model
