# -*- coding: utf-8 -*-
import torch
from torch import nn

from models.losses.basic_loss import BalanceLoss
import pdb


class SegLoss(nn.Module):
    def __init__(self, ohem_ratio=3, reduction='mean', **kawargs):
        """
        Implement UNet Loss.
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.balance_row_loss = BalanceLoss(negative_ratio=ohem_ratio, main_loss_type='DiceLoss')
        self.balance_col_loss = BalanceLoss(negative_ratio=ohem_ratio, main_loss_type='DiceLoss')
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        # pdb.set_trace()
        row_maps = pred[:, 0, :, :]
        col_maps = pred[:, 1, :, :]
        # todo: 修改难例挖掘策略，mask里表格线非常近的难例
        loss_row_maps = self.balance_row_loss(row_maps, batch['row_map'], batch['mask'])
        loss_col_maps = self.balance_col_loss(col_maps, batch['col_map'], batch['mask'])
        loss = loss_row_maps + loss_col_maps
        loss_dict = dict(loss=loss, loss_rowk_maps=loss_row_maps, loss_col_maps=loss_col_maps)

        return loss_dict

