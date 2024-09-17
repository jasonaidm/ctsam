# -*- coding: utf-8 -*-
# @Time    : 2020/12/4 14:39
# @Author  : jasonai
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class BalanceLoss(nn.Module):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    '''

    def __init__(self, negative_ratio=3.0, eps=1e-6, main_loss_type='CrossEntropy'):
        super(BalanceLoss, self).__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        if main_loss_type == "CrossEntropy":
            self.loss = nn.BCELoss()
        elif main_loss_type == "DiceLoss":
            self.loss = DiceLoss(self.eps)
        else:
            raise NotImplementedError("lazy ...")

    def forward(self,
                pred: torch.Tensor,
                gt: torch.Tensor,
                mask: torch.Tensor,
                return_origin=False):
        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''
        # pdb.set_trace()
        positive = (gt * mask).byte()
        negative = ((1 - gt) * mask).byte()
        positive_count = int(positive.float().sum())
        negative_count = min(int(negative.float().sum()), int(positive_count * self.negative_ratio))
        # loss = nn.functional.binary_cross_entropy(pred, gt, reduction='none')
        loss = self.loss(pred, gt, mask=mask)
        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()
        # negative_loss, _ = torch.topk(negative_loss.view(-1).contiguous(), negative_count)
        negative_loss, _ = negative_loss.view(-1).topk(negative_count)

        balance_loss = (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)

        if return_origin:
            return balance_loss, loss
        return balance_loss


class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        # pdb.set_trace()
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean', zero_infinity=True, loss_type='CTCLoss', **kwargs):
        super(CTCLoss, self).__init__()
        self.ctc = nn.CTCLoss(blank, reduction, zero_infinity)
        self.loss_type = loss_type
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, outputs, batch, batch_first=True):
        if self.loss_type == 'CrossEntropy':
            predicts = outputs.squeeze(0)
            target = torch.tensor(batch['targets']).to('cuda')
            loss = self.crossentropy(predicts, target)
            batch['labels'] = [[int(i.numpy()[0]) for i in batch['labels']]]
        else:
            log_probs = torch.log_softmax(outputs, dim=2).to(torch.float64)
            if batch_first:
                input_lengths = torch.LongTensor([log_probs.shape[1]] * log_probs.shape[0])
                log_probs = log_probs.permute(1, 0, 2)  # (b, w, c) --> (w, b, c)
            else:
                input_lengths = torch.LongTensor([log_probs.shape[0]] * log_probs.shape[1])
            loss = self.ctc(log_probs, batch['targets'], input_lengths, batch['target_lengths'])
        return loss


class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.center = nn.Parameter(10 * torch.randn(10, 2))
        self.lamda = 0.2
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature, label):
        batch_size = label.size()[0]
        nCenter = self.center.index_select(dim=0, index=label)
        distance = feature.dist(nCenter)
        centerloss = (1 / 2.0 / batch_size) * distance
        out = feature.mm(self.weight)
        ceLoss = F.cross_entropy(out, label)
        return out, ceLoss + self.lamda * centerloss


