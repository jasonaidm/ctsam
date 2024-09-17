# -*- coding: utf-8 -*-
# @Author  : Jasonaidm
import numpy as np
# from seqeval.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support


def cal_text_accuracy(text_indexes, labels):
    hit_count = 0
    batch_size = len(labels)
    for i in range(batch_size):
        pred_idx = text_indexes[i]
        label_idx = labels[i]
        if pred_idx == label_idx:
            hit_count += 1
    acc = hit_count / batch_size
    return acc


class ClsMetric(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, preds, labels):
        hit_count = 0
        batch_size = len(labels)
        for i in range(batch_size):
            pred_idx = preds[i]
            label_idx = labels[i]
            if pred_idx == label_idx:
                hit_count += 1
        acc = hit_count / batch_size
        return acc


class F1ScoreMetric(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, preds: list, labels: list):
        precision, recall, fscore, _ = precision_recall_fscore_support(labels, preds, average='macro')
        return fscore


class runningScore(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)

        if np.sum((label_pred[mask] < 0)) > 0:
            print(label_pred[label_pred < 0])
        hist = np.bincount(n_class * label_true[mask].astype(int) +
                           label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        # print label_trues.dtype, label_preds.dtype
        for lt, lp in zip(label_trues, label_preds):
            try:
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
            except:
                pass

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        # 混淆矩阵计算
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
        # 类别平均
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
        acc_cls = np.nanmean(acc_cls)
        # pdb.set_trace()
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.0001)
        mean_iu = np.nanmean(iu)
        # zjx add:
        pos_iu = iu[-1]
        freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {'Overall Acc': acc,
                'Mean Acc': acc_cls,
                'FreqW Acc': fwavacc,
                'Mean IoU': mean_iu,
                'Positive Iou': pos_iu}, cls_iu

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class NERMetric(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, labels, preds):
        return {
            "precision": precision_score(labels, preds),
            "recall": recall_score(labels, preds),
            "hmean": f1_score(labels, preds)
        }


class SeqEvalMetric(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, labels, preds):
        mean_acc = 0.
        batch_size = len(labels)
        for label, pred in zip(labels, preds):
            seq_len = len(label)
            acc = 0
            for i in range(seq_len):
                if label[i] == pred[i]:
                    acc += 1
            acc /= seq_len
            mean_acc += acc
        mean_acc /= batch_size
        return {"acc": mean_acc}
