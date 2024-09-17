# -*- coding: utf-8 -*-
# @Time    : 2020/09/04 09:58
# @Author  : Jasonaidm


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
