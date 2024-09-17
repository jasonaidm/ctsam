# -*- coding: utf-8 -*-
# @Author  : Jasonaidm
import copy
import torch
import PIL
import numpy as np
from .cls_dataset import *
from torch.utils.data import DataLoader
from torchvision import transforms
import pdb


def get_dataset(data_path, module_name, transform, dataset_args):
    """
    获取训练dataset
    :param data_path: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    # from data.data_loader import det_dataset
    s_dataset = eval(module_name)(transform=transform, data_path=data_path,
                                  **dataset_args)
    return s_dataset


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


class RecTargetCollectFN(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        targets = []
        target_lengths = []
        img = []
        labels = []
        for sample in batch:
            targets.extend(sample['targets'])
            target_lengths.append(sample['target_lengths'])
            img.append(sample['img'])
            labels.append(sample['targets'])

        data_dict = dict(
            targets=torch.tensor(targets, dtype=torch.int64),
            target_lengths=torch.tensor(target_lengths, dtype=torch.int64),
            img=torch.stack(img, 0),
            labels=labels
        )

        return data_dict



class TSRCollectFN(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        targets = []
        padded_targets = []
        bboxes = []
        bbox_masks = []
        imgs = []
        labels = []
        img_paths = []
        for sample in batch:
            targets.append(sample['target'])
            padded_targets.append(sample['padded_target'])
            bboxes.append(sample['bbox'])
            bbox_masks.append(sample['bbox_masks'])
            imgs.append(sample['img'])
            labels.append(sample['label'])
            img_paths.append(sample['img_path'])

        data_dict = dict(
            targets=targets,
            padded_targets=torch.stack(padded_targets, 0).long(),
            padded_bboxes=torch.stack(bboxes, 0).float(),
            padded_bbox_masks=torch.stack(bbox_masks, 0).long(),
            img=torch.stack(imgs, 0),
            labels=labels,
            img_paths=img_paths
        )

        return data_dict


class VolDetCollectFN(object):
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, batch):
        imgs = []
        segs = []
        bboxes_list = []
        category_list = []
        for sample in batch:
            imgs.append(sample['img'][0])
            segs.append(torch.from_numpy(sample['seg']))
            bbox_list = []
            for label_item in sample['gt_label_list']:
                bbox_list.append(label_item[0]['bbox'])
            bboxes = torch.from_numpy(np.array(bbox_list, dtype=np.float32))
            bboxes_list.append(bboxes)
            category = label_item[0]['category_id']
            category_list.append(category)
        data_dict = dict(
            img=torch.stack(imgs, 0),
            seg=torch.stack(segs, 0),
            bboxes=torch.stack(bboxes_list, 0),
            category=category_list  # torch.tensor(category, dtype=torch.int8)
        )
        return data_dict


def get_dataloader(module_config, distributed=False):
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    # 创建数据集
    dataset_name = config['dataset']['type']
    data_path = dataset_args.pop('data_path')
    if data_path == None:
        return None
    if isinstance(data_path, list):
        data_path = [x for x in data_path if x is not None]
    if len(data_path) == 0:
        return None
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        # config['loader']['collate_fn'] = None
        config['loader']['collate_fn'] = torch.utils.data.dataloader.default_collate
    else:
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])()

    _dataset = get_dataset(data_path=data_path, module_name=dataset_name, transform=img_transfroms, dataset_args=dataset_args)
    # pdb.set_trace()
    # _dataset.zjx(0)
    # _dataset.zjx(2473)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        # 3）使用DistributedSampler
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False
        config['loader']['pin_memory'] = True
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])
    return loader
