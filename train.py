# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import pdb

# os.environ['MASTER_PORT'] = '29501'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' 

def init_args():
    parser = argparse.ArgumentParser(description='REC.DEV')
    parser.add_argument('-d', '--device_id', default=None, type=str)
    parser.add_argument('-c', '--config_file', default='configs/lung_vpi_sam_cls3_cocojson_all.yaml', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=os.getenv('LOCAL_RANK', 0), type=int, help='Use distributed training')

    args = parser.parse_args()
    return args


def main(args, config):
    from models import build_model, build_loss
    from data.data_loader import get_dataloader
    from utils import get_metric
    if args.device_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        config['distributed'] = True 
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://', 
                                             world_size=num_gpus,
                                             rank=args.local_rank)
    else:
        config['distributed'] = False
    
    config['local_rank'] = args.local_rank
    # pdb.set_trace()
    # train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    # assert train_loader is not None
    # pdb.set_trace()
    train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    if 'validate' in config['dataset'] and config['dataset']['validate']['dataset']['args']['data_path'][0]:
        validate_loader = get_dataloader(config['dataset']['validate'], False)
    else:
        validate_loader = None
    
    assert train_loader is not None
    try:
        criterion = build_loss(config['loss']).cuda()
    except AssertionError:
        loss_config = config['loss']
        loss_type = loss_config.pop('type')
        criterion = nn.__dict__.get(loss_type, None)(**loss_config)
        assert criterion is not None, "{} is unsupported!".format(config['loss']['type'])

    if 'backbone' in config['arch'] and 'in_channels' in config['arch']['backbone']:
        config['arch']['backbone']['in_channels'] = 3 \
            if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
    
    if 'head' in config['arch']:
        if config['arch']['head'].get('num_classes') is None:
            config['arch']['head']['num_classes'] = train_loader.dataset.num_classes

    # train_loader
    model = build_model(config['arch'])
    # pdb.set_trace()
    metric = get_metric(config['metric'])
    t = importlib.import_module('trainers')
    trainer = t.__dict__[config['train_module']](
        config=config,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        metric_cls=metric,
        validate_loader=validate_loader)
    trainer.train()

def join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])


if __name__ == '__main__':
    import sys
    import pathlib
    import yaml
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    from utils import parse_config
    args = init_args()
    
    # register the tag handler
    yaml.add_constructor('!join', join)
    assert os.path.exists(args.config_file)
    with open(args.config_file, errors='ignore') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    if 'base' in config:
        config = parse_config(config)
    main(args, config)
