# -*- coding: utf-8 -*-

import os
import pathlib
import shutil
from pprint import pformat
import sys
import anyconfig
import torch
from utils import setup_logger
# from mqbench.prepare_by_platform import prepare_by_platform, BackendType
import pdb


class BaseTrainer:
    def __init__(self, config, model, criterion):
        config['trainer']['output_dir'] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent),
                                                       config['trainer']['output_dir'])
        
        # pdb.set_trace()
        if hasattr(model, 'name'):
            config['name'] = config['name'] + '_' + model.name
        self.save_dir = os.path.join(config['trainer']['output_dir'], config['name'])
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        local_rank = config['local_rank']
        
        # if config['trainer']['resume_checkpoint'] == '' and config['trainer']['finetune_checkpoint'] == '':
        #     shutil.rmtree(self.save_dir, ignore_errors=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.start_epoch = 0
        self.config = config
        self.model = model
        self.criterion = criterion
        
        # logger and tensorboard
        self.tensorboard_enable = self.config['trainer']['tensorboard']
        self.epochs = self.config['trainer']['epochs']
        self.log_iter = self.config['trainer']['log_iter']
        if local_rank == 0:
            anyconfig.dump(config, os.path.join(self.save_dir, 'config.yaml'))
            self.logger = setup_logger(os.path.join(self.save_dir, 'train.log'))
            self.logger_info(pformat(self.config))

        # 统计模型参数量和flops
        if self.config.get('get_flops'):
            from ptflops import get_model_complexity_info
            input_size = self.config['dataset'].get('input_size', [224, 224])
            input_channel = self.config['dataset'].get('input_channel', 3)
            input_shape = [input_channel] + input_size
            self.logger.info(f"输入shape为{input_shape}时，模型参数量和flops信息如下：")
            flops, params = get_model_complexity_info(model, tuple(input_shape), as_strings=True, print_per_layer_stat=True, verbose=True)
            self.logger.info(f"模型参数量为：{params}；计算复杂度FLOPs为：{flops}")
        # device
        torch.manual_seed(self.config['trainer']['seed'])  # 为CPU设置随机种子
        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.device = torch.device("cuda", local_rank)
            torch.cuda.manual_seed(self.config['trainer']['seed'])  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(self.config['trainer']['seed'])  # 为所有GPU设置随机种子
        else:
            self.with_cuda = False
            self.device = torch.device("cpu")
        self.logger_info('train with device {} and pytorch {}'.format(self.device, torch.__version__))
        # metrics
        self.metrics = {'acc': 0., 'train_loss': float('inf'), 'best_model_epoch': 0}
        # 冻结指定层参数
        if 'freeze' in self.config['trainer']:
            optim_params = self._freeze_param(self.model, **self.config['trainer']['freeze'])
        else:
            optim_params = model.parameters()
        self.optimizer = self._initialize('optimizer', torch.optim, optim_params)
        # resume or finetune
        if self.config['trainer']['resume_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
        elif self.config['trainer']['finetune_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['finetune_checkpoint'], resume=False)

        if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
            self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
    
        self.model.to(self.device)

        if self.tensorboard_enable and local_rank == 0:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)
            try:
                # add graph
                image_size = config['dataset']['train']['dataset']['args']['pre_processes'][1]['args']['size']
                in_channels = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
                dummy_input = torch.zeros(1, in_channels, image_size[0], image_size[1]).to(self.device)
                self.writer.add_graph(self.model, dummy_input)
                torch.cuda.empty_cache()
            except:
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warn('add graph to tensorboard failed')
        # 分布式训练
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank],
                                                                   output_device=local_rank,
                                                                   broadcast_buffers=False,
                                                                   find_unused_parameters=True
                                                                   )
        

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            try:
                # 在分布式模式下，需要在每个 epoch 开始时调用set_epoch()方法，然后再创建 DataLoader 迭代器，
                # 以使shuffle 操作能够在多个 epoch 中正常工作
                if self.config['distributed']:  
                    self.train_loader.sampler.set_epoch(epoch)  
                self.epoch_result = self._train_epoch(epoch)
                if self.config['lr_scheduler']['type'] != 'WarmupPolyLR':
                    self.scheduler.step()
            except KeyboardInterrupt:
                print("current model weight will be saved ...")
                self._on_epoch_finish()
                sys.exit()

            self._on_epoch_finish()
        if self.config['local_rank'] == 0 and self.tensorboard_enable:
            self.writer.close()
        self._on_train_finish()

    def _freeze_param(self, net, **kwargs):
        optim_params = []
        freeze_type = kwargs.get('type', 'filter')
        layers = kwargs.get('layers', [])
        # pdb.set_trace()
        for n, p in net.named_parameters():
            if freeze_type == 'filter':
                p.requires_grad = False
                for layer in layers:
                    if layer in n:
                        p.requires_grad = True
                        optim_params.append(p)
                        break
            else:
                # p.requires_grad = True
                for layer in layers:
                    if layer in n:
                        p.requires_grad = False
                
                if p.requires_grad:
                    optim_params.append(p)
        # pdb.set_trace()
        return optim_params

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval(self, epoch):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_epoch_finish(self):
        raise NotImplementedError

    def _on_train_finish(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self.model.module.state_dict() if self.config['distributed'] else self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        if self.config['optimizer'].get('is_save', False):
            state['optimizer'] = self.optimizer.state_dict()

        filename = os.path.join(self.checkpoint_dir, file_name)
        
        # if self.config['trainer']['quantize']:
        #     filename = filename.replace('.pth', '_q.pth')
        
        torch.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        match_info = self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            #pdb.set_trace()
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            # pdb.set_trace()
            if checkpoint.get('optimizer', None):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger_info("resume from checkpoint {} (epoch {}); {}".format(checkpoint_path, self.start_epoch, match_info))
        else:
            self.logger_info("finetune from checkpoint {}; {}".format(checkpoint_path, match_info))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        if self.config[name].get('is_ext', False):
            from tools import optimizer as ext_optimizer
            module = ext_optimizer
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)  # torch.optimizer.Adam

    def logger_info(self, s):
        if self.config['local_rank'] == 0:
            self.logger.info(s)
