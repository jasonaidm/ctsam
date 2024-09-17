# -*- coding: utf-8 -*-
import time

import torch
from torch.optim import AdamW
import numpy as np
from trainers.base_trainer import BaseTrainer
from utils import WarmupPolyLR

from monai.losses import DiceCELoss, DiceLoss
import pdb
from loguru import logger


class SegTrainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader, metric_cls, post_process=None):
        super().__init__(config, model, criterion)
        self.show_images_iter = self.config['trainer']['show_images_iter']
        self.train_loader = train_loader
        # if validate_loader is not None:
            # assert post_process is not None and metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.metrics['score'] = 0.0
        self.max_epoch = 500
        self.model = model
        self.criterion = criterion

        if train_loader is not None:
            self.train_loader_len = len(train_loader)
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
                if self.start_epoch > 1:
                    self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
                self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                            warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
            if self.validate_loader is not None:
                self.val_loader_len = len(self.validate_loader)
                self.logger_info(
                    'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                        len(self.train_loader.dataset), self.train_loader_len,
                        len(self.validate_loader.dataset), len(self.validate_loader))
                )
            else:
                self.logger_info('train dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len)
                    )
        lr = self.optimizer.param_groups[0]['lr']
        self.dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
        self.loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)

        self.encoder_opt = AdamW([i for i in self.model.img_encoder.parameters() if i.requires_grad==True], lr=lr, weight_decay=0)
        self.encoder_scheduler = torch.optim.lr_scheduler.LinearLR(self.encoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)

        # for idx in range(len(self.model.prompt_encoder_list)):
        #     self.model.prompt_encoder_list[idx] = self.model.prompt_encoder_list[idx].to(self.device)
        self.model.prompt_encoder = self.model.prompt_encoder.to(self.device)

        self.feature_opt = AdamW(self.model.parameter_list, lr=lr, weight_decay=0)
        self.feature_scheduler = torch.optim.lr_scheduler.LinearLR(self.feature_opt, start_factor=1.0, end_factor=0.01,
                                                            total_iters=500)
        self.decoder_opt = AdamW([i for i in self.model.mask_decoder.parameters() if i.requires_grad == True], lr=lr, weight_decay=0)
        self.decoder_scheduler = torch.optim.lr_scheduler.LinearLR(self.decoder_opt, start_factor=1.0, end_factor=0.01, total_iters=500)


    def _train_epoch(self, epoch):
        # self.model.train()
        epoch_start = time.time()
        train_loss = 0.
        # running_metric_text = runningScore(2)batch
        lr = self.optimizer.param_groups[0]['lr']

        loss_summary = []  
        self.model.img_encoder.train()
        self.model.prompt_encoder.train()
        self.model.mask_decoder.train()
        for idx, batch in enumerate(self.train_loader):
            # pdb.set_trace()
            # if i >= self.train_loader_len:
            #     break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            
            masks = self.model(batch)

            seg = batch['seg'].unsqueeze(1)
            loss = self.loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
            self.encoder_opt.zero_grad()
            self.decoder_opt.zero_grad()
            self.feature_opt.zero_grad()
            loss.backward()
            logger.info(
                'epoch: {}/{}, iter: {}/{}'.format(epoch, self.max_epoch, idx, len(self.train_loader)) + ": loss:" + str(
                    loss_summary[-1].flatten()[0]))
            torch.nn.utils.clip_grad_norm_(self.model.img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.mask_decoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.prompt_encoder.parameters(), 1.0)
            self.encoder_opt.step()
            self.feature_opt.step()
            self.decoder_opt.step()
        self.encoder_scheduler.step()
        self.feature_scheduler.step()
        self.decoder_scheduler.step()

        train_loss = np.mean(loss_summary)
        logger.info(f"- Train loss: {train_loss}")

        return {'train_acc': None, 'train_loss': train_loss, 'lr': lr, 'time': time.time() - epoch_start, 'epoch': epoch}


    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        # TODO: implement evaluate function
        self.model.img_encoder.eval()
        self.model.prompt_encoder.eval()
        self.model.mask_decoder.eval()
        loss_summary = []
        gt_pos_num = 0
        pred_pos_num = 0
        pos_hit_num = 0
        pos_error_num = 0
        for idx, batch in enumerate(self.validate_loader):
            self.global_step += 1
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            
            masks = self.model(batch)
            seg = batch['seg'].unsqueeze(1)

            # TODO： 使用bbox的mAP评估方法
            # 计算f1 score, 暂不考虑
            masks_softmax = torch.softmax(masks, 1)
            cls_map = masks_softmax.argmax(1)
            roi_pix_num = len(cls_map[cls_map==1])
            neg_pix_num = len(cls_map[cls_map==2])
            roi_gt_num = len(seg[seg==1])
            # neg_gt_num = len(seg[seg==2])
            
            if roi_pix_num > neg_pix_num:
                pred_pos_num += 1
                if roi_gt_num > 0:
                    pos_hit_num += 1
                else:
                    pos_error_num += 1
            
            if roi_gt_num > 0:
                gt_pos_num += 1
            
            loss = self.loss_cal(masks, seg)
            loss_summary.append(loss.detach().cpu().numpy())
        eval_loss = np.mean(loss_summary)
        precision = pos_hit_num / pred_pos_num
        recall = pos_hit_num / gt_pos_num

        f1_score = 2 * precision * recall / (precision + recall)
        logger.info(f"- Eval loss: {eval_loss}, f1 score: {f1_score}, precision: {precision}, recall: {recall}")

        return precision, recall, f1_score


    def _on_epoch_finish(self):
        # self.logger_info('[{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, time: {:.4f}, lr: {}'.format(
        #     self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['train_acc'],
        #     self.epoch_result['time'], self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            save_best = False
            if self.validate_loader is not None:  # 使用f1作为最优模型指标
                pecision, recall, score = self._eval()

                # self.logger_info('test: acc: {:.6f}, score: {:.6f}'.format(acc, score))
                if score > self.metrics['score']:
                    save_best = True
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
                    self.metrics['score'] = score
                    self.metrics['precision'] = pecision
                    self.metrics['recall'] = recall
                self.metrics['train_loss'] = self.epoch_result['train_loss']

            best_str = 'current best, '
            if 'acc' in self.metrics:
                self.metrics['acc']
            for k, v in self.metrics.items():
                best_str += '{}: {:.6f}, '.format(k, v)
            self.logger_info(best_str)
            if save_best:
                import shutil
                shutil.copy(net_save_path, net_save_path_best)
                self.logger_info("Saving current best: {}".format(net_save_path_best))
            else:
                self.logger_info("Saving checkpoint: {}".format(net_save_path))

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))
        
        
        self.logger_info('finish train')

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
