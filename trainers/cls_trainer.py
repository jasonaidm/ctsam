# -*- coding: utf-8 -*-
import time

import torch
from tqdm import tqdm
import numpy as np
from trainers.base_trainer import BaseTrainer
from sklearn.metrics import precision_recall_fscore_support
from utils import WarmupPolyLR
import pdb
from loguru import logger


class ClsTrainer(BaseTrainer):
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
        self.metric_leader = config['metric'].get('leader', None)
        self.do_metric_weighting = config['metric'].get('do_weighting', None)
        
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

        
    def _train_epoch(self, epoch):
        self.model.train()
        if 'freeze' in self.config['trainer']:
            self.model.apply(self.fix_bn)  # fix batchnorm
        epoch_start = time.time()
        batch_start = time.time()
        short_run_loss = 0.
        train_loss = 0.
        short_run_acc = 0.
        train_acc = 0.
        # running_metric_text = runningScore(2)batch
        lr = self.optimizer.param_groups[0]['lr']

        for i, batch in enumerate(self.train_loader):
            # pdb.set_trace()
            # if i >= self.train_loader_len:
            #     break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']
            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]
            outputs = self.model(batch['img'])
            if self.config['loss']['type'] == 'BCEWithLogitsLoss':
                loss = self.criterion(outputs, torch.nn.functional.one_hot(batch['label'],self.config['arch']['head']['num_classes']).float())
            else:
                loss = self.criterion(outputs, batch['label'])
                
            # loss = self.criterion(outputs, batch['label'])
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                self.scheduler.step()

            # cal loss and acc
            total_loss = loss.item()
            short_run_loss += total_loss
            train_loss += total_loss
            pred_indexes = outputs.argmax(1)
            pred_indexes = pred_indexes.detach().cpu().numpy().tolist()
            acc = self.metric_cls(pred_indexes, batch['label'])
            short_run_acc += acc
            train_acc += acc
            if self.global_step % self.log_iter == 0:
                short_run_loss /= self.log_iter
                short_run_acc /= self.log_iter
                loss_str = 'total_loss: {:.4f}, '.format(short_run_loss)
                batch_time = time.time() - batch_start
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, {}, '
                    'acc: {:.2f}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step,
                        self.log_iter * cur_batch_size / batch_time, loss_str, short_run_acc, lr, batch_time))
                batch_start = time.time()
                short_run_loss = 0.
                short_run_acc = 0.

        return {'train_acc': train_acc / self.train_loader_len, 'train_loss': train_loss / self.train_loader_len,
                'lr': lr, 'time': time.time() - epoch_start, 'epoch': epoch}

    @torch.no_grad()
    def _eval(self):
        self.model.eval()
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        total_frame = 0.0
        total_time = 0.0
        total_preds = []
        total_labels = []
        for batch in tqdm(self.validate_loader, total=len(self.validate_loader), desc='test model'):
            start = time.time()
            batch['img'] = batch['img'].to(self.device)
            # batch['labels'] = [[int(i.numpy()[0]) for i in batch['labels']]]
            outputs = self.model(batch['img'])
            # _, pred_index = preds.max(2)
            # pdb.set_trace()
            pred_indexes = outputs.argmax(1)
            pred_indexes = pred_indexes.detach().cpu().numpy().tolist()
            total_preds.extend(pred_indexes)
            total_labels.extend(batch['label'].tolist())
            total_frame += batch['img'].size()[0]
            total_time += time.time() - start
        acc = self.metric_cls(total_preds, total_labels)
        if self.metric_leader:
            precision, recall, fscore, _ = precision_recall_fscore_support(total_labels, total_preds)
            self.logger_info("### Each fscore: {}; average fscore: {:.3f}".format(fscore, fscore.mean()))
            leader_precision = precision[self.metric_leader]
            leader_recall = recall[self.metric_leader]
            leader_fscore = fscore[self.metric_leader]
            leader_weighted_score = 0.7 * leader_precision + 0.3 * leader_recall
            self.logger_info("test leader precision: {:.3f};leader recall: {:.3f};leader_fscore: {:.3f}; leader_weighted_score: {:.3f}".format(
                leader_precision, leader_recall, leader_fscore, leader_weighted_score)
                             )
            fscore = leader_weighted_score if self.do_metric_weighting else leader_fscore
                
        else:
            precision, recall, fscore, _ = precision_recall_fscore_support(total_labels, total_preds, average='macro')
            self.logger_info("test average precision: {:.3f};average recall: {:.3f}".format(precision, recall))
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        return acc, fscore

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['train_acc'],
            self.epoch_result['time'], self.epoch_result['lr']))
        net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
        net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)

        if self.config['local_rank'] == 0:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            save_best = False
            if self.validate_loader is not None:  # 使用f1作为最优模型指标
                acc, score = self._eval()

                if self.tensorboard_enable:
                    self.writer.add_scalar('EVAL/fscore', score, self.global_step)
                self.logger_info('test: acc: {:.6f}, score: {:.6f}'.format(acc, score))
                if score > self.metrics['score']:
                    save_best = True
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
                    self.metrics['score'] = score
                    self.metrics['acc'] = acc
                self.metrics['train_loss'] = self.epoch_result['train_loss']

            best_str = 'current best, '
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
