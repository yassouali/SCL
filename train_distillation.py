import os, argparse
import time, sys, math, random

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from models import model_dict, model_pool
from models.contrast import ContrastResNet
from models.util import create_model, get_teacher_name

from utils import adjust_learning_rate, accuracy, AverageMeter, warmup_learning_rate, set_seed
from dataset.loaders import get_train_loaders
from dataset.transform_cfg import transforms_options, transforms_list

from losses import DistillKL, contrast_distill
from tqdm import tqdm
from dataset.transform_cfg import transforms_list
from copy import deepcopy
from dataset.utils import AUG_TYPES

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # general
    parser.add_argument('--eval_freq', type=int, default=100, help='meta-eval frequency')
    parser.add_argument('--save_freq', type=int, default=500, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--tb_freq', type=int, default=100, help='tb frequency')
    parser.add_argument('--use_tb', default=False, action='store_true')
    parser.add_argument('--syncBN', action='store_true', help='using synchronized batch normalization')
    parser.add_argument('--trial', type=str, default=None, help='the experiment id')
    parser.add_argument('--seed', type=int, default=31)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=5e-2, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default=None, help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # dataset and model
    parser.add_argument('--model_s', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_t', type=str, default=None, choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100', 'cross'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
    parser.add_argument('--aug_type', type=str, default='simclr', choices=AUG_TYPES)

    # path to teacher model
    parser.add_argument('--model_path_t', type=str, default=None, help='teacher model snapshot')

    # weights of the total loss
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--lambda_cls', default=0., type=float, help='weight for classification')
    parser.add_argument('--lambda_KD', default=0., type=float, help='weight balance for KL div loss')
    parser.add_argument('--lambda_contrast_g', default=0., type=float, help='weight balance for contrastive loss')
    parser.add_argument('--lambda_contrast_s', default=0., type=float, help='weight balance for contrastive loss')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--double_transform', action='store_true')

    # setting for meta-learning
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')

    opt = parser.parse_args()
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
    
    if opt.model_t is None:
        opt.model_t = opt.model_s

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_distilled'
    if not opt.tb_path and opt.use_tb:
        opt.tb_path = './tensorboard'

    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
        if opt.dataset == "cross":
            opt.data_root = opt.data_root.replace("cross", "miniImageNet")
    opt.data_aug = True

    # learning rate decay
    if opt.lr_decay_epochs is None:
        decay_steps = opt.epochs // 10
        opt.lr_decay_epochs = [opt.epochs - 3*decay_steps, opt.epochs - 2*decay_steps, opt.epochs - decay_steps] 
    else:
        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

    # set model name
    if opt.model_name is None:
        if opt.use_trainval:
            opt.trial = opt.trial + '_trainval'

        opt.model_name = 'S:{}_T:{}_{}_trans_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.transform)
        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)

        if opt.trial is not None:
            opt.model_name = '{}_{}'.format(opt.model_name, opt.trial)

    if opt.use_tb:
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()

    return opt


def load_teacher(model_path, n_cls):
    """load the teacher model"""
    print('==> loading teacher model')

    ckpt = torch.load(model_path)
    opt = ckpt['opt']

    model = ContrastResNet(opt, n_cls)
    model.load_state_dict(ckpt['model'])

    print('==> done')
    return model, opt


def main():
    opt = parse_option()
    set_seed(opt.seed)

    # tensorboard logger
    if opt.use_tb:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    train_loader, val_loader, n_cls = get_train_loaders(opt, train_partition)
    opt.n_cls = n_cls

    # teacher
    opt.model = opt.model_t
    model_t, ckpt_opt = load_teacher(opt.model_path_t, n_cls)
    ckpt_opt.data_root = opt.data_root
    assert ckpt_opt.dataset == opt.dataset, "The teacher is trained on a different dataset."

    # student
    opt.model = opt.model_s
    model_s = ContrastResNet(ckpt_opt, n_cls)

    # losses
    criterion_cls = nn.CrossEntropyLoss()
    criterion_contrast = contrast_distill
    criterion_div = DistillKL(opt.kd_T)

    # optimizer
    params = [{'params': model_s.parameters()}]
    optimizer = optim.SGD(params, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # Set cuda params
    if opt.syncBN:
        model_t = apex.parallel.convert_syncbn_model(model_t)
        model_s = apex.parallel.convert_syncbn_model(model_s)

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model_t = nn.DataParallel(model_t)
            model_s = nn.DataParallel(model_s)
        model_t = model_t.cuda()
        model_s = model_s.cuda()
        criterion_cls = criterion_cls.cuda()
        criterion_div = criterion_div.cuda()
        cudnn.benchmark = True

    criterion_list = [criterion_cls, criterion_contrast, criterion_div]

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # distillation routine
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)

        time1 = time.time()
        train_loss = train(epoch, train_loader, model_t, model_s, criterion_list, optimizer, opt, ckpt_opt)
        time2 = time.time()

        print('epoch: {}, total time: {:.2f}, train loss: {:.3f}'.format(epoch, time2 - time1, train_loss))

        if opt.use_tb and (epoch % opt.tb_freq) == 0:
            logger.log_value('train_loss', train_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': ckpt_opt,
                'model': model_s.state_dict() if opt.n_gpu <= 1 else model_s.module.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': ckpt_opt,
        'model': model_s.state_dict() if opt.n_gpu <= 1 else model_s.module.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_student_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


def train(epoch, train_loader, model_t, model_s, criterion_list, optimizer, opt, ckpt_opt):
    """One epoch training"""
    model_s.train()
    model_t.eval()

    criterion_cls, criterion_contrast, criterion_div = criterion_list

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, loss_kd, loss_cont = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    tbar = tqdm(train_loader, ncols=130)

    for idx, (input, target, _) in enumerate(tbar):
        # fetch data
        data_time.update(time.time() - end)

        # send to gpu
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            if opt.double_transform:
                input = torch.cat([input[0].cuda(non_blocking=True).float(), 
                                    input[1].cuda(non_blocking=True).float()], dim=0)
            else:
                input = input.cuda(non_blocking=True).float()

        bz = target.size(0)

        # ===================forward=====================
        logits, spatial_f, global_f, avg_pool_feat = model_s(input)
        with torch.no_grad():
            logits_t, spatial_f_t, _, avg_pool_feat_t = model_t(input)
            logits_t, avg_pool_feat_t = logits_t.detach(), avg_pool_feat_t.detach()
            spatial_f_t = spatial_f_t.detach()

        # ===================losses================
        # losses - KL & CE
        loss_cls = criterion_cls(logits[:bz], target)
        loss_div = criterion_div(logits, logits_t)
        
        # losses - contrastive distillation - global
        loss_contrast_global = criterion_contrast(avg_pool_feat, avg_pool_feat_t)

        # losses - contrastive distillation - spatial
        B, C, H, W = spatial_f_t.size()
        
        spatial_f = spatial_f.view(B, C, H*W).permute(0, 2, 1).contiguous()
        spatial_f = spatial_f.view(B*H*W, C)
        
        spatial_f_t = spatial_f_t.view(B, C, H*W).permute(0, 2, 1).contiguous()
        spatial_f_t = spatial_f_t.view(B*H*W, C)

        loss_contrast_spatial = criterion_contrast(spatial_f, spatial_f_t)
        
        # total loss
        loss_contrast = opt.lambda_contrast_g * loss_contrast_global + opt.lambda_contrast_s * loss_contrast_spatial
        loss = opt.lambda_cls * loss_cls + opt.lambda_KD * loss_div + loss_contrast

        # ===================update losses================
        losses.update(loss.item())
        loss_kd.update(loss_div.item())
        loss_cont.update(loss_contrast.item())

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        tbar.set_description('Epoch: [{0}] Loss {losses.avg:.3f} - L kd {loss_kd.avg:.3f} L cont {loss_cont.avg:.3f}'
                                .format(epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                                        losses=losses,loss_kd=loss_kd, loss_cont=loss_cont))

    return losses.avg


if __name__ == '__main__':
    main()
