import os, argparse, random
import time, sys, math
from tqdm import tqdm
import numpy as np

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_pool
from models.contrast import ContrastResNet
from models.attention import AttentionSimilarity

from utils import adjust_learning_rate, accuracy, AverageMeter, warmup_learning_rate, write_results, set_seed
from losses import ContrastiveLoss

from dataset.transform_cfg import transforms_list
from dataset.loaders import get_train_loaders
from dataset.utils import AUG_TYPES
from copy import deepcopy

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # general
    parser.add_argument('--eval_freq', type=int, default=200, help='meta-eval frequency')
    parser.add_argument('--save_freq', type=int, default=400, help='save frequency')
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
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100', 'cross'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')
    parser.add_argument('--aug_type', type=str, default='simclr', choices=AUG_TYPES)

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')
    parser.add_argument('--model_name', type=str, default=None, help='model name')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=1000, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')

    # contrastive learning
    parser.add_argument('--feat_dim', default=80, type=int, help='the dimension of the projection features')
    parser.add_argument('--temperature_s', type=float, default=10.0, help='temperature for spatial contrastive loss')
    parser.add_argument('--temperature_g', type=float, default=10.0, help='temperature for global contrastive loss')
    parser.add_argument('--aggregation', type=str, default="mean", choices=["mean", "max", "sum", "logsum"],
                            help='the aggregation function used to compute the total similarity')

    # loss weights
    parser.add_argument('--lambda_cls', default=0., type=float)
    parser.add_argument('--lambda_global', default=1., type=float)
    parser.add_argument('--lambda_spatial', default=1., type=float)

    # contrastive loss
    parser.add_argument('--spatial_cont_loss', action='store_true', help='contrast spatial features')
    parser.add_argument('--global_cont_loss', action='store_true', help='contrast global features')
    parser.add_argument('--similarity_measure', type=str, default='cosine', choices=['cosine', 'mse'],
                            help='similarity measure used in the contrastive loss')
    parser.add_argument('--use_selfsup_loss', action='store_true', help='use the standard unsupervised contrastive loss')
    parser.add_argument('--double_transform', action='store_true')


    # parse & define standard parameters
    opt = parser.parse_args()
    opt.n_gpu = torch.cuda.device_count()
    opt.data_aug = True
    
    # apply the augmentations two times over a single batch (N inputs -> 2N aug inputs)
    opt.double_transform = True if opt.spatial_cont_loss or opt.global_cont_loss else False

    # set transforms
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    # set the paths
    if not opt.model_path:
        opt.model_path = './models_pretrained'

    if not opt.tb_path and opt.use_tb:
        opt.tb_path = './tensorboard'

    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
        if opt.dataset == "cross":
            opt.data_root = opt.data_root.replace("cross", "miniImageNet")
    
    # set the model name
    if opt.model_name is None:
        if opt.use_trainval:
            opt.trial = opt.trial + '_trainval'

        opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate, opt.weight_decay, opt.transform)
        if opt.cosine:
            opt.model_name = '{}_cosine'.format(opt.model_name)
        if opt.adam:
            opt.model_name = '{}_useAdam'.format(opt.model_name)
        if opt.warm:
            opt.model_name = '{}_warm'.format(opt.model_name)

    if opt.trial is not None:
        opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    # learning rate decay
    if opt.lr_decay_epochs is None:
        decay_steps = opt.epochs // 10
        opt.lr_decay_epochs = [opt.epochs - 3*decay_steps, opt.epochs - 2*decay_steps, opt.epochs - decay_steps] 
    else:
        iterations = opt.lr_decay_epochs.split(',')
        opt.lr_decay_epochs = list([])
        for it in iterations:
            opt.lr_decay_epochs.append(int(it))

    # warm-up for large-batch training
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    # create save folders
    if opt.use_tb:
        opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
        if not os.path.isdir(opt.tb_folder):
            os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    opt = parse_option()
    set_seed(opt.seed)

    print("Starting Training ..... \n\n")

    # create dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    train_loader, val_loader, n_cls = get_train_loaders(opt, train_partition)
    opt.n_cls = n_cls

    # CE loss
    criterion_cls = nn.CrossEntropyLoss()

    # create model
    model = ContrastResNet(opt, n_cls)

    # training parameters
    params = [{'params': model.encoder.parameters(), 'lr': opt.learning_rate}]

    # spatial contrastive loss
    if opt.spatial_cont_loss:
        attention = AttentionSimilarity(hidden_size=model.encoder.feat_dim, inner_size=opt.feat_dim, aggregation=opt.aggregation)
        params = params + [{'params': attention.parameters(), 'lr': opt.learning_rate}]

        criterion_contrast_spatial = ContrastiveLoss(temperature=opt.temperature_s)
    else:
        attention, criterion_contrast_spatial = None, None

    # global contrastive loss
    if opt.global_cont_loss:
        params = params + [{'params': model.head.parameters(), 'lr': opt.learning_rate}]

        criterion_contrast = ContrastiveLoss(temperature=opt.temperature_g)
    else:
        criterion_contrast = None

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimizer = optim.SGD(params, lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)

    # Set cuda params 
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion_cls = criterion_cls.cuda()
        if opt.global_cont_loss:
            criterion_contrast = criterion_contrast.cuda()
        if opt.spatial_cont_loss:
            attention = attention.cuda()
            criterion_contrast_spatial = criterion_contrast_spatial.cuda()
        cudnn.benchmark = True
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)

    # tensorboard
    if opt.use_tb:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # training routine
    for epoch in range(1, opt.epochs + 1):

        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)

        time1 = time.time()
        train_loss = train(epoch, train_loader, model, criterion_cls, criterion_contrast, criterion_contrast_spatial, attention, optimizer, opt)
        time2 = time.time()

        print('epoch: {}, total time: {:.2f}, train loss: {:.3f}'.format(epoch, time2 - time1, train_loss))

        if opt.use_tb and (epoch % opt.tb_freq) == 0:
            logger.log_value('train_loss', train_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': opt,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                'attention': attention.state_dict() if opt.spatial_similarity else None
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
        'attention': attention.state_dict() if opt.spatial_cont_loss else None
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


def train(epoch, train_loader, model, criterion_cls, criterion_contrast, criterion_contrast_spatial, attention, optimizer, opt):
    """
    One training epoch
    """

    model = model.train()
    if attention is not None:
        attention = attention.train()

    batch_time, data_time = AverageMeter(), AverageMeter()
    losses, loss_spa, loss_glo, loss_ce = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    tbar = tqdm(train_loader, ncols=130)

    # training lab
    for idx, (input, target, indices) in enumerate(tbar):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            if opt.double_transform:
                input = torch.cat([input[0].cuda(non_blocking=True).float(), 
                                    input[1].cuda(non_blocking=True).float()], dim=0)
            else:
                input = input.cuda(non_blocking=True).float()

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # ===================forward=====================
        bsz = target.shape[0]
        outputs, spatial_f, global_f, avg_pool_feat = model(input)

        # ===================Losses=====================

        # standard CE loss
        if opt.double_transform:
            loss_cls = (criterion_cls(outputs[:bsz], target) + criterion_cls(outputs[bsz:], target)) / 2.
        else:
            loss_cls = criterion_cls(outputs, target)

        # ignore labels for the self-supervised formulation
        labels = None if opt.use_selfsup_loss else target

        # compute global contrastive loss
        if opt.global_cont_loss:
            loss_contrast_global = criterion_contrast(global_f, labels=labels)
        else:
            loss_contrast_global = torch.zeros_like(loss_cls)

        # compute spatialcontrastive loss
        if opt.spatial_cont_loss:
            loss_contrast_spatial = criterion_contrast_spatial(spatial_f, labels=labels, attention=attention)
        else:
            loss_contrast_spatial = torch.zeros_like(loss_cls)

        # compute the total loss
        loss = loss_contrast_global * opt.lambda_global + loss_contrast_spatial * opt.lambda_spatial +  opt.lambda_cls * loss_cls

        # update the losses
        losses.update(loss.item())
        loss_glo.update(loss_contrast_global.item())
        loss_spa.update(loss_contrast_spatial.item())
        loss_ce.update(loss_cls.item())

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters & print=====================
        batch_time.update(time.time() - end)
        end = time.time()

        tbar.set_description('Epoch: [{0}] Loss {losses.avg:.3f} | Lce {loss_ce.avg:.3f} - Lgl {loss_glo.avg:.3f} - '
                            'Lsp {loss_spa.avg:.3f}'.format(epoch, losses=losses, loss_ce=loss_ce,
                            loss_spa=loss_spa, loss_glo=loss_glo))
        
    return losses.avg


if __name__ == '__main__':
    main()
