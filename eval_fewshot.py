import os, argparse, time
import sys, random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np

from models import model_dict, model_pool
from models.util import create_model
from models.contrast import Projection, ContrastResNet
from models.attention import AttentionSimilarity

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list
from dataset.loaders import get_eval_loaders, DATASETS

from eval.meta_eval import meta_test_torch, meta_test_scikit_learn

def parse_option():
    parser = argparse.ArgumentParser('Argument for few-shot evaluation.')

    # load pretrained model
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--model_path', type=str, default=None, help='Absolute path to .pth model')

    # dataset
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=DATASETS)
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='Use trainval set')

    # meta setting
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N', help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')
    parser.add_argument('--data_root', type=str, default='/workdir/oualiy/Datasets/', metavar='N', help='Root dataset')
    parser.add_argument('--num_workers', type=int, default=3, metavar='N', help='Number of workers for dataloader')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size', help='Size of test batch)')

    # evaluation setting & parameters
    parser.add_argument('--use_torch_impl', action='store_true', help='Use a linear classifier impl. in pytorch instead of scikit learn')
    parser.add_argument('--use_spatial_feat', action='store_true', help='When using torch impl, use spatial features to train the classifier')
    parser.add_argument('--use_global_feat', action='store_true', help='When using torch impl, use global features to train the classifier')
    parser.add_argument('--aggregation', default="sum", choices=["max", "sum"], help='How to aggregate the logits when using spa. and glob. feature')
    parser.add_argument('--cross', action='store_true', help='Cross domain evaluation')
    parser.add_argument('--weight_inprint', action='store_true', help='Using class prototypes to initialize the weights of the classifier')

    opt = parser.parse_args()

    # set data augmentation type
    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'
    elif opt.dataset in ['cub', 'cars', 'places', 'plantae']:
        opt.transform = 'C'

    # set the data path
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    return opt


def main_eval(opt):

    # load mode
    ckpt = torch.load(opt.model_path)

    # check if the training dataset matches the eval dataset
    if not opt.cross:
        assert ckpt['opt'].dataset == opt.dataset , "Model trained on a different dataset."
    else:
        assert ckpt['opt'].dataset == "cross"

    # load model
    model = ContrastResNet(ckpt['opt'], ckpt['opt'].n_cls)
    model.load_state_dict(ckpt['model'])


    # Set cuda params 
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # eval and testing sets
    meta_testloader, meta_valloader, n_cls = get_eval_loaders(opt)
    
    # validation
    print(f"Validation ...")

    if opt.use_torch_impl:
        val_acc_feat, val_std_feat = meta_test_torch(model, testloader=meta_valloader, opt=opt)
    else:
        val_acc_feat, val_std_feat = meta_test_scikit_learn(model, testloader=meta_valloader, opt=opt)

    val_acc_feat, val_std_feat = np.round(val_acc_feat*100, 2), np.round(val_std_feat*100, 2)
    print(f"    Validation accuracy: {val_acc_feat} +/- {val_std_feat}\n\n")

    # testing
    print(f"Testing ...")

    if opt.use_torch_impl:
        test_acc_feat, test_std_feat = meta_test_torch(model, testloader=meta_testloader, opt=opt)
    else:
        test_acc_feat, test_std_feat = meta_test_scikit_learn(model, testloader=meta_testloader, opt=opt)

    test_acc_feat, test_std_feat = np.round(test_acc_feat*100, 2), np.round(test_std_feat*100, 2)

    print(f"    Test accuracy: {test_acc_feat} +/- {test_std_feat}")
    
    return test_acc_feat

if __name__ == '__main__':
    opt = parse_option()
    main_eval(opt)


