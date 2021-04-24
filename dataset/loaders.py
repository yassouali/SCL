import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list
from dataset.utils import get_contrastive_aug
from dataset.cub import CUB, MetaCUB

DATASETS = ['miniImageNet', 'tieredImageNet', 'CIFAR-FS' , 'FC100', 'cub', 'cars', 'places', 'plantae', 'cross']

class TwoCropTransform:
    """
    Create two crops of the same image
    """
    def __init__(self, transformA, transformB=None):
        self.transformA = transformA
        if transformB is None:
            self.transformB = transformA
        else:
            self.transformB = transformB

    def __call__(self, x):
        return [self.transformA(x), self.transformB(x)]


def get_train_loaders(opt, train_partition, worker_init_fn=None):
    """
    Create the training dataloaders
    """
    if opt.double_transform:
        train_trans_standard, test_trans = transforms_options[opt.transform]
        train_trans_contrast = get_contrastive_aug(dataset=opt.dataset, aug_type=opt.aug_type)
        train_trans = TwoCropTransform(train_trans_standard, train_trans_contrast)
    else:
        train_trans, test_trans = transforms_options[opt.transform]

    # ImagetNet derivatives - miniImageNet
    if opt.dataset == 'miniImageNet':
        assert opt.transform == "A"
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                num_workers=opt.num_workers, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2, worker_init_fn=worker_init_fn)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

    # ImagetNet derivatives - tieredImageNet
    elif opt.dataset == 'tieredImageNet':
        assert opt.transform == "A"
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2, worker_init_fn=worker_init_fn)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

    # CIFAR-100 derivatives - both CIFAR-FS & FC100
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        assert opt.transform == "D" or opt.transform == "Dcontrast"
        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                num_workers=opt.num_workers, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2, worker_init_fn=worker_init_fn)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    
    # For cross-domain experiments we train on all of the sets (train, val and test)
    elif opt.dataset == 'cross':
        assert opt.transform == "A"
        
        train_dataset = ImageNet(args=opt, partition='train', transform=train_trans)
        val_dataset = ImageNet(args=opt, partition='val', transform=train_trans)
        test_dataset = ImageNet(args=opt, partition='test', transform=train_trans)
        
        all_datasets = ConcatDataset([train_dataset, val_dataset, test_dataset])

        train_loader = DataLoader(all_datasets, batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                num_workers=opt.num_workers, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2, worker_init_fn=worker_init_fn)
        n_cls = 64+16+20 # train + val + test

    else:
        raise NotImplementedError(opt.dataset)


    return train_loader, val_loader, n_cls





def get_eval_loaders(opt):
    """
    Create the evaluation dataloaders
    """
    train_trans, test_trans = transforms_options[opt.transform]

    # ImagetNet derivatives - miniImageNet
    if opt.dataset == 'miniImageNet':
        assert opt.transform == "A"
        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test', train_transform=train_trans,
                                    test_transform=test_trans, fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans, fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64

    # ImagetNet derivatives - tieredImageNet
    elif opt.dataset == 'tieredImageNet':
        assert opt.transform == "A"
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                    train_transform=train_trans, test_transform=test_trans, fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans, fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351

    # CIFAR-100 derivatives - both CIFAR-FS & FC100
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        assert opt.transform == "D"
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test', train_transform=train_trans,
                                    test_transform=test_trans, fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans, fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))

    # For cross-domain - we evaluate on a new dataset / domain
    elif opt.dataset in ['cub', 'cars', 'places', 'plantae']:
        train_classes = {'cub': 100, 'cars': 98, 'places': 183, 'plantae': 100}
        assert opt.transform == "C"
        assert not opt.use_trainval, f"Train val option not possible for dataset {opt.dataset}"

        meta_testloader = DataLoader(MetaCUB(args=opt, partition='novel',
                                    train_transform=train_trans, test_transform=test_trans, fix_seed=False),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCUB(args=opt, partition='val', train_transform=train_trans,
                                    test_transform=test_trans, fix_seed=False),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        n_cls = train_classes[opt.dataset]

    else:
        raise NotImplementedError(opt.dataset)

    return meta_testloader, meta_valloader, n_cls
