from __future__ import print_function
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from torch import nn
import torch.nn.functional as F
import random
from PIL import ImageFilter

"""
ImageNet style transformation
"""

mean_imagenet = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std_imagenet = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]
normalize_imagenet = transforms.Normalize(mean=mean_imagenet, std=std_imagenet)

transform_A = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(84, padding=8),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x).copy(),
        transforms.ToTensor(),
        normalize_imagenet
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_imagenet
    ])
]


"""
CIFAR style transformation
"""

mean_cifar = [0.5071, 0.4867, 0.4408]
std_cifar = [0.2675, 0.2565, 0.2761]
normalize_cifar100 = transforms.Normalize(mean=mean_cifar, std=std_cifar)

### -> Supervised transforms

transform_D = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x).copy(),
        transforms.ToTensor(),
        normalize_cifar100
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        normalize_cifar100
    ])
]


"""
CUB / CARS / places / plantae style transformation
"""

mean_cub = [0.485, 0.456, 0.406]
std_cub = [0.229, 0.224, 0.225]
normalize_cub = transforms.Normalize(mean=mean_cub, std=std_cub)

### -> Supervised transforms
        
transform_C = [
    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(),
        lambda x: np.asarray(x).copy(),
        transforms.ToTensor(),
        # normalize_cub
        normalize_imagenet
    ]),

    transforms.Compose([
        lambda x: Image.fromarray(x),
        transforms.ToTensor(),
        # normalize_cub
        normalize_imagenet
    ])
]

"""
All possible transorms
"""

transforms_list = ['A', 'D', 'C']

transforms_options = {
    'A': transform_A,
    'D': transform_D,
    'C': transform_C,
}
