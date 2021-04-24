from dataset.rand_augment import rand_augment_transform
from PIL import ImageFilter
from dataset import auto_augment
import random
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn 
import torch

mean_cifar = [0.5071, 0.4867, 0.4408]
std_cifar = [0.2675, 0.2565, 0.2761]

mean_imagenet = [120.39586422 / 255.0, 115.59361427 / 255.0, 104.54012653 / 255.0]
std_imagenet = [70.68188272 / 255.0, 68.27635443 / 255.0, 72.54505529 / 255.0]

MEAN_STD = {
    "imagenet": (mean_imagenet, std_imagenet),
    "cifar": (mean_cifar, std_cifar),
}

AUG_TYPES = ["standard", "simclr", "stacked_randaug", "autoaugment"]

class GaussianBlur(object):
    """
    blur a single image on CPU
    """
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img


def get_contrastive_aug(dataset, aug_type="simclr"):

    if dataset == "miniImageNet" or dataset == "tieredImageNet" or dataset == "cross":
        mean, std = MEAN_STD["imagenet"]
        crop_size = 84
    elif dataset == "cross_large":
        mean, std = MEAN_STD["mini_large"]
        crop_size = 224
    elif dataset == "CIFAR-FS" or dataset == "FC100":
        mean, std = MEAN_STD["cifar"]
        crop_size = 32
    else:
        raise NotImplementedError('dataset not found: {}'.format(dataset))

    normalize = transforms.Normalize(mean=mean, std=std)

    if aug_type == 'standard':
        # used in the standard supervised setting
        train_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == 'simclr':
        # used in MoCoV2, SimCLR
        train_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    elif aug_type == 'stacked_randaug':
        # used in InfoMin
        kernel_size = crop_size // 10
        ra_params = dict(
            translate_const=int(crop_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),)

        train_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomApply([GaussianBlur(kernel_size)], p=0.5),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ])

    elif aug_type == 'autoaugment':
        # used in AutoAugment
        auto_augment.IMAGE_SIZE = crop_size
        train_transform = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            auto_augment.AutoAugment(crop_size),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise NotImplementedError('transform not found: {}'.format(aug_type))
        
    return train_transform



