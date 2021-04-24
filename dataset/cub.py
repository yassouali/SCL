import os
import pickle
from PIL import Image
import numpy as np
import json

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CUB(Dataset):
    """support CUB"""
    def __init__(self, args, partition='base', transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.image_size = 84

        if self.partition == 'base':
            self.resize_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
                transforms.RandomCrop(size=84)
            ])
        else:
            self.resize_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
                transforms.CenterCrop(self.image_size)
            ])

        if transform is None:
            if self.partition == 'base' and self.data_aug:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x).copy(),
                    transforms.ToTensor(),
                    self.normalize
                ])
            else:
                self.transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.transform = transform

        self.data = {}
        self.file_pattern = '%s.json'

        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            meta = json.load(f)

        self.imgs = []
        labels = []
        for i in range(len(meta['image_names'])):
            image_path = os.path.join(meta['image_names'][i])
            self.imgs.append(image_path)

            label = meta['image_labels'][i]
            labels.append(label)

        # adjust sparse labels to labels from 0 to n.
        cur_class = 0
        label2label = {}
        for idx, label in enumerate(labels):
            if label not in label2label:
                label2label[label] = cur_class
                cur_class += 1
        new_labels = []
        for idx, label in enumerate(labels):
            new_labels.append(label2label[label])
        self.labels = new_labels
        self.num_classes = np.unique(np.array(self.labels)).shape[0]

    def __getitem__(self, item):
        image_path = self.imgs[item]
        img = Image.open(image_path).convert('RGB')
        img = np.array(img).astype('uint8')
        img = np.asarray(self.resize_transform(img)).astype('uint8')
        img = self.transform(img)
        target = self.labels[item]
        return img, target, item

    def __len__(self):
        return len(self.labels)


class MetaCUB(CUB):
    def __init__(self, args, partition='base', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaCUB, self).__init__(args, partition)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        self.classes = list(self.data.keys())
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples

        self.resize_transform_train = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
            transforms.RandomCrop(size=84)
        ])

        self.resize_transform_test = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.Resize([int(self.image_size*1.15), int(self.image_size*1.15)]),
            transforms.CenterCrop(self.image_size)
        ])

        if train_transform is None:
            self.train_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                lambda x: np.asarray(x).copy(),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                lambda x: Image.fromarray(x),
                transforms.ToTensor(),
                self.normalize
            ])
        else:
            self.test_transform = test_transform

        self.data = {}
        for idx in range(len(self.imgs)):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def _load_imgs(self, img_paths, transform):
        imgs = []
        for image_path in img_paths:
            img = Image.open(image_path).convert('RGB')
            img = np.array(img).astype('uint8')
            img = transform(img)
            imgs.append(np.asarray(img).astype('uint8'))
        return np.asarray(imgs).astype('uint8')

    def __getitem__(self, item):
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs_paths = self.data[cls]
            support_xs_ids_sampled = np.random.choice(range(len(imgs_paths)), self.n_shots, False)

            support_paths = [imgs_paths[i] for i in support_xs_ids_sampled]
            support_imgs = self._load_imgs(support_paths, transform=self.resize_transform_train)
            support_xs.append(support_imgs)
            support_ys.append([idx] * self.n_shots)

            query_xs_ids = np.setxor1d(np.arange(len(imgs_paths)), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_paths = [imgs_paths[i] for i in query_xs_ids]

            query_imgs = self._load_imgs(query_paths, transform=self.resize_transform_test)
            query_xs.append(query_imgs)
            query_ys.append([idx] * query_xs_ids.shape[0])

        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way,))

        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1,)), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)

        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))

        return support_xs, support_ys, query_xs, query_ys

    def __len__(self):
        return self.n_test_runs
