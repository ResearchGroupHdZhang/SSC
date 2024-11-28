from torch.utils.data import Dataset
from PIL import Image
# import cv2
import os
import numpy as np
from glob import glob
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import torch
import math
import torch.utils.data as data
NUM_DATASET_WORKERS = 8
SCALE_MIN = 0.75
SCALE_MAX = 0.95
import random
from torch.utils.data import DataLoader

class HR_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self,):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class IMG_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config,data):
        self.imgs = data.imgs
        # for dir in data_dir:
        #     self.imgs += glob(os.path.join(dir, '*.jpg'))
        #     self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self,):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path,_ = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed 

    def __len__(self):
        return len(self.imgs)

class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()


    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img
    def __len__(self):
        return len(self.imgs)

class CIFAR10(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.len = dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item % self.len)[0]

    def __len__(self):
        return self.len   #为什么这里要乘10呢？
        # return self.len
class BatchIndexDataLoader(DataLoader):
    def __init__(self, *args, snr_values,**kwargs):
        super().__init__(*args, **kwargs)
        self.snr_values = snr_values
        self.batch_idx = 0

    def __iter__(self):
        self.batch_idx = 0
        for batch in super().__iter__():
            yield collate_fn(batch, self.snr_values,self.batch_idx)
            self.batch_idx += 1


class BatchIndexDataLoader_mul(DataLoader):
    def __init__(self, *args, snr_values1,snr_values2, **kwargs):
        super().__init__(*args, **kwargs)
        self.snr_values1 = snr_values1
        self.snr_values2 = snr_values2
        self.batch_idx = 0

    def __iter__(self):
        self.batch_idx = 0
        for batch in super().__iter__():
            yield collate_fn_mul(batch, self.snr_values1,self.snr_values2, self.batch_idx)
            self.batch_idx += 1
def collate_fn_mul(batch, snr_values1,snr_values2, batch_idx):
    snr1 = snr_values1[batch_idx]
    snr2 = snr_values2[batch_idx]
    return batch,  snr1,snr2
def generate_snr_values_for_batch( low, high):
    random.seed(42)  # 固定随机种子以保证生成相同的 SNR 值
    return random.uniform(low, high)
def generate_snr_values(num_batches, low, high, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, num_batches)
def collate_fn(batch, snr_values, batch_idx):
    snr = snr_values[batch_idx]
    return batch,  snr
def collate_fn_multi(batch, low1, high1,low2,high2):
    snr1 = generate_snr_values_for_batch( low1, high1)
    snr2 = generate_snr_values_for_batch( low2, high2)
    images = batch
    images = torch.stack(images)
    return images,  snr1, snr2
def get_loader_multi(args, config,low1,high1,low2,high2):
    if args.trainset == 'DIV2K':
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)
    elif args.trainset == 'CIFAR10':
        batch_size_test = 1024
        dataset_ = datasets.CIFAR10
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
        train_dataset = dataset_(root=config.train_data_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=False)

        test_dataset = dataset_(root=config.test_data_dir,
                                train=False,
                                transform=transform_test,
                                download=False)

        train_dataset = CIFAR10(train_dataset)
        test_dataset = CIFAR10(test_dataset)

    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        ])
        train_dataset = datasets.ImageFolder(root='/home/zhoujh/code/train_out', transform=transform)
        train_dataset = IMG_image(config,train_dataset)
        test_dataset = Datasets(config.test_data_dir)
        batch_size_test = 1
    batch_size_train = config.batch_size
    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)
    num_train_batches = len(train_dataset) // batch_size_train + (1 if len(train_dataset) % batch_size_train != 0 else 0)
    num_test_batches = len(test_dataset) // batch_size_test + (1 if len(test_dataset) % batch_size_test != 0 else 0)
    train_snr_values1 = generate_snr_values(num_train_batches, low1, high1, seed=42)
    test_snr_values1 = generate_snr_values(num_test_batches, low1, high1, seed=42 + 1)  # 使用不同的种子
    train_snr_values2 = generate_snr_values(num_train_batches, low2, high2, seed=42)
    test_snr_values2 = generate_snr_values(num_test_batches, low2, high2, seed=42 + 1)  # 使用不同的种子
    train_loader = BatchIndexDataLoader_mul(
    dataset=train_dataset,
    snr_values1=train_snr_values1,
    snr_values2=train_snr_values2,
    num_workers=NUM_DATASET_WORKERS,
    pin_memory=True,
    batch_size=batch_size_train,
    worker_init_fn=worker_init_fn_seed,
    shuffle=True,
    drop_last=True
    )
    if args.trainset == 'CIFAR10':
        test_loader = BatchIndexDataLoader_mul(
        dataset=test_dataset,
        snr_values1=test_snr_values1,
        snr_values2=test_snr_values2,
        batch_size=batch_size_test,
        shuffle=False)

    else:
        test_loader = BatchIndexDataLoader_mul(
        dataset=test_dataset,
        snr_values1=test_snr_values1,
        snr_values2=test_snr_values2,
        batch_size=batch_size_test,
        shuffle=False
    )

    return train_loader, test_loader



def get_loader(args, config,low,high):
    if args.trainset == 'DIV2K':
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)
        batch_size_test = 1
    elif args.trainset == 'CIFAR10':
        batch_size_test = 1024
        dataset_ = datasets.CIFAR10
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
        train_dataset = dataset_(root=config.train_data_dir,
                                 train=True,
                                 transform=transform_train,
                                 download=False)

        test_dataset = dataset_(root=config.test_data_dir,
                                train=False,
                                transform=transform_test,
                                download=False)

        train_dataset = CIFAR10(train_dataset)
        test_dataset = CIFAR10(test_dataset)

    else:
        transform = transforms.Compose([
        transforms.ToTensor(),
        ])
        train_dataset = datasets.ImageFolder(root='/home/zhoujh/code/train_out', transform=transform)
        train_dataset = IMG_image(config,train_dataset)
        test_dataset = Datasets(config.test_data_dir)
        batch_size_test = 1
    batch_size_train = config.batch_size
    
    num_train_batches = len(train_dataset) // batch_size_train + (1 if len(train_dataset) % batch_size_train != 0 else 0)
    num_test_batches = len(test_dataset) // batch_size_test + (1 if len(test_dataset) % batch_size_test != 0 else 0)
    train_snr_values = generate_snr_values(num_train_batches, low, high, seed=42)
    test_snr_values = generate_snr_values(num_test_batches, low, high, seed=42 + 1)  # 使用不同的种子
    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
    #                                            num_workers=NUM_DATASET_WORKERS,
    #                                            pin_memory=True,
    #                                            batch_size=config.batch_size,
    #                                            worker_init_fn=worker_init_fn_seed,
    #                                            shuffle=True,
    #                                            drop_last=True,collate_fn=lambda batch: collate_fn(batch, low, high))
    train_loader = BatchIndexDataLoader(
    dataset=train_dataset,
    snr_values=train_snr_values,
    num_workers=NUM_DATASET_WORKERS,
    pin_memory=True,
    batch_size=batch_size_train,
    worker_init_fn=worker_init_fn_seed,
    shuffle=True,
    drop_last=True
    )

    
    if args.trainset == 'CIFAR10':
        test_loader = BatchIndexDataLoader(
        dataset=test_dataset,
        snr_values=test_snr_values,
        batch_size=batch_size_test,
        shuffle=False
    )

    else:
        test_loader = BatchIndexDataLoader(
        dataset=test_dataset,
        snr_values=test_snr_values,
        batch_size=batch_size_test,
        shuffle=False
    )

    return train_loader, test_loader

