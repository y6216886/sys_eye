import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import transform as auxtransform
import os
from mydataloader import Myloader


class DataLoader(object):
    def __init__(self, dataset, data_path, label_path, batch_size, n_threads=4, ten_crop=False, dataset_ratio=1.0):
        self.dataset = dataset
        self.data_path = data_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        
        if self.dataset == "asoct":
            print("|===>Creating %s Data Loader" % (self.dataset))
            self.train_loader, self.test_loader = self.asoct_data(data_path=self.data_path, label_path=self.label_path)
        elif self.dataset == "validation":
            print("|===>Creating %s Data Loader" % (self.dataset))
            # self.val_loader = self.palmdata(dataset=self.dataset, data_path=self.data_path)
            self.val_loader = self.asoct_data_val(data_path=self.data_path, label_path=self.label_path)
        else:
            assert False, "invalid data set"


    def getloader(self):
        if self.dataset == "validation":
            return self.val_loader
        else:
            return self.train_loader, self.test_loader

    def asoct_data_val(self, data_path, label_path):
        imgSize = 512
        # test_dir = data_path + "test.txt"
        test_dir = data_path + "trainval.txt"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        test_transform = transforms.Compose([
                # transforms.Scale(600),
                transforms.Scale(520),
                transforms.CenterCrop(imgSize),
                transforms.ToTensor(),
                normalize,
            ])
        test_loader = torch.utils.data.DataLoader(
            Myloader(test_dir, label_path, test_transform),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False)
        return test_loader

    def asoct_data(self, data_path, label_path):
        imgSize = 512
        train_dir = data_path + "train.txt"  #
        test_dir = data_path + "test.txt"    #

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        imagenet_pca = {
            'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
            'eigvec': torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        }
        train_loader = torch.utils.data.DataLoader(
            Myloader(train_dir, label_path, transforms.Compose([
                # transforms.Scale(600),
                transforms.Scale(520),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(15),
                transforms.RandomSizedCrop(imgSize),
                transforms.ToTensor(),
                auxtransform.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                auxtransform.Lighting(alphastd=0.1, eigval=imagenet_pca['eigval'], eigvec=imagenet_pca['eigvec']),
                normalize,
            ])),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False)

        test_transform = transforms.Compose([
                # transforms.Scale(600),
                transforms.Scale(520),
                transforms.CenterCrop(imgSize),
                transforms.ToTensor(),
                normalize,
            ])
        test_loader = torch.utils.data.DataLoader(
            Myloader(test_dir, label_path, test_transform),
            batch_size=int(self.batch_size/2),
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=False)
        return train_loader, test_loader



