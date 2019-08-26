import torch.utils.data as data
import torchvision.datasets as dsets
from PIL import Image
import os
import os.path
import random
import numbers
import torchvision.transforms as transforms
import pickle
from os.path import exists, join
import sys
from os import makedirs
import numpy as np
import re
import pandas as pd

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    isImage = False
    for extension in IMG_EXTENSIONS:
        if filename.endswith(extension):
            isImage = True
    return isImage


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx



def pil_loader(path):
    return Image.open(path).convert('RGB')




def make_dataset(dir, label_df):
    images = []
    for line in open(dir):
        img_path = line.rstrip('\n')
        img_path = img_path.rstrip ('\r')
        img_id = re.sub('.jpg', '', img_path.split('/')[-1])
        img_left_label = label_df.loc[img_id, 'left']
        img_right_label = label_df.loc[img_id, 'right']
        if img_left_label == 0 or img_right_label == 0:
            continue
        if img_left_label == 2 or img_left_label == 3 or img_left_label == 4:
            img_left_label = 2
        if img_right_label == 2 or img_right_label == 3 or  img_right_label == 4:
            img_right_label = 2
        item = (img_path, img_left_label, img_right_label)
        images.append(item)
    return images



def get_label(label_dir):
    label_df = pd.read_csv(label_dir)
    label_df = label_df.set_index('filename')
    return label_df



class Myloader(data.Dataset):
    def __init__(self, root, label_dir, transform=None):
        self.root = root
        self.label_dir = label_dir
        self.loader = pil_loader
        self.transform = transform

        self.label_df = get_label(label_dir)
        self.imgs = make_dataset(root, self.label_df)


    def __getitem__(self, index):
        path, left, right = self.imgs[index]
        img = self.loader(path)
        left_region = (0, 0, img.size[0]/2, img.size[1])
        right_region = (img.size[0]/2, 0, img.size[0], img.size[1])

        if random.random() < 0.5:
            img = img.crop(left_region)
            label = left
        else:
            img = img.crop(right_region)
            label = right

        img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)