#author: rmodi
import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import glob 
from pathlib import Path 
import cv2
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import random
#no aug
train_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),\
    transforms.RandomHorizontalFlip(p=0.5),\
    transforms.RandomRotation(degrees=(-45, 45)),\
    transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5, 0.5,0.5]), #not imagenet weights, downstream bbn is uniform trained
])

simple_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5,0.5], std=[0.5, 0.5,0.5]), #not imagenet weights, downstream bbn is uniform trained
])

class pneumoniadataset(Dataset):
    def __init__(self,\
        split = 'train',\
        data_root = './data'):
    
        super(pneumoniadataset, self).__init__()
        self.split = split 
        self.data_root = data_root
        
        if split=='train':
            self.transform = train_transform
            print("choose aug transform..")
        else:
            self.transform = simple_transform
            print("choose simple transform..")
            
        self.img_dir = Path(self.data_root)/self.split
        self.img_paths = sorted(glob.glob(str(Path(self.img_dir)/'**/*.jpeg'), recursive = True))
        random.shuffle(self.img_paths)
        print("img dir", self.img_dir)
        print("read", len(self.img_paths), "images from ssd....")
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        if 'NORMAL' in img_path:
            label = 0
        else:
            label = 1
        img = Image.open(img_path)
        img = np.array(img)
        n_dims = len(img.shape)
        if n_dims == 2:
            img = repeat(img, 'h w -> h w c', c=3)    
                
        img = Image.fromarray(img)
        img = self.transform(img)
        
        return img, label
if __name__ == '__main__':
    dataset = pneumoniadataset(split='train')
    # dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    
    # for idx, (img, label) in enumerate(dataloader):
    #     print("img shape", img.shape)
    # print("len", len(dataset))
    # print("img", dataset[0][0].shape)
    # print("label", dataset[0][1])
    # dataset = pneumoniadataset(split='test')
    for i in range(len(dataset)):
        print("img", dataset[i][0].shape)
    #     print("label", dataset[i][1])
    # print("len", len(dataset))
    # print("img", dataset[0][0].shape)
    # print("label", dataset[0][1])