import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob 
from pathlib import Path 
import cv2
from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch.nn as nn
import math 

from pytorch_pretrained_vit import ViT



class Model(nn.Module):
    def __init__(self,is_pretrained):    
        super(Model, self).__init__()
        if is_pretrained:
            self.backbone = ViT('B_16_imagenet1k', pretrained=True)
            print("loaded with weights...")
        else:
            self.backbone = ViT('B_16_imagenet1k', pretrained=False)
            print("loaded without weights...")
        self.fc = nn.Linear(768, 2)
        
    def forward(self, x):
        x,_ = self.backbone(x)
        print(x.shape)
        x = self.fc(x)
        return x
    
if __name__ == "__main__":
    model = Model(False).cuda()
    x = torch.randn(1, 3, 384*2, 384*2).cuda()
    pre = model(x)
    print(pre.shape)
    print(pre)
    print("done")