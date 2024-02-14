import os
import torch
import numpy as np
# import matplotlib.pyplot as plt
import glob 
from pathlib import Path 
import cv2
from einops import rearrange, reduce, repeat
from model import Model
from dataloader import pneumoniadataset


if __name__ == '__main__':
    
    
    device = 'cuda'
    save_dir = Path('./checkpoints')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = 'checkpoints_pretrained/val/model_1.pth'
    test_dataset = pneumoniadataset(split='test')
    
    
    test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    model = Model(is_pretrained = False).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path), strict = True)
    model.eval()
    print("loaded model")
    
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            print("idx", idx)   
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
    print('Finished Testing')