#author: rmodi
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
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    batch_size = 16
    num_workers = 8
    num_epochs = 15
    lr = 1e-4
    device = 'cuda'
    train_print_freq = 1
    save_dir = Path('./checkpoints_without_pretrained')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    
    train_dataset = pneumoniadataset(split='train')
    val_dataset = pneumoniadataset(split = 'val')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
    
    model = Model(is_pretrained = False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    writer = SummaryWriter(log_dir="runs/pa1")  # Replace with your desired log directory

    
    # for idx, (img, label) in enumerate(train_dataloader):
    #         img = img.to(device)
    #         print("img shape", img.shape)
    # exit(1)
            
    def train_loop(model, dataloader, optimizer):
        model.train()
        total_loss = 0
        av_loss = []     
        n_iters = 0    
        for idx, (img, label) in enumerate(dataloader):
            img = img.to(device)
            # print("img shape", img.shape)
            # exit(1)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(img)
            loss = torch.nn.functional.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            av_loss.append(loss.item())
            writer.add_scalar("train_loss", np.mean(av_loss),n_iters)
            n_iters += 1
            if idx % train_print_freq == 0:
                print(f"idx: {idx}, loss: {loss.item()}")
                print("av loss", np.mean(av_loss), idx, len(dataloader))
            # break
        return total_loss / len(dataloader)
    
    def eval_loop(model, dataloader):
        model.eval()
        total_loss = 0
        av_loss = []  
        n_iters = 0    
        for idx, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = torch.nn.functional.cross_entropy(output, label)
            av_loss.append(loss.item())
            total_loss += loss.item()
            writer.add_scalar("val_loss", np.mean(av_loss),n_iters)
            n_iters += 1
            print("av loss", np.mean(av_loss), idx, len(dataloader))
            # break
        return total_loss / len(dataloader)
    
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        train_loss = train_loop(model, train_dataloader, optimizer)
        val_loss = eval_loop(model, val_dataloader)
        writer.add_scalar("epochwise_train_loss", train_loss,epoch)
        writer.add_scalar("epochwise_val_loss", val_loss,epoch)
        print("------------ final", epoch, train_loss, val_loss)
        train_save_dir = save_dir / "train"
        train_save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), train_save_dir / f"model_{epoch}.pth")
        
        if val_loss < best_loss:
            best_loss = val_loss
            val_save_dir = save_dir / "val"
            val_save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), val_save_dir / f"model_{epoch}.pth")
            
        print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")
    # train_on_dataloader(model, train_dataloader, optimizer)
    