import torch
import torchvision
import torchvision.transforms as transforms
import ny
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import json
import os

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def get_dataset(path):
    ny_dataset = ny.NyDataset(path)
    # print(len(ny_dataset))
    # dataset = torch.utils.data.DataLoader(dataset=ny_dataset, batch_size=64, shuffle=True)

    # Set split lenght
    train_ratio = 0.8
    train_len = int(len(ny_dataset) * train_ratio)
    val_len = len(ny_dataset) - train_len

    train_dataset, val_dataset = torch.utils.data.random_split(ny_dataset, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

    return train_loader, val_loader

def train(data):
    train_loader, val_loader = data