import torch
import torchvision
import torchvision.transforms as transforms
import ny
import resnet
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    train_loader, test_loader = data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(53)
    if device == 'cuda':
        torch.cuda.manual_seed_all(53)

    model = resnet.get_model()
    model = model.to(device)

    # Optimize
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)

    import time
    start_time = time.time()
    min_loss = int(1e9)
    history = {'loss': [], 'val_acc': []}
    for epoch in range(1):  # loop over the dataset multiple times
        epoch_loss = 0.0
        tk0 = tqdm(train_loader, total=len(train_loader), leave=False)
        for step, (inputs, labels) in enumerate(tk0, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # validation
        if epoch % 10 == 0:
            class_correct = list(0. for i in range(1000))
            class_total = list(0. for i in range(1000))
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data
                    images = images.cuda()
                    labels = labels.cuda()
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).squeeze()
                    for i in range(labels.size()[0]):
                        label = labels[i].item()
                        class_correct[label] += c[i].item()
                        class_total[label] += 1
            val_acc = sum(class_correct) / sum(class_total) * 100
        else:
            val_acc = 0

        # print statistics
        tqdm.write('[Epoch : %d] train_loss: %.5f val_acc: %.2f Total_elapsed_time: %d 분' %
                   (epoch + 1, epoch_loss / 272, val_acc, (time.time() - start_time) / 60))
        history['loss'].append(epoch_loss / 272)
        history['val_acc'].append(val_acc)

        if epoch in [36, 64, 92]:
            for g in optimizer.param_groups:
                g['lr'] /= 10
            print('Loss 1/10')

    print(time.time() - start_time)
    print('Finished Training')
