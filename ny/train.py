import torch
import torchvision
import torchvision.transforms as transforms
import ny.ml_model
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


class RMSELoss(nn.Module):
    """Calculate RMSE Loss for validating test data.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        """Calculate RMSE Loss from two vectors.
        Plus epsilon for preventing zero division.
        """
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss

class Train:
    def __init__(self):
        self.save_path = 'gdrive/MyDrive/Colab Notebooks/'

    def train(self, data):
        train_loader, test_loader = data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # torch.manual_seed(53)
        # if device == 'cuda':
        #     torch.cuda.manual_seed_all(53)

        model = ny.ml_model.MIS()

        if torch.cuda.device_count() > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
            model = nn.DataParallel(model, output_device=1)
            print('Multi GPU!!!!!!!!!!!!!!')

        model = model.to(device)

        # Optimize
        # criterion = nn.CrossEntropyLoss().cuda()
        criterion = nn.SmoothL1Loss().cuda()
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        epochs = 10

        import time
        start_time = time.time()
        min_loss = int(1e9)
        history = {'loss': [], 'val_acc': []}
        for epoch in range(epochs):  # loop over the dataset multiple times
            epoch_loss = 0.0
            tk0 = tqdm(train_loader, total=len(train_loader), leave=False)
            for step, (inputs, labels) in enumerate(tk0, 0):
                image_inputs = inputs['image']
                coordinate_inputs = torch.stack([val for val in inputs['target_coordinate'][0]], dim=0).to(device)
                image_inputs, labels = image_inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model((image_inputs, coordinate_inputs))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            train_loader.dataset.dataset.read_img = True
            train_loader.dataset.dataset.read_depth = True
            # print('make true')
            # validation
            # if epoch % 10 == 0:
            validation_criterion = RMSELoss()
            rmse_loss = 0
            with torch.no_grad():
                for data in test_loader:
                    inputs, labels = data
                    image_inputs = inputs['image']
                    coordinate_inputs = torch.stack([val for val in inputs['target_coordinate'][0]], dim=0).cuda()

                    images = image_inputs.cuda()
                    labels = labels.cuda()
                    outputs = model((images, coordinate_inputs))
                    _, predicted = torch.max(outputs, 1)
                    rmse_loss = validation_criterion(labels, predicted)

            # print statistics
            tqdm.write('[Epoch : %d] train_loss: %.5f val_acc: %.2f Total_elapsed_time: %d minute' %
                       (epoch + 1, epoch_loss / len(train_loader), rmse_loss, (time.time() - start_time) / 60))
            history['loss'].append(epoch_loss / len(train_loader))
            history['val_acc'].append(rmse_loss)

            # if epoch in [36, 64, 92]:
            #     for g in optimizer.param_groups:
            #         g['lr'] /= 10
            #     print('Loss 1/10')

        print((time.time() - start_time) / 60)
        print('Finished Training')

        torch.save(model.state_dict(), os.path.join(self.save_path, 'model_state_dict.pth'))
