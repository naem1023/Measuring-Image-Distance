import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

# construct model on cuda if available
use_cuda = torch.cuda.is_available()

import torch

def get_model():
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
    return model

# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet152', pretrained=True)
# model.eval()