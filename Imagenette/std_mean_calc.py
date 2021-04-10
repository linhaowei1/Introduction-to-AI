import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from tensorboardX import SummaryWriter
import random
from model import resnet18, resnet34, resnet50, resnet101, resnet152, wide_resnet101_2, wide_resnet50_2, resnext101_32x8d, resnext50_32x4d, resnet152
from utils import get_args

def getStat(train_data):
    device = 'cuda:4'
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=os.cpu_count(),
        pin_memory=True)
    mean = torch.zeros(3).to(device)
    std = torch.zeros(3).to(device)
    for X, _ in tqdm(train_loader):
        X.to(device)
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.cpu().numpy()), list(std.cpu().numpy())
 
if __name__ == '__main__':
    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  
    image_path = os.path.join(data_root, "imagenette2-320")  
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                            transform=transforms.ToTensor())
    print(getStat(train_dataset))