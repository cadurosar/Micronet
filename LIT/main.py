'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import os
import argparse

from models import *
from utils import progress_bar, load_data, train, test
from torch.optim.lr_scheduler import MultiStepLR


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_trainloader, trainloader, testloader = load_data(128, cutout=True)
#    net = ResNet8(wide=5,first_layer=16)
    net = EfficientNetB0()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    for epoch in range(200):
        print('Epoch: %d' % epoch)
        train(net,trainloader,criterion,scheduler, device, optimizer)
#        test(net,clean_trainloader,criterion, device,save_name="no")
        test(net,testloader,criterion, device, save_name="b0")

if __name__ == "__main__":
    main()



