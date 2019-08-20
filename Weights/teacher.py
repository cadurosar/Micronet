'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import mobilenet
import resnet20
import os
import argparse
import sat_parameters
from models import *
from utils import progress_bar, load_data, train, test
from torch.optim.lr_scheduler import MultiStepLR
import binaryconnect


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_trainloader, trainloader, testloader = load_data(100, cutout=True, batch_clean=100)
    net = resnet20.SATResNet26(num_classes=100)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    bc = binaryconnect.BC(net)
    for epoch in range(200):
        print('Epoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer,cutmix=True,bc=bc)
        test(net,testloader, device, save_name="SAL_246",bc=bc)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    for epoch in range(5):
        print('Epoch: %d' % epoch)
        train(net,clean_trainloader,scheduler, device, optimizer,cutmix=False,mixup_alpha=0)
        test(net,testloader, device, save_name="SAL_246-2")
if __name__ == "__main__":
    main()



