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
    clean_trainloader, trainloader, testloader = load_data(1000, cutout=False, batch_clean=100)
    teacher_file = "checkpoint/resnet11010.pth"
    teacher = torch.load(file)["net"].module
    net = ResNet14(wide=4,first_layer=16)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    for epoch in range(200):
        print('Epoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer,cutmix=True)
        test(net,testloader, device, save_name="teacher_11010_student_144")
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)
    for epoch in range(100):
        print('Epoch: %d' % epoch)
        train(net,clean_trainloader,scheduler, device, optimizer,cutmix=False,mixup_alpha=0,teacher=teacher)
        test(net,testloader, device, save_name="Teacher11010_student144")
if __name__ == "__main__":
    main()



