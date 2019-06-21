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
    clean_trainloader, trainloader, testloader = load_data(32,cutout=True)
    net = ResNet56(wide=10,first_layer=16)
    net = net.to(device)
    teacher = torch.load("checkpoint/teacher.pth")["net"]
    teacher.eval()

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60,100, 120], gamma=0.1)
    for epoch in range(175):
        print('\nEpoch: %d' % epoch)
        train(net,trainloader,criterion,scheduler, device, optimizer, teacher=teacher)
        test(net,clean_trainloader,criterion, device, "no")
        test(net,testloader,criterion, device, "56_teacher_56_student")
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[35,55], gamma=0.1)
    for epoch in range(75):
        print('\nEpoch: %d' % epoch)
        train(net,trainloader,criterion,scheduler, device, optimizer, teacher=teacher,lit=False)
        test(net,clean_trainloader,criterion, device, "no")
        test(net,testloader,criterion, device, "56_teacher_56_student")

if __name__ == "__main__":
    main()



