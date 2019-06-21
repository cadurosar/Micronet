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
    clean_trainloader, trainloader, testloader = load_data(32, cutout=True)
    net = torch.load("checkpoint/14-5_nocutout.pth")["net"]
    net = net.to(device)
    teacher = torch.load("checkpoint/14-102.pth")["net"]
    teacher.eval()

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    test(net,testloader,criterion, device, "no")
    test(teacher,testloader,criterion, device, "no")
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100,175], gamma=0.1)
    for epoch in range(20):
        print('\nEpoch: %d' % epoch)
        train(net,clean_trainloader,criterion,scheduler, device, optimizer, teacher=teacher,lit=False, mixup_alpha=0)
        test(net,testloader,criterion, device, "no")
if __name__ == "__main__":
    main()



