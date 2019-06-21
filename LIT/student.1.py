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
    clean_trainloader, trainloader, testloader = load_data(128, cutout=False)
    net = torch.load("checkpoint/b0.pth")["net"]
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    test(net,testloader,criterion, device, save_name="no")
    for epoch in range(20):
        print('Epoch: %d' % epoch)
        train(net,clean_trainloader,criterion,scheduler, device, optimizer, mixup_alpha=0)
#        test(net,clean_trainloader,criterion, device,save_name="no")
        test(net,testloader,criterion, device, save_name="b0_post.pth")

if __name__ == "__main__":
    main()



