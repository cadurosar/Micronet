'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import mobilenet

import os
import argparse

from models import *
from utils import progress_bar, load_data, train, test
from torch.optim.lr_scheduler import MultiStepLR


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_trainloader, trainloader, testloader = load_data(128, cutout=True, batch_clean=100)
#    file = "checkpoint/resnet11010WD.pth"
#    net = torch.load(file)["net"].module.cpu()
#    net = ResNet14(wide=4,first_layer=16)
#    net = mobilenet.mobilenet_v2(width_mult=1.4, num_classes=100)
    net = densenet_cifar(93,7)#,groups=True)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    for epoch in range(200):
        print('Epoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer,cutmix=True)
        test(net,testloader, device, save_name="densenet937-strongcutout-single_layer")
    
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
#    optimizer = optim.SGD(net.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    
    for epoch in range(5):
        print('Epoch: %d' % epoch)
        train(net,clean_trainloader,scheduler, device, optimizer,cutmix=False,mixup_alpha=0)
        test(net,testloader, device, save_name="densenet937-strongcutout-single_layer-2")
if __name__ == "__main__":
    main()



