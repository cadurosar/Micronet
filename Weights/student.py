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
import mobilenet

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    clean_trainloader, trainloader, testloader = load_data(32, cutout=True, batch_clean=100)
    teacher_file = "checkpoint/densenet10064-strongcutout.pth"
    teacher = torch.load(teacher_file)["net"].module
#    net = ResNet14(wide=4,first_layer=16)
    net = densenet_cifar(196,8,groups=True)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100, 175], gamma=0.1)
    test(teacher,testloader, device, save_name="no")
    for epoch in range(250):
        print('Epoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer,cutmix=True, teacher = teacher)
        test(net,testloader, device, save_name="swapteacher_dense10064-groups_student_dense1968-groups")
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50, 150], gamma=0.1)
    teacher_file = "checkpoint/densenet10064-strongcutout-2.pth"
    teacher = torch.load(teacher_file)["net"].module
    for epoch in range(5):
        print('Epoch: %d' % epoch)
        train(net,clean_trainloader,scheduler, device, optimizer,cutmix=False,mixup_alpha=0,teacher=teacher)
        test(net,testloader, device, save_name="teacher_dense10064-groups_student_dense1968-groups-2")
if __name__ == "__main__":
    main()



