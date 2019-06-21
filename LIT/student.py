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
    net = ResNet8(wide=10,first_layer=16)
    net = net.to(device)
    teacher = torch.load("checkpoint/26-102.pth")["net"]
    teacher.eval()

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    test(teacher,testloader,criterion, device, "no")
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60,100, 120], gamma=0.1)
    for epoch in range(175):
        print('\nEpoch: %d' % epoch)
        train(net,trainloader,criterion,scheduler, device, optimizer, teacher=teacher)
#        test(net,clean_trainloader,criterion, device, "no")
        test(net,testloader,criterion, device, "2610_teacher_810_student")
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[35,55], gamma=0.1)
    for epoch in range(75):
        print('\nEpoch: %d' % epoch)
        train(net,trainloader,criterion,scheduler, device, optimizer, teacher=teacher,lit=False)
#        test(net,clean_trainloader,criterion, device, "no")
        test(net,testloader,criterion, device, "2610_teacher_810_student_kd")
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[35,55], gamma=0.1)
    for epoch in range(5):
        print('\nEpoch: %d' % epoch)
        train(net,clean_trainloader,criterion,scheduler, device, optimizer,mixup_alpha=0)
#        test(net,clean_trainloader,criterion, device, "no")
        test(net,testloader,criterion, device, "2610_teacher_810_student_kd_5")

if __name__ == "__main__":
    main()



