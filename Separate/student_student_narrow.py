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
    clean_trainloader, trainloader, testloader = load_data(32, cutout=True,batch_clean=128)

#    net = torch.load("checkpoint/145_teacher_108_student.pth")["net"].module
    net = ResNet14(wide=1,first_layer=32,samesize=True).cuda()
    net = net.to(device)
    teacher = torch.load("checkpoint/145_teacher_145_student-2.pth")["net"]
    teacher.eval()

    test(teacher,testloader, device, "no")
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    """
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100,175], gamma=0.1)
    for epoch in range(250):
        print('\nEpoch: %d' % epoch)
        train(net,trainloader,scheduler, device, optimizer, teacher=teacher)
        test(net,testloader, device, "145_teacher_108_student")
    """
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[100,175], gamma=0.1)
    for epoch in range(5):
        print('\nEpoch: %d' % epoch)
        train(net,clean_trainloader,scheduler, device, optimizer, teacher=teacher, mixup_alpha=0)
        test(net,testloader, device, "145_teacher_108_student-2")
if __name__ == "__main__":
    main()



