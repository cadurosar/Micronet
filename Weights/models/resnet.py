'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import identity
import pooling
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.identity = identity.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.identity(self.shortcut(x))
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, first_layer=16, wide=1, samesize=False):
        super(ResNet, self).__init__()
        self.in_planes = first_layer
        self.num_blocks = num_blocks
        self.conv1 = nn.Conv2d(3, first_layer, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(first_layer)
        self.relu = nn.ReLU()
        self.global_average_pooling = pooling.GlobalAveragePooling()
        if samesize:
            self.layer1 = self._make_layer(block, first_layer*wide, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, first_layer*wide, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, first_layer*wide, num_blocks[2], stride=2)
            if len(self.num_blocks) > 3:
                self.layer4 = self._make_layer(block, first_layer*wide, num_blocks[3], stride=2)
            self.linear = nn.Linear(first_layer*wide, num_classes)
        else:
            self.layer1 = self._make_layer(block, first_layer*wide, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, first_layer*wide*2, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, first_layer*wide*4, num_blocks[2], stride=2)
            if len(self.num_blocks) > 3:
                self.layer4 = self._make_layer(block, first_layer*wide*8, num_blocks[3], stride=2)
                self.linear = nn.Linear(first_layer*wide*8, num_classes)
            else:
                self.linear = nn.Linear(first_layer*wide*4, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        try:
            if len(self.num_blocks) > 3:
                out = self.layer4(out)
        except:
            pass

        out = global_average_pooling(out)
        out = self.linear(out)
        return out


def ResNet8(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [1,1,1],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet10(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [1,1,1,1],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet142(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [1,1,6],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet14(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [2,2,2],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet20(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [3,3,3],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet26(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [4,4,4],first_layer=first_layer,wide=wide,samesize=samesize)


def ResNet32(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [5,5,5],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet38(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [6,6,6],first_layer=first_layer,wide=wide,samesize=samesize)


def ResNet44(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [7,7,7],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet50(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [8,8,8],first_layer=first_layer,wide=wide,samesize=samesize)


def ResNet56(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [9,9,9],first_layer=first_layer,wide=wide,samesize=samesize)

def ResNet110(first_layer=16,wide=10,samesize=False):
    return ResNet(BasicBlock, [18,18,18],first_layer=first_layer,wide=wide,samesize=samesize)

#def global_average_pooling(inputs):
#    reshaped = inputs.view(inputs.size(0), inputs.size(1), -1)
#    pooled = torch.mean(reshaped, 2)
#    return pooled.view(pooled.size(0), -1)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
