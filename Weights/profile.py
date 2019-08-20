import argparse

import torch
import torch.nn as nn
import utils
import identity
import sat
import sat2
import binaryconnect
import pooling
def count_conv2d(m, x, y):
    global binary_connect
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = kh * kw * cin * (1/32)**m.binary  
    kernel_add = kh * kw * cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sat(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = cin * (1/32)**m.binary
    kernel_add = cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_sat2(m, x, y):
    x = x[0]

    cin = m.in_channels // m.groups
    cout = m.out_channels // m.groups
    kh, kw = m.kernel_size
    batch_size = x.size()[0]

    # ops per output element
    kernel_mul = 2*cin * (1/32)**m.binary
    kernel_add = 2*cin - 1
    bias_ops = 1 if m.bias is not None else 0
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_sub = nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])

def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]

    batch_size, nfeatures = x.size()

    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)

    m.total_ops += torch.Tensor([int(total_ops)])

def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features * (1/32)**m.binary
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])


def count_identity(m, x, y):
    total_ops = y.numel()
    m.total_ops += torch.Tensor([int(total_ops)]) 


def count_globalAveragePooling(m, x, y):
    total_add = x.shape[2]*x.shape[3]
    total_div = 1
    total_ops = y.numel() * (total_add + total_div)
    m.total_ops += torch.Tensor([int(total_ops)])     

def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))
        if isinstance(m, sat.ShiftAttention):
            
            for p in m.parameters():     
                m.total_params += torch.Tensor([p.numel()*36/(18*32)])
        elif isinstance(m, sat2.Shift2Attention):
            
            for p in m.parameters():
                if m.binary == 1:     
                    m.total_params += torch.Tensor([p.numel()*2*36/(18*32)]) / 32
                else: 
                    m.total_params += torch.Tensor([p.numel()*2*36/(18*32)])  
        else:
            
            for p in m.parameters():
                if m.binary == 1:
                    m.total_params += torch.Tensor([p.numel()]) / 32
                else:
                    m.total_params += torch.Tensor([p.numel()])
        
        if isinstance(m, sat.ShiftAttention):
            m.register_forward_hook(count_sat)
        elif isinstance(m, sat2.Shift2Attention):
            m.register_forward_hook(count_sat2)
        elif isinstance(m, nn.Conv2d):    
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            m.register_forward_hook(count_maxpool)
        elif isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
  
        elif isinstance(m, identity.Identity):
            m.register_forward_hook(count_identity)
        elif isinstance(m, pooling.GlobalAveragePooling):
            m.register_forward_hook(count_globalAveragePooling)
        
        elif isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            pass
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params
    total_ops = total_ops
    total_params = total_params

    return total_ops, total_params

def main():
    import mobilenet
    import models.densenet
    import resnet20
#    model = mobilenet.mobilenet_v2(width_mult=1.4,num_classes=1000)
#    model = models.densenet.densenet_cifar(n=93,growth_rate=7)#(width_mult=1.4,num_classes=100)
#    print(model)
    model = resnet20.SATResNet26()#
#    model = torch.load("checkpoint/resnet110samesize.pth")["net"].module.cpu()
#    file = "checkpoint/teacher_dense1968-groups_student_dense1968-groups-2.pth"
#    model = torch.load(file)["net"].module.cpu()
    for m in model.modules():
        m.register_buffer('binary', torch.zeros(1))
    bc = binaryconnect.BC(model)
    
       
    flops, params = profile(model, (1,3,32,32))
#    flops, params = profile(model, (1,3,224,224))
    flops, params = flops.item(), params.item()
    mobilenet_params = 6900000
    mobilenet_flops = 1170000000
    score_flops = flops/mobilenet_flops
    score_params = params/mobilenet_params
    score = score_flops + score_params
    print("Flops: {}, Params: {}".format(flops,params))
    print("Score flops: {} Score Params: {}".format(score_flops,score_params))
    print("Final score: {}".format(score))

    #model = torch.load(file)["net"].module

    clean_trainloader, trainloader, testloader = utils.load_data(32, cutout=True)
    utils.test(model,testloader, "cuda", "no")

if __name__ == "__main__":
    main()
