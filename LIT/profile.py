from models import *
from thop import profile
from efficientnet_pytorch import EfficientNet
import utils

def count_convNd(m, x, y):
    x = x[0]
    cin = m.in_channels
    # batch_size = x.size(0)

    kernel_ops = m.weight.size()[2:].numel()
    bias_ops = 1 if m.bias is not None else 0
    ops_per_element = kernel_ops + bias_ops
    output_elements = y.nelement()

    # cout x oW x oH
    total_ops = cin * output_elements * ops_per_element // m.groups
    m.total_ops = torch.Tensor([int(total_ops)])


model = EfficientNet.from_name('efficientnet-b1')
print(type(model._conv_stem))
#model = torch.load("checkpoint/b0.pth")["net"]
clean_trainloader, trainloader, testloader = utils.load_data(32, cutout=True)
criterion = nn.CrossEntropyLoss()
criterion.to("cuda")
flops, params = profile(model, input_size=(1, 3, 224,224),device="cuda", custom_ops={type(model._conv_stem):count_convNd})
mobilenet_params = 6900000 
mobilenet_flops = 1170000000
score_flops = flops/mobilenet_flops
score_params = params/mobilenet_params
score = score_flops + score_params
print("Flops: {}, Params: {}".format(flops,params))
print("Score flops: {} Score Params: {}".format(score_flops,score_params))
print("Final score: {}".format(score))
utils.test(model,testloader,criterion, "cuda", "no")
