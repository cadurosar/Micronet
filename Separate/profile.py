from models import *
from thop import profile
import utils
import torchvision.models
import numpy as np
import mobilenet

#model = mobilenet.mobilenet_v2(width_mult=1.4).cuda()
model = torch.load("checkpoint/resnet11010-2.pth")["net"].module
#model = ResNet8(wide=5,first_layer=16).cuda()
#model = ResNet10(wide=8,first_layer=16,samesize=True).cuda()

clean_trainloader, trainloader, testloader = utils.load_data(32, cutout=True)

input = torch.randn(1, 3, 32, 32).cuda()

flops, params = profile(model, inputs=(input,))
#model_parameters = model.parameters()
#params = sum([np.prod(p.size()) for p in model_parameters])

mobilenet_params = 6900000
mobilenet_flops = 1170000000
score_flops = 2*flops/mobilenet_flops
score_params = params/mobilenet_params
score = score_flops + score_params
print("Flops: {}, Params: {}".format(flops,params))
print("Score flops: {} Score Params: {}".format(score_flops,score_params))
print("Final score: {}".format(score))
utils.test(model,testloader, "cuda", "no")

