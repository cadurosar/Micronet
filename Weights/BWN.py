import torch.nn as nn
import numpy
from torch.autograd import Variable
import torch
class BC():
    def __init__(self, model):
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
                

        start_range = 0
        end_range = 1#count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        #self.bc_layers = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    
                    m.binary = torch.ones(1)
                    m.bwn = torch.ones(1)
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    

    def binarization(self): 
        self.save_params()
        for index in range(self.num_of_params):
            E=self.target_modules[index].data.abs().mean()
            self.target_modules[index].data.copy_(self.target_modules[index].data.sign() *E)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def save_score(self):
        for index in range(self.num_of_params):
            
            self.saved_params[index].copy_(self.target_modules[index].data)


    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):
        pass

    
