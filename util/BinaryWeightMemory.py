import torch, torchvision, torch.nn as nn
import math, numpy as np

class BinaryWeightMemory():
  def __init__(self, model:nn.Module):
    """
    Hold the pointer to the weights and the quantized representation associated
    From Courbariaux & al. 2015
    """
    self.saved_params = []
    self.actual_params = []
    self.params = 0  
    for m in model.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        self.saved_params.append(m.weight.data.clone())
        self.actual_params.append(m.weight)
        self.params += 1

    
  def binarize(self, BWN=True, determinist = True):
    """ 
    Modify the weights to show their binary counterpart at inference time
    """
    for i in range(self.params):
      true_value = self.actual_params[i].data
      self.saved_params[i].data.copy_(self.actual_params[i])
      if determinist:
        quantized = true_value.sign()
      else:
        """ From Courbariaux & al. 2015, w_b = +1 with p=hardsigmoid(w), -1 with q = 1-p """
        raise ValueError('Stochastic binarization is not supported')
      if BWN: quantized *= torch.mean(true_value)
      self.actual_params[i].data.copy_(quantized)
  
  def restore(self):
    for i in range(self.params):
      self.actual_params[i].data.copy_(self.saved_params[i])
        
  def clip(self):
    """ From Courbariaux & al. 2015, 2.4 - Clip weights after update to -1;1 
    since it doesn't impact the sign() ops while preventing overly large weights
    """
    for i in range(self.params):
      self.actual_params[i].data.copy_(torch.clamp(self.actual_params[i], min=-1, max=1))
      
  def __str__(self):
    """ Return a string representing the first param of the weight manager """
    return "Saved params \n {} \n Actual params \n {}".format(self.saved_params[0], self.actual_params[0])
  

    
