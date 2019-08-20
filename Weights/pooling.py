import torch
import torch.nn as nn


class GlobalAveragePooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        reshaped = inputs.view(inputs.size(0), inputs.size(1), -1)
        pooled = torch.mean(reshaped, 2)
        return pooled.view(pooled.size(0), -1)
        
