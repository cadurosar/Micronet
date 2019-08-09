import torch
import torch.nn
import sat_parameters as parameters

class Shift2Attention(torch.nn.Conv2d):	

    def __init__(self, in_features, out_features, stride=1, bias=False, padding = 0, kernel_size=3):
        super(ShiftAttention, self).__init__(in_features, out_features, stride = stride, bias = bias, padding = padding, kernel_size = kernel_size)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_width = kernel_size
        # self.temperature = temperature
        # self.bias = bias
        # self.stride = stride
        # self.padding = padding
        # self.conv = torch.nn.Conv2d()
        self.attentionWeights1 = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features, kernel_size* kernel_size))
        torch.nn.init.uniform_(self.attentionWeights1)
        self.attentionWeights2 = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features, kernel_size* kernel_size))
        torch.nn.init.uniform_(self.attentionWeights2)
        #b=torch.max(self.attentionWeights,dim=2)
        #print(b.shape)
        #self.attentionWeights.data[:,:,4]=b[0][:,:]

    def forward(self,input):
        # attention = self.attentionWeights / self.attentionWeights.std(dim=2).view(self.out_features, self.in_features, -1)
        # sortAt,_ = torch.sort(self.attentionWeights, descending = True, dim=2)
        attention1 = self.attentionWeights1 / self.attentionWeights1.std(dim=2).view(self.out_features, self.in_features, -1)
        attention2 = self.attentionWeights2 / self.attentionWeights2.std(dim=2).view(self.out_features, self.in_features, -1)
        # attention = self.attentionWeights / (sortAt[:,:,0]-sortAt[:,:,1]).view(self.out_features, self.in_features, -1)

        attention1 = torch.nn.functional.softmax(parameters.temperature * attention1, dim = 2)
        attention2 = torch.nn.functional.softmax(parameters.temperature * attention2, dim = 2)
        attention = torch.max(attention1,attention2)
        attention = attention.view(self.out_features, self.in_features, self.kernel_width, self.kernel_width)
        attention1 = attention1.view(self.out_features, self.in_features, self.kernel_width, self.kernel_width)
        attention2 = attention2.view(self.out_features, self.in_features, self.kernel_width, self.kernel_width)
        parameters.maxvalue = torch.max(attention[0,0,:]).item()
        if parameters.display:
            print(torch.max(attention[0,0,:]))
        if parameters.binarized:
            
            attention = torch.round(attention)
            attention1 = torch.round(attention1)
            attention2 = torch.round(attention2)
        
        attention = attention * self.weight
        return torch.nn.functional.conv2d(input, attention, bias = self.bias, stride = self.stride, padding = self.padding)
