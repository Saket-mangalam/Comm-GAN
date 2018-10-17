# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 08:34:25 2018

@author: saket
"""
import torch
from torch import nn
#from torch import Tensor

class SpectralNormalization(nn.Module):
    def __init__(self, module, power_iterations=1):
        super(SpectralNormalization, self).__init__()
        self.module = module
        self.power_iterations = power_iterations
        
    
    def normalize(self, x, eps =1e-10):
        return x/(x.norm()+eps)
            
    def spectral_norm(self):
        height = getattr(self.module, "weight_u")
        width = getattr(self.module, "weight_v")
        weight = getattr(self.module, "weight_bar")

        h = weight.data.shape[0]
        for i in range(self.power_iterations):
            width.data = normalize(torch.mv(torch.t(weight.view(h,-1).data), height.data))
            height.data = normalize(torch.mv(weight.view(h,-1).data, width.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = height.dot(weight.view(h, -1).mv(width))
        setattr(self.module, weight / sigma.expand_as(weight))
        
    def forward(self, *args):
        self.normalize(*args)
        return self.module.forward(*args)