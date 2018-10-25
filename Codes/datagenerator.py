# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:56:22 2018

@author: saket
"""


import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

class Datagen(Dataset):
    
    
    def __init__(self, args, transform=None):
        self.args = args
        self.dataset = args.dataset
        self.img_size = args.img_size
        self.root_dir = args.root_dir
        self.transform = transform

        
    def __len__(self):
        return self.args.batch_size
    
    
    def __getitem__(self, idx):
        if self.dataset == 'mypic':
            img_name = os.path.join(self.root_dir,
                                'Picture1.jpg')
            image = Image.open(img_name)
            cuda = True if torch.cuda.is_available() else False
            
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            if self.transform:
                image = self.transform(image)
                
            label = Variable(Tensor(1).fill_(1.0), requires_grad=False)
            return (image,label)
        
    

