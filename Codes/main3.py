# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:51:01 2018

@author: saket
"""

from __future__ import print_function
#%matplotlib inline
#import argparse
import os
import random
import torch
#import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
#import torch.optim as optim
import torch.utils.data
#import torch.nn.init.xavier_uniform as initialize 
#import torchvision.datasets as dset
import torchvision.transforms as transforms
#import torchvision.utils as vutils
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from IPython.display import HTML
#import numpy as np
#import os
import torch
from get_args import get_args
import csv

from utils import channel, errors_ber, weights_init_normal
random.seed(999)
torch.manual_seed(999)

cudnn.benchmark = True

#import torchvision.transforms as transforms
from torchvision.utils import save_image
#from datagenerator import Datagen
#from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable




if __name__ == '__main__':

    # give each run a random ID
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)

    args = get_args()
    #directory for saving images
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/'+identity, exist_ok=True)
    #directory for saving loss log
    os.makedirs('logbook', exist_ok=True)
    #os.makedirs('logbook/'+identity, exist_ok=True)

    img_shape = (args.img_channel, args.img_size, args.img_size)
    cuda = True if torch.cuda.is_available() else False

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    device = torch.device("cuda" if cuda else "cpu")

    # Loss function, we just use BCE this time.
    BCELoss = torch.nn.BCELoss()
    MSELoss = torch.nn.MSELoss()
    
    ########################################################
    # Setup GAN structures: TBD: select the right G,D and Dec
    ########################################################
    #initialize generator for gan
    from generators import DCGANGenerator as Generator
    #initialize generator for encoder
    #from generators import Enc_Generator as EncGenerator
    #initialize discriminator for gan
    from discriminators import DCGANDiscriminator as Discriminator
    #initialize discriminator for encoder
    #from discriminators import Enc_Discriminator as EncDiscriminator
    #initialize decoder
    #from decoders import Enc_Decoder as Decoder
    
    generator     = Generator(args)
    #encgenerator     = EncGenerator(args)
    #decoder          = Decoder(args)
    discriminator = Discriminator(args)
    #encdiscriminator = EncDiscriminator(args)
    
    if cuda:
        generator.cuda().to(device)
        #encgenerator.cuda()
        discriminator.cuda().to(device)
        #encdiscriminator.cuda()
        #decoder.cuda()
        BCELoss.cuda().to(device)
        MSELoss.cuda().to(device)
    else:
        print('models', generator, discriminator)

    # Initialize weights
    generator.apply(weights_init_normal)
    #encgenerator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    #encdiscriminator.apply(weights_init_normal)
    #decoder.apply(weights_init_normal)
    
    #########################################
    #### Dataloader #########################
    #########################################
    if args.dataset == 'cifar10':
        os.makedirs('./data/cifar10', exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True,
                           transform=transforms.Compose([
                                transforms.Resize(64),
                                #transforms.CenterCrop(64),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])),
            batch_size=args.batch_size, shuffle=True)
        
    
    #optimizers
    optimizer_G   = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    #optimizer_encG   = torch.optim.Adam(encgenerator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    #optimizer_Dec    = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D   = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    #optimizer_encD   = torch.optim.Adam(encdiscriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    #initialize tensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    fixed_noise = torch.randn((args.batch_size,args.sample_noise,1,1), dtype=torch.float).to(device)
    ##############################
    #  Training ##################
    ##############################
    with open('logbook/'+identity+'.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for epoch in range(args.num_epoch):
            for i, (imgs, _) in enumerate(dataloader):
                # Adversarial ground truths
                #print(imgs.shape)
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    
                if valid.shape[0]!=args.batch_size:
                    continue
                
                # Configure input
                #noise input for gan
                real_noise = torch.randn((args.batch_size,args.sample_noise,1,1), dtype=torch.float).to(device)
                #real images
                real_imgs = Variable(imgs.type(Tensor))
                #encoded message
                #u = torch.randint(0, 2, (args.batch_size, args.block_len), dtype=torch.float).to(device)
                
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                #optimizer_Dec.zero_grad()
                #optimizer_ganD.zero_grad()
                #optimizer_encD.zero_grad()
                
                #discriminate 
                real = discriminator(real_imgs).view(-1)
                real_dloss = BCELoss(real,valid)
                real_dloss.backward()
                D_x = real.mean().item()
                
                fake_imgs = generator(fixed_noise)
                nonreal = discriminator(fake_imgs.detach()).view(-1)
                fake_dloss = BCELoss(nonreal,fake)
                fake_dloss.backward()
                D_G_z1 = nonreal.mean().item()
                
                errD = fake_dloss + real_dloss
                # Update D
                optimizer_D.step()
                
                #generate
                nreal = discriminator(fake_imgs).view(-1)
                fake_gloss = BCELoss(nreal,valid)
                fake_gloss.backward()
                D_G_z2 = nreal.mean().item()
                
                errG = fake_gloss
                #update G
                optimizer_G.step()
                
                #print results
                if i%100 == 0:
                    
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [logD(x): %f] [log(D(G(z))): %f] [log(D(G(z))): %f]" % (epoch, args.num_epoch, i, len(dataloader),
                                                                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                    
                batches_done = epoch * len(dataloader) + i
                #saving log
                if batches_done == 0:
                    filewriter.writerow(['Batchnumber','D loss','G loss','D(x)','1 D(G(z))','2 D(G(z))'])
                    filewriter.writerow([batches_done,errD.item(),errG.item(),D_x,D_G_z1,D_G_z2])
                else:
                    filewriter.writerow([batches_done,errD.item(),errG.item(),D_x,D_G_z1,D_G_z2])
                    
                #saving generated image
                if batches_done % args.sample_interval == 0:
                    fake_imgs = generator(fixed_noise)
                    save_image(fake_imgs.data[:25], 'images/'+identity+'/%d.png' % batches_done, nrow=5, normalize=True)
                    
    

                
                