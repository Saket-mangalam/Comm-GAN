# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:37:38 2018

@author: saket
"""

import numpy as np
import os
import torch
from get_args import get_args
import csv

from utils import channel, errors_ber, weights_init_normal

import torchvision.transforms as transforms
from torchvision.utils import save_image
from datagenerator import Datagen
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

    if args.g_type == 'hidden5':
        from generators import Hidden_Generator_5 as Generator
    else:
        from generators import FCNN_Generator as Generator

    if args.dec_type == 'hidden5':
        from decoders import Hidden_Decoder_5 as Decoder
    else:
        from decoders import gridrnn_Decoder as Decoder

    if args.d_type == 'dcgan':
        from discriminators import DCGAN_discriminator as Discriminator
    else:
        from discriminators import Hidden_discriminator as Discriminator

    generator     = Generator(args)
    decoder       = Decoder(args)
    discriminator = Discriminator(args)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        decoder.cuda()
        BCELoss.cuda()
        MSELoss.cuda()
    else:
        print('models', generator, discriminator, decoder)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    decoder.apply(weights_init_normal)


    ########################################################
    # Dataloader: TBD CIFAR10, FashionMNIST, etc.
    ########################################################
    # Configure data loader
    if args.dataset == 'mypic':
        train_dataloader = torch.utils.data.DataLoader(Datagen(args,
                              transform=transforms.Compose([
                                      transforms.Resize((32,32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                           ])),
                        batch_size=args.batch_size, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(Datagen(args,
                              transform=transforms.Compose([
                                      transforms.Resize((32,32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                           ])),
                        batch_size=args.batch_size, shuffle=True)
    elif args.dataset == 'mnist':
        os.makedirs('./data/mnist', exist_ok=True)
        train_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])),
            batch_size=args.batch_size, shuffle=True)

        test_dataloader = torch.utils.data.DataLoader(
            datasets.MNIST('./data/mnist', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])),
            batch_size=args.batch_size, shuffle=True)

    elif args.dataset == 'cifar10':
        os.makedirs('./data/cifar10', exist_ok=True)
        train_dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True,
                           transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])),
            batch_size=args.batch_size, shuffle=True)

        test_dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data/cifar10', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                           ])),
            batch_size=args.batch_size, shuffle=True)
    else:
        print('hahahaha dataset is unknown')

    
    #optimizers
    optimizer_G   = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_Dec = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D   = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    #initialize tensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    d_loss = 1
    epoch = 0
    # ----------
    #  Training
    # ----------
    with open('logbook/'+identity+'.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for epoch in range(args.num_epoch):
            for i, (imgs, _) in enumerate(train_dataloader):
    
                # Adversarial ground truths
                #print(imgs.shape)
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
    
                if valid.shape[0]!=args.batch_size:
                    continue
                
                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                #encoded message
                u = torch.randint(0, 2, (args.batch_size, args.block_len), dtype=torch.float).to(device)
                
                #print(encodable_u.shape)
                
                '''where do i put this initialization'''
                optimizer_G.zero_grad()
                optimizer_D.zero_grad()
                optimizer_Dec.zero_grad()
                
                # -----------------
                #  Train G
                # -----------------
                for idx in range(args.num_train_G):
                    
                    
                    # Generate a batch of images
                    gen_imgs = generator(real_imgs, u)
                    # Loss measures generator's ability to fool the discriminator
                    gen_imgs = channel(gen_imgs, args.noise_std, channel_type = args.channel_type, device = device)
                    
                    #for every iteration of G train discriminator n times
                    for k in range(args.num_train_D):
                        
                        # Discriminate real or fake
                        valid_gen = discriminator(real_imgs)
                        fake_gen = discriminator(gen_imgs)
                        #train discriminator only
                        real_loss = BCELoss(valid_gen,valid)
                        fake_loss = BCELoss(fake_gen, fake)
                        d_loss = (real_loss + fake_loss) / 2
                        d_loss.backward(retain_graph=True)
                        optimizer_D.step()
                        
                    #for every iteration of G train Dec n times
                    for j in range(args.num_train_Dec):
                        
                        # Decode it
                        decoded_info = decoder(gen_imgs)
                        # train decoder only
                        dec_loss =  BCELoss(decoded_info, u)
                        dec_loss.backward(retain_graph=True)
                        optimizer_Dec.step()
                        
                    
                    #train generator
                    g_loss = (1.0 - args.lambda_I - args.lambda_G)*BCELoss(decoded_info,u) + \
                                    args.lambda_I * MSELoss(gen_imgs,real_imgs) + \
                                    args.lambda_G *((BCELoss(discriminator(gen_imgs), fake) + BCELoss(discriminator(real_imgs),valid))/2)
                                    
                    g_loss.backward(retain_graph=True)
                    optimizer_G.step()
                    
                #calculate ber loss
                decoded_info = decoded_info.detach()
                u            = u.detach()
                this_ber = errors_ber(decoded_info, u)
                
                if i%100 == 0:
                    
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [batch Dec BER: %f]" % (epoch, args.num_epoch, i, len(train_dataloader),
                                                                        d_loss.item(), g_loss.item(), this_ber))
                
                batches_done = epoch * len(train_dataloader) + i
                #saving log
                if batches_done == 0:
                    filewriter.writerow(['Batchnumber','D loss','Dec loss','G loss','ber loss'])
                    filewriter.writerow([batches_done,d_loss.item(),dec_loss.item(),g_loss.item(),this_ber])
                else:
                    filewriter.writerow([batches_done,d_loss.item(),dec_loss.item(),g_loss.item(),this_ber])
                    
                
                if batches_done % args.sample_interval == 0:
                    save_image(gen_imgs.data[:25], 'images/'+identity+'/%d.png' % batches_done, nrow=5, normalize=True)
                    
            
            
    # --------------------------
    #  Testing: only for BER
    # --------------------------
    ber_count = 0.0
    count = 0
    for i, (imgs, _) in enumerate(test_dataloader):
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        if valid.shape[0]!=args.batch_size:
            continue

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        #encoded message
        u = torch.randint(0, 2, (args.batch_size, args.block_len), dtype=torch.float).to(device)
        #x = torch.zeros(args.img_size)
        #u = torch.add(u,1,x)
        #u = u.unsqueeze_(-1)
        #encodable_u = u.expand(args.batch_size,args.block_len,args.img_size,args.img_size)
        #print(u.shape)

        # Generate a batch of images
        gen_imgs = generator(real_imgs, u)

        # Loss measures generator's ability to fool the discriminator
        gen_imgs = channel(gen_imgs, args.noise_std, channel_type = args.channel_type, device = device)

        decoded_info = decoder(gen_imgs)

        decoded_info = decoded_info.detach()
        u            = u.detach()

        decode_ber = errors_ber(decoded_info, u)
        ber_count += decode_ber
        count += 1

    print('The BER of image code is,', ber_count/count)
    print('model id is', identity)
    print(args)
