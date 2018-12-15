# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:29:20 2018

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
    #initialize generator for gan
    from generators import Gan_Generator as GanGenerator
    #initialize generator for encoder
    from generators import Enc_Generator as EncGenerator
    #initialize discriminator for gan
    from discriminators import Gan_Discriminator as GanDiscriminator
    #initialize discriminator for encoder
    from discriminators import Enc_Discriminator as EncDiscriminator
    #initialize decoder
    from decoders import Enc_Decoder as Decoder
    
    gangenerator     = GanGenerator(args)
    encgenerator     = EncGenerator(args)
    decoder          = Decoder(args)
    gandiscriminator = GanDiscriminator(args)
    encdiscriminator = EncDiscriminator(args)
    
    if cuda:
        gangenerator.cuda()
        encgenerator.cuda()
        gandiscriminator.cuda()
        encdiscriminator.cuda()
        decoder.cuda()
        BCELoss.cuda()
        MSELoss.cuda()
    else:
        print('models', gangenerator, encgenerator, gandiscriminator, encdiscriminator, decoder)

    # Initialize weights
    gangenerator.apply(weights_init_normal)
    encgenerator.apply(weights_init_normal)
    gandiscriminator.apply(weights_init_normal)
    encdiscriminator.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
    
    #########################################
    #### Dataloader #########################
    #########################################
    if args.dataset == 'cifar10':
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
    
    #optimizers
    optimizer_ganG   = torch.optim.Adam(gangenerator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_encG   = torch.optim.Adam(encgenerator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_Dec    = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_ganD   = torch.optim.Adam(gandiscriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_encD   = torch.optim.Adam(encdiscriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    
    #initialize tensor
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    ##############################
    #  Training ##################
    ##############################
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
                #noise input for gan
                real_noise = Variable(0.8 * torch.randn((args.batch_size,args.sample_noise), dtype=torch.float).to(device))
                #real images
                real_imgs = Variable(imgs.type(Tensor))
                #encoded message
                u = torch.randint(0, 2, (args.batch_size, args.block_len), dtype=torch.float).to(device)
                
                optimizer_ganG.zero_grad()
                optimizer_ganD.zero_grad()
                optimizer_Dec.zero_grad()
                optimizer_ganD.zero_grad()
                optimizer_encD.zero_grad()
                
                #################################
                # Generator to create images ####
                #################################
                for j in range(args.num_train_G):
                    
                    #generate image
                    gen_imgs = gangenerator(real_noise)
                    #pass through channel
                    gen_imgs = channel(gen_imgs, args.noise_std, channel_type = args.channel_type, device = device)
                    
                    #for every iteration of G train discriminator n times
                    for k in range(args.num_train_D):
                        
                        # Discriminate real or fake
                        valid_gen = gandiscriminator(real_imgs)
                        fake_gen = gandiscriminator(gen_imgs)
                        #train discriminator only
                        real_loss = BCELoss(valid_gen,valid)
                        fake_loss = BCELoss(fake_gen, fake)
                        gand_loss = (real_loss + fake_loss) / 2
                        gand_loss.backward(retain_graph=True)
                        optimizer_ganD.step()
                        
                    #train the  gan generator now
                    gang_loss = (args.lambda_G)*MSELoss(gen_imgs,real_imgs) + \
                            (1-args.lambda_G)*((BCELoss(gandiscriminator(gen_imgs), fake) + BCELoss(gandiscriminator(real_imgs),valid))/2)
                    
                    gang_loss.backward(retain_graph=True)
                    optimizer_ganG.step()
                    
                    #create an encoded image
                    enc_gan_imgs = encgenerator(gen_imgs,u)
                    enc_real_imgs = encgenerator(real_imgs,u)
                    
                    #for every iteration of encoding train discriminator n times
                    for l in range(args.num_train_D):
                        
                        #discriminate encoded or not
                        valid_enc = encdiscriminator(real_imgs)
                        fake_gan_enc = encdiscriminator(enc_gan_imgs)
                        fake_real_enc = encdiscriminator(enc_real_imgs)
                        #train discriminator
                        encd_loss = (BCELoss(valid_enc,valid)+BCELoss(fake_gan_enc,fake)+BCELoss(fake_real_enc,fake))/3
                        encd_loss.backward(retain_graph=True)
                        optimizer_encD.step()
                        
                    #for every iteration of encoding train decoder n times
                    for m in range(args.num_train_Dec):
                        
                        #decode info
                        decoded_info_1 = decoder(enc_gan_imgs)
                        decoded_info_2 = decoder(enc_real_imgs)
                        # train decoder only
                        dec_loss =  (BCELoss(decoded_info_1, u) + BCELoss(decoded_info_2, u))/2
                        dec_loss.backward(retain_graph=True)
                        optimizer_Dec.step()
                        
                    #train the enc generator now
                    encg_loss = args.lambda_I*(MSELoss(enc_gan_imgs,gen_imgs)+MSELoss(enc_real_imgs,real_imgs))/2 + \
                            args.lambda_G*(BCELoss(encdiscriminator(real_imgs),valid)+BCELoss(encdiscriminator(enc_gan_imgs),fake)+BCELoss(encdiscriminator(enc_real_imgs),fake))/3 + \
                            (1-args.lambda_I-args.lambda_G)*((BCELoss(decoder(enc_gan_imgs), u) + BCELoss(decoder(enc_real_imgs), u))/2)
                    
                    encg_loss.backward(retain_graph=True)
                    optimizer_ganD.step()
                    
                #calculate ber loss
                decoded_info_1 = decoded_info_1.detach()
                decoded_info_2 = decoded_info_2.detach()
                u            = u.detach()
                this_ber = (errors_ber(decoded_info_1, u) + errors_ber(decoded_info_2, u))/2
                
                if i%100 == 0:
                    
                    print ("[Epoch %d/%d] [Batch %d/%d] [Gan D loss: %f] [Gan G loss: %f] [Enc D loss: %f] [Enc G Loss: %f] [batch Dec BER: %f]" % (epoch, args.num_epoch, i, len(train_dataloader),
                                                                        gand_loss.item(), gang_loss.item(),encd_loss.item(),encg_loss.item(), this_ber))
                    
                batches_done = epoch * len(train_dataloader) + i
                #saving log
                if batches_done == 0:
                    filewriter.writerow(['Batchnumber','Gan D loss','Dec loss','Gan G loss','Enc D Loss','Enc G Loss','ber loss'])
                    filewriter.writerow([batches_done,gand_loss.item(),dec_loss.item(),gang_loss.item(),encd_loss.item(),encg_loss.item(),this_ber.item()])
                else:
                    filewriter.writerow([batches_done,gand_loss.item(),dec_loss.item(),gang_loss.item(),encd_loss.item(),encg_loss.item(),this_ber.item()])
                    
                #saving generated image
                if batches_done % args.sample_interval == 0:
                    save_image(enc_gan_imgs.data[:25], 'images/'+identity+'/%d.png' % batches_done, nrow=5, normalize=True)
                    
    
    ###############
    # BER LOSS ####
    ###############
    '''need to design this'''
    
    print('model id is', identity)
    print(args)