'''
Yihan Jiang's Update 04/15/2019 on new QR code.
(1) Dataset has only grayscale quantized dataset.
(2) Channel has only BSC/BEC.

'''
from __future__ import print_function
# from get_args import get_args
from get_args_qr import get_args
import os
import csv
import random
import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

from utils import errors_ber, weights_init
from channels import channel, channel_test
from quantizer import quantizer, bsc, add_qr # a differentiable quantizer.
from fidscore import calculate_fid

# Set random seem for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


# import arguments
args = get_args()
nc = args.img_channels
nz = args.zlatent
bs = args.batch_size
#ud = int(args.bitsperpix/args.code_rate)  # bitsperpix is 3, code rate is 3, then the actual bits perpix is 1
ud = int(args.bitsperpix)
im = args.img_size
im_u = args.info_img_size
#import initial training ratios suggested by trainer
encdec_wt = args.enc_lambda_Dec
encdisc_wt = args.enc_lambda_D
encmse_wt = args.enc_mse_wt
gend1_wt = args.G_lambda_D1
gend2_wt = args.G_lambda_D2
gendec_wt = args.G_lambda_Dec

######################################################
##### Select models #################################
#######################################################

# G
if args.gtype == 'dcgan':
    from Generators import DCGANGenerator as Generator

# Enc
if args.etype == 'basic':
    from Encoders import Basic_Encoder as Encoder
elif args.etype == 'res':
    from Encoders import Residual_Encoder as Encoder
elif args.etype == 'dense':
    from Encoders import Dense_Encoder as Encoder
elif args.etype == 'dres':
    from Encoders import DenseRes_Encoder as Encoder
elif args.etype == 'turbo_dres':
    from TurboCodec import  TurboEncoder as Encoder
elif args.etype == 'sadense':
    from Encoders import SADense_Encoder as Encoder

# Dec
if args.dectype == 'basic':
    from Decoders import Basic_Decoder as Decoder
elif args.dectype == 'dense':
    from Decoders import Dense_Decoder as Decoder
elif args.dectype == 'turbo_dense':
    from TurboCodec import TurboDecoder as Decoder

#D2
if args.d2type == 'dcgan':
    from Discriminators import DCGANDiscriminator as EncDiscriminator
elif args.d2type == 'sngan':
    from Discriminators import SNGANDiscriminator as EncDiscriminator
elif args.d2type == 'sagan':
    from Discriminators import SAGANDiscriminator as EncDiscriminator

#D1
if args.d1type == 'dcgan':
    from Discriminators import DCGANDiscriminator as GanDiscriminator
elif args.d1type == 'sngan':
    from Discriminators import SNGANDiscriminator as GanDiscriminator
elif args.d1type == 'sagan':
    from Discriminators import SAGANDiscriminator as GanDiscriminator

######################################################################
# Data
######################################################################
if args.img_channels == 1: # grayscale
    if args.data == 'test3':
        dataset = dset.ImageFolder(root='./data/test3',
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.Resize(args.img_size),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
    elif args.data == 'celeba':
        dataset = dset.ImageFolder(root='./data/celeba',
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.Resize(args.img_size),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
    elif args.data == 'mnist':
        dataloader = torch.utils.data.DataLoader(
            dset.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Grayscale(num_output_channels=1),
                           transforms.Resize(args.img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    elif args.data == 'cifar10':
        dataloader = torch.utils.data.DataLoader(
            dset.CIFAR10('./data/cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Grayscale(num_output_channels=1),
                             transforms.Resize(args.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        print('coco and lsun not supported yet!')
else: # 3 channels colorful picture
    if args.data == 'test3':
        dataset = dset.ImageFolder(root='./data/test3',
                                   transform=transforms.Compose([
                                       #transforms.Grayscale(num_output_channels=3),
                                       transforms.Resize(args.img_size),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
    elif args.data == 'celeba':
        dataset = dset.ImageFolder(root='./data/celeba',
                                   transform=transforms.Compose([
                                       transforms.Resize(args.img_size),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
    elif args.data == 'mnist':
        dataloader = torch.utils.data.DataLoader(
            dset.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize(args.img_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    elif args.data == 'cifar10':
        dataloader = torch.utils.data.DataLoader(
            dset.CIFAR10('./data/cifar10', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(args.img_size),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ])),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    else:
        print('coco and lsun not supported yet!')




# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

# Create the generator
netG = Generator(args).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netG = nn.DataParallel(netG, list(range(args.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
if args.model_id is 'default':
    netG.apply(weights_init)
    pass
else:
    try:
        pretrained_model = torch.load('./Generators/'+args.model_id+'.pt')
        try:
            netG.load_state_dict(pretrained_model.state_dict())
        except:
            netG.load_state_dict(pretrained_model)
    except:
        print('G weight not match, random init')
# Print the model
print(netG)

# Create the encoder
netE = Encoder(args).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netE = nn.DataParallel(netE, list(range(args.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
if args.model_id is 'default':
    netE.apply(weights_init)
    pass
else:
    try:
        pretrained_model = torch.load('./Encoders/'+args.model_id+'.pt')
        try:
            netE.load_state_dict(pretrained_model.state_dict())
        except:
            netE.load_state_dict(pretrained_model)
    except:
        print('Encoder weight not match, random init')

# Print the model
print(netE)

# Create the decoder
netDec = Decoder(args).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netDec = nn.DataParallel(netDec, list(range(args.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
if args.model_id is 'default':
    netDec.apply(weights_init)
    pass
else:
    try:
        pretrained_model = torch.load('./Decoders/'+args.model_id+'.pt')
        try:
            netDec.load_state_dict(pretrained_model.state_dict())
        except:
            netDec.load_state_dict(pretrained_model)
    except:
        print('Decoder weight not match, random init')

# Print the model
print(netDec)

# Create the gan discriminator
netGD = GanDiscriminator(args).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netGD = nn.DataParallel(netGD, list(range(args.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
if args.model_id is 'default':
    netGD.apply(weights_init)
else:
    try:
        pretrained_model = torch.load('./GDiscriminators/'+args.model_id+'.pt')
        try:
            netGD.load_state_dict(pretrained_model.state_dict())
        except:
            netGD.load_state_dict(pretrained_model)
    except:
        print('D1 weight not match, random init')

# Print the model
print('Net D1',netGD)

# Create the enc discriminator
netED = EncDiscriminator(args).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netED = nn.DataParallel(netED, list(range(args.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
if args.model_id is 'default':
    netED.apply(weights_init)
    pass
else:
    try:
        pretrained_model = torch.load('./EDiscriminators/'+args.model_id+'.pt')
        try:
            netED.load_state_dict(pretrained_model.state_dict())
        except:
            netED.load_state_dict(pretrained_model)
    except:
        print('D2 weight not match, random init')

# Print the model
print('Net D2', netED)


# Initialize Loss function
criterion = nn.BCELoss()
img_Loss  = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(args.batch_size, nz, 1, 1, dtype=torch.float,device=device)
fixed_u     = torch.randint(0, 2, (args.batch_size, ud,im_u,im_u), dtype=torch.float, device=device)
fixed_u = fixed_u/255
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerGD  = optim.Adam(netGD.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerED  = optim.Adam(netED.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerG   = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerE   = optim.Adam(netE.parameters(), lr=args.codec_lr)
optimizerDec = optim.Adam(netDec.parameters(), lr=args.codec_lr)

iters = 0
identity = str(np.random.random())[2:8]
print(identity)
#os.makedirs('images', exist_ok=True)
os.makedirs('images/' + identity)
# directory for saving loss log
#os.makedirs('logbook', exist_ok=True)
#os.makedirs('EDiscriminators', exist_ok=True)
#os.makedirs('GDiscriminators', exist_ok=True)
#os.makedirs('Generators', exist_ok=True)
#os.makedirs('Encoders', exist_ok=True)
#os.makedirs('Decoders', exist_ok=True)
#os.makedirs('test_ber', exist_ok=True)

#############################################################
######## training #########################
print("Starting Training Loop...")
# For each epoch
#
with open('logbook/' + identity + '.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for epoch in range(args.num_epoch):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # Format batch

            real_cpu                = data[0].to(device)
            #real_cpu_quantize       = quantizer(real_cpu, args)
            #real_cpu_quantize_noise = bsc(real_cpu_quantize, args.bsc_p, device)
            b_size   = real_cpu.size(0)

            noise = torch.randn(b_size, nz, 1, 1, dtype=torch.float,device=device)
            labelr     = torch.full((b_size,), real_label, device=device)
            labelf     = torch.full((b_size,), fake_label, device=device)

            #normalize = transforms.Normalize(mean=0.5, std=0.5)
            #######################################################################
            # Train gan discriminator (D1). maximize log(D(x)) + log(1 - D(G(z)))
            #######################################################################

            for run in range(args.num_train_D1):
                netGD.zero_grad()

                # forward pass real batch
                nreal_cpu = channel(real_cpu.detach(), args.awgn, args)
                routput    = netGD(nreal_cpu).view(-1)
                errGD_real = criterion(routput, labelr)
                errGD_real.backward()

                # forward pass fake batch
                fake_img   = netG(noise)
                nfake_img = channel(fake_img.detach(), args.awgn, args)
                foutput    = netGD(nfake_img.detach()).view(-1)
                errGD_fake = criterion(foutput, labelf)
                errGD_fake.backward()

                errGD = errGD_fake - errGD_real

                optimizerGD.step()

            #######################################################################
            # Train enc discriminator (D2), maximize log(D(x)) + log(1 - D(G(z))),
            # Idea is to discriminate encoded image/non-info image.
            #######################################################################

            for run in range(args.num_train_D2):
                netED.zero_grad()

                u     = torch.randint(0, 2, (b_size, ud, im_u, im_u), dtype=torch.float, device=device)
                #ustar = u/255

                # 1st pass: real batch, not encoded, QR code quantize/add noise the image
                routput     = netED(nreal_cpu).view(-1)
                errED_real1 = criterion(routput, labelr)
                errED_real1.backward()

                # 2nd pass: real image ,encoded
                enc_img = netE(real_cpu, u)
                nenc_img = channel(enc_img.detach(), args.awgn, args)
                foutput = netED(nenc_img.detach()).view(-1)
                errED_fake1 = criterion(foutput, labelf)
                errED_fake1.backward()

                errED_1 = errED_fake1 - errED_real1

                # 3rd pass fake image, not encoded, use quantized version.
                fake_img   = netG(noise)
                nfake_img = channel(fake_img.detach(), args.awgn, args)
                routput    = netED(nfake_img.detach()).view(-1)
                errED_real2 = criterion(routput, labelr)
                errED_real2.backward()

                # 4th pass fake enc img batch, encoded
                fake_enc = netE(fake_img.detach(), u)
                nfake_enc = channel(fake_enc.detach(), args.awgn, args)
                foutput = netED(nfake_enc.detach()).view(-1)
                errED_fake2 = criterion(foutput, labelf)
                errED_fake2.backward()

                errED_2 = errED_fake2 - errED_real2

                errED = (errED_1 + errED_2)/2.0

                optimizerED.step()
            #######################################################################
            # Train decoder, maximize Dec Info Loss, with BCE
            #######################################################################

            for run in range(args.num_train_Dec):
                netDec.zero_grad()

                u     = torch.randint(0, 2, (b_size, ud, im_u, im_u), dtype=torch.float, device=device)
                #ustar = u/255

                # forward pass fake encoded images
                # add noise to image
                fake_img    = netG(noise)
                fake_enc    = netE(fake_img.detach(), u)
                nfake_enc   = channel(fake_enc.detach(), args.awgn, args)
                foutput     = netDec(nfake_enc.detach())
                #foutput     = foutput*255
                #foutput = normalize(foutput)
                errDec_fakeenc = criterion(foutput,u)
                errDec_fakeenc.backward()

                fber = errors_ber(foutput, u)

                #forward pass real encoded images
                # add noise to enc image
                enc_img = netE(real_cpu,u)
                nenc_img = channel(enc_img.detach(), args.awgn, args)
                routput = netDec(nenc_img)
                #routput = routput*255
                #routput = normalize(routput)
                errDec_realenc = criterion(routput,u)
                errDec_realenc.backward()
                rber  = errors_ber(routput, u)

                errDec = (errDec_fakeenc + errDec_realenc)/2.0
                #errDec.backward()

                ber = (fber.item() + rber.item())/2.0

                optimizerDec.step()

            #######################################################################
            # Train Encoder, minimize
            # Encoder should encode real+fake images
            #######################################################################

            for run in range(args.num_train_Enc):
                netE.zero_grad()

                u     = torch.randint(0, 2, (b_size, ud, im_u, im_u), dtype=torch.float, device=device)
                #ustar = u/255

                ############################
                # D2 Discriminator Loss
                ############################
                # forward pass real encoded
                enc_img = netE(real_cpu, u)
                nenc_img = channel(enc_img, args.awgn, args)
                foutput = netED(nenc_img).view(-1)
                errGD_fake1 = criterion(foutput, labelr)

                #forward pass real encoded
                fake_img   = netG(noise)
                fake_enc   = netE(fake_img.detach(), u)
                nfake_enc = channel(fake_enc, args.awgn, args)
                routput = netED(nfake_enc).view(-1)
                errGD_fake2 = criterion(routput, labelr)

                errE_critic = errGD_fake1 + errGD_fake2

                ############################
                # Decoder Loss
                ############################
                # forward pass for decoder loss
                #nenc_img = channel(enc_img, args.awgn, args)
                u1 = netDec(nenc_img)
                #u1 = u1*255
                #u1 = normalize(u1)

                errDec_fake1 = criterion(u1,u)

                # forward pass for decoder loss
                #nfake_enc  = channel(fake_enc, args.awgn, args)
                u2 = netDec(nfake_enc)
                #u2 = u2*255
                #u2 = normalize(u2)
                errDec_fake2 = criterion(u2,u)

                errE_dec = errDec_fake1 + errDec_fake2

                # forward pass image reconstruction loss
                recon_loss = (img_Loss(nfake_img,nfake_enc) + img_Loss(nreal_cpu, nenc_img))/2.0

                # weight different losses.
                errE = encdisc_wt*errE_critic + encdec_wt*errE_dec + encmse_wt*recon_loss
                errE.backward()

                optimizerE.step()

            ########################################################################
            # Train Generator, minimize shit
            # Yihan: This part minimize GD loss, Dec loss, but no EncD loss. Which need some thinking
            ########################################################################

            for run in range(args.num_train_G):
                netG.zero_grad()

                u     = torch.randint(0, 2, (b_size, ud, im_u, im_u), dtype=torch.float, device=device)
                #ustar = u/255

                # forward pass fake batch,D1 loss
                fake_img   = netG(noise)
                nfake_img = channel(fake_img, args.awgn, args)
                foutput    = netGD(nfake_img).view(-1)
                errGDisc   = criterion(foutput, labelr)

                # forward pass fake batch, D2 loss
                fake_enc    = netE(fake_img, u)
                nfake_enc = channel(fake_enc.detach(), args.awgn, args)
                foutput     = netED(nfake_enc).view(-1)
                errEDisc1   = criterion(foutput, labelr)

                # forward pass fake batch, D2 loss.
                foutput     = netED(nfake_img).view(-1)
                errEDisc2   = criterion(foutput, labelf)

                errEDisc = errEDisc1 + errEDisc2
                # forward pass fake batch, Dec loss
                # forward pass encoded fake batch
                u3 = netDec(channel(fake_enc, args.awgn, args))
                #u3 = u3*255
                #u3 = normalize(u3)
                errGDec = criterion(u3, u)

                errG = gend1_wt*errGDisc  + gend2_wt*errEDisc + gendec_wt * errGDec

                errG.backward()

                optimizerG.step()

            #####################################################################
            ##  ADPTIVE TRAINING : SAKET VERSION#######################
            #### LETS HOPE THIS WORKS ##########
            #####################################################################
            fidgen=0
            fidenc=0
            #if epoch>10:
            #    fidgen = calculate_fid(real_cpu, fake_img, cuda=True, dims=2048)
            #    fidenc1 = calculate_fid(real_cpu, enc_img, cuda=True, dims=2048)
            #    fidenc2 = calculate_fid(fake_img, fake_enc, cuda=True, dims=2048)
            #    fidenc = (fidenc1+fidenc2)/2
                #if (0.5*math.exp(-0.2*epoch))>ber and epoch>10:
                    #this means decoder loss is not going down
                    #calculate fid for generator

                    #continue

            #normalize to sum 1 generator's lambdas and encoder's lambdas separately
            #sumgen = gend1_wt + gend2_wt + gendec_wt
            #sumenc = encmse_wt + encdec_wt + encdisc_wt
            #gend1_wt = float(gend1_wt)/sumgen
            #gend2_wt = float(gend2_wt)/sumgen
            #gendec_wt = float(gendec_wt)/sumgen
            #encdisc_wt = float(encdisc_wt)/sumenc
            #encdec_wt = float(encdec_wt)/sumenc
            #encmse_wt = float(encmse_wt)/sumenc


            #######################################################################
            # Output training stats
            #######################################################################
            if i % 50 == 0:
                if args.num_train_D1 and args.num_train_D2 and args.num_train_Enc and args.num_train_Dec and args.num_train_G:
                    print('[%d/%d][%d/%d]\tLoss_GD: %.4f\tLoss_ED: %.4f\tLoss_G: %.4f\tLoss_Dec: %.4f\tLoss_Enc: %.4f\tBER_Loss: %.4f\tFID gen: %.4f\tFID enc: %.4f'
                          % (epoch, args.num_epoch, i, len(dataloader),
                             errGD.item(), errED.item(), errG.item(), errDec.item(), errE.item(), ber, fidgen, fidenc))
                else:
                    print('[%d/%d][%d/%d]\tBER:%.4f'% (epoch, args.num_epoch, i, len(dataloader), ber))
                    errGD, errG, errDec, errED, errE = 0.0,0.0, 0.0, 0.0, 0.0

            batches_done = epoch * len(dataloader) + i

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 30 == 0) or ((epoch == args.num_epoch - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    v_noise    = torch.randn(real_cpu.shape[0], nz, 1, 1, dtype=torch.float,device=device)
                    v_u        = torch.randint(0, 2, (real_cpu.shape[0], ud, im_u, im_u), dtype=torch.float,device=device)
                    #v_u = v_u/255

                    v_fake     = netG(v_noise).detach()
                    v_fake_enc = netE(v_fake, v_u).detach()

                    v_real     = real_cpu
                    v_real_enc = netE(v_real, v_u).detach()
                    v_real     = v_real.detach()

                    v_fake_enc_rec = channel(v_fake_enc, args.awgn, args)
                    v_real_enc_rec = channel(v_real_enc, args.awgn, args)

                # save image.
                save_image(v_fake.data[:25], 'images/' + identity + '/%d_fake.png' % batches_done, nrow=5,
                           normalize=True, pad_value=1.0)
                save_image(v_fake_enc.data[:25], 'images/' + identity + '/%d_fake_enc.png' % batches_done, nrow=5,
                           normalize=True, pad_value=1.0)

                save_image(v_real.data[:25], 'images/' + identity + '/%d_real.png' % batches_done, nrow=5,
                           normalize=True, pad_value=1.0)
                save_image(v_real_enc.data[:25], 'images/' + identity + '/%d_real_enc.png' % batches_done, nrow=5,
                           normalize=True, pad_value=1.0)

                save_image(v_fake_enc_rec.data[:25], 'images/' + identity + '/%d_fake_enc_rec.png' % batches_done, nrow=5,
                           normalize=True, pad_value=1.0)
                save_image(v_real_enc_rec.data[:25], 'images/' + identity + '/%d_real_enc_rec.png' % batches_done, nrow=5,
                           normalize=True, pad_value=1.0)

            # saving log
            if batches_done == 0:
                filewriter.writerow(
                        ['Batchnumber', 'GD loss', 'G loss', 'Dec Loss', 'ED Loss', 'E Loss', 'BER Loss', 'FID Gen', 'FID Enc'])
                filewriter.writerow(
                    [batches_done, errGD.item(), errG.item(), errDec.item(), errED.item(), errE.item(), ber, fidgen, fidenc])
            else:
                filewriter.writerow(
                    [batches_done, errGD.item(), errG.item(), errDec.item(), errED.item(), errE.item(), ber, fidgen, fidenc])

            iters += 1

# save models
torch.save(netGD.state_dict(), 'GDiscriminators/' + identity + '.pt')
torch.save(netED.state_dict(), 'EDiscriminators/' + identity + '.pt')
torch.save(netE.state_dict(), 'Encoders/' + identity + '.pt')
torch.save(netG.state_dict(), 'Generators/' + identity + '.pt')
torch.save(netDec.state_dict(), 'Decoders/' + identity + '.pt')

#print weights ratios
print("encoder weight",encdisc_wt,encdec_wt,encmse_wt)
print("generator weight", gend1_wt,gend2_wt,gendec_wt)
# Initialize for evaluation
netG.eval()
netGD.eval()
netED.eval()
netE.eval()
netDec.eval()

#testing parameters
print('\nTESTING FOR DIFFERENT SNRs NOW>>>>>>>>')
num_test_epochs = 100

for i, data in enumerate(dataloader, 0):
    # Format batch
    real_cpu                = data[0].to(device)
    break

print('Fake/Real image test on awgn channel')
noise_ratio = [item*0.05 for item in range(11)]

with open('test_ber/' + identity+'_awgn' + '.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    # sample a batch of real image

    real_ber, fake_ber = [], []
    for snr in noise_ratio:
        real_this_ber, fake_this_ber = 0.0, 0.0
        for epoch in range(num_test_epochs):
            with torch.no_grad():

                noise = torch.randn(args.batch_size, nz, 1, 1, device=device)
                u = torch.randint(0, 2, (args.batch_size, ud, im_u, im_u), device=device)
                #ustar = u/255
                fake = netG(noise)

                fake_enc = netE(fake,u)
                noisy_fake = channel_test(fake_enc,snr, args, mode = 'awgn')
                output = netDec(noisy_fake)
                #output = output*255
                fake_this_ber += errors_ber(output,u).item()

                real_enc = netE(real_cpu,u)
                noisy_real = channel_test(real_enc,snr, args, mode = 'awgn')
                output = netDec(noisy_real)
                #output = output*255
                real_this_ber += errors_ber(output,u).item()

        real_avg_ber = real_this_ber / num_test_epochs
        fake_avg_ber = fake_this_ber / num_test_epochs
        real_ber.append(real_avg_ber)
        fake_ber.append(fake_avg_ber)

        print('AWGN ber for snr : %.2f \t is %.4f (real) and  %.4f (fake)' %(snr, real_avg_ber,fake_avg_ber ))
        filewriter.writerow([snr , real_avg_ber])

print('fake ber', fake_ber)
print('real ber', real_ber)

# print('test on BSC channel')
# noise_ratio = [item*0.05 for item in range(11)]
# with open('test_ber/' + identity+'_awgn' + '.csv', 'w') as csvfile:
#     filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
#
#     for snr in noise_ratio:
#         ber = 0.0
#         for epoch in range(num_test_epochs):
#             with torch.no_grad():
#
#                 noise = torch.randn(args.batch_size, nz, 1, 1, device=device)
#                 u = torch.randint(0, 2, (args.batch_size, ud, im_u, im_u), device=device)
#                 fake = netG(noise)
#                 fake_enc = netE(fake,u)
#                 noisy_fake = channel_test(fake_enc, snr, args, mode = 'bsc')
#                 output = netDec(noisy_fake)
#                 ber += errors_ber(output,u).item()
#
#         avg_ber = ber / num_test_epochs
#         print('BSC ber for snr : %.2f \t is %.4f' %(snr, avg_ber))
#         filewriter.writerow([snr , avg_ber])



