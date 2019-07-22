from __future__ import print_function
# from get_args import get_args
from get_args_qr import get_args
import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from utils import errors_ber, weights_init, pos_ber, pos_mse
from channels import channel, channel_test

from fidscore import calculate_fid

# Configure Channel Coding Encoder
def get_cce(args):

    if args.cce_type == 'turboae1d':
        from cce import CCE_Turbo_Encoder1D as CCE

    elif args.cce_type == 'turboae2d': # 2D turbo encoder
        from cce import CCE_Turbo_Encoder2D as CCE
    elif args.cce_type == 'turboae2d_img': # 2D turbo encoder
        from cce import CCE_Turbo_Encoder2D_img as CCE

    elif args.cce_type == 'cnn2d':
        from cce import CCE_CNN_Encoder2D as CCE
    elif args.cce_type == 'cnn2d_img':
        from cce import CCE_CNN_Encoder2D_img as CCE

    elif args.cce_type == 'repeat':
        from cce import CCE_Repeat as CCE


    return CCE

# Configure Image Encoder
def get_ie(args):

    if args.ie_type == 'sadense':
        from ie import IE_SADense as IE
    elif args.ie_type == 'dense':
        from ie import IE_Dense as IE

    return IE

# Configure Image Decoder
def get_idec(args):
    if args.idec_type == 'sadense':
        from idec import SADense_Decoder as IDec
    elif args.idec_type == 'dense':
        from idec import Dense_Decoder as IDec

    return IDec

# Configure Channel Coding Decoder
def get_cdec(args):

    if args.cdec_type == 'turboae2d' or args.cdec_type == 'turboae2d_img':
        from cdec import TurboAE_decoder2D as CDec

    elif args.cdec_type == 'turboae1d':
        from cdec import TurboAE_decoder1D as CDec
    elif args.cdec_type == 'cnn2d':
        from cdec import CNN_decoder2D as CDec

    return CDec

def get_D(args):
    #D2
    # if args.d2type == 'dcgan':
    #     from Discriminators import DCGANDiscriminator as EncDiscriminator
    # elif args.d2type == 'sngan':
    #     from Discriminators import SNGANDiscriminator as EncDiscriminator
    # elif args.d2type == 'sagan':
    #     from Discriminators import SAGANDiscriminator as EncDiscriminator
    # else:
    #     print('Discriminator not specified!')

    from discriminator import DCGANDiscriminator as EncDiscriminator

    return EncDiscriminator


def get_data(args):
    ######################################################################
    # Data
    ######################################################################
    if args.img_channels == 1: # grayscale
        if args.data == 'test':
            dataset = dset.ImageFolder(root='./data/test',
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
        if args.data == 'test':
            dataset = dset.ImageFolder(root='./data/test',
                                       transform=transforms.Compose([
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

        elif args.data == 'coco':
            dataloader = torch.utils.data.DataLoader(
                dset.CocoCaptions('./data/coco/', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(args.img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


        else:
            print('coco and lsun not supported yet!')

    return dataloader


if __name__ == '__main__':

    args = get_args()

    # Set random seem for reproducibility
    manualSeed = 999
    # manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    # For Interleavers
    from numpy import arange
    from numpy.random import mtrand
    seed = np.random.randint(0, 1)
    rand_gen = mtrand.RandomState(seed)
    p_array = rand_gen.permutation(arange(args.img_size**2))
    #print('p_array is',p_array )
    args.p_array = p_array


    ####################################################################
    # Network Setup
    ####################################################################

    CCE              = get_cce(args)
    IE               = get_ie(args)
    CDec             = get_cdec(args)
    IDec             = get_idec(args)
    EncDiscriminator = get_D(args)

    # Decide which device we want to run on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the encoder
    netCCE = CCE(args, p_array).to(device)

    if args.model_id is 'default':
        pass
    else:
        try:
            pretrained_model = torch.load('./CCE/'+args.model_id+'.pt')
            try:
                netCCE.load_state_dict(pretrained_model.state_dict())
            except:
                netCCE.load_state_dict(pretrained_model)
        except:
            print('CCE weight not match, random init')

    # Print the model
    print(netCCE)

    # Create the encoder
    netIE = IE(args).to(device)

    if args.model_id is 'default':
        pass
    else:
        try:
            pretrained_model = torch.load('./IE/'+args.model_id+'.pt')
            try:
                netIE.load_state_dict(pretrained_model.state_dict())
            except:
                netIE.load_state_dict(pretrained_model)
        except:
            print('IE weight not match, random init')

    # Print the model
    print(netIE)

    # Create the Image Decoder
    netIDec = IDec(args).to(device)

    if args.model_id is 'default':
        pass
    else:
        try:
            pretrained_model = torch.load('./IDec/'+args.model_id+'.pt')
            try:
                netIDec.load_state_dict(pretrained_model.state_dict())
            except:
                netIDec.load_state_dict(pretrained_model)
        except:
            print('Image Decoder weight not match, random init')

    # Print the model
    print(netIDec)


    # Create the Channel decoder
    netCDec = CDec(args, p_array).to(device)

    if args.model_id is 'default':
        pass
    else:
        try:
            pretrained_model = torch.load('./CDec/'+args.model_id+'.pt')
            try:
                netCDec.load_state_dict(pretrained_model.state_dict())
            except:
                netCDec.load_state_dict(pretrained_model)
        except:
            print('Channel Decoder weight not match, random init')

    # Print the model
    print(netCDec)

    # Create the enc discriminator
    if args.num_train_D2>0:
        netED = EncDiscriminator(args).to(device)

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
        optimizerED  = optim.Adam(netED.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

    # Initialize Loss function

    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()

    # Optimizers

    optimizerCCE   = optim.Adam(netCCE.parameters(), lr=args.codec_lr)
    optimizerIE   = optim.Adam(netIE.parameters(), lr=args.codec_lr)
    optimizerIDec = optim.Adam(netIDec.parameters(), lr=args.codec_lr)
    optimizerCDec = optim.Adam(netCDec.parameters(), lr=args.codec_lr)
    ####################################################################
    # Data
    ####################################################################
    # Get Data Loader
    dataloader = get_data(args)


    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_u     = torch.randint(0, 2, (args.batch_size, args.code_rate_k ,args.img_size,args.img_size),
                                dtype=torch.float, device=device)
    for i, data in enumerate(dataloader, 0):
        fixed_img  = data[0].to(device)
        fixed_one_img = fixed_img[10, :, :, :]
        fixed_img = torch.stack([fixed_one_img for _ in range(args.batch_size)], dim=0)
        print(fixed_img.shape)

        break
    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    iters = 0
    identity = str(np.random.random())[2:8]
    print(identity)
    os.makedirs('images/' + identity)

    #############################################################
    ######## training #########################
    print("Starting Training Loop...")
    labelr     = torch.full((args.batch_size,), real_label, device=device)
    labelf     = torch.full((args.batch_size,), fake_label, device=device)
    # For each epoch
    #
    with open('logbook/' + identity + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for epoch in range(args.num_epoch):
            # For each batch in the dataloader
            for i, data in enumerate(dataloader, 0):
                # Format batch
                real_cpu                = data[0].to(device)
                if real_cpu.shape[0]!=args.batch_size:
                    print('batch size mismatch!')
                    continue

                #######################################################################
                # Train enc discriminator (D2), maximize log(D(x)) + log(1 - D(G(z))),
                # Idea is to discriminate encoded image/non-info image.
                #######################################################################
                for run in range(args.num_train_D2):
                    netED.zero_grad()

                    routput     = netED(real_cpu).view(-1)
                    errED_real1 = BCE_loss(routput, labelr)
                    errED_real1.backward()

                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    x_message = netCCE(u_message, real_cpu)

                    fake_enc = netIE(x_message, real_cpu)

                    foutput = netED(fake_enc.detach()).view(-1)
                    errED_fake2 = BCE_loss(foutput, labelf)
                    errED_fake2.backward()

                    errED_2 = errED_fake2 - errED_real1

                    errED = errED_2

                    optimizerED.step()
                #######################################################################
                # Train Channel decoder, maximize Dec Info Loss, with BCE
                #######################################################################

                for run in range(args.num_train_CDec):
                    netCDec.zero_grad()

                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    x_message = netCCE(u_message, real_cpu)

                    fake_enc = netIE(x_message, real_cpu)
                    # channels, AWGN for now.
                    noise    = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    encoded  = fake_enc + noise

                    nfake_enc = torch.clamp(encoded, -1.0, +1.0)
                    nfake_noiseless = fake_enc

                    f_idec  = netIDec(nfake_enc.detach())
                    foutput = netCDec(nfake_enc.detach(), f_idec)

                    f_idc_noiseless   = netIDec(nfake_noiseless.detach())
                    foutput_noiseless = netCDec(nfake_noiseless.detach(), f_idc_noiseless)

                    # noisy
                    errDec_fakeenc = BCE_loss(foutput,u_message) + BCE_loss(foutput_noiseless, u_message)
                    errDec_fakeenc.backward()

                    optimizerCDec.step()

                    fber = errors_ber(foutput, u_message)
                    ber = fber.item()

                    fber_noiseless = errors_ber(foutput_noiseless, u_message)
                    ber_noiseless = fber_noiseless.item()


                #######################################################################
                # Train Image Decoder, maximize Dec Info Loss, with BCE
                #######################################################################

                for run in range(args.num_train_IDec):
                    netIDec.zero_grad()

                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    if args.norm_input_ie:
                        x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                    else:
                        x_message = 2.0*torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device) - 1.0

                    fake_enc = netIE(x_message, real_cpu)
                    # channels, AWGN for now.
                    noise    = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    encoded  = fake_enc + noise

                    nfake_enc = torch.clamp(encoded, -1.0, +1.0)
                    nfake_noiseless = fake_enc

                    f_idec  = netIDec(nfake_enc.detach())
                    foutput = f_idec

                    f_idc_noiseless   = netIDec(nfake_noiseless.detach())
                    foutput_noiseless = f_idc_noiseless

                    errDec_fakeenc = MSE_loss(foutput,x_message) + MSE_loss(foutput_noiseless, x_message)
                    errDec_fakeenc.backward()

                    errDec = errDec_fakeenc/2

                    optimizerIDec.step()

                #######################################################################
                # Train Image Encoder, minimize
                # Encoder should encode real+fake images
                #######################################################################

                for run in range(args.num_train_IE):
                    netIE.zero_grad()

                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    if args.norm_input_ie:
                        x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                    else:
                        x_message = 2.0*torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device) - 1.0
                        #x_message = netCCE(u_message, real_cpu)

                    ############################
                    # D2 Discriminator Loss
                    ############################
                    # forward pass real encoded
                    enc_img = netIE(x_message, real_cpu)
                    foutput = netED(enc_img).view(-1)
                    errGD_fake1 = BCE_loss(foutput, labelr)
                    errE_critic = errGD_fake1

                    ############################
                    # Decoder Loss
                    ############################
                    # forward pass for decoder loss
                    # AWGN Noise on image:
                    noise    = args.enc_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    encoded  = enc_img + noise
                    nenc_img = torch.clamp(encoded, -1.0, +1.0)

                    # noisy
                    uu = netIDec(nenc_img)
                    IE_MSE_dec_noisy = MSE_loss(uu,x_message)

                    # noiseless
                    uu = netIDec(enc_img)
                    IE_MSE_dec_noiseless = MSE_loss(uu, x_message)

                    # weight different losses.
                    errE = args.enc_lambda_D*errE_critic + args.enc_lambda_Dec*(IE_MSE_dec_noisy/4 + IE_MSE_dec_noiseless/4)
                    errE.backward()

                    optimizerIE.step()

                #######################################################################
                # Train Channel Coding Encoder
                # Encoder should encode real+fake images
                #######################################################################

                for run in range(args.num_train_CCE):
                    netCCE.zero_grad()

                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    x_message = netCCE(u_message, real_cpu)

                    ############################
                    # Decoder Loss
                    ############################
                    # forward pass for decoder loss
                    # AWGN Noise on image:

                    enc_img  = netIE(x_message, real_cpu)

                    noise    = args.enc_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    encoded  = enc_img + noise
                    nenc_img = torch.clamp(encoded, -1.0, +1.0)

                    # noisy
                    uu = netIDec(nenc_img)
                    u1 = netCDec(nenc_img, uu)
                    u1 = torch.clamp(u1, 0, 1)

                    errE_dec = BCE_loss(u1,u_message)

                    # noiseless
                    uu = netIDec(encoded)
                    u2 = netCDec(encoded, uu)
                    u2 = torch.clamp(u2, 0, 1)

                    errE_dec_noiseless = BCE_loss(u2, u_message)

                    # weight different losses.
                    errE = errE_dec + errE_dec_noiseless
                    errE.backward()

                    optimizerCCE.step()

                #######################################################################
                # Output training stats
                #######################################################################
                if i % 100 == 0:
                    try:
                        if args.img_channels == 3:
                            fidgen = calculate_fid(fixed_img, enc_img, cuda=use_cuda, dims=2048)
                        else:
                            fidgen = MSE_loss(real_cpu, enc_img)/(args.img_channels * args.img_size**2)
                    except:
                        fidgen = 0.0


                    try:
                        print('[%d/%d][%d/%d]\tMSE (noiseless) of Image Coding: %.4f\tMSE (noisy) of Image Coding: %.4f\tBER (noiseless): %.4f \tBER (noisy):%.4f \tFID_score:%.4f'
                                  % (epoch, args.num_epoch, i, len(dataloader),
                                    errDec.item(), IE_MSE_dec_noisy.item(), ber_noiseless, ber,  fidgen ))
                        this_mse = errDec.item()
                    except:
                        try:
                            print('[%d/%d][%d/%d]\tMSE (noiseless) of Image Coding: %.4f\tMSE (noisy) of Image Coding: %.4f \tFID_score:%.4f'
                                      % (epoch, args.num_epoch, i, len(dataloader),
                                        errDec.item(), IE_MSE_dec_noisy.item(), fidgen ))
                            this_mse = errDec.item()
                        except:
                            print('[%d/%d][%d/%d]\tBER (noiseless): %.4f \tBER (noisy):%.4f \tFID_score:%.4f'
                                  % (epoch, args.num_epoch, i, len(dataloader),
                                    ber_noiseless, ber,  fidgen ))




                    # Change to Image Quality weight adaptation?

                    if args.IE_weight_adapt:
                        if fidgen>args.fid_thd_low:
                        #if ber_noiseless<args.ber_thd_low:
                            args.enc_lambda_D = min(0.99, args.enc_lambda_D + 0.01)
                            args.enc_lambda_Dec  = 1.0 - args.enc_lambda_D
                            print('FID not satisfied, increase weight for D to %.4f\t, decrease weight for Dec to %.4f\t'
                                  %(args.enc_lambda_D, args.enc_lambda_Dec))
                        elif fidgen<args.fid_thd_high:
                        #elif ber_noiseless>args.ber_thd_high:
                            args.enc_lambda_Dec  = min(0.99, args.enc_lambda_Dec + 0.01)
                            args.enc_lambda_D = 1.0 - args.enc_lambda_Dec
                            print('FID good, decrease weight for D to %.4f\t, increase weight for Dec to %.4f\t'
                                  %(args.enc_lambda_D, args.enc_lambda_Dec))
                        else:
                            pass


                batches_done = epoch * len(dataloader) + i


                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == args.num_epoch - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        u_message  = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                        x_message  = netCCE(u_message, real_cpu)
                        if args.same_img:
                            real_cpu = fixed_img
                        v_fake_enc = netIE(x_message, real_cpu).detach()
                        # AWGN channnel
                        noise      = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                        encoded    = v_fake_enc + noise

                        real_cpu_noised = real_cpu + args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                        img_encode_mask  = torch.abs(v_fake_enc - real_cpu)
                    # save image.

                    save_image(v_fake_enc.data[:25], 'images/' + identity + '/%d_fake_enc.png' % batches_done, nrow=5,
                               normalize=True, pad_value=1.0)
                    save_image(encoded.data[:25], 'images/' + identity + '/%d_fake_enc_rec.png' % batches_done, nrow=5,
                               normalize=True, pad_value=1.0)

                    save_image(real_cpu.data[:25], 'images/' + identity + '/%d_real.png' % batches_done, nrow=5,
                               normalize=True, pad_value=1.0)
                    save_image(real_cpu_noised.data[:25], 'images/' + identity + '/%d_real_feedtoD.png' % batches_done, nrow=5,
                               normalize=True, pad_value=1.0)

                    save_image(img_encode_mask.data[:25], 'images/' + identity + '/%d_enc_mask_diff.png' % batches_done, nrow=5,
                               normalize=True, pad_value=1.0)

                # saving log
                try:
                    if batches_done == 0:
                        filewriter.writerow(
                                ['Batchnumber', 'Dec Loss', 'ED Loss', 'BER Loss'])
                        filewriter.writerow(
                            [batches_done, errDec, errED, ber])
                    else:
                        filewriter.writerow(
                            [batches_done, errDec, errED,  ber])
                except:
                    pass

                iters += 1

    if args.save_models:
        # save models
        torch.save(netED.state_dict(), 'EDiscriminators/' + identity + '.pt')
        torch.save(netIE.state_dict(), 'IE/' + identity + '.pt')
        torch.save(netCCE.state_dict(), 'CCE/' + identity + '.pt')
        torch.save(netIDec.state_dict(), 'IDec/' + identity + '.pt')
        torch.save(netCDec.state_dict(), 'CDec/' + identity + '.pt')
    # Initialize for evaluation
    netED.eval()
    netIE.eval()
    netCCE.eval()
    netCDec.eval()
    netIDec.eval()

    #testing parameters
    print('\nTESTING FOR DIFFERENT SNRs NOW>>>>>>>>')
    num_test_epochs = args.num_test_epochs

    for i, data in enumerate(dataloader, 0):
        # Format batch
        real_cpu                = data[0].to(device)
        break

    print('Fake/Real image test on awgn channel')
    noise_ratio = [item*0.01 for item in range(11)]

    ############################
    # Decoder Loss
    ############################

    MSE_pos_loss = nn.MSELoss()

    if args.print_pos_ber:
        #sample a batch of real image
        num_test_batch = 10000.0
        with torch.no_grad():
            for idx in range(int(num_test_batch)):
                if args.same_img:
                    real_cpu = fixed_img

                if args.use_cce:
                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    x_message = netCCE(u_message, real_cpu)
                else:
                    if args.norm_input_ie:
                        x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                    else:
                        x_message = 2.0*torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device) - 1.0

                real_enc   = netIE(x_message, real_cpu)
                noisy_fake = real_enc # no noise

                if args.use_cce:
                    output     = netCDec(noisy_fake, netIDec(noisy_fake))
                    if idx == 0:
                        pos_ber_tes    = pos_ber(output, u_message)
                    else:
                        pos_ber_tes += pos_ber(output, u_message)

                else:
                    output     = netIDec(noisy_fake)
                    if idx == 0:
                        pos_ber_tes = pos_mse(output,x_message)
                    else:
                        pos_ber_tes += pos_mse(output,x_message)

        pos_ber_tes = pos_ber_tes/num_test_batch

        with open('logbook/' + identity+'haha' + '.txt', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(pos_ber_tes.cpu().numpy().astype(np.float16).tolist()[0])
            filewriter.writerow(pos_ber_tes.cpu().numpy().astype(np.float16).tolist()[1])
            filewriter.writerow(pos_ber_tes.cpu().numpy().astype(np.float16).tolist()[2])

        with torch.no_grad():
            u_message  = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
            x_message  = netCCE(u_message, real_cpu)
            if args.same_img:
                real_cpu = fixed_img
            v_fake_enc = netIE(x_message, real_cpu).detach()
            # AWGN channnel
            noise      = args.dec_noise * torch.randn(v_fake_enc.shape, dtype=torch.float).to(device)
            encoded    = v_fake_enc + noise

            real_cpu_noised = real_cpu + args.dec_noise * torch.randn(v_fake_enc.shape, dtype=torch.float).to(device)
            img_encode_mask  = torch.abs(v_fake_enc - real_cpu)
        # save image.

        save_image(v_fake_enc.data[:25], 'images/' + identity + '/fake_enc.png', nrow=5,
                   normalize=True, pad_value=1.0)
        save_image(encoded.data[:25], 'images/' + identity + '/fake_enc_rec.png', nrow=5,
                   normalize=True, pad_value=1.0)

        save_image(real_cpu.data[:25], 'images/' + identity + '/real.png', nrow=5,
                   normalize=True, pad_value=1.0)
        save_image(real_cpu_noised.data[:25], 'images/' + identity + '/real_feedtoD.png', nrow=5,
                   normalize=True, pad_value=1.0)

        save_image(img_encode_mask.data[:25], 'images/' + identity + '/enc_mask_diff.png', nrow=5,
                   normalize=True, pad_value=1.0)


    real_ber, real_mse = [], []
    for snr in noise_ratio:
        real_this_ber, real_this_mse = 0.0 , 0.0
        for epoch in range(num_test_epochs):
            with torch.no_grad():
                u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                x_message = netCCE(u_message, real_cpu)

                real_enc   = netIE(x_message, real_cpu)
                noisy_fake = channel_test(real_enc, snr, args, mode = 'awgn')

                output_t   = netIDec(noisy_fake)
                output     = netCDec(noisy_fake, output_t)

                real_this_ber  += errors_ber(output,u_message).item()
                real_this_mse  += MSE_loss(x_message,output_t).item()/(args.img_channels * args.img_size**2)

        real_avg_ber = real_this_ber / num_test_epochs
        real_avg_mse = real_this_mse / num_test_epochs
        real_ber.append(real_avg_ber)
        real_mse.append(real_avg_mse)

        print('AWGN ber for snr : %.2f \t is %.4f (ber) \t is %.4f (MSE)' %(snr, real_avg_ber , real_avg_mse))

    print('real ber', real_ber)
    print('real mse', real_mse)
    print('mode id is',identity)

