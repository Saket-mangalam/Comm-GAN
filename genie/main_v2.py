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
from config import config
from data_loader import get_data
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

from utils import errors_ber, weights_init, pos_ber, pos_mse, psnr
from channels import channel, channel_test

from fidscore import calculate_fid



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
    configuration = config(args)
    CCE              = configuration.get_cce()
    IE               = configuration.get_ie()
    CDec             = configuration.get_cdec()
    IDec             = configuration.get_idec()
    EncDiscriminator = configuration.get_D()

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
    netED = EncDiscriminator(args).to(device)

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

    BCE_loss = nn.BCELoss()
    MSE_loss = nn.MSELoss()
    img_Loss  = nn.MSELoss()

    # Optimizers
    optimizerED  = optim.Adam(netED.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
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
                    nreal_cpu = channel_test(real_cpu, args)
                    routput     = netED(real_cpu).view(-1)
                    errED_real1 = BCE_loss(routput, labelr)
                    errED_real1.backward()

                    if args.use_cce:
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                        x_message = netCCE(u_message)
                    else: # pre-train the whole channel, no cce. co
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                        x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)

                    fake_enc = netIE(x_message, real_cpu)

                    foutput = netED(fake_enc.detach()).view(-1)
                    errED_fake2 = BCE_loss(foutput, labelf)
                    errED_fake2.backward()

                    errED = errED_fake2 - errED_real1

                    #errED = errED_2

                    optimizerED.step()

                #######################################################################
                # Train Channel decoder, maximize Dec Info Loss, with BCE
                #######################################################################

                for run in range(args.num_train_CDec):
                    netCDec.zero_grad()

                    if args.use_cce:
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                        x_message = netCCE(u_message)

                    else: # pre-train the whole channel, no cce. co
                        break  # no need to train CDec when not using CCE.

                    fake_enc = netIE(x_message, real_cpu)
                    # channels, AWGN for now.
                    #noise    = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    #encoded  = fake_enc + noise
                    encoded = channel_test(fake_enc, args)
                    nfake_enc = torch.clamp(encoded, -1.0, +1.0)
                    nfake_noiseless = fake_enc

                    f_idec  = netIDec(nfake_enc.detach())
                    foutput = netCDec(nfake_enc.detach(), f_idec)

                    f_idc_noiseless   = netIDec(nfake_noiseless.detach())
                    foutput_noiseless = netCDec(nfake_noiseless.detach(), f_idc_noiseless)

                    # noisy
                    errDec_fakeenc = BCE_loss(foutput,u_message)
                    errDec_fakeenc.backward()

                    # noiseless
                    errDec_fakeenc_noiseless = BCE_loss(foutput_noiseless, u_message)  # both noiseless and noisy decoding should be there.
                    errDec_fakeenc_noiseless.backward()

                    errDec = errDec_fakeenc

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

                    if args.use_cce:
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                        x_message = netCCE(u_message)
                    else: # pre-train the whole channel, no cce. co
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                        #x_message = u_message
                        x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)

                    fake_enc = netIE(x_message, real_cpu)
                    # channels, AWGN for now.
                    #noise    = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    #encoded  = fake_enc + noise
                    encoded = channel_test(fake_enc, args)
                    nfake_enc = torch.clamp(encoded, -1.0, +1.0)
                    nfake_noiseless = fake_enc

                    f_idec  = netIDec(nfake_enc.detach())
                    if not args.use_cce:
                        foutput = f_idec
                    else:
                        foutput = netCDec(nfake_enc.detach(), f_idec)

                    f_idc_noiseless   = netIDec(nfake_noiseless.detach())
                    if not args.use_cce:
                        foutput_noiseless = f_idc_noiseless
                    else:
                        foutput_noiseless = netCDec(nfake_noiseless.detach(), f_idc_noiseless)

                    # noisy
                    if args.use_cce:
                        errDec_fakeenc = BCE_loss(foutput,u_message)
                    else:
                        errDec_fakeenc = MSE_loss(foutput,x_message)

                    errDec_fakeenc.backward()

                    # noiseless
                    if args.use_cce:
                        errDec_fakeenc_noiseless = BCE_loss(foutput_noiseless, u_message)  # both noiseless and noisy decoding should be there.
                    else:
                        errDec_fakeenc_noiseless = MSE_loss(foutput_noiseless, x_message)

                    errDec_fakeenc_noiseless.backward()
                    #errDec_fakeenc.backward()
                    errDec = errDec_fakeenc + errDec_fakeenc_noiseless

                    optimizerIDec.step()

                    if args.use_cce:
                        fber = errors_ber(foutput, u_message)
                        ber = fber.item()

                        fber_noiseless = errors_ber(foutput_noiseless, u_message)
                        ber_noiseless = fber_noiseless.item()
                    else:
                        fber = MSE_loss(foutput, x_message)
                        ber = fber.item()

                        fber_noiseless = MSE_loss(foutput_noiseless, x_message)
                        ber_noiseless = fber_noiseless.item()

                    #find psrn
                    psnrdec = psnr(fake_enc[0], real_cpu[0])

                #######################################################################
                # Train Image Encoder, minimize
                # Encoder should encode real+fake images
                #######################################################################

                for run in range(args.num_train_IE):
                    netIE.zero_grad()

                    if args.use_cce:
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                        x_message = netCCE(u_message)
                    else: # pre-train the whole channel, no cce. co
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                        #x_message = u_message
                        x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)


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
                    #noise    = args.enc_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    #encoded  = enc_img + noise
                    nenc_img = channel_test(enc_img, args)
                    nenc_img = torch.clamp(encoded, -1.0, +1.0)

                    # noisy
                    nu = netIDec(nenc_img)
                    if not args.use_cce:
                        errE_dec = MSE_loss(nu,x_message)
                        mse_img_noisy = errE_dec.item()
                    else:
                        u1 = netCDec(nenc_img, nu)
                        u1 = torch.clamp(u1, 0, 1)
                        errE_dec = BCE_loss(u1,u_message)

                    # noiseless
                    uu = netIDec(enc_img)
                    if not args.use_cce:
                        errE_dec_noiseless = MSE_loss(uu, x_message)
                        mse_img_noiseless = errE_dec_noiseless.item()
                    else:
                        u2 = netCDec(encoded, uu)
                        u2 = torch.clamp(u2, 0, 1)
                        errE_dec_noiseless = BCE_loss(u2, u_message)

                    # weight different losses.
                    errE = args.enc_lambda_D*errE_critic + args.enc_lambda_Dec*(errE_dec + errE_dec_noiseless)
                    errE.backward(retain_graph=True)


                    optimizerIE.step()

                #######################################################################
                # Train Channel Coding Encoder
                # Encoder should encode real+fake images
                #######################################################################

                for run in range(args.num_train_CCE):
                    netCCE.zero_grad()

                    if args.use_cce:
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                        x_message = netCCE(u_message)
                    else: # pre-train the whole channel, no cce. co
                        #break
                        u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                        #x_message = u_message
                        x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)

                    ############################
                    # Decoder Loss
                    ############################
                    # forward pass for decoder loss
                    # AWGN Noise on image:

                    enc_img  = netIE(x_message, real_cpu)

                    #noise    = args.enc_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    #encoded  = enc_img + noise
                    encoded = channel_test(enc_img, args)
                    nenc_img = torch.clamp(encoded, -1.0, +1.0)

                    # noisy
                    uu = netIDec(nenc_img)
                    if not args.use_cce:
                        u1 = uu
                    else:
                        u1 = netCDec(nenc_img, uu)
                        u1 = torch.clamp(u1, 0, 1)

                    if args.use_cce:
                        errE_dec = BCE_loss(u1,u_message)
                    else:
                        errE_dec = MSE_loss(u1,x_message)

                    # noiseless
                    uu = netIDec(encoded)
                    if not args.use_cce:
                        u2 = uu
                    else:
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

                    if args.use_cce:
                        print('[%d/%d][%d/%d]\tLoss_ED: %.4f\tLoss_Dec: %.4f\tMSE_Loss: %.4f \tMSE_noiseless:%.4f \tFID_score:%.4f'
                                  % (epoch, args.num_epoch, i, len(dataloader),
                                      errED.item(), errDec.item(), mse_img_noisy.item(), mse_img_noiseless.item(), fidgen ))
                    else:
                        print('[%d/%d][%d/%d]\tLoss_Dec: %.4f\tBER_Loss: %.4f \tBER_noiseless:%.4f \tFID_score:%.4f \tNoisyMSE:%.4f \tNLessMSE:%.4f'
                                  % (epoch, args.num_epoch, i, len(dataloader),
                                    errDec.item(), ber,ber_noiseless, fidgen, mse_img_noisy, mse_img_noiseless ))


                    if epoch>=0 and not args.use_cce:
                        if mse_img_noiseless<args.ber_thd_low:
                        #if ber_noiseless<args.ber_thd_low:
                            args.enc_lambda_D = min(0.5, args.enc_lambda_D + 0.01)
                            args.enc_lambda_Dec  = 1.0 - args.enc_lambda_D
                            print('BER satisfied, increase weight for D to %.4f\t, decrease weight for Dec to %.4f\t'
                                  %(args.enc_lambda_D, args.enc_lambda_Dec))
                        elif mse_img_noiseless>args.ber_thd_high:
                        #elif ber_noiseless>args.ber_thd_high:
                            args.enc_lambda_Dec  = min(0.99, args.enc_lambda_Dec + 0.01)
                            args.enc_lambda_D = 1.0 - args.enc_lambda_Dec
                            print('BER bad, decrease weight for D to %.4f\t, increase weight for Dec to %.4f\t'
                                  %(args.enc_lambda_D, args.enc_lambda_Dec))
                        else:
                            if np.random.random()>0.99:
                                args.enc_lambda_D = min(0.5, args.enc_lambda_D + 0.01)
                                args.enc_lambda_Dec  = 1.0 - args.enc_lambda_D
                                print('Random, increase weight for D to %.4f\t, decrease weight for Dec to %.4f\t'%(args.enc_lambda_D, args.enc_lambda_Dec))
                            elif np.random.random()<0.01:
                                args.enc_lambda_Dec  = min(0.99, args.enc_lambda_Dec + 0.01)
                                args.enc_lambda_D = 1.0 - args.enc_lambda_Dec
                                print('Random, decrease weight for D to %.4f\t, increase weight for Dec to %.4f\t'
                                      %(args.enc_lambda_D, args.enc_lambda_Dec))


                batches_done = epoch * len(dataloader) + i


                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == args.num_epoch - 1) and (i == len(dataloader) - 1)):
                    with torch.no_grad():
                        if args.use_cce:
                            u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                            x_message = netCCE(u_message)
                        else: # pre-train the whole channel, no cce. co
                            u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                            x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)



                        # AWGN channnel
                        #noise      = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                        #encoded    = v_fake_enc + noise
                        #encoded = channel_test(v_fake_enc, args)
                        #real_cpu_noised = real_cpu + args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                        real_cpu_noised = channel_test(real_cpu, args)
                        v_fake_enc = netIE(x_message, real_cpu_noised).detach()
                        img_encode_mask  = v_fake_enc - real_cpu
                    # save image.

                    save_image(v_fake_enc.data[:25], 'images/' + identity + '/%d_fake_enc.png' % batches_done, nrow=5,
                               normalize=True, pad_value=1.0)
                    #save_image(encoded.data[:25], 'images/' + identity + '/%d_fake_enc_rec.png' % batches_done, nrow=5,
                    #           normalize=True, pad_value=1.0)

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
                                ['Batchnumber', 'Dec Loss', 'ED Loss', 'BER noisy Loss', 'BER Noiseless loss', 'Decoder frac', 'Discriminator frac'])
                        filewriter.writerow(
                            [batches_done, errDec, errED, ber, ber_noiseless, args.enc_lambda_Dec, args.enc_lambda_D])
                    else:
                        filewriter.writerow(
                            [batches_done, errDec, errED,  ber, ber_noiseless, args.enc_lambda_Dec, args.enc_lambda_D])
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
    num_test_epochs = 10

    for i, data in enumerate(dataloader, 0):
        # Format batch
        real_cpu                = data[0].to(device)
        break

    print('Fake/Real image test on awgn channel')
    noise_ratio = [item*0.01 for item in range(11)]

    ############################
    # Decoder Loss
    ############################

    if args.print_pos_ber:
        #sample a batch of real image
        num_test_batch = 10000.0
        with torch.no_grad():
            for idx in range(int(num_test_batch)):
                if args.use_cce:
                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    x_message = netCCE(u_message)
                else: # pre-train the whole channel, no cce. co
                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                    #x_message = u_message
                    x_message = torch.randn((args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)


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



        pos_ber_tes = pos_ber_tes/num_test_batch
        print(pos_ber_tes.cpu().numpy().tolist()[0])
        print(pos_ber_tes.cpu().numpy().shape)

        with open('logbook/' + identity+'haha' + '.txt', 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(pos_ber_tes.cpu().numpy().tolist()[0])
            filewriter.writerow(pos_ber_tes.cpu().numpy().tolist()[1])
            filewriter.writerow(pos_ber_tes.cpu().numpy().tolist()[2])


    real_ber = []
    for snr in noise_ratio:
        real_this_ber, fake_this_ber = 0.0, 0.0
        for epoch in range(num_test_epochs):
            with torch.no_grad():
                if args.use_cce:
                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    x_message = netCCE(u_message)
                else: # pre-train the whole channel, no cce. co
                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_n, args.img_size, args.img_size), dtype=torch.float, device=device)
                    x_message = u_message


                real_enc   = netIE(x_message, real_cpu)
                noisy_fake = channel_test(real_enc, snr, args, mode = 'awgn')

                output_t   = netIDec(noisy_fake)
                if not args.use_cce:
                    output = output_t
                else:
                    output = netCDec(noisy_fake, output_t)

                real_this_ber  += errors_ber(output,u_message).item()


        real_avg_ber = real_this_ber / num_test_epochs
        real_ber.append(real_avg_ber)

        print('AWGN ber for snr : %.2f \t is %.4f (real)' %(snr, real_avg_ber ))

    print('real ber', real_ber)
    print('mode id is',identity)

