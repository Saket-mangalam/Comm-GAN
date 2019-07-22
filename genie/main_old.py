from __future__ import print_function
# from get_args import get_args
from get_args_qr import get_args
import os
import csv
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

from utils import errors_ber, weights_init
from channels import channel, channel_test

from fidscore import calculate_fid

def get_encoder():

    from encoder import WholeEncoder as Encoder
    return Encoder

def get_decoder():

    if args.dec_type == 'turboae2d':
        from decoder import TurboAE_decoder2D as Decoder
    elif args.dec_type == 'turboae1d':
        from decoder import TurboAE_decoder1D as Decoder
    elif args.dec_type == 'cnn2d':
        from decoder import CNN_decoder2D as Decoder

    return Decoder

def get_D():
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
    print('p_array is',p_array )
    args.p_array = p_array


    ####################################################################
    # Network Setup
    ####################################################################

    Encoder          = get_encoder()
    Decoder          = get_decoder()
    EncDiscriminator = get_D()

    # Decide which device we want to run on
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create the encoder
    netE = Encoder(args, p_array).to(device)

    if args.model_id is 'default':
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
    netDec = Decoder(args, p_array).to(device)


    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    if args.model_id is 'default':
        #netDec.apply(weights_init)
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
    criterion = nn.BCELoss()
    img_Loss  = nn.MSELoss()

    # Optimizers
    optimizerED  = optim.Adam(netED.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    optimizerE   = optim.Adam(netE.parameters(), lr=args.codec_lr)
    optimizerDec = optim.Adam(netDec.parameters(), lr=args.codec_lr)

    ####################################################################
    # Data
    ####################################################################
    # Get Data Loader
    dataloader = get_data(args)


    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_u     = torch.randint(0, 2, (args.batch_size, args.code_rate_k ,args.img_size,args.img_size),
                                dtype=torch.float, device=device)
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
                #real_cpu_quantize       = quantizer(real_cpu, args)
                #real_cpu_quantize_noise = bsc(real_cpu_quantize, args.bsc_p, device)

                labelr     = torch.full((args.batch_size,), real_label, device=device)
                labelf     = torch.full((args.batch_size,), fake_label, device=device)

                #######################################################################
                # Train enc discriminator (D2), maximize log(D(x)) + log(1 - D(G(z))),
                # Idea is to discriminate encoded image/non-info image.
                #######################################################################
                for run in range(args.num_train_D2):
                    netED.zero_grad()
                    u_message = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    routput     = netED(real_cpu).view(-1)
                    errED_real1 = criterion(routput, labelr)
                    errED_real1.backward()

                    fake_enc = netE(u_message, real_cpu)
                    foutput = netED(fake_enc.detach()).view(-1)
                    errED_fake2 = criterion(foutput, labelf)
                    errED_fake2.backward()

                    errED_2 = errED_fake2 - errED_real1

                    errED = errED_2

                    optimizerED.step()
                #######################################################################
                # Train decoder, maximize Dec Info Loss, with BCE
                #######################################################################

                for run in range(args.num_train_Dec):
                    netDec.zero_grad()

                    u_message     = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)
                    # repeatutive code with code rate 1/3

                    fake_enc    = netE(u_message, real_cpu)
                    #nfake_enc   = channel(fake_enc.detach(), args.awgn, args)
                    # AWGN Noise on image:
                    noise         = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    encoded = fake_enc + noise
                    nfake_enc = torch.clamp(encoded, -1.0, +1.0)
                    nfake_noiseless = fake_enc

                    foutput     = netDec(nfake_enc.detach())
                    foutput_noiseless = netDec(nfake_noiseless.detach())

                    # noisy
                    errDec_fakeenc = criterion(foutput,u_message)
                    errDec_fakeenc.backward()

                    # noiseless
                    errDec_fakeenc_noiseless = criterion(foutput_noiseless, u_message)  # both noiseless and noisy decoding should be there.
                    errDec_fakeenc_noiseless.backward()

                    errDec = errDec_fakeenc

                    optimizerDec.step()

                    fber = errors_ber(foutput, u_message)
                    ber = fber.item()

                    fber_noiseless = errors_ber(foutput_noiseless, u_message)
                    ber_noiseless = fber_noiseless.item()

                #######################################################################
                # Train Encoder+Generator, minimize
                # Encoder should encode real+fake images
                #######################################################################

                for run in range(args.num_train_Enc):
                    netE.zero_grad()

                    u     = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float, device=device)

                    ############################
                    # D2 Discriminator Loss
                    ############################
                    # forward pass real encoded
                    enc_img = netE(u, real_cpu)
                    foutput = netED(enc_img).view(-1)
                    errGD_fake1 = criterion(foutput, labelr)

                    errE_critic = errGD_fake1

                    ############################
                    # Decoder Loss
                    ############################
                    # forward pass for decoder loss
                    # AWGN Noise on image:
                    noise         = args.enc_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                    encoded = enc_img + noise
                    nenc_img = torch.clamp(encoded, -1.0, +1.0)

                    # noisy
                    u1 = netDec(nenc_img)
                    u1 = torch.clamp(u1, 0, 1)
                    errE_dec = criterion(u1,u)

                    # noiseless
                    u2 = netDec(encoded)
                    u2 = torch.clamp(u2, 0, 1)
                    errE_dec_noiseless = criterion(u2, u)

                    # weight different losses.
                    errE = args.enc_lambda_D*errE_critic + args.enc_lambda_Dec*(errE_dec + errE_dec_noiseless)
                    errE.backward()

                    optimizerE.step()

                #######################################################################
                # Output training stats
                #######################################################################
                if i % 100 == 0:
                    if args.img_channels == 3:
                        fidgen = calculate_fid(real_cpu, enc_img, cuda=use_cuda, dims=2048)
                    else:
                        fidgen = 0.0

                    print('[%d/%d][%d/%d]\tLoss_ED: %.4f\tLoss_Dec: %.4f\tBER_Loss: %.4f \tBER_noiseless:%.4f \tFID_score:%.4f'
                              % (epoch, args.num_epoch, i, len(dataloader),
                                  errED.item(), errDec.item(), ber,ber_noiseless, fidgen ))

                    if epoch>=0:
                        if ber_noiseless<0.01:
                            args.enc_lambda_D = min(0.5, args.enc_lambda_D + 0.01)
                            args.enc_lambda_Dec  = 1.0 - args.enc_lambda_D
                            print('BER satisfied, increase weight for D to %.4f\t, decrease weight for Dec to %.4f\t'
                                  %(args.enc_lambda_D, args.enc_lambda_Dec))
                        elif ber_noiseless>0.02:
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
                        v_u        = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), dtype=torch.float,device=device)
                        # repeative code

                        v_fake_enc = netE(v_u, real_cpu).detach()

                        noise   = args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                        encoded = v_fake_enc + noise

                        real_cpu_noised = real_cpu + args.dec_noise * torch.randn(fake_enc.shape, dtype=torch.float).to(device)
                        img_encode_mask  = v_fake_enc - real_cpu
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
                if batches_done == 0:
                    filewriter.writerow(
                            ['Batchnumber', 'Dec Loss', 'ED Loss', 'BER Loss'])
                    filewriter.writerow(
                        [batches_done, errDec, errED, ber])
                else:
                    filewriter.writerow(
                        [batches_done, errDec, errED,  ber])

                iters += 1

    # save models
    torch.save(netED.state_dict(), 'EDiscriminators/' + identity + '.pt')
    torch.save(netE.state_dict(), 'Encoders/' + identity + '.pt')
    torch.save(netDec.state_dict(), 'Decoders/' + identity + '.pt')

    # Initialize for evaluation
    netED.eval()
    netE.eval()
    netDec.eval()

    #testing parameters
    print('\nTESTING FOR DIFFERENT SNRs NOW>>>>>>>>')
    num_test_epochs = 10

    for i, data in enumerate(dataloader, 0):
        # Format batch
        real_cpu                = data[0].to(device)
        break

    print('Fake/Real image test on awgn channel')
    noise_ratio = [item*0.01 for item in range(11)]

    with open('test_ber/' + identity+'_awgn' + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        # sample a batch of real image

        real_ber, fake_ber = [], []
        for snr in noise_ratio:
            real_this_ber, fake_this_ber = 0.0, 0.0
            for epoch in range(num_test_epochs):
                with torch.no_grad():

                    u = torch.randint(0, 2, (args.batch_size, args.code_rate_k, args.img_size, args.img_size), device=device)

                    fake_enc = netE(u, real_cpu)
                    noisy_fake = channel_test(fake_enc,snr, args, mode = 'awgn')
                    output = netDec(noisy_fake)
                    fake_this_ber += errors_ber(output,u).item()

                    real_enc = netE(u, real_cpu)
                    noisy_real = channel_test(fake_enc,snr, args, mode = 'awgn')
                    output = netDec(noisy_real)
                    real_this_ber += errors_ber(output,u).item()

            real_avg_ber = real_this_ber / num_test_epochs
            fake_avg_ber = fake_this_ber / num_test_epochs
            real_ber.append(real_avg_ber)
            fake_ber.append(fake_avg_ber)

            print('AWGN ber for snr : %.2f \t is %.4f (real) and  %.4f (fake)' %(snr, real_avg_ber,fake_avg_ber ))
            filewriter.writerow([snr , real_avg_ber])

    print('fake ber', fake_ber)
    print('real ber', real_ber)
    print('mode id is',identity)

