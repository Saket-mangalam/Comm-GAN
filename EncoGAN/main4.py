

from __future__ import print_function
#%matplotlib inline
from get_args import get_args
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
from utils import channel, errors_ber, weights_init

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
ud = args.bitsperpix
im = args.img_size

######################################################
##### Select models #################################
#######################################################
if args.gtype == 'dcgan':
    from Generators import DCGANGenerator as Generator

if args.etype == 'basic':
    from Encoders import Basic_Encoder as Encoder
elif args.etype == 'res':
    from Encoders import Residual_Encoder as Encoder
elif args.etype == 'dense':
    from Encoders import Dense_Encoder as Encoder

if args.dectype == 'basic':
    from Decoders import Basic_Decoder as Decoder
elif args.dectype == 'dense':
    from Decoders import Dense_Decoder as Decoder

if args.dtype == 'dcgan':
    from Discriminators import DCGANDiscriminator as EncDiscriminator
    from Discriminators import DCGANDiscriminator as GanDiscriminator

######################################################################
# Data
# ----
#
# We can use an image folder dataset the way we have it setup.
# Create the dataset
# Create the dataset
if args.data == 'celeba':
    dataset = dset.ImageFolder(root=args.list_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(args.img_size),
                                   transforms.CenterCrop(args.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=args.num_workers)

# os.makedirs(args.list_dir, exist_ok=True)
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
                         # transforms.CenterCrop(args.img_size),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ])),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

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
    pretrained_model = torch.load('./Generators/'+args.model_id+'.pt')
    try:
        netG.load_state_dict(pretrained_model.state_dict())
    except:
        netG.load_state_dict(pretrained_model)

# Print the model
#print(netG)

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
    pretrained_model = torch.load('./Encoders/'+args.model_id+'.pt')
    try:
        netE.load_state_dict(pretrained_model.state_dict())
    except:
        netE.load_state_dict(pretrained_model)

# Print the model
#print(netE)

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
    pretrained_model = torch.load('./Decoders/'+args.model_id+'.pt')
    try:
        netDec.load_state_dict(pretrained_model.state_dict())
    except:
        netDec.load_state_dict(pretrained_model)

# Print the model
#print(netDec)

# Create the gan discriminator
netGD = GanDiscriminator(args).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (args.ngpu > 1):
    netGD = nn.DataParallel(netGD, list(range(args.ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
if args.model_id is 'default':
    netGD.apply(weights_init)
    pass
else:
    pretrained_model = torch.load('./GDiscriminators/'+args.model_id+'.pt')
    try:
        netGD.load_state_dict(pretrained_model.state_dict())
    except:
        netGD.load_state_dict(pretrained_model)

# Print the model
#print(netGD)

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
    pretrained_model = torch.load('./EDiscriminators/'+args.model_id+'.pt')
    try:
        netED.load_state_dict(pretrained_model.state_dict())
    except:
        netED.load_state_dict(pretrained_model)

# Print the model
#print(netED)


# Initialize Loss function
criterion = nn.BCELoss()
img_Loss = nn.MSELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fixed_u = torch.randint(0, 2, (64, ud,im,im), dtype=torch.float, device=device)
# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerGD = optim.Adam(netGD.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerED = optim.Adam(netED.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerE = optim.Adam(netE.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
optimizerDec = optim.Adam(netDec.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))

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

    for epoch in range(args.end_epoch):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            u = torch.randint(0, 2, (b_size, ud, im, im), dtype=torch.float, device=device)






            ########## Train gan discriminator. maximize log(D(x)) + log(1 - D(G(z))) ##############
            netGD.zero_grad()
            # forward pass real batch
            labelr = torch.full((b_size,), real_label, device=device)
            routput = netGD(real_cpu).view(-1)
            errGD_real = criterion(routput, labelr)
            errGD_real.backward()

            #forward pass fake batch
            labelf = torch.full((b_size,), fake_label, device=device)
            fake_img = netG(noise)
            foutput = netGD(fake_img.detach()).view(-1)
            errGD_fake = criterion(foutput, labelf)
            errGD_fake.backward()

            errGD = errGD_fake - errGD_real
            #errGD.backward()

            optimizerGD.step()

            ########## Train enc discriminator, maximize log(D(x)) + log(1 - D(G(z))) ##############
            netED.zero_grad()
            #forward pass real batch
            routput = netED(real_cpu).view(-1)
            errED_real1 = criterion(routput, labelr)
            errED_real1.backward()

            # forward pass fake batch
            enc_img = netE(real_cpu,u)
            foutput = netED(enc_img.detach()).view(-1)
            errED_fake1 = criterion(foutput, labelf)
            errED_fake1.backward()

            errED_1 = errED_fake1 - errED_real1
            #errED.backward()

            #forward pass fake image batch
            routput = netED(fake_img.detach()).view(-1)
            errED_real2 = criterion(routput, labelr)
            errED_real2.backward()

            #forward pass fake enc img batch
            fake_enc = netE(fake_img.detach(), u)
            foutput = netED(fake_enc.detach()).view(-1)
            errED_fake2 = criterion(foutput, labelf)
            errED_fake2.backward()

            errED_2 = errED_fake2 - errED_fake1

            errED = (errED_1 + errED_2)/2.0

            optimizerED.step()

            ########## Train decoder, maximize decinfoloss ################################
            netDec.zero_grad()
            # forward pass fake encoded images
            # add noise to image
            nfake_enc = channel(fake_enc.detach(), args.noise)
            foutput = netDec(nfake_enc.detach())
            errDec_fakeenc = criterion(foutput,u)
            errDec_fakeenc.backward()
            fber = errors_ber(foutput, u)

            #forward pass real encoded images
            # add noise to enc image
            nenc_img = channel(enc_img.detach(), args.noise)
            routput = netDec(nenc_img)
            errDec_realenc = criterion(routput,u)
            errDec_realenc.backward()
            rber  = errors_ber(routput, u)

            errDec = (errDec_fakeenc + errDec_realenc)/2.0
            #errDec.backward()

            ber = (fber.item() + rber.item())/2.0

            optimizerDec.step()

            ########## Train Encoder, minimize ##########################################
            netE.zero_grad()
            # forward pass fake batch
            foutput = netED(enc_img).view(-1)
            errGD_fake1 = criterion(foutput, labelr)
            #errGD_fake1.backward(retain_graph=True)

            #forward pass real batch
            routput = netED(fake_enc).view(-1)
            errGD_fake2 = criterion(routput, labelr)
            #errGD_fake2.backward(retain_graph=True)

            errE_critic = errGD_fake1 + errGD_fake2

            # forward pass for decoder loss
            u1 = netDec(enc_img)
            errDec_fake1 = criterion(u1,u)
            #errDec_fake1.backward()

            # forward pass for decoder loss
            u2 = netDec(fake_enc)
            errDec_fake2 = criterion(u2,u)
            #errDec_fake2.backward()

            errE_dec = errDec_fake1 + errDec_fake2

            # forward pass image reconstruction loss
            recon_loss = img_Loss(enc_img,fake_img.detach())


            errE = args.lambda_D*errE_critic + args.lambda_Dec*errE_dec + args.mse_wt*recon_loss
            errE.backward()

            optimizerE.step()

            ######### Train Generator, minimize shit ####################################
            netG.zero_grad()
            # forward pass fake batch
            foutput = netGD(fake_img).view(-1)
            errGDisc = criterion(foutput, labelr)

            fake_enc = netE(fake_img, u)
            # forward pass encoded fake batch
            u3 = netDec(fake_enc)
            errGDec = criterion(u3, u)

            errG = errGDisc + errGDec

            errG.backward()

            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                # decoded_u = netDec(fake).detach()

                print('[%d/%d][%d/%d]\tLoss_GD: %.4f\tLoss_ED: %.4f\tLoss_G: %.4f\tLoss_Dec: %.4f\tLoss_Enc: %.4f\tBER_Loss: %.4f'
                      % (epoch, args.end_epoch, i, len(dataloader),
                         errGD.item(), errED.item(), errG.item(), errDec.item(), errE.item(), ber))

            batches_done = epoch * len(dataloader) + i

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == args.end_epoch - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    v_noise = torch.randn(64, nz, 1, 1, device=device)
                    v_u = torch.randint(0, 2, (64, ud, im, im), device=device)
                    v_fake = netG(v_noise).detach()
                    v_enc = netE(v_fake, v_u).detach()
                    fake = netG(fixed_noise).detach()
                    fake_enc = netE(fake, fixed_u).detach()

                save_image(fake.data[:25], 'images/' + identity + '/fake%d.png' % batches_done, nrow=5,
                           normalize=True)
                save_image(fake_enc.data[:25], 'images/' + identity + '/fake_enc%d.png' % batches_done, nrow=5,
                           normalize=True)
                save_image(v_fake.data[:25], 'images/' + identity + '/v_fake%d.png' % batches_done, nrow=5,
                           normalize=True)
                save_image(v_enc.data[:25], 'images/' + identity + '/v_enc%d.png' % batches_done, nrow=5,
                           normalize=True)

            # saving log
            if batches_done == 0:
                filewriter.writerow(
                        ['Batchnumber', 'GD loss', 'G loss', 'Dec Loss', 'ED Loss', 'E Loss', 'BER Loss'])
                filewriter.writerow(
                    [batches_done, errGD, errG, errDec, errED, errE, ber])
            else:
                filewriter.writerow(
                    [batches_done, errGD, errG, errDec, errED, errE, ber])

            iters += 1

# save models
torch.save(netGD.state_dict(), 'GDiscriminators/' + identity + '.pt')
torch.save(netED.state_dict(), 'EDiscriminators/' + identity + '.pt')
torch.save(netE.state_dict(), 'Decoders/' + identity + '.pt')
torch.save(netG.state_dict(), 'Generators/' + identity + '.pt')
torch.save(netDec.state_dict(), 'Decoders/' + identity + '.pt')

# Initialize for evaluation
netG.eval()
netGD.eval()
netED.eval()
netE.eval()
netDec.eval()

#testing parameters
print('\nTESTING FOR DIFFERENT SNRs NOW>>>>>>>>')
num_test_epochs = 10

noise_ratio = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
with open('test_ber/' + identity + '.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for snr in noise_ratio:
        ber = 0.0
        for epoch in range(num_test_epochs):
            with torch.no_grad():

                noise = torch.randn(args.batch_size, nz, 1, 1, device=device)
                u = torch.randint(0, 2, (args.batch_size, ud * nc, im, im), device=device)
                fake = netG(noise)
                fake_enc = netE(fake,u)
                noisy_fake = channel(fake_enc, snr)
                output = netDec(noisy_fake)
                ber += errors_ber(output,u).item()

        avg_ber = ber / num_test_epochs
        print('ber for snr : %.1f \t is %.4f' %(snr, avg_ber))
        filewriter.writerow([snr , avg_ber])



