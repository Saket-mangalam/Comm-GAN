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

from utils import errors_ber, weights_init, encoded_message, decoded_message
from channels import channel, channel_test
from quantizer import quantizer, bsc, add_qr # a differentiable quantizer.

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
# TEST Data
######################################################################
if args.img_channels == 1: # grayscale
    dataset = dset.ImageFolder(root='./data/'+args.test_folder,
                                   transform=transforms.Compose([
                                       transforms.Grayscale(num_output_channels=1),
                                       transforms.Resize(args.img_size),
                                       transforms.CenterCrop(args.img_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

                                   ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                                 shuffle=True, num_workers=args.num_workers)
else:
    dataset = dset.ImageFolder(root='./data/'+args.test_folder,  #./data/celeba
                               transform=transforms.Compose([
                                   transforms.RandomRotation((90, 90)),
                                   transforms.Resize(args.img_size),
                                   transforms.CenterCrop(args.img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs,
                                             shuffle=True, num_workers=args.num_workers)


# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")


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


with open('logbook/' + args.model_id + 'noise.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

    for j in range(20):
        truepositive = 0
        falsepositive = 0
        truenegative = 0
        falsenegative = 0

        for i, data in enumerate(dataloader, 0):
            # Format batch

            with torch.no_grad():
                real_cpu = data[0].to(device)

                #get message to encode, in string format
                message = args.message
                #send it to function that converts it to bits
                u = encoded_message(bs,ud, im, message).to(device)
                print(u.shape, real_cpu.shape)

                #encode this message on image real_cpu
                encoded = netE(real_cpu,u)
                #add noise to it
                noisy_encoded = channel(encoded, (j * args.awgn), args)
                #save this encoded image
                save_image(encoded.data, 'images/test/' + args.model_id + '%d_encoded.png' % i,
                           normalize=True, pad_value=1.0)
                save_image(noisy_encoded.data, 'images/test/'+args.model_id+'%d_noisy.png'%i,
                           normalize=True, pad_value=1.0)

                #decode message from encoded
                decoded_u = netDec(encoded)
                noisy_decoded = netDec(noisy_encoded)
                ber = errors_ber(decoded_u, u)
                nber = errors_ber(noisy_decoded, u)
                #print(ber.item(), nber.item())
                #find in entire batch
                for i in range(len(decoded_u)):

                    #convert u to bits array
                    single_u = torch.round(decoded_u[i]).view(-1)
                    noisysingle_u = torch.round(noisy_decoded[i]).view(-1)
                    #print(single_u.size())
                    #single_u = torch.round(torch.sum(torch.round(decoded_u[i]), dim=0) / 3).view(-1)
                    #print(single_u.size())

                    #print(decoded_u.data)
                    #convert to string from bits
                    dec_message1 = decoded_message(single_u)
                    dec_message2 = decoded_message(noisysingle_u)


                    #print(dec_message)


                    if (dec_message1 == message) & (dec_message2 == message):
                        #print("Message retrieved")
                        truepositive +=1
                    elif (dec_message1 != message) & (dec_message2 == message):
                        falsenegative +=1
                    elif (dec_message2 != message) & (dec_message1 == message):
                        falsepositive +=1
                    else:
                        truenegative+=1

        noiselessdec= (truepositive+falsepositive)
        noisydec = (truepositive+falsenegative)
        invariance = (truepositive/(falsenegative+falsepositive+truepositive))
        print("total number of noiseless decoded:",noiselessdec)
        print("total number of noisy decoded:",noisydec)
        print("Invariance over noise:", invariance)

        filewriter.writerow([j, noiselessdec, noisydec, invariance])

        if invariance<0.5:
            break













