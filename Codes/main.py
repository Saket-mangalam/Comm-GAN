
import numpy as np
import os
import torch
from get_args import get_args

from utils import channel, errors_ber, weights_init_normal

import torchvision.transforms as transforms
from torchvision.utils import save_image
from datagenerator import Datagen
#from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

# WGAN-GP utility
import torch.autograd as autograd
def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == '__main__':

    # give each run a random ID
    identity = str(np.random.random())[2:8]
    print('[ID]', identity)

    args = get_args()

    os.makedirs('images', exist_ok=True)
    os.makedirs('images/'+identity, exist_ok=True)

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

    if args.g_type == 'dcgan':
        from generators import DCGAN_Generator as Generator
    elif args.g_type == 'rnn_dcnn':
        from generators import rnn_Generator as Generator
    elif args.g_type == 'gridrnn_dcnn':
        from generators import gridrnn_Generator as Generator
    elif args.g_type == 'hidden':
        from generators import Hidden_Generator as Generator
    else:
        from generators import FCNN_Generator as Generator

    if args.dec_type == 'dcgan':
        from decoders import DCGAN_Decoder as Decoder
    elif args.dec_type == 'rnn_dcnn':
        from decoders import rnn_Decoder as Decoder
    elif args.dec_type == 'gridrnn_dcnn':
        from decoders import gridrnn_Decoder as Decoder
    elif args.dec_type == 'hidden':
        from decoders import Hidden_Decoder as Decoder
    else:
        from decoders import gridrnn_Decoder as Decoder

    if args.d_type == 'dcgan':
        from discriminators import DCGAN_discriminator as Discriminator
    #elif args.d_type == 'hidden':
    else:
        from discriminators import Hidden_discriminator as Discriminator
    #else:
    #from discriminators import DCGAN_discriminator as Discriminator

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
    if args.dataset == 'mnist':
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
    elif args.dataset == 'test_selfie':
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
    elif args.dataset == 'mypic':
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
    else:
        print('hahahaha dataset is unknown')




    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    ########################################################
    # Optimizers
    ########################################################
    if args.gan_type == 'wgan_gp' or args.gan_type == 'dcgan':
        optimizer_G   = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        optimizer_Dec = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        optimizer_D   = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    elif args.gan_type == 'wgan':
        optimizer_G   = torch.optim.RMSprop(generator.parameters(), lr=args.lr)
        optimizer_Dec = torch.optim.RMSprop(decoder.parameters(), lr=args.lr)
        optimizer_D   = torch.optim.RMSprop(discriminator.parameters(), lr=args.lr)

    else: # default is DCGAN optimizer
        optimizer_G   = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        optimizer_Dec = torch.optim.Adam(decoder.parameters(), lr=args.lr, betas=(args.b1, args.b2))
        optimizer_D   = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(args.num_epoch):
        for i, (imgs, _) in enumerate(train_dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            if valid.shape[0]!=args.batch_size:
                continue

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train G
            # -----------------
            for idx in range(args.num_train_G):
                optimizer_G.zero_grad()

                # Sample noise as generator input
                #z = real_imgs
                #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
                u = torch.randint(0, 2, (args.block_len, 1), dtype=torch.float).to(device)
                
                # Generate a batch of images
                gen_imgs = generator(real_imgs, u)

                # Loss measures generator's ability to fool the discriminator
                #received_imgs = channel(gen_imgs, args.noise_std, channel_type = args.channel_type, device = device)
                received_imgs = gen_imgs
                decoded_info = decoder(received_imgs)

                if args.gan_type == 'dcgan':
                    # DCGAN
                    # train generator to optimize both Channel AE + GAN
                    g_loss =    (1.0 - args.dec_weight) * BCELoss(discriminator(gen_imgs), valid) + \
                                   args.dec_weight * BCELoss(decoded_info, u)
                
                elif args.gan_type == 'hidden':
                    #hidden loss to optimize Channel Encoder + image reconstruction + Adversary
                    g_loss = (1.0 - args.lambda_I - args.lambda_G)*BCELoss(decoded_info,u) + \
                                args.lambda_I * MSELoss(gen_imgs,real_imgs) + \
                                args.lambda_G *((BCELoss(discriminator(gen_imgs), fake) + BCELoss(discriminator(real_imgs),valid))/2)
                    

                elif args.gan_type == 'wgan' or args.gan_type == 'wgan_gp':
                    g_loss = (1.0 - args.dec_weight) * (-torch.mean(discriminator(gen_imgs))) + \
                                args.dec_weight * BCELoss(decoded_info, u)

                else: # dcgan by default
                    g_loss =    (1.0 - args.dec_weight) * BCELoss(discriminator(gen_imgs), valid) + \
                                args.dec_weight * BCELoss(decoded_info, u)


                g_loss.backward()
                optimizer_G.step()

            # -----------------
            #  Train Decoder
            # -----------------
            for idx in range(args.num_train_Dec):
                optimizer_Dec.zero_grad()

                # Sample noise as generator input
                #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
                u = torch.randint(0, 2, (args.block_len, 1), dtype=torch.float).to(device)

                # Generate a batch of images
                gen_imgs = generator(real_imgs, u)

                # Loss measures generator's ability to fool the discriminator
                received_imgs = channel(gen_imgs, args.noise_std, channel_type = args.channel_type, device = device)

                decoded_info = decoder(received_imgs)
                # train decoder only for decoder.
                dec_loss =  BCELoss(decoded_info, u)

                dec_loss.backward()
                optimizer_Dec.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            for idx in range(args.num_train_D):
                optimizer_D.zero_grad()
                
                u = torch.randint(0, 2, (args.block_len, 1), dtype=torch.float).to(device)
                gen_imgs = generator(real_imgs,u)
                
                if args.gan_type == 'dcgan':
                    # Measure discriminator's ability to classify real from generated samples
                    real_loss = BCELoss(discriminator(real_imgs), valid)
                    fake_loss = BCELoss(discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_D.step()

                elif args.gan_type == 'wgan_gp':
                    # Real images
                    fake_imgs = gen_imgs.detach()
                    real_validity = discriminator(real_imgs)

                    # Fake images
                    fake_validity = discriminator(fake_imgs)

                    gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                    # Adversarial loss
                    d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty
                    d_loss.backward()
                    optimizer_D.step()
                    
                elif args.gan_type == 'hidden':
                    real_loss = BCELoss(discriminator(real_imgs),valid)
                    fake_loss = BCELoss(discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_D.step()

                else: # 'wgan'
                    fake_imgs = generator(z).detach()
                    # Adversarial loss
                    loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))

                    loss_D.backward()
                    optimizer_D.step()

                    # Clip weights of discriminator
                    for p in discriminator.parameters():
                        p.data.clamp_(-args.clip_value, args.clip_value)

            if i%100 == 0:
                decoded_info = decoded_info.detach()
                u            = u.detach()
                this_ber = errors_ber(decoded_info, u)
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [batch Dec BER: %f]" % (epoch, args.num_epoch, i, len(train_dataloader),
                                                                    d_loss.item(), g_loss.item(), this_ber))


            batches_done = epoch * len(train_dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], 'images/'+identity+'/%d.png' % batches_done, nrow=5, normalize=True)




    # --------------------------
    #  Testing: only for BER
    # --------------------------
    ber_count = 0.0
    count = 0
    for i, (imgs, _) in enumerate(test_dataloader):
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        if valid.shape[0]!=args.batch_size:
            continue

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))
        u = torch.randint(0, 2, (args.batch_size, args.block_len, 1), dtype=torch.float).to(device)

        # Generate a batch of images
        gen_imgs = generator(z, u)

        # Loss measures generator's ability to fool the discriminator
        received_imgs = channel(gen_imgs, args.noise_std, channel_type = args.channel_type, device = device)

        decoded_info = decoder(received_imgs)

        decoded_info = decoded_info.detach()
        u            = u.detach()

        decode_ber = errors_ber(decoded_info, u)
        ber_count += decode_ber
        count += 1

    print('The BER of image code is,', ber_count/count)
    print('model id is', identity)
    print(args)