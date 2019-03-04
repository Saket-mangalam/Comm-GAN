import torch.nn as nn
import torch
import numpy as np

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#BER Loss of Decoder
def errors_ber(y_pred, y_true):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return k

def channel(encoded_imgs, noise_std, channel_type = 'awgn', device = torch.device("cuda") ):

    if channel_type == 'awgn':
        noise        = noise_std * torch.randn(encoded_imgs.shape, dtype=torch.float).to(device)
        return encoded_imgs + noise

    elif channel_type == 'slides':
        batch_size, s, l1, l2 = encoded_imgs.shape[0], encoded_imgs.shape[1] , encoded_imgs.shape[2] ,encoded_imgs.shape[3]
        # only support MNIST for now
        random_line = int(np.ceil(noise_std * l1))
        random_start = np.random.randint(1, l1- random_line)
        if np.random.rand()>0.5:
            random_noise = torch.cat([torch.ones((batch_size, s, random_start, l2), dtype=torch.float),
                                      torch.zeros((batch_size, s, random_line, l2), dtype=torch.float),
                                      torch.ones((batch_size, s, l1 - random_start - random_line, l2), dtype=torch.float)],
                                     dim = 2).to(device)
        else:
            random_noise = torch.cat([torch.ones((batch_size, s, l1, random_start), dtype=torch.float),
                                      torch.zeros((batch_size, s, l1, random_line), dtype=torch.float),
                                      torch.ones((batch_size, s, l1, l2 - random_start - random_line), dtype=torch.float)],
                                     dim = 3).to(device)

        received_imgs= random_noise * encoded_imgs
        #save_image(received_imgs, 'images/tmp/tmp.png', nrow=10, normalize=True)
        return received_imgs
    elif channel_type == 'basic_quantize':
        # quantize to -1, -.5, 0.0, 0.5, 1.
        encoded_imgs = encoded_imgs.detach()
        q = 0.5
        y = q * np.round(encoded_imgs/q)
        return encoded_imgs

    else:
        noise        = noise_std * torch.randn(encoded_imgs.shape, dtype=torch.float).to(device)
        return encoded_imgs + noise