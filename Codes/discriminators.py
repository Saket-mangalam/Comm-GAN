


import torch.nn as nn
#import torch.nn.functional as F
import torch
from SpectralNormalization import SpectralNormalization

class DCGAN_discriminator(nn.Module):
    def __init__(self, args):
        super(DCGAN_discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True, specnorm = True):
            if specnorm :
                block = [   SpectralNormalization(nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            else:
                block = [ nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.img_channel, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


class Hidden_discriminator(nn.Module):
    def __init__(self,args):
        super(Hidden_discriminator,self).__init__()
        
        self.args = args
        
        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")
        
        def block(in_filters,out_filters, normalize=True, specnorm = True):
            if specnorm:
                block = [ SpectralNormalization(nn.Conv2d(in_filters,out_filters,3,stride=1,padding=1))]
            else:
                block = [ nn.Conv2d(in_filters,out_filters,3,stride=1,padding=1)]
            if normalize:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.model = nn.Sequential(
                *block(self.args.img_channel,64),
                *block(64,64),
                *block(64,64))
        
        #ds_size = args.img_size // 2**4
        self.avgpool = nn.Sequential(nn.AvgPool3d(kernel_size = (64,self.args.img_size,self.args.img_size), stride = 1, padding=0),
                                  nn.Sigmoid())
        #self.adv_layer = nn.Sequential( nn.Linear(64*ds_size**2, 1),nn.Sigmoid())

        
    def forward(self,img):
        conv = self.model(img)
        conv = self.avgpool(conv)
        conv = conv.view(conv.shape[0],-1)
        validity = conv
        return validity
    
        