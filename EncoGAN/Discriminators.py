import torch.nn as nn
import torch

import numpy as np
from torch.nn.utils import spectral_norm

class DCGANDiscriminator(nn.Module):
    def __init__(self, args):
        super(DCGANDiscriminator, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.ndf = self.args.disc_lat

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")
        if self.args.img_size==128:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 16),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(self.ndf * 16, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif self.args.img_size==64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif self.args.img_size == 32:
            # 64 uses kernel = 4, stride = 2, padding = 1. To reduce (64,64) to 1 via 5 layers.
            # 32 uses kernel = 4, strude = 2, padding = 1. to reduce (32,32) to 1 via 4 layers.

            # state size. (ndf) x 32 x 32
            self.l21 = nn.Conv2d(self.nc, self.ndf * 2, 4, 2, 1, bias=False)
            self.l22 = nn.BatchNorm2d(self.ndf * 2)
            self.l23 = nn.LeakyReLU(0.2, inplace=False)

            # state size. (ndf*2) x 16 x 16
            self.l31 = nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)
            self.l32 = nn.BatchNorm2d(self.ndf * 4)
            self.l33 = nn.LeakyReLU(0.2, inplace=False)

            # state size. (ndf*4) x 8 x 8
            self.l41 = nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)
            self.l42 = nn.BatchNorm2d(self.ndf * 8)
            self.l43 = nn.LeakyReLU(0.2, inplace=False)

            # state size. (ndf*8) x 4 x 4
            self.l51 = nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
            self.l52 = nn.Sigmoid()

    def forward(self, input):

        if self.args.img_size==32:

            # x = self.l13(self.l12(self.l11(input)))

            x = self.l23(self.l22(self.l21(input)))
            x = self.l33(self.l32(self.l31(x)))
            x = self.l43(self.l42(self.l41(x)))

            x = self.l52(self.l51(x)).view(-1)
        else:
            return self.main(input).view(-1)

            return x



class SNGANDiscriminator(nn.Module):
    def __init__(self, args):
        super(SNGANDiscriminator, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.ndf = self.args.disc_lat

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")
        if self.args.img_size==64:
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                spectral_norm(nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf) x 32 x 32
                spectral_norm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(self.ndf * 2),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*2) x 16 x 16
                spectral_norm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(self.ndf * 4),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*4) x 8 x 8
                spectral_norm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)),
                nn.BatchNorm2d(self.ndf * 8),
                nn.LeakyReLU(0.2, inplace=False),
                # state size. (ndf*8) x 4 x 4
                spectral_norm(nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)),
                nn.Sigmoid()
            )
        elif self.args.img_size == 32:
            # 64 uses kernel = 4, stride = 2, padding = 1. To reduce (64,64) to 1 via 5 layers.
            # 32 uses kernel = 4, strude = 2, padding = 1. to reduce (32,32) to 1 via 4 layers.

            # state size. (ndf) x 32 x 32
            self.l21 = spectral_norm(nn.Conv2d(self.nc, self.ndf * 2, 4, 2, 1, bias=False))
            self.l22 = nn.BatchNorm2d(self.ndf * 2)
            self.l23 = nn.LeakyReLU(0.2, inplace=False)

            # state size. (ndf*2) x 16 x 16
            self.l31 = spectral_norm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False))
            self.l32 = nn.BatchNorm2d(self.ndf * 4)
            self.l33 = nn.LeakyReLU(0.2, inplace=False)

            # state size. (ndf*4) x 8 x 8
            self.l41 = spectral_norm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False))
            self.l42 = nn.BatchNorm2d(self.ndf * 8)
            self.l43 = nn.LeakyReLU(0.2, inplace=False)

            # state size. (ndf*8) x 4 x 4
            self.l51 = spectral_norm(nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False))
            self.l52 = nn.Sigmoid()

    def forward(self, input):

        if self.args.img_size==64:
            return self.main(input).view(-1)
        else:
            #x = self.l13(self.l12(self.l11(input)))


            x = self.l23(self.l22(self.l21(input)))
            x = self.l33(self.l32(self.l31(x)))
            x = self.l43(self.l42(self.l41(x)))

            x = self.l52(self.l51(x)).view(-1)

            return x



# Self-Attention GAN
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention

class Generator(nn.Module):
    """Generator."""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(spectral_norm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2

class SAGANDiscriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, args):
        super(SAGANDiscriminator, self).__init__()
        self.args = args
        self.imsize = args.img_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        conv_dim = self.args.disc_lat
        layer1.append(spectral_norm(nn.Conv2d(args.img_channels, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

        self.sgmd = nn.Sigmoid()

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return self.sgmd(out)

        #return out.squeeze(), p1, p2