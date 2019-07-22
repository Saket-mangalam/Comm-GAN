__author__ = 'SaketM'
import torch.nn as nn
import torch
# need some more engineering on discriminator now.......

class DCGANDiscriminator(nn.Module):
    def __init__(self, args):
        super(DCGANDiscriminator, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.ndf = self.args.disc_lat

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")
        if self.args.img_size==64:
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
                nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
            )
        elif self.args.img_size == 128:
            self.main = nn.Sequential(
                # input is (nc) x 128 x 128
                nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ndf),
                nn.LeakyReLU(0.2, inplace=False),
                # input is (nc) x 64 x 64
                nn.Conv2d(self.ndf, self.ndf, 4, 2, 1, bias=False),
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
                nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False)
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

    def forward(self, input):

        if self.args.img_size==64 or self.args.img_size==128:
            return torch.sigmoid(self.main(input)).view(-1)
        else:
            #x = self.l13(self.l12(self.l11(input)))


            x = self.l23(self.l22(self.l21(input)))
            x = self.l33(self.l32(self.l31(x)))
            x = self.l43(self.l42(self.l41(x)))

            x = torch.sigmoid(self.l51(x)).view(-1)

            return x
