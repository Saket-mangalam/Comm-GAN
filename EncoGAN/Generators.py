import torch.nn as nn
import torch

from quantizer import quantizer


class DCGANGenerator(nn.Module):
    def __init__(self, args):
        super(DCGANGenerator, self).__init__()

        self.args = args
        self.nc = self.args.img_channels
        self.nz = self.args.zlatent
        self.ngf = self.args.gen_lat
        self.imgsize = self.args.img_size

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        if self.imgsize == 128:
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(self.nz, self.ngf * 16, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.ngf * 16),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(self.ngf * 4, self.ngf*2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf*2),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(self.ngf*2, self.ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(True),
                # state size 64x64
                nn.ConvTranspose2d(self.ngf,self.nc,4,2,1,bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        elif self.imgsize == 64:
            self.main = nn.Sequential(
                        # input is Z, going into a convolution
                        nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(self.ngf * 8),
                        nn.ReLU(True),
                        # state size. (ngf*8) x 4 x 4
                        nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.ngf * 4),
                        nn.ReLU(True),
                        # state size. (ngf*4) x 8 x 8
                        nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.ngf * 2),
                        nn.ReLU(True),
                        # state size. (ngf*2) x 16 x 16
                        nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(self.ngf),
                        nn.ReLU(True),
                        # state size. (ngf) x 32 x 32
                        nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
                        nn.Tanh()
                        # state size. (nc) x 64 x 64
                    )
        elif self.args.img_size == 32:
            # 64 uses kernel = 4, stride = 1, padding = 0. To reduce (64,64) to 1 via 5 layers.
            # 32 uses kernel = 4, strude = 2, padding = 1. to reduce (32,32) to 1 via 4 layers

            # input is Z, going into a convolution
            self.l11 =nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False)
            self.l12 =nn.BatchNorm2d(self.ngf * 8)
            self.l13 =nn.ReLU(True)

            # state size. (self.ngf) x 4 x 4
            self.l21 = nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)
            self.l22 = nn.BatchNorm2d(self.ngf * 4)
            self.l23 = nn.ReLU(True)

            # state size. (ngf*4) x 8 x 8
            self.l31 = nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)
            self.l32 = nn.BatchNorm2d(self.ngf * 2)
            self.l33 = nn.ReLU(True)

            # state size. (ngf*2) x 16 x 16
            self.l41 = nn.ConvTranspose2d(self.ngf*2, self.nc, 4, 2, 1, bias=False)
            self.l42 = nn.Tanh()
            # state size. (nc) x 32 x 32



    def forward(self, z):
        if self.args.img_size == 32:
            x = self.l13(self.l12(self.l11(z)))
            x = self.l23(self.l22(self.l21(x)))
            x = self.l33(self.l32(self.l31(x)))
            output = self.l42(self.l41(x))
        else:
            output = self.main(z)

            #output = quantizer(output, self.args)

        return output





    

