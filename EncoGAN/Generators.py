import torch.nn as nn
import torch




class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        self.args = args
        self.nc = self.args.img_channels
        self.nz = self.args.zlatent
        self.ngf = self.args.gen_lat

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

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

    def forward(self, z):
        output = self.main(z)
        return output



    

