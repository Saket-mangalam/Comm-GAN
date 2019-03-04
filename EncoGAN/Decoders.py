import torch.nn as nn
import torch



class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.ndf = self.args.dec_lat
        self.ud = self.args.bitsperpix

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is ndf x 64 x64
            nn.Conv2d(self.ndf, self.ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size is ndf x 64 x64
            nn.Conv2d(self.ndf, self.nc * self.ud, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        output= self.main(img)
        return output

