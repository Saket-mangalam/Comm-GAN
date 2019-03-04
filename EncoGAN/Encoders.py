import torch.nn as nn
import torch



class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.nef = self.args.enc_lat
        self.ud = self.args.bitsperpix

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.enc_Img = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encode = nn.Sequential(
            #input is (nef+ 3*ud) x 64 x 64
            nn.Conv2d((self.nef + self.nc * self.ud), self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace= True),
            # input is nef x 64 x64
            nn.Conv2d(self.nef, self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace= True),
            # input is ned x 64 x64
            nn.Conv2d(self.nef, self.nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size nc x 64 x64
        )

    def forward(self, img, u):
        eimg = self.enc_Img(img)
        eimg = torch.cat([eimg,u], dim=1)
        output = self.encode(eimg)
        return output

