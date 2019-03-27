import torch.nn as nn
import torch



class Basic_Decoder(nn.Module):
    def __init__(self, args):
        super(Basic_Decoder, self).__init__()
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
            nn.LeakyReLU(0.2, inplace=False),
            # state size is ndf x 64 x64
            nn.Conv2d(self.ndf, self.ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=False),
            # state size is ndf x 64 x64
            nn.Conv2d(self.ndf, self.ud, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        output= self.main(img)
        return output


class Dense_Decoder(nn.Module):
    def __init__(self, args):
        super(Dense_Decoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.ndf = self.args.dec_lat
        self.ud = self.args.bitsperpix

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.dec_Img = nn.Sequential(
            #input is nc x 64 x64
            nn.Conv2d(self.nc, self.ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.decode1 = nn.Sequential(
            #input is ndf x 64 x64
            nn.Conv2d(self.ndf, self.ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.decode2 = nn.Sequential(
            #input is 2*ndf x 64x 64
            nn.Conv2d((2*self.ndf), self.ndf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.decode3 = nn.Sequential(
            #input is 3*ndf x 64 x64
            nn.Conv2d((3*self.ndf), self.ud, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        output1 = self.dec_Img(img)
        output2 = self.decode1(output1)
        output3 = torch.cat([output1, output2], dim=1)
        output3 = self.decode2(output3)
        output4 = torch.cat([output1, output2, output3], dim=1)
        output4 = self.decode3(output4)
        return output4

