import torch.nn as nn
import torch



class Basic_Encoder(nn.Module):
    def __init__(self, args):
        super(Basic_Encoder, self).__init__()
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
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.encode = nn.Sequential(
            #input is (nef+ ud) x 64 x 64
            nn.Conv2d((self.nef + self.ud), self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace= False),
            # input is nef x 64 x64
            nn.Conv2d(self.nef, self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace= False),
            # input is ned x 64 x64
            nn.Conv2d(self.nef, self.nc, 3, 1, 1, bias=False),
            nn.Tanh()
            # state size nc x 64 x64
        )

    def forward(self, img, u):
        output = self.enc_Img(img)
        output = torch.cat([output,u], dim=1)
        output = self.encode(output)
        return output


class Residual_Encoder(nn.Module):
    def __init__(self, args):
        super(Residual_Encoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.nef = self.args.enc_lat
        self.ud = self.args.bitsperpix

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.enc_Img = nn.Sequential(
            #input is nc X 64X64
            nn.Conv2d(self.nc,self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False)

        )

        self.encode = nn.Sequential(
            #input is nef + ud X64X64
            nn.Conv2d((self.nef + self.ud), self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False),
            #input is nef x 64 x64
            nn.Conv2d(self.nef, self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False),
            #input is nef X 64x64
            nn.Conv2d(self.nef,self.nc, 3, 1, 1, bias=False)
        )

        self.tanh = nn.Sequential(
            nn.Tanh()
            # output image
        )

    def forward(self, img, u):
        output = self.enc_Img(img)
        output = torch.cat([output,u], dim=1)
        output = self.encode(output)
        output += img
        output = self.tanh(output)
        return output


class Dense_Encoder(nn.Module):
    def __init__(self, args):
        super(Dense_Encoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.nef = self.args.enc_lat
        self.ud = self.args.bitsperpix

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.enc_Img = nn.Sequential(
            #state size nc X 64 x64
            nn.Conv2d(self.nc, self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.encode1 = nn.Sequential(
            #statesize is nef+_ud X 64 x64
            nn.Conv2d((self.nef+self.ud), self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.encode2 = nn.Sequential(
            #statesize is 2*nef + ud X 64 x64
            nn.Conv2d((2*self.nef + self.ud), self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.encode3 = nn.Sequential(
            #state size is 3*nef + ud X 64 x64
            nn.Conv2d((3*self.nef + self.ud), self.nc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nc),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.tanh = nn.Sequential(
            nn.Tanh()
            # output image
        )

    def forward(self, img, u):
        output1 = self.enc_Img(img)
        output2 = torch.cat([output1,u], dim=1)
        output2 = self.encode1(output2)
        output3 = torch.cat([output1,output2,u], dim=1)
        output3 = self.encode2(output3)
        output4 = torch.cat([output1, output2, output3, u], dim=1)
        output4 = self.encode3(output4)
        output4 += img
        output4 = self.tanh(output4)
        return output4


