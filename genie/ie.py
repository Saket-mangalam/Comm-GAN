__author__ = 'SaketM'
import torch.nn as nn
import torch
from sagan_utils import Self_Attn

###########################################
# Image Encoders (IE)
# IE maps u-tensor and image to encoded image
###########################################
# not done
class IE_Basic_Encoder(nn.Module):
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
            nn.Conv2d(self.args.img_channels, self.args.code_rate_n, 3, 1, 1, bias=False),
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

# not done
class IE_Residual_Encoder(nn.Module):
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

# not done
class DenseRes_Encoder(nn.Module):
    def __init__(self, args):
        super(DenseRes_Encoder, self).__init__()
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

# not done
class IE_Dense(nn.Module):
    def __init__(self, args):
        super(IE_Dense, self).__init__()
        self.args = args
        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.enc_layers    = torch.nn.ModuleList()
        self.sa_attn       = torch.nn.ModuleList()

        for iter in range(args.ie_num_layer):

            this_input_channel = (iter)*self.args.ie_num_unit + self.args.code_rate_n + self.args.img_channels
            #this_input_channel = (iter)*self.nef + self.ud + self.img_channels
            if iter!=args.ie_num_layer-1:
                this_output_channel = self.args.ie_num_unit
            else:
                this_output_channel = self.args.img_channels

            this_enc =  nn.Sequential(
                #statesize is nef+_ud X 64 x64
                nn.Conv2d(this_input_channel, this_output_channel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(this_output_channel),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.enc_layers.append(this_enc)

        self.tanh = nn.Sequential(
            nn.Tanh()
        )

    def forward(self,message, image):
        dense_input = torch.cat([message, image], dim=1)
        for iter in range(self.args.ie_num_layer):
            this_output = self.enc_layers[iter](dense_input)
            if iter!=self.args.ie_num_layer-1:
                dense_input = torch.cat([this_output,dense_input], dim=1)

        output4 = self.tanh(this_output)
        return output4


class IE_SADense(nn.Module):
    def __init__(self, args):
        super(IE_SADense, self).__init__()
        self.args = args
        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.enc_layers    = torch.nn.ModuleList()
        self.sa_attn       = torch.nn.ModuleList()

        for iter in range(args.num_layer_enc):

            this_input_channel = (iter+1)*self.code_rate_n + self.img_channels
            #this_input_channel = (iter)*self.nef + self.ud + self.img_channels
            if iter!=args.num_layer_Enc-1:
                this_output_channel = self.nef
            else:
                this_output_channel = self.nc

            #only last 2 layers put this on
            if iter in[ args.num_layer_Enc-2, args.num_layer_Enc-1]:
                self.sa_attn.append(Self_Attn(this_input_channel, 'relu'))

            this_enc =  nn.Sequential(
                #statesize is nef+_ud X 64 x64
                nn.Conv2d(this_input_channel, this_output_channel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(this_output_channel),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.enc_layers.append(this_enc)

        self.tanh = nn.Sequential(
            nn.Tanh()
        )

    def forward(self,message, image):
        dense_input = torch.cat([message, image], dim=1)
        for iter in range(self.args.num_layer_Enc):
            this_output = self.enc_layers[iter](dense_input)

            if iter == self.args.num_layer_Enc-2:
                dense_input, _= self.sa_attn[0](dense_input)

            if iter == self.args.num_layer_Enc-1:
                dense_input, _= self.sa_attn[1](dense_input)

            if iter!=self.args.num_layer_Enc-1:
                dense_input = torch.cat([this_output,dense_input], dim=1)

        output4 = self.tanh(this_output)


        return output4

