import torch.nn as nn
import torch

from TurboCodec import ENC_interRNN

from quantizer import quantizer

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


class Dense_Encoder(nn.Module):
    def __init__(self, args):
        super(Dense_Encoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.nef = self.args.enc_lat
        self.ud = self.args.bitsperpix

        self.cc_channel = 1

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        if args.info_img_size == 16:
            self.cc_encode = nn.Sequential(
                #input is cc_channel,
                nn.ConvTranspose2d(1, self.nef, 3, 2, 1, bias=True),
                nn.BatchNorm2d(self.nef),
                nn.LeakyReLU(0.2, inplace=False),
                nn.ConvTranspose2d(self.nef, self.cc_channel, 4, 1, 1, bias=True),
                nn.BatchNorm2d(self.cc_channel),
                nn.LeakyReLU(0.2, inplace=False)
            )
        elif args.info_img_size ==18:
            self.turbo_enc = ENC_interRNN(args)


        self.enc_Img = nn.Sequential(
            #state size nc X 64 x64
            nn.Conv2d(self.nc, self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.enc_layers    = torch.nn.ModuleList()

        for iter in range(args.num_layer_Enc):
            #
            this_input_channel = (iter+1)*self.nef + self.ud
            if iter!=args.num_layer_Enc-1:
                this_output_channel = self.nef
            else:
                this_output_channel = self.nc
            this_enc =  nn.Sequential(
                #statesize is nef+_ud X 64 x64
                nn.Conv2d(this_input_channel, this_output_channel, 3, 1, 1, bias=False),
                nn.BatchNorm2d(this_output_channel),
                nn.LeakyReLU(0.2, inplace=False)
            )
            self.enc_layers.append(this_enc)


        # self.encode1 = nn.Sequential(
        #     #statesize is nef+_ud X 64 x64
        #     nn.Conv2d((self.nef+ self.ud ), self.nef, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(self.nef),
        #     nn.LeakyReLU(0.2, inplace=False)
        # )
        #
        # self.encode2 = nn.Sequential(
        #     #statesize is 2*nef + ud X 64 x64
        #     nn.Conv2d((2*self.nef+self.ud ), self.nef, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(self.nef),
        #     nn.LeakyReLU(0.2, inplace=False)
        # )
        #
        # self.encode3 = nn.Sequential(
        #     #state size is 3*nef + ud X 64 x64
        #     nn.Conv2d((3*self.nef+self.ud), self.nc, 3, 1, 1, bias=False),
        #     nn.BatchNorm2d(self.nc),
        #     nn.LeakyReLU(0.2, inplace=False)
        # )

        self.tanh = nn.Sequential(
            nn.Tanh()
            # output image
        )

    def forward(self, img, u):
        output1 = self.enc_Img(img)
        u_shape = u.shape

        if self.args.img_size == 32:
            # Channel Encoding part. Start by u_shape = (B, L, 1), end by (B, 1, H, W).
            # for 32*32 image, 1024 bits - 64*3=192 =>832 bits , try to embedded 5
            if self.args.info_img_size==16:
                uprime  = self.cc_encode(u)

            elif self.args.info_img_size==32:
                uprime = u

            elif self.args.info_img_size == 18:
                # channel coding 1D encoding...
                # chop u to multiple blocks, with size (B*M, L, 1), M is the number of blocks
                u_shape = u.shape

                u      = u.view(u_shape[0]*4, int(self.args.info_img_size**2/4), 1)
                encoded = self.turbo_enc(u)
                encoded = encoded.view(u_shape[0], -1)
                encoded_shape = encoded.shape
                encoded_padded = torch.cat([encoded, encoded[:, :self.args.img_size**2 - encoded_shape[1]]], dim=1)
                uprime = encoded_padded.view(u_shape[0], 1, self.args.img_size,self.args.img_size)
        else: # img_size = 64
            uprime = u

        # Dense Encoder part:
        dense_input = torch.cat([output1,uprime], dim=1)
        for iter in range(self.args.num_layer_Enc):
            this_output = self.enc_layers[iter](dense_input)

            if iter!=self.args.num_layer_Enc-1:
                dense_input = torch.cat([this_output,dense_input ], dim=1)

        # resnet connection?
        this_output += img
        output4 = self.tanh(this_output)

        # # Encoder part
        # output2 = torch.cat([output1,uprime], dim=1)
        # output2 = self.encode1(output2)
        # output3 = torch.cat([output1,output2,uprime], dim=1)
        # output3 = self.encode2(output3)
        # output4 = torch.cat([output1, output2, output3, uprime], dim=1)
        # output4 = self.encode3(output4)
        # #output4 += img
        # output4 = self.tanh(output4)
        # #output4 = quantizer(output4, self.args)

        return output4


from Discriminators import Self_Attn

class SADense_Encoder(nn.Module):
    def __init__(self, args):
        super(SADense_Encoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.nef = self.args.enc_lat
        self.ud = self.args.bitsperpix

        self.cc_channel = 1


        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        if args.info_img_size == 16:
            self.cc_encode = nn.Sequential(
                #input is cc_channel,
                nn.ConvTranspose2d(1, self.nef, 3, 2, 1, bias=True),
                nn.BatchNorm2d(self.nef),
                nn.LeakyReLU(0.2, inplace=False),
                nn.ConvTranspose2d(self.nef, self.cc_channel, 4, 1, 1, bias=True),
                nn.BatchNorm2d(self.cc_channel),
                nn.LeakyReLU(0.2, inplace=False)
            )
        elif args.info_img_size ==18:
            self.turbo_enc = ENC_interRNN(args)


        self.enc_Img = nn.Sequential(
            #state size nc X 64 x64
            nn.Conv2d(self.nc, self.nef, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.nef),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.enc_layers    = torch.nn.ModuleList()
        self.sa_attn       = torch.nn.ModuleList()

        for iter in range(args.num_layer_Enc):
            #
            this_input_channel = (iter+1)*self.nef + self.ud
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

    def forward(self, img, u):
        output1 = self.enc_Img(img)
        u_shape = u.shape

        if self.args.img_size == 32:
            # Channel Encoding part. Start by u_shape = (B, L, 1), end by (B, 1, H, W).
            # for 32*32 image, 1024 bits - 64*3=192 =>832 bits , try to embedded 5
            if self.args.info_img_size==16:
                uprime  = self.cc_encode(u)

            elif self.args.info_img_size==32:
                uprime = u

            elif self.args.info_img_size == 18:
                # channel coding 1D encoding...
                # chop u to multiple blocks, with size (B*M, L, 1), M is the number of blocks
                u_shape = u.shape

                u      = u.view(u_shape[0]*4, int(self.args.info_img_size**2/4), 1)
                encoded = self.turbo_enc(u)
                encoded = encoded.view(u_shape[0], -1)
                encoded_shape = encoded.shape
                encoded_padded = torch.cat([encoded, encoded[:, :self.args.img_size**2 - encoded_shape[1]]], dim=1)
                uprime = encoded_padded.view(u_shape[0], 1, self.args.img_size,self.args.img_size)
        else: # img_size = 64
            uprime = u

        # Dense Encoder part:
        dense_input = torch.cat([output1,uprime], dim=1)
        for iter in range(self.args.num_layer_Enc):
            this_output = self.enc_layers[iter](dense_input)

            if iter == self.args.num_layer_Enc-2:
                dense_input, _= self.sa_attn[0](dense_input)

            if iter == self.args.num_layer_Enc-1:
                dense_input, _= self.sa_attn[1](dense_input)

            if iter!=self.args.num_layer_Enc-1:
                dense_input = torch.cat([this_output,dense_input ], dim=1)

        # resnet connection?
        this_output += img
        output4 = self.tanh(this_output)

        # # Encoder part
        # output2 = torch.cat([output1,uprime], dim=1)
        # output2 = self.encode1(output2)
        # output3 = torch.cat([output1,output2,uprime], dim=1)
        # output3 = self.encode2(output3)
        # output4 = torch.cat([output1, output2, output3, uprime], dim=1)
        # output4 = self.encode3(output4)
        # #output4 += img
        # output4 = self.tanh(output4)
        # #output4 = quantizer(output4, self.args)

        return output4

