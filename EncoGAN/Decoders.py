import torch.nn as nn
import torch

from TurboCodec import DEC_LargeRNN



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

        self.cc_channel = 1

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")
        if args.img_size==32:
            if args.info_img_size == 16:
                self.cc_decode = nn.Sequential(
                    #input is cc_channel,
                    nn.Conv2d(self.cc_channel, self.ndf, 5, 2, 2, bias=True),
                    nn.LeakyReLU(0.2, inplace=False),
                    nn.Conv2d(self.ndf, 1, 3, 1, 1, bias=True)
                )
            elif args.info_img_size == 18:
                self.turbo_dec = DEC_LargeRNN(args)
        else: # img_size=64
            if args.info_img_size ==32:
                pass

            elif args.info_img_size == 64:
                pass

        self.dec_layers = torch.nn.ModuleList()

        for iter in range(args.num_layer_Dec):
            if iter == 0:
                this_input_channel = self.nc
            else:
                this_input_channel = iter*self.ndf +self.nc

            if iter!=args.num_layer_Dec-1:
                this_output_channel = self.ndf
            else:
                this_output_channel = self.ud

            if iter!=args.num_layer_Dec-1:
                this_enc = nn.Sequential(
                    nn.Conv2d(this_input_channel, this_output_channel, 3, 1, 1, bias=True),
                    nn.BatchNorm2d(this_output_channel),
                    nn.LeakyReLU(0.2, inplace=False)
                )
            else:
                this_enc = nn.Sequential(
                    nn.Conv2d(this_input_channel, this_output_channel, 3, 1, 1, bias=True),
                )
            self.dec_layers.append(this_enc)


        # self.dec_Img = nn.Sequential(
        #     #input is nc x 64 x64
        #     nn.Conv2d(self.nc, self.ndf, 3, 1, 1, bias=False),
        #     #nn.BatchNorm2d(self.ndf),
        #     nn.LeakyReLU(0.2, inplace=False)
        # )
        #
        # self.decode1 = nn.Sequential(
        #     #input is ndf x 64 x64
        #     nn.Conv2d(self.ndf, self.ndf, 3, 1, 1, bias=False),
        #     #nn.BatchNorm2d(self.ndf),
        #     nn.LeakyReLU(0.2, inplace=False)
        # )
        #
        # self.decode2 = nn.Sequential(
        #     #input is 2*ndf x 64x 64
        #     nn.Conv2d((2*self.ndf), self.ndf, 3, 1, 1, bias=False),
        #     #nn.BatchNorm2d(self.ndf),
        #     nn.LeakyReLU(0.2, inplace=False)
        # )
        #
        # self.decode3 = nn.Sequential(
        #     #input is 3*ndf x 64 x64
        #     nn.Conv2d((3*self.ndf), self.ud, 3, 1, 1, bias=False)
        # )
        self.sigmd = nn.Sigmoid()

    def forward(self, img):
        # representation learning part
        # output1 = self.dec_Img(img)
        # output2 = self.decode1(output1)
        # output3 = torch.cat([output1, output2], dim=1)
        # output3 = self.decode2(output3)
        # output4 = torch.cat([output1, output2, output3], dim=1)
        # repr_inputs = self.decode3(output4)

        dense_input = img

        for iter in range(self.args.num_layer_Dec):
            this_output = self.dec_layers[iter](dense_input)
            if iter!=self.args.num_layer_Dec-1:
                dense_input = torch.cat([this_output,dense_input ], dim=1)

        repr_inputs = this_output

        # from representation to channel coding decoding
        # cc_decode map (B, C, H, W) to a block of (B, L, 1)
        # V1: CNN encoder, L = 16*16, H=W=32, C=1
        if self.args.info_img_size == 16:
            final = self.cc_decode(repr_inputs)
        elif self.args.info_img_size == 18:
            u_shape = repr_inputs.shape
            u_get = repr_inputs.view(u_shape[0], u_shape[2]*u_shape[3])
            u_get = u_get[:, :self.args.info_img_size**2*3].contiguous()
            repr_inputs = u_get.view(u_shape[0]*4,int(self.args.info_img_size**2/4), 3)

            final = self.turbo_dec(repr_inputs)
            final = final.view(u_shape[0],1, self.args.info_img_size, self.args.info_img_size )
        else:
            final = repr_inputs

        final = self.sigmd(final)

        return final

