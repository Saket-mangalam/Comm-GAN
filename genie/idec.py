__author__ = 'SaketM'
import torch
import torch.nn.functional as F
import torch.nn as nn

from interleavers import Interleaver, Interleaver2D, DeInterleaver, DeInterleaver2D
from cnn_utils import SameShapeConv2d, SameShapeConv1d


class Dense_Decoder(nn.Module):
    def __init__(self, args):
        super(Dense_Decoder, self).__init__()
        self.args = args
        #self.nc = self.args.img_channels
        #self.ndf = self.args.dec_lat
        #self.ud = self.args.bitsperpix

        #self.cc_channel = 1

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.dec_layers = torch.nn.ModuleList()

        for iter in range(args.idec_num_layer):
            if iter == 0:
                this_input_channel = self.args.img_channels
            else:
                this_input_channel = iter*self.args.idec_num_unit +self.args.img_channels

            if iter!=args.idec_num_layer-1:
                this_output_channel = self.args.idec_num_unit
            else:
                this_output_channel = self.args.code_rate_n

            if iter!=args.idec_num_layer-1:
                this_enc = nn.Sequential(
                    nn.Conv2d(this_input_channel, this_output_channel, 3, 1, 1, bias=True),
                    nn.LeakyReLU(0.2, inplace=False)
                )
            else:
                this_enc = nn.Sequential(
                    nn.Conv2d(this_input_channel, this_output_channel, 3, 1, 1, bias=True),
                )
            self.dec_layers.append(this_enc)

            self.norm = torch.nn.BatchNorm2d(self.args.code_rate_n,  affine=False)


    def forward(self, img):
        dense_input = img

        for iter in range(self.args.idec_num_layer):
            this_output = self.dec_layers[iter](dense_input)
            if iter!=self.args.idec_num_layer-1:
                dense_input = torch.cat([this_output,dense_input ], dim=1)

        repr_inputs = this_output
        final = self.norm(repr_inputs)

        return final
