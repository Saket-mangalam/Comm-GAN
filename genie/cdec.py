__author__ = 'SaketM'
import torch
import torch.nn.functional as F
import torch.nn as nn

from interleavers import Interleaver, Interleaver2D, DeInterleaver, DeInterleaver2D
from cnn_utils import SameShapeConv2d, SameShapeConv1d


class TurboAE_decoder1D(nn.Module):
    def __init__(self, args, p_array):
        super(TurboAE_decoder1D, self).__init__()
        self.args = args
        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.cdec_num_unit, kernel_size = args.dec_kernel_size)
            )

            self.dec2_cnns.append(SameShapeConv1d(num_layer=args.dec_num_layer, in_channels=2 + args.num_iter_ft,
                                                  out_channels= args.cdec_num_unit, kernel_size = args.dec_kernel_size)
            )
            self.dec1_outputs.append(torch.nn.Linear(args.cdec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(args.cdec_num_unit, args.code_rate_k))
            else:
                self.dec2_outputs.append(torch.nn.Linear(args.cdec_num_unit, args.num_iter_ft))

        # also need some CNN for f
        self.ftstart =  SameShapeConv2d(num_layer=args.dec_num_layer, in_channels=args.img_channels,
                                                  out_channels= args.cdec_num_unit, kernel_size = args.dec_kernel_size)
        self.ftend   =  SameShapeConv2d(num_layer=1, in_channels=args.cdec_num_unit,
                                                  out_channels= args.img_channels, kernel_size = args.dec_kernel_size)


    def forward(self, received, ie_decoded):
        received = received.type(torch.FloatTensor).to(self.this_device)

        received = self.ftstart(received)
        received = self.ftend(received)

        # Turbo Decoder
        r_sys     = received[:,0,:, :].view((self.args.batch_size, self.args.img_size*self.args.img_size, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,1,:, :].view((self.args.batch_size, self.args.img_size*self.args.img_size, 1))
        r_par2    = received[:,2,:, :].view((self.args.batch_size, self.args.img_size*self.args.img_size, 1))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.img_size*self.args.img_size, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)
            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))
        final      = final.view(self.args.batch_size, self.args.code_rate_k, self.args.img_size,self.args.img_size )

        return final


class TurboAE_decoder2D(nn.Module):
    def __init__(self, args, p_array):
        super(TurboAE_decoder2D, self).__init__()
        self.args = args
        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.interleaver          = Interleaver2D(args, p_array)
        self.deinterleaver        = DeInterleaver2D(args, p_array)

        self.dec1_cnns      = torch.nn.ModuleList()
        self.dec2_cnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_cnns.append(SameShapeConv2d(num_layer=args.cdec_num_layer, in_channels=args.img_channels - 1 + args.num_iter_ft,
                                                  out_channels= args.cdec_num_unit, kernel_size = args.dec_kernel_size)
            )

            self.dec2_cnns.append(SameShapeConv2d(num_layer=args.cdec_num_layer, in_channels=args.img_channels - 1 + args.num_iter_ft,
                                                  out_channels= args.cdec_num_unit, kernel_size = args.dec_kernel_size)
            )
            self.dec1_outputs.append(SameShapeConv2d(1, args.cdec_num_unit,args.num_iter_ft , kernel_size=1))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(SameShapeConv2d(1, args.cdec_num_unit, args.code_rate_k, kernel_size=1, no_act = True))
            else:
                self.dec2_outputs.append(SameShapeConv2d(1, args.cdec_num_unit,args.num_iter_ft , kernel_size=1))

    def forward(self, received, ie_decoded):
        if self.args.cdec_type == 'turboae2d_img':
            received = received.type(torch.FloatTensor).to(self.this_device)
        else:
            received = ie_decoded.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,0,:, :].view((self.args.batch_size, 1, self.args.img_size, self.args.img_size))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,1,:, :].view((self.args.batch_size, 1, self.args.img_size, self.args.img_size))
        r_par2    = received[:,2,:, :].view((self.args.batch_size, 1, self.args.img_size, self.args.img_size))

        #num_iteration,
        prior = torch.zeros((self.args.batch_size, self.args.num_iter_ft, self.args.img_size, self.args.img_size)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 1)
            x_dec  = self.dec1_cnns[idx](x_this_dec)
            x_plr      = self.dec1_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 1)

            x_dec  = self.dec2_cnns[idx](x_this_dec)

            x_plr      = self.dec2_outputs[idx](x_dec)

            if self.args.extrinsic:
                x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 1)

        x_dec     = self.dec1_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec1_outputs[self.args.num_iteration - 1](x_dec)

        if self.args.extrinsic:
            x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 1)

        x_dec     = self.dec2_cnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec2_outputs[self.args.num_iteration - 1](x_dec)

        final      = torch.sigmoid(self.deinterleaver(x_plr))

        return final

# Wanna try dense decoder.....

class CNN_decoder2D(nn.Module):
    def __init__(self, args, p_array):
        super(CNN_decoder2D, self).__init__()
        self.args = args
        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")



        self.cnn   = SameShapeConv2d(num_layer=args.cdec_num_layer, in_channels=args.img_channels,
                                     out_channels= args.cdec_num_unit, kernel_size = args.dec_kernel_size)

        self.final = SameShapeConv2d(num_layer=1, in_channels=args.cdec_num_unit,
                                                  out_channels= args.code_rate_k, kernel_size = 1, no_act = True)


    def forward(self, img, ie_decoded):
        received   = img.type(torch.FloatTensor).to(self.this_device)
        ie_decoded = ie_decoded.type(torch.FloatTensor).to(self.this_device)

        x_plr    = self.cnn(received)
        final    = torch.sigmoid(self.final(x_plr))

        return final
