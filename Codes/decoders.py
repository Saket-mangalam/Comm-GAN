__author__ = 'yihanjiang'


import torch.nn as nn
import torch.nn.functional as F
import torch


class DCGAN_Decoder(nn.Module):
    def __init__(self, args):
        super(DCGAN_Decoder, self).__init__()

        self.args = args

        self.u_rnn = nn.GRU(args.code_rate,  args.dec_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.u_fcnn = nn.Linear(2*args.dec_num_unit, 1, bias=True)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.img_channel, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, args.block_len * args.code_rate))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        received = self.adv_layer(out)
        received = received.view(out.shape[0], self.args.block_len, self.args.code_rate)
        # final decoder
        dec, _   = self.u_rnn(received)
        decoded  = F.sigmoid(self.u_fcnn(dec))

        return decoded


# this is 1D RNN encoder + DC-CNN as G.
class rnn_Decoder(nn.Module):
    def __init__(self, args):
        super(rnn_Decoder, self).__init__()

        self.args = args

        # self.u_rnn =  GridGRU(args.code_rate, num_unit = args.enc_num_unit, num_layer = 2)
        # self.u_fcnn = nn.Linear(4*args.enc_num_unit, 1, bias=True)

        self.u_rnn =  nn.GRU(args.code_rate, args.dec_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.u_fcnn = nn.Linear(2*args.dec_num_unit, 1, bias=True)

        def discriminator_block(in_filters, out_filters, kernel_size,   bn=True, stride = 2, padding = 0):
            block = [   nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.img_channel, 16, 3,stride = 1,  bn=False, padding=2),
            *discriminator_block(16, 32, 3, stride=2, padding=2),
            *discriminator_block(32, 64, 3, stride=2, padding=2),
            *discriminator_block(64, self.args.code_rate, 3, stride=1, padding=1),
        )

        # The height and width of downsampled image
        #ds_size = args.img_size // 2**4
        #self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, args.block_len * args.code_rate))

    def forward(self, img):
        out = self.model(img)

        out = out.permute(0, 2,3, 1)
        received = out.view(out.shape[0], self.args.block_len, self.args.code_rate)
        # final decoder
        dec, _   = self.u_rnn(received)
        decoded  = F.sigmoid(self.u_fcnn(dec))

        return decoded

# this is 2D RNN encoder + DC-CNN as G.
#from rnns.gridgru import GridGRU
class gridrnn_Decoder(nn.Module):
    def __init__(self, args):
        super(gridrnn_Decoder, self).__init__()

        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.u_rnn =  nn.GRU(args.code_rate, num_unit = args.dec_num_unit, num_layer = 2, device = self.this_device)
        self.u_fcnn = nn.Linear(4*args.dec_num_unit, 1, bias=True)

        # self.u_rnn =  nn.GRU(args.code_rate, args.dec_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        # self.u_fcnn = nn.Linear(2*args.dec_num_unit, 1, bias=True)

        def discriminator_block(in_filters, out_filters, kernel_size,   bn=True, stride = 2, padding = 0):
            block = [   nn.Conv2d(in_filters, out_filters, kernel_size, stride, padding, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.img_channel, 16, 3,stride = 1,  bn=False, padding=2),
            *discriminator_block(16, 32, 3, stride=2, padding=2),
            *discriminator_block(32, 64, 3, stride=2, padding=2),
            *discriminator_block(64, self.args.code_rate, 3, stride=1, padding=1),
        )

        # The height and width of downsampled image
        #ds_size = args.img_size // 2**4
        #self.adv_layer = nn.Sequential( nn.Linear(128*ds_size**2, args.block_len * args.code_rate))

    def forward(self, img):
        out = self.model(img)

        out = out.permute(0, 2, 3, 1)
        received = out.view(out.shape[0], self.args.block_len, self.args.code_rate)
        # final decoder
        dec      = self.u_rnn(received)
        decoded  = F.sigmoid(self.u_fcnn(dec))

        return decoded

