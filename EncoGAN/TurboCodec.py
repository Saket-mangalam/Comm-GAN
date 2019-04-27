__author__ = 'yihanjiang'
import torch.nn as nn
import torch
import torch.functional as F
import numpy as np
seed = 0



class Interleaver(nn.Module):
    def __init__(self, args, p_array):
        super(Interleaver, self).__init__()
        self.args = args
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def set_parray(self, p_array):
        self.p_array = torch.LongTensor(p_array).view(len(p_array))

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2)
        res    = inputs[self.p_array]
        res    = res.permute(1, 0, 2)

        return res

class DeInterleaver(nn.Module):
    def __init__(self, args, p_array):
        super(DeInterleaver, self).__init__()
        self.args = args

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[int(p_array[idx])] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(len(p_array))

    def set_parray(self, p_array):

        self.reverse_p_array = [0 for _ in range(len(p_array))]
        for idx in range(len(p_array)):
            self.reverse_p_array[p_array[idx]] = idx

        self.reverse_p_array = torch.LongTensor(self.reverse_p_array).view(self.args.block_len)

    def forward(self, inputs):
        inputs = inputs.permute(1,0,2)
        res    = inputs[self.reverse_p_array]
        res    = res.permute(1, 0, 2)

        return res


class ENC_interRNN(nn.Module):
    def __init__(self, args):
        # turbofy only for code rate 1/3
        super(ENC_interRNN, self).__init__()
        self.args             = args
        rand_gen = np.random.mtrand.RandomState(seed)

        p_array = rand_gen.permutation(np.arange(args.block_len))
        self.norm_layer = torch.nn.BatchNorm1d(args.code_rate,  affine=False)
        # Encoder
        if args.enc_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.enc_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.enc_rnn_1       = RNN_MODEL(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_1    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.enc_rnn_2       = RNN_MODEL(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_2    = torch.nn.Linear(2*args.enc_num_unit, 1)

        self.enc_rnn_3       = RNN_MODEL(1, args.enc_num_unit,
                                           num_layers=args.enc_num_layer, bias=True, batch_first=True,
                                           dropout=0, bidirectional=True)

        self.enc_linear_3    = torch.nn.Linear(2*args.enc_num_unit, 1)


        self.interleaver      = Interleaver(args, p_array)


    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def set_parallel(self):
        self.enc_rnn_1 = torch.nn.DataParallel(self.enc_rnn_1)
        self.enc_rnn_2 = torch.nn.DataParallel(self.enc_rnn_2)
        self.enc_rnn_3 = torch.nn.DataParallel(self.enc_rnn_3)
        self.enc_linear_1 = torch.nn.DataParallel(self.enc_linear_1)
        self.enc_linear_2 = torch.nn.DataParallel(self.enc_linear_2)
        self.enc_linear_3 = torch.nn.DataParallel(self.enc_linear_3)

    def enc_act(self, inputs):
        return inputs

    def forward(self, inputs):

        x_sys, _   = self.enc_rnn_1(inputs)
        x_sys      = self.enc_act(self.enc_linear_1(x_sys))

        x_p1, _    = self.enc_rnn_2(inputs)
        x_p1       = self.enc_act(self.enc_linear_2(x_p1))

        x_sys_int  = self.interleaver(inputs)

        x_p2, _    = self.enc_rnn_3(x_sys_int)
        x_p2       = self.enc_act(self.enc_linear_3(x_p2))

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)

        x_input_tmp  = x_tx.permute(0,2,1).contiguous()
        x_input_norm = self.norm_layer(x_input_tmp)
        x_tx         = x_input_norm.permute(0,2,1).contiguous()

        return x_tx


class DEC_LargeRNN(nn.Module):
    def __init__(self, args):
        super(DEC_LargeRNN, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        rand_gen = np.random.mtrand.RandomState(seed)
        p_array = rand_gen.permutation(np.arange(args.block_len))

        self.interleaver          = Interleaver(args, p_array)
        self.deinterleaver        = DeInterleaver(args, p_array)

        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.dropout = torch.nn.Dropout(args.dropout)

        self.dec1_rnns      = torch.nn.ModuleList()
        self.dec2_rnns      = torch.nn.ModuleList()
        self.dec1_outputs   = torch.nn.ModuleList()
        self.dec2_outputs   = torch.nn.ModuleList()

        for idx in range(args.num_iteration):
            self.dec1_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                                        num_layers=2, bias=True, batch_first=True,
                                                        dropout=args.dropout, bidirectional=True)
            )

            self.dec2_rnns.append(RNN_MODEL(2 + args.num_iter_ft,  args.dec_num_unit,
                                           num_layers=2, bias=True, batch_first=True,
                                           dropout=args.dropout, bidirectional=True)
            )
            self.dec1_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))

            if idx == args.num_iteration -1:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, 1))
            else:
                self.dec2_outputs.append(torch.nn.Linear(2*args.dec_num_unit, args.num_iter_ft))


    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.dec_act == 'linear':
            return inputs
        else:
            return F.elu(inputs)

    def set_parallel(self):
        for idx in range(self.args.num_iteration):
            self.dec1_rnns[idx] = torch.nn.DataParallel(self.dec1_rnns[idx])
            self.dec2_rnns[idx] = torch.nn.DataParallel(self.dec2_rnns[idx])
            self.dec1_outputs[idx] = torch.nn.DataParallel(self.dec1_outputs[idx])
            self.dec2_outputs[idx] = torch.nn.DataParallel(self.dec2_outputs[idx])

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)
        self.deinterleaver.set_parray(p_array)


    def forward(self, received):
        rec_shape = received.shape
        received = received.type(torch.FloatTensor).to(self.this_device)
        # Turbo Decoder
        r_sys     = received[:,:,0].view((rec_shape[0], self.args.block_len, 1))
        r_sys_int = self.interleaver(r_sys)
        r_par1    = received[:,:,1].view((rec_shape[0], self.args.block_len, 1))
        r_par2    = received[:,:,2].view((rec_shape[0], self.args.block_len, 1))

        #num_iteration,
        prior = torch.zeros((rec_shape[0], self.args.block_len, self.args.num_iter_ft)).to(self.this_device)

        for idx in range(self.args.num_iteration - 1):
            x_this_dec = torch.cat([r_sys, r_par1, prior], dim = 2)

            x_dec, _   = self.dec1_rnns[idx](x_this_dec)

            #x_plr      = self.dec1_outputs[idx](x_dec)
            x_plr      = self.dec_act(self.dropout(self.dec1_outputs[idx](x_dec)))

            x_plr = x_plr - prior

            x_plr_int  = self.interleaver(x_plr)

            x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

            x_dec, _   = self.dec2_rnns[idx](x_this_dec)

            x_plr      = self.dec_act(self.dropout(self.dec2_outputs[idx](x_dec)))

            x_plr = x_plr - x_plr_int

            prior      = self.deinterleaver(x_plr)

        # last round
        x_this_dec = torch.cat([r_sys,r_par1, prior], dim = 2)
        x_dec, _   = self.dec1_rnns[self.args.num_iteration - 1](x_this_dec)
        x_plr      = self.dec_act(self.dropout(self.dec1_outputs[self.args.num_iteration - 1](x_dec)))

        x_plr = x_plr - prior

        x_plr_int  = self.interleaver(x_plr)

        x_this_dec = torch.cat([r_sys_int, r_par2, x_plr_int ], dim = 2)

        x_dec, _   = self.dec2_rnns[self.args.num_iteration - 1](x_this_dec)

        x_plr      = self.dec_act(self.dropout(self.dec2_outputs[self.args.num_iteration - 1](x_dec)))

        logit      = self.deinterleaver(x_plr)

        final      = torch.sigmoid(logit)

        return final


class TurboEncoder(nn.Module):
    def __init__(self, args):
        super(TurboEncoder, self).__init__()
        self.args = args
        self.nc        = self.args.img_channels
        self.nef       = self.args.enc_lat
        self.ud        = self.args.bitsperpix
        self.code_rate = self.args.code_rate

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        self.turbo_enc = ENC_interRNN(args)

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
        # channel coding 1D encoding...
        # chop u to multiple blocks, with size (B*M, L, 1), M is the number of blocks
        u_shape = u.shape
        num_block_per_batch = int(self.args.img_size*self.args.img_size/self.args.block_len)
        u      = u.view(self.args.batch_size, num_block_per_batch, self.args.block_len, 1)
        u      = u.view(self.args.batch_size*num_block_per_batch, self.args.block_len, 1)

        encoded = self.turbo_enc(u)
        u       = encoded.view(self.args.batch_size, self.code_rate, u_shape[2], u_shape[3])

        # project the 1d encoding to image...
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

class TurboDecoder(nn.Module):
    def __init__(self, args):
        super(TurboDecoder, self).__init__()
        self.args = args
        self.nc = self.args.img_channels
        self.ndf = self.args.dec_lat
        self.ud = self.args.bitsperpix

        self.decoder = DEC_LargeRNN(args)

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

        num_block_per_batch = int(self.args.img_size*self.args.img_size/self.args.block_len)

        y = output4.view(self.args.batch_size*num_block_per_batch, self.args.block_len, self.args.code_rate)
        final = self.decoder(y)

        final = final.view(self.args.batch_size, int(self.args.bitsperpix/self.args.code_rate), self.args.img_size, self.args.img_size)

        return final

