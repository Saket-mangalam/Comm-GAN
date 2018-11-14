

import torch.nn as nn
import torch.nn.functional as F
import torch

import numpy as np


# FCNN generator. One big generator for everything.
class FCNN_Generator(nn.Module):
    def __init__(self, args):
        super(FCNN_Generator, self).__init__()

        self.args      = args
        self.img_shape = (args.img_channel, args.img_size, args.img_size)

        # self.embed_rnn = torch.nn.GRU(1, args.enc_num_unit,
        #                                    num_layers=args.enc_num_layer, bias=True, batch_first=True,
        #                                    dropout=0, bidirectional=True)

        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.args.latent_dim+self.args.block_len, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, infos):
        # Concatenate label embedding and image to produce input

        info_embedding = infos
        gen_input = torch.cat((info_embedding, noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img

class DCGAN_Generator(nn.Module):
    def __init__(self, args):
        super(DCGAN_Generator, self).__init__()

        self.args      = args

        self.init_size = args.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim + args.code_rate*args.block_len, 128*self.init_size**2))

        self.u_rnn = nn.GRU(1,  args.enc_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.u_fcnn = nn.Linear(2*args.enc_num_unit, args.code_rate, bias=True)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.args.img_channel, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z, u):
        rnn_enc, _ = self.u_rnn(u)
        rnn_enc    = self.u_fcnn(rnn_enc)
        zz         = torch.cat([z, rnn_enc.view(z.shape[0], -1)], dim = 1)
        out        = self.l1(zz)
        out        = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img        = self.conv_blocks(out)
        return img

#from rnns.gridgru import GridGRU

class rnn_Generator(nn.Module):
    def __init__(self, args):
        super(rnn_Generator, self).__init__()

        self.args      = args

        #self.init_size = args.img_size // 4
        self.init_size = int(np.sqrt(args.block_len))

        # self.u_rnn =  GridGRU(1, num_unit = args.enc_num_unit, num_layer = 2)
        # self.u_fcnn = nn.Linear(4*args.enc_num_unit, args.code_rate, bias=True)
        #
        # self.z_rnn =  GridGRU(1, num_unit = args.enc_num_unit, num_layer = 2)
        # self.z_fcnn = nn.Linear(4*args.enc_num_unit, args.code_rate, bias=True)

        self.u_rnn =  nn.GRU(1, args.enc_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.u_fcnn = nn.Linear(2*args.enc_num_unit, args.code_rate, bias=True)

        self.z_rnn =  nn.GRU(1, args.enc_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.z_fcnn = nn.Linear(2*args.enc_num_unit, args.code_rate, bias=True)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block


        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.args.code_rate *2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(self.args.code_rate *2, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 16, 5, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, args.img_channel, 7, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, z, u):
        u_enc, _ = self.u_rnn(u)
        u_enc    = self.u_fcnn(u_enc)

        z        = z.view(z.shape[0], z.shape[1], 1)
        z_enc, _ = self.z_rnn(z)
        z_enc    = self.z_fcnn(z_enc)

        zz       = torch.cat([u_enc, z_enc], dim = 2)
        zz       = zz.view(zz.shape[0], self.init_size, self.init_size, zz.shape[2])
        zz       = zz.permute(0,3,1,2)

        zz       = zz.contiguous()

        img      = self.conv_blocks(zz)

        return img



class gridrnn_Generator(nn.Module):
    def __init__(self, args):
        super(gridrnn_Generator, self).__init__()

        self.args      = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        #self.init_size = args.img_size // 4
        self.init_size = int(np.sqrt(args.block_len))

        self.u_rnn =  GridGRU(1, num_unit = args.enc_num_unit, num_layer = 2, device = self.this_device)
        self.u_fcnn = nn.Linear(4*args.enc_num_unit, args.code_rate, bias=True)
        #
        self.z_rnn =  GridGRU(1, num_unit = args.enc_num_unit, num_layer = 2, device = self.this_device)
        self.z_fcnn = nn.Linear(4*args.enc_num_unit, args.code_rate, bias=True)

        # self.u_rnn =  nn.GRU(1, args.enc_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        # self.u_fcnn = nn.Linear(2*args.enc_num_unit, args.code_rate, bias=True)

        # self.z_rnn =  nn.GRU(1, args.enc_num_unit, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        # self.z_fcnn = nn.Linear(2*args.enc_num_unit, args.code_rate, bias=True)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block


        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.args.code_rate *2),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(self.args.code_rate *2, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(128, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 16, 5, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, args.img_channel, 7, stride=1, padding=1),
            nn.Tanh()
        )


    def forward(self, z, u):
        u_enc    = self.u_rnn(u)
        u_enc    = self.u_fcnn(u_enc)

        z        = z.view(z.shape[0], z.shape[1], 1)
        z_enc    = self.z_rnn(z)
        z_enc    = self.z_fcnn(z_enc)

        zz       = torch.cat([u_enc, z_enc], dim = 2)
        zz       = zz.view(zz.shape[0], self.init_size, self.init_size, zz.shape[2])
        zz       = zz.permute(0,3,1,2)

        zz       = zz.contiguous()

        img      = self.conv_blocks(zz)

        return img
    
class Hidden_Generator_1(nn.Module):
    def __init__(self, args):
        super(Hidden_Generator_1, self).__init__()

        self.args      = args
        #self.img_shape = (args.img_channel, args.img_size, args.img_size)

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        #self.init_size = args.img_size // 4
        #self.init_size = args.img_size
        
        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Conv2d(in_feat, out_feat, 3, stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.conv_block_1 = nn.Sequential(
                *block(self.args.img_channel,64),
                *block(64,64),
                *block(64,64),
                *block(64,64)
                )
        
        
        self.conv_block_2 = nn.Sequential(
                *block((64+self.args.block_len+self.args.img_channel),64),
                nn.Conv2d(64, self.args.img_channel, 1, stride = 1, padding = 0),
                nn.BatchNorm2d(self.args.img_channel, 0.8)
                )
        
        
    def forward(self,z,u):
        encready_z = self.conv_block_1(z)
            #u = self.l1(u)
            #u = u.view(u.shape[0], 1, self.init_size, self.init_size)
            
            #u.unsqueeze_(-1)
            #u = u.expand(self.args.batch_size, self.args.block_len, self.args.img_size, self.args.img_size)
            #u.unsqueeze_(-1)
            #u = u.expand(self.args.block_len,self.args.img_size,self.args.img_size)
            
        # x = torch.zeros(self.args.img_size)
        #u = torch.add(u,1,x)
        u = u.unsqueeze_(-1)
        u = u.expand(self.args.batch_size,self.args.block_len,self.args.img_size)
        u = u.unsqueeze_(-1)
        #u = u.expand(self.args.batch_size,self.args.block_len,self.args.img_size,self.args.img_size) 
        encodable_u = u.expand(self.args.batch_size,self.args.block_len,self.args.img_size,self.args.img_size)
        enc = torch.cat([encodable_u,encready_z,z], dim = 1)
        enc = self.conv_block_2(enc)
        return enc
        


class Hidden_Generator_2(nn.Module):
    def __init__(self, args):
        super(Hidden_Generator_2, self).__init__()

        self.args      = args
        #self.img_shape = (args.img_channel, args.img_size, args.img_size)

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        #self.init_size = args.img_size // 4
        #self.init_size = args.img_size
        
        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Conv2d(in_feat, out_feat, 3, stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def c_block(in_feat, out_feat, normalize=True):
            layers = [ nn.Conv1d(in_feat,out_feat,3,stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ELU(0.2, inplace=True))
            return layers
        
        self.conv1d_block = nn.Sequential(
                *c_block(1,3),
                *c_block(3,3),
                *c_block(3,3))
        
        self.conv_block_1 = nn.Sequential(
                *block(self.args.img_channel,64),
                *block(64,64),
                *block(64,64),
                *block(64,64)
                )
        
        
        self.conv_block_2 = nn.Sequential(
                *block((64+(3*self.args.block_len)+self.args.img_channel),64),
                nn.Conv2d(64, self.args.img_channel, 1, stride = 1, padding = 0),
                nn.BatchNorm2d(self.args.img_channel, 0.8)
                )
        
        
    def forward(self,z,u):
        encready_z = self.conv_block_1(z)
        #u = self.l1(u)
        #u = u.view(u.shape[0], 1, self.init_size, self.init_size)
        
        #u.unsqueeze_(-1)
        #u = u.expand(self.args.batch_size, self.args.block_len, self.args.img_size, self.args.img_size)
        #u.unsqueeze_(-1)
        #u = u.expand(self.args.block_len,self.args.img_size,self.args.img_size)
            
        # x = torch.zeros(self.args.img_size)
        #u = torch.add(u,1,x)
        u = u.unsqueeze_(-1)
        #u = u.expand(self.args.batch_size,self.args.block_len,1)
        #u = u.unsqueeze_(0)
        #u = u.expand(self.args.batch_size,self.args.block_len,self.args.img_size,self.args.img_size)
        #permute and convolve
        u = u.permute(1,2,0)
        u = self.conv1d_block(u)
        #how to concat all along dim 1 to dim 2
        u = u.permute(2,0,1)
        u = u.contiguous() 
        u = u.view(self.args.batch_size,-1)
        u = u.unsqueeze_(-1)
        encodable_u = u.expand(self.args.batch_size,3*self.args.block_len,self.args.img_size)
        u = u.unsqueeze_(-1)
        encodable_u = u.expand(self.args.batch_size,3*self.args.block_len,self.args.img_size,self.args.img_size)
        enc = torch.cat([encodable_u,encready_z,z], dim = 1)
        enc = self.conv_block_2(enc)
        return enc
        
class Hidden_Generator_3(nn.Module):
    def __init__(self, args):
        super(Hidden_Generator_3, self).__init__()

        self.args      = args
        #self.img_shape = (args.img_channel, args.img_size, args.img_size)

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")


        
        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Conv2d(in_feat, out_feat, 3, stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def c_block(in_feat, out_feat, normalize=True):
            layers = [ nn.Conv1d(in_feat,out_feat,3,stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ELU(0.2, inplace=True))
            return layers
        
        self.conv1d_block = nn.Sequential(
                *c_block(1,3),
                *c_block(3,3),
                *c_block(3,3))
        
        self.conv_block_1 = nn.Sequential(
                *block(self.args.img_channel,64),
                *block(64,64),
                *block(64,64),
                *block(64,64)
                )
        
        
        self.conv_block_2 = nn.Sequential(
                *block((67+self.args.img_channel),64),
                nn.Conv2d(64, self.args.img_channel, 1, stride = 1, padding = 0),
                nn.BatchNorm2d(self.args.img_channel, 0.8)
                )
        
        
    def forward(self,z,u):
        encready_z = self.conv_block_1(z)
        #add extra dimension to the tensor to make it 3D
        u = u.unsqueeze_(-1)#64,(32*32),1
        #u = u.expand(self.args.batch_size,self.args.block_len,1)
        #permute and convolve
        u = u.permute(0,2,1)        #64, 1, 32*32
        u = self.conv1d_block(u)    #64,3,32*32
        #how to concat all along dim 1 to dim 2
        #u = u.permute(2,0,1)#64,(32*32),3
        u = u.contiguous() 
        #u = u.view(self.args.batch_size,-1)#64,(32*32*3)
        
        encodable_u = u.view(self.args.batch_size, 3,self.args.img_size, self.args.img_size)
        #64,3,32,32
        enc = torch.cat([encodable_u,encready_z,z], dim = 1)#64,70,32,32
        enc = self.conv_block_2(enc)#64,3,32,32
        return enc
    

class Hidden_Generator_4(nn.Module):
    def __init__(self, args):
        super(Hidden_Generator_4, self).__init__()

        self.args      = args
        #self.img_shape = (args.img_channel, args.img_size, args.img_size)

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")


        
        def block(in_feat, out_feat, normalize=True):
            layers = [  nn.Conv2d(in_feat, out_feat, 3, stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def c_block(in_feat, out_feat, normalize=True):
            layers = [ nn.Conv1d(in_feat,out_feat,3,stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.ELU(0.2, inplace=True))
            return layers
        
        self.conv1d_block = nn.Sequential(
                *c_block(1,64),
                *c_block(64,128),
                *c_block(128,256))
        
        self.conv_block_1 = nn.Sequential(
                *block(self.args.img_channel,64),
                *block(64,64),
                *block(64,64),
                *block(64,64)
                )
        
        
        self.conv_block_2 = nn.Sequential(
                *block((65+self.args.img_channel),64),
                nn.Conv2d(64, self.args.img_channel, 1, stride = 1, padding = 0),
                nn.BatchNorm2d(self.args.img_channel, 0.8)
                )
        
        
    def forward(self,z,u):
        #z is the image, u is message
        encready_z = self.conv_block_1(z)
        # u is of size 64, (4/4^2/4^3)
        u = u.unsqueeze_(-1) # add 3rd dimension 64,4,1
        u = u.permute(0,2,1)#make it 64,1,4
        u = self.conv1d_block(u)# 64,256,4
        #u = u.view(self.args.batch_size,-1) # 64,1024
        u = u.view(self.args.batch_size,1,self.args.img_size,self.args.img_size)#64,1,32,32
        enc = torch.cat([u,encready_z,z],dim=1)
        enc = self.conv_block_2(enc)
        return enc
    
    
if __name__ == '__main__':
    print('Generators initialized')