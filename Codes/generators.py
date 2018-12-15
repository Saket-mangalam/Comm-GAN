

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
                *c_block(3,6),
                *c_block(6,12))
        
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
                *c_block(1,2),
                *c_block(2,4),
                *c_block(4,4))
        
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

class Hidden_Generator_5(nn.Module):
    def __init__(self, args):
        super(Hidden_Generator_5, self).__init__()

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
            layers = [ nn.Conv2d(in_feat,out_feat,3,stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ELU(0.8, inplace=True))
            return layers
        
        self.conv2d_block = nn.Sequential(
                *c_block(1,2),
                *c_block(2,4),
                *c_block(4,4))
        
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
        # u is of size 64,x^2
        u = u.view(self.args.batch_size,1,int(self.args.block_len**0.5),int(self.args.block_len**0.5)) # 64,1,x,x
        u = self.conv2d_block(u)
        #u = u.view(self.args.batch_size,-1) # 64,1024
        u = u.view(self.args.batch_size,1,self.args.img_size,self.args.img_size)#64,1,32,32
        enc = torch.cat([u,encready_z,z],dim=1)
        enc = self.conv_block_2(enc)
        return enc
    


class Hidden_Generator_6(nn.Module):
    def __init__(self, args):
        super(Hidden_Generator_6, self).__init__()

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
            layers = [ nn.Conv2d(in_feat,out_feat,3,stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ELU(0.8, inplace=True))
            return layers
        
        def down_block(in_feat,out_feat,normalize=True):
            block = [nn.Linear(in_feat,out_feat)]
            if normalize:
                block.append(nn.BatchNorm1d(out_feat, 0.8))
            block.append(nn.LeakyReLU(0.2,inplace=True))
            return block
        
        def up_block(in_feat, out_feat, normalize=True):
            block = [nn.ConvTranspose2d(in_feat,out_feat,3,stride=1,padding=1)]
            if normalize:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        
        self.linear_1 = nn.Sequential(
                *down_block(self.args.block_len,512),
                *down_block(512,int((self.args.img_size**2)/4)))
        
        self.linear_2 = nn.Sequential(
                *down_block(self.args.sample_noise,512),
                *down_block(512,int((self.args.img_size**2)/4)))
        
        self.conv2d_block = nn.Sequential(
                *c_block(1,2),
                *c_block(2,4),
                *c_block(4,4))
        
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
        
        self.transformed = nn.Sequential(
                *up_block(1,16),
                *up_block(16,32),
                *up_block(32,64),
                *up_block(64,4*self.args.img_channel))
        
    def forward(self,z,u):
        #z is the awgn, u is message
        #Transform z to image first
        z = self.linear_2(z)
        z = z.view(self.args.batch_size,1,int(self.args.img_size/2),int(self.args.img_size/2))
        #tranform it in an image shape
        z = self.transformed(z)
        z = z.view(self.args.batch_size,self.args.img_channel,self.args.img_size,self.args.img_size)
        encready_z = self.conv_block_1(z)
        # u is of size 64,x^2
        u = self.linear_1(u)
        u = u.view(self.args.batch_size,1,int(self.args.img_size/2),int(self.args.img_size/2)) # 64,1,x,x
        u = self.conv2d_block(u)
        #u = u.view(self.args.batch_size,-1) # 64,1024
        u = u.view(self.args.batch_size,1,self.args.img_size,self.args.img_size)#64,1,32,32
        enc = torch.cat([u,encready_z,z],dim=1)
        enc = self.conv_block_2(enc)
        return enc

class Hidden_Generator_7(nn.Module):
    def __init__(self, args):
        super(Hidden_Generator_7, self).__init__()

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
            layers = [ nn.Conv2d(in_feat,out_feat,3,stride = 1, padding = 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.ELU(0.8, inplace=True))
            return layers
        
        def down_block(in_feat,out_feat,normalize=True):
            block = [nn.Linear(in_feat,out_feat)]
            if normalize:
                block.append(nn.BatchNorm1d(out_feat, 0.8))
            block.append(nn.LeakyReLU(0.2,inplace=True))
            return block
        
        def up_block(in_feat, out_feat, normalize=True):
            block = [nn.Upsample(scale_factor=2,mode='bilinear')]
            block.append(nn.Conv2d(in_feat,out_feat,3,stride=1,padding=1))
            if normalize:
                block.append(nn.BatchNorm2d(out_feat, 0.8))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        
        self.linear = nn.Sequential(
                *down_block(self.args.block_len,512),
                *down_block(512,int((self.args.img_size**2)/4)))
        
        
        self.conv2d_block = nn.Sequential(
                *c_block(1,2),
                *c_block(2,4),
                *c_block(4,4))
        
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
        
        self.transformed = nn.Sequential(
                *up_block(self.args.sample_noise,64), #2x2
                *up_block(64,128), #4x4
                *up_block(128,256), #8x8
                *up_block(256,128), #16x16
                *up_block(128,64), #32x32
                *block(64,self.args.img_channel)
                )
        
    def forward(self,z,u):
        #z is the awgn, u is message
        #Transform z to image first
        z = z.view(self.args.batch_size,self.args.sample_noise,1,1)
        
        #tranform it in an image shape
        z = self.transformed(z)
        #z = z.view(self.args.batch_size,self.args.img_channel,self.args.img_size,self.args.img_size)
        encready_z = self.conv_block_1(z)
        # u is of size 64,x^2
        u = self.linear(u)
        u = u.view(self.args.batch_size,1,int(self.args.img_size/2),int(self.args.img_size/2)) # 64,1,x,x
        u = self.conv2d_block(u)
        #u = u.view(self.args.batch_size,-1) # 64,1024
        u = u.view(self.args.batch_size,1,self.args.img_size,self.args.img_size)#64,1,32,32
        enc = torch.cat([u,encready_z,z],dim=1)
        enc = self.conv_block_2(enc)
        return enc

class Gan_Generator(nn.Module):
    def __init__(self, args):
        super(Gan_Generator, self).__init__()

        self.args      = args
        #self.img_shape = (args.img_channel, args.img_size, args.img_size)

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        
        def conv_block(in_feat, out_feat, normalize=True, kernel = 5, pad = 2):
            layers = [nn.Upsample(scale_factor=2,mode='bilinear')]
            layers.append(nn.Conv2d(in_feat, out_feat, kernel, stride = 1, padding = pad))
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def lin_block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat,bias=False)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        
        self.linear = nn.Sequential(
                *lin_block(self.args.sample_noise, 512),
                *lin_block(512,1024),
                *lin_block(1024,2048))
        
        self.conv = nn.Sequential(
                *conv_block(512,256),
                *conv_block(256,128),
                *conv_block(128,64),
                *conv_block(64,3))
        
        self.tanh = nn.Sequential(nn.Tanh())
        
    def forward(self,z):
        #z is awgn dim- batchsizexnoisesize(100)
        #convert z from 100 to 2048
        z = self.linear(z)
        # change the shape to 64x512x2x2
        z = z.view(self.args.batch_size,512,2,2)
        # upgrade and upsample using convolutions to 64x3x32x32
        z = self.conv(z)
        z = self.tanh(z)
        return z
    
class Enc_Generator(nn.Module):
    def __init__(self, args):
        super(Enc_Generator, self).__init__()

        self.args      = args
        #self.img_shape = (args.img_channel, args.img_size, args.img_size)

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        
        def up_conv_block(in_feat, out_feat, normalize=True, kernel = 5, pad = 2):
            layers =  [nn.Upsample(scale_factor=2,mode='bilinear')]
            layers.append(nn.Conv2d(in_feat, out_feat, kernel, stride = 1, padding = pad))
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        def conv_block(in_feat, out_feat, normalize=True, kernel_size = 5, pad = 2):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size, stride = 1, padding = pad)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.conv1 = nn.Sequential(
                *up_conv_block(16,8),
                *up_conv_block(8,4),
                *up_conv_block(4,1))
        
        self.conv2 = nn.Sequential(
                *conv_block(self.args.img_channel+1,32),
                *conv_block(32,64),
                *conv_block(64,32),
                *conv_block(32,self.args.img_channel))
        
        self.tanh = nn.Sequential(nn.Tanh())
        
    def forward(self,z,u):
        #z is image shape 64x3x32x32 u is message shape 64x256
        #convert u to be 64x16x4x4
        u = u.view(self.args.batch_size,16,4,4)
        # upsample and convolve to 64x1x32x32
        u = F.sigmoid(self.conv1(u))
        # add it to image
        z = torch.cat([z,u],dim=1)
        #convolve the image to be final 64x3x32x32
        z = self.conv2(z)
        z = self.tanh(z)
        return z

if __name__ == '__main__':
    print('Generators initialized')
