__author__ = 'SaketM'
import torch.nn as nn
import torch

###########################################
# Channel Coding Encoders (CCE)
# CCE maps u to a tensor
###########################################
from interleavers import Interleaver2D, DeInterleaver2D, Interleaver, DeInterleaver
from cnn_utils import SameShapeConv2d, SameShapeConv1d
from sagan_utils import Self_Attn



# 1D TurboAE Encoder
# Input shape: u (1, M , M) -> (R, M, M)
class CCE_Turbo_Encoder1D(nn.Module):
    def __init__(self, args, p_array):
        super(CCE_Turbo_Encoder1D, self).__init__()
        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        # Define 1D Network for TurboAE
        self.enc_cnn_1       = SameShapeConv1d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.cce_num_unit, kernel_size = args.dec_kernel_size)


        self.enc_linear_1    = torch.nn.Linear(args.cce_num_unit, 1)

        self.enc_cnn_2       = SameShapeConv1d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.cce_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_2    = torch.nn.Linear(args.cce_num_unit, 1)

        self.enc_cnn_3       = SameShapeConv1d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.cce_num_unit, kernel_size = args.dec_kernel_size)

        self.enc_linear_3    = torch.nn.Linear(args.cce_num_unit, 1)

        self.interleaver     = Interleaver(args, p_array)

        self.norm            = torch.nn.BatchNorm1d(self.args.code_rate_n,  affine=True)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def forward(self, u_message):
        inputs     = 2.0*u_message - 1.0
        inputs     = inputs.view(self.args.batch_size, self.args.code_rate_k, self.args.img_size**2).permute(0,2,1)

        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_linear_1(x_sys)

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_linear_2(x_p1)

        x_sys_int  = self.interleaver(inputs)

        x_p2       = self.enc_cnn_3(x_sys_int)
        x_p2       = self.enc_linear_3(x_p2)

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 2)
        x_tx       = x_tx.permute(0,2,1)
        x_tx       = self.norm(x_tx)

        u_tensor = x_tx.view(self.args.batch_size,  self.args.code_rate_n,self.args.img_size,self.args.img_size)

        return u_tensor

# 2D TurboAE Encoder
# Input shape: u (1, M , M) -> (R, M, M)
class CCE_Turbo_Encoder2D(nn.Module):
    def __init__(self, args, p_array):
        super(CCE_Turbo_Encoder2D, self).__init__()
        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        # Define 1D Network for TurboAE
        self.enc_cnn_1       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)


        self.enc_linear_1    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= 1, kernel_size = 1)

        self.enc_cnn_2       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_2    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= 1, kernel_size = 1)

        self.enc_cnn_3       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k,
                                                  out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_3    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= 1, kernel_size = 1)

        self.interleaver     = Interleaver2D(args, p_array)

        self.norm            = torch.nn.BatchNorm2d(self.args.code_rate_n,  affine=True)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def forward(self, u_message):
        inputs     = 2.0*u_message - 1.0

        x_sys      = self.enc_cnn_1(inputs)
        x_sys      = self.enc_linear_1(x_sys)

        x_p1       = self.enc_cnn_2(inputs)
        x_p1       = self.enc_linear_2(x_p1)

        x_sys_int  = self.interleaver(inputs)

        x_p2       = self.enc_cnn_3(x_sys_int)
        x_p2       = self.enc_linear_3(x_p2)

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 1)

        u_tensor   = self.norm(x_tx)

        return u_tensor


# 2D CNN Encoder
class CCE_CNN_Encoder2D(nn.Module):
    def __init__(self, args, p_array):
        super(CCE_CNN_Encoder2D, self).__init__()
        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        # Define 1D Network for TurboAE
        self.enc_cnn       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k,
                                             out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)


        self.enc_linear    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= args.code_rate_n, kernel_size = 1)
        self.norm          = torch.nn.BatchNorm2d(self.args.code_rate_n,  affine=True)

    def forward(self, u_message):
        inputs     = 2.0*u_message - 1.0

        x_sys      = self.enc_cnn(inputs)
        x_sys      = self.enc_linear(x_sys)
        u_tensor   = self.norm(x_sys)

        return u_tensor



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

    def forward(self,dense_input):
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

    def forward(self,dense_input):
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


###########################################
# Image Encoders (IE)
# IE maps u-tensor and image to encoded image
###########################################

class WholeEncoder(nn.Module):
    def __init__(self, args, p_array):
        super(WholeEncoder, self).__init__()
        self.args = args
        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        # DTA Channel coding encoder layer
        if self.args.cce_type == 'turboae1d':
            self.cce = CCE_Turbo_Encoder1D(args, p_array)
        elif self.args.cce_type == 'turboae2d': # 2D turbo encoder
            self.cce = CCE_Turbo_Encoder2D(args, p_array)
        elif self.args.cce_type == 'cnn2d':
            self.cce = CCE_CNN_Encoder2D(args, p_array)

        if self.args.ie_type == 'sadense':
            self.ie = IE_SADense(args)
        elif self.args.ie_type == 'dense':
            self.ie = IE_Dense(args)
        else:
            pass

    def forward(self,u_message, images):
        ###########################################
        # CCE
        ###########################################
        u_coded = self.cce(u_message)

        ###########################################
        # IE
        ###########################################


        dense_input = torch.cat([u_coded, images], dim=1)
        encoded_img = self.ie(dense_input)

        return encoded_img

