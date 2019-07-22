
import torch.nn as nn
import torch

###########################################
# Channel Coding Encoders (CCE)
# CCE maps u to a tensor
###########################################
from interleavers import Interleaver2D, DeInterleaver2D, Interleaver, DeInterleaver
from cnn_utils import SameShapeConv2d, SameShapeConv1d
from sagan_utils import Self_Attn



# STE implementation
class MyQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, args):

        ctx.save_for_backward(inputs)
        ctx.args = args

        outputs_int = torch.sign(inputs)

        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):

        # STE implementations
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        return grad_input, None

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

        self.norm            = torch.nn.BatchNorm1d(self.args.code_rate_n,  affine=False)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def forward(self, u_message, real_cpu):
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

        if self.args.use_cce_ste:
            myquantize = MyQuantize.apply
            u_tensor = myquantize(u_tensor, self.args)

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

        self.norm            = torch.nn.BatchNorm2d(self.args.code_rate_n,  affine=False)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def forward(self, u_message, real_cpu):
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

        if self.args.use_cce_ste:
            myquantize = MyQuantize.apply
            u_tensor = myquantize(u_tensor, self.args)

        return u_tensor


class CCE_Turbo_Encoder2D_img(nn.Module):
    def __init__(self, args, p_array):
        super(CCE_Turbo_Encoder2D_img, self).__init__()
        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        # Define 1D Network for TurboAE
        self.enc_cnn_1       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k + args.img_channels,
                                                  out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)


        self.enc_linear_1    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= 1, kernel_size = 1)

        self.enc_cnn_2       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k+ args.img_channels,
                                                  out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_2    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= 1, kernel_size = 1)

        self.enc_cnn_3       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k+ args.img_channels,
                                                  out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)

        self.enc_linear_3    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= 1, kernel_size = 1)

        self.interleaver     = Interleaver2D(args, p_array)

        self.norm            = torch.nn.BatchNorm2d(self.args.code_rate_n,  affine=False)

    def set_interleaver(self, p_array):
        self.interleaver.set_parray(p_array)

    def forward(self, u_message, real_cpu):
        inputs     = 2.0*u_message - 1.0
        inputs_img = torch.cat([inputs,real_cpu ], dim = 1)

        x_sys      = self.enc_cnn_1(inputs_img)
        x_sys      = self.enc_linear_1(x_sys)

        x_p1       = self.enc_cnn_2(inputs_img)
        x_p1       = self.enc_linear_2(x_p1)

        x_sys_int      = self.interleaver(inputs)
        inputs_int_img = torch.cat([x_sys_int,real_cpu ], dim = 1)

        x_p2       = self.enc_cnn_3(inputs_int_img)
        x_p2       = self.enc_linear_3(x_p2)

        x_tx       = torch.cat([x_sys,x_p1, x_p2], dim = 1)

        u_tensor   = self.norm(x_tx)

        if self.args.use_cce_ste:
            myquantize = MyQuantize.apply
            u_tensor = myquantize(u_tensor, self.args)

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
        self.norm          = torch.nn.BatchNorm2d(self.args.code_rate_n,  affine=False)

    def forward(self, u_message, real_cpu):
        inputs     = 2.0*u_message - 1.0

        x_sys      = self.enc_cnn(inputs)
        x_sys      = self.enc_linear(x_sys)
        u_tensor   = self.norm(x_sys)

        if self.args.use_cce_ste:
            myquantize = MyQuantize.apply
            u_tensor = myquantize(u_tensor, self.args)

        return u_tensor



# 2D CNN Encoder
class CCE_CNN_Encoder2D_img(nn.Module):
    def __init__(self, args, p_array):
        super(CCE_CNN_Encoder2D_img, self).__init__()
        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")

        # Define 1D Network for TurboAE
        self.enc_cnn       = SameShapeConv2d(num_layer=args.cce_num_layer, in_channels=args.code_rate_k + args.img_channels,
                                             out_channels= args.cce_num_unit, kernel_size = args.enc_kernel_size)


        self.enc_linear    = SameShapeConv2d(num_layer=1, in_channels=args.cce_num_unit,
                                                  out_channels= args.code_rate_n, kernel_size = 1)
        self.norm          = torch.nn.BatchNorm2d(self.args.code_rate_n,  affine=False)

    def forward(self, u_message, real_cpu):
        inputs     = 2.0*u_message - 1.0

        img_inputs = torch.cat([inputs, real_cpu], dim=1)

        x_sys      = self.enc_cnn(img_inputs)
        x_sys      = self.enc_linear(x_sys)
        u_tensor   = self.norm(x_sys)

        if self.args.use_cce_ste:
            myquantize = MyQuantize.apply
            u_tensor = myquantize(u_tensor, self.args)

        return u_tensor


# Dummy Encoder
class CCE_Repeat(nn.Module):
    def __init__(self, args, p_array):
        super(CCE_Repeat, self).__init__()
        self.args = args

        cuda = True if torch.cuda.is_available() else False
        self.this_device = torch.device("cuda" if cuda else "cpu")
        self.dummy = nn.Linear(1,1)

        self.norm          = torch.nn.BatchNorm2d(int(self.args.code_rate_n/self.args.code_rate_k),  affine=False)

    def forward(self, u_message, real_cpu):
        inputs   = u_message
        u_tensor = torch.cat([inputs for _ in range(int(self.args.code_rate_n/self.args.code_rate_k))], dim=1)
        u_tensor = self.norm(u_tensor)
        return u_tensor






