import torch
import numpy as np

class MyQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs,args):
        ctx.args = args

        x_lim_abs  = args.enc_value_limit
        x_lim_range = 2.0 * x_lim_abs
        x_input_norm =  torch.clamp(inputs, -x_lim_abs, x_lim_abs)

        if args.quantize == 2:
            outputs_int = torch.sign(x_input_norm)
        else:
            outputs_int  = torch.round((x_input_norm +x_lim_abs) * ((args.quantize - 1.0)/x_lim_range)) * x_lim_range/(args.quantize - 1.0) - x_lim_abs


        return outputs_int

    @staticmethod
    def backward(ctx, grad_output):

        # STE implementations
        if ctx.args.enc_clipping in ['inputs', 'both']:
            input, = ctx.saved_tensors
            grad_output[input>1.0]=0
            grad_output[input<-1.0]=0

        if ctx.args.enc_clipping in ['gradient', 'both']:
            grad_output = torch.clamp(grad_output, -ctx.args.enc_grad_limit, ctx.args.enc_grad_limit)

        #print torch.min(grad_output), torch.max(grad_output)

        grad_input = grad_output.clone()

        return grad_input, None

def quantizer(imgs, args):

    myquantize = MyQuantize.apply
    encoded    = myquantize(imgs, args)
    encoded_imgs = encoded

    return encoded_imgs

def add_qr(imgs):

    # add some things
    # add item for up left.
    new_imgs = imgs
    new_imgs[:, :, :7, :7] = -1.0
    new_imgs[:, :, 1, 1:6] = +1.0
    new_imgs[:, :, 5, 1:6] = +1.0
    new_imgs[:, :, 1:6, 1] = +1.0
    new_imgs[:, :, 1:6, 5] = +1.0
    new_imgs[:, :, 7, :8]  = +1.0
    new_imgs[:, :, :8, 7]  = +1.0

    # add item for left down.
    new_imgs[:, :, -7:, :7] = -1.0
    new_imgs[:, :, -2, 1:6] = +1.0
    new_imgs[:, :, -6, 1:6] = +1.0
    new_imgs[:, :,  -6:-2, 1] = +1.0
    new_imgs[:, :, -6:-2, 5] = +1.0
    new_imgs[:, :, -8, :8]  = +1.0
    new_imgs[:, :, -8:, 7]  = +1.0

    # add item for right up
    new_imgs[:, :, :7, -7:] = -1.0
    new_imgs[:, :, 1, -6:-2] = +1.0
    new_imgs[:, :, 5, -6:-2] = +1.0
    new_imgs[:, :, 1:6, -2] = +1.0
    new_imgs[:, :, 1:6, -6] = +1.0
    new_imgs[:, :, 7, -7:]  = +1.0
    new_imgs[:, :, :8, -8]  = +1.0

    return new_imgs



def bsc(imgs, bsc_p, device):
    fwd_noise = torch.from_numpy(np.random.choice([-1.0, 1.0], imgs.shape,
                                        p=[bsc_p, 1 - bsc_p])).type(torch.FloatTensor).to(device)
    imgs = imgs*fwd_noise

    return imgs




