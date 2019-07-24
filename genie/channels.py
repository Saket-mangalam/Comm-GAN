import torch.nn as nn
import torch
import numpy as np
import torch.functional as F
#from quantizer import add_qr

# STE implementation
class MyQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, args, quantize_level = 0):
        ctx.args = args

        if quantize_level == 0:
            if args.quantize == 2:
                inputs = artanh(inputs)      # inverse tanh
                ctx.save_for_backward(inputs)
                x_input_norm =  torch.clamp(inputs, -1.0, 1.0)
                outputs_int = torch.sign(x_input_norm)
            else:
                # quantize>2 not tested yet!
                ctx.save_for_backward(inputs)
                x_input_norm =  torch.clamp(inputs, -1.0, 1.0)
                x_lim_range = 2.0
                outputs_int  = torch.round((x_input_norm +1.0) * ((args.quantize - 1.0)/x_lim_range)) * x_lim_range/(args.quantize - 1.0) - 1.0
        elif quantize_level >1:
            ctx.save_for_backward(inputs)
            x_input_norm =  torch.clamp(inputs, -1.0, 1.0)
            x_lim_range = 2.0
            outputs_int  = torch.round((x_input_norm +1.0) * ((quantize_level - 1.0)/x_lim_range)) * x_lim_range/(quantize_level - 1.0) - 1.0

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




def quaitize(inputs, quantize_level):

    x_input_norm =  torch.clamp(inputs, -1.0, 1.0)
    x_lim_range = 2.0
    outputs_int  = torch.round((x_input_norm +1.0) * ((quantize_level - 1.0)/x_lim_range)) * x_lim_range/(quantize_level - 1.0) - 1.0

    return outputs_int




def artanh(y):
    x = 0.5 * torch.log((1+y)/(1-y))
    return x


def salt_and_pepper(img, prob):
    """salt and pepper noise for mnist"""
    rnd = torch.rand(img.shape)

    noisy = img.clone()
    noisy[rnd < prob/2] = -1.
    #noisy[rnd > 1 - prob/2] = 1.

    return noisy


def channel(encoded_imgs, args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.awgn!=0.0:
        noise         = args.noise_std * torch.randn(encoded_imgs.shape, dtype=torch.float).to(device)
        encoded = encoded_imgs + noise
        encoded_imgs = torch.clamp(encoded, -1.0, +1.0)

    # if args.sp!=0.0:
    #     tmp = encoded_imgs
    #     encoded = salt_and_pepper(tmp, args.sp)
    #     encoded_imgs = encoded
    #
    # if args.spmf!=0.0:
    #     tmp = encoded_imgs
    #     encoded_imgs = salt_and_pepper(tmp, args.sp)
    #     print('Salt-Pepper and Median Filter not implemented!!!!!!! Yihan Jiang is super lazy!!!!')
    #
    # if args.downsampling != 1:
    #     DownPool = nn.AvgPool2d(2, stride=2)
    #     UpPool = nn.Upsample(scale_factor=2, mode='nearest')
    #
    #     output = DownPool(encoded_imgs)
    #     encoded = UpPool(output)
    #     encoded_imgs = encoded
    #
    # if args.quantize !=1:
    #     # quantize >2 not tested!!
    #     myquantize = MyQuantize.apply
    #     encoded    = myquantize(encoded_imgs, args)
    #     encoded_imgs = encoded
    #
    # if args.channel_bsc_p!=0.0:
    #     myquantize = MyQuantize.apply
    #     encoded    = myquantize(encoded_imgs, args, quantize_level = 2)
    #     encoded_imgs = encoded
    #
    #     channel_bsc_p = args.channel_bsc_p
    #     fwd_noise = torch.from_numpy(np.random.choice([-1.0, 1.0], encoded_imgs.shape,
    #                                     p=[channel_bsc_p, 1.0 - channel_bsc_p])).type(torch.FloatTensor).to(device)
    #     encoded_imgs = encoded_imgs*fwd_noise
        #encoded_imgs = add_qr(encoded_imgs)

    return encoded_imgs

def slide_channel(encoded_imgs, sl_size, sl_dim):
    #mask = torch.zeros(encoded_imgs.size).to(device)
    batch_size, s, l1, l2 = encoded_imgs.shape[0], encoded_imgs.shape[1], encoded_imgs.shape[2], encoded_imgs.shape[3]
    noisy = encoded_imgs.clone()
    if sl_dim == 'height':
        random_start = np.random.randint(1, (l1-sl_size))
        noisy[:,:,random_start:(random_start+sl_size),:]  = -1
    else:
        random_start = np.random.randint(1, (l2-sl_size))
        noisy[:,:,:,random_start:(random_start+sl_size)]  = -1

    return noisy



def channel_test(encoded_imgs, args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.awgn:
        noise         = args.noise_std * torch.randn(encoded_imgs.shape, dtype=torch.float).to(device)
        encoded = encoded_imgs + noise
        encoded_imgs = torch.clamp(encoded, -1.0, +1.0)
        #encoded_imgs = add_qr(encoded_imgs)
    elif args.sp:
        tmp = encoded_imgs
        encoded_imgs = salt_and_pepper(tmp, args.sp_size)
    elif args.slide:
        encoded_imgs = slide_channel(encoded_imgs, args.sl_size, args.sl_dim)
    else:
        print('unknown test channel!')

    return encoded_imgs