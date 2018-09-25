
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-is_save_model', type=int, default=0, help ='0:not save weight. 1: save weight') # not functional

    parser.add_argument('-g_type', choices = ['dcgan','rnn_dcnn','gridrnn_dcnn'], default='dcgan', help='Choose G')
    parser.add_argument('-d_type', choices = ['dcgan'], default='dcgan', help='Choose D')
    parser.add_argument('-dec_type', choices = ['dcgan','rnn_dcnn','gridrnn_dcnn'], default='dcgan', help='Choose Dec')

    # not functional
    parser.add_argument('-dataset', choices = ['mnist','cifar10','yihan_selfie'], default='mnist', help='choose dataset for GAN training')


    parser.add_argument('-num_train_G', type=int, default=1)
    parser.add_argument('-num_train_D', type=int, default=1)
    parser.add_argument('-num_train_Dec', type=int, default=1)

    parser.add_argument('-num_epoch', type=int, default=25, help='number of epochs of training')
    parser.add_argument('-batch_size', type=int, default=64, help='size of the batches')

    parser.add_argument('-enc_num_unit', type=int, default=25)
    parser.add_argument('-dec_num_unit', type=int, default=100)
    parser.add_argument('-code_rate', type=int, default=2)

    parser.add_argument('-dec_weight', type=float, default=0.99, help = 'how much to address the decoder (0 to 1)')

    parser.add_argument('-gan_type', choices = ['dcgan', 'wgan', 'wgan_gp'], default='wgan_gp', help='gan structures.')
    parser.add_argument('-lambda_gp', type=int, default=10)

    parser.add_argument('-lr', type=float, default=0.0002, help='adam: 0.0002, rmsprop 0.00005')
    parser.add_argument('-b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('-b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

    parser.add_argument('-n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')

    # For Mnist, use (32,32) with 1 channel.
    parser.add_argument('-latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('-n_classes', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('-img_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('-img_channel', type=int, default=1, help='number of image channels')

    parser.add_argument('-sample_interval', type=int, default=400, help='interval between image sampling')

    parser.add_argument('-block_len', type=int, default=10, help='how much bit information to disguise')
    parser.add_argument('-channel_type', choices = ['slides', 'awgn', 'basic_quantize'], default='basic_quantize', help='channel')

    parser.add_argument('-sample_noise_image', type=int, default=1, help='sample data with noise pattern')

    parser.add_argument('-noise_std', type=float, default=0.1, help='noise as a regularizer')
    
    parser.add_argument('-Channels', type=int, default=3, help='number of channels in image file, 1 if grayscale, 3 if rgb')
    parser.add_argument('-Latent_channels', type=int, default=5, help='L no. of channels of message added for convolution as LxHxW')

    opt = parser.parse_args()
    print(opt)

    return opt
