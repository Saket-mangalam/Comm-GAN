import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="training of CommGAN")
    parser.add_argument("-ngpu", type=int, default=1, help="gpu number to use, \
                    multiple ids should be e.g. 0,1,2,3)")
    parser.add_argument("-num_workers", type=int, default=2, help="number of parallel workers")

    #GAN structure related arguments
    parser.add_argument("-gtype", choices=['dcgan','wgan'], default='dcgan', help="generator type, wgan not there")

    parser.add_argument("-d1type", choices=['dcgan','wgan', 'sngan', 'sagan'], default='sagan', help="D1 type")
    parser.add_argument("-d2type", choices=['dcgan', 'wgan', 'sngan', 'sagan'], default='sagan', help="D2 type")

    parser.add_argument("-etype", choices=['basic', 'res', 'dense', 'dres', 'turbo_dres', 'sadense'], default='sadense', help="encoder type")
    parser.add_argument("-dectype", choices=['basic', 'dense', 'turbo_dense'], default='dense', help="decoder type")

    # Dataset (Image) setup
    parser.add_argument("-message", type=str, default="Saket Mangalam", help="Message to be encoded")
    parser.add_argument("-img_channels", type=int, default=1, help="length for height/width of square patch")
    parser.add_argument("-img_size", type=int, default=64, help="image size for training")
    parser.add_argument("-info_img_size", type=int, default=64, help="image size for training")

    parser.add_argument('-block_len', type=int, default=100)

    parser.add_argument("-data", choices=['test3', 'test', 'test2', 'celeba','mnist','cifar10','lsun','coco'], default='mnist', help="Lsun/Coco not supported yet.")
    parser.add_argument("-test_folder", choices=['test','test2','test3','celeba'], default='test3', help="folder name of testset")

    # Training Hyperparameter setup
    parser.add_argument("-batch_size", type=int, default=4, help="mini-batch size")
    parser.add_argument("-learning_rate", type=float, default=0.0002, help="learning rate, use value from origin paper as default")
    parser.add_argument("-codec_lr", type=float, default=0.0002, help="learning rate, use value from origin paper as default")
    parser.add_argument("-beta1", type=float, default=0.5, help="momentum")
    parser.add_argument("-beta2", type=float, default=0.99, help="momentum")

    parser.add_argument("-num_epoch", type=int, default=20, help="end epoch for training(exclusive)")

    # number of train steps
    parser.add_argument('-num_train_G', type=int, default=1, help = 'Generator for G')
    parser.add_argument('-num_train_D1', type=int, default=1, help = 'Discriminator for G')
    parser.add_argument('-num_train_D2', type=int, default=1, help = 'Discriminator for Encoder')
    parser.add_argument('-num_train_Enc', type=int, default=2, help = 'Encoder')
    parser.add_argument('-num_train_Dec', type=int, default=5, help = 'Decoder')

    # design the complexity of encoder and decoder
    parser.add_argument('-num_layer_Enc', type=int, default=4, help = 'Encoder CNN layers, Dense Encoder only')
    parser.add_argument('-num_layer_Dec', type=int, default=4, help = 'Decoder CNN layers')

    # training objective ratio
    parser.add_argument("-enc_lambda_D", type=float, default = 0.01, help = "relative weight of discriminator (D2)")
    parser.add_argument("-enc_lambda_Dec", type=float, default=0.99, help="relative weight of decoder")
    parser.add_argument("-enc_mse_wt", type=float, default=0.0, help="relative weight of mse loss")

    parser.add_argument("-G_lambda_D1", type=float, default = 0.9, help = "relative weight of discriminator")
    parser.add_argument("-G_lambda_D2", type=float, default= 0.09, help="relative weight of decoder")
    parser.add_argument("-G_lambda_Dec", type=float, default=0.01, help="relative weight of mse loss")

    parser.add_argument("-bitsperpix", type=int, default=3, help="number of bits to encode per pixel")

    parser.add_argument("-zlatent", type=int, default=100, help="number of latent space in z used for generating images")
    parser.add_argument("-gen_lat", type=int, default=64, help="hidden feature numbers of generator")
    parser.add_argument("-enc_lat", type=int, default=64, help="hidden feature numbers of encoder")
    parser.add_argument("-dec_lat", type=int, default=64, help="hidden feature numbers of decoder")
    parser.add_argument("-disc_lat", type=int, default=64, help="hidden feature numbers of discriminator")

    parser.add_argument("-use_data", type=bool, default=False, help="use data or generated image for encoding")


    parser.add_argument("-print_freq", type=int, default=50, help="summary info(for tensorboard) writing frequency(of batches)")
    parser.add_argument("-save_freq", type=int, default=500, help="checkpoint saving freqency(of epoches)")
    parser.add_argument("-val_freq", type=int, default=50, help="model validation frequency(of epoches)")
    parser.add_argument("-model_id", type=str, default='default', help="model id of saved pretrained model weights")

    # Channel Arguments
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('-channel_bsc_p', type=float, default=0.00, help="=")
    parser.add_argument('-channel_bec_p', type=float, default=0.0, help="=")
    parser.add_argument('-awgn', type=float, default=0.0, help="put noise std. 0.0 means there is no AWGN noise")
    parser.add_argument('-sp', type=float, default=0.0, help="put salt&pepper prob. 0.0 means there is no s-p noise")
    parser.add_argument('-spmf', type=float, default=0.0, help="put salt&pepper&median filter prob. 0.0 means there is no s-p-mf noise, not done yet..")
    parser.add_argument('-downsampling', type=int, default=1, help="downsampling ratio, downsampling and upsampling. 1 means there is no downsampling")


    # Channel Coding encoder and decoder arugments
    parser.add_argument('-enc_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-dec_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-enc_num_layer', type=int, default=2)
    parser.add_argument('-dec_num_layer', type=int, default=2)
    parser.add_argument('-enc_num_unit', type=int, default=100)
    parser.add_argument('-dec_num_unit', type=int, default=100)
    parser.add_argument('-num_iteration', type=int, default=2)
    parser.add_argument('-num_iter_ft', type=int, default=5)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-is_parallel', type=int, default=0)



    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')
    parser.add_argument('-dec_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')

    parser.add_argument('-code_rate', type=int, default=3)


    parser.add_argument('-bsc_p', type=float, default=0.1)

    # Quantization related value
    parser.add_argument('-quantize', type=int, default=1, help="downsampling levels. 1 means there is no quantization")
    parser.add_argument('-enc_value_limit', type=float, default=1.0, help = 'only valid for group_norm quantization')
    parser.add_argument('-enc_grad_limit', type=float, default=0.01, help = 'only valid for group_norm quantization')
    parser.add_argument('-enc_clipping', choices=['inputs', 'gradient', 'both', 'default'], default='default', help = 'only valid for group_norm quantization')


    opt = parser.parse_args()
    print(opt)

    return opt