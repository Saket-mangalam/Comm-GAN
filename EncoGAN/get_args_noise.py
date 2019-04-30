import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="training of MC-CNN")
    parser.add_argument("-ngpu", type=int, default=1, help="gpu number to use, \
                    multiple ids should be e.g. 0,1,2,3)")
    parser.add_argument("-gtype", choices=['dcgan','wgan'], default='dcgan', help="generator type")
    parser.add_argument("-etype", choices=['basic', 'res', 'dense', 'dres', 'turbo_dres'], default='turbo_dres', help="encoder type")
    parser.add_argument("-dtype", choices=['dcgan', 'wgan'], default='dcgan', help="discriminator type")
    parser.add_argument("-dectype", choices=['basic', 'dense', 'turbo_dense'], default='turbo_dense', help="decoder type")
    parser.add_argument("-num_workers", type=int, default=2, help="number of parallel workers")
    parser.add_argument("-img_channels", type=int, default=1, help="length for height/width of square patch")
    parser.add_argument("-batch_size", type=int, default=4, help="mini-batch size")
    parser.add_argument("-img_size", type=int, default=64, help="image size for training")
    parser.add_argument("-learning_rate", type=float, default=0.0002, help="learning rate, \
                    use value from origin paper as default")
    parser.add_argument("-beta1", type=float, default=0.5, help="momentum")
    parser.add_argument("-beta2", type=float, default=0.99, help="momentum")
    parser.add_argument("-data", choices=['celeba','mnist','cifar','coco'], default='mnist', help="Dataset to use")
    parser.add_argument("-list_dir", type=str, default="./data/celeba", help="path to dir containing training & validation \
                    should be list_dir")
    parser.add_argument("-lambda_D", type=float, default = 0.5, help = "relative weight of discriminator")
    parser.add_argument("-lambda_Dec", type=float, default=0.5, help="relative weight of decoder")
    parser.add_argument("-mse_wt", type=float, default=1.0, help="relative weight of mse loss")
    parser.add_argument("-bitsperpix", type=int, default=3, help="number of bits to encode per pixel")
    parser.add_argument("-zlatent", type=int, default=100, help="number of latent space in z used for generating images")
    parser.add_argument("-use_data", type=bool, default=False, help="use data or genimage for encoding")
    parser.add_argument("-gen_lat", type=int, default=64, help="hidden feature numbers of generator")
    parser.add_argument("-enc_lat", type=int, default=64, help="hidden feature numbers of encoder")
    parser.add_argument("-dec_lat", type=int, default=64, help="hidden feature numbers of decoder")
    parser.add_argument("-disc_lat", type=int, default=64, help="hidden feature numbers of discriminator")
    parser.add_argument("-start_epoch", type=int, default=0, help="start epoch for training(inclusive)")
    parser.add_argument("-end_epoch", type=int, default=100, help="end epoch for training(exclusive)")
    parser.add_argument("-print_freq", type=int, default=50, help="summary info(for tensorboard) writing frequency(of batches)")
    parser.add_argument("-save_freq", type=int, default=500, help="checkpoint saving freqency(of epoches)")
    parser.add_argument("-val_freq", type=int, default=50, help="model validation frequency(of epoches)")
    parser.add_argument("-model_id", type=str, default='default', help="model id of saved pretrained model weights")

    # Channel Arguments
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('-awgn', type=float, default=0.0, help="put noise std. 0.0 means there is no AWGN noise")
    parser.add_argument('-sp', type=float, default=0.0, help="put salt&pepper prob. 0.0 means there is no s-p noise")
    parser.add_argument('-spmf', type=float, default=0.0, help="put salt&pepper&median filter prob. 0.0 means there is no s-p-mf noise, not done yet..")
    parser.add_argument('-downsampling', type=int, default=1, help="downsampling ratio, downsampling and upsampling. 1 means there is no downsampling")
    parser.add_argument('-quantize', type=int, default=1, help="downsampling levels. 1 means there is no quantization")
    parser.add_argument('-enc_clipping', choices=['inputs', 'gradient', 'both', 'default'], default='both', help = 'only valid for group_norm quantization')


    # Channel Coding encoder and decoder arugments
    parser.add_argument('-enc_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-dec_rnn', choices=['gru', 'lstm', 'rnn'], default='gru')
    parser.add_argument('-enc_num_layer', type=int, default=2)
    parser.add_argument('-dec_num_layer', type=int, default=2)
    parser.add_argument('-enc_num_unit', type=int, default=25)
    parser.add_argument('-dec_num_unit', type=int, default=50)
    parser.add_argument('-num_iteration', type=int, default=6)
    parser.add_argument('-num_iter_ft', type=int, default=5)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-is_parallel', type=int, default=0)

    parser.add_argument('-block_len', type=int, default=64) # TBD

    parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')
    parser.add_argument('-dec_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')

    parser.add_argument('-code_rate', type=int, default=3)


    opt = parser.parse_args()
    print(opt)

    return opt