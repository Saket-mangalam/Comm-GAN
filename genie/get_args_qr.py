import argparse

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description="training of CommGAN")

    parser.add_argument('-num_workers', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-num_test_epochs', type=int, default=100)


    #Neural structure related arguments
    # Encoders
    parser.add_argument("-cce_type", choices=['turboae1d','turboae2d','turboae2d_img', 'cnn2d','cnn2d_img','repeat'], default='repeat', help="Channel Coding Encoder with TurboAE-2D, 1D..")
    parser.add_argument("-ie_type", choices=['sadense','dense'], default='dense', help="Image Encoder")
    parser.add_argument("-cdec_type", choices=['turboae1d','turboae2d','turboae2d_img', 'cnn2d'], default='cnn2d', help="Channel Decoder")
    parser.add_argument("-idec_type", choices=['sadense','dense'], default='dense', help="Image Decoder")

    parser.add_argument('--norm_input_ie', action='store_true', default=False,
                        help='IE input is N(0,1) (True) or Binary (-1,+1) (False, Default)')
    parser.add_argument('--use_cce', action='store_true', default=False,
                        help='print positional BER/MSE')
    parser.add_argument('--use_cce_ste', action='store_true', default=False,
                        help='Only useful for print positional BER/MSE')

    parser.add_argument('--same_img', action='store_true', default=False,
                        help='Only useful for print positional BER/MSE')
    parser.add_argument('--print_pos_ber', action='store_true', default=False,
                        help='Channel Coding on Encoder Side')

    parser.add_argument('--save_models', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--IE_weight_adapt', action='store_true', default=False,
                        help='disables CUDA training')


    # design the complexity of encoder and decoder
    parser.add_argument('-cce_num_layer', type=int, default=2, help = 'Encoder CNN layers')
    parser.add_argument('-ie_num_layer', type=int, default=4, help = 'Encoder CNN layers')
    parser.add_argument('-idec_num_layer', type=int, default=4, help = 'Decoder CNN layers')
    parser.add_argument('-cdec_num_layer', type=int, default=2, help = 'Decoder CNN layers')

    parser.add_argument('-cce_num_unit', type=int, default=50)
    parser.add_argument('-ie_num_unit', type=int, default=50)
    parser.add_argument('-cdec_num_unit', type=int, default=50)
    parser.add_argument('-idec_num_unit', type=int, default=50)

    parser.add_argument('-code_rate_k', type=int, default=1)
    parser.add_argument('-code_rate_n', type=int, default=1)

    parser.add_argument("-enc_noise", type=float, default=0.01, help="momentum")
    parser.add_argument("-dec_noise", type=float, default=0.1, help="momentum")

    parser.add_argument("-message", type=str, default="Saket Mangalam", help="Message to be encoded")
    parser.add_argument("-img_channels", type=int, default=1, help="length for height/width of square patch")
    parser.add_argument("-img_size", type=int, default=32, help="image size for training")

    parser.add_argument("-data", choices=['celeba','mnist','cifar10','lsun','coco', 'test'], default='mnist', help="Lsun not supported yet.")
    parser.add_argument("-test_folder", choices=['test','test2','test3','celeba'], default='test3', help="folder name of testset")

    # Training Hyperparameter setup
    parser.add_argument("-batch_size", type=int, default=32, help="mini-batch size")
    parser.add_argument("-learning_rate", type=float, default=0.0001, help="learning rate, use value from origin paper as default")
    parser.add_argument("-codec_lr", type=float, default=0.0001, help="learning rate, use value from origin paper as default")
    parser.add_argument("-beta1", type=float, default=0.5, help="momentum")
    parser.add_argument("-beta2", type=float, default=0.99, help="momentum")

    parser.add_argument("-num_epoch", type=int, default=0, help="end epoch for training(exclusive)")

    # D2 noise parameters
    parser.add_argument("-d2_awgn", type=float, default=0.1, help="awgn noise added to real image added to D")
    parser.add_argument("-d2_bsc", type=float, default=0.0, help="awgn noise added to real image added to D")

    # number of train steps
    parser.add_argument('-num_train_D2', type=int, default=1, help = 'Discriminator for Encoder')
    parser.add_argument('-num_train_IE', type=int, default=5, help = 'Encoder, Image Encoder')
    parser.add_argument('-num_train_CCE', type=int, default=2, help = 'Encoder, Channel Coding Encoder')
    parser.add_argument('-num_train_CDec', type=int, default=5, help = 'Decoder')
    parser.add_argument('-num_train_IDec', type=int, default=5, help = 'Decoder')

    # weight ratio for adaptive encoder training
    parser.add_argument("-fid_thd_low", type=float, default = 120.0, help = "relative weight of discriminator (D2)")
    parser.add_argument("-fid_thd_high", type=float, default = 125.0, help = "relative weight of discriminator (D2)")

    # weight ratio for adaptive encoder training
    parser.add_argument("-ber_thd_low", type=float, default=0.05, help="relative weight of discriminator (D2)")
    parser.add_argument("-ber_thd_high", type=float, default=0.2, help="relative weight of discriminator (D2)")

    # training objective ratio
    parser.add_argument("-enc_lambda_D", type=float, default = 0.2, help = "relative weight of discriminator (D2)")
    parser.add_argument("-enc_lambda_Dec", type=float, default=0.8, help="relative weight of decoder")


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

    parser.add_argument('-channel_bsc_p', type=float, default=0.00, help="=")
    parser.add_argument('-channel_bec_p', type=float, default=0.0, help="=")
    parser.add_argument('-awgn', type=bool, default=False, help="put noise std. 0.0 means there is no AWGN noise")
    parser.add_argument('-noise_std', type=float, default=0.0, help="standard deviation of noise being added as AWGN")
    parser.add_argument('-sp', type=bool, default=False, help="put salt&pepper prob. 0.0 means there is no s-p noise")
    parser.add_argument('-sp_size', type=float, default=0.5, help="block size of salt and pepper channel")
    parser.add_argument('-slide', type=bool, default=False, help="put sliding window or not")
    parser.add_argument('-sl_size', type=int, default= 10, help="thickness of sliding window")
    parser.add_argument('-sl_dim', type=str, default = 'height', help ="slide channel along height or width")
    parser.add_argument('-spmf', type=float, default=0.0, help="put salt&pepper&median filter prob. 0.0 means there is no s-p-mf noise, not done yet..")
    parser.add_argument('-downsampling', type=int, default=1, help="downsampling ratio, downsampling and upsampling. 1 means there is no downsampling")




    parser.add_argument('-num_iteration', type=int, default=2)
    parser.add_argument('-num_iter_ft', type=int, default=5)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-is_parallel', type=int, default=0)

    parser.add_argument('-enc_kernel_size', type=int, default=5)
    parser.add_argument('-dec_kernel_size', type=int, default=5)
    parser.add_argument('-extrinsic', type=int, default=1)



    # parser.add_argument('-enc_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')
    # parser.add_argument('-dec_act', choices=['tanh', 'selu', 'relu', 'elu', 'sigmoid', 'linear'], default='linear')
    #



    parser.add_argument('-bsc_p', type=float, default=0.1)

    # Quantization related value
    parser.add_argument('-quantize', type=int, default=1, help="downsampling levels. 1 means there is no quantization")
    parser.add_argument('-enc_value_limit', type=float, default=1.0, help = 'only valid for group_norm quantization')
    parser.add_argument('-enc_grad_limit', type=float, default=0.01, help = 'only valid for group_norm quantization')
    parser.add_argument('-enc_clipping', choices=['inputs', 'gradient', 'both', 'default'], default='default', help = 'only valid for group_norm quantization')


    opt = parser.parse_args()
    print(opt)

    return opt