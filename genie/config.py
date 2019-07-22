__author__ = 'saketM'
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

class config(object):
    def __init__(self, args):
        self.args = args
    # Configure Channel Coding Encoder
    def get_cce(self):

        if self.args.cce_type == 'turboae1d':
            from cce import CCE_Turbo_Encoder1D as CCE

        elif self.args.cce_type == 'turboae2d': # 2D turbo encoder
            from cce import CCE_Turbo_Encoder1D as CCE

        elif self.args.cce_type == 'cnn2d':
            from cce import CCE_CNN_Encoder2D as CCE
        elif self.args.cce_type == 'repeat':
            from cce import CCE_Repeat as CCE

        return CCE

    # Configure Image Encoder
    def get_ie(self):

        if self.args.ie_type == 'sadense':
            from ie import IE_SADense as IE
        elif self.args.ie_type == 'dense':
            from ie import IE_Dense as IE

        return IE

    # Configure Image Decoder
    def get_idec(self):
        if self.args.idec_type == 'sadense':
            from idec import SADense_Decoder as IDec
        elif self.args.idec_type == 'dense':
            from idec import Dense_Decoder as IDec

        return IDec

    # Configure Channel Coding Decoder
    def get_cdec(self):

        if self.args.cdec_type == 'turboae2d':
            from cdec import TurboAE_decoder2D as CDec
        elif self.args.cdec_type == 'turboae1d':
            from cdec import TurboAE_decoder1D as CDec
        elif self.args.cdec_type == 'cnn2d':
            from cdec import CNN_decoder2D as CDec

        return CDec

    def get_D(self):
        #D2
        # if args.d2type == 'dcgan':
        #     from Discriminators import DCGANDiscriminator as EncDiscriminator
        # elif args.d2type == 'sngan':
        #     from Discriminators import SNGANDiscriminator as EncDiscriminator
        # elif args.d2type == 'sagan':
        #     from Discriminators import SAGANDiscriminator as EncDiscriminator
        # else:
        #     print('Discriminator not specified!')

        from discriminator import DCGANDiscriminator as EncDiscriminator

        return EncDiscriminator
