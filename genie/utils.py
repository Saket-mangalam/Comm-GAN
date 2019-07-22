__author__ = 'SaketM'
import torch.nn as nn
import torch
import numpy as np
import os
from collections import Counter
import math

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     nn.init.normal_(m.weight.data, 1.0, 0.02)
    #     nn.init.constant_(m.bias.data, 0)

#BER Loss of Decoder
def errors_ber(y_pred, y_true):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return k

#PSNR calculator
def psnr(img1, img2):
    mse = torch.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def pos_ber(y_pred, y_true):

    myOtherTensor = torch.ne(torch.round(y_true), torch.round(y_pred)).float()
    res = myOtherTensor.mean(dim=0)
    return res


def pos_mse(y_pred, y_true):
    MSE_pos_loss = (y_true - y_pred)**2
    res = MSE_pos_loss.mean(dim=0)
    return res


def channel(encoded_imgs, noise_std, channel_type = 'awgn', device = torch.device("cuda") ):

    if channel_type == 'awgn':
        noise        = noise_std * torch.randn(encoded_imgs.shape, dtype=torch.float).to(device)
        return encoded_imgs + noise

    elif channel_type == 'slides':
        batch_size, s, l1, l2 = encoded_imgs.shape[0], encoded_imgs.shape[1] , encoded_imgs.shape[2] ,encoded_imgs.shape[3]
        # only support MNIST for now
        random_line = int(np.ceil(noise_std * l1))
        random_start = np.random.randint(1, l1- random_line)
        if np.random.rand()>0.5:
            random_noise = torch.cat([torch.ones((batch_size, s, random_start, l2), dtype=torch.float),
                                      torch.zeros((batch_size, s, random_line, l2), dtype=torch.float),
                                      torch.ones((batch_size, s, l1 - random_start - random_line, l2), dtype=torch.float)],
                                     dim = 2).to(device)
        else:
            random_noise = torch.cat([torch.ones((batch_size, s, l1, random_start), dtype=torch.float),
                                      torch.zeros((batch_size, s, l1, random_line), dtype=torch.float),
                                      torch.ones((batch_size, s, l1, l2 - random_start - random_line), dtype=torch.float)],
                                     dim = 3).to(device)

        received_imgs= random_noise * encoded_imgs
        #save_image(received_imgs, 'images/tmp/tmp.png', nrow=10, normalize=True)
        return received_imgs
    elif channel_type == 'basic_quantize':
        # quantize to -1, -.5, 0.0, 0.5, 1.
        encoded_imgs = encoded_imgs.detach()
        q = 0.5
        y = q * np.round(encoded_imgs/q)
        return encoded_imgs

    else:
        noise        = noise_std * torch.randn(encoded_imgs.shape, dtype=torch.float).to(device)
        return encoded_imgs + noise



import zlib
from math import exp

import torch
from reedsolo import RSCodec
from torch.nn.functional import conv2d

rs = RSCodec(250)


def text_to_bits(text):
    """Convert text to a list of ints in {0, 1}"""
    return bytearray_to_bits(text_to_bytearray(text))


def bits_to_text(bits):
    """Convert a list of ints in {0, 1} to text"""
    return bytearray_to_text(bits_to_bytearray(bits))


def bytearray_to_bits(x):
    """Convert bytearray to a list of bits"""
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])

    return result


def bits_to_bytearray(bits):
    """Convert a list of bits to a bytearray"""
    ints = []
    #print(bits)
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        #print(int(''.join([str(int(bit)) for bit in byte]), 2))
        ints.append(int(''.join([str(int(bit)) for bit in byte]), 2))
    #print(bytearray(ints))
    return bytearray(ints)


def text_to_bytearray(text):
    """Compress and add error correction"""
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))

    return x


def bytearray_to_text(x):
    """Apply error correction and decompress"""
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        #text = text.decode("utf-8")
        return text.decode("utf-8")
    except BaseException:
        return False

def _make_payload(width, height, depth, text):
    """
    This takes a piece of text and encodes it into a bit vector. It then
    fills a matrix of size (width, height) with copies of the bit vector.
    """
    message = text_to_bits(text) + [0] * 32

    payload = message
    while len(payload) < width * height * depth:
        payload += message

    payload = payload[:width * height * depth]

    return torch.FloatTensor(payload).view(1, depth, height, width)

def encoded_message(batchsize, data_depth, img_size, text):
    """Encode an image.
    Args:
        cover (str): Path to the image to be used as cover.
        output (str): Path where the generated image will be saved.
        text (str): Message to hide inside the image.
    """

    payload = _make_payload(img_size, img_size, 1, text)
    #print(payload.size())
    payload = payload.expand(batchsize, data_depth, img_size, img_size)
    #replicate payload over batchsize
    #payload = payload.expand(batchsize, data_depth, img_size, img_size)
    #print(payload.size())
    #payload = payload.to(device)
    return payload

def decoded_message(bit_message):

    # split and decode messages
    #bit_message = bit_message.view(-1)
    candidates = Counter()
    bits = bit_message.data.cpu().numpy().tolist()
    #print(bits)
    #print(len(bits))
    for candidate in bits_to_bytearray(bits).split(b'\x00\x00\x00\x00'):
        candidate = bytearray_to_text(bytearray(candidate))
        #print(candidate)
        if candidate:
            candidates[candidate] += 1


    # choose most common message
    if len(candidates) == 0:
        #raise ValueError('Failed to find message.')
        candidate = 'No message found'
    else:
        #candidate, _ = candidates.most_common(1)[0]
        candidate, count = candidates.most_common(1)[0]

    return candidate
