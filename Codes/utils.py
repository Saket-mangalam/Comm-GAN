
import torch
import numpy as np
from torchvision.models.inception import inception_v3
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import entropy

def channel(encoded_imgs, noise_std, channel_type = 'awgn', device = torch.device("cpu") ):
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


def errors_ber(y_true, y_pred):
    y_true = y_true.view(y_true.shape[0], -1, 1)
    y_pred = y_pred.view(y_pred.shape[0], -1, 1)

    myOtherTensor = np.not_equal(np.round(y_true), np.round(y_pred)).float()
    k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
    return k


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0.0)


def inception_score(imgs, cuda=False, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)