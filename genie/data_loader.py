__author__ = 'saketM'
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms


def get_data(args):
    ######################################################################
    # Data
    ######################################################################
    if args.img_channels == 1: # grayscale
        if args.data == 'test':
            dataset = dset.ImageFolder(root='./data/test',
                                       transform=transforms.Compose([
                                           transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize(args.img_size),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

            # Create the dataloader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                     shuffle=True, num_workers=args.num_workers)
        elif args.data == 'celeba':
            dataset = dset.ImageFolder(root='./data/celeba',
                                       transform=transforms.Compose([
                                           transforms.Grayscale(num_output_channels=1),
                                           transforms.Resize(args.img_size),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

            # Create the dataloader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                     shuffle=True, num_workers=args.num_workers)
        elif args.data == 'mnist':
            dataloader = torch.utils.data.DataLoader(
                dset.MNIST('./data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Grayscale(num_output_channels=1),
                               transforms.Resize(args.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        elif args.data == 'cifar10':
            dataloader = torch.utils.data.DataLoader(
                dset.CIFAR10('./data/cifar10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.Resize(args.img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        else:
            print('coco and lsun not supported yet!')
    else: # 3 channels colorful picture
        if args.data == 'test':
            dataset = dset.ImageFolder(root='./data/test',
                                       transform=transforms.Compose([
                                           transforms.Resize(args.img_size),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

            # Create the dataloader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                     shuffle=True, num_workers=args.num_workers)
        elif args.data == 'celeba':
            dataset = dset.ImageFolder(root='./data/celeba',
                                       transform=transforms.Compose([
                                           transforms.Resize(args.img_size),
                                           transforms.CenterCrop(args.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))

            # Create the dataloader
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                     shuffle=True, num_workers=args.num_workers)
        elif args.data == 'mnist':
            dataloader = torch.utils.data.DataLoader(
                dset.MNIST('./data/mnist', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(args.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ])),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        elif args.data == 'cifar10':
            dataloader = torch.utils.data.DataLoader(
                dset.CIFAR10('./data/cifar10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(args.img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])),
                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        else:
            print('coco and lsun not supported yet!')

    return dataloader