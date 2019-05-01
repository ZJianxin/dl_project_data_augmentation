import torch
import numpy as np
from torch import Tensor
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def to_var(x, use_cuda):
    x = Variable(x)
    if use_cuda:
        x = x.cuda()
    return x

def one_hot(labels, class_size, use_cuda):
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return to_var(targets, use_cuda)

def EMNIST_process(raw):
    img = np.flip(raw.reshape(28, 28), 1)
    return np.rot90(img, 1)

def get_dataset(name, subset=None):
    if name == 'EMNIST':
        dataset = datasets.EMNIST('./data/EMNIST', train=True, download=True, split='byclass', transform=transforms.ToTensor())
    elif name == 'MNIST':
        dataset = datasets.EMNIST('./data/MNIST', train=True, download=True, transform=transforms.ToTensor())
    elif name == 'CIFAR10':
        dataset = datasets.CIFAR10('./data/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
    elif name == 'SVHN':
        dataset = datasets.SVHN('./data/SVHN', split='train', download=True, transform=transforms.ToTensor())
    else:
        raise Exception

    if (subset is None) or (subset >= len(dataset)):
        return dataset
    else:
        split = (subset, len(dataset) - subset)
        subset, _ = torch.utils.data.random_split(dataset, split)
        return subset

def standard_sample(generation_model, num_classes, latent_size, use_cuda, save_path, image_size=(1, 28, 28), transformation=None):
    # Generate images with condition labels
    channel, w, h = image_size
    c = torch.eye(num_classes, num_classes) # [one hot labels for 0-9]
    c = to_var(c, use_cuda)
    z = to_var(torch.randn(num_classes, latent_size), use_cuda)
    samples = generation_model(z, c).data.cpu().numpy()

    fig = plt.figure(figsize=(num_classes, 1))
    gs = gridspec.GridSpec(1, num_classes)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if not (transformation is None):
            sample = transformation(sample)
        if channel == 1:
            plt.imshow(sample.reshape(w, h), cmap='Greys_r')
        else:
            plt.imshow(np.transpose(sample.reshape((channel, w, h)), (1, 2, 0)))
    plt.savefig(save_path)
    plt.close()