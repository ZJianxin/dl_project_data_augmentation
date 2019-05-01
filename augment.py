import torch
from Dataset import Dataset
from utils import *
from models import *
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split

def cvae_agu_per_class(model, num_classes, label, size, latent_size, use_cuda=torch.cuda.is_available()):
    # augment data of one class by cvae
    #z = to_var(torch.randn(size, latent_size), use_cuda)
    z = to_var(torch.FloatTensor(size, latent_size).uniform_(-2048, 2048), use_cuda)
    c = torch.zeros(size, num_classes)
    c[:, label] = torch.ones(size)
    c = to_var(c, use_cuda)
    faked_X = model.generation_model(z, c).data
    return faked_X.cpu()

def cvae_aug(model, num_class, per_class, use_cuda=torch.cuda.is_available(), x_dim=(1, 28, 28)):
    latent_size = model.latent_size
    c, w, h = x_dim
    faked_X = torch.empty(0, c, w, h)
    y = torch.empty(0)
    for i in range(num_class):
        faked_X_in_class = cvae_agu_per_class(model, num_class, i, per_class, latent_size, use_cuda)
        faked_X = torch.cat((faked_X, faked_X_in_class.view(per_class, c, w, h)))
        labels = i * torch.ones(per_class)
        y = torch.cat((y, labels))
    return Dataset(faked_X, y)

def perform_augmentation(train_set, augmentation_function, augmentation_model, num_classes, augmentation_per_class,
                         use_cuda, **params):
    if (isinstance(augmentation_model, CVAE)):
        output_dim = params['output_dim'] if 'output_dim' in params.keys() else (1, 28, 28)
        subset = params['subset'] if 'subset' in params.keys() else None
        if not (subset is None):
            split = (subset, len(train_set) - subset)
            train_set, _ = random_split(train_set, split)
        augmented_set = augmentation_function(augmentation_model, num_classes, augmentation_per_class, use_cuda,
                                              output_dim)
        return ConcatDataset((train_set, augmented_set))
