from __future__ import print_function
import torch.utils.data
from torch import optim
from torch import Tensor

from utils import *
from models import *

def train_epoch(epoch, cvae, discriminator, train_loader, cvae_optimizer, d_optimizer, num_classes, use_cuda):
    #cvae training for each epoch
    device = torch.device('cuda' if use_cuda else 'cpu')
    cvae.train()
    discriminator.train()
    latent_size = cvae.latent_size
    #train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        batch_size = data.size()[0]

        x_real = to_var(data, use_cuda)#.view(data.shape[0], -1)
        labels = one_hot(labels, num_classes, use_cuda)
        x_recon, mu, logvar = cvae(x_real, labels)

        z_sample = to_var(torch.randn(batch_size, latent_size), use_cuda)
        x_sample = cvae.generation_model(z_sample, labels)

        feature_real, logits_real = discriminator(x_real, labels)
        feature_recon, logits_recon = discriminator(x_recon, labels)
        feature_sample, logits_sample = discriminator(x_sample, labels)

        # train discriminator
        d_optimizer.zero_grad()
        loss_D = discriminator_loss(logits_real, logits_recon, logits_sample, batch_size, device)
        loss_D.backward(retain_graph=True)
        d_optimizer.step()

        # train encoder and decoder
        cvae_optimizer.zero_grad()
        loss_EG = cvae_loss(x_real, x_recon, feature_real, feature_recon, mu, logvar, logits_sample, batch_size, device)
        loss_EG.backward()
        #train_loss += loss_EG.data
        cvae_optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tDLoss: {:.6f}\tEGLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_D.data, loss_EG.data))

def cvae_loss(x_real, x_recon, feature_real, feature_recon, mu, logvar, logits_sample, batch_size, device, weights=(1.0, 5e-2, 5e-2)):
    ones = torch.ones(batch_size).to(device)
    w_gen, w_recon, w_KLD = weights
    loss_gen = bce_loss(logits_sample, ones)
    #loss_recon = F.binary_cross_entropy(x_recon, feature_real, reduction='sum')
    #loss_recon = 0.5 * (0.01*(x_recon - x_real).pow(2).sum() + (feature_recon - feature_real.detach()).pow(2).sum())
    loss_recon = (0.01(x_recon - x_real).pow(2).sum() + (feature_recon - feature_real).pow(2).sum())
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #return (w_gen * loss_gen + w_recon * loss_recon + w_KLD * KLD) / x_real.size(0)
    return (w_gen * loss_gen + w_recon * loss_recon + w_KLD * KLD) / x_real.size(0)

def bce_loss(input, target):
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()

def discriminator_loss(logits_real, logits_recon, logits_sample, batch_size, device,
                       weights=(1.0, 0.5, 0.5)):
    w_real, w_recon, w_sample = weights
    ones = torch.ones(batch_size).to(device)
    zeros = torch.zeros(batch_size).to(device)
    return w_real * bce_loss(logits_real, ones) \
           + w_recon * bce_loss(logits_recon, zeros) \
           + w_sample * bce_loss(logits_sample, zeros)

def train_cvaegan(cvae, discriminator, cvae_optimizer, d_optimizer, dataset_name, **params):
    # train a cvae
    use_cuda = params['use_cuda'] if 'use_cuda' in params.keys() else torch.cuda.is_available()
    #input_size = params['input_size'] if 'input_size' in params.keys() else 28 * 28
    batch_size = params['batch_size'] if 'batch_size' in params.keys() else 512
    latent_size = params['latent_size'] if 'latent_size' in params.keys() else 1024 * 8
    num_classes = params['num_classes'] if 'num_classes' in params.keys() else  62
    num_epochs = params['num_epochs'] if 'num_epochs' in params.keys() else 20
    size_data = params['size_data'] if 'size_data' in params.keys() else 697932
    save_model_freq = params['save_model_freq'] if 'save_model_freq' in params.keys() else 10
    save_sample_freq = params['save_sample_freq'] if 'save_sample_freq' in params.keys() else 1
    save_model_path = params['save_model_path'] if 'save_model_path' in params.keys() else './saved_model/cvae_gan_' + dataset_name + '/'
    save_sample_path = params['save_sample_path'] if 'save_sample_path' in params.keys() else './res/cvaegan_res_' + dataset_name + '/'

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    #load data
    dataset = get_dataset(dataset_name, subset=size_data)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    #sample
    if dataset_name == 'EMNIST':
        sample_transformation = EMNIST_process
    else:
        sample_transformation = None
    #train
    for epoch in range(1, num_epochs):
        train_epoch(epoch, cvae, discriminator, train_loader, cvae_optimizer, d_optimizer, num_classes, use_cuda)
        if epoch % save_sample_freq == 0:
            if dataset_name == 'CIFAR10':
                image_size = (3, 32, 32)
            else:
                image_size = (1, 28, 28)
            save_epoch = save_sample_path + 'epoch_' + str(epoch)
            standard_sample(cvae.generation_model, num_classes, latent_size, use_cuda, save_epoch, image_size=image_size, transformation=sample_transformation)
        if epoch % save_model_freq == 0:
            torch.save(cvae.state_dict(), save_model_path + str(epoch) + '_cvae.pt')
            torch.save(discriminator.state_dict(), save_model_path + str(epoch) + '_discriminator.pt')

def main():
    use_cuda = torch.cuda.is_available()
    batch_size = 512
    latent_size = 1024 * 8
    #latent_size = 1024 * 32
    #num_classes = 10
    num_classes = 62
    num_epochs = 100
    #subset = None
    subset = 500

    #cvae = CVAE_CIFAR10(latent_size, num_classes)
    #discriminator = Discriminator_CIFAR10(num_classes)
    cvae = CVAE(latent_size, num_classes)
    discriminator = Discriminator(num_classes)
    if use_cuda:
        cvae = cvae.cuda()
        discriminator = discriminator.cuda()

    #cvae_optimizer = optim.Adam(cvae.parameters(), lr=2e-4, betas=(0.5, 0.999))
    #d_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    cvae_optimizer = optim.RMSprop(cvae.parameters(), lr=1e-4)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=1e-4)

    #name_dataset = 'EMNIST'
    name_dataset = 'CIFAR10'

    train_cvaegan(cvae, discriminator, cvae_optimizer, d_optimizer, name_dataset, use_cuda=use_cuda, batch_size=batch_size,
                  latent_size=latent_size, num_classes=num_classes, num_epochs=num_epochs, size_data=subset)

    cvae_gan_savepath = './saved_model/cvae_gan_' + name_dataset
    torch.save(cvae.state_dict(), cvae_gan_savepath + 'cvae.pt')
    torch.save(discriminator.state_dict(), cvae_gan_savepath + 'discriminator.pt')

if __name__ == "__main__":
  main()