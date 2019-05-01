from __future__ import print_function
import torch.utils.data
from torch import optim

from utils import *
from models import *

def train(epoch, model, train_loader, optimizer, num_classes, use_cuda):
    #cvae training for each epoch
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = to_var(data, use_cuda)#.view(data.shape[0], -1)
        labels = one_hot(labels, num_classes, use_cuda)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))

def loss_function(x_hat, x, mu, logvar):
    # define the loss function for cvae
    BCE = F.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = (BCE + KLD) / x.size(0)
    return loss

def train_cvae(model, optimizer, dataset_name, **params):
    # train a cvae
    use_cuda = params['use_cuda'] if 'use_cuda' in params.keys() else torch.cuda.is_available()
    #input_size = params['input_size'] if 'input_size' in params.keys() else 28 * 28
    batch_size = params['batch_size'] if 'batch_size' in params.keys() else 512
    latent_size = params['latent_size'] if 'latent_size' in params.keys() else 1024 * 8
    num_classes = params['num_classes'] if 'num_classes' in params.keys() else  62
    num_epochs = params['num_epochs'] if 'num_epochs' in params.keys() else 20
    size_data = params['size_data'] if 'size_data' in params.keys() else 697932
    save_freq = params['save_freq'] if 'save_freq' in params.keys() else 1
    save_path = params['save_path'] if 'save_path' in params.keys() else './res/'+dataset_name+'_res/'

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
    for epoch in range(1, num_epochs+1):
        train(epoch, model, train_loader, optimizer, num_classes, use_cuda)
        if epoch % save_freq == 0:
            save_epoch = save_path + 'epoch_' + str(epoch)
            if dataset_name == 'CIFAR10':
                image_size = (3, 32, 32)
            else:
                image_size = (1, 28, 28)
            standard_sample(model.generation_model, num_classes, latent_size, use_cuda, save_epoch, image_size=image_size, transformation=sample_transformation)

def main():
    use_cuda = torch.cuda.is_available()
    #input_size = 28 * 28
    batch_size = 512
    latent_size = 1024 * 8
    num_classes = 10
    num_epochs = 100
    subset = None
    #subset = 1000

    #model = CVAE(latent_size, num_classes)
    model = CVAE_CIFAR10(latent_size, num_classes)
    if use_cuda:
        model = model.cuda()
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
    #optimizer = optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer = optim.Adagrad(model.parameters(), lr=1e-3, lr_decay=1e-3)

    #name_dataset = 'EMNIST'
    #name_dataset = 'SVHN'
    name_dataset = 'CIFAR10'

    train_cvae(model, optimizer, name_dataset, use_cuda=use_cuda, batch_size=batch_size,
               latent_size=latent_size, num_classes=num_classes, num_epochs=num_epochs, size_data=subset)

    #model_savepath = './saved_model/cvae_local_test_' + name_dataset
    model_savepath = './saved_model/cvae_' + name_dataset
    #model_savepath = './saved_model/cvae_' + name_dataset + str(subset) +'.pt'
    torch.save(model.state_dict(), model_savepath)

if __name__ == "__main__":
  main()

