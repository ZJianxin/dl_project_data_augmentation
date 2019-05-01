from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from models import *

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def train_cnn(model, optimizer, train_set, test_set, **params):
    use_cuda = params['use_cuda'] if 'use_cuda' in params.keys() else torch.cuda.is_available()
    num_epochs = params['num_epochs'] if 'num_epochs' in params.keys() else 20
    batch_size = params['batch_size'] if 'batch_size' in params.keys() else 512
    log_interval = params['log_interval'] if 'log_interval' in params.keys() else 10
    save_path = params['save_path'] if 'save_path' in params.keys() else None
    training_size = params['training_size'] if 'training_size' in params.keys() else None

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    device = torch.device("cuda" if use_cuda else "cpu")
    #data
    if not (training_size is None):
        split = (training_size, len(train_set) - training_size)
        train_set, _ = torch.utils.data.random_split(training_size, split)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, **kwargs)

    #train
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval=log_interval)
        test(model, device, test_loader)
    if not (save_path is None):
        torch.save(model.state_dict(), save_path)

def main():
    torch.manual_seed(999)
    lr = 0.05
    momentum = 0.5
    use_cuda = torch.cuda.is_available()
    batch_size = 512
    num_epochs = 10
    log_interval = 10
    save_path = None

    #train_set = datasets.EMNIST('./data/EMNIST', train=True, download=True, split='byclass', transform=transforms.ToTensor())
    #test_set = datasets.EMNIST('./data/EMNIST', train=False, download=True, split='byclass', transform=transforms.ToTensor())
    train_set = datasets.CIFAR10('./data/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.CIFAR10('./data/CIFAR10', train=False, download=True, transform=transforms.ToTensor())
    if use_cuda:
        #model = CNN(62).cuda()
        #model = CNN(10).cuda()
        model = CNN_CIFAR10(10).cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # cnn_optimizer = optim.Adagrad(cnn_model.parameters(), lr=5e-3, lr_decay=1e-3)
    train_cnn(model, optimizer, train_set, test_set, use_cuda=use_cuda, batch_size=batch_size,
              num_epochs=num_epochs, log_interval=log_interval, save_path=save_path)


if __name__ == '__main__':
    main()