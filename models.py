from __future__ import print_function
import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

class CVAE(nn.Module):
    def __init__(self, latent_size, class_size):
        super(CVAE, self).__init__()
        #self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = 1024

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, self.units)
        )
        self.conditional = nn.Sequential(
            nn.Linear(self.class_size, self.units),
            nn.Tanh(),
            nn.Linear(self.units, self.units),
            nn.Tanh()
        )
        self.recog_head = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, self.units),
            nn.Tanh()
        )
        self.mu = nn.Sequential(nn.Linear(self.units, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512, self.latent_size),
        )
        self.logvar = nn.Sequential(nn.Linear(self.units, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512, self.latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.class_size, self.units),
            #nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(7 * 7 * 128),
            Unflatten(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def recognition_model(self, x, c):
        """
        Computes the parameters of the posterior distribution q(z | x, c) using the
        recognition network defined in the constructor

        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar PyTorch Variable of shape (batch_size, latent_size) for the posterior
          variance in log space
        """
        x = self.encoder(x)
        c = self.conditional(c)
        temp = self.recog_head(x + c)
        mu = self.mu(temp)
        logvar = self.logvar(temp)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, z, c):  # P(x|z, c)
        """
        Computes the generation output from the generative distribution p(x | z, c)
        using the generation network defined in the constructor

        Inputs:
        - z: PyTorch Variable of shape (batch_size, latent_size) for the latent vector
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        """
        z_c = torch.cat((z, c), dim=1)
        x_hat = self.decoder(z_c)
        return x_hat

    def forward(self, x, c):
        """
        Performs the inference and generation steps of the CVAE model using
        the recognition_model, reparametrization trick, and generation_model

        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch Variable of shape (batch_size, latent_size)
                  for the posterior logvar
        """
        batch_size, _, _, _ = x.size()
        mu, logvar = self.recognition_model(x, c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn([batch_size, self.latent_size]).to('cuda' if torch.cuda.is_available() else 'cpu')
        z_hat = eps * std + mu
        x_hat = self.generation_model(z_hat, c)
        return x_hat, mu, logvar

class CNN(nn.Module):
    def __init__(self, class_size=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, class_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Discriminator(nn.Module):
    def __init__(self, class_size):
        super(Discriminator, self).__init__()
        self.shared = nn.Sequential(
            Unflatten(-1, 1, 28, 28),
            nn.Conv2d(1, 32, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
        )
        self.label_head = nn.Sequential(
            #Flatten(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
        #self.feature_head = nn.Sequential(
        #    nn.Conv2d(64, 64, 5),
        #    nn.LeakyReLU(0.01),
        #    nn.Conv2d(64, 128, 5),
        #    nn.LeakyReLU(0.01),
        #)
        self.conditional = nn.Sequential(
            nn.Linear(class_size, 1024),
            nn.Tanh(),
            nn.Linear(1024, 1024),
            nn.Tanh()
        )
    def forward(self, input, classes):
        temp_x = self.shared(input)
        temp_c = self.conditional(classes)
        features = temp_x
        labels = self.label_head(Flatten()(temp_x) + temp_c)
        return features, labels

class CNN_CIFAR10(nn.Module):
    def __init__(self, class_size=10):
        super(CNN_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(1250, 512)
        self.fc2 = nn.Linear(512, class_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CVAE_CIFAR10(nn.Module):
    def __init__(self, latent_size, class_size):
        super(CVAE_CIFAR10, self).__init__()
        #self.input_size = input_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.units = 2048

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            Flatten(),
            nn.Linear(3200, 2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, self.units)
        )
        self.conditional = nn.Sequential(
            nn.Linear(self.class_size, self.units),
            nn.Tanh(),
            nn.Linear(self.units, self.units),
            nn.Tanh()
        )
        self.recog_head = nn.Sequential(
            nn.Linear(self.units, 2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, self.units),
            nn.Tanh()
        )
        self.mu = nn.Sequential(
            nn.Linear(self.units, 2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512, self.latent_size),
        )
        self.logvar = nn.Sequential(
            nn.Linear(self.units, 2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512, self.latent_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size + self.class_size, self.units),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 8 * 8 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(8 * 8 * 128),
            nn.Linear(8 * 8 * 128, 8 * 8 * 128),
            nn.ReLU(),
            nn.BatchNorm1d(8 * 8 * 128),
            Unflatten(-1, 128, 8, 8),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def recognition_model(self, x, c):
        """
        Computes the parameters of the posterior distribution q(z | x, c) using the
        recognition network defined in the constructor

        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar PyTorch Variable of shape (batch_size, latent_size) for the posterior
          variance in log space
        """
        x = self.encoder(x)
        c = self.conditional(c)
        temp = self.recog_head(x + c)
        mu = self.mu(temp)
        logvar = self.logvar(temp)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def generation_model(self, z, c):  # P(x|z, c)
        """
        Computes the generation output from the generative distribution p(x | z, c)
        using the generation network defined in the constructor

        Inputs:
        - z: PyTorch Variable of shape (batch_size, latent_size) for the latent vector
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        """
        z_c = torch.cat((z, c), dim=1)
        x_hat = self.decoder(z_c)
        return x_hat

    def forward(self, x, c):
        """
        Performs the inference and generation steps of the CVAE model using
        the recognition_model, reparametrization trick, and generation_model

        Inputs:
        - x: PyTorch Variable of shape (batch_size, input_size) for the input data
        - c: PyTorch Variable of shape (batch_size, num_classes) for the input data class

        Returns:
        - x_hat: PyTorch Variable of shape (batch_size, input_size) for the generated data
        - mu: PyTorch Variable of shape (batch_size, latent_size) for the posterior mu
        - logvar: PyTorch Variable of shape (batch_size, latent_size)
                  for the posterior logvar
        """
        batch_size, _, _, _ = x.size()
        mu, logvar = self.recognition_model(x, c)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn([batch_size, self.latent_size]).to('cuda' if torch.cuda.is_available() else 'cpu')
        z_hat = eps * std + mu
        x_hat = self.generation_model(z_hat, c)
        return x_hat, mu, logvar

class Discriminator_CIFAR10(nn.Module):
    def __init__(self, class_size):
        super(Discriminator_CIFAR10, self).__init__()
        self.shared = nn.Sequential(
            Unflatten(-1, 3, 32, 32),
            nn.Conv2d(3, 64, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, stride=2),
        )
        self.label_head = nn.Sequential(
            #Flatten(),
            nn.Linear(3200, 2048),
            nn.LeakyReLU(0.01),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 1)
        )
        self.conditional = nn.Sequential(
            nn.Linear(class_size, 2048),
            nn.Tanh(),
            nn.Linear(2048, 3200),
            nn.Tanh()
        )
    def forward(self, input, classes):
        temp_x = self.shared(input)
        temp_c = self.conditional(classes)
        features = temp_x
        labels = self.label_head(Flatten()(temp_x) + temp_c)
        return features, labels