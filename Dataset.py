import torch
from torch.utils.data import Dataset
from torch import Tensor

class Dataset(Dataset):

    def __init__(self, X, y):
        self.X = Tensor(X)
        self.y = Tensor(y).long()

    def __len__(self):
        return self.X.size()[0]

    def __getitem__(self, index):
        return (self.X[index, :], self.y[index])
