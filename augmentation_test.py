from train_cvae import *
from train_cnn import train_cnn
from augment import *
from models import *

use_cuda = True
num_classes = 62
augmentation_per_class = 20000
latent_size = 1024 * 8

model_path = './saved_model/cvae_EMNIST'
cvae_model = CVAE(latent_size, num_classes)
cvae_model.load_state_dict(torch.load(model_path))
cvae_model.eval()
print("trained model loaded, start augmentation")

if use_cuda:
    cvae_model = cvae_model.cuda()

dataset_name = 'EMNIST'
if dataset_name == 'EMNIST':
    data = datasets.EMNIST
elif dataset_name == 'MNIST':
    data = datasets.MNIST

batch_size = 512
train_set = data('./data/'+dataset_name, train=True, download=True, split='byclass', transform=transforms.ToTensor())
augmented = perform_augmentation(train_set, cvae_aug, cvae_model, num_classes, augmentation_per_class, use_cuda)
test_set = data('./data/'+dataset_name, train=False, download=True, split='byclass', transform=transforms.ToTensor())
print("augmentation completed")

cnn_model = CNN(class_size=62)
if use_cuda:
    cnn_model = cnn_model.cuda()
cnn_optimizer = optim.SGD(cnn_model.parameters(), lr=5e-2, momentum=0.5)
#cnn_optimizer = optim.Adagrad(cnn_model.parameters(), lr=5e-3, lr_decay=1e-3)
cnn_save_path = './saved_model/cnn_augmented_EMNIST.pt'
print("start training")
train_cnn(cnn_model, cnn_optimizer, augmented, test_set, save_path=cnn_save_path, num_epochs=30)
print("done")