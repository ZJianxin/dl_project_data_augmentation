# dl_project_data_augmentation

1. Neural nets architecture can be viewed and modified in models.py
2. To train a CNN, run `python train_cnn.py` after specifying the hyperparameters, datasets and paths in the file.
3. To train a CVAE, run `python train_cvae.py` after specifying the hyperparameters, datasets and paths in the file. Samples will be automatically generated.
4. To train a CVAE-GAN, run `python train_cvae_gan.py` after specifying the hyperparameters, datasets and paths in the file. Samples will be automatically generated.
5. To perform augmentation on with a pretrained model, modify the hyperparameters, datasets and paths in augmentation_test.py and run `python augmentation_test.py`
6. To use one of the MNIST, EMNIST, CIFAR-10 loaded torchvison package, create the folder `./data/NAME_OF_DATASET` and it will be downloaded while running the files.
7. Please make sure the paths to save samples and trained parameters exist; otherwise python will prompt an exception.
