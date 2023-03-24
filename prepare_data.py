from torchvision import datasets, transforms
import random
import wget
import os
from loggings import logger
import torch
import numpy as np

class CIFAR10_TRAIN(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])(img)
        return img, target
class CIFAR10_VAL(datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])(img)
        return img, target

def make_dataset(dataset,attack_method=''):
    logger.info(f"Loading {dataset} dataset")
    if dataset=="mnist":
        mnist_train=datasets.mnist.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mnist_val=datasets.mnist.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        data=list(zip(mnist_train.data,mnist_train.targets))
        random.shuffle(data)
        mnist_train.data,mnist_train.targets=zip(*data)
        mnist_train.data=torch.stack(mnist_train.data)
        mnist_train.targets=torch.stack(mnist_train.targets)
        train,val=mnist_train,mnist_val

    elif dataset=="cifar10":
        train_transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        val_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        if attack_method=="dba":
            train_transform=transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        cifar10_train=datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        cifar10_val=datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        if attack_method=="constrain_scale":
            poisoned_images=[30696,33105,33615,33907,36848,40713,41706]
            poisoned_test_images=[330,568,3934,12336,30560]
            indexes=np.array([i for i in range(len(cifar10_train.data)) if i not in poisoned_images+poisoned_test_images])
            cifar10_train.data=cifar10_train.data[indexes]
            cifar10_train.targets=np.array(cifar10_train.targets)[indexes]
            # cifar10_train.targets=cifar10_train.targets[indexes]
        data=list(zip(cifar10_train.data,cifar10_train.targets))
        random.shuffle(data)
        cifar10_train.data,cifar10_train.targets=zip(*data)
        cifar10_train.data=np.stack(cifar10_train.data)
        # cifar10_train.data=torch.from_numpy(cifar10_train.data)
        cifar10_train.targets=np.array(cifar10_train.targets)
        train,val=cifar10_train,cifar10_val

    elif dataset=="tiny_imagenet":
        if os.path.exists("./data/tiny-imagenet-200")==False:
            if os.path.exists("./data/tiny-imagenet-200.zip")==False:
                wget.download("http://cs231n.stanford.edu/tiny-imagenet-200.zip",out="./data")
            os.system("unzip ./data/tiny-imagenet-200.zip -d ./data > /dev/null")
        tiny_imagenet_train=datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transforms.ToTensor())
        tiny_imagenet_val=datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transforms.ToTensor())
        data=list(zip(tiny_imagenet_train.imgs,tiny_imagenet_train.targets))
        random.shuffle(data)
        tiny_imagenet_train.imgs,tiny_imagenet_train.targets=zip(*data)
        tiny_imagenet_train.imgs=torch.stack(tiny_imagenet_train.imgs)
        tiny_imagenet_train.targets=torch.stack(tiny_imagenet_train.targets)

        train,val=tiny_imagenet_train,tiny_imagenet_val
    logger.info(f"Loaded {dataset} dataset")
    return train,val
if __name__ == "__main__":
    train_data,val_data=make_dataset("cifar10")
    print(train_data[0][0].shape)
    print(val_data[0][0].shape)