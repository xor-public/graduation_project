from torchvision import datasets, transforms
import random
import wget
import os
from loggings import logger

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
def make_dataset(dataset):
    logger.info(f"Loading {dataset} dataset")
    if dataset=="mnist":
        mnist_train=datasets.mnist.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mnist_val=datasets.mnist.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        mnist_train=list(mnist_train)
        random.shuffle(mnist_train)
        train,val=mnist_train,mnist_val

    elif dataset=="cifar10":
        # cifar10_train=CIFAR10_TRAIN(root='./data', train=True, download=True, transform=transforms.ToTensor())
        # cifar10_val=CIFAR10_TRAIN(root='./data', train=False, download=True, transform=transforms.ToTensor())
        cifar10_train=datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        cifar10_val=datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
        cifar10_train=list(cifar10_train)
        random.shuffle(cifar10_train)
        train,val=cifar10_train,cifar10_val

    elif dataset=="tiny_imagenet":
        if os.path.exists("./data/tiny-imagenet-200")==False:
            if os.path.exists("./data/tiny-imagenet-200.zip")==False:
                wget.download("http://cs231n.stanford.edu/tiny-imagenet-200.zip",out="./data")
            os.system("unzip ./data/tiny-imagenet-200.zip -d ./data > /dev/null")
        tiny_imagenet_train=datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transforms.ToTensor())
        tiny_imagenet_val=datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transforms.ToTensor())
        tiny_imagenet_train=list(tiny_imagenet_train)
        random.shuffle(tiny_imagenet_train)
        train,val=tiny_imagenet_train,tiny_imagenet_val
    logger.info(f"Loaded {dataset} dataset")
    return train,val
if __name__ == "__main__":
    train_data,val_data=make_dataset("cifar10")
    print(train_data[0][0].shape)
    print(val_data[0][0].shape)