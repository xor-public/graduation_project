from torchvision import datasets, transforms
import random

def make_dataset(dataset):
    random.seed(42)
    if dataset=="mnist":
        mnist_train=datasets.mnist.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        mnist_val=datasets.mnist.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        mnist_train=list(mnist_train)
        random.shuffle(mnist_train)
        return mnist_train,mnist_val

    elif dataset=="cifar10":
        cifar10_train=datasets.cifar.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        cifar10_val=datasets.cifar.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
        cifar10_train=list(cifar10_train)
        random.shuffle(cifar10_train)
        return cifar10_train,cifar10_val

    elif dataset=="tiny_imagenet":
        tiny_imagenet_train=datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transforms.ToTensor())
        tiny_imagenet_val=datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transforms.ToTensor())
        tiny_imagenet_train=list(tiny_imagenet_train)
        random.shuffle(tiny_imagenet_train)
        return tiny_imagenet_train,tiny_imagenet_val