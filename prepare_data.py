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
        return cifar10_train,cifar10_val
