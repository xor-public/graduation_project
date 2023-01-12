from models.cnn_mnist import CNN_MNIST
from models.mlp_mnist import MLP_MNIST
from models.cnn_cifar import CNN_CIFAR
from models.res18_cifar import RES18_CIFAR

class ModelLoader:
    def __init__(self, config):
        self.config = config
    def load_model(self):
        if self.config["model"] == "mlp_mnist":
            return MLP_MNIST()
        elif self.config["model"] == "cnn_mnist":
            return CNN_MNIST()
        elif self.config["model"] == "cnn_cifar":
            return CNN_CIFAR()
        elif self.config["model"] == "res18_cifar":
            return RES18_CIFAR()