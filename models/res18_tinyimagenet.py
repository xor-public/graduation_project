from .res18_cifar import ResNet,BasicBlock
def RES18_TINYIMAGENET():
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=200)