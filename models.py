import torch.nn as nn

class CNN_Cifar(nn.Module):
    """
    This CNN is inspired by LeNet-5. It differs from Lenet-5 in few things such as 
    using ReLU instead of Sigmoid and using MaxPooling instead of AveragePooling.
    This architecture is only suitable for CIFAR dataset.
    """
    def __init__(self,num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                                    nn.Conv2d(3,6,5,1,2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(6,16,5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
                                 )
        self.fully_connected = nn.Sequential(
                                  nn.Linear(576,120),
                                  nn.ReLU(),
                                  nn.Linear(120,84),
                                  nn.ReLU(),
                                  nn.Linear(84,num_classes),
                               )
    def forward(self,x):
        x = self.feature_extractor(x)
        x = nn.Flatten()(x)
        x = self.fully_connected(x)
        return x


class CNN_Mnist(nn.Module):
    """
    This CNN is inspired by LeNet-5. It differs from Lenet-5 in few things such as 
    using ReLU instead of Sigmoid and using MaxPooling instead of AveragePooling.
    This architecture is only suitable for MNIST dataset.
    """
    def __init__(self,num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
                                    nn.Conv2d(1,6,5,1,2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2),
                                    nn.Conv2d(6,16,5),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2,2)
                                )
        self.fully_connected = nn.Sequential(
                                  nn.Linear(400,120),
                                  nn.ReLU(),
                                  nn.Linear(120,84),
                                  nn.ReLU(),
                                  nn.Linear(84,num_classes),
                                )
    def forward(self,x):
        x = self.feature_extractor(x)
        x = nn.Flatten()(x)
        x = self.fully_connected(x)
        return x
        
