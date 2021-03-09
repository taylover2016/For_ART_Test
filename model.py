# -*- coding: utf-8 -*
import torch
from torch import nn, optim
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
#from logger import Logger

class Net(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Net, self).__init__()    # super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv = nn.Sequential(     #padding=2保证输入输出尺寸相同(参数依次是:输入深度，输出深度，ksize，步长，填充)
            nn.Conv2d(in_dim, 32, 5, stride=1, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d((2,2),(2,2) ),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d((2,2) ,(2,2)))


        self.fc1 = nn.Linear(14*12*64, 2000)
        self.fc2 = nn.Linear(2000, n_class)


    def forward(self, x):
        x = self.conv(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)

       # out = self.fc1(x)
        out = nn.functional.relu(self.fc1(x))

       # out = nn.ReLU(out)
        out = self.fc2(out)
        return out
