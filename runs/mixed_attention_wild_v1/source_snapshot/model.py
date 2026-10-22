"""Original sequential CNN, randomly initialized. No pretrained architecture."""
import torch
from torch import nn

CLASSES = ["Healthy", "Unhealthy"]

class LeafCNN(nn.Module):
    def __init__(self, width=1, attention=False):
        super().__init__()
        layers=[]
        previous=3
        for count in [32,64,128,192]:
            channels=int(count*width)
            layers += [nn.Conv2d(previous,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels),nn.SiLU(inplace=True),
                       nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels),nn.SiLU(inplace=True)]
            if attention: layers.append(ChannelGate(channels))
            layers.append(nn.MaxPool2d(2))
            previous=channels
        self.features=nn.Sequential(*layers)
        self.classifier=nn.Sequential(nn.AdaptiveAvgPool2d(2),nn.Flatten(),nn.Linear(previous*4,128),
                                      nn.SiLU(),nn.Dropout(.4),nn.Linear(128,2))

    def forward(self,x):
        return self.classifier(self.features(x))

class ChannelGate(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.gate=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,max(8,channels//8),1),
                                nn.SiLU(),nn.Conv2d(max(8,channels//8),channels,1),nn.Sigmoid())

    def forward(self,x):
        return x*self.gate(x)
