import torch
from torch import nn
from torch.nn import functional as F

#Здесь описываются модели
class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(64**2, 320)
        self.fc2 = nn.Linear(320, 160)
        self.out = nn.Linear(160, 101)


    def forward(self, x): 
        x = self.conv1(x) 
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=3, stride=2)
        x = self.conv2(x)        
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=3, stride=2, padding=1) 
        x = self.conv3(x)
        x = F.relu(x)
        x = F.adaptive_max_pool3d(x, output_size=(1, 1, 64))
        x = x.reshape(-1, 64**2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return F.softmax(x)