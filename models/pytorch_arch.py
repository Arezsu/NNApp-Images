# app/models/pytorch_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TabularNet(nn.Module):
    def __init__(self, input_dim=30, n_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,n_classes),
            nn.Sigmoid()
        )
    def forward(self,x): return self.net(x)

class ImageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(64*8*8,64), nn.ReLU(),
            nn.Linear(64,10)
        )
    def forward(self,x): return self.net(x)

class AudioCNN(nn.Module):
    def __init__(self, n_classes=35):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64, n_classes)
        )
    def forward(self,x): return self.net(x)


# ---------------------------------------------------------------
# Clasificación de animales (15 clases). Son las mismas clases que
# se definen en notebooks/CNN_Animals_PyTorch.ipynb
# ---------------------------------------------------------------
class AnimalCNN(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)


class AnimalEfficientNet(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.base = models.efficientnet_b0(weights=None)
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.base(x)
