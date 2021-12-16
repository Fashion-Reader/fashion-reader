import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class CropedClassificationModel(nn.Module):
    def __init__(self):
        super(CropedClassificationModel, self).__init__()
        self.model = EfficientNet.from_pretrained(
            'efficientnet-b4', num_classes=1024)
        self.fc1 = nn.Linear(1024, 22)
        self.fc2 = nn.Linear(1024, 14)

    def forward(self, x):
        x = self.model(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        return x1, x2


class StyleClassificationModel(nn.Module):
    def __init__(self):
        super(StyleClassificationModel, self).__init__()
        self.model = EfficientNet.from_pretrained(
            'efficientnet-b4', num_classes=24)

    def forward(self, x):
        x = self.model(x)
        return x
