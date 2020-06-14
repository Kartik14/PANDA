import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class enetv2(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(enetv2, self).__init__()
        self.enet = model = EfficientNet.from_pretrained(enet_type, num_classes=out_dim)

    def forward(self, x):
        x = self.enet(x)
        return x