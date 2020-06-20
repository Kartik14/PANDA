import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from pooling import GeM

class enetv2(nn.Module):
    def __init__(self, enet_type, out_dim, attn_dim, n_tiles):
        super(enetv2, self).__init__()
        self.enet = model = EfficientNet.from_pretrained(enet_type)
        fd_dim = self.enet._fc.in_features
        self.n_tiles = n_tiles
        self.enet._fc = nn.Identity()
        self.enet._avg_pooling = GeM()
        attn_layer1 = nn.Linear(in_features=fd_dim, out_features=attn_dim)
        attn_layer2 = nn.Linear(in_features=attn_dim, out_features=1)
        self.attn = nn.Sequential(attn_layer1, nn.Tanh(), attn_layer2)
        self.fc = nn.Linear(in_features=fd_dim, out_features=out_dim)

    def forward(self, inp):
        fd = self.enet(inp).squeeze() # bs*N x fd_dim
        attn_weights = F.softmax(self.attn(fd).reshape(-1, self.n_tiles), dim=1).reshape(-1, 1) # bs*N x 1
        fd = fd*attn_weights # (bs*N) x fd_dim
        fd = fd.reshape(-1, self.n_tiles, fd.shape[-1]).sum(1) # bs x fd_dim
        logits = self.fc(fd)

        return logits