import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from pooling import GeM

class enet(nn.Module):
    def __init__(self, enet_type, out_dim, attn_dim):
        super(enet, self).__init__()
        self.enet = EfficientNet.from_pretrained(enet_type)
        fd_dim = self.enet._fc.in_features
        self.enet._fc = nn.Identity()
        self.enet._avg_pooling = GeM()
        attn_layer1 = nn.Linear(in_features=fd_dim, out_features=attn_dim)
        attn_layer2 = nn.Linear(in_features=attn_dim, out_features=1)
        self.attn = nn.Sequential(attn_layer1, nn.Tanh(), attn_layer2)
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=fd_dim, out_features=out_dim)

    def forward(self, inp, num_tiles):

        fd = self.enet(inp).squeeze() # (n_1 + .. n_bs) x fd_dim        
        attn_out = self.attn(fd) # (n_1 + .. n_bs) x 1
        attn_weights = []
        offset = 0
        for n in num_tiles:
            attn_weights.append(F.softmax(attn_out[offset:offset+n], dim=0))
            offset += n
        attn_weights = torch.cat(attn_weights) # (n_1 + .. n_bs) x 1

        fd = fd*attn_weights
        cumulative_fds = []
        offset = 0
        for n in num_tiles:
            cumulative_fds.append(fd[offset:offset+n].sum(0))
            offset += n
        cumulative_fds = torch.stack(cumulative_fds) # bs x fd_dim
        logits = self.fc(self.drop(cumulative_fds))

        return logits, attn_weights