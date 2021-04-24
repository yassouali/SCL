
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from models.util import create_model


class Projection(nn.Module):
    """
    projection head
    """
    def __init__(self, dim, projection_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size, bias=False))
        
    def forward(self, x):
        return self.net(x)


class ContrastResNet(nn.Module):
    """
    defining the backbone and projection head
    """
    def __init__(self, opt, n_cls):
        super(ContrastResNet, self).__init__()

        self.encoder = create_model(opt.model, n_cls, dataset=opt.dataset)
        dim_in = self.encoder.feat_dim
        projection_size = opt.feat_dim
        self.head = Projection(dim=dim_in, projection_size=projection_size, hidden_size=dim_in)

        self.global_cont_loss = opt.global_cont_loss
        self.spatial_cont_loss = opt.spatial_cont_loss
        
    def forward(self, x):
        # forward pass through the embedding model, feat is a list of features
        feat, outputs = self.encoder(x, is_feat=True)

        # spatial features before avg pool
        spatial_f = feat[-2]

        # global features after avg pool
        avg_pool_feat = feat[-1]

        # projected global features
        global_f = self.head(avg_pool_feat)
        return outputs, spatial_f, global_f, avg_pool_feat