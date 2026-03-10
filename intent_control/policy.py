import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(self, latent_dim=128, joint_dim=24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, joint_dim)
        )

    def forward(self, z):
        return self.net(z)
