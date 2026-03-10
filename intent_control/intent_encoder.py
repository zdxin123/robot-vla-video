import torch
import torch.nn as nn


class IntentEncoder(nn.Module):
    def __init__(self, joint_dim=24, seq_len=64, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_dim * seq_len, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        """
        x: [B, T, joint_dim]
        """
        x = x.reshape(x.size(0), -1)
        return self.net(x)
