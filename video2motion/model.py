import torch
import torch.nn as nn


class Video2MotionNet(nn.Module):
    def __init__(self, input_dim=99, hidden_dim=256, joint_dim=24, num_layers=3, nhead=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, joint_dim)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        return: [B, T, joint_dim]
        """
        x = self.input_proj(x)
        x = self.encoder(x)
        return self.output_head(x)
