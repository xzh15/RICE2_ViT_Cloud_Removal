import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        return x