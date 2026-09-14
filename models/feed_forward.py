import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int = 512, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = embed_dim * 4
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)


    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x