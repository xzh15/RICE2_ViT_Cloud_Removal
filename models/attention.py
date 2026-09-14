import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能够被 num_heads 整除")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, True)

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D =x.shape
        qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)

        attn = attn / (self.head_dim ** 0.5)

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v

        out = out.transpose(1, 2)
        out = out.reshape(B, N, D)
        out = self.proj(out)
        return out