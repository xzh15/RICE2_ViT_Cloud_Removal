import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention
from .feed_forward import FeedForward

class TransformerEncoderBlock(nn.Module):
    """
    Pre-Norm Transformer Encoder Block
    输入：
        [B, N, D]
    输出：
        [B, N, D]
    """
    def __init__(self, embed_dim: int = 512,num_heads: int = 8,mlp_ratio: float = 4.0,dropout: float = 0.0,
    ):
        super().__init__()
        # 第一部分：Attention
        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim=embed_dim,num_heads=num_heads,dropout=dropout,)

        # 第二部分：FFN
        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim,mlp_ratio=mlp_ratio,dropout=dropout,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一阶段
        # X1 = X + MHA(LN(X))
        x = x + self.attn(self.norm1(x))

        # 第二阶段
        # X2 = X1 + FFN(LN(X1))
        x = x + self.ffn(self.norm2(x))

        return x