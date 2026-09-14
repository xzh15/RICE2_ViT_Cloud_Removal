import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .position_embedding import PositionEmbedding
from .transformer_block import TransformerEncoderBlock
class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder
    输入:
        [B, 3, H, W]
    输出:
        [B, N, D]
    """
    def __init__( self,img_size: int = 256,patch_size: int = 16,in_channels: int = 3,embed_dim: int = 512,
                  depth: int = 8,num_heads: int = 8,mlp_ratio: float = 4.0,dropout: float = 0.0,):
        super().__init__()
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels=in_channels,embed_dim=embed_dim,patch_size=patch_size)
        # 2. 计算 Patch 数量
        if img_size % patch_size != 0:
            raise ValueError("img_size 必须能够被 patch_size 整除。")
        num_patches = (img_size // patch_size) ** 2
        # 3. Position Embedding
        self.pos_embed = PositionEmbedding(num_patches=num_patches,embed_dim=embed_dim,)
        # 4. Transformer Encoder Blocks
        self.blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,
                                                             dropout=dropout) for _ in range(depth)])
        # 5. 最终 LayerNorm
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        # 输入
        # [B, 3, H, W]
        x = self.patch_embed(x)
        # [B, N, D]
        x = self.pos_embed(x)
        # [B, N, D]
        for block in self.blocks:
            x = block(x)
        # [B, N, D]
        x = self.norm(x)
        return x