import torch
import torch.nn as nn

from .vit_encoder import ViTEncoder
from .reconstruction_head import ReconstructionHead


class ViTRestoration(nn.Module):
    """
    ViT 遥感图像去云模型
    输入：
        [B,3,256,256]
    输出：
        [B,3,256,256]
    """

    def __init__(self,img_size: int = 256,patch_size: int = 16,in_channels: int = 3,embed_dim: int = 512,
        depth: int = 8,num_heads: int = 8,mlp_ratio: float = 4.0,dropout: float = 0.0,):
        super().__init__()
        self.encoder = ViTEncoder(img_size=img_size,patch_size=patch_size,in_channels=in_channels,embed_dim=embed_dim,
            depth=depth,num_heads=num_heads,mlp_ratio=mlp_ratio,dropout=dropout,
        )

        self.decoder = ReconstructionHead(img_size=img_size,patch_size=patch_size,
            embed_dim=embed_dim, out_channels=in_channels,)

    def forward(self, x: torch.Tensor,) -> torch.Tensor:
        # Encoder
        x = self.encoder(x)
        # [B,256,512]

        # Decoder
        x = self.decoder(x)
        # [B,3,256,256]
        return x