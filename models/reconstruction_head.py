import torch
import torch.nn as nn


class TokenToFeatureMap(nn.Module):
    """
    将 ViT Token 从 [B, N, D]
    恢复成二维特征图 [B, D, H, W]
    """
    def __init__(self,img_size: int = 256, patch_size: int = 16,embed_dim: int = 512,):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(
                "img_size 必须能够被 patch_size 整除。"
            )
        self.feature_size = img_size // patch_size
        self.num_patches = self.feature_size ** 2
        self.embed_dim = embed_dim

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        # [B, N, D]
        B, N, D = x.shape
        if N != self.num_patches:
            raise ValueError(f"Token数量错误："f"当前 N={N}，"f"期望 N={self.num_patches}")
        if D != self.embed_dim:
            raise ValueError(f"Embedding维度错误："f"当前 D={D}，"f"期望 D={self.embed_dim}")
        # [B, N, D] --> [B, D, N]
        x = x.transpose(1, 2)
        # [B, 512, 256] --> [B, 512, 16, 16]
        x = x.reshape(B,D,self.feature_size,self.feature_size,)
        return x

class UpBlock(nn.Module):
    """
    一个简单的上采样模块：
    ConvTranspose2d
        ↓
    Conv2d
        ↓
    GELU
    """
    def __init__(self,in_channels: int,out_channels: int,):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels, kernel_size=2,stride=2,padding=0,output_padding=0,)
        self.conv = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=True,)

        self.act = nn.GELU()

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class ReconstructionHead(nn.Module):
    """
    ViT Token → RGB 图像
    """
    def __init__(self,img_size: int = 256,patch_size: int = 16,embed_dim: int = 512,out_channels: int = 3,):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size 必须能够被 patch_size 整除。")
        feature_size = img_size // patch_size
        if feature_size != 16:
            raise ValueError("当前这个 Decoder 示例假设初始特征图为 16×16。")

        # Token → Feature Map
        self.token_to_feature = TokenToFeatureMap(img_size=img_size,patch_size=patch_size,embed_dim=embed_dim,)
        # 16×16 → 32×32
        self.up1 = UpBlock(in_channels=512,out_channels=256,)
        # 32×32 → 64×64
        self.up2 = UpBlock(in_channels=256,out_channels=128,)
        # 64×64 → 128×128
        self.up3 = UpBlock(in_channels=128,out_channels=64,)
        # 128×128 → 256×256
        self.up4 = UpBlock(in_channels=64,out_channels=32,)
        # 32 → RGB 3
        self.out_conv = nn.Conv2d(in_channels=32,out_channels=out_channels,kernel_size=3,stride=1,padding=1,bias=True,)

    def forward(self,x: torch.Tensor,) -> torch.Tensor:
        # [B,256,512]
        x = self.token_to_feature(x)
        # [B,512,16,16]
        x = self.up1(x)
        # [B,256,32,32]
        x = self.up2(x)
        # [B,128,64,64]
        x = self.up3(x)
        # [B,64,128,128]
        x = self.up4(x)
        # [B,32,256,256]
        x = self.out_conv(x)
        # [B,3,256,256]

        return x