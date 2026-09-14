import torch
import torch.nn as nn

'''
  ViT Patch Embedding

  输入：
      [B, 3, H, W]

  输出：
      [B, N, D]

  其中：
      B = batch size
      N = patch 数量
      D = embedding dimension
'''

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int=512, patch_size: int=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels = in_channels, out_channels = embed_dim, kernel_size = patch_size, stride = patch_size, padding = 0)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x