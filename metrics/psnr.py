import torch


def calculate_psnr(prediction: torch.Tensor,target: torch.Tensor,max_value: float = 1.0,) -> float:
    """
    计算一批图像的平均 PSNR。
    prediction:
        [B, C, H, W]
    target:
        [B, C, H, W]
    max_value:
        像素最大值。
        如果图像范围为 [0,1]，则为 1.0。
    """

    mse = torch.mean((prediction - target) ** 2,dim=(1, 2, 3),)

    psnr = 10.0 * torch.log10(
        (max_value ** 2) / mse.clamp_min(1e-10)
    )

    return psnr.mean().item()