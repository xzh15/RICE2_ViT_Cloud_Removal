import torch

from torchmetrics.image import StructuralSimilarityIndexMeasure

from metrics.psnr import calculate_psnr


def validate(model,dataloader,criterion,device,):
    """
    在验证集上计算：
    1. Loss
    2. PSNR
    3. SSIM
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    with torch.no_grad():
        for batch in dataloader:
            cloud = batch["cloud"].to(device)
            label = batch["label"].to(device)
            # Forward
            prediction = model(cloud)
            # Loss
            loss = criterion(prediction,label,)
            # PSNR
            psnr = calculate_psnr(prediction,label,max_value=1.0,)

            # SSIM
            ssim = ssim_metric(prediction,label,).item()
            batch_size = cloud.size(0)
            total_loss += (loss.item() * batch_size)

            total_psnr += (psnr * batch_size)

            total_ssim += (ssim * batch_size)

            total_samples += batch_size

    average_loss = (total_loss / total_samples)

    average_psnr = (total_psnr / total_samples)

    average_ssim = (total_ssim / total_samples)

    return (average_loss,average_psnr,average_ssim,)