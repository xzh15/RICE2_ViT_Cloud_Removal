import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from torchmetrics.image import StructuralSimilarityIndexMeasure

from datasets import create_dataloaders
from datasets.rice2_dataset import RICE2Dataset
from models import ViTRestoration
from losses import ReconstructionLoss
from metrics import calculate_psnr


def main():
    # 1. 配置
    seed = 42

    image_size = 256
    batch_size = 4

    num_workers = 0

    # ViT 参数
    patch_size = 16
    embed_dim = 512
    depth = 8
    num_heads = 8
    mlp_ratio = 4.0
    dropout = 0.0

    data_root = "./data/RICE2"

    checkpoint_path = (
        "./checkpoints/best_model.pth"
    )

    result_dir = "./results"

    os.makedirs(
        result_dir,
        exist_ok=True,
    )

    for f in os.listdir(result_dir):
        fp = os.path.join(result_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    # 2. Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("RICE2 ViT Cloud Removal - Test")
    print("=" * 70)

    print(f"Device: {device}")

    # 3. 创建 DataLoader（只测试训练用的 100.png ~ 103.png 共 4 张图）

    print("\nCreating DataLoader (mini test set)...")

    torch.manual_seed(seed)

    full_dataset = RICE2Dataset(
        root_dir=data_root,
        image_size=image_size,
        use_mask=True,
    )

    target_filenames = {f"{i}.png" for i in range(100, 104)}

    target_indices = [
        idx
        for idx, name in enumerate(full_dataset.filenames)
        if name in target_filenames
    ]

    if len(target_indices) != len(target_filenames):
        raise RuntimeError(
            f"目标图片未全部找到，只在数据集中匹配到 {len(target_indices)} 张，"
            f"期望 {len(target_filenames)} 张：{sorted(target_filenames)}"
        )

    mini_test_dataset = Subset(full_dataset, target_indices)

    test_loader = DataLoader(
        dataset=mini_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"Mini test dataset size: {len(mini_test_dataset)} "
        f"(files: {sorted(target_filenames)})"
    )
    print(f"Test batches: {len(test_loader)}")

    # 4. 创建模型

    print("\nCreating model...")

    model = ViTRestoration(
        img_size=image_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )

    model = model.to(device)

    # 5. 创建 Loss
    criterion = ReconstructionLoss()

    # 6. 加载最佳模型
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best Epoch: {checkpoint['epoch']}")
    print(f"Best Validation PSNR: {checkpoint['val_psnr']:.4f} dB")
    print(f"Best Validation SSIM: {checkpoint['val_ssim']:.4f}")

    # 7. Evaluation Mode
    model.eval()

    # 8. SSIM Metric
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # 9. 统计变量

    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    # 10. Test

    print("\nStart testing...")
    print("=" * 70)

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            # 输入
            cloud = batch["cloud"].to(device)
            label = batch["label"].to(device)
            filenames = batch["filename"]

            # Forward
            prediction = model(cloud)

            # Loss
            loss = criterion(prediction, label, )

            # PSNR
            psnr = calculate_psnr(prediction, label, max_value=1.0, )

            # SSIM
            ssim = ssim_metric(prediction, label, ).item()

            # 统计
            current_batch_size = cloud.size(0)

            total_loss += (loss.item() * current_batch_size)

            total_psnr += (psnr * current_batch_size)

            total_ssim += (ssim * current_batch_size)

            total_samples += (current_batch_size)

            # 11. 保存部分可视化结果
            # 只保存前几个 Batch，
            # 防止 results 文件夹产生大量图片。

            if batch_idx < 5:
                for i in range(
                        current_batch_size
                ):
                    filename = filenames[i]
                    stem = os.path.splitext(
                        filename
                    )[0]

                    # Cloud
                    save_image(
                        cloud[i].cpu(),
                        os.path.join(result_dir, f"{stem}_cloud.png", ),
                    )

                    # Prediction
                    prediction_image = (
                        prediction[i]
                        .cpu()
                        .clamp(0.0, 1.0)
                    )

                    save_image(
                        prediction_image,
                        os.path.join(
                            result_dir,
                            f"{stem}_prediction.png",
                        ),
                    )

                    # Ground Truth
                    save_image(label[i].cpu(), os.path.join(result_dir, f"{stem}_label.png", ),
                               )

    # 12. 平均结果
    test_loss = (total_loss / total_samples)
    test_psnr = (total_psnr / total_samples)
    test_ssim = (total_ssim / total_samples)

    # 13. 打印最终结果
    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Test Loss : {test_loss:.6f}")
    print(f"Test PSNR : {test_psnr:.4f} dB")
    print(f"Test SSIM : {test_ssim:.4f}")
    print(f"\nVisualization saved to:"f"\n{result_dir}"
          )

    print("=" * 70)


if __name__ == "__main__":
    main()