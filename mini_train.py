import os

import torch
from torch.utils.data import DataLoader, Subset

from datasets import create_dataloaders
from datasets.rice2_dataset import RICE2Dataset
from models import ViTRestoration
from losses import ReconstructionLoss
from validate import validate


def main():

    # ========================================================
    # 1. 超参数
    # ========================================================

    seed = 42

    image_size = 256
    batch_size = 4

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    num_workers = 0

    epochs = 500

    learning_rate = 1e-4
    weight_decay = 1e-4

    # ViT
    patch_size = 16
    embed_dim = 512
    depth = 8
    num_heads = 8
    mlp_ratio = 4.0
    dropout = 0.0

    data_root = "./data/RICE2"

    checkpoint_dir = "./checkpoints"

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    best_model_path = os.path.join(
        checkpoint_dir,
        "best_model.pth",
    )

    # ========================================================
    # 2. 固定随机种子
    # ========================================================

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # ========================================================
    # 3. Device
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ========================================================
    # 4. 创建 DataLoader（只使用 100.png ~ 103.png 共 4 张图）
    # ========================================================

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

    mini_dataset = Subset(full_dataset, target_indices)

    print(
        f"Mini dataset size: {len(mini_dataset)} "
        f"(files: {sorted(target_filenames)})"
    )

    train_loader = DataLoader(
        dataset=mini_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=mini_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ========================================================
    # 5. 创建模型
    # ========================================================

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

    # ========================================================
    # 6. 创建 Loss
    # ========================================================

    criterion = ReconstructionLoss()

    # ========================================================
    # 7. 创建 Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ========================================================
    # 8. Best PSNR
    # ========================================================

    best_psnr = -float("inf")

    # ========================================================
    # 9. Training Loop
    # ========================================================

    for epoch in range(1, epochs + 1):

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        # ----------------------------------------------------
        # 一个 Epoch
        # ----------------------------------------------------

        for batch in train_loader:
            cloud = batch["cloud"].to(device)
            label = batch["label"].to(device)

            # =========================
            # Forward
            # =========================

            prediction = model(cloud)

            # =========================
            # Loss
            # =========================

            loss = criterion(
                prediction,
                label,
            )

            # =========================
            # 清空梯度
            # =========================

            optimizer.zero_grad()

            # =========================
            # Backward
            # =========================

            loss.backward()

            # =========================
            # 更新参数
            # =========================

            optimizer.step()

            # =========================
            # 统计 Loss
            # =========================

            current_batch_size = cloud.size(0)

            total_train_loss += (
                    loss.item()
                    * current_batch_size
            )

            total_train_samples += (
                current_batch_size
            )

        # ----------------------------------------------------
        # 平均 Train Loss
        # ----------------------------------------------------

        train_loss = (
                total_train_loss
                / total_train_samples
        )

        # ====================================================
        # Validation
        # ====================================================

        val_loss, val_psnr, val_ssim = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # ====================================================
        # 打印
        # ====================================================

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"| Train Loss: {train_loss:.6f} "
            f"| Val Loss: {val_loss:.6f} "
            f"| PSNR: {val_psnr:.4f} dB "
            f"| SSIM: {val_ssim:.4f}"
        )

        # ====================================================
        # 保存最佳模型
        # ====================================================

        if val_psnr > best_psnr:
            best_psnr = val_psnr

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_psnr": val_psnr,
                    "val_ssim": val_ssim,
                },
                best_model_path,
            )

            print(
                f"Best model saved! "
                f"PSNR = {val_psnr:.4f} dB"
            )


if __name__ == "__main__":
    main()