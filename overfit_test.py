import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset

from datasets import RICE2Dataset
from models import ViTRestoration


def main():

    # =====================================================
    # 1. Device
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # =====================================================
    # 2. Dataset
    # =====================================================

    dataset = RICE2Dataset(
        root_dir="./data/RICE2",
        image_size=256,
        use_mask=True,
    )

    # =====================================================
    # 3. 只取4张
    # =====================================================

    tiny_dataset = Subset(
        dataset,
        [0, 1, 2, 3],
    )

    print(
        "Tiny dataset size:",
        len(tiny_dataset)
    )

    # =====================================================
    # 4. DataLoader
    # =====================================================

    loader = DataLoader(
        dataset=tiny_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # =====================================================
    # 5. 取出这4张
    # =====================================================

    batch = next(iter(loader))

    cloud = batch["cloud"].to(device)

    label = batch["label"].to(device)

    print("Cloud shape :", cloud.shape)
    print("Label shape :", label.shape)

    print(
        "Cloud range:",
        cloud.min().item(),
        "~",
        cloud.max().item(),
    )

    print(
        "Label range:",
        label.min().item(),
        "~",
        label.max().item(),
    )

    print(
        "Filenames:",
        batch["filename"],
    )

    # =====================================================
    # 6. Model
    # =====================================================

    model = ViTRestoration(
        img_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    )

    model = model.to(device)

    # =====================================================
    # 7. Loss
    # =====================================================

    criterion = nn.MSELoss()

    # =====================================================
    # 8. Optimizer
    # =====================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )

    # =====================================================
    # 9. 训练
    # =====================================================

    iterations = 3000

    for iteration in range(1, iterations + 1):

        model.train()

        # ---------------------------------------------
        # Forward
        # ---------------------------------------------

        prediction = model(cloud)

        # ---------------------------------------------
        # Loss
        # ---------------------------------------------

        loss = criterion(
            prediction,
            label,
        )

        # ---------------------------------------------
        # Backward
        # ---------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ---------------------------------------------
        # 打印
        # ---------------------------------------------

        if (
            iteration == 1
            or iteration % 100 == 0
        ):

            with torch.no_grad():

                mse = torch.mean(
                    (prediction - label) ** 2
                )

                psnr = (
                    10.0
                    * torch.log10(
                        1.0
                        / mse.clamp_min(1e-10)
                    )
                )

            print(
                f"Iteration [{iteration:4d}/"
                f"{iterations}] "
                f"Loss: {loss.item():.8f} "
                f"PSNR: {psnr.item():.4f} dB"
            )


if __name__ == "__main__":
    main()