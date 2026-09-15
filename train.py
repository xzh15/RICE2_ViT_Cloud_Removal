import torch
from torch.utils.data import DataLoader
from datasets import RICE2Dataset, create_dataloaders
from models import ViTRestoration
from losses import ReconstructionLoss

def main():
    # 1. Device
    device = torch.device("cuda"if torch.cuda.is_available()else "cpu")
    print("Device:", device)

    # 2. DataLoader
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir="./data/RICE2",
        image_size=256,
        batch_size=4,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        num_workers=0,
    )

    # 3. Model
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

    # 4. Loss
    criterion = ReconstructionLoss()

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4,)

    # 6. Training
    epochs = 20
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            cloud = batch["cloud"].to(device)
            label = batch["label"].to(device)

            # Forward
            prediction = model(cloud)

            # Loss
            loss = criterion(prediction,label)

            # 清空梯度
            optimizer.zero_grad()

            # Backward
            loss.backward()

            # 更新参数
            optimizer.step()
        print(f"Epoch [{epoch + 1}/{epochs}] "f"Loss: {loss.item():.6f}")


if __name__ == "__main__":
    main()

