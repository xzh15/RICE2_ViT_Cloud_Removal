from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset

from .rice2_dataset import RICE2Dataset


def create_dataloaders(
    root_dir: str,
    image_size: int = 256,
    batch_size: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建 RICE2 的 Train / Validation / Test DataLoader。
    参数
    root_dir:RICE2 数据集根目录。
    image_size:输入图像调整后的尺寸。例如 256 表示 256×256。
    batch_size:每个 batch 中包含多少张图像。
    train_ratio:训练集比例。
    val_ratio:验证集比例。
    test_ratio:测试集比例。
    seed:随机种子，用于保证数据划分可复现。
    num_workers:DataLoader 使用的子进程数量。Windows 初学阶段建议先使用 0。
    """
    # 1. 检查比例
    total_ratio = train_ratio + val_ratio + test_ratio

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio 必须等于 1。"
        )

    # 2. 创建完整 Dataset
    full_dataset = RICE2Dataset(
        root_dir=root_dir,
        image_size=image_size,
        use_mask=True,
    )
    total_size = len(full_dataset)
    print(f"总样本数量：{total_size}")

    # 3. 根据比例计算数量
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    # 剩余样本全部放到 test
    test_size = total_size - train_size - val_size

    print(f"训练集数量：{train_size}")
    print(f"验证集数量：{val_size}")
    print(f"测试集数量：{test_size}")

    # 4. 创建固定随机生成器
    generator = torch.Generator()
    generator.manual_seed(seed)

    # 5. 随机划分
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    # 6. 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,)

    return train_loader, val_loader, test_loader