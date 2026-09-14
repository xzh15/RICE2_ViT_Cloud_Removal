from pathlib import Path
from typing import Optional, Callable

from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class RICE2Dataset(Dataset):
    def __init__(self,root_dir: str,image_size: Optional[int] = 256,use_mask: bool = True,):
        super().__init__()
        # 根目录 data/RICE2
        self.root_dir = Path(root_dir)
        # data/RICE2/cloud
        self.cloud_dir = self.root_dir / "cloud"
        # data/RICE2/label
        self.label_dir = self.root_dir / "label"
        # data/RICE2/mask
        self.mask_dir = self.root_dir / "mask"
        self.image_size = image_size
        self.use_mask = use_mask

        # 检查目录
        if not self.cloud_dir.exists():
            raise FileNotFoundError(f"找不到 cloud 文件夹：{self.cloud_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"找不到 label 文件夹：{self.label_dir}")
        if self.use_mask and not self.mask_dir.exists():
            raise FileNotFoundError(f"找不到 mask 文件夹：{self.mask_dir}")

        # self.cloud_dir.glob("*.png")获取 cloud 中所有 png 文件
        # p.name 取文件名
        # lambda匿名函数，输入x返回x的stem（文件名）
        # Dataset 只从 cloud 获取文件名，配对数据集，确保一一对应。filenames相当于编号，第 1 张图片对应的 label 和 mask
        self.filenames = sorted([p.name for p in self.cloud_dir.glob("*.png")], key=lambda x: int(Path(x).stem))

        if len(self.filenames) == 0:
            raise RuntimeError(f"cloud 文件夹中没有找到 PNG 文件：{self.cloud_dir}")

        # 检查 cloud 和 label 是否一一对应
        for filename in self.filenames:
            cloud_path = self.cloud_dir / filename
            label_path = self.label_dir / filename
            if not label_path.exists():
                raise FileNotFoundError(f"找不到对应 label：{label_path}")
            if self.use_mask:
                mask_path = self.mask_dir / filename
                if not mask_path.exists():
                    raise FileNotFoundError(f"找不到对应 mask：{mask_path}")
        print(f"RICE2 数据集读取成功，共 {len(self.filenames)} 个样本")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        cloud_path = self.cloud_dir / filename
        label_path = self.label_dir / filename
        # 1. 读取 cloud，转换成三通道
        cloud = Image.open(cloud_path).convert("RGB")
        # 2. 读取 label
        label = Image.open(label_path).convert("RGB")
        # 3. 读取 mask
        if self.use_mask:
            mask_path = self.mask_dir / filename
            mask = Image.open(mask_path).convert("L")
        # 4. 尺寸处理
        if self.image_size is not None:
            cloud = TF.resize(cloud,[self.image_size, self.image_size])
            label = TF.resize(label,[self.image_size, self.image_size])
            if self.use_mask:
                mask = TF.resize(mask,[self.image_size, self.image_size])
        # 5. 转 Tensor
        cloud = TF.to_tensor(cloud)
        label = TF.to_tensor(label)
        if self.use_mask:
            mask = TF.to_tensor(mask)
        # 6. 返回
        if self.use_mask:
            return {
                "cloud": cloud,
                "label": label,
                "mask": mask,
                "filename": filename,
            }

        return {
            "cloud": cloud,
            "label": label,
            "filename": filename,
        }

