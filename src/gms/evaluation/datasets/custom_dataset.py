"""ImageNet子集和自定义数据集加载器模块

提供：
- ImageNetSubsetDataset: 加载ImageNet的子集（部分类别）
- CustomImageDataset: 从文件夹结构加载数据
- 通用接口用于自定义图像分类数据集
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from PIL import Image

from ..data_transforms import (
    imagenet_train_transforms,
    imagenet_test_transforms,
    DataTransformFactory,
    TransformConfig,
)

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}


class ImageNetSubsetDataset(Dataset):
    """ImageNet子集数据集类
    
    支持加载ILSVRC2012格式的ImageNet子集，可以选择性加载部分类别。
    
    Attributes:
        root: 数据集根目录（应包含train/val/test文件夹）
        subset_classes: 使用的类别列表或None表示全部
        split: 数据划分类型 ('train', 'val', 'test')
        transform: 图像变换函数
        num_classes: 类别数量
        
    Example:
        >>> dataset = ImageNetSubsetDataset(
        ...     root='./data/imagenet_subset',
        ...     split='train',
        ...     subset_classes=['cat', 'dog'],
        ...     transform=imagenet_train_transforms()
        ... )
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        subset_classes: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (224, 224),
        augmentation_strength: float = 0.7,
    ) -> None:
        """初始化ImageNet子集数据集
        
        Args:
            root: 数据集根目录，应包含train/val/test子目录
            split: 数据划分类型，支持 'train', 'val', 'test'
            transform: 图像变换函数
            target_transform: 标签变换函数
            subset_classes: 要使用的类别名称列表，如果为None则使用全部类别
            image_size: 目标图像尺寸
            augmentation_strength: 增强强度（仅在transform为None时使用）
            
        Raises:
            ValueError: 如果split不被支持或根目录不存在
            FileNotFoundError: 如果指定的split目录不存在
        """
        self.root = Path(root)
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"不支持的split类型: {split}。支持: train, val, test")
        
        if not self.root.exists():
            raise FileNotFoundError(f"数据集根目录不存在: {root}")
        
        split_dir = self.root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split目录不存在: {split_dir}")
        
        if transform is None:
            if self.split == 'train':
                config = TransformConfig(
                    image_size=image_size,
                    augmentation_strength=augmentation_strength,
                )
                factory = DataTransformFactory(config)
                self.transform = factory.create_train_transforms()
            else:
                self.transform = imagenet_test_transforms(image_size=image_size)
        
        self.samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        if subset_classes is not None:
            available_classes = [
                d.name for d in sorted(split_dir.iterdir())
                if d.is_dir() and d.name not in {'.DS_Store'}
            ]
            invalid_classes = [c for c in subset_classes if c not in available_classes]
            if invalid_classes:
                logger.warning(f"以下类别不存在将被忽略: {invalid_classes}")
            self.classes = [c for c in subset_classes if c in available_classes]
        else:
            self.classes = sorted([
                d.name for d in split_dir.iterdir()
                if d.is_dir() and d.name not in {'.DS_Store'}
            ])
        
        if len(self.classes) == 0:
            raise ValueError(f"在{split_dir}中未找到任何类别")
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            
            label = self.class_to_idx[class_name]
            image_files = []
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.extend(class_dir.glob(f'*{ext}'))
                image_files.extend(class_dir.glob(f'*{ext.upper()}'))
            
            for img_path in image_files:
                self.samples.append((img_path, label))
        
        logger.info(
            f"加载ImageNet-{self.split}子集: "
            f"{len(self.classes)}个类别, {len(self.samples)}张图像"
        )
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """获取单个样本
        
        Args:
            index: 样本索引
            
        Returns:
            元组：(image_tensor, label, index)
            
        Raises:
            IndexError: 索引超出范围
            IOError: 图像文件损坏或无法读取
        """
        if index >= len(self):
            raise IndexError(f"索引{index}超出范围，数据集大小为{len(self)}")
        
        img_path, label = self.samples[index]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"无法读取图像 {img_path}: {e}")
            raise IOError(f"无法读取图像文件: {img_path}") from e
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return (image, label, index)
    
    @property
    def num_classes(self) -> int:
        """返回类别数量"""
        return len(self.classes)
    
    @property
    def class_names(self) -> List[str]:
        """返回类别名称列表"""
        return self.classes.copy()


class CustomImageDataset(Dataset):
    """通用自定义图像数据集
    
    从标准文件夹结构（每个类别一个文件夹）加载图像数据，
    自动识别类别并支持灵活的配置选项。
    
    Attributes:
        data_dir: 数据根目录
        classes: 类别名称列表
        samples: 样本路径和标签列表
        
    Example:
        >>> dataset = CustomImageDataset(
        ...     data_dir='./my_dataset',
        ...     split='train',
        ...     train_ratio=0.8,
        ...     val_ratio=0.1,
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        image_size: Tuple[int, int] = (224, 224),
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        augmentation_strength: float = 0.5,
    ) -> None:
        """初始化自定义图像数据集
        
        Args:
            data_dir: 数据根目录，包含按类别组织的子文件夹
            split: 数据划分 ('train', 'val', 'test')
            transform: 图像变换函数
            target_transform: 标签变换函数
            train_ratio: 训练集比例（默认0.8）
            val_ratio: 验证集比例（默认0.1）
            test_ratio: 测试集比例（默认0.1）
            seed: 随机种子确保可重复划分
            image_size: 目标图像尺寸
            mean: 标准化均值，如果为None使用ImageNet默认值
            std: 标准化标准差，如果为None使用ImageNet默认值
            augmentation_strength: 增强强度
            
        Raises:
            ValueError: 如果比例之和不等于1.0或split不被支持
            FileNotFoundError: 如果数据目录不存在
        """
        self.data_dir = Path(data_dir)
        self.split = split.lower()
        self.transform = transform
        self.target_transform = target_transform
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"训练/验证/测试比例之和必须等于1.0，"
                f"当前: {train_ratio + val_ratio + test_ratio}"
            )
        
        if self.split not in ['train', 'val', 'test']:
            raise ValueError(f"不支持的split: {split}。支持: train, val, test")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        _mean = mean or TransformConfig().mean
        _std = std or TransformConfig().std
        
        if transform is None:
            config = TransformConfig(
                image_size=image_size,
                mean=_mean,
                std=_std,
                augmentation_strength=augmentation_strength if self.split == 'train' else 0.0,
            )
            factory = DataTransformFactory(config)
            if self.split == 'train':
                self.transform = factory.create_train_transforms()
            else:
                self.transform = factory.create_test_transforms()
        
        all_samples: List[Tuple[Path, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        
        class_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith('.')
        ])
        
        if len(class_dirs) == 0:
            raise ValueError(f"在{self.data_dir}中未找到任何类别文件夹")
        
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            image_files = []
            for ext in SUPPORTED_IMAGE_EXTENSIONS:
                image_files.extend(class_dir.glob(f'*{ext}'))
                image_files.extend(class_dir.glob(f'*{ext.upper()}'))
            
            for img_path in sorted(image_files):
                all_samples.append((img_path, label))
        
        if len(all_samples) == 0:
            raise ValueError(f"未找到任何图像文件在{self.data_dir}中")
        
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator).tolist()
        
        if self.split == 'train':
            selected_indices = indices[:n_train]
        elif self.split == 'val':
            selected_indices = indices[n_train:n_train + n_val]
        else:
            selected_indices = indices[n_train + n_val:]
        
        self.samples = [all_samples[i] for i in selected_indices]
        
        logger.info(
            f"加载CustomDataset-{self.split}: "
            f"{len(self.classes)}个类别, {len(self.samples)}张图像 "
            f"(总计{n_total}张)"
        )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        if index >= len(self):
            raise IndexError(f"索引{index}超出范围，数据集大小为{len(self)}")
        
        img_path, label = self.samples[index]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"无法读取图像 {img_path}: {e}")
            raise IOError(f"无法读取图像文件: {img_path}") from e
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return (image, label, index)
    
    @property
    def num_classes(self) -> int:
        return len(self.classes)
    
    @property
    def class_names(self) -> List[str]:
        return self.classes.copy()


def get_imagenet_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    subset_classes: Optional[List[str]] = None,
    image_size: Tuple[int, int] = (224, 224),
    augmentation_strength: float = 0.7,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> Dict[str, DataLoader]:
    """创建ImageNet子集数据加载器集合
    
    Args:
        data_root: ImageNet数据集根目录
        batch_size: 批次大小
        num_workers: 工作进程数
        subset_classes: 要使用的类别子集，None表示全部
        image_size: 目标图像尺寸
        augmentation_strength: 训练增强强度
        pin_memory: 是否固定内存
        drop_last: 训练集是否丢弃最后不完整批次
        
    Returns:
        包含'train', 'val', 'test'的DataLoader字典
        
    Example:
        >>> loaders = get_imagenet_dataloaders(
        ...     data_root='./data/imagenet',
        ...     subset_classes=['n01440764', 'n01443537'],  # 使用WNID
        ...     batch_size=64
        ... )
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = ImageNetSubsetDataset(
                root=data_root,
                split=split,
                subset_classes=subset_classes,
                image_size=image_size,
                augmentation_strength=augmentation_strength,
            )
            
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last and (split == 'train'),
            )
            logger.info(f"创建ImageNet-{split} DataLoader: {len(dataloaders[split])} batches")
            
        except FileNotFoundError as e:
            logger.warning(f"跳过{split}集: {e}")
    
    return dataloaders


def get_custom_dataloader(
    data_dir: Union[str, Path],
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    """创建自定义数据集的DataLoader
    
    从文件夹结构加载数据并返回指定划分的DataLoader。
    
    Args:
        data_dir: 数据目录路径
        split: 返回哪个划分的数据 ('train', 'val', 'test')
        batch_size: 批次大小
        num_workers: 工作进程数
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
        **kwargs: 传递给CustomImageDataset的其他参数
        
    Returns:
        指定划分的DataLoader
        
    Example:
        >>> loader = get_custom_dataloader(
        ...     data_dir='./my_images',
        ...     split='train',
        ...     batch_size=16,
        ...     image_size=(128, 128),
        ...     augmentation_strength=0.3
        ... )
    """
    dataset = CustomImageDataset(
        data_dir=data_dir,
        split=split,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        **kwargs,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    logger.info(
        f"创建CustomDataLoader ({split}): "
        f"{len(dataset)}样本, {len(loader)}batches"
    )
    
    return loader


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("ImageNet子集和自定义数据集演示")
    print("=" * 60)
    
    print("\n注意：此演示需要实际的数据集目录")
    print("请准备以下格式的数据集进行测试:")
    print("-" * 60)
    print("""
    目录结构示例:
    
    data/
    ├── imagenet_subset/
    │   ├── train/
    │   │   ├── cat/
    │   │   │   ├── cat_001.jpg
    │   │   │   └── ...
    │   │   └── dog/
    │   │       ├── dog_001.jpg
    │   │       └── ...
    │   ├── val/
    │   └── test/
    └── custom_dataset/
        ├── class_a/
        │   ├── img_001.png
        │   └── ...
        └── class_b/
            ├── img_001.png
            └── ...
    """)
    
    print("\n✓ 模块已成功导入!")
    print(f"\n支持的图像格式: {SUPPORTED_IMAGE_EXTENSIONS}")
