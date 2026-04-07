"""CIFAR-10数据集加载器模块

基于torchvision.datasets.CIFAR10构建，提供：
- 自定义数据预处理管道
- 可配置的数据增强策略
- 分布式训练支持
- 灵活的数据划分选项
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from ..data_transforms import (
    cifar10_train_transforms,
    cifar10_test_transforms,
    DataTransformFactory,
    TransformConfig,
)

logger = logging.getLogger(__name__)


class CIFAR10Dataset(Dataset):
    """CIFAR-10数据集包装类
    
    在torchvision的CIFAR-10基础上添加了：
    - 返回索引信息用于追踪样本
    - 支持自定义transforms管道
    - 可选的高级数据增强
    
    Attributes:
        root: 数据集根目录
        train: 是否为训练集模式
        transform: 图像变换函数
        target_transform: 标签变换函数
        download: 是否自动下载
        
    Example:
        >>> dataset = CIFAR10Dataset(
        ...     root='./data',
        ...     train=True,
        ...     transform=cifar10_train_transforms()
        ... )
        >>> image, label, idx = dataset[0]
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
        augmentation_strength: float = 0.5,
        use_advanced_augmentation: bool = False,
    ) -> None:
        """初始化CIFAR-10数据集
        
        Args:
            root: 数据集存储根目录
            train: True为训练集，False为测试集
            transform: 可选的自定义图像变换，如果为None则使用默认变换
            target_transform: 标签变换函数
            download: 如果数据不存在是否自动下载
            augmentation_strength: 增强强度（0.0-1.0），仅在transform为None时使用
            use_advanced_augmentation: 是否使用高级增强（CutOut等）
        """
        self.root = Path(root)
        self.train = train
        self.augmentation_strength = augmentation_strength
        
        if transform is None:
            if train:
                config = TransformConfig(
                    image_size=(32, 32),
                    augmentation_strength=augmentation_strength,
                    use_cutout=use_advanced_augmentation,
                    use_erasing=use_advanced_augmentation,
                )
                factory = DataTransformFactory(config)
                transform = factory.create_train_transforms()
            else:
                transform = cifar10_test_transforms()
        
        logger.info(f"加载CIFAR-10{'训练' if train else '测试'}集，根目录: {root}")
        
        try:
            self.dataset = datasets.CIFAR10(
                root=str(self.root),
                train=train,
                download=download,
                transform=transform,
                target_transform=target_transform,
            )
            logger.info(f"成功加载{len(self.dataset)}个样本")
        except Exception as e:
            logger.error(f"加载CIFAR-10数据集失败: {e}")
            raise
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """获取单个样本
        
        Args:
            index: 样本索引
            
        Returns:
            元组：(image_tensor, label, index)
            
        Raises:
            IndexError: 如果索引超出范围
        """
        if index >= len(self):
            raise IndexError(f"索引{index}超出范围，数据集大小为{len(self)}")
        
        image, label = self.dataset[index]
        return (image, label, index)
    
    @property
    def num_classes(self) -> int:
        """返回类别数量"""
        return 10
    
    @property
    def class_names(self) -> List[str]:
        """返回类别名称列表"""
        return [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    def get_class_distribution(self) -> Dict[int, int]:
        """计算每个类别的样本分布
        
        Returns:
            字典：{class_label: sample_count}
        """
        distribution = {}
        for _, label, _ in [self[i] for i in range(len(self))]:
            distribution[label] = distribution.get(label, 0) + 1
        return distribution


def get_cifar10_dataloaders(
    data_root: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1,
    augmentation_strength: float = 0.5,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    seed: int = 42,
    custom_train_transform: Optional[Callable] = None,
    custom_test_transform: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """创建CIFAR-10数据加载器集合
    
    创建训练、验证和测试集的DataLoader，支持自动划分验证集。
    
    Args:
        data_root: 数据集根目录路径
        batch_size: 批次大小，默认64
        num_workers: 数据加载工作进程数，默认4
        val_split: 从训练集中划分的比例作为验证集，默认0.1（10%）
        augmentation_strength: 训练集增强强度（0.0-1.0）
        pin_memory: 是否将数据固定在内存中（加速GPU传输）
        drop_last: 训练集是否丢弃最后不完整的批次
        distributed: 是否使用分布式训练采样器
        seed: 随机种子，确保可重复性
        custom_train_transform: 自定义训练集变换（覆盖默认）
        custom_test_transform: 自定义测试集变换（覆盖默认）
        
    Returns:
        包含DataLoader的字典：
        {
            'train': DataLoader,  # 训练集
            'val': DataLoader,    # 验证集（从训练集划分）
            'test': DataLoader    # 测试集
        }
        
    Raises:
        ValueError: 如果val_split不在(0, 1)范围内
        FileNotFoundError: 如果数据根目录不存在且无法创建
        
    Example:
        >>> loaders = get_cifar10_dataloaders(
        ...     data_root='./data/cifar10',
        ...     batch_size=128,
        ...     num_workers=8,
        ...     val_split=0.15
        ... )
        >>> for images, labels, indices in loaders['train']:
        ...     # 训练循环
        ...     pass
    """
    if not 0 < val_split < 1:
        raise ValueError(f"val_split必须在(0, 1)之间，当前值: {val_split}")
    
    data_path = Path(data_root)
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建数据目录: {data_path}")
    
    logger.info("=" * 60)
    logger.info("创建CIFAR-10数据加载器")
    logger.info(f"批次大小: {batch_size}, 工作进程: {num_workers}")
    logger.info(f"验证集比例: {val_split}, 增强强度: {augmentation_strength}")
    logger.info("=" * 60)
    
    full_train_dataset = CIFAR10Dataset(
        root=data_path,
        train=True,
        transform=custom_train_transform,
        augmentation_strength=augmentation_strength,
        download=True,
    )
    
    test_dataset = CIFAR10Dataset(
        root=data_path,
        train=False,
        transform=custom_test_transform,
        download=True,
    )
    
    n_total = len(full_train_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_train_dataset,
        [n_train, n_val],
        generator=generator,
    )
    
    logger.info(f"训练集: {n_val}样本")
    logger.info(f"测试集: {len(test_dataset)}样本")
    
    dataloaders = {}
    
    train_sampler = (
        DistributedSampler(train_subset, shuffle=True, seed=seed)
        if distributed else None
    )
    
    dataloaders['train'] = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=train_sampler,
        worker_init_fn=_worker_init_fn,
    )
    
    val_sampler = (
        DistributedSampler(val_subset, shuffle=False, seed=seed)
        if distributed else None
    )
    
    dataloaders['val'] = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
        worker_init_fn=_worker_init_fn,
    )
    
    dataloaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
    )
    
    for name, loader in dataloaders.items():
        logger.info(f"{name}集DataLoader已创建，共{len(loader)}个batch")
    
    return dataloaders


def _worker_init_fn(worker_id: int) -> None:
    """Worker初始化函数
    
    为每个DataLoader worker设置不同的随机种子，
    确保数据增强的随机性。
    
    Args:
        worker_id: Worker进程ID
    """
    import numpy as np
    import random
    
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def create_balanced_sampler(
    dataset: Dataset,
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> torch.utils.data.WeightedRandomSampler:
    """创建平衡采样器
    
    对于不平衡数据集，创建按类别权重采样的采样器。
    
    Args:
        dataset: 数据集对象
        num_samples: 每个epoch的采样数，如果为None则使用数据集大小
        replacement: 是否有放回采样
        
    Returns:
        WeightedRandomSampler实例
        
    Example:
        >>> sampler = create_balanced_sampler(dataset)
        >>> loader = DataLoader(dataset, batch_size=64, sampler=sampler)
    """
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label)
    
    class_counts = {}
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    weights = []
    for label in labels:
        weight = 1.0 / class_counts[label]
        weights.append(weight)
    
    weights_tensor = torch.DoubleTensor(weights)
    
    if num_samples is None:
        num_samples = len(dataset)
    
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=num_samples,
        replacement=replacement,
    )
    
    logger.info(f"创建平衡采样器，num_samples={num_samples}")
    return sampler


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("CIFAR-10数据集加载器演示")
    print("=" * 60)
    
    print("\n1. 测试数据集加载:")
    print("-" * 40)
    try:
        loaders = get_cifar10_dataloaders(
            data_root='./data/cifar10_demo',
            batch_size=32,
            num_workers=0,
            val_split=0.2,
            augmentation_strength=0.5,
        )
        
        print(f"\n✓ 成功创建数据加载器:")
        for name, loader in loaders.items():
            print(f"  - {name}: {len(loader)} batches")
        
        print("\n2. 测试数据迭代:")
        print("-" * 40)
        for i, (images, labels, indices) in enumerate(loaders['train']):
            print(f"Batch {i}: shape={images.shape}, labels={labels[:5]}, indices={indices[:5]}")
            if i >= 2:
                break
        
        print("\n3. 测试类别信息:")
        print("-" * 40)
        dataset = loaders['train'].dataset.dataset
        if hasattr(dataset, 'dataset'):
            base_dataset = dataset.dataset
        else:
            base_dataset = dataset
        if isinstance(base_dataset, CIFAR10Dataset):
            print(f"类别数量: {base_dataset.num_classes}")
            print(f"类别名称: {base_dataset.class_names}")
        
        print("\n✓ 所有测试通过!")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
