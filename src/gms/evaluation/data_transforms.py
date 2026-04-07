"""数据预处理和增强管道模块

提供标准化的图像transforms pipeline，支持不同数据集（CIFAR-10、ImageNet等）
和可配置的数据增强策略。包含预定义的transforms集合和高级增强技术。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Transform配置类
    
    存储所有与数据增强相关的配置参数。
    
    Attributes:
        image_size: 目标图像尺寸（高度，宽度）
        mean: 标准化均值（RGB通道）
        std: 标准化标准差（RGB通道）
        augmentation_strength: 增强强度（0.0-1.0）
        use_color_jitter: 是否使用颜色抖动
        use_random_crop: 是否使用随机裁剪
        use_horizontal_flip: 是否使用水平翻转
        use_randaugment: 是否使用RandAugment
        use_autoaugment: 是否使用AutoAugment
        use_cutout: 是否使用CutOut
        use_erasing: 是否使用RandomErasing
        cutout_size: CutOut的尺寸比例
        erasing_probability: RandomErasing的概率
        
    Example:
        >>> config = TransformConfig(
        ...     image_size=(32, 32),
        ...     mean=(0.4914, 0.4822, 0.4465),
        ...     std=(0.2470, 0.2435, 0.2616),
        ...     augmentation_strength=0.5
        ... )
    """
    
    image_size: Tuple[int, int] = (224, 224)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    augmentation_strength: float = 0.5
    use_color_jitter: bool = True
    use_random_crop: bool = True
    use_horizontal_flip: bool = True
    use_randaugment: bool = False
    use_autoaugment: bool = False
    use_cutout: bool = False
    use_erasing: bool = False
    cutout_size: float = 0.25
    erasing_probability: float = 0.5


class DataTransformFactory:
    """数据变换工厂类
    
    创建标准化的transforms pipeline，支持不同图像尺寸、增强强度和自定义配置。
    提供工厂方法用于快速创建常见数据集的标准transforms。
    
    Attributes:
        config: TransformConfig实例，存储变换配置
        
    Example:
        >>> factory = DataTransformFactory(TransformConfig(image_size=(32, 32)))
        >>> train_transform = factory.create_train_transforms()
        >>> test_transform = factory.create_test_transforms()
    """
    
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2470, 0.2435, 0.2616)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    
    def __init__(self, config: Optional[TransformConfig] = None) -> None:
        """初始化变换工厂
        
        Args:
            config: 变换配置对象，如果为None则使用默认配置
        """
        self.config = config or TransformConfig()
        logger.info(f"初始化DataTransformFactory，图像尺寸: {self.config.image_size}")
    
    def create_train_transforms(self) -> transforms.Compose:
        """创建训练集变换pipeline
        
        根据配置创建包含各种增强的训练变换。
        
        Returns:
            训练集变换组合
        """
        transform_list = []
        h, w = self.config.image_size
        
        if self.config.use_randaugment:
            try:
                n = max(1, int(2 * self.config.augmentation_strength))
                m = max(5, int(15 * self.config.augmentation_strength))
                transform_list.append(transforms.RandAugment(n=n, m=m))
                logger.debug(f"添加RandAugment: n={n}, m={m}")
            except AttributeError:
                logger.warning("当前PyTorch版本不支持RandAugment")
        
        if self.config.use_autoaugment:
            try:
                transform_list.append(transforms.AutoAugment(
                    policy=transforms.AutoAugmentPolicy.IMAGENET
                ))
                logger.debug("添加AutoAugment (ImageNet策略)")
            except AttributeError:
                logger.warning("当前PyTorch版本不支持AutoAugment")
        
        transform_list.append(transforms.Resize((h + 8, w + 8)))
        
        if self.config.use_random_crop:
            crop_padding = int(4 * self.config.augmentation_strength)
            transform_list.append(transforms.RandomCrop(
                size=(h, w),
                padding=crop_padding
            ))
            logger.debug(f"添加RandomCrop, padding={crop_padding}")
        
        if self.config.use_horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            logger.debug("添加RandomHorizontalFlip")
        
        if self.config.use_color_jitter and self.config.augmentation_strength > 0:
            brightness = 0.2 * self.config.augmentation_strength
            contrast = 0.2 * self.config.augmentation_strength
            saturation = 0.1 * self.config.augmentation_strength
            hue = 0.05 * self.config.augmentation_strength
            transform_list.append(transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ))
            logger.debug(f"添加ColorJitter: brightness={brightness}, contrast={contrast}")
        
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(
            mean=self.config.mean,
            std=self.config.std
        ))
        
        if self.config.use_cutout:
            transform_list.append(CutOut(
                size_ratio=self.config.cutout_size,
                probability=0.5 * self.config.augmentation_strength
            ))
            logger.debug("添加CutOut")
        
        if self.config.use_erasing:
            p = min(1.0, self.config.erasing_probability * self.config.augmentation_strength)
            transform_list.append(transforms.RandomErasing(p=p))
            logger.debug(f"添加RandomErasing, p={p}")
        
        result = transforms.Compose(transform_list)
        logger.info(f"创建训练transform pipeline，共{len(transform_list)}个操作")
        return result
    
    def create_test_transforms(self) -> transforms.Compose:
        """创建测试/验证集变换pipeline
        
        创建不包含增强的标准化测试变换。
        
        Returns:
            测试集变换组合
        """
        h, w = self.config.image_size
        transform_list = [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std),
        ]
        
        result = transforms.Compose(transform_list)
        logger.info(f"创建测试transform pipeline，共{len(transform_list)}个操作")
        return result
    
    def create_val_transforms(self) -> transforms.Compose:
        """创建验证集变换pipeline
        
        验证集通常使用与测试集相同的变换（无数据增强）。
        
        Returns:
            验证集变换组合
        """
        return self.create_test_transforms()


class CutOut(object):
    """CutOut数据增强
    
    随机遮挡图像的一个矩形区域，提高模型对局部特征的鲁棒性。
    
    Attributes:
        size_ratio: 遮挡区域相对于图像边长的比例
        probability: 应用CutOut的概率
        fill_value: 遮挡区域的填充值
        
    Example:
        >>> cutout = CutOut(size_ratio=0.25, probability=0.5)
        >>> transformed_image = cutout(image_tensor)
    """
    
    def __init__(
        self,
        size_ratio: float = 0.25,
        probability: float = 0.5,
        fill_value: float = 0.0
    ) -> None:
        """初始化CutOut增强
        
        Args:
            size_ratio: 遮挡区域相对于图像最小边的比例（0-1之间）
            probability: 应用此增强的概率
            fill_value: 遮挡区域的填充值
        """
        self.size_ratio = size_ratio
        self.probability = probability
        self.fill_value = fill_value
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """应用CutOut增强
        
        Args:
            img: 输入图像张量，形状为 (C, H, W)
            
        Returns:
            可能被遮挡的图像张量
        """
        if torch.rand(1).item() > self.probability:
            return img
        
        _, h, w = img.shape
        cut_h = int(h * self.size_ratio)
        cut_w = int(w * self.size_ratio)
        
        top = torch.randint(0, h - cut_h + 1, (1,)).item()
        left = torch.randint(0, w - cut_w + 1, (1,)).item()
        
        img[:, top:top + cut_h, left:left + cut_w] = self.fill_value
        return img
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size_ratio={self.size_ratio}, probability={self.probability})"


def cifar10_train_transforms(
    image_size: Tuple[int, int] = (32, 32),
    augmentation_strength: float = 0.5
) -> transforms.Compose:
    """获取CIFAR-10训练集标准增强transforms
    
    为CIFAR-10训练集创建优化的数据增强pipeline，
    包含RandomCrop、RandomHorizontalFlip和ColorJitter等。
    
    Args:
        image_size: 目标图像尺寸，默认为(32, 32)
        augmentation_strength: 增强强度（0.0-1.0），控制增强程度
        
    Returns:
        训练集transforms组合
        
    Example:
        >>> train_tfms = cifar10_train_transforms(augmentation_strength=0.7)
        >>> transformed_img = train_tfms(original_pil_image)
    """
    config = TransformConfig(
        image_size=image_size,
        mean=DataTransformFactory.CIFAR10_MEAN,
        std=DataTransformFactory.CIFAR10_STD,
        augmentation_strength=augmentation_strength,
        use_color_jitter=True,
        use_random_crop=True,
        use_horizontal_flip=True,
    )
    factory = DataTransformFactory(config)
    return factory.create_train_transforms()


def cifar10_test_transforms(
    image_size: Tuple[int, int] = (32, 32)
) -> transforms.Compose:
    """获取CIFAR-10测试集标准化transforms
    
    为CIFAR-10测试集创建标准化变换，仅包含Resize、ToTensor和Normalize。
    
    Args:
        image_size: 目标图像尺寸，默认为(32, 32)
        
    Returns:
        测试集transforms组合
        
    Example:
        >>> test_tfms = cifar10_test_transforms()
        >>> normalized_img = test_tfms(test_pil_image)
    """
    config = TransformConfig(
        image_size=image_size,
        mean=DataTransformFactory.CIFAR10_MEAN,
        std=DataTransformFactory.CIFAR10_STD,
    )
    factory = DataTransformFactory(config)
    return factory.create_test_transforms()


def imagenet_train_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_strength: float = 0.7
) -> transforms.Compose:
    """获取ImageNet训练集增强transforms
    
    为ImageNet训练集创建标准的数据增强pipeline，
    使用更强的增强以适应大规模数据集。
    
    Args:
        image_size: 目标图像尺寸，默认为(224, 224)
        augmentation_strength: 增强强度（0.0-1.0），默认为0.7
        
    Returns:
        训练集transforms组合
        
    Example:
        >>> train_tfms = imagenet_train_transforms(augmentation_strength=0.8)
    """
    config = TransformConfig(
        image_size=image_size,
        mean=DataTransformFactory.IMAGENET_MEAN,
        std=DataTransformFactory.IMAGENET_STD,
        augmentation_strength=augmentation_strength,
        use_color_jitter=True,
        use_random_crop=True,
        use_horizontal_flip=True,
        use_randaugment=False,
        use_autoaugment=False,
        use_erasing=True,
        erasing_probability=0.3,
    )
    factory = DataTransformFactory(config)
    return factory.create_train_transforms()


def imagenet_test_transforms(
    image_size: Tuple[int, int] = (224, 224)
) -> transforms.Compose:
    """获取ImageNet测试集标准化transforms
    
    为ImageNet测试集创建标准化变换。
    
    Args:
        image_size: 目标图像尺寸，默认为(224, 224)
        
    Returns:
        测试集transforms组合
    """
    config = TransformConfig(
        image_size=image_size,
        mean=DataTransformFactory.IMAGENET_MEAN,
        std=DataTransformFactory.IMAGENET_STD,
    )
    factory = DataTransformFactory(config)
    return factory.create_test_transforms()


def custom_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """根据配置字典创建自定义transforms
    
    从配置字典动态构建transforms pipeline，提供最大的灵活性。
    
    Args:
        config: 配置字典，支持以下键：
            - 'image_size': tuple(int, int), 图像尺寸
            - 'mean': tuple(float, float, float), 标准化均值
            - 'std': tuple(float, float, float), 标准化方差
            - 'train': bool, 是否为训练模式
            - 'use_color_jitter': bool, 颜色抖动
            - 'brightness': float, 亮度变化范围
            - 'contrast': float, 对比度变化范围
            - 'horizontal_flip_prob': float, 水平翻转概率
            - 'rotation_degrees': float, 旋转角度范围
            - 'random_crop_padding': int, 随机裁剪padding
            
    Returns:
        自定义transforms组合
        
    Raises:
        ValueError: 如果缺少必要的配置项
        
    Example:
        >>> config = {
        ...     'image_size': (64, 64),
        ...     'mean': (0.5, 0.5, 0.5),
        ...     'std': (0.5, 0.5, 0.5),
        ...     'train': True,
        ...     'use_color_jitter': True,
        ...     'rotation_degrees': 15
        ... }
        >>> tfms = custom_transforms(config)
    """
    required_keys = ['image_size', 'mean', 'std']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置中缺少必要字段: {key}")
    
    is_train = config.get('train', False)
    image_size = config['image_size']
    mean = config['mean']
    std = config['std']
    
    transform_list = []
    
    if is_train:
        if config.get('rotation_degrees', 0) > 0:
            transform_list.append(transforms.RandomRotation(
                degrees=config['rotation_degrees']
            ))
        
        resize_with_padding = (
            image_size[0] + config.get('random_crop_padding', 4),
            image_size[1] + config.get('random_crop_padding', 4)
        )
        transform_list.append(transforms.Resize(resize_with_padding))
        transform_list.append(transforms.RandomCrop(size=image_size))
        
        flip_prob = config.get('horizontal_flip_prob', 0.5)
        if flip_prob > 0:
            transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
        
        if config.get('use_color_jitter', False):
            brightness = config.get('brightness', 0.2)
            contrast = config.get('contrast', 0.2)
            saturation = config.get('saturation', 0.1)
            hue = config.get('hue', 0.05)
            transform_list.append(transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue
            ))
    else:
        transform_list.append(transforms.Resize(image_size))
    
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    result = transforms.Compose(transform_list)
    mode = "训练" if is_train else "测试"
    logger.info(f"从配置创建自定义transform，{mode}模式，"
                f"共{len(transform_list)}个操作")
    return result


def get_normalization_stats(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """获取常用数据集的标准化统计量
    
    返回指定数据集的标准均值和标准差。
    
    Args:
        dataset_name: 数据集名称，支持：
            - 'cifar10': CIFAR-10数据集
            - 'imagenet': ImageNet数据集
            - 'cifar100': CIFAR-100数据集
            - 'mnist': MNIST数据集
            
    Returns:
        元组：(mean, std)，每个都是3元组或1元组
        
    Raises:
        ValueError: 如果数据集名称不被支持
        
    Example:
        >>> mean, std = get_normalization_stats('cifar10')
        >>> print(f"CIFAR-10 Mean: {mean}, Std: {std}")
    """
    stats_map = {
        'cifar10': (DataTransformFactory.CIFAR10_MEAN, DataTransformFactory.CIFAR10_STD),
        'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        'imagenet': (DataTransformFactory.IMAGENET_MEAN, DataTransformFactory.IMAGENET_STD),
        'mnist': ((0.1307,), (0.3081,)),
    }
    
    dataset_name_lower = dataset_name.lower().strip()
    if dataset_name_lower not in stats_map:
        supported = ', '.join(stats_map.keys())
        raise ValueError(f"不支持的dataset_name: {dataset_name}。支持的数据集: {supported}")
    
    return stats_map[dataset_name_lower]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    print("=" * 60)
    print("数据预处理和增强管道模块演示")
    print("=" * 60)
    
    print("\n1. CIFAR-10 训练集 Transforms:")
    print("-" * 40)
    train_tfms = cifar10_train_transforms(augmentation_strength=0.7)
    print(train_tfms)
    
    print("\n2. CIFAR-10 测试集 Transforms:")
    print("-" * 40)
    test_tfms = cifar10_test_transforms()
    print(test_tfms)
    
    print("\n3. ImageNet 训练集 Transforms:")
    print("-" * 40)
    imagenet_train = imagenet_train_transforms(augmentation_strength=0.8)
    print(imagenet_train)
    
    print("\n4. 自定义配置 Transforms:")
    print("-" * 40)
    custom_config = {
        'image_size': (128, 128),
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        'train': True,
        'use_color_jitter': True,
        'rotation_degrees': 20,
        'horizontal_flip_prob': 0.3,
    }
    custom_tfms = custom_transforms(custom_config)
    print(custom_tfms)
    
    print("\n5. 获取标准化统计量:")
    print("-" * 40)
    for ds in ['cifar10', 'imagenet', 'cifar100']:
        mean, std = get_normalization_stats(ds)
        print(f"{ds.upper()}: mean={mean}, std={std}")
