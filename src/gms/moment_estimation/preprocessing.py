"""图像预处理模块

提供标准化的图像预处理管道，支持多种图像格式、
尺寸调整、归一化和数据增强功能。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# 支持的图像格式
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"}

# ImageNet标准化参数（用于预训练模型）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ImagePreprocessor:
    """图像预处理器
    
    提供完整的图像预处理管道，包括：
    - 图像加载和格式转换
    - 尺寸调整和裁剪
    - 归一化处理
    - 数据增强（可选）
    - 批量处理支持
    
    使用torchvision.transforms构建灵活的预处理pipeline。
    
    Attributes:
        image_size: 目标图像尺寸 (height, width)
        normalize: 是否进行归一化
        augmentation: 是否启用数据增强
        transform: torchvision.transforms.Compose对象
        
    Example:
        >>> preprocessor = ImagePreprocessor(
        ...     image_size=(224, 224),
        ...     normalize=True,
        ...     augmentation=False
        ... )
        >>> processed = preprocessor(image_path_or_tensor)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        augmentation: bool = False,
        augmentation_config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """初始化图像预处理器
        
        Args:
            image_size: 目标图像尺寸 (height, width)，默认(224, 224)
            normalize: 是否对图像进行归一化，默认True
            mean: 归一化均值，默认使用ImageNet均值
            std: 归一化标准差，默认使用ImageNet标准差
            augmentation: 是否启用数据增强，默认False
            augmentation_config: 数据增强配置字典
            device: 输出张量的目标设备
        """
        self.image_size = image_size
        self.normalize = normalize
        self.augmentation = augmentation
        self.device = device or torch.device("cpu")

        # 归一化参数
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD

        # 构建变换管道
        self.transform = self._build_transform(augmentation, augmentation_config)

        logger.info(
            f"初始化图像预处理器: "
            f"尺寸={image_size}, 归一化={normalize}, "
            f"增强={augmentation}"
        )

    def _build_transform(
        self,
        augmentation: bool,
        config: Optional[Dict[str, Any]] = None,
    ) -> transforms.Compose:
        """构建torchvision变换管道
        
        Args:
            augmentation: 是否包含数据增强
            config: 增强配置
            
        Returns:
            组合好的变换对象
        """
        transform_list = []

        # 基础变换：调整尺寸
        transform_list.append(
            transforms.Resize(self.image_size)
        )

        # 数据增强（可选）
        if augmentation:
            config = config or {}
            transform_list.extend(self._get_augmentation_transforms(config))

        # 转换为张量
        transform_list.append(transforms.ToTensor())

        # 归一化
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=self.mean, std=self.std)
            )

        return transforms.Compose(transform_list)

    def _get_augmentation_transforms(
        self, config: Dict[str, Any]
    ) -> List[transforms.Transform]:
        """获取数据增强变换列表
        
        Args:
            config: 增强配置字典
            
        Returns:
            增强变换列表
        """
        augmentations = []

        # 随机水平翻转
        if config.get("horizontal_flip", True):
            augmentations.append(transforms.RandomHorizontalFlip(p=0.5))

        # 随机垂直翻转
        if config.get("vertical_flip", False):
            augmentations.append(transforms.RandomVerticalFlip(p=0.5))

        # 随机旋转
        rotation_degrees = config.get("rotation", 0)
        if rotation_degrees > 0:
            augmentations.append(
                transforms.RandomRotation(degrees=rotation_degrees)
            )

        # 随机裁剪
        crop_scale = config.get("random_crop", None)
        if crop_scale is not None:
            augmentations.append(
                transforms.RandomResizedCrop(
                    size=self.image_size,
                    scale=(crop_scale, 1.0),
                    ratio=(0.75, 1.33)
                )
            )
            # 移除之前的Resize，因为RandomResizedCrop已经处理了
            augmentations.pop(0)  # 移除Resize

        # 颜色抖动
        color_jitter = config.get("color_jitter", None)
        if color_jitter is not None:
            augmentations.append(
                transforms.ColorJitter(
                    brightness=color_jitter.get("brightness", 0.2),
                    contrast=color_jitter.get("contrast", 0.2),
                    saturation=color_jitter.get("saturation", 0.2),
                    hue=color_jitter.get("hue", 0.1)
                )
            )

        # 高斯模糊
        blur_prob = config.get("gaussian_blur", 0)
        if blur_prob > 0:
            augmentations.append(
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=blur_prob)
            )

        logger.debug(f"添加 {len(augmentations)} 个数据增强变换")
        return augmentations

    def __call__(
        self,
        image: Union[str, Path, Image.Image, torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        """处理单张图像
        
        Args:
            image: 输入图像，可以是：
                   - 文件路径（str或Path）
                   - PIL Image对象
                   - PyTorch张量
                   - NumPy数组
                   
        Returns:
            处理后的张量，形状为 (C, H, W)
            
        Raises:
            ValueError: 如果输入类型不支持或文件不存在
            IOError: 如果图像文件无法读取
        """
        pil_image = self._load_image(image)
        processed = self.transform(pil_image)
        return processed.to(self.device)

    def _load_image(
        self, image: Union[str, Path, Image.Image, torch.Tensor, np.ndarray]
    ) -> Image.Image:
        """将各种输入格式转换为PIL Image
        
        Args:
            image: 输入图像
            
        Returns:
            PIL Image对象
            
        Raises:
            ValueError: 如果输入类型不支持
        """
        if isinstance(image, (str, Path)):
            return self._load_from_path(Path(image))
        
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        
        elif isinstance(image, torch.Tensor):
            return self._tensor_to_pil(image)
        
        elif isinstance(image, np.ndarray):
            return self._array_to_pil(image)
        
        else:
            raise ValueError(
                f"不支持的图像类型: {type(image)}。"
                f"支持: str, Path, PIL.Image, torch.Tensor, np.ndarray"
            )

    def _load_from_path(self, path: Path) -> Image.Image:
        """从文件路径加载图像
        
        Args:
            path: 图像文件路径
            
        Returns:
            PIL Image对象
            
        Raises:
            FileNotFoundError: 如果文件不存在
            ValueError: 如果文件格式不支持
        """
        if not path.exists():
            raise FileNotFoundError(f"图像文件不存在: {path}")
        
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise ValueError(
                f"不支持的图像格式: {suffix}。"
                f"支持格式: {SUPPORTED_FORMATS}"
            )

        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
                logger.debug(f"成功加载图像: {path}")
                return image
        except Exception as e:
            raise IOError(f"无法读取图像文件 {path}: {e}")

    @staticmethod
    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """将PyTorch张量转换为PIL Image
        
        Args:
            tensor: 输入张量，形状为 (C, H, W) 或 (B, C, H, W) 或 (H, W, C)
            
        Returns:
            PIL Image对象
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一张图
        
        if tensor.dim() == 3:
            if tensor.shape[0] in [1, 3]:
                tensor = tensor.permute(1, 2, 0)  # CHW -> HWC
        
        # 转换为numpy数组
        array = tensor.cpu().numpy()
        
        # 归一化到[0, 255]
        if array.dtype != np.uint8:
            array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
        
        return Image.fromarray(array)

    @staticmethod
    def _array_to_pil(array: np.ndarray) -> Image.Image:
        """将NumPy数组转换为PIL Image
        
        Args:
            array: 输入数组，形状为 (H, W), (H, W, C) 或 (C, H, W)
            
        Returns:
            PIL Image对象
        """
        if array.ndim == 2:
            return Image.fromarray(array.astype(np.uint8), mode='L').convert('RGB')
        
        elif array.ndim == 3:
            if array.shape[0] in [1, 3]:  # CHW格式
                array = np.transpose(array, (1, 2, 0))
            
            if array.shape[2] == 1:  # 灰度图
                array = np.repeat(array, 3, axis=2)
            
            if array.dtype != np.uint8:
                array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
            
            return Image.fromarray(array)
        
        else:
            raise ValueError(f"不支持的数组维度: {array.ndim}")

    def process_batch(
        self,
        images: List[
            Union[str, Path, Image.Image, torch.Tensor, np.ndarray]
        ],
    ) -> torch.Tensor:
        """批量处理多张图像
        
        Args:
            images: 图像列表，每个元素可以是任意支持的格式
            
        Returns:
            批量处理后的张量，形状为 (B, C, H, W)
            
        Note:
            所有图像将被调整为相同大小并堆叠成批次
        """
        if not images:
            raise ValueError("图像列表不能为空")

        processed_images = []
        for i, img in enumerate(images):
            try:
                processed = self(img)
                processed_images.append(processed)
            except Exception as e:
                logger.error(f"处理第{i}张图像时出错: {e}")
                raise

        batch = torch.stack(processed_images, dim=0)
        logger.debug(f"批量处理完成: {len(images)} 张图像")
        return batch

    def process_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.*",
        recursive: bool = False,
    ) -> Tuple[torch.Tensor, List[Path]]:
        """从目录中加载并处理所有图像
        
        Args:
            directory: 图像目录路径
            pattern: 文件匹配模式，默认 '*.*'
            recursive: 是否递归搜索子目录
            
        Returns:
            元组 (processed_tensors, file_paths):
            - processed_tensors: 批量处理的张量
            - file_paths: 成功处理的文件路径列表
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"目录不存在: {directory}")

        glob_pattern = "**/*" if recursive else "*"
        image_paths = []
        
        for path in sorted(directory.glob(glob_pattern)):
            if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS:
                if pattern == "*.*" or path.match(pattern):
                    image_paths.append(path)

        if not image_paths:
            logger.warning(f"目录中未找到图像文件: {directory}")
            return torch.empty(0), []

        logger.info(f"在{directory}中发现 {len(image_paths)} 张图像")
        tensors = self.process_batch(image_paths)
        return tensors, image_paths

    def to(self, device: torch.device) -> "ImagePreprocessor":
        """将预处理器移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 返回自身以支持链式调用
        """
        self.device = device
        return self

    def get_transform_info(self) -> Dict[str, Any]:
        """获取当前变换管道的详细信息
        
        Returns:
            包含变换配置的字典
        """
        return {
            "image_size": self.image_size,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std,
            "augmentation": self.augmentation,
            "device": str(self.device),
            "num_transforms": len(self.transform.transforms),
        }

    def __repr__(self) -> str:
        """返回预处理器的字符串表示"""
        return (
            f"ImagePreprocessor("
            f"size={self.image_size}, "
            f"normalize={self.normalize}, "
            f"augmentation={self.augmentation})"
        )


def create_standard_preprocessor(
    model_type: str = "imagenet",
    **kwargs: Any,
) -> ImagePreprocessor:
    """创建标准预处理器工厂函数
    
        根据常用模型类型创建预配置的预处理器。
        
        Args:
            model_type: 模型类型，支持 'imagenet'（默认）等
            **kwargs: 覆盖默认配置的额外参数
            
        Returns:
            配置好的ImagePreprocessor实例
            
        Example:
            >>> preprocessor = create_standard_preprocessor('imagenet')
            >>> preprocessor = create_standard_preprocessor(
            ...     model_type='imagenet',
            ...     image_size=(299, 299),
            ...     augmentation=True
            ... )
        """
    defaults = {
        "imagenet": {
            "image_size": (224, 224),
            "normalize": True,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        }
    }

    if model_type not in defaults:
        logger.warning(f"未知的模型类型: {model_type}，使用imagenet默认值")
        model_type = "imagenet"

    config = defaults[model_type].copy()
    config.update(kwargs)
    
    return ImagePreprocessor(**config)
