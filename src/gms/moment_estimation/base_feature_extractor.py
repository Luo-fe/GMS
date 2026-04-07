"""特征提取器基类模块

定义所有特征提取器的抽象基类和标准接口。
提供设备管理、预处理配置等通用功能。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """特征提取器抽象基类
    
    所有特征提取器的基类，定义标准接口和通用功能。
    支持输入图像预处理配置、设备管理（CPU/GPU）等。
    
    Attributes:
        device (torch.device): 计算设备（CPU或GPU）
        preprocessor: 图像预处理器（可选）
        output_dim (int): 输出特征维度
        
    Example:
        >>> class MyExtractor(BaseFeatureExtractor):
        ...     def __init__(self):
        ...         super().__init__(device=torch.device('cpu'))
        ...         
        ...     def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        ...         # 实现特征提取逻辑
        ...         return images.flatten(1)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        preprocessor: Optional[Any] = None,
    ) -> None:
        """初始化特征提取器
        
        Args:
            device: 计算设备，如果为None则自动选择最佳设备
            preprocessor: 图像预处理器实例，如果为None则使用默认配置
        """
        self.device = device or self._get_default_device()
        self.preprocessor = preprocessor
        self.output_dim: int = 0
        self._is_frozen: bool = False
        logger.info(f"初始化特征提取器，设备: {self.device}")

    @staticmethod
    def _get_default_device() -> torch.device:
        """获取默认计算设备
        
        优先使用CUDA（如果可用），否则使用CPU。
        
        Returns:
            torch.device: 可用的计算设备
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @abstractmethod
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """从图像中提取特征（抽象方法）
        
        Args:
            images: 输入图像张量，形状为 (B, C, H, W) 或 (C, H, W)
            
        Returns:
            提取的特征张量
            
        Raises:
            NotImplementedError: 子类必须实现此方法
        """
        raise NotImplementedError("子类必须实现 extract_features 方法")

    def preprocess(
        self,
        images: Union[torch.Tensor, "np.ndarray"],
    ) -> torch.Tensor:
        """对输入图像进行预处理
        
        如果配置了预处理器，则使用它进行预处理；
        否则执行基本的张量转换和设备移动。
        
        Args:
            images: 输入图像，可以是PyTorch张量或NumPy数组
            
        Returns:
            预处理后的张量，位于当前设备上
        """
        if self.preprocessor is not None:
            return self.preprocessor(images)

        if isinstance(images, torch.Tensor):
            return images.to(self.device)
        
        import numpy as np
        tensor = torch.from_numpy(images).float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor.to(self.device)

    def to(self, device: torch.device) -> "BaseFeatureExtractor":
        """将模型移动到指定设备
        
        Args:
            device: 目标设备
            
        Returns:
            self: 返回自身以支持链式调用
        """
        self.device = device
        if hasattr(self, 'model') and self.model is not None:
            self.model.to(device)
        if self.preprocessor is not None and hasattr(self.preprocessor, 'to'):
            self.preprocessor.to(device)
        logger.debug(f"模型已移动到设备: {device}")
        return self

    def freeze(self) -> None:
        """冻结模型参数（停止梯度计算）"""
        self._is_frozen = True
        if hasattr(self, 'model') and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
        logger.info("模型参数已冻结")

    def unfreeze(self) -> None:
        """解冻模型参数（启用梯度计算）"""
        self._is_frozen = False
        if hasattr(self, 'model') and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = True
        logger.info("模型参数已解冻")

    @property
    def is_frozen(self) -> bool:
        """检查模型是否处于冻结状态"""
        return self._is_frozen

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """获取模型参数数量
        
        Args:
            trainable_only: 是否只统计可训练参数
            
        Returns:
            参数数量
        """
        if not hasattr(self, 'model') or self.model is None:
            return 0
        
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())

    def get_output_dim(self) -> int:
        """获取输出特征维度
        
        Returns:
            输出特征维度
        """
        if self.output_dim == 0:
            logger.warning("输出维度尚未设置，返回0")
        return self.output_dim

    def set_output_dim(self, dim: int) -> None:
        """设置输出特征维度
        
        Args:
            dim: 特征维度
        """
        self.output_dim = dim
        logger.debug(f"输出维度设置为: {dim}")

    def __repr__(self) -> str:
        """返回模型的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"output_dim={self.output_dim}, "
            f"frozen={self._is_frozen})"
        )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """使实例可调用，直接进行特征提取
        
        Args:
            images: 输入图像张量
            
        Returns:
            提取的特征张量
        """
        return self.extract_features(images)
