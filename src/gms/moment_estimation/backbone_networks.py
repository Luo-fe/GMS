"""骨干网络模块

集成预训练的ResNet和VGG网络用于特征提取。
支持多种网络架构、层选择、权重加载和冻结选项。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
from torchvision import models

from .base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)

# 支持的ResNet架构
RESNET_ARCHITECTURES = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
}

# 支持的VGG架构
VGG_ARCHITECTURES = {
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
}

# ResNet特征层配置
RESNET_FEATURE_CONFIGS = {
    "layer1": {"output_channels": 256, "spatial_reduction": 4},
    "layer2": {"output_channels": 512, "spatial_reduction": 8},
    "layer3": {"output_channels": 1024, "spatial_reduction": 16},
    "layer4": {"output_channels": 2048, "spatial_reduction": 32},
    "avgpool": {"output_channels": 2048, "spatial_reduction": 32},
    "fc": {"output_channels": 1000, "spatial_reduction": 1},
}

# VGG特征层配置
VGG_FEATURE_CONFIGS = {
    "features_12": {"output_channels": 256, "spatial_reduction": 8},
    "features_22": {"output_channels": 512, "spatial_reduction": 16},
    "features_32": {"output_channels": 512, "spatial_reduction": 16},
    "avgpool": {"output_channels": 512, "spatial_reduction": 16},
    "classifier": {"output_channels": 1000, "spatial_reduction": 1},
}


class ResNetFeatureExtractor(BaseFeatureExtractor):
    """基于ResNet的特征提取器
    
    使用预训练的ResNet网络（18/34/50/101）提取图像特征。
    支持从不同层提取特征，并可选择性冻结/解冻网络参数。
    
    Attributes:
        architecture: ResNet架构名称
        feature_layer: 特征提取的目标层
        use_pretrained: 是否使用预训练权重
        model: ResNet模型实例
        
    Example:
        >>> extractor = ResNetFeatureExtractor(
        ...     architecture='resnet50',
        ...     feature_layer='layer3',
        ...     use_pretrained=True
        ... )
        >>> features = extractor.extract_features(images)
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        feature_layer: str = "layer3",
        use_pretrained: bool = True,
        freeze_layers: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        preprocessor: Optional[Any] = None,
    ) -> None:
        """初始化ResNet特征提取器
        
        Args:
            architecture: ResNet架构名称，支持 'resnet18/34/50/101'
            feature_layer: 特征提取目标层，支持 'layer1-4', 'avgpool', 'fc'
            use_pretrained: 是否使用ImageNet预训练权重
            freeze_layers: 要冻结的层列表，如 ['layer1', 'layer2']
            device: 计算设备
            preprocessor: 图像预处理器
            
        Raises:
            ValueError: 如果architecture或feature_layer无效
        """
        super().__init__(device=device, preprocessor=preprocessor)

        if architecture not in RESNET_ARCHITECTURES:
            raise ValueError(
                f"不支持的ResNet架构: {architecture}。"
                f"支持的架构: {list(RESNET_ARCHITECTURES.keys())}"
            )

        if feature_layer not in RESNET_FEATURE_CONFIGS:
            raise ValueError(
                f"不支持的特征层: {feature_layer}。"
                f"支持的层: {list(RESNET_FEATURE_CONFIGS.keys())}"
            )

        self.architecture = architecture
        self.feature_layer = feature_layer
        self.use_pretrained = use_pretrained

        # 加载预训练模型
        logger.info(f"加载{architecture}模型，预训练={use_pretrained}")
        model_fn = RESNET_ARCHITECTURES[architecture]
        weights = "IMAGENET1K_V1" if use_pretrained else None
        self.model = model_fn(weights=weights)

        # 设置特征提取层
        self._setup_feature_extraction(feature_layer)

        # 冻结指定层
        if freeze_layers:
            self._freeze_specific_layers(freeze_layers)

        # 移动到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()

        # 设置输出维度
        config = RESNET_FEATURE_CONFIGS[feature_layer]
        self.set_output_dim(config["output_channels"])

        logger.info(
            f"ResNet特征提取器初始化完成: "
            f"{architecture}, 层={feature_layer}, "
            f"输出维度={self.output_dim}"
        )

    def _setup_feature_extraction(self, feature_layer: str) -> None:
        """设置特征提取层
        
        根据指定的层创建特征提取的前向传播路径。
        
        Args:
            feature_layer: 目标特征层名称
        """
        self._feature_layer = feature_layer

        if feature_layer in ["layer1", "layer2", "layer3", "layer4"]:
            layer_idx = int(feature_layer[-1]) - 1
            layers_to_use = list(self.model.children())[:5 + layer_idx]
            self.feature_extractor = nn.Sequential(*layers_to_use)
        elif feature_layer == "avgpool":
            layers_to_use = list(self.model.children())[:9]
            self.feature_extractor = nn.Sequential(*layers_to_use)
        elif feature_layer == "fc":
            self.feature_extractor = self.model
        else:
            raise ValueError(f"未知的特征层: {feature_layer}")

    def _freeze_specific_layers(self, layers: List[str]) -> None:
        """冻结指定的网络层
        
        Args:
            layers: 要冻结的层名称列表
        """
        for layer_name in layers:
            if hasattr(self.model, layer_name):
                layer = getattr(self.model, layer_name)
                for param in layer.parameters():
                    param.requires_grad = False
                logger.debug(f"已冻结层: {layer_name}")
            else:
                logger.warning(f"未找到层: {layer_name}，跳过冻结")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """从图像中提取ResNet特征
        
        Args:
            images: 输入图像张量，形状为 (B, C, H, W)
            
        Returns:
            提取的特征张量
            
        Note:
            如果feature_layer为'fc'，返回形状为 (B, 1000) 的分类向量；
            其他层返回形状为 (B, C, H', W') 的特征图或 (B, C) 的池化特征。
        """
        with torch.no_grad():
            if self.feature_layer == "fc":
                return self.model(images)
            else:
                features = self.feature_extractor(images)
                
                if self.feature_layer == "avgpool":
                    features = torch.flatten(features, 1)
                    
                return features

    def get_feature_map_size(
        self, input_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[int, int]:
        """计算给定输入尺寸下的特征图大小
        
        Args:
            input_size: 输入图像尺寸 (H, W)
            
        Returns:
            特征图尺寸 (H', W')
        """
        config = RESNET_FEATURE_CONFIGS[self.feature_layer]
        reduction = config["spatial_reduction"]
        h, w = input_size
        return (h // reduction, w // reduction)

    def __repr__(self) -> str:
        """返回模型的字符串表示"""
        return (
            f"ResNetFeatureExtractor("
            f"arch={self.architecture}, "
            f"layer={self.feature_layer}, "
            f"pretrained={self.use_pretrained}, "
            f"device={self.device})"
        )


class VGGFeatureExtractor(BaseFeatureExtractor):
    """基于VGG的特征提取器
    
    使用预训练的VGG网络（11/13/16/19）提取图像特征。
    支持从不同层提取特征，并提供与ResNet类似的接口。
    
    Attributes:
        architecture: VGG架构名称
        feature_layer: 特征提取的目标层
        use_pretrained: 是否使用预训练权重
        model: VGG模型实例
        
    Example:
        >>> extractor = VGGFeatureExtractor(
        ...     architecture='vgg16',
        ...     feature_layer='features_22',
        ...     use_pretrained=True
        ... )
        >>> features = extractor.extract_features(images)
    """

    def __init__(
        self,
        architecture: str = "vgg16",
        feature_layer: str = "features_22",
        use_pretrained: bool = True,
        freeze_features: bool = True,
        device: Optional[torch.device] = None,
        preprocessor: Optional[Any] = None,
    ) -> None:
        """初始化VGG特征提取器
        
        Args:
            architecture: VGG架构名称，支持 'vgg11/13/16/19'
            feature_layer: 特征提取目标层，支持 'features_12/22/32', 'avgpool', 'classifier'
            use_pretrained: 是否使用ImageNet预训练权重
            freeze_features: 是否冻结特征提取部分
            device: 计算设备
            preprocessor: 图像预处理器
            
        Raises:
            ValueError: 如果architecture或feature_layer无效
        """
        super().__init__(device=device, preprocessor=preprocessor)

        if architecture not in VGG_ARCHITECTURES:
            raise ValueError(
                f"不支持的VGG架构: {architecture}。"
                f"支持的架构: {list(VGG_ARCHITECTURES.keys())}"
            )

        if feature_layer not in VGG_FEATURE_CONFIGS:
            raise ValueError(
                f"不支持的特征层: {feature_layer}。"
                f"支持的层: {list(VGG_FEATURE_CONFIGS.keys())}"
            )

        self.architecture = architecture
        self.feature_layer = feature_layer
        self.use_pretrained = use_pretrained

        # 加载预训练模型
        logger.info(f"加载{architecture}模型，预训练={use_pretrained}")
        model_fn = VGG_ARCHITECTURES[architecture]
        weights = "IMAGENET1K_V1" if use_pretrained else None
        self.model = model_fn(weights=weights)

        # 设置特征提取层
        self._setup_feature_extraction(feature_layer)

        # 冻结特征提取部分
        if freeze_features:
            self._freeze_features()

        # 移动到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()

        # 设置输出维度
        config = VGG_FEATURE_CONFIGS[feature_layer]
        self.set_output_dim(config["output_channels"])

        logger.info(
            f"VGG特征提取器初始化完成: "
            f"{architecture}, 层={feature_layer}, "
            f"输出维度={self.output_dim}"
        )

    def _setup_feature_extraction(self, feature_layer: str) -> None:
        """设置特征提取层
        
        Args:
            feature_layer: 目标特征层名称
        """
        self._feature_layer = feature_layer

        if feature_layer.startswith("features_"):
            layer_idx = int(feature_layer.split("_")[1])
            self.feature_extractor = nn.Sequential(
                *list(self.model.features.children())[:layer_idx]
            )
        elif feature_layer == "avgpool":
            self.feature_extractor = nn.Sequential(
                self.model.features,
                self.model.avgpool
            )
        elif feature_layer == "classifier":
            self.feature_extractor = self.model
        else:
            raise ValueError(f"未知的特征层: {feature_layer}")

    def _freeze_features(self) -> None:
        """冻结特征提取部分的所有参数"""
        for param in self.model.features.parameters():
            param.requires_grad = False
        logger.debug("已冻结VGG特征提取部分")

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """从图像中提取VGG特征
        
        Args:
            images: 输入图像张量，形状为 (B, C, H, W)
            
        Returns:
            提取的特征张量
        """
        with torch.no_grad():
            if self.feature_layer == "classifier":
                return self.model(images)
            else:
                features = self.feature_extractor(images)
                
                if self.feature_layer == "avgpool":
                    features = torch.flatten(features, 1)
                    
                return features

    def get_feature_map_size(
        self, input_size: Tuple[int, int] = (224, 224)
    ) -> Tuple[int, int]:
        """计算给定输入尺寸下的特征图大小
        
        Args:
            input_size: 输入图像尺寸 (H, W)
            
        Returns:
            特征图尺寸 (H', W')
        """
        config = VGG_FEATURE_CONFIGS[self.feature_layer]
        reduction = config["spatial_reduction"]
        h, w = input_size
        return (h // reduction, w // reduction)

    def __repr__(self) -> str:
        """返回模型的字符串表示"""
        return (
            f"VGGFeatureExtractor("
            f"arch={self.architecture}, "
            f"layer={self.feature_layer}, "
            f"pretrained={self.use_pretrained}, "
            f"device={self.device})"
        )


def create_feature_extractor(
    name: str = "resnet50",
    **kwargs: Any,
) -> BaseFeatureExtractor:
    """工厂函数：创建特征提取器实例
    
    根据名称自动选择合适的特征提取器类型。
    
    Args:
        name: 特征提取器名称，格式为 '架构名' 或 '架构名_层名'
               例如: 'resnet50', 'resnet50_layer3', 'vgg16_features_22'
        **kwargs: 传递给特征提取器的额外参数
        
    Returns:
        配置好的特征提取器实例
        
    Raises:
        ValueError: 如果名称无法识别
        
    Example:
        >>> extractor = create_feature_extractor('resnet50_layer3')
        >>> extractor = create_feature_extractor('vgg16', feature_layer='features_22')
    """
    name_lower = name.lower()

    # 解析架构和层
    parts = name_lower.split("_")
    architecture = parts[0]

    if architecture.startswith("resnet"):
        kwargs.setdefault("architecture", architecture)
        if len(parts) > 1 and parts[1].startswith("layer"):
            kwargs.setdefault("feature_layer", "_".join(parts[1:]))
        return ResNetFeatureExtractor(**kwargs)
    
    elif architecture.startswith("vgg"):
        kwargs.setdefault("architecture", architecture)
        if len(parts) > 1 and (parts[1].startswith("features") or 
                               parts[1] in ["avgpool", "classifier"]):
            kwargs.setdefault("feature_layer", "_".join(parts[1:]))
        return VGGFeatureExtractor(**kwargs)
    
    else:
        raise ValueError(
            f"无法识别的特征提取器: {name}。"
            f"请使用 'resnet*' 或 'vgg*' 格式。"
        )
