"""矩估计统一接口模块

整合MeanHead、VarianceHead、SkewnessHead，提供统一的矩估计接口。
支持选择性启用/禁用特定输出头，并返回结构化的结果。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mean_head import MeanHead
from .variance_head import VarianceHead
from .skewness_head import SkewnessHead

logger = logging.getLogger(__name__)


@dataclass
class MomentResult:
    """矩估计结果数据类

    存储所有矩估计结果及其精度指标。

    Attributes:
        mean: 均值向量 μ (batch, output_dim) 或 (output_dim,)
        variance: 方差 σ² 或协方差矩阵 Σ
        skewness: 偏度系数 γ
        mean_metrics: 均值预测的精度评估指标字典
        variance_metrics: 方差预测的精度评估指标字典
        skewness_metrics: 偏度预测的精度评估指标字典
        metadata: 额外的元数据信息

    Example:
        >>> result = MomentResult(
        ...     mean=torch.randn(10),
        ...     variance=torch.abs(torch.randn(10)),
        ...     skewness=torch.randn(10)
        ... )
        >>> print(result.mean.shape)
        torch.Size([10])
    """

    mean: Optional[torch.Tensor] = None
    variance: Optional[torch.Tensor] = None
    skewness: Optional[torch.Tensor] = None

    mean_metrics: Dict[str, float] = field(default_factory=dict)
    variance_metrics: Dict[str, float] = field(default_factory=dict)
    skewness_metrics: Dict[str, float] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_mean(self) -> bool:
        """检查是否包含均值结果"""
        return self.mean is not None

    @property
    def has_variance(self) -> bool:
        """检查是否包含方差结果"""
        return self.variance is not None

    @property
    def has_skewness(self) -> bool:
        """检查是否包含偏度结果"""
        return self.skewness is not None

    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式

        Returns:
            包含所有结果的字典
        """
        result_dict = {}

        if self.has_mean:
            result_dict["mean"] = self.mean

        if self.has_variance:
            result_dict["variance"] = self.variance

        if self.has_skewness:
            result_dict["skewness"] = self.skewness

        result_dict["metrics"] = {
            "mean": self.mean_metrics,
            "variance": self.variance_metrics,
            "skewness": self.skewness_metrics,
        }

        if self.metadata:
            result_dict["metadata"] = self.metadata

        return result_dict

    def detach(self) -> "MomentResult":
        """分离所有张量（停止梯度追踪）

        Returns:
            新的MomentResult实例，所有张量已分离
        """
        new_result = MomentResult(
            mean=self.mean.detach() if self.has_mean else None,
            variance=self.variance.detach() if self.has_variance else None,
            skewness=self.skewness.detach() if self.has_skewness else None,
            mean_metrics=self.mean_metrics.copy(),
            variance_metrics=self.variance_metrics.copy(),
            skewness_metrics=self.skewness_metrics.copy(),
            metadata=self.metadata.copy(),
        )
        return new_result

    def cpu(self) -> "MomentResult":
        """将所有张量移动到CPU

        Returns:
            新的MomentResult实例，所有张量在CPU上
        """
        new_result = MomentResult(
            mean=self.mean.cpu() if self.has_mean else None,
            variance=self.variance.cpu() if self.has_variance else None,
            skewness=self.skewness.cpu() if self.has_skewness else None,
            mean_metrics=self.mean_metrics.copy(),
            variance_metrics=self.variance_metrics.copy(),
            skewness_metrics=self.skewness_metrics.copy(),
            metadata=self.metadata.copy(),
        )
        return new_result


class MomentEstimator(nn.Module):
    """矩估计器 - 统一的多输出头管理器

    整合MeanHead、VarianceHead和SkewnessHead，
    提供统一的接口进行多阶矩的联合估计。

    Attributes:
        feature_dim: 输入特征维度
        output_dim: 输出维度
        enable_mean: 是否启用均值头
        enable_variance: 是否启用的方差头
        enable_skewness: 是否启用偏度头
        mean_head: 均值输出头实例
        variance_head: 方差输出头实例
        skewness_head: 偏度输出头实例

    Example:
        >>> estimator = MomentEstimator(
        ...     feature_dim=1024,
        ...     output_dim=10,
        ...     enable_mean=True,
        ...     enable_variance=True,
        ...     enable_skewness=True
        ... )
        >>> features = torch.randn(32, 1024)
        >>> result = estimator(features)
        >>> print(result.mean.shape)
        torch.Size([32, 10])
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        enable_mean: bool = True,
        enable_variance: bool = True,
        enable_skewness: bool = True,
        mean_hidden_dims: Optional[Union[int, Tuple[int, ...]]] = None,
        variance_hidden_dims: Optional[Union[int, Tuple[int, ...]]] = None,
        skewness_hidden_dims: Optional[Union[int, Tuple[int, ...]]] = None,
        variance_mode: str = "diagonal",
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        **kwargs,
    ) -> None:
        """初始化矩估计器

        Args:
            feature_dim: 骨干网络特征维度
            output_dim: 所有输出头的输出维度
            enable_mean: 是否启用均值估计
            enable_variance: 是否启用方差/协方差估计
            enable_skewness: 是否启用偏度估计
            mean_hidden_dims: 均值头隐藏层配置
            variance_hidden_dims: 方差头隐藏层配置
            skewness_hidden_dims: 偏度头隐藏层配置
            variance_mode: 方差模式，'diagonal' 或 'full'
            activation: 激活函数类型
            dropout: Dropout比率
            use_batch_norm: 是否使用批归一化
            **kwargs: 传递给各个头的额外参数

        Raises:
            ValueError: 如果参数无效或至少未启用一个输出头
        """
        super().__init__()

        if not (enable_mean or enable_variance or enable_skewness):
            raise ValueError("至少需要启用一个输出头（均值、方差或偏度）")

        if feature_dim <= 0:
            raise ValueError(f"feature_dim必须是正整数，当前值: {feature_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim必须是正整数，当前值: {output_dim}")

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.enable_mean = enable_mean
        self.enable_variance = enable_variance
        self.enable_skewness = enable_skewness

        if enable_mean:
            self.mean_head = MeanHead(
                feature_dim=feature_dim,
                output_dim=output_dim,
                hidden_dims=mean_hidden_dims,
                activation=activation,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
            )

        if enable_variance:
            self.variance_head = VarianceHead(
                feature_dim=feature_dim,
                output_dim=output_dim,
                mode=variance_mode,
                hidden_dims=variance_hidden_dims,
                activation=activation,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                **{k: v for k, v in kwargs.items()
                   if k in ["variance_activation", "min_variance"]},
            )

        if enable_skewness:
            self.skewness_head = SkewnessHead(
                feature_dim=feature_dim,
                output_dim=output_dim,
                hidden_dims=skewness_hidden_dims,
                activation="tanh",
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                **{k: v for k, v in kwargs.items()
                   if k in ["clamp_range"]},
            )

        logger.info(
            f"MomentEstimator初始化完成: "
            f"特征维度={feature_dim}, "
            f"输出维度={output_dim}, "
            f"启用=[均值={enable_mean}, 方差={enable_variance}, 偏度={enable_skewness}]"
        )

    def forward(self, features: torch.Tensor) -> MomentResult:
        """前向传播：执行所有启用的矩估计

        Args:
            features: 骨干网络提取的特征张量，
                     形状为 (batch_size, feature_dim) 或 (feature_dim,)

        Returns:
            MomentResult对象，包含所有启用的矩估计结果
        """
        result = MomentResult()

        if self.enable_mean:
            result.mean = self.mean_head(features)

        if self.enable_variance:
            result.variance = self.variance_head(features)

        if self.enable_skewness:
            result.skewness = self.skewness_head(features)

        return result

    def forward_with_targets(
        self,
        features: torch.Tensor,
        target_mean: Optional[torch.Tensor] = None,
        target_variance: Optional[torch.Tensor] = None,
        target_skewness: Optional[torch.Tensor] = None,
    ) -> MomentResult:
        """带目标值的前向传播，同时计算精度指标

        在执行前向传播的同时计算各输出的精度评估。

        Args:
            features: 输入特征张量
            target_mean: 真实的均值（可选）
            target_variance: 真实的方差/协方差（可选）
            target_skewness: 真实的偏度（可选）

        Returns:
            包含预测结果和精度指标的MomentResult对象
        """
        result = self.forward(features)

        if self.enable_mean and target_mean is not None:
            result.mean_metrics = self.mean_head.evaluate_accuracy(
                result.mean, target_mean
            )

        if self.enable_variance and target_variance is not None:
            result.variance_metrics = self.variance_head.evaluate_accuracy(
                result.variance, target_variance
            )

        if self.enable_skewness and target_skewness is not None:
            result.skewness_metrics = self.skewness_head.evaluate_accuracy(
                result.skewness, target_skewness
            )

        return result

    def get_enabled_heads(self) -> List[str]:
        """获取当前启用的输出头列表

        Returns:
            启用的输出头名称列表
        """
        enabled = []
        if self.enable_mean:
            enabled.append("mean")
        if self.enable_variance:
            enabled.append("variance")
        if self.enable_skewness:
            enabled.append("skewness")
        return enabled

    def set_head_enabled(
        self,
        head_name: str,
        enabled: bool,
    ) -> None:
        """动态启用或禁用指定的输出头

        Args:
            head_name: 输出头名称，'mean' | 'variance' | 'skewness'
            enabled: 是否启用

        Raises:
            ValueError: 如果head_name无效
        """
        if head_name == "mean":
            if not hasattr(self, "mean_head"):
                raise ValueError("均值头未被初始化")
            self.enable_mean = enabled
        elif head_name == "variance":
            if not hasattr(self, "variance_head"):
                raise ValueError("方差头未被初始化")
            self.enable_variance = enabled
        elif head_name == "skewness":
            if not hasattr(self, "skewness_head"):
                raise ValueError("偏度头未被初始化")
            self.enable_skewness = enabled
        else:
            raise ValueError(
                f"无效的head_name: {head_name}。"
                f"支持的名称: ['mean', 'variance', 'skewness']"
            )

        logger.info(f"{head_name}头已{'启用' if enabled else '禁用'}")

    def compute_total_loss(
        self,
        result: MomentResult,
        target_mean: Optional[torch.Tensor] = None,
        target_variance: Optional[torch.Tensor] = None,
        target_skewness: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """计算总损失

        计算所有启用的输出头的加权损失总和。

        Args:
            result: 前向传播的结果
            target_mean: 目标均值
            target_variance: 目标方差
            target_skewness: 目标偏度
            weights: 各损失的权重字典，默认为等权重。
                     格式: {"mean": 1.0, "variance": 1.0, "skewness": 1.0}

        Returns:
            总损失标量

        Raises:
            ValueError: 如果缺少必要的目标值
        """
        if weights is None:
            weights = {
                "mean": 1.0,
                "variance": 1.0,
                "skewness": 1.0,
            }

        device = result.mean.device if result.has_mean else (
            result.variance.device if result.has_variance else
            (result.skewness.device if result.has_skewness else "cpu")
        )
        total_loss = torch.tensor(0.0, device=device)

        if self.enable_mean:
            if target_mean is None:
                raise ValueError("启用了均值头但未提供target_mean")
            mean_loss = F.mse_loss(result.mean, target_mean)
            total_loss = total_loss + weights.get("mean", 1.0) * mean_loss

        if self.enable_variance:
            if target_variance is None:
                raise ValueError("启用了方差头但未提供target_variance")
            var_loss = F.mse_loss(result.variance, target_variance)
            total_loss = total_loss + weights.get("variance", 1.0) * var_loss

        if self.enable_skewness:
            if target_skewness is None:
                raise ValueError("启用了偏度头但未提供target_skewness")
            skew_loss = F.mse_loss(result.skewness, target_skewness)
            total_loss = total_loss + weights.get("skewness", 1.0) * skew_loss

        return total_loss

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """获取总参数数量

        Args:
            trainable_only: 是否只统计可训练参数

        Returns:
            参数总数或可训练参数数
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_parameters_by_head(self) -> Dict[str, int]:
        """获取每个输出头的参数数量

        Returns:
            字典，键为输出头名称，值为参数数量
        """
        params_by_head = {}

        if self.enable_mean:
            params_by_head["mean"] = self.mean_head.get_num_parameters()

        if self.enable_variance:
            params_by_head["variance"] = self.variance_head.get_num_parameters()

        if self.enable_skewness:
            params_by_head["skewness"] = self.skewness_head.get_num_parameters()

        return params_by_head

    def __repr__(self) -> str:
        """返回模型的字符串表示"""
        num_params = self.get_num_parameters()
        enabled_heads = self.get_enabled_heads()

        return (
            f"MomentEstimator("
            f"feature_dim={self.feature_dim}, "
            f"output_dim={self.output_dim}, "
            f"enabled_heads={enabled_heads}, "
            f"total_params={num_params})"
        )


def create_moment_estimator(
    feature_dim: int,
    output_dim: int,
    config: Optional[Dict[str, Any]] = None,
) -> MomentEstimator:
    """工厂函数：创建配置好的矩估计器

    根据配置字典创建并返回MomentEstimator实例。

    Args:
        feature_dim: 输入特征维度
        output_dim: 输出维度
        config: 配置字典，可以包含以下键：
                - enable_mean: bool
                - enable_variance: bool
                - enable_skewness: bool
                - variance_mode: str
                - hidden_dims: int or tuple
                - activation: str
                - dropout: float
                - use_batch_norm: bool
                以及其他传递给MomentEstimator的参数

    Returns:
        配置好的MomentEstimator实例

    Example:
        >>> estimator = create_moment_estimator(1024, 10, {
        ...     "enable_mean": True,
        ...     "enable_variance": True,
        ...     "enable_skewness": False,
        ...     "activation": "gelu"
        ... })
    """
    if config is None:
        config = {}

    default_config = {
        "enable_mean": True,
        "enable_variance": True,
        "enable_skewness": True,
        "variance_mode": "diagonal",
        "activation": "relu",
        "dropout": 0.0,
        "use_batch_norm": False,
    }

    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    estimator = MomentEstimator(
        feature_dim=feature_dim,
        output_dim=output_dim,
        **config
    )

    logger.info(f"通过工厂函数创建MomentEstimator: {estimator}")
    return estimator
