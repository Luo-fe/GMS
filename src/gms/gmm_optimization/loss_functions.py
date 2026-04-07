"""GMM优化器损失函数 - 实现矩匹配误差计算"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimizer_base import TargetMoments

logger = logging.getLogger(__name__)


@dataclass
class LossConfig:
    """损失函数配置数据类

    Attributes:
        mean_weight: 一阶矩（均值）损失的权重
        variance_weight: 二阶矩（方差/协方差）损失的权重
        skewness_weight: 三阶矩（偏度）损失的权重
        covariance_epsilon: 协方差矩阵正则化系数（添加εI）
        gradient_clip_max: 梯度裁剪最大值
        use_frobenius_norm: 是否使用Frobenius范数计算协方差损失
        normalize_by_dimension: 是否按维度归一化损失
        reduction: 损失约简方式 ('mean', 'sum', 'none')

    Example:
        >>> config = LossConfig(
        ...     mean_weight=1.0,
        ...     variance_weight=2.0,
        ...     skewness_weight=0.5
        ... )
    """

    mean_weight: float = 1.0
    variance_weight: float = 1.0
    skewness_weight: float = 0.5
    covariance_epsilon: float = 1e-6
    gradient_clip_max: float = 10.0
    use_frobenius_norm: bool = True
    normalize_by_dimension: bool = False
    reduction: str = "mean"

    def __post_init__(self):
        """验证配置参数"""
        if self.mean_weight < 0:
            raise ValueError(f"mean_weight不能为负，当前值: {self.mean_weight}")
        if self.variance_weight < 0:
            raise ValueError(f"variance_weight不能为负，当前值: {self.variance_weight}")
        if self.skewness_weight < 0:
            raise ValueError(f"skewness_weight不能为负，当前值: {self.skewness_weight}")
        if self.covariance_epsilon < 0:
            raise ValueError(f"covariance_epsilon不能为负，当前值: {self.covariance_epsilon}")
        valid_reductions = ["mean", "sum", "none"]
        if self.reduction not in valid_reductions:
            raise ValueError(
                f"reduction必须是 {valid_reductions} 之一，当前值: {self.reduction}"
            )


class MomentMatchingLoss(nn.Module):
    """矩匹配损失函数

    计算GMM预测矩与目标矩之间的加权误差。
    支持一阶、二阶、三阶矩的灵活配置和数值稳定性处理。

    损失组成:
        L₁ = ||μ_pred - μ_target||² (均值损失)
        L₂ = ||Σ_pred - Σ_target||_F² (协方差Frobenius范数损失)
        L₃ = ||γ_pred - γ_target||² (偏度损失)
        L_total = w₁L₁ + w₂L₂ + w₃L₃

    Attributes:
        config: 损失配置
        _last_loss_components: 上次计算的各分量损失（用于监控）

    Example:
        >>> loss_fn = MomentMatchingLoss(LossConfig(mean_weight=2.0))
        >>> params = {'means': ..., 'covariances': ..., 'weights': ...}
        >>> target = TargetMoments(mean=..., covariance=...)
        >>> loss = loss_fn(params, target)
    """

    def __init__(self, config: Optional[LossConfig] = None):
        """初始化矩匹配损失函数

        Args:
            config: 损失配置，如果为None则使用默认配置
        """
        super().__init__()
        self.config = config or LossConfig()
        self._last_loss_components: Dict[str, float] = {}

    def forward(
        self,
        params: Dict[str, torch.Tensor],
        target_moments: TargetMoments,
    ) -> torch.Tensor:
        """前向传播：计算总损失

        Args:
            params: GMM参数字典，包含:
                   - 'means': 预测均值 (n_components, n_features)
                   - 'covariances': 预测协方差
                   - 'weights': 混合权重 (n_components,)
            target_moments: 目标矩对象

        Returns:
            总损失值（标量张量）

        Raises:
            ValueError: 如果缺少必要的参数或目标矩
        """
        device = self._get_device(params)
        total_loss = torch.tensor(0.0, device=device)
        components = {}

        means = params.get('means')
        covariances = params.get('covariances')
        weights = params.get('weights')

        if means is not None and target_moments.mean is not None and self.config.mean_weight > 0:
            mean_loss = self._compute_mean_loss(means, weights, target_moments.mean)
            components['mean'] = mean_loss.item()
            total_loss = total_loss + self.config.mean_weight * mean_loss

        if covariances is not None and target_moments.covariance is not None and self.config.variance_weight > 0:
            var_loss = self._compute_covariance_loss(covariances, weights, target_moments.covariance)
            components['variance'] = var_loss.item()
            total_loss = total_loss + self.config.variance_weight * var_loss

        if target_moments.skewness is not None and self.config.skewness_weight > 0:
            skewness_pred = self._predict_skewness(params)
            skew_loss = self._compute_skewness_loss(skewness_pred, target_moments.skewness)
            components['skewness'] = skew_loss.item()
            total_loss = total_loss + self.config.skewness_weight * skew_loss

        self._last_loss_components = components

        return total_loss

    def __call__(
        self,
        params: Dict[str, torch.Tensor],
        target_moments: TargetMoments,
    ) -> torch.Tensor:
        """使实例可调用"""
        return self.forward(params, target_moments)

    def _get_device(self, params: Dict[str, torch.Tensor]) -> torch.device:
        """获取参数所在的设备

        Args:
            params: 参数字典

        Returns:
            设备对象
        """
        for param in params.values():
            return param.device
        return torch.device('cpu')

    def _compute_mean_loss(
        self,
        predicted_means: torch.Tensor,
        weights: Optional[torch.Tensor],
        target_mean: torch.Tensor,
    ) -> torch.Tensor:
        """计算一阶矩（均值）损失

        L₁ = ||μ_pred - μ_target||²

        如果提供了混合权重，则计算加权平均均值。

        Args:
            predicted_means: 预测的各分量均值 (n_components, n_features)
            weights: 混合权重 (n_components,)
            target_mean: 目标均值 (n_features,)

        Returns:
            均值损失值
        """
        if weights is not None:
            weighted_mean = torch.sum(
                weights.unsqueeze(1) * predicted_means, dim=0
            )
        else:
            weighted_mean = predicted_means.mean(dim=0)

        diff = weighted_mean - target_mean
        loss = torch.sum(diff ** 2)

        if self.config.normalize_by_dimension:
            loss = loss / target_mean.numel()

        if self.config.reduction == "mean":
            loss = loss / target_mean.numel()

        return loss

    def _compute_covariance_loss(
        self,
        predicted_covs: torch.Tensor,
        weights: Optional[torch.Tensor],
        target_cov: torch.Tensor,
    ) -> torch.Tensor:
        """计算二阶矩（协方差）损失

        L₂ = ||Σ_pred - Σ_target||_F² （Frobenius范数）

        包含协方差矩阵正则化以确保数值稳定性。

        Args:
            predicted_covs: 预测的协方差矩阵
                           形状: (n_components, n_features, n_features) 或 (n_components, n_features)
            weights: 混合权重 (n_components,)
            target_cov: 目标协方差矩阵

        Returns:
            协方差损失值
        """
        predicted_covs = self._regularize_covariance(predicted_covs)

        if weights is not None:
            if predicted_covs.dim() == 3:
                weighted_cov = torch.sum(
                    weights.unsqueeze(1).unsqueeze(2) * predicted_covs, dim=0
                )
            else:
                weighted_cov = torch.sum(
                    weights.unsqueeze(1) * predicted_covs, dim=0
                )
        else:
            weighted_cov = predicted_covs.mean(dim=0)

        if self.config.use_frobenius_norm:
            diff = weighted_cov - target_cov
            loss = torch.sum(diff ** 2)
        else:
            loss = F.mse_loss(weighted_cov, target_cov, reduction=self.config.reduction)

        if self.config.normalize_by_dimension:
            n_elements = target_cov.numel()
            loss = loss / n_elements

        return loss

    def _compute_skewness_loss(
        self,
        predicted_skewness: torch.Tensor,
        target_skewness: torch.Tensor,
    ) -> torch.Tensor:
        """计算三阶矩（偏度）损失

        L₃ = ||γ_pred - γ_target||²

        Args:
            predicted_skewness: 预测的偏度
            target_skewness: 目标偏度

        Returns:
            偏度损失值
        """
        diff = predicted_skewness - target_skewness
        loss = torch.sum(diff ** 2)

        if self.config.normalize_by_dimension:
            loss = loss / target_skewness.numel()

        if self.config.reduction == "mean":
            loss = loss / target_skewness.numel()

        return loss

    def _predict_skewness(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从GMM参数预测偏度

        对于高斯分布，理论偏度为0。
        此方法返回零向量作为默认预测。

        Args:
            params: GMM参数字典

        Returns:
            零偏度张量
        """
        means = params.get('means')
        if means is not None:
            return torch.zeros_like(means[0])
        return torch.tensor([0.0])

    def _regularize_covariance(self, cov: torch.Tensor) -> torch.Tensor:
        """对协方差矩阵进行正则化

        添加 εI 以确保正定性，防止数值不稳定。

        Args:
            cov: 协方差矩阵或对角方差向量

        Returns:
            正则化后的协方差
        """
        epsilon = self.config.covariance_epsilon

        if cov.dim() == 3:
            identity = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
            cov = cov + epsilon * identity.unsqueeze(0)
        elif cov.dim() == 2:
            cov = cov + epsilon * torch.eye(cov.shape[1], device=cov.device, dtype=cov.dtype)
        elif cov.dim() == 1:
            cov = torch.clamp(cov, min=epsilon)

        return cov

    def get_last_loss_components(self) -> Dict[str, float]:
        """获取上次计算的各分量损失

        Returns:
            各分量的损失字典，如 {'mean': 0.123, 'variance': 0.456}
        """
        return self._last_loss_components.copy()

    def clip_gradients(self, parameters: Any) -> None:
        """裁剪梯度以防止梯度爆炸

        Args:
            parameters: 模型参数（可以是参数列表或生成器）
        """
        if self.config.gradient_clip_max > 0:
            torch.nn.utils.clip_grad_value_(parameters, self.config.gradient_clip_max)


class WeightedMSELoss(nn.Module):
    """加权均方误差损失

    允许对不同特征维度应用不同的权重。

    Attributes:
        weights: 特征权重向量
        reduction: 约简方式

    Example:
        >>> loss_fn = WeightedMSELoss(weights=torch.tensor([1.0, 2.0, 0.5]))
        >>> pred = torch.randn(10, 3)
        >>> target = torch.randn(10, 3)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """初始化加权MSE损失

        Args:
            weights: 特征权重，形状 (n_features,)。如果为None则等权重
            reduction: 约简方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.weights = weights
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """计算加权MSE损失

        Args:
            prediction: 预测值
            target: 目标值

        Returns:
            加权损失值
        """
        squared_diff = (prediction - target) ** 2

        if self.weights is not None:
            if prediction.dim() == 2:
                weighted_diff = squared_diff * self.weights.unsqueeze(0)
            else:
                weighted_diff = squared_diff * self.weights
        else:
            weighted_diff = squared_diff

        if self.reduction == "mean":
            return weighted_diff.mean()
        elif self.reduction == "sum":
            return weighted_diff.sum()
        else:
            return weighted_diff


class HuberMomentLoss(nn.Module):
    """Huber矩损失

    对异常值更鲁棒的损失函数，在小误差时为二次，
    大误差时变为线性。

    Attributes:
        delta: 从二次到线性的转折点
        config: 基础损失配置

    Example:
        >>> loss_fn = HuberMomentLoss(delta=1.0)
        >>> loss = loss_fn(params, target_moments)
    """

    def __init__(
        self,
        delta: float = 1.0,
        config: Optional[LossConfig] = None,
    ):
        """初始化Huber矩损失

        Args:
            delta: Huber损失的delta参数
            config: 损失配置
        """
        super().__init__()
        self.delta = delta
        self.config = config or LossConfig()

    def forward(
        self,
        params: Dict[str, torch.Tensor],
        target_moments: TargetMoments,
    ) -> torch.Tensor:
        """计算Huber矩损失

        Args:
            params: GMM参数
            target_moments: 目标矩

        Returns:
            总损失值
        """
        base_loss_fn = MomentMatchingLoss(self.config)
        base_loss = base_loss_fn(params, target_moments)

        abs_loss = torch.abs(base_loss)
        quadratic = torch.min(abs_loss, torch.tensor(self.delta, device=abs_loss.device))
        linear = abs_loss - quadratic

        huber_loss = 0.5 * quadratic ** 2 + self.delta * linear

        return huber_loss


def create_loss_function(
    loss_type: str = "moment_matching",
    config: Optional[LossConfig] = None,
    **kwargs,
) -> nn.Module:
    """工厂函数：创建损失函数实例

    根据类型字符串创建对应的损失函数。

    Args:
        loss_type: 损失函数类型:
                  - 'moment_matching': 标准矩匹配损失
                  - 'weighted_mse': 加权MSE损失
                  - 'huber': Huber矩损失
        config: 损失配置
        **kwargs: 传递给特定损失函数的额外参数

    Returns:
        损失函数实例

    Raises:
        ValueError: 如果loss_type无效

    Example:
        >>> loss_fn = create_loss_function("moment_matching", LossConfig(mean_weight=2.0))
        >>> huber_fn = create_loss_function("huber", delta=1.5)
    """
    loss_creators = {
        "moment_matching": lambda: MomentMatchingLoss(config),
        "weighted_mse": lambda: WeightedMSELoss(**kwargs),
        "huber": lambda: HuberMomentLoss(config=config, **kwargs),
    }

    if loss_type not in loss_creators:
        raise ValueError(
            f"无效的loss_type: {loss_type}。支持的类型: {list(loss_creators.keys())}"
        )

    loss_fn = loss_creators[loss_type]()
    logger.info(f"创建损失函数: {type(loss_fn).__name__}")

    return loss_fn
