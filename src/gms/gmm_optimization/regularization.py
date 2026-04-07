"""GMM优化器正则化和稳定性约束模块"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import warnings
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RegularizationConfig:
    """正则化配置数据类

    Attributes:
        l2_lambda: L2正则化系数
        l1_lambda: L1正则化系数（可选）
        variance_floor: 方差下界（防止退化）
        weight_clamp_min: 权重最小值
        weight_clamp_max: 权重最大值
        mean_range: 均值范围限制 (min, max)，None表示不限制
        use_sigmoid_weights: 是否使用sigmoid约束权重到(0,1)
        entropy_regularization: 熵正则化系数（鼓励均匀混合）
        correlation_penalty: 相关性惩罚系数

    Example:
        >>> config = RegularizationConfig(
        ...     l2_lambda=0.01,
        ...     variance_floor=1e-6,
        ...     use_sigmoid_weights=True
        ... )
    """

    l2_lambda: float = 1e-4
    l1_lambda: float = 0.0
    variance_floor: float = 1e-6
    weight_clamp_min: float = 0.001
    weight_clamp_max: float = 0.999
    mean_range: Optional[Tuple[float, float]] = None
    use_sigmoid_weights: bool = False
    entropy_regularization: float = 0.0
    correlation_penalty: float = 0.0

    def __post_init__(self):
        """验证配置参数"""
        if self.l2_lambda < 0:
            raise ValueError(f"l2_lambda不能为负，当前值: {self.l2_lambda}")
        if self.l1_lambda < 0:
            raise ValueError(f"l1_lambda不能为负，当前值: {self.l1_lambda}")
        if self.variance_floor <= 0:
            raise ValueError(f"variance_floor必须为正数，当前值: {self.variance_floor}")
        if self.weight_clamp_min < 0 or self.weight_clamp_max > 1:
            raise ValueError("weight_clamp必须在[0,1]范围内")
        if self.weight_clamp_min >= self.weight_clamp_max:
            raise ValueError("weight_clamp_min必须小于weight_clamp_max")
        if self.mean_range is not None:
            if len(self.mean_range) != 2 or self.mean_range[0] >= self.mean_range[1]:
                raise ValueError("mean_range必须是(min, max)元组且min < max")


class RegularizationTerm(nn.Module):
    """正则化项计算模块

    实现多种正则化策略以防止过拟合和数值不稳定。

    支持的正则化类型:
        - L2 正则化: λ||θ||² (防止参数过大)
        - L1 正则化: λ||θ||₁ (稀疏性)
        - 方差下界约束: σ² ≥ ε (防止退化)
        - 权重范围约束: sigmoid(w) 或 clamp
        - 均值范围限制 (可选)
        - 熵正则化: 鼓励均匀混合
        - 协方差相关性惩罚

    Attributes:
        config: 正则化配置

    Example:
        >>> reg = RegularizationTerm(RegularizationConfig(l2_lambda=0.01))
        >>> params = {'means': ..., 'covariances': ..., 'weights': ...}
        >>> reg_loss = reg(params)
    """

    def __init__(self, config: Optional[RegularizationConfig] = None):
        """初始化正则化项

        Args:
            config: 正则化配置，如果为None则使用默认配置
        """
        super().__init__()
        self.config = config or RegularizationConfig()

    def forward(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算总正则化损失

        Args:
            params: GMM参数字典

        Returns:
            总正则化损失值
        """
        device = self._get_device(params)
        total_reg = torch.tensor(0.0, device=device)

        total_reg = total_reg + self._compute_l2_regularization(params)
        total_reg = total_reg + self._compute_l1_regularization(params)
        total_reg = total_reg + self._compute_variance_penalty(params)
        total_reg = total_reg + self._compute_entropy_regularization(params)
        total_reg = total_reg + self._compute_correlation_penalty(params)

        return total_reg

    def __call__(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使实例可调用"""
        return self.forward(params)

    def _get_device(self, params: Dict[str, torch.Tensor]) -> torch.device:
        """获取设备信息"""
        for param in params.values():
            return param.device
        return torch.device('cpu')

    def _compute_l2_regularization(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算L2正则化项

        λ||θ||²

        Args:
            params: 参数字典

        Returns:
            L2正则化损失
        """
        if self.config.l2_lambda == 0:
            return torch.tensor(0.0, device=self._get_device(params))

        l2_norm_sq = torch.tensor(0.0, device=self._get_device(params))

        for name, param in params.items():
            if name == 'weights' and self.config.use_sigmoid_weights:
                continue
            l2_norm_sq = l2_norm_sq + torch.sum(param ** 2)

        return self.config.l2_lambda * l2_norm_sq

    def _compute_l1_regularization(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算L1正则化项

        λ||θ||₁

        Args:
            params: 参数字典

        Returns:
            L1正则化损失
        """
        if self.config.l1_lambda == 0:
            return torch.tensor(0.0, device=self._get_device(params))

        l1_norm = torch.tensor(0.0, device=self._get_device(params))

        for param in params.values():
            l1_norm = l1_norm + torch.sum(torch.abs(param))

        return self.config.l1_lambda * l1_norm

    def _compute_variance_penalty(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算方差下界惩罚

        对低于下界的方差施加惩罚。

        Args:
            params: 参数字典

        Returns:
            方差惩罚项
        """
        covariances = params.get('covariances')
        if covariances is None:
            return torch.tensor(0.0, device=self._get_device(params))

        floor = self.config.variance_floor

        if covariances.dim() == 3:
            eigenvalues = torch.linalg.eigvalsh(covariances)
            violations = torch.clamp(floor - eigenvalues, min=0)
            penalty = torch.sum(violations ** 2)
        elif covariances.dim() == 2:
            diag = torch.diagonal(covariances, dim1=-2, dim2=-1)
            violations = torch.clamp(floor - diag, min=0)
            penalty = torch.sum(violations ** 2)
        elif covariances.dim() == 1:
            violations = torch.clamp(floor - covariances, min=0)
            penalty = torch.sum(violations ** 2)
        else:
            penalty = torch.tensor(0.0, device=covariances.device)

        return penalty

    def _compute_entropy_regularization(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算熵正则化

        鼓励混合权重更均匀分布。
        H(w) = -Σ wᵢ log(wᵢ)，最大化熵。

        Args:
            params: 参数字典

        Returns:
            熵正则化损失（负熵）
        """
        if self.config.entropy_regularization == 0:
            return torch.tensor(0.0, device=self._get_device(params))

        weights = params.get('weights')
        if weights is None:
            return torch.tensor(0.0, device=self._get_device(params))

        eps = 1e-10
        weights_clamped = torch.clamp(weights, min=eps)
        entropy = -torch.sum(weights_clamped * torch.log(weights_clamped))

        return -self.config.entropy_regularization * entropy

    def _compute_correlation_penalty(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算协方差相关性惩罚

        惩罚协方差矩阵的非对角元素（鼓励独立性）。

        Args:
            params: 参数字典

        Returns:
            相关性惩罚项
        """
        if self.config.correlation_penalty == 0:
            return torch.tensor(0.0, device=self._get_device(params))

        covariances = params.get('covariances')
        if covariances is None or covariances.dim() != 3:
            return torch.tensor(0.0, device=self._get_device(params))

        n_components, n_features, _ = covariances.shape
        off_diag_sum = torch.tensor(0.0, device=covariances.device)

        for k in range(n_components):
            cov_k = covariances[k]
            off_diag = cov_k - torch.diag(torch.diag(cov_k))
            off_diag_sum = off_diag_sum + torch.sum(off_diag ** 2)

        return self.config.correlation_penalty * off_diag_sum

    def apply_constraints(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """应用硬约束到参数

        执行不可微分的约束操作（如裁剪）。

        Args:
            params: 参数字典

        Returns:
            约束后的参数字典
        """
        constrained_params = {}

        for name, param in params.items():
            constrained_param = param.clone()

            if name == 'means':
                constrained_param = self._constrain_means(constrained_param)
            elif name == 'covariances':
                constrained_param = self._constrain_covariances(constrained_param)
            elif name == 'weights':
                constrained_param = self._constrain_weights(constrained_param)

            constrained_params[name] = constrained_param

        return constrained_params

    def _constrain_means(self, means: torch.Tensor) -> torch.Tensor:
        """约束均值在指定范围内

        Args:
            means: 均值张量

        Returns:
            裁剪后的均值
        """
        if self.config.mean_range is not None:
            means = torch.clamp(
                means,
                min=self.config.mean_range[0],
                max=self.config.mean_range[1]
            )
        return means

    def _constrain_covariances(self, covariances: torch.Tensor) -> torch.Tensor:
        """确保协方差矩阵满足约束

        - 对角线元素 >= variance_floor
        - 确保半正定性

        Args:
            covariances: 协方差张量

        Returns:
            约束后的协方差
        """
        floor = self.config.variance_floor

        if covariances.dim() == 3:
            n_comp, n_feat, _ = covariances.shape
            identity = torch.eye(n_feat, device=covariances.device, dtype=covariances.dtype)

            for k in range(n_comp):
                eigvals, eigvecs = torch.linalg.eigh(covariances[k])
                eigvals = torch.clamp(eigvals, min=floor)
                covariances[k] = eigvecs @ torch.diag(eigvals) @ eigvecs.T

        elif covariances.dim() == 2:
            n_feat = covariances.shape[1]
            identity = torch.eye(n_feat, device=covariances.device, dtype=covariances.dtype)

            eigvals, eigvecs = torch.linalg.eigh(covariances)
            eigvals = torch.clamp(eigvals, min=floor)
            covariances = eigvecs @ torch.diag(eigvals) @ eigvecs.T

        elif covariances.dim() == 1:
            covariances = torch.clamp(covariances, min=floor)

        return covariances

    def _constrain_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """约束混合权重

        如果use_sigmoid_weights=True，应用sigmoid变换。
        否则简单裁剪并归一化。

        Args:
            weights: 权重张量

        Returns:
            约束后的权重
        """
        if self.config.use_sigmoid_weights:
            weights = torch.sigmoid(weights)
        else:
            weights = torch.clamp(
                weights,
                min=self.config.weight_clamp_min,
                max=self.config.weight_clamp_max
            )

        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum

        return weights


class StabilityConstraints:
    """稳定性约束和检查器

    提供数值稳定性检测、参数合法性验证和自动修正功能。

    Attributes:
        config: 正则化配置（用于获取阈值）

    Example:
        >>> checker = StabilityConstraints()
        >>> is_valid, issues = checker.check_params(params)
        >>> corrected_params = checker.correct_params(params)
    """

    def __init__(
        self,
        config: Optional[RegularizationConfig] = None,
        tolerance: float = 1e-6,
    ):
        """初始化稳定性检查器

        Args:
            config: 正则化配置
            tolerance: 数值容差
        """
        self.config = config or RegularizationConfig()
        self.tolerance = tolerance

    def check_params(self, params: Dict[str, torch.Tensor]) -> Tuple[bool, List[str]]:
        """检查参数的合法性和数值稳定性

        Args:
            params: GMM参数字典

        Returns:
            (是否合法, 问题列表)
        """
        issues = []
        is_valid = True

        if 'weights' in params:
            valid, weight_issues = self._check_weights(params['weights'])
            if not valid:
                is_valid = False
                issues.extend(weight_issues)

        if 'covariances' in params:
            valid, cov_issues = self._check_covariances(params['covariances'])
            if not valid:
                is_valid = False
                issues.extend(cov_issues)

        if 'means' in params:
            valid, mean_issues = self._check_means(params['means'])
            if not valid:
                is_valid = False
                issues.extend(mean_issues)

        return is_valid, issues

    def _check_weights(self, weights: torch.Tensor) -> Tuple[bool, List[str]]:
        """检查权重的合法性

        Args:
            weights: 权重张量

        Returns:
            (是否合法, 问题列表)
        """
        issues = []

        if torch.isnan(weights).any():
            issues.append("权重包含NaN值")
            return False, issues

        if torch.isinf(weights).any():
            issues.append("权重包含无穷值")
            return False, issues

        if (weights < 0).any():
            issues.append(f"存在负权重: 最小值={weights.min().item():.6f}")

        if abs(weights.sum().item() - 1.0) > self.tolerance:
            issues.append(f"权重和不等于1: sum={weights.sum().item():.6f}")

        return len(issues) == 0, issues

    def _check_covariances(self, covariances: torch.Tensor) -> Tuple[bool, List[str]]:
        """检查协方差矩阵的合法性

        Args:
            covariances: 协方差张量

        Returns:
            (是否合法, 问题列表)
        """
        issues = []

        if torch.isnan(covariances).any():
            issues.append("协方差包含NaN值")
            return False, issues

        if torch.isinf(covariances).any():
            issues.append("协方差包含无穷值")
            return False, issues

        if covariances.dim() >= 2:
            if covariances.dim() == 3:
                for k in range(covariances.shape[0]):
                    cov_k = covariances[k]
                    if not self._is_positive_definite(cov_k):
                        issues.append(f"分量{k}的协方差矩阵不是正定的")
                        break
            elif covariances.dim() == 2:
                if not self._is_positive_definite(covariances):
                    issues.append("协方差矩阵不是正定的")

        if covariances.dim() >= 1:
            min_val = covariances.min().item()
            if min_val < 0:
                issues.append(f"协方差存在负值: {min_val:.6f}")

            if abs(min_val) < self.config.variance_floor and min_val > 0:
                issues.append(f"协方差接近零下界: {min_val:.6e}")

        return len(issues) == 0, issues

    def _check_means(self, means: torch.Tensor) -> Tuple[bool, List[str]]:
        """检查均值的合法性

        Args:
            means: 均值张量

        Returns:
            (是否合法, 问题列表)
        """
        issues = []

        if torch.isnan(means).any():
            issues.append("均值包含NaN值")
            return False, issues

        if torch.isinf(means).any():
            issues.append("均值包含无穷值")
            return False, issues

        if self.config.mean_range is not None:
            min_val, max_val = self.config.mean_range
            if (means < min_val).any():
                issues.append(f"均值超出下界: 最小值={means.min().item():.4f}")
            if (means > max_val).any():
                issues.append(f"均值超出上界: 最大值={means.max().item():.4f}")

        return len(issues) == 0, issues

    def _is_positive_definite(self, matrix: torch.Tensor) -> bool:
        """检查矩阵是否正定

        使用特征值分解判断正定性。

        Args:
            matrix: 方阵

        Returns:
            是否正定
        """
        try:
            eigvals = torch.linalg.eigvalsh(matrix)
            return bool((eigvals > -self.tolerance).all())
        except Exception:
            return False

    def correct_params(
        self,
        params: Dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """自动修正非法参数

        对检测到的数值问题进行自动修复。

        Args:
            params: 参数字典
            inplace: 是否原地修改

        Returns:
            修正后的参数字典
        """
        if not inplace:
            params = {k: v.clone() for k, v in params.items()}

        if 'weights' in params:
            params['weights'] = self._correct_weights(params['weights'])

        if 'covariances' in params:
            params['covariances'] = self._correct_covariances(params['covariances'])

        if 'means' in params:
            params['means'] = self._correct_means(params['means'])

        return params

    def _correct_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """修正权重问题

        Args:
            weights: 权重张量

        Returns:
            修正后的权重
        """
        weights = torch.where(torch.isnan(weights), torch.ones_like(weights), weights)
        weights = torch.where(torch.isinf(weights), torch.ones_like(weights), weights)
        weights = torch.clamp(weights, min=0.0)
        weight_sum = weights.sum()

        if weight_sum == 0:
            weights = torch.ones_like(weights) / len(weights)
        else:
            weights = weights / weight_sum

        return weights

    def _correct_covariances(self, covariances: torch.Tensor) -> torch.Tensor:
        """修正协方差问题

        Args:
            covariances: 协方差张量

        Returns:
            修正后的协方差
        """
        covariances = torch.where(
            torch.isnan(covariances),
            torch.ones_like(covariances) * self.config.variance_floor,
            covariances
        )
        covariances = torch.where(
            torch.isinf(covariances),
            torch.ones_like(covariances),
            covariances
        )

        reg_term = RegularizationTerm(self.config)
        covariances = reg_term._constrain_covariances(covariances)

        return covariances

    def _correct_means(self, means: torch.Tensor) -> torch.Tensor:
        """修正均值问题

        Args:
            means: 均值张量

        Returns:
            修正后的均值
        """
        means = torch.where(torch.isnan(means), torch.zeros_like(means), means)
        means = torch.where(torch.isinf(means), torch.zeros_like(means), means)

        if self.config.mean_range is not None:
            means = torch.clamp(
                means,
                min=self.config.mean_range[0],
                max=self.config.mean_range[1]
            )

        return means

    def detect_numerical_issues(self, tensor: torch.Tensor, name: str = "tensor") -> List[str]:
        """检测张量的数值问题

        Args:
            tensor: 要检查的张量
            name: 张量名称（用于日志）

        Returns:
            发现的问题列表
        """
        issues = []

        if torch.isnan(tensor).any():
            count = torch.isnan(tensor).sum().item()
            issues.append(f"{name}: 包含{count}个NaN值")

        if torch.isinf(tensor).any():
            pos_inf = (tensor == float('inf')).sum().item()
            neg_inf = (tensor == float('-inf')).sum().item()
            issues.append(f"{name}: 包含{pos_inf}个+∞和{neg_inf}个-∞值")

        if tensor.numel() > 0:
            max_val = tensor.max().item()
            min_val = tensor.min().item()

            if abs(max_val) > 1e10:
                issues.append(f"{name}: 最大值过大 ({max_val:.2e})")
            if abs(min_val) > 1e10 and abs(min_val) != float('inf'):
                issues.append(f"{name}: 最小值的绝对值过大 ({min_val:.2e})")

        return issues
