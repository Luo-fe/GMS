"""双分量高斯混合模型参数定义

定义GMM的核心参数数据类和配置类，包括:
- GMMParameters: 存储双分量GMM的所有参数
- GMMParametersConfig: GMM参数的配置选项

支持对角协方差和全协方差两种模式，
提供参数验证、修正和与优化器无缝对接的功能。

Example:
    >>> import torch
    >>> from gms.gmm_optimization.gmm_parameters import GMMParameters
    >>>
    >>> # 创建双分量GMM参数
    >>> params = GMMParameters(
    ...     weight=0.6,
    ...     mean1=torch.tensor([1.0, 2.0]),
    ...     mean2=torch.tensor([-1.0, -2.0]),
    ...     variance1=torch.tensor([0.5, 0.5]),
    ...     variance2=torch.tensor([1.0, 1.0])
    ... )
    >>>
    >>> # 验证参数
    >>> params.validate()
    >>> print(f"维度: {params.dimensionality}, 权重2: {params.weight2:.3f}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union
import torch
import logging
import warnings

logger = logging.getLogger(__name__)


class CovarianceType(Enum):
    """协方差类型枚举

    定义GMM支持的协方差矩阵类型。
    """
    DIAGONAL = "diagonal"
    FULL = "full"


class InitializationStrategy(Enum):
    """初始化策略枚举

    定义支持的参数初始化方法。
    """
    KMEANS = "kmeans"
    RANDOM = "random"
    HEURISTIC = "heuristic"


@dataclass
class GMMParametersConfig:
    """GMM参数配置类

    配置GMM参数的创建和初始化选项。

    Attributes:
        dimensionality: 特征维度d
        covariance_type: 协方差类型 ('diagonal' 或 'full')
        initialization_strategy: 初始化策略
        min_variance: 最小方差值（用于数值稳定性）
        max_variance: 最大方差值
        weight_range: 权重w的有效范围 (min, max)
        validate_on_creation: 创建时是否自动验证
        device: 计算设备
        dtype: 数据类型

    Example:
        >>> config = GMMParametersConfig(
        ...     dimensionality=3,
        ...     covariance_type='diagonal',
        ...     initialization_strategy='kmeans'
        ... )
    """

    dimensionality: int = 2
    covariance_type: str = "diagonal"
    initialization_strategy: str = "kmeans"
    min_variance: float = 1e-6
    max_variance: float = 1e6
    weight_range: Tuple[float, float] = (1e-6, 1.0 - 1e-6)
    validate_on_creation: bool = True
    device: Union[str, torch.device] = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        """验证配置参数"""
        if self.dimensionality <= 0:
            raise ValueError(f"dimensionality必须为正整数，当前值: {self.dimensionality}")

        if self.covariance_type not in ["diagonal", "full"]:
            raise ValueError(
                f"covariance_type必须是'diagonal'或'full'，当前值: {self.covariance_type}"
            )

        if self.initialization_strategy not in ["kmeans", "random", "heuristic"]:
            raise ValueError(
                f"不支持的初始化策略: {self.initialization_strategy}"
            )

        if self.min_variance <= 0:
            raise ValueError(f"min_variance必须为正数，当前值: {self.min_variance}")

        if self.max_variance <= self.min_variance:
            raise ValueError(
                f"max_variance({self.max_variance})必须大于min_variance({self.min_variance})"
            )

        if not (0 < self.weight_range[0] < self.weight_range[1] < 1):
            raise ValueError(
                f"weight_range必须在(0,1)范围内，当前值: {self.weight_range}"
            )


@dataclass
class GMMParameters:
    """双分量高斯混合模型参数数据类

    存储双分量GMM的所有参数，包括权重、均值和协方差矩阵。
    支持对角协方差和全协方差两种模式。

    数学表示:
        p(x) = w · N(x|μ₁, Σ₁) + (1-w) · N(x|μ₂, Σ₂)

    Attributes:
        weight: 第一个分量的权重 w (0 < w < 1)
        mean1: 第一个分量的均值 μ₁，形状 (d,)
        mean2: 第二个分量的均值 μ₂，形状 (d,)
        variance1: 第一个分量的方差/协方差 Σ₁
                  - 对角模式: 形状 (d,)
                  - 全协方差模式: 形状 (d, d)
        variance2: 第二个分量的方差/协方差 Σ₂
                  - 对角模式: 形状 (d,)
                  - 全协方差模式: 形状 (d, d)

    Example:
        >>> import torch
        >>> params = GMMParameters(
        ...     weight=0.7,
        ...     mean1=torch.tensor([1.0, 0.5]),
        ...     mean2=torch.tensor([-1.0, -0.5]),
        ...     variance1=torch.tensor([0.3, 0.3]),
        ...     variance2=torch.tensor([0.8, 0.8])
        ... )
        >>>
        >>> print(f"维度: {params.dimensionality}")
        >>> print(f"是否对角协方差: {params.is_diagonal}")
        >>> print(f"第二个分量权重: {params.weight2:.3f}")
    """

    weight: float
    mean1: torch.Tensor
    mean2: torch.Tensor
    variance1: torch.Tensor
    variance2: torch.Tensor
    _config: Optional[GMMParametersConfig] = field(default=None, repr=False)

    def __post_init__(self):
        """初始化后处理"""
        if self._config is None:
            self._config = GMMParametersConfig()

        if self._config.validate_on_creation:
            try:
                self.validate()
            except ValueError as e:
                logger.warning(f"参数验证警告: {e}")

    @property
    def weight2(self) -> float:
        """获取第二个分量的权重

        Returns:
            1 - weight
        """
        return 1.0 - self.weight

    @property
    def dimensionality(self) -> int:
        """获取特征维度

        Returns:
            特征维度 d
        """
        return self.mean1.shape[-1]

    @property
    def is_diagonal(self) -> bool:
        """判断是否为对角协方差模式

        Returns:
            True如果是对角协方差，False如果是全协方差
        """
        return self.variance1.dim() == 1

    @property
    def means(self) -> torch.Tensor:
        """获取所有分量的均值（堆叠形式）

        Returns:
            形状为 (2, d) 的张量
        """
        return torch.stack([self.mean1, self.mean2], dim=0)

    @property
    def covariances(self) -> torch.Tensor:
        """获取所有分量的协方差（堆叠形式）

        Returns:
            形状为 (2, d) 或 (2, d, d) 的张量
        """
        return torch.stack([self.variance1, self.variance2], dim=0)

    @property
    def weights(self) -> torch.Tensor:
        """获取所有分量的权重（张量形式）

        Returns:
            形状为 (2,) 的张量 [w, 1-w]
        """
        return torch.tensor([self.weight, self.weight2],
                          dtype=self.mean1.dtype,
                          device=self.mean1.device)

    def validate(self) -> bool:
        """验证参数合法性

        检查以下条件:
        1. 权重在有效范围内 (0, 1)
        2. 均值向量维度一致
        3. 方差/协方差矩阵维度正确
        4. 方差值为正（正定性）
        5. 两个分量的维度匹配

        Returns:
            True 如果所有参数合法

        Raises:
            ValueError: 如果任何参数非法
        """
        config = self._config

        if not (config.weight_range[0] < self.weight < config.weight_range[1]):
            raise ValueError(
                f"权重必须在({config.weight_range[0]}, {config.weight_range[1]})范围内，"
                f"当前值: {self.weight}"
            )

        if self.mean1.shape != self.mean2.shape:
            raise ValueError(
                f"均值向量维度不一致: mean1{tuple(self.mean1.shape)} vs "
                f"mean2{tuple(self.mean2.shape)}"
            )

        d = self.dimensionality

        if self.is_diagonal:
            if self.variance1.shape != (d,) or self.variance2.shape != (d,):
                raise ValueError(
                    f"对角模式下方差形状应为({d},)，"
                    f"variance1{tuple(self.variance1.shape)}, "
                    f"variance2{tuple(self.variance2.shape)}"
                )

            if (self.variance1 <= 0).any():
                raise ValueError("variance1包含非正值")

            if (self.variance2 <= 0).any():
                raise ValueError("variance2包含非正值")
        else:
            if self.variance1.shape != (d, d) or self.variance2.shape != (d, d):
                raise ValueError(
                    f"全协方差模式下协方差形状应为({d}, {d})，"
                    f"variance1{tuple(self.variance1.shape)}, "
                    f"variance2{tuple(self.variance2.shape)}"
                )

            if not self._is_positive_definite(self.variance1):
                raise ValueError("variance1不是正定矩阵")

            if not self._is_positive_definite(self.variance2):
                raise ValueError("variance2不是正定矩阵")

        logger.debug("参数验证通过")
        return True

    def clamp(self) -> "GMMParameters":
        """修正非法参数到合法范围

        对超出范围的参数进行裁剪:
        - 权重裁剪到有效范围
        - 方差裁剪到 [min_variance, max_variance]
        - 确保协方差矩阵的正定性

        Returns:
            修正后的新GMMParameters实例（原地修改）

        Example:
            >>> params.clamp()
            >>> params.validate()  # 现在应该通过
        """
        config = self._config

        self.weight = float(torch.tensor(self.weight).clamp(
            config.weight_range[0],
            config.weight_range[1]
        ))

        with torch.no_grad():
            if self.is_diagonal:
                self.variance1 = self.variance1.clamp(
                    config.min_variance, config.max_variance
                )
                self.variance2 = self.variance2.clamp(
                    config.min_variance, config.max_variance
                )
            else:
                self.variance1 = self._ensure_positive_definite(
                    self.variance1, config.min_variance
                )
                self.variance2 = self._ensure_positive_definite(
                    self.variance2, config.min_variance
                )

        logger.debug("参数已修正到合法范围")
        return self

    def to_device(self, device: Union[str, torch.device]) -> "GMMParameters":
        """将所有张量移动到指定设备

        Args:
            device: 目标设备 ('cpu', 'cuda' 或 torch.device)

        Returns:
            新的GMMParameters实例（张量在目标设备上）
        """
        if isinstance(device, str):
            device = torch.device(device)

        return GMMParameters(
            weight=self.weight,
            mean1=self.mean1.to(device),
            mean2=self.mean2.to(device),
            variance1=self.variance1.to(device),
            variance2=self.variance2.to(device),
            _config=self._config,
        )

    def to_dtype(self, dtype: torch.dtype) -> "GMMParameters":
        """转换数据类型

        Args:
            dtype: 目标数据类型

        Returns:
            新的GMMParameters实例（使用新数据类型）
        """
        return GMMParameters(
            weight=self.weight,
            mean1=self.mean1.to(dtype),
            mean2=self.mean2.to(dtype),
            variance1=self.variance1.to(dtype),
            variance2=self.variance2.to(dtype),
            _config=self._config,
        )

    def to_optimizer_params(self) -> dict:
        """转换为优化器兼容的参数字典

        将GMMParameters转换为BaseGMMOptimizer.optimize()所需的格式。

        Returns:
            包含 'means', 'covariances', 'weights' 的字典

        Example:
            >>> optimizer = AdamOptimizer(OptimizationConfig())
            >>> result = optimizer.optimize(target_moments, params.to_optimizer_params())
        """
        return {
            'means': self.means.detach().clone(),
            'covariances': self.covariances.detach().clone(),
            'weights': self.weights.detach().clone(),
        }

    @classmethod
    def from_optimizer_params(
        cls,
        optimized_params: "OptimizedParams",
        config: Optional[GMMParametersConfig] = None
    ) -> "GMMParameters":
        """从优化器结果创建GMMParameters

        将OptimizedParams转换回GMMParameters格式。

        Args:
            optimized_params: OptimizedParams实例
            config: 可选的配置

        Returns:
            GMMParameters实例

        Example:
            >>> result = optimizer.optimize(target_moments, initial_params)
            >>> gmm_params = GMMParameters.from_optimizer_params(result)
        """
        if optimized_params.means is None or optimized_params.weights is None:
            raise ValueError("OptimizedParams缺少必需的字段")

        means = optimized_params.means
        covariances = optimized_params.covariances
        weights = optimized_params.weights

        return cls(
            weight=float(weights[0]),
            mean1=means[0],
            mean2=means[1],
            variance1=covariances[0] if covariances is not None else torch.eye(means.shape[1]),
            variance2=covariances[1] if covariances is not None else torch.eye(means.shape[1]),
            _config=config,
        )

    def to_dict(self) -> dict:
        """转换为字典格式（用于序列化）

        Returns:
            包含所有参数的字典
        """
        return {
            'weight': self.weight,
            'mean1': self.mean1.cpu().numpy().tolist(),
            'mean2': self.mean2.cpu().numpy().tolist(),
            'variance1': self.variance1.cpu().numpy().tolist(),
            'variance2': self.variance2.cpu().numpy().tolist(),
            'is_diagonal': self.is_diagonal,
            'dimensionality': self.dimensionality,
        }

    @staticmethod
    def _is_positive_definite(matrix: torch.Tensor) -> bool:
        """检查矩阵是否正定

        使用Cholesky分解检验正定性。

        Args:
            matrix: 方阵

        Returns:
            True如果正定
        """
        try:
            torch.linalg.cholesky(matrix)
            return True
        except RuntimeError:
            return False

    @staticmethod
    def _ensure_positive_definite(
        matrix: torch.Tensor,
        min_eigenvalue: float = 1e-6
    ) -> torch.Tensor:
        """确保矩阵正定

        通过特征值分解修正非正定矩阵。

        Args:
            matrix: 输入方阵
            min_eigenvalue: 最小特征值

        Returns:
            正定矩阵
        """
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        eigenvalues = eigenvalues.clamp(min=min_eigenvalue)

        corrected = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        return (corrected + corrected.T) / 2  # 确保对称性

    def __repr__(self) -> str:
        """返回可读的字符串表示"""
        return (
            f"GMMParameters("
            f"w={self.weight:.4f}, "
            f"d={self.dimensionality}, "
            f"diag={self.is_diagonal}, "
            f"μ1={[f'{x:.3f}' for x in self.mean1.tolist()]}, "
            f"μ2={[f'{x:.3f}' for x in self.mean2.tolist()]})"
        )
