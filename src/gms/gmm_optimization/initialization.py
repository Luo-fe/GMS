"""GMM参数初始化策略

提供多种参数初始化方法，用于为优化器提供良好的起始点:
- KMeansInitializer: K-means++聚类初始化（推荐）
- RandomInitializer: 随机初始化（用于基准对比）
- HeuristicInitializer: 启发式规则初始化

好的初始化可以显著提高:
- 优化收敛速度
- 最终解的质量
- 对局部最优的鲁棒性

Example:
    >>> import torch
    >>> from gms.gmm_optimization import (
    ...     GMMParameters,
    ...     KMeansInitializer,
    ...     create_initializer
    ... )
    >>>
    >>> # 生成示例数据
    >>> data = torch.randn(1000, 2)
    >>>
    >>> # 使用K-means++初始化
    >>> initializer = KMeansInitializer(n_components=2, max_iter=100)
    >>> params = initializer.initialize(data)
    >>>
    >>> # 或使用工厂函数
    >>> initializer = create_initializer('kmeans')
    >>> params = initializer.initialize(data)
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import logging
from .gmm_parameters import GMMParameters, GMMParametersConfig, InitializationStrategy

logger = logging.getLogger(__name__)


class BaseInitializer(ABC):
    """初始化器抽象基类

    定义所有初始化器的通用接口。

    子类需要实现:
    - initialize(): 执行初始化并返回GMMParameters
    """

    @abstractmethod
    def initialize(
        self,
        data: torch.Tensor,
        seed: Optional[int] = None,
    ) -> GMMParameters:
        """执行参数初始化

        Args:
            data: 训练数据，形状 (n_samples, n_features)
            seed: 可选的随机种子

        Returns:
            初始化后的GMMParameters实例
        """
        pass


class KMeansInitializer(BaseInitializer):
    """K-means++初始化器

    使用K-means++算法进行智能初始化，通常能提供高质量的初始参数。

    算法流程:
    1. K-means++选择初始中心点（概率与距离平方成正比）
    2. 迭代运行标准K-means直到收敛
    3. 从聚类结果估计GMM参数

    Attributes:
        n_components: 聚类数量（固定为2）
        max_iter: K-means最大迭代次数
        tol: 收敛阈值
        n_init: 运行K-means的次数（选择最佳）
        distance_metric: 距离度量函数
        random_state: 随机种子

    Example:
        >>> initializer = KMeansInitializer(
        ...     n_components=2,
        ...     max_iter=100,
        ...     n_init=10
        ... )
        >>> params = initializer.initialize(data, seed=42)
    """

    def __init__(
        self,
        n_components: int = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 10,
        distance_metric: str = 'euclidean',
        random_state: Optional[int] = None,
    ):
        """初始化K-means++初始化器

        Args:
            n_components: 分量数量（必须为2）
            max_iter: 每次运行的最大迭代次数
            tol: 收敛阈值（中心点变化小于此值时停止）
            n_init: 使用不同初始值运行的次数
            distance_metric: 距离度量 ('euclidean', 'manhattan', 'cosine')
            random_state: 随机种子
        """
        if n_components != 2:
            raise ValueError(f"双分量GMM要求n_components=2，当前值: {n_components}")

        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.distance_metric = distance_metric
        self.random_state = random_state

        logger.debug(
            f"KMeansInitializer创建: "
            f"n={n_components}, max_iter={max_iter}, n_init={n_init}"
        )

    def initialize(
        self,
        data: torch.Tensor,
        seed: Optional[int] = None,
    ) -> GMMParameters:
        """执行K-means++初始化

        Args:
            data: 输入数据，形状 (n_samples, d)
            seed: 随机种子（覆盖构造函数中的设置）

        Returns:
            初始化后的GMMParameters
        """
        actual_seed = seed or self.random_state

        best_params = None
        best_inertia = float('inf')

        for run in range(self.n_init):
            run_seed = None if actual_seed is None else actual_seed + run * 1000

            try:
                centers, labels, inertia = self._kmeans_plusplus(
                    data, seed=run_seed
                )

                params = self._estimate_gmm_params(data, centers, labels)

                if inertia < best_inertia:
                    best_inertia = inertia
                    best_params = params

                logger.debug(
                    f"K-means运行 {run+1}/{self.n_init}: "
                    f"inertia={inertia:.4f}"
                )

            except Exception as e:
                logger.warning(f"K-means运行 {run+1} 失败: {e}")
                continue

        if best_params is None:
            logger.error("所有K-means运行都失败，回退到随机初始化")
            return RandomInitializer().initialize(data, seed=actual_seed)

        logger.info(
            f"K-means++初始化完成: 最佳inertia={best_inertia:.4f}, "
            f"权重={best_params.weight:.4f}"
        )

        return best_params

    def _kmeans_plusplus(
        self,
        data: torch.Tensor,
        seed: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """执行K-means++算法

        Args:
            data: 输入数据
            seed: 随机种子

        Returns:
            (centers, labels, inertia) 元组
        """
        n_samples, n_features = data.shape
        device = data.device

        if seed is not None:
            torch.manual_seed(seed)

        centers = torch.empty(self.n_components, n_features, device=device)

        idx_0 = torch.randint(0, n_samples, (1,)).item()
        centers[0] = data[idx_0]

        for k in range(1, self.n_components):
            distances = self._compute_distances_to_centers(data, centers[:k])
            min_distances, _ = distances.min(dim=1)

            probs = min_distances ** 2
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs = probs / probs_sum
            else:
                probs = torch.ones(n_samples, device=device) / n_samples

            idx_k = torch.multinomial(probs.unsqueeze(0), 1).squeeze().item()
            centers[k] = data[idx_k]

        labels, inertia = self._assign_clusters(data, centers)

        for iteration in range(self.max_iter):
            new_centers = self._update_centers(data, labels)

            shift = torch.norm(new_centers - centers).item()
            centers = new_centers

            labels, new_inertia = self._assign_clusters(data, centers)

            if abs(inertia - new_inertia) < self.tol:
                logger.debug(f"K-means收敛于第{iteration+1}次迭代")
                break

            inertia = new_inertia

        return centers, labels, inertia

    def _compute_distances_to_centers(
        self,
        data: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        """计算数据点到各中心的距离"""
        if self.distance_metric == 'euclidean':
            diff = data.unsqueeze(1) - centers.unsqueeze(0)
            return (diff ** 2).sum(dim=2)
        elif self.distance_metric == 'manhattan':
            diff = data.unsqueeze(1) - centers.unsqueeze(0)
            return diff.abs().sum(dim=2)
        elif self.distance_metric == 'cosine':
            data_norm = F.normalize(data, p=2, dim=1)
            centers_norm = F.normalize(centers, p=2, dim=1)
            similarity = torch.mm(data_norm, centers_norm.T)
            return 1.0 - similarity
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")

    def _assign_clusters(
        self,
        data: torch.Tensor,
        centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """分配样本到最近的中心点"""
        distances = self._compute_distances_to_centers(data, centers)
        labels = distances.argmin(dim=1)

        min_distances, _ = distances.min(dim=1)
        inertia = min_distances.sum().item()

        return labels, inertia

    def _update_centers(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """更新中心点为簇均值"""
        new_centers = torch.zeros(self.n_components, data.shape[1], device=data.device, dtype=data.dtype)

        for k in range(self.n_components):
            mask = (labels == k)
            if mask.sum() > 0:
                new_centers[k] = data[mask].mean(dim=0)
            else:
                new_centers[k] = data[
                    torch.randint(0, data.shape[0], (1,)).item()
                ]

        return new_centers

    def _estimate_gmm_params(
        self,
        data: torch.Tensor,
        centers: torch.Tensor,
        labels: torch.Tensor,
    ) -> GMMParameters:
        """从聚类结果估计GMM参数

        Args:
            data: 原始数据
            centers: 聚类中心
            labels: 聚类标签

        Returns:
            GMMParameters实例
        """
        config = GMMParametersConfig(dimensionality=data.shape[1])

        weights = []
        means = []
        variances = []

        for k in range(self.n_components):
            mask = (labels == k)
            cluster_data = data[mask]

            n_k = cluster_data.shape[0]
            weight_k = n_k / data.shape[0]
            weights.append(weight_k)

            mean_k = cluster_data.mean(dim=0)
            means.append(mean_k)

            if config.covariance_type == 'diagonal':
                var_k = cluster_data.var(dim=0, unbiased=False)
                var_k = var_k.clamp(min=config.min_variance)
                variances.append(var_k)
            else:
                cov_k = torch.cov(cluster_data.T) if cluster_data.shape[0] > 1 else torch.eye(data.shape[1])
                cov_k = (cov_k + cov_k.T) / 2
                eigenvalues = torch.linalg.eigvalsh(cov_k)
                if (eigenvalues <= 0).any():
                    cov_k = cov_k + (abs(eigenvalues.min()) + config.min_variance) * torch.eye(data.shape[1])
                variances.append(cov_k)

        params = GMMParameters(
            weight=float(weights[0]),
            mean1=means[0],
            mean2=means[1],
            variance1=variances[0],
            variance2=variances[1],
            _config=config,
        )

        return params


class RandomInitializer(BaseInitializer):
    """随机初始化器

    使用随机方式生成初始参数，主要用于:
    - 基准对比
    - 测试优化器的鲁棒性
    - 多起点优化的随机起点

    Attributes:
        weight_range: 权重范围 (min, max)
        mean_range: 均值范围 (min, max) 或 (min, max, d)
        variance_range: 方差范围 (min, max)
        seed: 随机种子

    Example:
        >>> initializer = RandomInitializer(
        ...     weight_range=(0.3, 0.7),
        ...     mean_range=(-5.0, 5.0),
        ...     seed=42
        ... )
        >>> params = initializer.initialize(data)
    """

    def __init__(
        self,
        weight_range: Tuple[float, float] = (0.3, 0.7),
        mean_range: Tuple[float, float] = (-3.0, 3.0),
        variance_range: Tuple[float, float] = (0.5, 2.0),
        seed: Optional[int] = None,
    ):
        """初始化随机初始化器

        Args:
            weight_range: 权重w的范围
            mean_range: 均值的范围
            variance_range: 方差的范围
            seed: 随机种子（用于可复现性）
        """
        self.weight_range = weight_range
        self.mean_range = mean_range
        self.variance_range = variance_range
        self.seed = seed

        logger.debug("RandomInitializer创建完成")

    def initialize(
        self,
        data: torch.Tensor,
        seed: Optional[int] = None,
    ) -> GMMParameters:
        """执行随机初始化

        Args:
            data: 输入数据（仅用于获取维度信息）
            seed: 随机种子

        Returns:
            随机初始化的GMMParameters
        """
        actual_seed = seed or self.seed
        if actual_seed is not None:
            torch.manual_seed(actual_seed)

        n_features = data.shape[1]
        device = data.device
        dtype = data.dtype

        config = GMMParametersConfig(dimensionality=n_features)

        weight = torch.empty(1).uniform_(self.weight_range[0], self.weight_range[1]).item()

        mean1 = torch.empty(n_features).uniform_(self.mean_range[0], self.mean_range[1]).to(device).to(dtype)
        mean2 = torch.empty(n_features).uniform_(self.mean_range[0], self.mean_range[1]).to(device).to(dtype)

        if config.covariance_type == 'diagonal':
            variance1 = torch.empty(n_features).uniform_(self.variance_range[0], self.variance_range[1]).to(device).to(dtype)
            variance2 = torch.empty(n_features).uniform_(self.variance_range[0], self.variance_range[1]).to(device).to(dtype)
        else:
            var1_val = torch.empty(1).uniform_(self.variance_range[0], self.variance_range[1]).item()
            var2_val = torch.empty(1).uniform_(self.variance_range[0], self.variance_range[1]).item()
            variance1 = var1_val * torch.eye(n_features, device=device, dtype=dtype)
            variance2 = var2_val * torch.eye(n_features, device=device, dtype=dtype)

        params = GMMParameters(
            weight=weight,
            mean1=mean1,
            mean2=mean2,
            variance1=variance1,
            variance2=variance2,
            _config=config,
        )

        logger.info(f"随机初始化完成: w={params.weight:.4f}")
        return params


class HeuristicInitializer(BaseInitializer):
    """启发式初始化器

    基于数据的统计特性进行启发式初始化，
    使用四分位数分割等规则。

    适用于:
        - 数据有明显分离特征的情况
        - 快速但质量中等的初始化需求
        - 作为其他方法的备选方案

    Example:
        >>> initializer = HeuristicInitializer(method='quantile')
        >>> params = initializer.initialize(data)
    """

    def __init__(
        self,
        method: str = 'quantile',
        quantile_threshold: float = 0.5,
    ):
        """初始化启发式初始化器

        Args:
            method: 启发式方法 ('quantile', 'percentile', 'statistical')
            quantile_threshold: 用于分割的分位数值
        """
        self.method = method
        self.quantile_threshold = quantile_threshold

        logger.debug(f"HeuristicInitializer创建: method={method}")

    def initialize(
        self,
        data: torch.Tensor,
        seed: Optional[int] = None,
    ) -> GMMParameters:
        """执行启发式初始化

        Args:
            data: 输入数据
            seed: 未使用（保留接口一致性）

        Returns:
            初始化后的GMMParameters
        """
        n_samples, n_features = data.shape
        device = data.device
        dtype = data.dtype

        config = GMMParametersConfig(dimensionality=n_features)

        if self.method == 'quantile':
            params = self._initialize_by_quantile(data, config)
        elif self.method == 'percentile':
            params = self._initialize_by_percentile(data, config)
        elif self.method == 'statistical':
            params = self._initialize_by_statistics(data, config)
        else:
            raise ValueError(f"不支持的启发式方法: {self.method}")

        logger.info(f"启发式初始化完成 ({self.method}): w={params.weight:.4f}")
        return params

    def _initialize_by_quantile(
        self,
        data: torch.Tensor,
        config: GMMParametersConfig,
    ) -> GMMParameters:
        """基于分位数分割的初始化"""
        projected = data.mean(dim=1)

        threshold = torch.quantile(projected, self.quantile_threshold)

        mask_lower = (projected <= threshold)
        mask_upper = (projected > threshold)

        group1 = data[mask_lower]
        group2 = data[mask_upper]

        weight = len(group1) / len(data)

        mean1 = group1.mean(dim=0) if len(group1) > 0 else data.mean(dim=0) - torch.ones(config.dimensionality, device=data.device)
        mean2 = group2.mean(dim=0) if len(group2) > 0 else data.mean(dim=0) + torch.ones(config.dimensionality, device=data.device)

        if config.covariance_type == 'diagonal':
            var1 = group1.var(dim=0, unbiased=False).clamp(min=config.min_variance) if len(group1) > 1 else torch.ones(config.dimensionality, device=data.device)
            var2 = group2.var(dim=0, unbiased=False).clamp(min=config.min_variance) if len(group2) > 1 else torch.ones(config.dimensionality, device=data.device)
        else:
            var1 = self._safe_covariance(group1, config) if len(group1) > 1 else torch.eye(config.dimensionality, device=data.device)
            var2 = self._safe_covariance(group2, config) if len(group2) > 1 else torch.eye(config.dimensionality, device=data.device)

        return GMMParameters(
            weight=float(weight),
            mean1=mean1,
            mean2=mean2,
            variance1=var1,
            variance2=var2,
            _config=config,
        )

    def _initialize_by_percentile(
        self,
        data: torch.Tensor,
        config: GMMParametersConfig,
    ) -> GMMParameters:
        """基于百分位的初始化"""
        p25 = torch.quantile(data, 0.25, dim=0)
        p75 = torch.quantile(data, 0.75, dim=0)

        median = torch.median(data, dim=0).values
        std = data.std(dim=0)

        weight = 0.5

        mean1 = p25.clone()
        mean2 = p75.clone()

        if config.covariance_type == 'diagonal':
            var1 = (std ** 2).clamp(min=config.min_variance)
            var2 = (std ** 2).clamp(min=config.min_variance)
        else:
            var1 = torch.diag((std ** 2).clamp(min=config.min_variance))
            var2 = torch.diag((std ** 2).clamp(min=config.min_variance))

        return GMMParameters(
            weight=weight,
            mean1=mean1,
            mean2=mean2,
            variance1=var1,
            variance2=var2,
            _config=config,
        )

    def _initialize_by_statistics(
        self,
        data: torch.Tensor,
        config: GMMParametersConfig,
    ) -> GMMParameters:
        """基于统计特性的初始化"""
        overall_mean = data.mean(dim=0)
        overall_std = data.std(dim=0)

        offset = overall_std * 1.5

        mean1 = overall_mean - offset
        mean2 = overall_mean + offset

        weight = 0.5

        if config.covariance_type == 'diagonal':
            var1 = (overall_std ** 2).clamp(min=config.min_variance)
            var2 = (overall_std ** 2).clamp(min=config.min_variance)
        else:
            var1 = torch.diag((overall_std ** 2).clamp(min=config.min_variance))
            var2 = torch.diag((overall_std ** 2).clamp(min=config.min_variance))

        return GMMParameters(
            weight=weight,
            mean1=mean1,
            mean2=mean2,
            variance1=var1,
            variance2=var2,
            _config=config,
        )

    @staticmethod
    def _safe_covariance(
        data: torch.Tensor,
        config: GMMParametersConfig,
    ) -> torch.Tensor:
        """安全计算协方差矩阵（确保正定）"""
        if data.shape[0] <= 1:
            return torch.eye(data.shape[1])

        cov = torch.cov(data.T)
        cov = (cov + cov.T) / 2

        eigenvalues = torch.linalg.eigvalsh(cov)
        if (eigenvalues <= config.min_variance).any():
            cov = cov + (config.min_variance - eigenvalues.min() + 1e-6) * torch.eye(data.shape[1])

        return cov


def create_initializer(
    strategy: Union[str, InitializationStrategy],
    **kwargs
) -> BaseInitializer:
    """工厂函数：根据名称创建初始化器实例

    Args:
        strategy: 初始化策略名称 ('kmeans', 'random', 'heuristic')
                 或 InitializationStrategy枚举值
        **kwargs: 传递给对应初始化器的额外参数

    Returns:
        初始化器实例

    Raises:
        ValueError: 如果策略名称无效

    Example:
        >>> # 创建K-means初始化器
        >>> init = create_initializer('kmeans', max_iter=200)
        >>>
        >>> # 创建随机初始化器
        >>> init = create_initializer('random', seed=42)
        >>>
        >>> # 使用枚举
        >>> init = create_initializer(InitializationStrategy.KMEANS)
    """
    if isinstance(strategy, InitializationStrategy):
        strategy_name = strategy.value
    else:
        strategy_name = strategy.lower()

    if strategy_name == 'kmeans':
        return KMeansInitializer(**kwargs)
    elif strategy_name == 'random':
        return RandomInitializer(**kwargs)
    elif strategy_name == 'heuristic':
        return HeuristicInitializer(**kwargs)
    else:
        valid_strategies = ['kmeans', 'random', 'heuristic']
        raise ValueError(
            f"不支持的初始化策略: {strategy}。"
            f"支持的策略: {valid_strategies}"
        )


class MultiStartInitializer(BaseInitializer):
    """多起点初始化器

    结合多种初始化策略，运行多次后选择最佳结果。

    适用于需要高质量初始化的场景。

    Example:
        >>> multi_init = MultiStartInitializer(
        ...     strategies=['kmeans', 'random'],
        ...     n_trials_per_strategy=5
        ... )
        >>> params = multi_init.initialize(data)
    """

    def __init__(
        self,
        strategies: List[str] = ['kmeans', 'random'],
        n_trials_per_strategy: int = 3,
        selection_criteria: str = 'likelihood',
    ):
        """初始化多起点初始化器

        Args:
            strategies: 要使用的策略列表
            n_trials_per_strategy: 每种策略的试验次数
            selection_criteria: 选择准则 ('likelihood', 'bic', 'aic')
        """
        self.strategies = strategies
        self.n_trials_per_strategy = n_trials_per_strategy
        self.selection_criteria = selection_criteria

    def initialize(
        self,
        data: torch.Tensor,
        seed: Optional[int] = None,
    ) -> GMMParameters:
        """执行多起点初始化"""
        from .probability_density import GaussianMixtureModel

        best_params = None
        best_score = float('-inf')

        total_trials = 0

        for strategy_name in self.strategies:
            initializer = create_initializer(strategy_name)

            for trial in range(self.n_trials_per_strategy):
                trial_seed = None if seed is None else seed + total_trials * 7919

                try:
                    params = initializer.initialize(data, seed=trial_seed)

                    model = GaussianMixtureModel(params)
                    log_likelihood = model.log_pdf(data).sum().item()

                    score = self._compute_score(params, model, data, log_likelihood)

                    if score > best_score:
                        best_score = score
                        best_params = params

                    logger.debug(
                        f"{strategy_name} trial {trial+1}: "
                        f"score={score:.4f}"
                    )

                except Exception as e:
                    logger.warning(f"{strategy_name} trial {trial+1} 失败: {e}")

                total_trials += 1

        if best_params is None:
            logger.error("所有初始化尝试都失败")
            raise RuntimeError("无法完成多起点初始化")

        logger.info(
            f"多起点初始化完成: 最佳score={best_score:.4f}, "
            f"总试验次数={total_trials}"
        )

        return best_params

    def _compute_score(
        self,
        params: GMMParameters,
        model: 'GaussianMixtureModel',
        data: torch.Tensor,
        log_likelihood: float,
    ) -> float:
        """计算选择分数"""
        n_samples, n_features = data.shape
        n_params = self._count_parameters(params)

        if self.selection_criteria == 'likelihood':
            return log_likelihood
        elif self.selection_criteria == 'aic':
            return -2 * log_likelihood + 2 * n_params
        elif self.selection_criteria == 'bic':
            return -2 * log_likelihood + n_params * math.log(n_samples)
        else:
            return log_likelihood

    @staticmethod
    def _count_parameters(params: GMMParameters) -> int:
        """计算自由参数数量"""
        n_means = 2 * params.dimensionality

        if params.is_diagonal:
            n_covs = 2 * params.dimensionality
        else:
            n_covs = 2 * params.dimensionality * (params.dimensionality + 1) // 2

        n_weights = 1

        return n_means + n_covs + n_weights


import math
import torch.nn.functional as F
