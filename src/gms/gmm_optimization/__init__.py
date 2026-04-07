"""高斯混合模型优化模块 - 用于 GMM 参数的优化和拟合

提供完整的GMM参数优化框架，包括:
- 优化器基础框架 (BaseGMMOptimizer, GradientDescentOptimizer, AdamOptimizer)
- 矩匹配损失函数 (MomentMatchingLoss, WeightedMSELoss, HuberMomentLoss)
- 正则化和稳定性约束 (RegularizationTerm, StabilityConstraints)
- 训练监控接口 (TrainingMonitor, MonitoringData)
- 早停和学习率调度机制 (EarlyStopping, StepLR, ExponentialLR等)
- 双分量高斯混合模型 (GMMParameters, GaussianMixtureModel)
- 概率密度函数计算 (PDF, log-PDF, 后验概率, 采样)
- 参数序列化 (JSON, Pickle, State Dict格式)
- 参数初始化策略 (K-means++, 随机, 启发式)

Example:
    >>> from src.gms.gmm_optimization import (
    ...     AdamOptimizer,
    ...     OptimizationConfig,
    ...     GMMParameters,
    ...     GaussianMixtureModel,
    ...     KMeansInitializer
    ... )
    >>>
    >>> # 创建GMM参数和模型
    >>> params = GMMParameters(
    ...     weight=0.6,
    ...     mean1=torch.tensor([1.0, 2.0]),
    ...     mean2=torch.tensor([-1.0, -2.0]),
    ...     variance1=torch.tensor([0.5, 0.5]),
    ...     variance2=torch.tensor([1.0, 1.0])
    ... )
    >>> model = GaussianMixtureModel(params)
    >>>
    >>> # 创建优化配置和优化器
    >>> config = OptimizationConfig(learning_rate=0.001, max_iterations=1000)
    >>> optimizer = AdamOptimizer(config)
    >>>
    >>> # 使用K-means++初始化
    >>> initializer = KMeansInitializer()
    >>> init_params = initializer.initialize(data)
    >>>
    >>> # 执行优化
    >>> result = optimizer.optimize(target_moments, init_params.to_optimizer_params())
"""

from .optimizer_base import (
    BaseGMMOptimizer,
    OptimizationConfig,
    OptimizedParams,
    TargetMoments,
    EpochCallbackData,
    GradientDescentOptimizer,
    AdamOptimizer,
)

from .loss_functions import (
    LossConfig,
    MomentMatchingLoss,
    WeightedMSELoss,
    HuberMomentLoss,
    create_loss_function,
)

from .regularization import (
    RegularizationConfig,
    RegularizationTerm,
    StabilityConstraints,
)

from .monitoring import (
    MonitoringData,
    TrainingMonitor,
    create_monitor,
)

from .schedulers import (
    EarlyStoppingConfig,
    EarlyStopping,
    LearningRateScheduler,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LambdaLR,
    create_scheduler,
)

from .gmm_parameters import (
    GMMParameters,
    GMMParametersConfig,
    CovarianceType,
    InitializationStrategy,
)

from .probability_density import (
    GaussianMixtureModel,
    compute_kl_divergence,
    compute_js_divergence,
    compute_wasserstein_distance,
)

from .serialization import (
    GMMSerializer,
    SerializationConfig,
    create_serializer,
)

from .initialization import (
    BaseInitializer,
    KMeansInitializer,
    RandomInitializer,
    HeuristicInitializer,
    MultiStartInitializer,
    create_initializer,
)

__all__ = [
    # Optimizer base
    "BaseGMMOptimizer",
    "OptimizationConfig",
    "OptimizedParams",
    "TargetMoments",
    "EpochCallbackData",
    "GradientDescentOptimizer",
    "AdamOptimizer",

    # Loss functions
    "LossConfig",
    "MomentMatchingLoss",
    "WeightedMSELoss",
    "HuberMomentLoss",
    "create_loss_function",

    # Regularization
    "RegularizationConfig",
    "RegularizationTerm",
    "StabilityConstraints",

    # Monitoring
    "MonitoringData",
    "TrainingMonitor",
    "create_monitor",

    # Schedulers
    "EarlyStoppingConfig",
    "EarlyStopping",
    "LearningRateScheduler",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "ReduceLROnPlateau",
    "LambdaLR",
    "create_scheduler",

    # GMM Model
    "GMMParameters",
    "GMMParametersConfig",
    "CovarianceType",
    "InitializationStrategy",
    "GaussianMixtureModel",
    "compute_kl_divergence",
    "compute_js_divergence",
    "compute_wasserstein_distance",

    # Serialization
    "GMMSerializer",
    "SerializationConfig",
    "create_serializer",

    # Initialization
    "BaseInitializer",
    "KMeansInitializer",
    "RandomInitializer",
    "HeuristicInitializer",
    "MultiStartInitializer",
    "create_initializer",
]
