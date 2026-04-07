"""采样模块 - 从 GMM 和扩散模型中生成样本

提供采样调度、时间步控制、检查点管理、进度监控和双分量高斯混合采样功能。
"""

from .sampling_scheduler import (
    BaseScheduler,
    LinearScheduler,
    CosineScheduler,
    ConstantScheduler,
    SqrtScheduler,
)

from .time_step_controller import (
    TimeStepController,
    AdaptationMode,
    StepHistory,
    TimeStepStats,
)

from .checkpoint_manager import (
    SamplingCheckpointManager,
    SamplingCheckpoint,
    CheckpointCleanupPolicy,
)

from .progress_monitor import (
    ProgressMonitor,
    SamplingProgress,
    SamplingEventType,
    TqdmProgressMonitor,
)

from .component_selector import ComponentSelector
from .gaussian_sampler import GaussianSampler
from .batch_sampler import BatchGaussianMixtureSampler
from .sampling_validator import SamplingValidator, ValidationReport
from .reproducibility import ReproducibleSampler

__all__ = [
    # 调度器
    "BaseScheduler",
    "LinearScheduler",
    "CosineScheduler",
    "ConstantScheduler",
    "SqrtScheduler",
    # 时间步控制器
    "TimeStepController",
    "AdaptationMode",
    "StepHistory",
    "TimeStepStats",
    # 检查点管理
    "SamplingCheckpointManager",
    "SamplingCheckpoint",
    "CheckpointCleanupPolicy",
    # 进度监控
    "ProgressMonitor",
    "SamplingProgress",
    "SamplingEventType",
    "TqdmProgressMonitor",
    # 双分量采样器
    "ComponentSelector",
    "GaussianSampler",
    "BatchGaussianMixtureSampler",
    "SamplingValidator",
    "ValidationReport",
    "ReproducibleSampler",
]
