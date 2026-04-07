"""GMS 端到端联合训练循环

实现完整的 GMS 训练流程，协调矩估计、GMM优化和扩散模型训练的联合优化。
支持多阶段训练策略、灵活的损失计算和完善的日志记录。

核心功能:
    - 协调矩估计、GMM优化、扩散模型训练的联合流程
    - 支持多阶段训练（预训练GMM → 联合微调）
    - 完善的检查点管理和断点续训
    - 详细的训练指标记录和可视化支持

与标准扩散模型训练的对比:
    标准 DDPM 训练:
        1. 固定的噪声调度
        2. 单一的去噪网络
        3. 简单的 MSE 损失

    GMS 联合训练:
        1. 动态的 GMS 噪声调度（由 GMM 参数控制）
        2. 多任务学习（去噪 + GMM 正则化）
        3. 自适应的损失加权机制

Example:
    >>> from gms.diffusion_integration.trainer import (
    ...     GMSTrainer, TrainingConfig, TrainingHistory
    ... )
    >>>
    >>> config = TrainingConfig(
    ...     epochs=100,
    ...     batch_size=32,
    ...     learning_rate=1e-4,
    ...     device='cuda'
    ... )
    >>>
    >>> trainer = GMSTrainer(
    ...     model=denoising_model,
    ...     noise_scheduler=scheduler,
    ...     forward_process=forward_proc,
    ...     backward_process=backward_proc,
    ...     config=config
    ... )
    >>>
    >>> history = trainer.train_full(epochs=100, dataloaders=train_loader)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Callable,
    Iterator, TypeVar, Generic
)
import os
import time
import json
import copy
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None


T = TypeVar('T')


class TrainingPhase(Enum):
    """训练阶段枚举

    PRETRAIN_GMM: 预训练 GMM 参数（不更新扩散模型）
    JOINT_FINE TUNE: 联合微调（同时更新所有参数）
    DIFFUSION_ONLY: 仅训练扩散模型（冻结 GMM）
    """
    PRETRAIN_GMM = "pretrain_gmm"
    JOINT_FINE_TUNE = "joint_fine_tune"
    DIFFUSION_ONLY = "diffusion_only"


class SchedulerType(Enum):
    """学习率调度器类型"""
    CONSTANT = "constant"
    LINEAR_WARMUP = "linear_warmup"
    COSINE = "cosine"
    COSINE_WITH_WARMUP = "cosine_with_warmup"
    STEP = "step"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"


@dataclass
class TrainingConfig:
    """训练超参数配置数据类

    包含所有训练相关的超参数，支持从 YAML/JSON 文件加载。

    Attributes:
        batch_size: 批次大小
        learning_rate: 基础学习率
        epochs: 总训练轮数
        weight_decay: 权重衰减系数
        gradient_clip_norm: 梯度裁剪范数（0 表示不裁剪）
        device: 计算设备 ('cpu', 'cuda', 'cuda:0' 等）
        dtype: 数据类型 (torch.float32, torch.float16 等)
        mixed_precision: 是否使用混合精度训练
        num_workers: 数据加载工作线程数
        pin_memory: 是否固定内存（加速 CPU->GPU 传输）

        GMS 特定配置:
            gmm_regularization_weight: GMM 正则化损失的权重
            gmm_update_frequency: GMM 参数更新的频率（每 N 个 step）
            use_gmm_guidance: 是否在训练中使用 GMS 引导
            guidance_scale: GMS 引导强度

        调度器配置:
            scheduler_type: 学习率调度器类型
            warmup_epochs: warmup 轮数
            scheduler_kwargs: 调度器的额外参数

        检查点配置:
            checkpoint_dir: 检查点保存目录
            checkpoint_every: 每 N 个 epoch 保存一次检查点
            save_best_only: 是否只保存最佳模型
            keep_n_best: 保留的最佳检查点数量

        日志配置:
            log_every: 每 N 个 step 打印一次日志
            log_dir: 日志保存目录
            tensorboard_enabled: 是否启用 TensorBoard 日志

    Example:
        >>> config = TrainingConfig(batch_size=64, learning_rate=2e-4)
        >>> # 从字典加载
        >>> config = TrainingConfig.from_dict({'batch_size': 32, 'epochs': 50})
    """

    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    weight_decay: float = 0.0
    gradient_clip_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    mixed_precision: bool = False
    num_workers: int = 4
    pin_memory: bool = True

    gmm_regularization_weight: float = 0.01
    gmm_update_frequency: int = 10
    use_gmm_guidance: bool = True
    guidance_scale: float = 1.0

    scheduler_type: str = "cosine_with_warmup"
    warmup_epochs: int = 5
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)

    checkpoint_dir: str = "./checkpoints"
    checkpoint_every: int = 10
    save_best_only: bool = True
    keep_n_best: int = 3

    log_every: int = 10
    log_dir: str = "./logs"
    tensorboard_enabled: bool = False

    seed: int = 42
    deterministic: bool = False

    def __post_init__(self):
        """初始化后验证"""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正数，当前值: {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate 必须为正数，当前值: {self.learning_rate}")
        if self.epochs <= 0:
            raise ValueError(f"epochs 必须为正整数，当前值: {self.epochs}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置实例

        Args:
            data: 配置字典

        Returns:
            TrainingConfig 实例
        """
        filtered_data = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered_data)

    @classmethod
    def from_yaml(cls, filepath: str) -> "TrainingConfig":
        """从 YAML 文件加载配置

        Args:
            filepath: YAML 文件路径

        Returns:
            TrainingConfig 实例
        """
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get('training', data))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            配置字典
        """
        result = {}
        for key, value in self.__dataclass_fields__.items():
            val = getattr(self, key)
            if isinstance(val, torch.dtype):
                result[key] = str(val)
            else:
                result[key] = val
        return result


@dataclass
class EpochMetrics:
    """单个 epoch 的训练指标

    Attributes:
        epoch: epoch 编号
        phase: 训练阶段 ('train', 'val')
        total_loss: 总损失
        diffusion_loss: 扩散损失（MSE）
        gmm_loss: GMM 正则化损失（可选）
        loss_weights: 各项损失的权重
        learning_rate: 当前学习率
        grad_norm: 平均梯度范数
        time_elapsed: 该 epoch 耗时（秒）
        samples_processed: 处理的样本数
        gpu_memory: GPU 内存使用量（MB，可选）
        extra_metrics: 其他自定义指标
    """

    epoch: int
    phase: str = "train"
    total_loss: float = 0.0
    diffusion_loss: float = 0.0
    gmm_loss: Optional[float] = None
    loss_weights: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    time_elapsed: float = 0.0
    samples_processed: int = 0
    gpu_memory: Optional[float] = None
    extra_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'epoch': self.epoch,
            'phase': self.phase,
            'total_loss': self.total_loss,
            'diffusion_loss': self.diffusion_loss,
            'gmm_loss': self.gmm_loss,
            'loss_weights': self.loss_weights,
            'learning_rate': self.learning_rate,
            'grad_norm': self.grad_norm,
            'time_elapsed': self.time_elapsed,
            'samples_processed': self.samples_processed,
            'gpu_memory': self.gpu_memory,
            **self.extra_metrics,
        }


class TrainingHistory:
    """训练历史记录类

    训练和验证过程中的所有指标、参数变化和元信息。
    支持序列化/反序列化、可视化和最佳模型选择。

    Attributes:
        config: 训练配置
        train_metrics: 每个 epoch 的训练指标列表
        val_metrics: 每个 epoch 的验证指标列表
        best_epoch: 最佳 epoch 编号
        best_val_loss: 最佳验证损失
        start_time: 训练开始时间
        end_time: 训练结束时间
        total_time: 总训练时间

    Example:
        >>> history = TrainingHistory(config=training_config)
        >>> history.record_epoch(epoch_metrics)
        >>> best_epoch = history.best_epoch
        >>> history.save_to_file('training_history.json')
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        """初始化训练历史

        Args:
            config: 训练配置（可选）
        """
        self.config = config
        self.train_metrics: List[EpochMetrics] = []
        self.val_metrics: List[EpochMetrics] = []
        self.best_epoch: int = 0
        self.best_val_loss: float = float('inf')
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.total_time: float = 0.0
        self._global_step: int = 0

    @property
    def global_step(self) -> int:
        """获取全局步数"""
        return self._global_step

    def increment_step(self, n: int = 1) -> None:
        """增加全局步数

        Args:
            n: 增加的步数
        """
        self._global_step += n

    def record_epoch(
        self,
        metrics: EpochMetrics,
        is_validation: bool = False,
    ) -> None:
        """记录一个 epoch 的指标

        Args:
            metrics: EpochMetrics 实例
            is_validation: 是否为验证集指标
        """
        if is_validation:
            self.val_metrics.append(metrics)

            if metrics.total_loss < self.best_val_loss:
                self.best_val_loss = metrics.total_loss
                self.best_epoch = metrics.epoch

                if logger:
                    logger.info(
                        f"新的最佳验证损失: {self.best_val_loss:.6f} "
                        f"(epoch {metrics.epoch})"
                    )
        else:
            self.train_metrics.append(metrics)

    def get_train_losses(self) -> List[float]:
        """获取所有训练损失

        Returns:
            训练损失列表
        """
        return [m.total_loss for m in self.train_metrics]

    def get_val_losses(self) -> List[float]:
        """获取所有验证损失

        Returns:
            验证损失列表
        """
        return [m.total_loss for m in self.val_metrics]

    def get_latest_train_metrics(self) -> Optional[EpochMetrics]:
        """获取最新的训练指标

        Returns:
            最新的 EpochMetrics 或 None
        """
        return self.train_metrics[-1] if self.train_metrics else None

    def get_latest_val_metrics(self) -> Optional[EpochMetrics]:
        """获取最新的验证指标

        Returns:
            最新的 EpochMetrics 或 None
        """
        return self.val_metrics[-1] if self.val_metrics else None

    def get_best_checkpoint_path(self, checkpoint_dir: str) -> Optional[str]:
        """获取最佳模型的检查点路径

        Args:
            checkpoint_dir: 检查点目录

        Returns:
            最佳检查点的完整路径，如果不存在则返回 None
        """
        best_ckpt = os.path.join(
            checkpoint_dir,
            f"best_model_epoch_{self.best_epoch}.pt"
        )

        if os.path.exists(best_ckpt):
            return best_ckpt

        return None

    def start_timing(self) -> None:
        """开始计时"""
        self.start_time = datetime.now()
        if logger:
            logger.info(f"训练开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def stop_timing(self) -> None:
        """停止计时"""
        self.end_time = datetime.now()
        if self.start_time:
            self.total_time = (self.end_time - self.start_time).total_seconds()
        if logger:
            logger.info(
                f"训练结束时间: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"总耗时: {self.total_time:.2f} 秒"
            )

    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要

        Returns:
            包含关键统计信息的字典
        """
        train_losses = self.get_train_losses()
        val_losses = self.get_val_losses()

        summary = {
            'total_epochs': len(self.train_metrics),
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'total_training_time': self.total_time,
            'global_steps': self._global_step,
        }

        if train_losses:
            summary['avg_train_loss'] = sum(train_losses) / len(train_losses)
            summary['min_train_loss'] = min(train_losses)
            summary['max_train_loss'] = max(train_losses)

        if val_losses:
            summary['avg_val_loss'] = sum(val_losses) / len(val_losses)
            summary['min_val_loss'] = min(val_losses)
            summary['max_val_loss'] = max(val_losses)

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典

        Returns:
            字典表示
        """
        return {
            'config': self.config.to_dict() if self.config else None,
            'train_metrics': [m.to_dict() for m in self.train_metrics],
            'val_metrics': [m.to_dict() for m in self.val_metrics],
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_time': self.total_time,
            'global_step': self._global_step,
            'summary': self.get_summary(),
        }

    def save_to_file(self, filepath: str) -> None:
        """保存到 JSON 文件

        Args:
            filepath: 输出文件路径
        """
        data = self.to_dict()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        if logger:
            logger.info(f"训练历史已保存到 {filepath}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingHistory":
        """从字典恢复训练历史

        Args:
            data: 序列化的字典

        Returns:
            TrainingHistory 实例
        """
        config = None
        if data.get('config'):
            config = TrainingConfig.from_dict(data['config'])

        history = cls(config=config)

        for m in data.get('train_metrics', []):
            metrics = EpochMetrics(**{k: v for k, v in m.items()
                                     if k in EpochMetrics.__dataclass_fields__})
            history.train_metrics.append(metrics)

        for m in data.get('val_metrics', []):
            metrics = EpochMetrics(**{k: v for k, v in m.items()
                                     if k in EpochMetrics.__dataclass_fields__})
            history.val_metrics.append(metrics)

        history.best_epoch = data.get('best_epoch', 0)
        history.best_val_loss = data.get('best_val_loss', float('inf'))
        history._global_step = data.get('global_step', 0)
        history.total_time = data.get('total_time', 0.0)

        if data.get('start_time'):
            history.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            history.end_time = datetime.fromisoformat(data['end_time'])

        return history

    @classmethod
    def from_file(cls, filepath: str) -> "TrainingHistory":
        """从 JSON 文件加载

        Args:
            filepath: JSON 文件路径

        Returns:
            TrainingHistory 实例
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class GMSTrainer:
    """GMS 端到端联合训练器

    协调矩估计、GMM优化和扩散模型训练的完整训练循环。
    支持多阶段训练策略、灵活的损失组合和完善的状态管理。

    核心职责:
        1. 管理前向/反向传播过程
        2. 协调多个优化器和调度器
        3. 计算 and 加权多任务损失
        4. 记录训练指标和历史
        5. 处理检查点保存和加载

    训练流程:
        对于每个 batch:
        1. 采样时间步 t ~ Uniform(1, T)
        2. 采样噪声 ε ~ N(0, I) 或 GMM 控制的噪声
        3. 计算加噪数据 x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        4. 通过去噪网络预测噪声 ε_θ(x_t, t)
        5. 计算扩散损失 L_diff = MSE(ε_θ, ε)
        6. （可选）计算 GMM 正则化损失 L_gmm
        7. 总损失 L = L_diff + λ · L_gmm
        8. 反向传播并更新参数

    Attributes:
        model: 去噪神经网络
        noise_scheduler: 噪声调度器
        forward_process: GMS 前向过程
        backward_process: GMS 反向过程
        config: 训练配置
        optimizer: 优化器
        scheduler: 学习率调度器
        history: 训练历史记录
        current_epoch: 当前 epoch 编号
        device: 计算设备

    Example:
        >>> trainer = GMSTrainer(
        ...     model=my_unet,
        ...     noise_scheduler=NoiseScheduler(num_steps=1000),
        ...     forward_process=GMSForwardProcess(scheduler),
        ...     backward_process=GMSBackwardProcess(scheduler),
        ...     config=TrainingConfig(epochs=50)
        ... )
        >>>
        >>> history = trainer.train_full(
        ...     epochs=50,
        ...     dataloaders={'train': train_loader, 'val': val_loader}
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: "NoiseScheduler",
        forward_process: "GMSForwardProcess",
        backward_process: "GMSBackwardProcess",
        config: TrainingConfig,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        gmm_parameters: Optional["GMMParameters"] = None,
        condition_encoder: Optional[nn.Module] = None,
        condition_injector: Optional[nn.Module] = None,
    ):
        """初始化 GMS 训练器

        Args:
            model: 去噪神经网络（如 UNet）
            noise_scheduler: NoiseScheduler 实例
            forward_process: GMSForwardProcess 实例
            backward_process: GMSBackwardProcess 实例
            config: TrainingConfig 训练配置
            optimizer: 优化器（如果不提供则自动创建）
            scheduler: 学习率调度器（可选）
            gmm_parameters: GMMParameters 实例（用于 GMS 增强）
            condition_encoder: 条件编码器（GMSEncoder）
            condition_injector: 条件注入器（GMSConditionInjector）

        Raises:
            ValueError: 如果必要组件缺失或配置非法
        """
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.forward_process = forward_process
        self.backward_process = backward_process
        self.config = config
        self.gmm_parameters = gmm_parameters
        self.condition_encoder = condition_encoder
        self.condition_injector = condition_injector

        self.device = torch.device(config.device)
        self.dtype = config.dtype

        self.model.to(self.device).to(self.dtype)

        if self.condition_encoder is not None:
            self.condition_encoder.to(self.device).to(self.dtype)
        if self.condition_injector is not None:
            self.condition_injector.to(self.device).to(self.dtype)

        self.forward_process.to(self.device)
        self.backward_process.to(self.device)
        self.noise_scheduler.to(self.device)

        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = scheduler

        self.history = TrainingHistory(config=config)
        self.current_epoch: int = 0
        self.global_step: int = 0

        self.scaler = GradScaler() if config.mixed_precision else None

        if config.seed is not None:
            self._set_seed(config.seed)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        if logger:
            logger.info(
                f"GMSTrainer 初始化完成: "
                f"device={self.device}, "
                f"mixed_precision={config.mixed_precision}, "
                f"params={sum(p.numel() for p in model.parameters()):,}"
            )

    def _set_seed(self, seed: int) -> None:
        """设置随机种子

        Args:
            seed: 随机种子值
        """
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if logger:
            logger.debug(f"随机种子设置为: {seed}")

    def _create_optimizer(self) -> optim.Optimizer:
        """创建默认优化器

        Returns:
            AdamW 优化器实例
        """
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """创建学习率调度器

        Returns:
            调度器实例
        """
        scheduler_type = SchedulerType(self.config.scheduler_type)

        if scheduler_type == SchedulerType.CONSTANT:
            return optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=1
            )

        elif scheduler_type == SchedulerType.LINEAR_WARMUP:
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_epochs,
            )

        elif scheduler_type == SchedulerType.COSINE:
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01,
            )

        elif scheduler_type == SchedulerType.COSINE_WITH_WARMUP:
            return optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[
                    optim.lr_scheduler.LinearLR(
                        self.optimizer,
                        start_factor=0.1,
                        total_iters=self.config.warmup_epochs,
                    ),
                    optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        T_max=self.config.epochs - self.config.warmup_epochs,
                        eta_min=self.config.learning_rate * 0.01,
                    ),
                ],
                milestones=[self.config.warmup_epochs],
            )

        elif scheduler_type == SchedulerType.STEP:
            step_size = self.config.scheduler_kwargs.get('step_size', 30)
            gamma = self.config.scheduler_kwargs.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )

        elif scheduler_type == SchedulerType.EXPONENTIAL:
            gamma = self.config.scheduler_kwargs.get('gamma', 0.95)
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=gamma
            )

        elif scheduler_type == SchedulerType.PLATEAU:
            mode = self.config.scheduler_kwargs.get('mode', 'min')
            factor = self.config.scheduler_kwargs.get('factor', 0.1)
            patience = self.config.scheduler_kwargs.get('patience', 10)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=mode, factor=factor, patience=patience
            )

        else:
            return optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=1
            )

    def compute_loss(
        self,
        x_0: torch.Tensor,
        model_output: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算训练损失

        计算扩散损失和可选的 GMM 正则化损失。

        损失组成:
            L_total = L_diffusion + λ · L_gmm

        其中:
            L_diffusion = MSE(ε_pred, ε_true)  （标准去噪损失）
            L_gmm = GMM 参数的正则化项（可选）

        Args:
            x_0: 干净数据，形状 (B, C, H, W)
            model_output: 模型预测输出，形状与 x_0 相同
            noise: 真实噪声，形状与 x_0 相同
            timestep: 时间步张量，形状 (B,)

        Returns:
            包含各项损失的字典:
                - 'total': 总损失
                - 'diffusion': 扩散损失
                - 'gmm': GMM 正则化损失（可能为 None）
        """
        loss_weights = {'diffusion': 1.0}

        diffusion_loss = nn.functional.mse_loss(model_output, noise)

        losses = {
            'diffusion': diffusion_loss,
            'total': diffusion_loss,
        }

        if (self.gmm_parameters is not None and
            self.config.gmm_regularization_weight > 0):

            gmm_loss = self._compute_gmm_regularization(x_0, timestep)
            losses['gmm'] = gmm_loss
            losses['total'] = (
                diffusion_loss +
                self.config.gmm_regularization_weight * gmm_loss
            )
            loss_weights['gmm'] = self.config.gmm_regularization_weight

        losses['_weights'] = loss_weights

        return losses

    def _compute_gmm_regularization(
        self,
        x_0: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """计算 GMM 正则化损失

        使用 GMM 参数对生成分布施加约束。
        鼓励生成的样本接近 GMM 定义的分布。

        正则化形式:
            L_gmm = E[-log p_gmm(x_0)] + ||params||²

        Args:
            x_0: 输入数据
            timestep: 时间步

        Returns:
            GMM 正则化损失标量
        """
        try:
            from gms.gmm_optimization.probability_density import compute_log_probability

            params = self.gmm_parameters.to_device(self.device)
            log_prob = compute_log_probability(x_0, params)

            reg_loss = -log_prob.mean()

            param_reg = (
                torch.norm(params.mean1) ** 2 +
                torch.norm(params.mean2) ** 2
            ) * 1e-4

            reg_loss = reg_loss + param_reg

            return reg_loss

        except Exception as e:
            if logger:
                logger.warning(f"GMM 正则化计算失败，使用零损失: {e}")
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> EpochMetrics:
        """训练一个 epoch

        完整的前向-反向-优化循环。

        Args:
            dataloader: 训练数据加载器
            epoch: 当前 epoch 编号

        Returns:
            EpochMetrics 包含该 epoch 的所有指标

        Example:
            >>> metrics = trainer.train_epoch(train_loader, epoch=1)
            >>> print(f"训练损失: {metrics.total_loss:.6f}")
        """
        self.model.train()

        if self.condition_encoder is not None:
            self.condition_encoder.train()
        if self.condition_injector is not None:
            self.condition_injector.train()

        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_gmm_loss = 0.0
        num_batches = 0
        total_samples = 0
        grad_norm_sum = 0.0

        epoch_start_time = time.time()

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch}/{self.config.epochs}",
            disable=not HAS_TQDM,
        ) if HAS_TQDM else dataloader

        for batch_idx, batch in enumerate(progress_bar):
            if isinstance(batch, (tuple, list)):
                x_0 = batch[0].to(self.device, dtype=self.dtype)
            else:
                x_0 = batch.to(self.device, dtype=self.dtype)

            batch_size = x_0.shape[0]
            total_samples += batch_size

            self.optimizer.zero_grad(set_to_none=True)

            if self.config.mixed_precision and self.scaler is not None:
                with autocast(enabled=True):
                    losses = self._train_step(x_0)

                self.scaler.scale(losses['total']).backward()

                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )
                else:
                    grad_norm = 0.0

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses = self._train_step(x_0)
                losses['total'].backward()

                if self.config.gradient_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm,
                    )
                else:
                    grad_norm = 0.0

                self.optimizer.step()

            batch_loss = losses['total'].item()
            diff_loss = losses['diffusion'].item()
            gmm_loss_val = losses.get('gmm')

            total_loss += batch_loss
            total_diffusion_loss += diff_loss
            if gmm_loss_val is not None:
                total_gmm_loss += gmm_loss_val.item()

            grad_norm_sum += grad_norm if isinstance(grad_norm, float) else 0.0
            num_batches += 1
            self.global_step += 1
            self.history.increment_step()

            if HAS_TQDM and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'lr': f'{self._get_current_lr():.6f}'
                })

            elif (batch_idx + 1) % self.config.log_every == 0:
                if logger:
                    logger.debug(
                        f"Epoch [{epoch}/{self.config.epochs}] "
                        f"Batch [{batch_idx+1}/{len(dataloader)}] "
                        f"Loss: {batch_loss:.6f}"
                    )

        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                pass
            else:
                self.scheduler.step()

        epoch_time = time.time() - epoch_start_time

        avg_loss = total_loss / max(num_batches, 1)
        avg_diff_loss = total_diffusion_loss / max(num_batches, 1)
        avg_gmm_loss = total_gmm_loss / max(num_batches, 1) if total_gmm_loss > 0 else None
        avg_grad_norm = grad_norm_sum / max(num_batches, 1)

        gpu_memory = self._get_gpu_memory()

        metrics = EpochMetrics(
            epoch=epoch,
            phase="train",
            total_loss=avg_loss,
            diffusion_loss=avg_diff_loss,
            gmm_loss=avg_gmm_loss,
            learning_rate=self._get_current_lr(),
            grad_norm=avg_grad_norm,
            time_elapsed=epoch_time,
            samples_processed=total_samples,
            gpu_memory=gpu_memory,
        )

        self.history.record_epoch(metrics, is_validation=False)

        if logger:
            logger.info(
                f"Epoch [{epoch}/{self.config.epochs}] 完成 - "
                f"Loss: {avg_loss:.6f}, "
                f"Diff Loss: {avg_diff_loss:.6f}, "
                f"Time: {epoch_time:.2f}s, "
                f"Samples: {total_samples}"
            )

        return metrics

    def _train_step(
        self,
        x_0: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """单步训练逻辑

        执行前向传播并返回损失字典。

        Args:
            x_0: 输入批次，形状 (B, C, H, W)

        Returns:
            损失字典
        """
        batch_size = x_0.shape[0]
        num_steps = self.noise_scheduler.num_steps

        t = torch.randint(
            0, num_steps, (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

        gmm_noise_params = None
        if (self.gmm_parameters is not None and
            self.forward_process.gmm_noise_enabled):

            if self.global_step % self.config.gmm_update_frequency == 0:
                gmm_noise_params = self._prepare_gmm_noise_params(batch_size)

        x_t, noise = self.forward_process(x_0, t, gmm_noise_params)

        gms_condition = None
        if self.condition_encoder is not None and self.gmm_parameters is not None:
            gmm_params_dict = self.gmm_parameters.to_dict()
            gms_condition = self.condition_encoder(gmm_params_dict)

            if gms_condition.dim() == 1:
                gms_condition = gms_condition.unsqueeze(0).expand(batch_size, -1)

        model_output = self.model(x_t, t, gms_condition=gms_condition)

        losses = self.compute_loss(x_0, model_output, noise, t)

        return losses

    def _prepare_gmm_noise_params(
        self,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """准备 GMM 噪声参数

        从当前的 GMM 参数中提取噪声均值和方差。

        Args:
            batch_size: 批次大小

        Returns:
            GMM 噪声参数字典
        """
        params = self.gmm_parameters.to_device(self.device)

        mean_mixture = params.weight * params.mean1 + params.weight2 * params.mean2
        var_mixture = (
            params.weight * params.variance1 +
            params.weight2 * params.variance2
        )

        mean_expanded = mean_mixture.unsqueeze(0).expand(batch_size, -1)
        var_expanded = var_mixture.unsqueeze(0).expand(batch_size, -1)

        return {
            'mean': mean_expanded,
            'variance': var_expanded,
        }

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> EpochMetrics:
        """验证循环

        在验证集上评估模型性能。

        Args:
            dataloader: 验证数据加载器
            epoch: 当前 epoch 编号

        Returns:
            EpochMetrics 验证指标
        """
        self.model.eval()

        if self.condition_encoder is not None:
            self.condition_encoder.eval()
        if self.condition_injector is not None:
            self.condition_injector.eval()

        total_loss = 0.0
        total_diffusion_loss = 0.0
        total_gmm_loss = 0.0
        num_batches = 0
        total_samples = 0

        val_start_time = time.time()

        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x_0 = batch[0].to(self.device, dtype=self.dtype)
            else:
                x_0 = batch.to(self.device, dtype=self.dtype)

            batch_size = x_0.shape[0]
            total_samples += batch_size

            num_steps = self.noise_scheduler.num_steps
            t = torch.randint(
                0, num_steps, (batch_size,),
                device=self.device,
                dtype=torch.long,
            )

            x_t, noise = self.forward_process(x_0, t)

            gms_condition = None
            if self.condition_encoder is not None and self.gmm_parameters is not None:
                gmm_params_dict = self.gmm_parameters.to_dict()
                gms_condition = self.condition_encoder(gmm_params_dict)
                if gms_condition.dim() == 1:
                    gms_condition = gms_condition.unsqueeze(0).expand(batch_size, -1)

            with torch.set_grad_enabled(False):
                model_output = self.model(x_t, t, gms_condition=gms_condition)
                losses = self.compute_loss(x_0, model_output, noise, t)

            total_loss += losses['total'].item()
            total_diffusion_loss += losses['diffusion'].item()

            if losses.get('gmm') is not None:
                total_gmm_loss += losses['gmm'].item()

            num_batches += 1

        val_time = time.time() - val_start_time

        avg_loss = total_loss / max(num_batches, 1)
        avg_diff_loss = total_diffusion_loss / max(num_batches, 1)
        avg_gmm_loss = total_gmm_loss / max(num_batches, 1) if total_gmm_loss > 0 else None

        metrics = EpochMetrics(
            epoch=epoch,
            phase="val",
            total_loss=avg_loss,
            diffusion_loss=avg_diff_loss,
            gmm_loss=avg_gmm_loss,
            learning_rate=self._get_current_lr(),
            time_elapsed=val_time,
            samples_processed=total_samples,
        )

        self.history.record_epoch(metrics, is_validation=True)

        if logger:
            logger.info(
                f"验证完成 - Epoch [{epoch}] - "
                f"Val Loss: {avg_loss:.6f}, "
                f"Val Diff Loss: {avg_diff_loss:.6f}"
            )

        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(avg_loss)

        return metrics

    def train_full(
        self,
        epochs: Optional[int] = None,
        dataloaders: Union[DataLoader, Dict[str, DataLoader]] = None,
        validation_freq: int = 1,
        early_stopping_patience: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> TrainingHistory:
        """完整训练流程

        执行完整的训练和验证循环。

        Args:
            epochs: 训练轮数（覆盖配置中的值）
            dataloaders: 数据加载器或 {'train': loader, 'val': loader}
            validation_freq: 验证频率（每 N 个 epoch）
            early_stopping_patience: 早停耐心值（None 表示不使用）
            callbacks: 回调函数列表

        Returns:
            TrainingHistory 训练历史

        Example:
            >>> history = trainer.train_full(
            ...     epochs=100,
            ...     dataloaders={'train': train_loader, 'val': val_loader},
            ...     validation_freq=5
            ... )
        """
        if epochs is None:
            epochs = self.config.epochs

        self.history.start_timing()

        if dataloaders is None:
            raise ValueError("必须提供 dataloaders")

        if isinstance(dataloaders, DataLoader):
            train_loader = dataloaders
            val_loader = None
        else:
            train_loader = dataloaders.get('train')
            val_loader = dataloaders.get('val')

        if train_loader is None:
            raise ValueError("dataloaders 必须包含 'train' 键")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.current_epoch + 1, self.current_epoch + epochs + 1):
            self.current_epoch = epoch

            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_begin(epoch, self)

            train_metrics = self.train_epoch(train_loader, epoch)

            should_validate = (
                val_loader is not None and
                epoch % validation_freq == 0
            )

            if should_validate:
                val_metrics = self.validate(val_loader, epoch)

                if early_stopping_patience is not None:
                    if val_metrics.total_loss < best_val_loss - 1e-6:
                        best_val_loss = val_metrics.total_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= early_stopping_patience:
                        if logger:
                            logger.info(
                                f"早停触发! 连续 {patience_counter} 个 epoch "
                                f"没有改善 (最佳: {best_val_loss:.6f})"
                            )
                        break

            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, self, train_metrics)

            if epoch % self.config.checkpoint_every == 0:
                self._save_checkpoint(epoch, train_metrics)

        self.history.stop_timing()

        if logger:
            summary = self.history.get_summary()
            logger.info(f"训练完成! 摘要: {summary}")

        return self.history

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: EpochMetrics,
    ) -> str:
        """保存检查点

        Args:
            epoch: 当前 epoch
            metrics: 当前指标

        Returns:
            保存的文件路径
        """
        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            'training_history': self.history.to_dict(),
            'config': self.config.to_dict(),
            'metrics': metrics.to_dict(),
            'rng_state': torch.get_rng_state(),
            'timestamp': datetime.now().isoformat(),
        }

        if self.scaler is not None:
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

        if self.gmm_parameters is not None:
            checkpoint_data['gmm_params'] = self.gmm_parameters.to_dict()

        if self.condition_encoder is not None:
            checkpoint_data['condition_encoder_state_dict'] = \
                self.condition_encoder.state_dict()

        if self.condition_injector is not None:
            checkpoint_data['condition_injector_state_dict'] = \
                self.condition_injector.state_dict()

        filename = f"checkpoint_epoch_{epoch}.pt"
        filepath = os.path.join(self.config.checkpoint_dir, filename)

        torch.save(checkpoint_data, filepath)

        if self.config.save_best_only and metrics.phase == "val":
            best_filepath = os.path.join(
                self.config.checkpoint_dir,
                "best_model.pt"
            )
            torch.save(checkpoint_data, best_filepath)

        if logger:
            logger.debug(f"检查点已保存: {filepath}")

        return filepath

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """加载检查点

        Args:
            checkpoint_path: 检查点文件路径
            load_optimizer: 是否加载优化器状态
            strict: 是否严格匹配模型参数

        Returns:
            检查点数据字典
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=strict,
        )

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if ('scheduler_state_dict' in checkpoint and
            self.scheduler is not None and
            checkpoint['scheduler_state_dict'] is not None):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)

        if 'gmm_params' in checkpoint and checkpoint['gmm_params'] is not None:
            from gms.gmm_optimization.gmm_parameters import GMMParameters
            self.gmm_parameters = GMMParameters.from_dict(checkpoint['gmm_params'])

        if ('condition_encoder_state_dict' in checkpoint and
            self.condition_encoder is not None):
            self.condition_encoder.load_state_dict(
                checkpoint['condition_encoder_state_dict']
            )

        if ('condition_injector_state_dict' in checkpoint and
            self.condition_injector is not None):
            self.condition_injector.load_state_dict(
                checkpoint['condition_injector_state_dict']
            )

        if 'training_history' in checkpoint:
            self.history = TrainingHistory.from_dict(checkpoint['training_history'])

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].to(self.device))

        if logger:
            logger.info(
                f"检查点已加载: {checkpoint_path} "
                f"(epoch={self.current_epoch}, step={self.global_step})"
            )

        return checkpoint

    def _get_current_lr(self) -> float:
        """获取当前学习率

        Returns:
            当前学习率
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def _get_gpu_memory(self) -> Optional[float]:
        """获取当前 GPU 内存使用量

        Returns:
            GPU 内存使用量（MB），如果不可用则返回 None
        """
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
        return None

    def set_learning_rate(
        self,
        new_lr: float,
        param_group_index: Optional[int] = None,
    ) -> None:
        """动态调整学习率

        Args:
            new_lr: 新的学习率
            param_group_index: 参数组索引（None 表示所有组）
        """
        if param_group_index is not None:
            self.optimizer.param_groups[param_group_index]['lr'] = new_lr
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

        if logger:
            logger.info(f"学习率调整为: {new_lr}")

    def get_model_state(self) -> Dict[str, Any]:
        """获取完整模型状态

        Returns:
            包含所有状态的字典
        """
        return {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
            'history_summary': self.history.get_summary(),
        }

    def export_for_inference(self) -> Dict[str, Any]:
        """导出推理所需的最小状态

        Returns:
            推理所需的精简状态字典
        """
        state = {
            'model_state_dict': self.model.state_dict(),
            'noise_scheduler_config': self.noise_scheduler.config,
            'backward_config': self.backward_process.export_state(),
            'config': {
                'device': str(self.device),
                'dtype': str(self.dtype),
            },
        }

        if self.gmm_parameters is not None:
            state['gmm_params'] = self.gmm_parameters.to_dict()

        if self.condition_encoder is not None:
            state['condition_encoder_state_dict'] = \
                self.condition_encoder.state_dict()

        return state


def create_trainer_from_config(
    model: nn.Module,
    config_path: str,
    noise_scheduler: Optional["NoiseScheduler"] = None,
    forward_process: Optional["GMSForwardProcess"] = None,
    backward_process: Optional["GMSBackwardProcess"] = None,
) -> GMSTrainer:
    """从配置文件快速创建训练器

    便捷函数，从 YAML 配置文件创建完整的训练器实例。

    Args:
        model: 去噪网络模型
        config_path: YAML 配置文件路径
        noise_scheduler: 噪声调度器（可选，将自动创建）
        forward_process: 前向过程（可选，将自动创建）
        backward_process: 反向过程（可选，将自动创建）

    Returns:
        配置完成的 GMSTrainer 实例

    Example:
        >>> trainer = create_trainer_from_config(
        ...     model=my_unet,
        ...     config_path='configs/training_config.yaml'
        ... )
    """
    training_config = TrainingConfig.from_yaml(config_path)

    if noise_scheduler is None:
        from .forward_process import NoiseScheduler
        noise_scheduler = NoiseScheduler(
            num_steps=1000,
            schedule_type='cosine',
            device=training_config.device,
        )

    if forward_process is None:
        from .forward_process import GMSForwardProcess
        forward_process = GMSForwardProcess(noise_scheduler)

    if backward_process is None:
        from .backward_process import GMSBackwardProcess
        backward_process = GMSBackwardProcess(noise_scheduler)

    return GMSTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
        forward_process=forward_process,
        backward_process=backward_process,
        config=training_config,
    )


if __name__ == "__main__":
    print("GMS Trainer 模块 - 用于 GMS-Diffusion 模型的端到端训练")
    print("\n主要组件:")
    print("  - TrainingConfig: 训练配置数据类")
    print("  - TrainingHistory: 训练历史记录")
    print("  - GMSTrainer: 主训练器类")
    print("\n快速开始:")
    print("  from gms.diffusion_integration.trainer import GMSTrainer, TrainingConfig")
    print("  config = TrainingConfig(epochs=100, batch_size=32)")
    print("  trainer = GMSTrainer(model, scheduler, forward, backward, config)")
    print("  history = trainer.train_full(epochs=100, dataloaders=loaders)")
