"""扩散模型集成模块 - 将 GMM 求解器与扩散模型结合

提供 GMS (Gaussian Mixture Solver) 与标准扩散模型 (DDPM/DDIM) 之间的完整集成层。

核心组件:
    - Adapter: GMM 参数到噪声调度的适配器
    - ForwardProcess: GMS 增强的前向加噪过程
    - BackwardProcess: GMS 引导的反向去噪过程
    - ConditionInjection: 多种条件注入机制
    - Trainer: 端到端联合训练循环
    - CheckpointManager: 检查点管理和状态恢复
    - InferencePipeline: 快速推理和生成
    - DistributedTrainer: 分布式训练支持

快速开始:
    >>> from gms.diffusion_integration import (
    ...     GMSDiffusionAdapter, NoiseScheduler,
    ...     GMSForwardProcess, GMSBackwardProcess,
    ...     GMSTrainer, TrainingConfig,
    ...     GMSInferencePipeline, InferenceConfig,
    ... )
    >>>
    >>> # 1. 创建调度器和过程
    >>> scheduler = NoiseScheduler(num_steps=1000, schedule_type='cosine')
    >>> forward = GMSForwardProcess(scheduler)
    >>> backward = GMSBackwardProcess(scheduler)
    >>>
    >>> # 2. 配置并创建训练器
    >>> config = TrainingConfig(epochs=100, batch_size=32)
    >>> trainer = GMSTrainer(model, scheduler, forward, backward, config)
    >>>
    >>> # 3. 训练
    >>> history = trainer.train_full(epochs=100, dataloaders=train_loader)
    >>>
    >>> # 4. 推理生成
    >>> pipeline = GMSInferencePipeline(model, scheduler, backward)
    >>> result = pipeline.generate(n_samples=16)

与标准扩散模型的对比:
    标准 DDPM:
        - 固定的高斯白噪声 ε ~ N(0, I)
        - 线性或余弦 β 调度
        - 无外部条件信息

    GMS 增强:
        - 时间步依赖的非均匀噪声（由 GMM 控制）
        - 自适应噪声调度（基于混合分布统计量）
        - 可注入的 GMM 参数条件引导

Example:
    完整的端到端流程::

        adapter = GMSDiffusionAdapter(num_steps=1000)
        schedule = adapter.adapt_gmm_to_diffusion(gmm_params)

        x_noisy, noise = forward(x_clean, timesteps)
        model_output = denoiser(x_noisy, timesteps, gms_condition=cond)
        x_denoised = backward.sample_step(x_noisy, timesteps, model_output)
"""

from .adapter import (
    GMSDiffusionAdapter,
    NoiseSchedule,
    AdaptationStrategy,
)

from .forward_process import (
    GMSForwardProcess,
    NoiseScheduler,
    ScheduleType,
    SchedulerConfig,
)

from .backward_process import (
    GMSBackwardProcess,
    DenoisingNetworkWrapper,
    PredictionType,
    BackwardConfig,
    compute_gms_guidance_scale,
    apply_classifier_free_guidance,
)

from .condition_injection import (
    GMSConditionInjector,
    GMSEncoder,
    ConditionType,
    FiLMLayer,
    AdaptiveGroupNorm,
    CrossAttentionInjector,
    InjectorConfig,
    build_full_conditioning_pipeline,
)

from .trainer import (
    GMSTrainer,
    TrainingConfig,
    TrainingHistory,
    EpochMetrics,
    TrainingPhase,
    create_trainer_from_config,
)

from .checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointConfig,
    create_checkpoint_manager,
)

from .inference import (
    GMSInferencePipeline,
    InferenceConfig,
    GenerationResult,
    SamplingMethod,
    create_inference_pipeline_from_trainer,
)

from .distributed import (
    DistributedTrainer,
    DistributedConfig,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    is_distributed_available,
    get_world_size,
    get_rank,
    synchronize,
    launch_distributed_training,
)

__all__ = [
    # Adapter
    "GMSDiffusionAdapter",
    "NoiseSchedule",
    "AdaptationStrategy",
    # Forward Process
    "GMSForwardProcess",
    "NoiseScheduler",
    "ScheduleType",
    "SchedulerConfig",
    # Backward Process
    "GMSBackwardProcess",
    "DenoisingNetworkWrapper",
    "PredictionType",
    "BackwardConfig",
    "compute_gms_guidance_scale",
    "apply_classifier_free_guidance",
    # Condition Injection
    "GMSConditionInjector",
    "GMSEncoder",
    "ConditionType",
    "FiLMLayer",
    "AdaptiveGroupNorm",
    "CrossAttentionInjector",
    "InjectorConfig",
    "build_full_conditioning_pipeline",
    # Trainer
    "GMSTrainer",
    "TrainingConfig",
    "TrainingHistory",
    "EpochMetrics",
    "TrainingPhase",
    "create_trainer_from_config",
    # Checkpoint
    "CheckpointManager",
    "CheckpointMetadata",
    "CheckpointConfig",
    "create_checkpoint_manager",
    # Inference
    "GMSInferencePipeline",
    "InferenceConfig",
    "GenerationResult",
    "SamplingMethod",
    "create_inference_pipeline_from_trainer",
    # Distributed
    "DistributedTrainer",
    "DistributedConfig",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "is_distributed_available",
    "get_world_size",
    "get_rank",
    "synchronize",
    "launch_distributed_training",
]
