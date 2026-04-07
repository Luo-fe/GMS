"""GMS 分布式训练支持

提供 DataParallel 和 DistributedDataParallel (DDP) 支持，
以及混合精度训练（AMP）功能，用于加速大规模模型训练。

核心功能:
    - DataParallel: 单机多卡并行训练
    - DDP: 多机多卡分布式训练
    - 混合精度训练: FP16/BF16 自动混合精度
    - 进程组管理和梯度同步
    - 分布式采样器支持

与标准 PyTorch 分布式训练的对比:
    标准 DDP 使用:
        1. 手动初始化进程组
        2. 自己包装模型
        3. 管理数据分发
        4. 处理梯度同步

    GMS DistributedTrainer:
        1. 一键初始化和配置
        2. 自动模型包装
        3. 内置数据分布管理
        4. 完整的同步和通信抽象

Example:
    >>> from gms.diffusion_integration.distributed import (
    ...     DistributedTrainer, setup_distributed, is_main_process
    ... )
    >>>
    >>> # 初始化分布式环境
    >>> setup_distributed(rank=0, world_size=4, backend='nccl')
    >>>
    >>> # 创建分布式训练器
    >>> dist_trainer = DistributedTrainer(
    ...     model=model,
    ...     config=training_config,
    ...     use_ddp=True,
    ...     mixed_precision='fp16'
    ... )
    >>>
    >>> if is_main_process():
    ...     history = dist_trainer.train_full(epochs=100, dataloaders=loaders)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Callable,
)
import os
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None


class PrecisionType(Enum):
    """精度类型枚举"""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class DistributedBackend(Enum):
    """分布式后端枚举"""
    GLOO = "gloo"
    NCCL = "nccl"


@dataclass
class DistributedConfig:
    """分布式训练配置

    Attributes:
        backend: 分布式后端 ('gloo' for CPU, 'nccl' for GPU)
        use_ddp: 是否使用 DDP（否则用 DataParallel）
        world_size: 总进程数/GPU 数
        rank: 当前进程排名
        local_rank: 本地 GPU 编号
        init_method: 初始化方法 (env:// 或 file://)
        precision: 混合精度类型
        find_unused_parameters: DDP 是否查找未使用参数
        bucket_cap_mb: DDP 梯度桶大小（MB）
        gradient_predivide_factor: 梯度预除因子
        sync_bn: 是否使用同步批归一化
    """

    backend: str = "nccl" if torch.cuda.is_available() else "gloo"
    use_ddp: bool = True
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    init_method: str = "env://"
    precision: str = "fp32"
    find_unused_parameters: bool = False
    bucket_cap_mb: float = 25.0
    gradient_predivide_factor: float = 1.0
    sync_bn: bool = False


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """初始化分布式进程组

    在每个进程中调用此函数以建立通信。

    Args:
        rank: 当前进程的全局排名 (0 到 world_size-1)
        world_size: 参与的总进程数
        backend: 后端类型 ('nccl', 'gloo')
        init_method: 初始化方法

    Example:
        >>> # 在每个进程中执行
        >>> import torch.multiprocessing as mp
        >>> def worker(rank, world_size):
        ...     setup_distributed(rank, world_size)
        ...     # 训练代码...
        >>>
        >>> mp.spawn(worker, args=(4,), nprocs=4)
    """
    try:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                rank=rank,
                world_size=world_size,
            )

            if torch.cuda.is_available():
                torch.cuda.set_device(rank)

            if logger:
                logger.info(
                    f"分布式环境已初始化: "
                    f"rank={rank}, world_size={world_size}, backend={backend}"
                )
    except Exception as e:
        if logger:
            logger.error(f"分布式初始化失败: {e}")
        raise


def cleanup_distributed() -> None:
    """清理分布式资源

    训练结束后必须调用此函数释放资源。
    """
    if dist.is_initialized():
        dist.destroy_process_group()

        if logger:
            logger.debug("分布式环境已清理")


def is_distributed_available() -> bool:
    """检查分布式是否可用

    Returns:
        True 如果分布式环境已初始化
    """
    return dist.is_initialized()


def get_world_size() -> int:
    """获取总进程数

    Returns:
        world size（未初始化则返回 1）
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """获取当前进程排名

    Returns:
        当前 rank（未初始化则返回 0）
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """获取本地 GPU 排名

    Returns:
        local rank
    """
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """判断当前进程是否为主进程

    通常只有主进程负责日志、检查点保存等。

    Returns:
        True 如果是主进程 (rank == 0)
    """
    return get_rank() == 0


def synchronize() -> None:
    """同步所有进程

    阻塞直到所有进程到达此点。
    用于确保所有进程在同一位置继续执行。

    Example:
        >>> if is_main_process():
        ...     save_checkpoint(...)
        >>> synchronize()  # 等待所有进程完成
        >>> continue_training()
    """
    if dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(
    tensor: torch.Tensor,
    average: bool = True,
) -> torch.Tensor:
    """跨所有进程进行 AllReduce 操作

    Args:
        tensor: 要聚合的张量
        average: 是否取平均值（否则求和）

    Returns:
        聚合后的张量
    """
    if dist.is_initialized():
        dist.all_reduce(tensor)

        if average:
            tensor /= dist.get_world_size()

    return tensor


def gather_tensors(
    tensor: torch.Tensor,
) -> List[torch.Tensor]:
    """从所有进程收集张量

    Args:
        tensor: 当前进程的张量

    Returns:
        所有进程张量的列表（仅主进程有完整列表）
    """
    if not dist.is_initialized():
        return [tensor]

    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)

    if is_main_process():
        return gathered
    else:
        return []


class DistributedTrainer:
    """GMS 分布式训练器

    包装 GMSTrainer 以支持多 GPU / 多节点训练。

    支持两种模式:
        1. DataParallel (DP): 单机多卡，简单易用
           - 自动数据分割和梯度聚合
           - 适合单机 2-8 卡场景

        2. DistributedDataParallel (DDP): 多机多卡，高性能
           - 每个进程独立控制一个 GPU
           - 更好的性能和可扩展性
           - 适合大规模训练

    混合精度训练:
        - FP16: 标准半精度，广泛支持
        - BF16: Brain Float 16，更好的数值范围（需要硬件支持）

    Attributes:
        base_trainer: 底层 GMSTrainer 实例
        config: DistributedConfig 配置
        model: （可能包装的）模型
        scaler: AMP 的 GradScaler

    Example:
        >>> dist_config = DistributedConfig(
        ...     use_ddp=True,
        ...     precision='fp16',
        ...     world_size=4
        ... )
        >>>
        >>> trainer = GMSTrainer(model, scheduler, forward, backward, train_config)
        >>> dist_trainer = DistributedTrainer(trainer, dist_config)
        >>>
        >>> history = dist_trainer.train_full(epochs=100, dataloaders=loaders)
    """

    def __init__(
        self,
        base_trainer: "GMSTrainer",
        config: Optional[DistributedConfig] = None,
    ):
        """初始化分布式训练器

        Args:
            base_trainer: 基础 GMSTrainer 实例
            config: 分布式配置（可选）
        """
        self.base_trainer = base_trainer
        self.config = config or DistributedConfig()

        self.model = base_trainer.model
        self.device = base_trainer.device
        self.dtype = base_trainer.dtype

        self.scaler: Optional[GradScaler] = None
        self._original_model = base_trainer.model
        self._is_setup = False

        self._setup_model()
        self._setup_precision()

        self._is_setup = True

        if logger and is_main_process():
            logger.info(
                f"DistributedTrainer 初始化完成: "
                f"mode={'DDP' if self.config.use_ddp else 'DP'}, "
                f"precision={self.config.precision}, "
                f"world_size={get_world_size()}"
            )

    def _setup_model(self) -> None:
        """设置模型的分布式包装"""
        if self.config.use_ddp and dist.is_initialized():
            if self.config.sync_bn:
                self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

            self.model = self.model.to(self.device)
            self.model = DDP(
                self.model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=self.config.find_unused_parameters,
                bucket_cap_mb=int(self.config.bucket_cap_mb * 1024 * 1024),
            )

            if logger:
                logger.debug("模型已包装为 DDP")

        elif torch.cuda.device_count() > 1 and not self.config.use_ddp:
            self.model = nn.DataParallel(self.model)

            if logger:
                logger.debug(f"模型已包装为 DataParallel ({torch.cuda.device_count()} GPUs)")

        else:
            self.model = self.model.to(self.device)

    def _setup_precision(self) -> None:
        """设置混合精度训练"""
        precision = PrecisionType(self.config.precision)

        if precision == PrecisionType.FP16:
            self.scaler = GradScaler()
            if logger:
                logger.debug("启用 FP16 混合精度训练")

        elif precision == PrecisionType.BF16:
            if torch.cuda.is_bf16_supported():
                self.scaler = GradScaler(dtype=torch.bfloat16)
                if logger:
                    logger.debug("启用 BF16 混合精度训练")
            else:
                if logger:
                    logger.warning("BF16 不受支持，回退到 FP32")
                self.scaler = None

        else:
            self.scaler = None

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Any:
        """分布式训练 epoch

        处理数据采样器的设置和梯度的全局同步。

        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch

        Returns:
            EpochMetrics（仅在主进程上有有效值）
        """
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        metrics = self.base_trainer.train_epoch(dataloader, epoch)

        if dist.is_initialized():
            loss_tensor = torch.tensor(
                metrics.total_loss, device=self.device
            )
            all_reduce_tensor(loss_tensor, average=True)
            metrics.total_loss = loss_tensor.item()

        return metrics

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int,
    ) -> Any:
        """分布式验证

        Args:
            dataloader: 验证数据加载器
            epoch: 当前 epoch

        Returns:
            验证指标
        """
        metrics = self.base_trainer.validate(dataloader, epoch)

        if dist.is_initialized():
            loss_tensor = torch.tensor(
                metrics.total_loss, device=self.device
            )
            all_reduce_tensor(loss_tensor, average=True)
            metrics.total_loss = loss_tensor.item()

        return metrics

    def train_full(
        self,
        epochs: int,
        dataloaders: Union[DataLoader, Dict[str, DataLoader]],
        **kwargs,
    ) -> Any:
        """完整分布式训练流程

        仅在主进程上执行完整训练循环，
        其他进程配合计算和同步。

        Args:
            epochs: 训练轮数
            dataloaders: 数据加载器
            **kwargs: 传递给 base_trainer.train_full 的额外参数

        Returns:
            TrainingHistory（仅主进程）
        """
        if is_main_process():
            history = self.base_trainer.train_full(epochs, dataloaders, **kwargs)
            return history
        else:
            for epoch in range(self.base_trainer.current_epoch + 1,
                              self.base_trainer.current_epoch + epochs + 1):
                self.base_trainer.current_epoch = epoch

                if isinstance(dataloaders, dict):
                    train_loader = dataloaders.get('train')
                else:
                    train_loader = dataloaders

                if train_loader:
                    self.train_epoch(train_loader, epoch)

                if isinstance(dataloaders, dict):
                    val_loader = dataloaders.get('val')
                    if val_loader:
                        self.validate(val_loader, epoch)

            return None

    def save_checkpoint(
        self,
        checkpoint_path: str,
        epoch: int,
        metrics: Optional[Dict] = None,
    ) -> None:
        """保存分布式检查点

        仅主进程负责写入磁盘。

        Args:
            checkpoint_path: 保存路径
            epoch: 当前 epoch
            metrics: 当前指标
        """
        if is_main_process():
            self.base_trainer._save_checkpoint(epoch, metrics or {})

        synchronize()

    def load_checkpoint(
        self,
        checkpoint_path: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """加载检查点

        所有进程都加载相同的状态。

        Args:
            checkpoint_path: 检查点路径
            **kwargs: 传递给 base_trainer.load_checkpoint 的参数

        Returns:
            检查点数据
        """
        return self.base_trainer.load_checkpoint(checkpoint_path, **kwargs)

    def unwrap_model(self) -> nn.Module:
        """解包模型获取原始模型

        从 DDP/DataParallel 包装中提取原始模型。

        Returns:
            原始 nn.Module 实例
        """
        model = self.model

        if isinstance(model, DDP):
            return model.module
        elif isinstance(model, nn.DataParallel):
            return model.module
        else:
            return model

    def get_effective_batch_size(self) -> int:
        """获取有效批次大小

        考虑到梯度累积和分布式设置的真正批次大小。

        Returns:
            有效批次大小
        """
        base_bs = self.base_trainer.config.batch_size
        world_size = get_world_size()

        return base_bs * world_size

    def export_for_inference(self) -> Dict[str, Any]:
        """导出推理状态（解包后）

        Returns:
            包含原始模型状态的字典
        """
        original_model = self.unwrap_model()
        export_state = {
            'model_state_dict': original_model.state_dict(),
            'noise_scheduler': self.base_trainer.noise_scheduler,
            'backward_process': self.base_trainer.backward_process,
            'config': {
                'device': str(self.device),
                'dtype': str(self.dtype),
            },
        }

        if self.base_trainer.gmm_parameters is not None:
            export_state['gmm_params'] = self.base_trainer.gmm_parameters.to_dict()

        return export_state


def create_distributed_sampler(
    dataset: "Dataset",
    shuffle: bool = True,
    seed: int = 42,
) -> DistributedSampler:
    """创建分布式采样器

    为每个进程分配不同的数据子集。

    Args:
        dataset: 数据集对象
        shuffle: 是否打乱
        seed: 随机种子

    Returns:
        DistributedSampler 实例

    Example:
        >>> sampler = create_distributed_sampler(train_dataset)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=32)
    """
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed,
    )


def launch_distributed_training(
    main_fn: Callable,
    num_gpus: int = -1,
    args: tuple = (),
) -> None:
    """启动分布式训练

    自动检测可用 GPU 并启动多个进程。

    Args:
        main_fn: 主函数（接收 rank 和 args 作为参数）
        num_gpus: 使用的 GPU 数量（-1 表示全部）
        args: 传递给 main_fn 的额外参数

    Example:
        >>> def training_worker(rank, config_path):
        ...     setup_distributed(rank, world_size=torch.cuda.device_count())
        ...     # 训练代码...
        ...     cleanup_distributed()
        >>>
        >>> launch_distributed_training(training_worker, num_gpus=4)
    """
    import torch.multiprocessing as mp

    if num_gpus <= 0:
        num_gpus = torch.cuda.device_count()

    if num_gpus < 1:
        if logger:
            logger.warning("没有可用的 GPU，使用单进程模式")
        main_fn(0, *args)
        return

    if logger:
        logger.info(f"启动 {num_gpus} 个分布式训练进程")

    mp.spawn(
        main_fn,
        args=(num_gpus,) + args,
        nprocs=num_gpus,
        join=True,
    )


if __name__ == "__main__":
    print("GMS Distributed Training Support")
    print("\n主要组件:")
    print("  - DistributedTrainer: 分布式训练器")
    print("  - DistributedConfig: 分布式配置")
    print("  - 工具函数:")
    print("    - setup_distributed(): 初始化进程组")
    print("    - cleanup_distributed(): 清理资源")
    print("    - is_main_process(): 判断主进程")
    print("    - synchronize(): 同步所有进程")
    print("\n快速开始:")
    print("  from gms.diffusion_integration.distributed import setup_distributed, DistributedTrainer")
    print("  setup_distributed(rank=0, world_size=4)")
    print("  dist_trainer = DistributedTrainer(base_trainer, DistributedConfig(use_ddp=True))")
