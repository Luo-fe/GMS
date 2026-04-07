"""GMS 检查点管理和状态恢复

提供完善的检查点保存、加载和恢复功能，支持：
- 自动定期保存检查点（按 epoch 或 step）
- 保存完整状态（模型参数、优化器状态、随机状态、GMM 参数等）
- 从检查点恢复训练（支持断点续训）
- 管理多个检查点版本（保留最佳 N 个）
- 清理旧检查点以节省磁盘空间

与标准 PyTorch 检查点的对比:
    标准 torch.save():
        - 仅保存模型 state_dict
        - 需要手动管理版本
        - 无自动清理机制

    GMS CheckpointManager:
        - 保存完整训练状态
        - 自动版本管理
        - 智能清理策略
        - 完整的元数据记录

Example:
    >>> from gms.diffusion_integration.checkpoint import CheckpointManager
    >>>
    >>> manager = CheckpointManager(checkpoint_dir='./checkpoints', keep_n_best=3)
    >>>
    >>> # 保存检查点
    >>> manager.save(trainer, epoch=10, metrics={'loss': 0.123})
    >>>
    >>> # 加载最新检查点
    >>> checkpoint = manager.load_latest()
    >>>
    >>> # 获取最佳模型路径
    >>> best_path = manager.get_best_checkpoint()
"""

from dataclasses import dataclass, field
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Callable
)
import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import torch

from .trainer import TrainingHistory

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None


@dataclass
class CheckpointMetadata:
    """检查点元数据

    存储检查点的描述性信息，用于快速查询和管理。

    Attributes:
        filename: 文件名
        filepath: 完整文件路径
        epoch: 对应的 epoch 编号
        global_step: 全局步数
        timestamp: 创建时间戳
        loss: 训练/验证损失
        file_size_mb: 文件大小（MB）
        is_best: 是否为最佳模型
        checksum: 文件校验和（SHA256）
        extra_info: 其他自定义信息
    """

    filename: str
    filepath: str
    epoch: int
    global_step: int
    timestamp: str
    loss: Optional[float] = None
    file_size_mb: float = 0.0
    is_best: bool = False
    checksum: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'filename': self.filename,
            'filepath': self.filepath,
            'epoch': self.epoch,
            'global_step': self.global_step,
            'timestamp': self.timestamp,
            'loss': self.loss,
            'file_size_mb': self.file_size_mb,
            'is_best': self.is_best,
            'checksum': self.checksum,
            **self.extra_info,
        }


@dataclass
class CheckpointConfig:
    """检查点管理配置

    Attributes:
        checkpoint_dir: 检查点根目录
        keep_n_best: 保留的最佳检查点数量
        max_checkpoints: 最大检查点总数限制（0 表示不限制）
        save_every_n_epochs: 每 N 个 epoch 保存一次
        save_every_n_steps: 每 N 个 step 保存一次（可选）
        filename_template: 文件名模板，支持 {epoch}, {step}, {timestamp} 占位符
        compress: 是否压缩检查点
        backup_before_overwrite: 覆盖前是否备份
        verify_integrity: 保存后是否验证完整性
    """

    checkpoint_dir: str = "./checkpoints"
    keep_n_best: int = 3
    max_checkpoints: int = 0
    save_every_n_epochs: int = 1
    save_every_n_steps: Optional[int] = None
    filename_template: str = "checkpoint_epoch_{epoch}_step_{step}.pt"
    compress: bool = False
    backup_before_overwrite: bool = True
    verify_integrity: bool = False

    def __post_init__(self):
        """初始化后验证"""
        if self.keep_n_best < 1:
            raise ValueError(f"keep_n_best 必须至少为 1，当前值: {self.keep_n_best}")


class CheckpointManager:
    """GMS 检查点管理器

    提供完整的检查点生命周期管理：
    - 保存：将完整训练状态序列化到磁盘
    - 加载：从磁盘恢复训练状态
    - 版本管理：维护多个检查点版本
    - 清理：删除过期或低质量的检查点

    检查点内容结构:
        {
            'epoch': int,                    # 当前 epoch
            'global_step': int,              # 全局步数
            'model_state_dict': dict,        # 模型参数
            'optimizer_state_dict': dict,    # 优化器状态
            'gmm_params': GMMParameters,     # GMM 参数
            'scheduler_state_dict': dict,    # 学习率调度器状态
            'training_history': TrainingHistory,  # 训练历史
            'metrics': dict,                 # 当前指标
            'config': TrainingConfig,        # 训练配置
            'rng_state': Tensor,             # 随机数生成器状态
            'timestamp': str,                # 时间戳
            'metadata': dict,                # 额外元数据
        }

    Attributes:
        config: CheckpointConfig 配置实例
        _checkpoint_registry: 已注册的检查点列表

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir='./my_checkpoints',
        ...     keep_n_best=5
        ... )
        >>>
        >>> # 在训练循环中保存
        >>> for epoch in range(100):
        ...     train(...)
        ...     if epoch % 10 == 0:
        ...         manager.save(trainer, epoch=epoch, metrics=metrics)
        >>>
        >>> # 恢复训练
        >>> checkpoint = manager.load_latest()
        >>> trainer.load_checkpoint(checkpoint)
    """

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        config: Optional[CheckpointConfig] = None,
    ):
        """初始化检查点管理器

        Args:
            checkpoint_dir: 检查点保存目录
            config: CheckpointConfig 配置实例（可选）
        """
        if config is None:
            config = CheckpointConfig(checkpoint_dir=checkpoint_dir)

        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoint_registry: Dict[str, CheckpointMetadata] = {}
        self._best_loss: float = float('inf')
        self._best_checkpoint_path: Optional[str] = None

        self._load_existing_checkpoints()

        if logger:
            logger.info(
                f"CheckpointManager 初始化完成: "
                f"dir={checkpoint_dir}, "
                f"existing={len(self._checkpoint_registry)}"
            )

    def _load_existing_checkpoints(self) -> None:
        """加载已存在的检查点到注册表"""
        for filepath in self.checkpoint_dir.glob("*.pt"):
            try:
                metadata = self._extract_metadata(filepath)
                if metadata:
                    self._checkpoint_registry[filepath.name] = metadata

                    if (metadata.loss is not None and
                        metadata.loss < self._best_loss):
                        self._best_loss = metadata.loss
                        self._best_checkpoint_path = str(filepath)
                        metadata.is_best = True
            except Exception as e:
                if logger:
                    logger.warning(f"无法加载检查点元数据 {filepath}: {e}")

    def _extract_metadata(
        self,
        filepath: Path,
    ) -> Optional[CheckpointMetadata]:
        """从检查点文件提取元数据

        Args:
            filepath: 检查点文件路径

        Returns:
            CheckpointMetadata 或 None（如果提取失败）
        """
        try:
            checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

            epoch = checkpoint.get('epoch', 0)
            step = checkpoint.get('global_step', 0)
            timestamp = checkpoint.get('timestamp', datetime.now().isoformat())
            metrics = checkpoint.get('metrics', {})
            loss = metrics.get('total_loss') if metrics else None

            file_size = filepath.stat().st_size / (1024 * 1024)

            return CheckpointMetadata(
                filename=filepath.name,
                filepath=str(filepath),
                epoch=epoch,
                global_step=step,
                timestamp=timestamp,
                loss=loss,
                file_size_mb=file_size,
            )

        except Exception as e:
            if logger:
                logger.debug(f"提取元数据失败 {filepath}: {e}")
            return None

    def _compute_checksum(self, filepath: Path) -> str:
        """计算文件的 SHA256 校验和

        Args:
            filepath: 文件路径

        Returns:
            十六进制校验和字符串
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def save(
        self,
        trainer: "GMSTrainer",
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        extra_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """保存检查点

        将完整的训练状态保存到磁盘。

        Args:
            trainer: GMSTrainer 实例
            epoch: 当前 epoch 编号
            metrics: 当前指标字典
            is_best: 是否标记为最佳模型
            extra_data: 要包含在检查点中的额外数据

        Returns:
            保存的文件路径

        Raises:
            IOError: 如果保存失败

        Example:
            >>> path = manager.save(
            ...     trainer=trainer,
            ...     epoch=50,
            ...     metrics={'train_loss': 0.123, 'val_loss': 0.145},
            ...     is_best=True
            ... )
            >>> print(f"检查点已保存到: {path}")
        """
        timestamp = datetime.now().isoformat()
        filename = self.config.filename_template.format(
            epoch=epoch,
            step=trainer.global_step,
            timestamp=timestamp.replace(':', '-'),
        )
        filepath = self.checkpoint_dir / filename

        checkpoint_data = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': (
                trainer.scheduler.state_dict() if trainer.scheduler else None
            ),
            'training_history': trainer.history.to_dict(),
            'config': trainer.config.to_dict(),
            'metrics': metrics or {},
            'rng_state': torch.get_rng_state(),
            'timestamp': timestamp,
            'metadata': {
                'python_version': '3.x',
                'pytorch_version': torch.__version__,
                'gms_version': '0.1.0',
            },
        }

        if trainer.gmm_parameters is not None:
            checkpoint_data['gmm_params'] = trainer.gmm_parameters.to_dict()

        if trainer.scaler is not None:
            checkpoint_data['scaler_state_dict'] = trainer.scaler.state_dict()

        if trainer.condition_encoder is not None:
            checkpoint_data['condition_encoder_state_dict'] = \
                trainer.condition_encoder.state_dict()

        if trainer.condition_injector is not None:
            checkpoint_data['condition_injector_state_dict'] = \
                trainer.condition_injector.state_dict()

        if extra_data:
            checkpoint_data.update(extra_data)

        if self.config.backup_before_overwrite and filepath.exists():
            backup_path = filepath.with_suffix('.pt.bak')
            shutil.copy2(filepath, backup_path)
            if logger:
                logger.debug(f"已备份旧检查点到: {backup_path}")

        try:
            torch.save(checkpoint_data, filepath)

            if self.config.verify_integrity:
                test_load = torch.load(filepath, map_location='cpu', weights_only=False)
                assert test_load['epoch'] == epoch
                del test_load

            file_size = filepath.stat().st_size / (1024 * 1024)
            checksum = self._compute_checksum(filepath) if self.config.verify_integrity else None

            current_loss = (metrics.get('total_loss') or
                          metrics.get('val_loss') or
                          metrics.get('train_loss'))

            metadata = CheckpointMetadata(
                filename=filename,
                filepath=str(filepath),
                epoch=epoch,
                global_step=trainer.global_step,
                timestamp=timestamp,
                loss=current_loss,
                file_size_mb=file_size,
                is_best=is_best,
                checksum=checksum,
            )

            self._checkpoint_registry[filename] = metadata

            if current_loss is not None and current_loss < self._best_loss:
                self._best_loss = current_loss
                self._best_checkpoint_path = str(filepath)
                metadata.is_best = True

                best_filepath = self.checkpoint_dir / "best_model.pt"
                torch.save(checkpoint_data, best_filepath)

                if logger:
                    logger.info(f"新的最佳模型! Loss: {current_loss:.6f}")

            self._cleanup_old_checkpoints()

            if logger:
                logger.info(
                    f"检查点已保存: {filepath} "
                    f"(epoch={epoch}, step={trainer.global_step}, "
                    f"size={file_size:.2f}MB)"
                )

            return str(filepath)

        except Exception as e:
            if filepath.exists():
                filepath.unlink(missing_ok=True)
            raise IOError(f"保存检查点失败: {e}") from e

    def load(
        self,
        checkpoint_path: str,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """加载指定检查点

        从指定路径加载检查点数据。

        Args:
            checkpoint_path: 检查点文件路径
            map_location: 映射到的设备（如 'cpu', 'cuda:0'）
            strict: 是否严格匹配模型参数键名

        Returns:
            检查点数据字典

        Raises:
            FileNotFoundError: 如果文件不存在
            RuntimeError: 如果加载失败或文件损坏

        Example:
            >>> ckpt = manager.load('./checkpoints/checkpoint_epoch_50.pt')
            >>> print(f"Epoch: {ckpt['epoch']}, Step: {ckpt['global_step']}")
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")

        try:
            if map_location is None:
                checkpoint = torch.load(
                    checkpoint_path,
                    weights_only=False
                )
            else:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=map_location,
                    weights_only=False
                )

            required_keys = ['epoch', 'model_state_dict']
            for key in required_keys:
                if key not in checkpoint:
                    raise RuntimeError(
                        f"检查点缺少必需字段: {key}。"
                        f"该文件可能损坏或不兼容。"
                    )

            if self.config.verify_integrity:
                expected_epoch = checkpoint['epoch']
                assert isinstance(expected_epoch, int), "epoch 字段类型错误"

            if logger:
                logger.info(
                    f"检查点已加载: {checkpoint_path} "
                    f"(epoch={checkpoint.get('epoch', '?')})"
                )

            return checkpoint

        except Exception as e:
            raise RuntimeError(
                f"加载检查点失败: {checkpoint_path}\n"
                f"错误: {e}"
            ) from e

    def load_latest(
        self,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """加载最新的检查点

        自动查找并加载时间戳最近的检查点。

        Args:
            map_location: 目标设备
            strict: 是否严格匹配参数

        Returns:
            最新的检查点数据，如果没有则返回 None

        Example:
            >>> ckpt = manager.load_latest()
            >>> if ckpt:
            ...     print(f"从 epoch {ckpt['epoch']} 恢复")
        """
        if not self._checkpoint_registry:
            if logger:
                logger.warning("没有找到任何检查点")
            return None

        latest_meta = max(
            self._checkpoint_registry.values(),
            key=lambda m: m.timestamp
        )

        try:
            return self.load(latest_meta.filepath, map_location, strict)
        except Exception as e:
            if logger:
                logger.error(f"加载最新检查点失败: {e}")

            sorted_checkpoints = sorted(
                self._checkpoint_registry.values(),
                key=lambda m: m.epoch,
                reverse=True
            )

            for meta in sorted_checkpoints[1:]:
                try:
                    return self.load(meta.filepath, map_location, strict)
                except Exception:
                    continue

            return None

    def get_best_checkpoint(self) -> Optional[str]:
        """获取最佳模型的路径

        Returns:
            最佳检查点的完整路径，如果不存在则返回 None

        Example:
            >>> best_path = manager.get_best_checkpoint()
            >>> if best_path:
            ...     trainer.load_checkpoint(best_path)
        """
        if self._best_checkpoint_path and os.path.exists(self._best_checkpoint_path):
            return self._best_checkpoint_path

        best_file = self.checkpoint_dir / "best_model.pt"
        if best_file.exists():
            return str(best_file)

        if self._checkpoint_registry:
            valid_checkpoints = [
                m for m in self._checkpoint_registry.values()
                if m.loss is not None
            ]

            if valid_checkpoints:
                best = min(valid_checkpoints, key=lambda m: m.loss)
                return best.filepath

        return None

    def list_checkpoints(
        self,
        sort_by: str = "epoch",
        descending: bool = True,
    ) -> List[CheckpointMetadata]:
        """列出所有已注册的检查点

        Args:
            sort_by: 排序依据 ('epoch', 'timestamp', 'loss', 'file_size')
            descending: 是否降序排列

        Returns:
            CheckpointMetadata 列表

        Example:
            >>> checkpoints = manager.list_checkpoints(sort_by='loss')
            >>> for ckpt in checkpoints[:5]:
            ...     print(f"Epoch {ckpt.epoch}: Loss={ckpt.loss}")
        """
        checkpoints = list(self._checkpoint_registry.values())

        valid_sort_keys = {'epoch', 'timestamp', 'loss', 'file_size'}
        if sort_by not in valid_sort_keys:
            sort_by = 'epoch'

        def sort_key(meta: CheckpointMetadata):
            value = getattr(meta, sort_by, None)
            if value is None:
                return float('inf') if descending else float('-inf')
            return value

        return sorted(checkpoints, key=sort_key, reverse=descending)

    def cleanup(
        self,
        keep_n_best: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[str]:
        """清理旧检查点

        删除多余的检查点，仅保留最佳的 N 个。

        Args:
            keep_n_best: 要保留的数量（默认使用配置值）
            dry_run: 如果为 True，只返回要删除的列表但不实际删除

        Returns:
            被删除的文件路径列表

        Example:
            >>> deleted = manager.cleanup(keep_n_best=3)
            >>> print(f"已删除 {len(deleted)} 个旧检查点")
        """
        if keep_n_best is None:
            keep_n_best = self.config.keep_n_best

        all_checkpoints = list(self._checkpoint_registry.values())

        if len(all_checkpoints) <= keep_n_best:
            if logger:
                logger.debug(f"无需清理: 当前 {len(all_checkpoints)} 个 <= 保留 {keep_n_best} 个")
            return []

        valid_with_loss = [
            c for c in all_checkpoints
            if c.loss is not None
        ]

        if len(valid_with_loss) > 0:
            sorted_by_loss = sorted(valid_with_loss, key=lambda c: c.loss)
            to_keep = set(c.filename for c in sorted_by_loss[:keep_n_best])
        else:
            sorted_by_epoch = sorted(all_checkpoints, key=lambda c: c.epoch, reverse=True)
            to_keep = set(c.filename for c in sorted_by_epoch[:keep_n_best])

        to_delete = [
            c for c in all_checkpoints
            if c.filename not in to_keep and c.filename != "best_model.pt"
        ]

        deleted_paths = []

        for meta in to_delete:
            filepath = Path(meta.filepath)

            if dry_run:
                deleted_paths.append(str(filepath))
                continue

            try:
                if filepath.exists():
                    filepath.unlink()
                    deleted_paths.append(str(filepath))

                    if meta.filename in self._checkpoint_registry:
                        del self._checkpoint_registry[meta.filename]

                    backup_path = filepath.with_suffix('.pt.bak')
                    if backup_path.exists():
                        backup_path.unlink()

                    if logger:
                        logger.debug(f"已删除检查点: {filepath}")

            except Exception as e:
                if logger:
                    logger.warning(f"删除检查点失败 {filepath}: {e}")

        if logger and not dry_run:
            logger.info(
                f"清理完成: 删除了 {len(deleted_paths)} 个检查点, "
                f"保留 {len(to_keep)} 个"
            )

        return deleted_paths

    def _cleanup_old_checkpoints(self) -> None:
        """内部方法：根据配置自动清理"""
        if self.config.max_checkpoints > 0:
            total = len(self._checkpoint_registry)
            if total > self.config.max_checkpoints:
                self.cleanup(keep_n_best=self.config.keep_n_best)

    def restore_training(
        self,
        trainer: "GMSTrainer",
        checkpoint_path: Optional[str] = None,
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """从检查点完全恢复训练状态

        一站式恢复函数，加载检查点并应用到训练器。

        Args:
            trainer: GMSTrainer 实例
            checkpoint_path: 检查点路径（None 则使用最新）
            load_optimizer: 是否恢复优化器状态

        Returns:
            加载的检查点数据

        Example:
            >>> checkpoint = manager.restore_training(trainer)
            >>> print(f"从 epoch {checkpoint['epoch']} 继续训练")
        """
        if checkpoint_path is None:
            checkpoint = self.load_latest(map_location=trainer.device)
        else:
            checkpoint = self.load(checkpoint_path, map_location=trainer.device)

        if checkpoint is None:
            raise RuntimeError("没有找到可用的检查点")

        trainer.model.load_state_dict(
            checkpoint['model_state_dict'],
            strict=True,
        )

        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if ('scheduler_state_dict' in checkpoint and
            trainer.scheduler is not None and
            checkpoint.get('scheduler_state_dict') is not None):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and trainer.scaler is not None:
            trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        trainer.current_epoch = checkpoint.get('epoch', 0)
        trainer.global_step = checkpoint.get('global_step', 0)

        if 'gmm_params' in checkpoint and checkpoint['gmm_params'] is not None:
            from gms.gmm_optimization.gmm_parameters import GMMParameters
            trainer.gmm_parameters = GMMParameters.from_dict(checkpoint['gmm_params'])

        if ('condition_encoder_state_dict' in checkpoint and
            trainer.condition_encoder is not None):
            trainer.condition_encoder.load_state_dict(
                checkpoint['condition_encoder_state_dict']
            )

        if ('condition_injector_state_dict' in checkpoint and
            trainer.condition_injector is not None):
            trainer.condition_injector.load_state_dict(
                checkpoint['condition_injector_state_dict']
            )

        if 'training_history' in checkpoint:
            trainer.history = TrainingHistory.from_dict(checkpoint['training_history'])

        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state'].to(trainer.device))

        if logger:
            logger.info(
                f"训练状态已恢复: "
                f"epoch={trainer.current_epoch}, "
                f"step={trainer.global_step}"
            )

        return checkpoint

    def export_checkpoint_summary(
        self,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """导出所有检查点的摘要信息

        Args:
            output_path: 输出 JSON 文件路径（可选）

        Returns:
            摘要信息字典
        """
        checkpoints = self.list_checkpoints(sort_by="epoch")

        summary = {
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': len(checkpoints),
            'best_loss': self._best_loss,
            'best_checkpoint': self._best_checkpoint_path,
            'checkpoints': [c.to_dict() for c in checkpoints],
            'total_size_mb': sum(c.file_size_mb for c in checkpoints),
            'generated_at': datetime.now().isoformat(),
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            if logger:
                logger.info(f"检查点摘要已导出到: {output_path}")

        return summary

    def verify_checkpoint(
        self,
        checkpoint_path: str,
    ) -> Tuple[bool, Optional[str]]:
        """验证检查点完整性

        Args:
            checkpoint_path: 检查点路径

        Returns:
            (是否有效, 错误消息) 元组
        """
        if not os.path.exists(checkpoint_path):
            return False, f"文件不存在: {checkpoint_path}"

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False
            )

            required_fields = ['epoch', 'model_state_dict']
            for field in required_fields:
                if field not in checkpoint:
                    return False, f"缺少必需字段: {field}"

            if not isinstance(checkpoint['epoch'], int):
                return False, "epoch 字段类型错误"

            state_dict = checkpoint['model_state_dict']
            if not isinstance(state_dict, dict):
                return False, "model_state_dict 不是字典类型"

            if self.config.verify_integrity:
                computed_checksum = self._compute_checksum(Path(checkpoint_path))
                stored_checksum = self._checkpoint_registry.get(
                    Path(checkpoint_path).name, CheckpointMetadata(
                        filename="", filepath="", epoch=0, global_step=0, timestamp=""
                    )
                ).checksum

                if stored_checksum and computed_checksum != stored_checksum:
                    return False, "校验和不匹配"

            return True, None

        except Exception as e:
            return False, f"验证失败: {str(e)}"

    def get_statistics(self) -> Dict[str, Any]:
        """获取检查点统计信息

        Returns:
            统计信息字典
        """
        checkpoints = list(self._checkpoint_registry.values())

        total_size = sum(c.file_size_mb for c in checkpoints)
        losses = [c.loss for c in checkpoints if c.loss is not None]

        stats = {
            'total_count': len(checkpoints),
            'total_size_mb': round(total_size, 2),
            'epoch_range': (
                min((c.epoch for c in checkpoints), default=0),
                max((c.epoch for c in checkpoints), default=0),
            ),
            'best_loss': min(losses) if losses else None,
            'worst_loss': max(losses) if losses else None,
            'avg_loss': sum(losses) / len(losses) if losses else None,
            'has_best_model': self._best_checkpoint_path is not None,
            'directory': str(self.checkpoint_dir),
        }

        return stats


def create_checkpoint_manager(
    checkpoint_dir: str = "./checkpoints",
    keep_n_best: int = 3,
) -> CheckpointManager:
    """快速创建检查点管理器

    便捷工厂函数。

    Args:
        checkpoint_dir: 检查点目录
        keep_n_best: 保留的最佳检查点数量

    Returns:
        CheckpointManager 实例

    Example:
        >>> manager = create_checkpoint_manager("./my_ckpt", keep_n_best=5)
        >>> manager.save(trainer, epoch=10, metrics={'loss': 0.1})
    """
    config = CheckpointConfig(
        checkpoint_dir=checkpoint_dir,
        keep_n_best=keep_n_best,
    )
    return CheckpointManager(config=config)


if __name__ == "__main__":
    print("GMS Checkpoint Manager - 用于训练状态的保存和恢复")
    print("\n主要组件:")
    print("  - CheckpointManager: 检查点管理器")
    print("  - CheckpointMetadata: 检查点元数据")
    print("  - CheckpointConfig: 检查点配置")
    print("\n快速开始:")
    print("  from gms.diffusion_integration.checkpoint import CheckpointManager")
    print("  manager = CheckpointManager(checkpoint_dir='./checkpoints')")
    print("  manager.save(trainer, epoch=10, metrics={'loss': 0.123})")
    print("  checkpoint = manager.load_latest()")
