"""检查点管理器 - 采样过程的中断和恢复机制"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import json
import pickle
import logging
import time
import shutil
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SamplingCheckpoint:
    """采样检查点数据类

    存储采样过程中的完整状态，用于中断后恢复。

    Attributes:
        checkpoint_id: 唯一标识符
        timestamp: 创建时间戳
        current_step: 当前步骤索引
        total_steps: 总步数
        random_state: 随机数生成器状态
        intermediate_results: 中间结果数据
        metadata: 额外的元信息
        scheduler_state: 调度器状态（可选）
        controller_state: 时间步控制器状态（可选）
    """

    checkpoint_id: str
    timestamp: float
    current_step: int
    total_steps: int
    random_state: Optional[Dict[str, Any]] = None
    intermediate_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    scheduler_state: Optional[Dict[str, Any]] = None
    controller_state: Optional[Dict[str, Any]] = None

    @property
    def progress(self) -> float:
        """计算完成进度 (0.0 - 1.0)"""
        if self.total_steps <= 0:
            return 0.0
        return min(self.current_step / self.total_steps, 1.0)

    @property
    def created_at(self) -> str:
        """返回人类可读的创建时间"""
        return datetime.fromtimestamp(self.timestamp).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式

        Returns:
            可序列化的字典
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingCheckpoint":
        """从字典创建实例

        Args:
            data: 字典数据

        Returns:
            检查点实例
        """
        return cls(**data)


class CheckpointCleanupPolicy:
    """检查点清理策略

    管理检查点的生命周期和存储空间。
    """

    KEEP_ALL = "keep_all"
    KEEP_LAST_N = "keep_last_n"
    KEEP_BY_TIME = "keep_by_time"
    KEEP_BEST = "keep_best"

    def __init__(
        self,
        strategy: str = KEEP_LAST_N,
        max_checkpoints: int = 5,
        max_age_hours: float = 24.0,
        min_free_space_mb: float = 100.0,
    ) -> None:
        """初始化清理策略

        Args:
            strategy: 清理策略类型
            max_checkpoints: 最大保留数量
            max_age_hours: 最大保留时间（小时）
            min_free_space_mb: 最小剩余空间（MB）
        """
        self.strategy = strategy
        self.max_checkpoints = max_checkpoints
        self.max_age_hours = max_age_hours
        self.min_free_space_mb = min_free_space_mb


class SamplingCheckpointManager:
    """采样检查点管理器

    管理采样过程的保存和恢复，支持：
    - 手动/自动定期保存
    - JSON 和 Pickle 格式
    - 检查点清理和管理
    - 完整的状态恢复

    Attributes:
        save_dir: 检查点保存目录
        auto_save_interval: 自动保存间隔（步数）
        format: 保存格式 ('json' 或 'pickle')
    """

    def __init__(
        self,
        save_dir: Union[str, Path],
        auto_save_interval: int = 10,
        format: str = "pickle",
        cleanup_policy: Optional[CheckpointCleanupPolicy] = None,
        compress: bool = False,
    ) -> None:
        """初始化检查点管理器

        Args:
            save_dir: 检查点保存目录路径
            auto_save_interval: 每隔多少步自动保存一次（0 表示不自动保存）
            format: 保存格式，'json' 或 'pickle'
            cleanup_policy: 清理策略配置
            compress: 是否压缩保存文件
        """
        if format not in ("json", "pickle"):
            raise ValueError(f"不支持的格式: {format}，必须是 'json' 或 'pickle'")

        self.save_dir = Path(save_dir)
        self.auto_save_interval = auto_save_interval
        self.format = format
        self.cleanup_policy = cleanup_policy or CheckpointCleanupPolicy()
        self.compress = compress

        # 创建目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 内部状态
        self._last_auto_save_step: int = 0
        self._checkpoint_count: int = 0
        self._checkpoints: List[SamplingCheckpoint] = []

        logger.info(
            f"初始化 SamplingCheckpointManager: "
            f"dir={self.save_dir}, "
            f"format={format}, "
            f"auto_save={auto_save_interval}"
        )

    def create_checkpoint(
        self,
        current_step: int,
        total_steps: int,
        random_state: Optional[Dict[str, Any]] = None,
        intermediate_results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scheduler_state: Optional[Dict[str, Any]] = None,
        controller_state: Optional[Dict[str, Any]] = None,
    ) -> SamplingCheckpoint:
        """创建新的检查点

        Args:
            current_step: 当前步骤
            total_steps: 总步数
            random_state: 随机状态
            intermediate_results: 中间结果
            metadata: 元数据
            scheduler_state: 调度器状态
            controller_state: 控制器状态

        Returns:
            创建的检查点对象
        """
        import uuid
        checkpoint_id = (
            f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            f"_{current_step}"
        )
        checkpoint = SamplingCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            current_step=current_step,
            total_steps=total_steps,
            random_state=random_state,
            intermediate_results=intermediate_results,
            metadata=metadata or {},
            scheduler_state=scheduler_state,
            controller_state=controller_state,
        )
        return checkpoint

    def save_checkpoint(
        self,
        checkpoint: SamplingCheckpoint,
        filename: Optional[str] = None,
    ) -> Path:
        """保存检查点到文件

        Args:
            checkpoint: 要保存的检查点对象
            filename: 自定义文件名（可选）

        Returns:
            保存文件的路径

        Raises:
            IOError: 如果保存失败
        """
        if filename is None:
            ext = ".pkl.gz" if self.compress else ".pkl"
            if self.format == "json":
                ext = ".json"
            filename = f"{checkpoint.checkpoint_id}{ext}"

        filepath = self.save_dir / filename

        try:
            if self.format == "json":
                self._save_json(checkpoint, filepath)
            else:
                self._save_pickle(checkpoint, filepath)

            self._checkpoint_count += 1
            self._checkpoints.append(checkpoint)

            logger.info(f"已保存检查点: {filepath.name} (步骤 {checkpoint.current_step})")

            # 执行清理
            self._cleanup_if_needed()

            return filepath

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            raise IOError(f"无法保存检查点到 {filepath}: {e}")

    def _save_json(self, checkpoint: SamplingCheckpoint, filepath: Path) -> None:
        """以 JSON 格式保存

        Args:
            checkpoint: 检查点对象
            filepath: 目标路径
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

    def _save_pickle(self, checkpoint: SamplingCheckpoint, filepath: Path) -> None:
        """以 Pickle 格式保存

        Args:
            checkpoint: 检查点对象
            filepath: 目标路径
        """
        protocol = pickle.HIGHEST_PROTOCOL
        if self.compress:
            import gzip
            with gzip.open(filepath, "wb") as f:
                pickle.dump(checkpoint, f, protocol=protocol)
        else:
            with open(filepath, "wb") as f:
                pickle.dump(checkpoint, f, protocol=protocol)

    def load_checkpoint(self, filepath: Union[str, Path]) -> SamplingCheckpoint:
        """从文件加载检查点

        Args:
            filepath: 检查点文件路径

        Returns:
            加载的检查点对象

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式无效
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {path}")

        try:
            if path.suffix == ".json":
                return self._load_json(path)
            else:
                return self._load_pickle(path)
        except Exception as e:
            logger.error(f"加载检查点失败: {path} - {e}")
            raise ValueError(f"无法加载检查点 {path}: {e}")

    def _load_json(self, filepath: Path) -> SamplingCheckpoint:
        """从 JSON 加载"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SamplingCheckpoint.from_dict(data)

    def _load_pickle(self, filepath: Path) -> SamplingCheckpoint:
        """从 Pickle 加载"""
        if filepath.suffix == ".gz":
            import gzip
            with gzip.open(filepath, "rb") as f:
                return pickle.load(f)
        else:
            with open(filepath, "rb") as f:
                return pickle.load(f)

    def should_auto_save(self, current_step: int) -> bool:
        """判断是否应该自动保存

        Args:
            current_step: 当前步骤

        Returns:
            是否需要自动保存
        """
        if self.auto_save_interval <= 0:
            return False

        if current_step <= 0:
            return False

        steps_since_last = current_step - self._last_auto_save_step
        if steps_since_last >= self.auto_save_interval:
            self._last_auto_save_step = current_step
            return True
        return False

    def get_latest_checkpoint(self) -> Optional[SamplingCheckpoint]:
        """获取最新的检查点

        Returns:
            最新的检查点对象，如果没有则返回 None
        """
        if not self._checkpoints:
            # 尝试从磁盘加载
            self._scan_checkpoints()

        if not self._checkpoints:
            return None

        return max(self._checkpoints, key=lambda c: c.timestamp)

    def list_checkpoints(self) -> List[SamplingCheckpoint]:
        """列出所有可用的检查点

        Returns:
            按时间排序的检查点列表
        """
        if not self._checkpoints:
            self._scan_checkpoints()
        return sorted(self._checkpoints, key=lambda c: c.timestamp)

    def _scan_checkpoints(self) -> None:
        """扫描目录中的所有检查点文件"""
        extensions = [".pkl", ".pkl.gz", ".json"]
        for ext in extensions:
            for filepath in self.save_dir.glob(f"*{ext}"):
                try:
                    ckpt = self.load_checkpoint(filepath)
                    if ckpt not in self._checkpoints:
                        self._checkpoints.append(ckpt)
                except Exception as e:
                    logger.warning(f"跳过损坏的检查点 {filepath}: {e}")

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除指定的检查点

        Args:
            checkpoint_id: 要删除的检查点 ID

        Returns:
            是否成功删除
        """
        for ckpt in self._checkpoints:
            if ckpt.checkpoint_id == checkpoint_id:
                for ext in [".pkl", ".pkl.gz", ".json"]:
                    filepath = self.save_dir / f"{checkpoint_id}{ext}"
                    if filepath.exists():
                        filepath.unlink()
                        logger.info(f"已删除检查点: {checkpoint_id}")
                        self._checkpoints.remove(ckpt)
                        return True
        return False

    def _cleanup_if_needed(self) -> None:
        """根据清理策略执行清理"""
        policy = self.cleanup_policy

        if policy.strategy == policy.KEEP_ALL:
            return

        elif policy.strategy == policy.KEEP_LAST_N:
            while len(self._checkpoints) > policy.max_checkpoints:
                oldest = min(self._checkpoints, key=lambda c: c.timestamp)
                self.delete_checkpoint(oldest.checkpoint_id)

        elif policy.strategy == policy.KEEP_BY_TIME:
            now = time.time()
            max_age_sec = policy.max_age_hours * 3600
            for ckpt in self._checkpoints[:]:
                if now - ckpt.timestamp > max_age_sec:
                    self.delete_checkpoint(ckpt.checkpoint_id)

    def cleanup_all(self) -> int:
        """清除所有检查点

        Returns:
            删除的检查点数量
        """
        count = 0
        for ckpt in self._checkpoints[:]:
            deleted = self.delete_checkpoint(ckpt.checkpoint_id)
            if deleted:
                count += 1

        # 清除磁盘上所有剩余文件
        for ext in [".pkl", ".pkl.gz", ".json"]:
            for filepath in self.save_dir.glob(f"*{ext}"):
                filepath.unlink()
                count += 1

        # 清空内部列表
        self._checkpoints.clear()

        logger.info(f"已清除所有检查点 ({count} 个)")
        return count

    def get_storage_info(self) -> Dict[str, Any]:
        """获取存储信息统计

        Returns:
            包含使用情况的字典
        """
        total_size = sum(
            f.stat().st_size
            for f in self.save_dir.glob("*")
            if f.is_file()
        )
        return {
            "directory": str(self.save_dir),
            "total_checkpoints": len(self._checkpoints),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "cleanup_policy": self.cleanup_policy.strategy,
        }

    def reset(self) -> None:
        """重置管理器状态"""
        self._last_auto_save_step = -1
        self._checkpoint_count = 0
        self._checkpoints.clear()
        logger.info("CheckpointManager 已重置")
