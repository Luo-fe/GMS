"""进度监控器 - 实时监控采样进度并提供回调接口"""

from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
import logging
import time
from datetime import timedelta

logger = logging.getLogger(__name__)


class SamplingEventType(Enum):
    """采样事件类型"""

    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    SAMPLING_START = "sampling_start"
    SAMPLING_COMPLETE = "sampling_complete"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class SamplingProgress:
    """采样进度信息数据类

    存储采样过程的详细进度信息。

    Attributes:
        current_step: 当前步骤索引
        total_steps: 总步数
        progress_percent: 进度百分比 (0-100)
        elapsed_time: 已用时间（秒）
        estimated_remaining_time: 预计剩余时间（秒）
        steps_per_second: 每秒步数
        custom_metrics: 自定义指标字典
    """

    current_step: int
    total_steps: int
    progress_percent: float
    elapsed_time: float
    estimated_remaining_time: float
    steps_per_second: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_str(self) -> str:
        """格式化的已用时间字符串"""
        return str(timedelta(seconds=int(self.elapsed_time)))

    @property
    def eta_str(self) -> str:
        """格式化的预计剩余时间字符串"""
        if self.estimated_remaining_time < 0:
            return "未知"
        return str(timedelta(seconds=int(self.estimated_remaining_time)))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percent": self.progress_percent,
            "elapsed_time": self.elapsed_time,
            "estimated_remaining_time": self.estimated_remaining_time,
            "steps_per_second": self.steps_per_second,
            "custom_metrics": self.custom_metrics,
        }


class ProgressCallback:
    """进度回调函数类型别名"""

    def __call__(
        self,
        progress: SamplingProgress,
        event_type: SamplingEventType,
        **kwargs
    ) -> None:
        """回调函数签名

        Args:
            progress: 当前进度信息
            event_type: 事件类型
            **kwargs: 额外的事件数据
        """
        pass


class ProgressMonitor:
    """采样进度监控器

    提供实时进度监控、ETA 估计和回调机制。
    支持与 tqdm 进度条集成。

    Attributes:
        total_steps: 总步数
        enable_tqdm: 是否启用 tqdm 进度条
        tqdm_desc: tqdm 描述文本
    """

    def __init__(
        self,
        total_steps: int,
        enable_tqdm: bool = True,
        tqdm_desc: str = "采样进度",
        log_interval: int = 10,
        eta_smoothing: float = 0.1,
    ) -> None:
        """初始化进度监控器

        Args:
            total_steps: 总采样步数
            enable_tqdm: 是否显示 tqdm 进度条
            tqdm_desc: tqdm 进度条描述
            log_interval: 日志输出间隔（步数）
            eta_smoothing: ETA 平滑系数 (0-1)，越小越平滑
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps 必须为正整数，得到 {total_steps}")
        if not 0 <= eta_smoothing <= 1:
            raise ValueError(f"eta_smoothing 必须在 [0, 1] 范围内，得到 {eta_smoothing}")

        self.total_steps = total_steps
        self.enable_tqdm = enable_tqdm
        self.tqdm_desc = tqdm_desc
        self.log_interval = log_interval
        self.eta_smoothing = eta_smoothing

        # 内部状态
        self._current_step: int = 0
        self._start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._step_times: List[float] = []
        self._is_running: bool = False
        self._callbacks: Dict[SamplingEventType, List[ProgressCallback]] = {}

        # tqdm 进度条（延迟初始化）
        self._tqdm: Optional[Any] = None

        # 自定义指标
        self._custom_metrics: Dict[str, Any] = {}

        logger.debug(
            f"初始化 ProgressMonitor: total={total_steps}, "
            f"tqdm={enable_tqdm}"
        )

    def register_callback(
        self,
        event_type: SamplingEventType,
        callback: ProgressCallback,
    ) -> None:
        """注册事件回调

        Args:
            event_type: 要监听的事件类型
            callback: 回调函数
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
        logger.debug(f"注册回调: {event_type.value}")

    def unregister_callback(
        self,
        event_type: SamplingEventType,
        callback: ProgressCallback,
    ) -> bool:
        """取消注册回调

        Args:
            event_type: 事件类型
            callback: 要移除的回调函数

        Returns:
            是否成功移除
        """
        if event_type in self._callbacks:
            try:
                self._callbacks[event_type].remove(callback)
                logger.debug(f"取消回调: {event_type.value}")
                return True
            except ValueError:
                pass
        return False

    def start(self) -> None:
        """开始监控"""
        if self._is_running:
            logger.warning("监控器已在运行中")
            return

        self._start_time = time.time()
        self._last_step_time = self._start_time
        self._is_running = True

        # 初始化 tqdm
        if self.enable_tqdm:
            try:
                from tqdm import tqdm

                self._tqdm = tqdm(
                    total=self.total_steps,
                    desc=self.tqdm_desc,
                    unit="step",
                )
            except ImportError:
                logger.warning("tqdm 未安装，禁用进度条")
                self.enable_tqdm = False

        self._emit_event(SamplingEventType.SAMPLING_START)
        logger.info(f"开始采样监控 (总步数: {self.total_steps})")

    def on_step_start(self, step: int) -> None:
        """步骤开始时的回调

        Args:
            step: 当前步骤索引
        """
        self._current_step = step
        self._emit_event(SamplingEventType.STEP_START)

    def on_step_complete(
        self,
        step: int,
        **metrics
    ) -> None:
        """步骤完成时的回调

        Args:
            step: 当前步骤索引
            **metrics: 额外的指标数据
        """
        now = time.time()
        if self._last_step_time is not None:
            step_duration = now - self._last_step_time
            self._step_times.append(step_duration)
            # 限制历史长度
            if len(self._step_times) > 100:
                self._step_times.pop(0)

        self._last_step_time = now
        self._current_step = step + 1

        # 更新自定义指标
        self._custom_metrics.update(metrics)

        # 更新 tqdm
        if self._tqdm is not None:
            self._tqdm.update(1)
            if metrics:
                self._tqdm.set_postfix(metrics)

        progress = self._get_progress()

        # 定期日志
        if step % self.log_interval == 0:
            logger.info(
                f"步骤 {step}/{self.total_steps} "
                f"({progress.progress_percent:.1f}%) "
                f"ETA: {progress.eta_str} "
                f"速度: {progress.steps_per_second:.2f} 步/秒"
            )

        self._emit_event(SamplingEventType.STEP_COMPLETE)

    def on_error(self, step: int, error: Exception) -> None:
        """错误回调

        Args:
            step: 出错的步骤
            error: 异常对象
        """
        self._emit_event(
            SamplingEventType.ERROR,
            step=step,
            error=str(error),
        )
        logger.error(f"步骤 {step} 发生错误: {error}")

    def on_warning(self, step: int, message: str) -> None:
        """警告回调

        Args:
            step: 步骤
            message: 警告信息
        """
        self._emit_event(
            SamplingEventType.WARNING,
            step=step,
            message=message,
        )
        logger.warning(f"步骤 {step}: {message}")

    def complete(self) -> None:
        """完成监控"""
        if not self._is_running:
            return

        self._is_running = False

        # 关闭 tqdm
        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None

        self._emit_event(SamplingEventType.SAMPLING_COMPLETE)

        progress = self._get_progress()
        logger.info(
            f"采样完成: {progress.elapsed_str}, "
            f"平均速度: {progress.steps_per_second:.2f} 步/秒"
        )

    def _get_progress(self) -> SamplingProgress:
        """计算当前进度

        Returns:
            进度信息对象
        """
        if self._start_time is None:
            elapsed = 0.0
        else:
            elapsed = time.time() - self._start_time

        progress_percent = (
            min(self._current_step / self.total_steps, 1.0) * 100
        )

        # 计算速度
        if elapsed > 0 and self._current_step > 0:
            steps_per_second = self._current_step / elapsed
        else:
            steps_per_second = 0.0

        # 计算 ETA
        if steps_per_second > 0:
            remaining_steps = self.total_steps - self._current_step
            eta = remaining_steps / steps_per_second
        else:
            eta = -1.0

        return SamplingProgress(
            current_step=self._current_step,
            total_steps=self.total_steps,
            progress_percent=progress_percent,
            elapsed_time=elapsed,
            estimated_remaining_time=eta,
            steps_per_second=steps_per_second,
            custom_metrics=self._custom_metrics.copy(),
        )

    def _emit_event(
        self,
        event_type: SamplingEventType,
        **kwargs
    ) -> None:
        """触发事件

        Args:
            event_type: 事件类型
            **kwargs: 事件数据
        """
        progress = self._get_progress()
        callbacks = self._callbacks.get(event_type, [])

        for callback in callbacks:
            try:
                callback(progress, event_type, **kwargs)
            except Exception as e:
                logger.error(f"回调函数执行失败 ({event_type.value}): {e}")

    def update_custom_metric(self, key: str, value: Any) -> None:
        """更新自定义指标

        Args:
            key: 指标名称
            value: 指标值
        """
        self._custom_metrics[key] = value

    def get_custom_metrics(self) -> Dict[str, Any]:
        """获取所有自定义指标

        Returns:
            指标字典
        """
        return self._custom_metrics.copy()

    def reset(self) -> None:
        """重置监控器状态"""
        self._current_step = 0
        self._start_time = None
        self._last_step_time = None
        self._step_times.clear()
        self._is_running = False
        self._custom_metrics.clear()

        if self._tqdm is not None:
            self._tqdm.close()
            self._tqdm = None

        logger.debug("ProgressMonitor 已重置")

    @property
    def is_running(self) -> bool:
        """是否正在运行"""
        return self._is_running

    @property
    def progress(self) -> SamplingProgress:
        """获取当前进度（只读）"""
        return self._get_progress()


class TqdmProgressMonitor(ProgressMonitor):
    """带 tqdm 集成的进度监控器

    提供更丰富的 tqdm 集成功能。
    """

    def __init__(
        self,
        total_steps: int,
        tqdm_desc: str = "采样进度",
        tqdm_postfix: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> None:
        """初始化 tqdm 进度监控器

        Args:
            total_steps: 总步数
            tqdm_desc: tqdm 描述
            tqdm_postfix: tqdm 后缀指标
            **kwargs: 传递给 ProgressMonitor 的参数
        """
        super().__init__(total_steps, enable_tqdm=True, tqdm_desc=tqdm_desc, **kwargs)
        self.tqdm_postfix = tqdm_postfix or {}

    def start(self) -> None:
        """开始监控（初始化 tqdm）"""
        super().start()
        if self._tqdm is not None and self.tqdm_postfix:
            self._tqdm.set_postfix(self.tqdm_postfix)

    def update_tqdm_postfix(self, **metrics) -> None:
        """更新 tqdm 后缀指标

        Args:
            **metrics: 要更新的指标
        """
        self.tqdm_postfix.update(metrics)
        if self._tqdm is not None:
            self._tqdm.set_postfix(self.tqdm_postfix)
