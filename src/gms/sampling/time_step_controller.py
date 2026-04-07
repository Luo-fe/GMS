"""时间步长控制器 - 动态调整采样过程中的时间步长 Δt"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Callable, Dict, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AdaptationMode(Enum):
    """自适应步长调整模式"""

    FIXED = "fixed"
    GRADIENT_BASED = "gradient_based"
    CURVATURE_BASED = "curvature_based"
    HYBRID = "hybrid"


@dataclass
class StepHistory:
    """步长历史记录数据类

    存储每个时间步的详细信息，用于分析和调试。
    """

    step: int
    dt: float
    gradient_norm: Optional[float] = None
    curvature: Optional[float] = None
    adaptation_reason: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class TimeStepStats:
    """步长统计信息"""

    mean_dt: float
    std_dt: float
    min_dt: float
    max_dt: float
    total_adaptations: int
    adaptation_rate: float


class TimeStepController:
    """时间步长控制器

    动态调整采样过程中的时间步长 Δt，支持：
    - 固定步长模式
    - 基于梯度的自适应步长
    - 基于曲率的自适应步长
    - 混合模式（梯度 + 曲率）

    Attributes:
        initial_dt: 初始步长
        min_dt: 最小允许步长
        max_dt: 最大允许步长
        mode: 自适应模式
        history: 步长历史记录列表
    """

    def __init__(
        self,
        initial_dt: float = 0.01,
        min_dt: float = 1e-4,
        max_dt: float = 0.1,
        mode: AdaptationMode = AdaptationMode.FIXED,
        safety_factor: float = 0.9,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.8,
        gradient_threshold: float = 1.0,
        curvature_threshold: float = 10.0,
        max_history_size: int = 10000,
    ) -> None:
        """初始化时间步长控制器

        Args:
            initial_dt: 初始时间步长
            min_dt: 最小允许步长（防止步长过小）
            max_dt: 最大允许步长（保证数值稳定性）
            mode: 自适应模式选择
            safety_factor: 安全系数，用于缩放计算出的步长 (0, 1]
            increase_factor: 步长增大因子 (>1)
            decrease_factor: 步长减小因子 (<1)
            gradient_threshold: 梯度范数阈值，超过此值减小步长
            curvature_threshold: 曲率阈值，超过此值减小步长
            max_history_size: 历史记录最大长度
        """
        if not 0 < min_dt <= initial_dt <= max_dt:
            raise ValueError(
                f"步长约束不满足: 0 < {min_dt} <= {initial_dt} <= {max_dt}"
            )
        if not 0 < safety_factor <= 1:
            raise ValueError(f"safety_factor 必须在 (0, 1] 范围内，得到 {safety_factor}")
        if increase_factor <= 1:
            raise ValueError(f"increase_factor 必须 > 1，得到 {increase_factor}")
        if not 0 < decrease_factor < 1:
            raise ValueError(f"decrease_factor 必须在 (0, 1) 范围内，得到 {decrease_factor}")

        self.initial_dt = initial_dt
        self.current_dt = initial_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.mode = mode
        self.safety_factor = safety_factor
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.gradient_threshold = gradient_threshold
        self.curvature_threshold = curvature_threshold
        self.max_history_size = max_history_size

        self.history: List[StepHistory] = []
        self._step_count = 0
        self._adaptation_count = 0

        logger.info(
            f"初始化 TimeStepController: dt={initial_dt}, "
            f"mode={mode.value}, range=[{min_dt}, {max_dt}]"
        )

    def get_current_dt(self) -> float:
        """获取当前时间步长

        Returns:
            当前的时间步长值
        """
        return self.current_dt

    def adapt_step(
        self,
        gradient_norm: Optional[float] = None,
        curvature: Optional[float] = None,
        loss_change: Optional[float] = None,
    ) -> float:
        """根据当前状态调整时间步长

        根据选定的模式和输入信息动态调整步长。

        Args:
            gradient_norm: 当前梯度范数（可选）
            curvature: 当前曲率估计（可选）
            loss_change: 损失函数变化量（可选）

        Returns:
            调整后的新步长
        """
        old_dt = self.current_dt
        new_dt = old_dt
        reason: Optional[str] = None

        if self.mode == AdaptationMode.FIXED:
            new_dt = old_dt

        elif self.mode == AdaptationMode.GRADIENT_BASED:
            if gradient_norm is not None:
                new_dt, reason = self._adapt_by_gradient(gradient_norm)

        elif self.mode == AdaptationMode.CURVATURE_BASED:
            if curvature is not None:
                new_dt, reason = self._adapt_by_curvature(curvature)

        elif self.mode == AdaptationMode.HYBRID:
            new_dt, reason = self._adapt_hybrid(
                gradient_norm, curvature, loss_change
            )
        else:
            raise ValueError(f"未知的适应模式: {self.mode}")

        # 应用约束
        new_dt = np.clip(new_dt, self.min_dt, self.max_dt)

        # 记录历史
        history_entry = StepHistory(
            step=self._step_count,
            dt=new_dt,
            gradient_norm=gradient_norm,
            curvature=curvature,
            adaptation_reason=reason,
        )
        self.history.append(history_entry)

        # 限制历史大小
        if len(self.history) > self.max_history_size:
            self.history.pop(0)

        # 更新统计
        if abs(new_dt - old_dt) > 1e-10:
            self._adaptation_count += 1

        self.current_dt = new_dt
        self._step_count += 1

        logger.debug(
            f"步骤 {self._step_count}: dt {old_dt:.6f} -> {new_dt:.6f}"
            + (f" ({reason})" if reason else "")
        )

        return new_dt

    def _adapt_by_gradient(self, gradient_norm: float) -> Tuple[float, Optional[str]]:
        """基于梯度的步长调整

        当梯度较大时减小步长以保持稳定性，
        当梯度较小时增大步长以提高效率。

        Args:
            gradient_norm: 梯度范数

        Returns:
            (新步长, 调整原因)
        """
        if gradient_norm > self.gradient_threshold * 10:
            # 梯度非常大，大幅减小步长
            new_dt = self.current_dt * self.decrease_factor ** 2
            return new_dt, f"高梯度 ({gradient_norm:.4f})"
        elif gradient_norm > self.gradient_threshold:
            # 梯度较大，适度减小步长
            new_dt = self.current_dt * self.decrease_factor
            return new_dt, f"中等梯度 ({gradient_norm:.4f})"
        elif gradient_norm < self.gradient_threshold / 10:
            # 梯度很小，可以增大步长
            new_dt = self.current_dt * self.increase_factor
            return new_dt, f"低梯度 ({gradient_norm:.4f})"
        else:
            return self.current_dt, None

    def _adapt_by_curvature(self, curvature: float) -> Tuple[float, Optional[str]]:
        """基于曲率的步长调整

        高曲率表示函数变化剧烈，需要更小的步长。

        Args:
            curvature: 曲率估计值

        Returns:
            (新步长, 调整原因)
        """
        if curvature > self.curvature_threshold * 10:
            new_dt = self.current_dt * self.decrease_factor ** 2
            return new_dt, f"高曲率 ({curvature:.4f})"
        elif curvature > self.curvature_threshold:
            new_dt = self.current_dt * self.decrease_factor
            return new_dt, f"中等曲率 ({curvature:.4f})"
        elif curvature < self.curvature_threshold / 10:
            new_dt = self.current_dt * self.increase_factor
            return new_dt, f"低曲率 ({curvature:.4f})"
        else:
            return self.current_dt, None

    def _adapt_hybrid(
        self,
        gradient_norm: Optional[float],
        curvature: Optional[float],
        loss_change: Optional[float],
    ) -> Tuple[float, Optional[str]]:
        """混合模式的步长调整

        综合考虑梯度、曲率和损失变化来决定步长调整。

        Args:
            gradient_norm: 梯度范数
            curvature: 曲率
            loss_change: 损失变化

        Returns:
            (新步长, 调整原因)
        """
        factors: List[Tuple[str, float]] = []
        reasons: List[str] = []

        if gradient_norm is not None:
            if gradient_norm > self.gradient_threshold:
                factors.append(("grad", self.decrease_factor))
                reasons.append(f"梯度={gradient_norm:.3f}")
            elif gradient_norm < self.gradient_threshold / 5:
                factors.append(("grad", self.increase_factor))
                reasons.append(f"低梯度")

        if curvature is not None:
            if curvature > self.curvature_threshold:
                factors.append(("curve", self.decrease_factor))
                reasons.append(f"曲率={curvature:.3f}")
            elif curvature < self.curvature_threshold / 5:
                factors.append(("curve", self.increase_factor))
                reasons.append(f"低曲率")

        if loss_change is not None and abs(loss_change) < 1e-8:
            factors.append(("loss", self.increase_factor))
            reasons.append("损失收敛")
        elif loss_change is not None and loss_change > 0:
            factors.append(("loss", self.decrease_factor))
            reasons.append("损失增加")

        if not factors:
            return self.current_dt, None

        # 取所有因子的最小值（最保守的调整）
        min_factor = min(f for _, f in factors)
        new_dt = self.current_dt * min_factor * self.safety_factor

        return new_dt, ", ".join(reasons)

    def reset(self) -> None:
        """重置控制器到初始状态"""
        self.current_dt = self.initial_dt
        self.history.clear()
        self._step_count = 0
        self._adaptation_count = 0
        logger.info("TimeStepController 已重置")

    def get_statistics(self) -> TimeStepStats:
        """获取步长使用统计信息

        Returns:
            包含均值、标准差、最值的统计对象
        """
        if not self.history:
            return TimeStepStats(
                mean_dt=self.initial_dt,
                std_dt=0.0,
                min_dt=self.initial_dt,
                max_dt=self.initial_dt,
                total_adaptations=0,
                adaptation_rate=0.0,
            )

        dts = [h.dt for h in self.history]
        arr = np.array(dts)
        adaptation_rate = self._adaptation_count / max(len(self.history), 1)

        return TimeStepStats(
            mean_dt=float(np.mean(arr)),
            std_dt=float(np.std(arr)),
            min_dt=float(np.min(arr)),
            max_dt=float(np.max(arr)),
            total_adaptations=self._adaptation_count,
            adaptation_rate=adaptation_rate,
        )

    def get_dt_sequence(self) -> List[float]:
        """获取完整的步长序列

        Returns:
            所有历史步长的列表
        """
        return [h.dt for h in self.history]

    def get_adaptation_events(self) -> List[StepHistory]:
        """获取所有发生步长调整的事件

        Returns:
            发生过调整的步骤历史记录列表
        """
        events = []
        for i, h in enumerate(self.history):
            if h.adaptation_reason is not None:
                events.append(h)
        return events

    def set_mode(self, mode: AdaptationMode) -> None:
        """切换自适应模式

        Args:
            mode: 新的模式
        """
        old_mode = self.mode
        self.mode = mode
        logger.info(f"模式从 {old_mode.value} 切换为 {mode.value}")

    def export_state(self) -> Dict[str, Any]:
        """导出控制器状态（用于检查点保存）

        Returns:
            包含所有重要状态的字典
        """
        return {
            "current_dt": self.current_dt,
            "initial_dt": self.initial_dt,
            "min_dt": self.min_dt,
            "max_dt": self.max_dt,
            "mode": self.mode.value,
            "step_count": self._step_count,
            "adaptation_count": self._adaptation_count,
            "config": {
                "safety_factor": self.safety_factor,
                "increase_factor": self.increase_factor,
                "decrease_factor": self.decrease_factor,
                "gradient_threshold": self.gradient_threshold,
                "curvature_threshold": self.curvature_threshold,
            },
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "TimeStepController":
        """从状态字典恢复控制器

        Args:
            state: export_state() 返回的状态字典

        Returns:
            恢复后的控制器实例
        """
        config = state.get("config", {})
        controller = cls(
            initial_dt=state["initial_dt"],
            min_dt=state["min_dt"],
            max_dt=state["max_dt"],
            mode=AdaptationMode(state["mode"]),
            **config,
        )
        controller.current_dt = state["current_dt"]
        controller._step_count = state.get("step_count", 0)
        controller._adaptation_count = state.get("adaptation_count", 0)
        return controller

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"TimeStepController(dt={self.current_dt:.6f}, "
            f"mode={mode.value}, "
            f"range=[{self.min_dt}, {self.max_dt}], "
            f"steps={self._step_count})"
        )
