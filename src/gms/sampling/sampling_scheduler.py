"""采样调度器 - 控制扩散/采样过程中的噪声调度策略"""

from abc import ABC, abstractmethod
from typing import List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class BaseScheduler(ABC):
    """采样调度器抽象基类

    定义噪声调度的标准接口，支持不同的噪声调度策略。
    调度器用于控制采样过程中每一步的噪声水平或参数值。

    Attributes:
        start_value: 调度起始值
        end_value: 调度结束值
        name: 调度器名称
    """

    def __init__(
        self,
        start_value: float = 0.0,
        end_value: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        """初始化调度器

        Args:
            start_value: 调度起始值（t=0 时的值）
            end_value: 调度结束值（t=T 时的值）
            name: 调度器名称，用于日志和标识
        """
        self.start_value = start_value
        self.end_value = end_value
        self.name = name or self.__class__.__name__
        logger.debug(
            f"初始化 {self.name}: start={start_value}, end={end_value}"
        )

    @abstractmethod
    def get_schedule(self, total_steps: int) -> List[float]:
        """生成完整的调度序列

        Args:
            total_steps: 总时间步数

        Returns:
            长度为 total_steps 的浮点数列表，表示每个时间步的调度值

        Raises:
            ValueError: 如果 total_steps <= 0
        """
        pass

    def get_value(self, step: int, total_steps: int) -> float:
        """获取单个时间步的调度值

        Args:
            step: 当前步骤索引 (0-based)
            total_steps: 总步数

        Returns:
            该步骤的调度值

        Raises:
            ValueError: 如果 step 超出范围或 total_steps <= 0
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps 必须为正整数，得到 {total_steps}")
        if step < 0 or step >= total_steps:
            raise ValueError(
                f"step 必须在 [0, {total_steps}) 范围内，得到 {step}"
            )
        schedule = self.get_schedule(total_steps)
        return schedule[step]

    def __repr__(self) -> str:
        return (
            f"{self.name}(start={self.start_value}, "
            f"end={self.end_value})"
        )


class LinearScheduler(BaseScheduler):
    """线性调度器

    实现线性插值的噪声调度：
    β(t) = start + (end - start) * (t / T)

    适用于需要均匀变化的场景。
    """

    def __init__(
        self,
        start_value: float = 1e-4,
        end_value: float = 0.02,
        name: Optional[str] = None,
    ) -> None:
        """初始化线性调度器

        Args:
            start_value: 起始 β 值（默认 1e-4，适合扩散模型）
            end_value: 结束 β 值（默认 0.02）
            name: 调度器名称
        """
        super().__init__(start_value, end_value, name or "LinearScheduler")

    def get_schedule(self, total_steps: int) -> List[float]:
        """生成线性调度序列

        Args:
            total_steps: 总时间步数

        Returns:
            线性递增的调度值列表

        Raises:
            ValueError: 如果 total_steps <= 0
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps 必须为正整数，得到 {total_steps}")

        schedule = np.linspace(self.start_value, self.end_value, total_steps)
        logger.debug(f"{self.name}: 生成 {total_steps} 步线性调度")
        return schedule.tolist()


class CosineScheduler(BaseScheduler):
    """余弦调度器

    实现基于余弦函数的平滑调度：
    β(t) = start + (end - start) * 0.5 * (1 - cos(π * t / T))

    相比线性调度，余弦调度在开始和结束时变化更平缓，
    中间变化更快，通常能产生更好的采样质量。

    Reference:
        Nichol & Dhariwal, "Improved Denoising Diffusion Probabilistic Models"
    """

    def __init__(
        self,
        start_value: float = 1e-4,
        end_value: float = 0.02,
        s: float = 0.008,
        name: Optional[str] = None,
    ) -> None:
        """初始化余弦调度器

        Args:
            start_value: 起始 β 值
            end_value: 结束 β 值
            s: 余弦偏移量，控制曲线形状（默认 0.008）
            name: 调度器名称
        """
        super().__init__(start_value, end_value, name or "CosineScheduler")
        self.s = s

    def get_schedule(self, total_steps: int) -> List[float]:
        """生成余弦调度序列

        使用改进的余弦调度公式，产生更平滑的过渡。

        Args:
            total_steps: 总时间步数

        Returns:
            余弦调度的调度值列表

        Raises:
            ValueError: 如果 total_steps <= 0
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps 必须为正整数，得到 {total_steps}")

        steps = np.arange(total_steps + 1, dtype=np.float64)
        t_normalized = steps / total_steps

        # 改进的余弦调度公式
        alpha_bar = np.cos((t_normalized + self.s) / (1 + self.s) * np.pi / 2) ** 2
        beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
        beta = np.clip(beta, self.start_value, self.end_value)

        logger.debug(f"{self.name}: 生成 {total_steps} 步余弦调度")
        return beta.tolist()


class ConstantScheduler(BaseScheduler):
    """常数调度器

    返回固定值的调度器，用于测试或特殊场景。
    """

    def __init__(self, value: float = 0.01, name: Optional[str] = None) -> None:
        """初始化常数调度器

        Args:
            value: 固定调度值
            name: 调度器名称
        """
        super().__init__(value, value, name or "ConstantScheduler")

    def get_schedule(self, total_steps: int) -> List[float]:
        """生成常数调度序列

        Args:
            total_steps: 总时间步数

        Returns:
            全部为相同值的列表

        Raises:
            ValueError: 如果 total_steps <= 0
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps 必须为正整数，得到 {total_steps}")

        logger.debug(f"{self.name}: 生成 {total_steps} 步常数调度")
        return [self.start_value] * total_steps


class SqrtScheduler(BaseScheduler):
    """平方根调度器

    实现平方根插值的调度：
    β(t) = start + (end - start) * sqrt(t / T)

    在开始时变化较快，后期趋于平缓。
    """

    def __init__(
        self,
        start_value: float = 1e-4,
        end_value: float = 0.02,
        name: Optional[str] = None,
    ) -> None:
        """初始化平方根调度器

        Args:
            start_value: 起始值
            end_value: 结束值
            name: 调度器名称
        """
        super().__init__(start_value, end_value, name or "SqrtScheduler")

    def get_schedule(self, total_steps: int) -> List[float]:
        """生成平方根调度序列

        Args:
            total_steps: 总时间步数

        Returns:
            平方根增长的调度值列表

        Raises:
            ValueError: 如果 total_steps <= 0
        """
        if total_steps <= 0:
            raise ValueError(f"total_steps 必须为正整数，得到 {total_steps}")

        t_normalized = np.linspace(0, 1, total_steps)
        schedule = self.start_value + (self.end_value - self.start_value) * np.sqrt(t_normalized)

        logger.debug(f"{self.name}: 生成 {total_steps} 步平方根调度")
        return schedule.tolist()
