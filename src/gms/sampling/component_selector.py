"""分量选择器 - 基于权重的高斯混合模型分量选择

实现基于 Bernoulli 分布的分量选择逻辑，支持随机和确定性模式。
"""

from typing import Optional, Union
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ComponentSelector:
    """高斯混合模型分量选择器

    基于权重参数 w，使用 Bernoulli 分布选择分量1或分量2：
    - P(选择分量2) = w
    - P(选择分量1) = 1 - w

    支持单样本、批量采样以及确定性（固定比例）分配。

    Attributes:
        weight: 分量2的权重 (0 < w < 1)
        deterministic: 是否使用确定性模式
    """

    def __init__(
        self,
        weight: float = 0.5,
        deterministic: bool = False,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """初始化分量选择器

        Args:
            weight: 分量2的权重，必须在 (0, 1) 范围内
            deterministic: 是否使用确定性模式（固定比例分配）
            device: 计算设备 ('cpu', 'cuda' 或 torch.device）

        Raises:
            ValueError: 如果 weight 不在 (0, 1) 范围内
        """
        if not 0 < weight < 1:
            raise ValueError(f"weight 必须在 (0, 1) 范围内，得到 {weight}")

        self.weight = weight
        self.deterministic = deterministic
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        logger.info(
            f"初始化 ComponentSelector: w={weight:.4f}, "
            f"deterministic={deterministic}, device={self.device}"
        )

    def select(self, size: int = 1, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """执行分量选择

        根据权重进行 Bernoulli 采样，返回分量索引。
        返回值：0 表示选择分量1，1 表示选择分量2。

        Args:
            size: 采样数量
            generator: PyTorch 随机数生成器（用于可复现性）

        Returns:
            形状为 (size,) 的张量，包含 0 或 1 的分量选择结果

        Example:
            >>> selector = ComponentSelector(weight=0.3)
            >>> choices = selector.select(1000)
            >>> # choices 中大约 30% 为 1（分量2），70% 为 0（分量1）
        """
        if self.deterministic:
            return self._deterministic_select(size)

        random_tensor = torch.rand(
            size,
            device=self.device,
            generator=generator,
        )
        choices = (random_tensor < self.weight).long()

        logger.debug(
            f"分量选择: size={size}, "
            f"实际分量2比例={choices.float().mean():.4f} "
            f"(理论={self.weight:.4f})"
        )

        return choices

    def _deterministic_select(self, size: int) -> torch.Tensor:
        """确定性分量选择（固定比例分配）

        按照精确的比例分配分量，确保分量2的数量为 round(size * weight)。

        Args:
            size: 采样数量

        Returns:
            确定性的分量选择张量
        """
        num_component_2 = int(round(size * self.weight))
        num_component_1 = size - num_component_2

        choices = torch.zeros(size, dtype=torch.long, device=self.device)
        if num_component_2 > 0:
            choices[:num_component_2] = 1

        # 打乱顺序以避免位置偏差
        perm = torch.randperm(size, device=self.device)
        choices = choices[perm]

        logger.debug(
            f"确定性分量选择: size={size}, "
            f"分量1数量={num_component_1}, 分量2数量={num_component_2}"
        )

        return choices

    def get_selection_stats(self, choices: torch.Tensor) -> dict:
        """计算分量选择的统计信息

        Args:
            choices: 分量选择结果张量

        Returns:
            包含统计信息的字典
        """
        total = choices.numel()
        count_1 = (choices == 0).sum().item()
        count_2 = (choices == 1).sum().item()

        stats = {
            "total_samples": total,
            "component_1_count": int(count_1),
            "component_2_count": int(count_2),
            "component_1_ratio": float(count_1 / max(total, 1)),
            "component_2_ratio": float(count_2 / max(total, 1)),
            "theoretical_weight": self.weight,
            "error_abs": abs(float(count_2 / max(total, 1)) - self.weight),
        }

        logger.info(
            f"分量选择统计: 分量1占比={stats['component_1_ratio']:.4f}, "
            f"分量2占比={stats['component_2_ratio']:.4f}, "
            f"绝对误差={stats['error_abs']:.4f}"
        )

        return stats

    def validate_long_term_ratio(
        self,
        num_samples: int = 100000,
        tolerance: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> bool:
        """验证长期运行的分量选择比例

        通过大量采样验证实际比例是否接近理论权重。

        Args:
            num_samples: 采样数量（默认 10 万）
            tolerance: 允许的最大偏差（默认 2%）
            generator: 随机数生成器

        Returns:
            如果实际比例在容差范围内返回 True
        """
        choices = self.select(num_samples, generator=generator)
        stats = self.get_selection_stats(choices)

        is_valid = stats["error_abs"] <= tolerance

        logger.info(
            f"长期比例验证: {'通过' if is_valid else '未通过'}, "
            f"误差={stats['error_abs']:.4f}, 容差={tolerance}"
        )

        return is_valid

    def set_weight(self, new_weight: float) -> None:
        """更新权重参数

        Args:
            new_weight: 新的权重值，必须在 (0, 1) 范围内

        Raises:
            ValueError: 如果 new_weight 不在 (0, 1) 范围内
        """
        if not 0 < new_weight < 1:
            raise ValueError(f"new_weight 必须在 (0, 1) 范围内，得到 {new_weight}")

        old_weight = self.weight
        self.weight = new_weight
        logger.info(f"权重更新: {old_weight:.4f} -> {new_weight:.4f}")

    def to(self, device: Union[str, torch.device]) -> "ComponentSelector":
        """移动到指定设备

        Args:
            device: 目标设备

        Returns:
            self（支持链式调用）
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        return self

    def __repr__(self) -> str:
        return (
            f"ComponentSelector(w={self.weight:.4f}, "
            f"deterministic={self.deterministic}, "
            f"device={self.device})"
        )
