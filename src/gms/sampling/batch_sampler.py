"""批量高斯混合采样器 - 高效的双分量 GMM 批量采样

整合分量选择器和高斯采样器，实现高效的向量化批量采样流程。
支持分块处理和性能优化。
"""

from typing import Optional, Union, Tuple, Dict, Any
import logging
import time
import numpy as np
import torch

from .component_selector import ComponentSelector
from .gaussian_sampler import GaussianSampler

logger = logging.getLogger(__name__)


class BatchGaussianMixtureSampler:
    """双分量高斯混合模型批量采样器

    完整的 GMM 采样流程：
    1. 根据权重 w 进行分量选择（Bernoulli 分布）
    2. 并行从两个高斯分量 N(μ₁, σ₁²) 和 N(μ₂, σ₂²) 采样
    3. 根据选择掩码组合结果

    性能优化策略：
    - 完全向量化操作，避免 Python 循环
    - 并行生成两个分量的样本
    - 使用高级索引进行高效掩码操作
    - 支持分块处理以适应内存限制

    Attributes:
        weight: 分量2的权重
        mean1, std1: 分量1的参数 (μ₁, σ₁)
        mean2, std2: 分量2的参数 (μ₂, σ₂)
    """

    def __init__(
        self,
        weight: float = 0.5,
        mean1: float = 0.0,
        std1: float = 1.0,
        mean2: float = 3.0,
        std2: float = 0.5,
        method: str = "direct",
        device: Union[str, torch.device] = "cpu",
        deterministic_selection: bool = False,
    ) -> None:
        """初始化批量 GMM 采样器

        Args:
            weight: 分量2的混合权重 (0 < w < 1)
            mean1: 分量1的均值 μ₁
            std1: 分量1的标准差 σ₁ (> 0)
            mean2: 分量2的均值 μ₂
            std2: 分量2的标准差 σ₂ (> 0)
            method: 高斯采样方法 ('box_muller' 或 'direct')
            device: 计算设备
            deterministic_selection: 是否使用确定性分量选择

        Raises:
            ValueError: 如果参数不合法
        """
        # 初始化分量选择器
        self.component_selector = ComponentSelector(
            weight=weight,
            deterministic=deterministic_selection,
            device=device,
        )

        # 初始化两个高斯采样器
        self.sampler1 = GaussianSampler(
            mean=mean1,
            std=std1,
            method=method,
            device=device,
        )
        self.sampler2 = GaussianSampler(
            mean=mean2,
            std=std2,
            method=method,
            device=device,
        )

        self.method = method
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        logger.info(
            f"初始化 BatchGMM: w={weight:.4f}, "
            f"N({mean1:.4f}, {std1:.4f}²) + N({mean2:.4f}, {std2:.4f}²), "
            f"device={self.device}"
        )

    def sample(
        self,
        size: int = 1000,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """执行批量 GMM 采样

        优化的向量化采样流程：
        1. 一次性生成分量选择掩码
        2. 预分配结果张量
        3. 分别计算两个分量需要的样本数
        4. 并行从两个分量采样
        5. 使用高级索引填充结果

        Args:
            size: 总样本数量
            generator: PyTorch 随机数生成器

        Returns:
            形状为 (size,) 的 GMM 样本张量

        Example:
            >>> sampler = BatchGaussianMixtureSampler(weight=0.3, mean1=0, mean2=5)
            >>> samples = sampler.sample(10000)
            >>> # 约 30% 来自 N(0, 1)，70% 来自 N(5, 0.25)
        """
        # 步骤 1：批量生成分量选择掩码
        component_mask = self.component_selector.select(size, generator=generator)

        # 步骤 2：预分配结果张量
        samples = torch.empty(size, device=self.device)

        # 步骤 3 & 4：计算各分量样本数并并行采样
        mask_1 = (component_mask == 0)
        mask_2 = (component_mask == 1)

        count_1 = mask_1.sum().item()
        count_2 = mask_2.sum().item()

        # 从分量1和分量2分别采样（可完全并行）
        if count_1 > 0:
            samples_1 = self.sampler1.sample(int(count_1), generator=generator)
        else:
            samples_1 = torch.empty(0, device=self.device)

        if count_2 > 0:
            samples_2 = self.sampler2.sample(int(count_2), generator=generator)
        else:
            samples_2 = torch.empty(0, device=self.device)

        # 步骤 5：根据掩码组合结果（向量化索引操作）
        samples[mask_1] = samples_1
        samples[mask_2] = samples_2

        logger.debug(
            f"GMM 采样完成: total={size}, "
            f"分量1={int(count_1)}, 分量2={int(count_2)}"
        )

        return samples

    def sample_chunked(
        self,
        total_size: int,
        chunk_size: int = 10000,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """分块处理的大规模采样

        当总样本量很大时，使用分块处理以控制内存使用。
        每个 chunk 独立采样，最后拼接结果。

        内存优化说明：
        - 峰值内存 ≈ chunk_size * sizeof(float) 而非 total_size * sizeof(float)
        - 适用于 GPU 显存受限或系统内存有限的情况

        Args:
            total_size: 总样本数量
            chunk_size: 每块的样本数量（默认 10k）
            generator: 随机数生成器

        Returns:
            形状为 (total_size,) 的完整样本张量
        """
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        all_samples = []

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_size)
            current_chunk_size = end_idx - start_idx

            logger.debug(f"处理分块 {i+1}/{num_chunks}: size={current_chunk_size}")

            chunk_samples = self.sample(current_chunk_size, generator=generator)
            all_samples.append(chunk_samples)

        result = torch.cat(all_samples, dim=0)

        logger.info(
            f"分块采样完成: total={total_size}, chunks={num_chunks}, "
            f"chunk_size={chunk_size}"
        )

        return result

    def sample_with_components(
        self,
        size: int = 1000,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样并返回分量归属信息

        除了返回样本外，还返回每个样本来自哪个分量的信息。

        Args:
            size: 总样本数量
            generator: 随机数生成器

        Returns:
            (samples, component_labels) 元组：
            - samples: 形状为 (size,) 的样本张量
            - component_labels: 形状为 (size,) 的标签张量（0 或 1）
        """
        component_labels = self.component_selector.select(size, generator=generator)
        samples = self._sample_from_labels(component_labels, generator)

        return samples, component_labels

    def _sample_from_labels(
        self,
        labels: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """根据给定的标签向量进行采样

        内部方法，用于从已知的分量分配中采样。

        Args:
            labels: 分量标签张量（0 或 1）
            generator: 随机数生成器

        Returns:
            对应的高斯分布样本
        """
        size = labels.numel()
        samples = torch.empty(size, device=self.device)

        mask_1 = (labels == 0)
        mask_2 = (labels == 1)

        count_1 = mask_1.sum().item()
        count_2 = mask_2.sum().item()

        if count_1 > 0:
            samples[mask_1] = self.sampler1.sample(int(count_1), generator=generator)
        if count_2 > 0:
            samples[mask_2] = self.sampler2.sample(int(count_2), generator=generator)

        return samples

    def get_theoretical_moments(self) -> Dict[str, float]:
        """计算理论矩（均值、方差、偏度等）

        对于双分量 GMM：
        E[X] = (1-w)*μ₁ + w*μ₂
        Var[X] = (1-w)*(σ₁² + μ₁²) + w*(σ₂² + μ₂²) - E[X]²

        Returns:
            包含理论矩的字典
        """
        w = self.component_selector.weight
        m1, s1 = self.sampler1.mean, self.sampler1.std
        m2, s2 = self.sampler2.mean, self.sampler2.std

        # 一阶矩（均值）
        mean = (1 - w) * m1 + w * m2

        # 二阶中心矩（方差）
        variance = (
            (1 - w) * (s1 ** 2 + m1 ** 2) +
            w * (s2 ** 2 + m2 ** 2) -
            mean ** 2
        )

        moments = {
            "mean": mean,
            "variance": variance,
            "std": np.sqrt(max(variance, 0)),
            "weight": w,
            "component1_mean": m1,
            "component1_std": s1,
            "component2_mean": m2,
            "component2_std": s2,
        }

        return moments

    def benchmark_vs_naive(
        self,
        sample_size: int = 100000,
        num_runs: int = 10,
    ) -> Dict[str, Any]:
        """性能基准测试：对比向量化实现与朴素循环实现的加速比

        朴素实现：逐个样本循环采样
        向量化实现：批量并行采样

        Args:
            sample_size: 测试样本数量
            num_runs: 重复运行次数

        Returns:
            包含详细性能对比结果的字典
        """
        results = {}

        # 向量化实现计时
        times_vectorized = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.sample(sample_size)
            elapsed = time.perf_counter() - start
            times_vectorized.append(elapsed)

        results["vectorized"] = {
            "mean_time": float(np.mean(times_vectorized)),
            "std_time": float(np.std(times_vectorized)),
            "min_time": float(np.min(times_vectorized)),
            "max_time": float(np.max(times_vectorized)),
        }

        # 朴素循环实现计时
        times_naive = []
        for _ in range(num_runs):
            start = time.perf_counter()

            # 朴素实现：逐个采样
            naive_samples = torch.empty(sample_size, device=self.device)
            for i in range(sample_size):
                comp_choice = self.component_selector.select(1)
                if comp_choice.item() == 0:
                    naive_samples[i] = self.sampler1.sample(1)
                else:
                    naive_samples[i] = self.sampler2.sample(1)

            elapsed = time.perf_counter() - start
            times_naive.append(elapsed)

        results["naive"] = {
            "mean_time": float(np.mean(times_naive)),
            "std_time": float(np.std(times_naive)),
            "min_time": float(np.min(times_naive)),
            "max_time": float(np.max(times_naive)),
        }

        # 计算加速比
        speedup = results["naive"]["mean_time"] / max(results["vectorized"]["mean_time"], 1e-10)
        results["speedup_ratio"] = speedup

        logger.info(
            f"性能基准测试 (size={sample_size}, runs={num_runs}):\n"
            f"  朴素循环: {results['naive']['mean_time']:.4f}s\n"
            f"  向量化:   {results['vectorized']['mean_time']:.4f}s\n"
            f"  加速比:   {speedup:.2f}x"
        )

        return results

    def set_parameters(
        self,
        weight: Optional[float] = None,
        mean1: Optional[float] = None,
        std1: Optional[float] = None,
        mean2: Optional[float] = None,
        std2: Optional[float] = None,
    ) -> None:
        """更新 GMM 参数

        Args:
            weight: 新的混合权重
            mean1, std1: 分量1的新参数
            mean2, std2: 分量2的新参数
        """
        if weight is not None:
            self.component_selector.set_weight(weight)

        if mean1 is not None or std1 is not None:
            self.sampler1.set_parameters(mean=mean1, std=std1)

        if mean2 is not None or std2 is not None:
            self.sampler2.set_parameters(mean=mean2, std=std2)

        logger.info("GMM 参数已更新")

    def to(self, device: Union[str, torch.device]) -> "BatchGaussianMixtureSampler":
        """移动到指定设备

        Args:
            device: 目标设备

        Returns:
            self（支持链式调用）
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.component_selector.to(device)
        self.sampler1.to(device)
        self.sampler2.to(device)
        return self

    def __repr__(self) -> str:
        return (
            f"BatchGMM(w={self.component_selector.weight:.4f}, "
            f"N({self.sampler1.mean:.2f}, {self.sampler1.std:.2f}) + "
            f"N({self.sampler2.mean:.2f}, {self.sampler2.std:.2f}), "
            f"device={self.device})"
        )
