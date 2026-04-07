"""高斯采样器 - 基于 Box-Muller 变换的高斯分布采样

实现高效的单分量高斯分布采样，支持 Box-Muller 变换和直接采样两种方法。
"""

from typing import Optional, Union, Tuple
import logging
import time
import numpy as np
import torch

logger = logging.getLogger(__name__)

# 数值稳定性常量
MIN_STD = 1e-8  # 最小允许的标准差


class GaussianSampler:
    """单分量高斯分布采样器

    使用 Box-Muller 变换或 PyTorch 内置方法从正态分布 N(μ, σ²) 中采样。

    Box-Muller 变换原理：
    给定两个独立的均匀分布 U1, U2 ~ Uniform(0, 1)，
    可以通过以下变换得到两个独立的标准正态分布样本：
        Z0 = sqrt(-2 * ln(U1)) * cos(2π * U2)
        Z1 = sqrt(-2 * ln(U1)) * sin(2π * U2)

    然后通过线性变换 Z = μ + σ * Z0 得到 N(μ, σ²) 的样本。

    Attributes:
        mean: 均值 μ
        std: 标准差 σ
        method: 采样方法 ('box_muller' 或 'direct')
    """

    def __init__(
        self,
        mean: float = 0.0,
        std: float = 1.0,
        method: str = "box_muller",
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """初始化高斯采样器

        Args:
            mean: 高斯分布的均值 μ
            std: 高斯分布的标准差 σ（必须 > 0）
            method: 采样方法：
                - 'box_muller': 使用 Box-Muller 变换（教学用途）
                - 'direct': 使用 torch.randn（性能更优）
            device: 计算设备 ('cpu', 'cuda' 或 torch.device)

        Raises:
            ValueError: 如果 std <= 0 或 method 不支持
        """
        if std <= 0:
            raise ValueError(f"std 必须为正数，得到 {std}")

        valid_methods = ["box_muller", "direct"]
        if method not in valid_methods:
            raise ValueError(f"method 必须是 {valid_methods} 之一，得到 {method}")

        self.mean = mean
        self.std = max(std, MIN_STD)  # 数值稳定性处理
        self.method = method
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        logger.info(
            f"初始化 GaussianSampler: μ={mean:.4f}, σ={std:.4f}, "
            f"method={method}, device={self.device}"
        )

    def sample(
        self,
        size: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """生成高斯分布样本

        根据配置的方法生成服从 N(mean, std²) 的样本。

        Args:
            size: 样本数量
            generator: PyTorch 随机数生成器（用于可复现性）

        Returns:
            形状为 (size,) 的张量，包含高斯分布样本

        Example:
            >>> sampler = GaussianSampler(mean=5.0, std=2.0)
            >>> samples = sampler.sample(1000)
            >>> # samples 近似服从 N(5, 4) 分布
        """
        if self.method == "box_muller":
            return self._sample_box_muller(size, generator)
        else:
            return self._sample_direct(size, generator)

    def _sample_box_muller(
        self,
        size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """使用 Box-Muller 变换进行采样（向量化实现）

        性能优化说明：
        - 完全向量化实现，避免 Python 循环
        - 利用 PyTorch 的广播机制
        - 对于奇数个样本，多生成一个并丢弃

        Args:
            size: 需要的样本数量
            generator: 随机数生成器

        Returns:
            高斯分布样本张量
        """
        # 计算需要的均匀分布对数（Box-Muller 每次生成 2 个样本）
        num_pairs = (size + 1) // 2

        # 生成均匀分布 U1, U2 ~ Uniform(0, 1)
        u1 = torch.rand(
            num_pairs,
            device=self.device,
            generator=generator,
        )
        u2 = torch.rand(
            num_pairs,
            device=self.device,
            generator=generator,
        )

        # 数值稳定性：避免 log(0)
        u1 = torch.clamp(u1, min=1e-10, max=1.0 - 1e-10)

        # Box-Muller 变换
        # 使用极坐标形式的 Box-Muller（数值更稳定）
        r = torch.sqrt(-2.0 * torch.log(u1))
        theta = 2.0 * np.pi * u2

        z0 = r * torch.cos(theta)
        z1 = r * torch.sin(theta)

        # 交替合并结果
        samples = torch.empty(2 * num_pairs, device=self.device)
        samples[0::2] = z0
        samples[1::2] = z1

        # 截取所需数量
        samples = samples[:size]

        # 线性变换到 N(μ, σ²)
        samples = self.mean + self.std * samples

        logger.debug(
            f"Box-Muller 采样: size={size}, "
            f"实际均值={samples.mean():.4f} (理论={self.mean:.4f}), "
            f"实际标准差={samples.std():.4f} (理论={self.std:.4f})"
        )

        return samples

    def _sample_direct(
        self,
        size: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """使用 PyTorch 内置方法直接采样

        直接调用 torch.randn 生成标准正态分布，然后进行线性变换。
        这种方法通常比 Box-Muller 更快，因为使用了优化的底层实现。

        性能优势：
        - 使用 PyTorch 的高度优化实现
        - 支持 GPU 加速
        - 避免了三角函数计算的开销

        Args:
            size: 样本数量
            generator: 随机数生成器

        Returns:
            高斯分布样本张量
        """
        standard_normal = torch.randn(
            size,
            device=self.device,
            generator=generator,
        )
        samples = self.mean + self.std * standard_normal

        logger.debug(
            f"直接采样: size={size}, "
            f"实际均值={samples.mean():.4f} (理论={self.mean:.4f}), "
            f"实际标准差={samples.std():.4f} (理论={self.std:.4f})"
        )

        return samples

    def sample_batch(
        self,
        batch_size: int,
        num_batches: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """批量采样（分块生成）

        用于内存受限场景下的分块采样。

        Args:
            batch_size: 每批次的样本数量
            num_batches: 批次数量
            generator: 随机数生成器

        Returns:
            形状为 (batch_size * num_batches,) 的样本张量
        """
        total_size = batch_size * num_batches
        all_samples = torch.empty(total_size, device=self.device)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            all_samples[start_idx:end_idx] = self.sample(batch_size, generator)

        logger.debug(
            f"批量采样完成: total_size={total_size}, "
            f"batch_size={batch_size}, num_batches={num_batches}"
        )

        return all_samples

    def get_sample_statistics(self, samples: torch.Tensor) -> dict:
        """计算样本统计信息

        Args:
            samples: 样本张量

        Returns:
            包含均值、方差、偏度等统计量的字典
        """
        samples_np = samples.detach().cpu().numpy()

        from scipy.stats import skew, kurtosis

        stats = {
            "count": len(samples),
            "mean": float(samples.mean()),
            "std": float(samples.std()),
            "variance": float(samples.var()),
            "min": float(samples.min()),
            "max": float(samples.max()),
            "median": float(samples.median()),
            "skewness": float(skew(samples_np)),
            "kurtosis": float(kurtosis(samples_np)),
            "theoretical_mean": self.mean,
            "theoretical_std": self.std,
            "mean_error": abs(float(samples.mean()) - self.mean),
            "std_error": abs(float(samples.std()) - self.std),
        }

        logger.info(
            f"样本统计: 均值误差={stats['mean_error']:.4f}, "
            f"标准差误差={stats['std_error']:.4f}"
        )

        return stats

    def benchmark_methods(
        self,
        sample_size: int = 100000,
        num_runs: int = 10,
    ) -> dict:
        """性能基准测试：对比不同采样方法的性能

        Args:
            sample_size: 每次采样的样本数量
            num_runs: 重复运行次数

        Returns:
            包含各方法平均耗时的字典
        """
        results = {}

        # 测试 Box-Muller 方法
        box_muller_sampler = GaussianSampler(
            mean=self.mean,
            std=self.std,
            method="box_muller",
            device=self.device,
        )

        times_bm = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = box_muller_sampler.sample(sample_size)
            elapsed = time.perf_counter() - start
            times_bm.append(elapsed)

        results["box_muller"] = {
            "mean_time": float(np.mean(times_bm)),
            "std_time": float(np.std(times_bm)),
            "min_time": float(np.min(times_bm)),
            "max_time": float(np.max(times_bm)),
        }

        # 测试直接方法
        direct_sampler = GaussianSampler(
            mean=self.mean,
            std=self.std,
            method="direct",
            device=self.device,
        )

        times_direct = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = direct_sampler.sample(sample_size)
            elapsed = time.perf_counter() - start
            times_direct.append(elapsed)

        results["direct"] = {
            "mean_time": float(np.mean(times_direct)),
            "std_time": float(np.std(times_direct)),
            "min_time": float(np.min(times_direct)),
            "max_time": float(np.max(times_direct)),
        }

        # 计算加速比
        speedup = results["box_muller"]["mean_time"] / max(results["direct"]["mean_time"], 1e-10)
        results["speedup_ratio"] = speedup

        logger.info(
            f"性能基准测试 (size={sample_size}, runs={num_runs}): "
            f"Box-Muller={results['box_muller']['mean_time']:.4f}s, "
            f"Direct={results['direct']['mean_time']:.4f}s, "
            f"加速比={speedup:.2f}x"
        )

        return results

    def set_parameters(self, mean: Optional[float] = None, std: Optional[float] = None) -> None:
        """更新分布参数

        Args:
            mean: 新的均值（可选）
            std: 新的标准差（可选，必须 > 0）

        Raises:
            ValueError: 如果 std <= 0
        """
        if mean is not None:
            self.mean = mean

        if std is not None:
            if std <= 0:
                raise ValueError(f"std 必须为正数，得到 {std}")
            self.std = max(std, MIN_STD)

        logger.info(f"参数更新: μ={self.mean:.4f}, σ={self.std:.4f}")

    def to(self, device: Union[str, torch.device]) -> "GaussianSampler":
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
            f"GaussianSampler(μ={self.mean:.4f}, σ={self.std:.4f}, "
            f"method={self.method}, device={self.device})"
        )
