"""可复现采样器 - 随机种子控制和结果复现

提供完整的随机状态管理，确保相同种子产生完全相同的采样序列。
与 PyTorch 的随机状态管理系统完全兼容。
"""

from typing import Optional, Dict, Any, Union, Tuple
import logging
import copy
import torch
import numpy as np

from .batch_sampler import BatchGaussianMixtureSampler

logger = logging.getLogger(__name__)


class ReproducibleSampler:
    """可复现的 GMM 采样器包装器

    管理随机种子的设置、保存和恢复，确保：
    - 相同的种子总是产生相同的采样序列
    - 可以保存和恢复随机状态以实现精确复现
    - 与 PyTorch 和 NumPy 的随机状态管理兼容

    使用示例：
        >>> sampler = ReproducibleSampler(seed=42)
        >>> samples1 = sampler.sample(100)
        >>> # 重置到初始状态
        >>> sampler.reset()
        >>> samples2 = sampler.sample(100)
        >>> assert torch.allclose(samples1, samples2)  # 完全相同

    Attributes:
        seed: 初始随机种子
        base_sampler: 底层的 BatchGaussianMixtureSampler 实例
    """

    def __init__(
        self,
        base_sampler: Optional[BatchGaussianMixtureSampler] = None,
        seed: int = 42,
        device: Union[str, torch.device] = "cpu",
        **sampler_kwargs,
    ) -> None:
        """初始化可复现采样器

        Args:
            base_sampler: 已有的 GMM 采样器实例（可选）
            seed: 随机种子（默认 42）
            device: 计算设备
            **sampler_kwargs: 如果未提供 base_sampler，则传递给 BatchGMM 构造函数
        """
        self.seed = seed

        if base_sampler is not None:
            self.base_sampler = base_sampler
        else:
            self.base_sampler = BatchGaussianMixtureSampler(
                device=device,
                **sampler_kwargs,
            )

        # 创建专用的随机数生成器
        self.generator = torch.Generator(device=self.base_sampler.device)
        self.generator.manual_seed(seed)

        # 保存初始状态作为参考点
        self._initial_state = self._get_full_random_state()

        logger.info(
            f"初始化 ReproducibleSampler: seed={seed}, "
            f"device={self.base_sampler.device}"
        )

    def sample(
        self,
        size: int = 1000,
    ) -> torch.Tensor:
        """执行可复现的 GMM 采样

        使用管理的随机生成器进行采样。

        Args:
            size: 样本数量

        Returns:
            GMM 样本张量
        """
        samples = self.base_sampler.sample(size, generator=self.generator)

        logger.debug(f"采样完成: size={size}, seed={self.seed}")

        return samples

    def sample_with_reproduction_check(
        self,
        size: int = 1000,
    ) -> Tuple[torch.Tensor, bool]:
        """采样并验证是否可以复现

        执行两次采样并比较结果，验证可复现性。

        Args:
            size: 样本数量

        Returns:
            (samples, is_reproducible) 元组
        """
        # 第一次采样
        state_before = self.save_state()
        samples1 = self.sample(size)

        # 恢复状态并重新采样
        self.restore_state(state_before)
        samples2 = self.sample(size)

        # 比较
        is_reproducible = torch.allclose(samples1, samples2, atol=1e-7)

        if not is_reproducible:
            logger.warning(
                f"可复现性检查失败! 最大差异: "
                f"{(samples1 - samples2).abs().max():.2e}"
            )

        return samples1, is_reproducible

    def set_seed(self, new_seed: int) -> None:
        """设置新的随机种子

        将所有相关随机数生成器重置为新种子。

        Args:
            new_seed: 新的种子值
        """
        old_seed = self.seed
        self.seed = new_seed
        self.generator.manual_seed(new_seed)
        self._initial_state = self._get_full_random_state()

        logger.info(f"种子更新: {old_seed} -> {new_seed}")

    def reset(self) -> None:
        """重置到初始状态

        恢复到构造时的随机状态。
        """
        self._set_full_random_state(self._initial_state)
        logger.debug("已重置到初始状态")

    def save_state(self) -> Dict[str, Any]:
        """保存当前随机状态

        Returns:
            包含完整随机状态的字典，可用于后续恢复
        """
        state = {
            "seed": self.seed,
            "generator_state": self.generator.get_state(),
            "torch_cpu_state": torch.get_rng_state(),
            "full_state": self._get_full_random_state(),
        }

        # 尝试保存 CUDA 状态（如果可用）
        if torch.cuda.is_available():
            try:
                state["torch_cuda_state"] = torch.cuda.get_rng_state_all()
            except Exception as e:
                logger.debug(f"无法保存 CUDA 状态: {e}")

        # 保存 NumPy 状态
        state["numpy_state"] = np.random.get_state()

        logger.debug("随机状态已保存")

        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """从保存的状态恢复

        Args:
            state: save_state() 返回的状态字典
        """
        # 恢复 PyTorch CPU 状态
        if "torch_cpu_state" in state:
            torch.set_rng_state(state["torch_cpu_state"])

        # 恢复 PyTorch CUDA 状态（如果可用）
        if torch.cuda.is_available() and "torch_cuda_state" in state:
            try:
                torch.cuda.set_rng_state_all(state["torch_cuda_state"])
            except Exception as e:
                logger.debug(f"无法恢复 CUDA 状态: {e}")

        # 恢复专用生成器状态
        if "generator_state" in state:
            self.generator.set_state(state["generator_state"])

        # 恢复 NumPy 状态
        if "numpy_state" in state:
            np.random.set_state(state["numpy_state"])

        # 更新种子记录
        if "seed" in state:
            self.seed = state["seed"]

        logger.debug("随机状态已恢复")

    def _get_full_random_state(self) -> Dict[str, Any]:
        """获取完整的随机状态快照"""
        return {
            "pytorch_cpu": torch.get_rng_state().clone(),
            "generator": self.generator.get_state().clone(),
            "numpy": list(np.random.get_state()[1][:10]),  # 只保存部分状态以节省空间
            "seed": self.seed,
        }

    def _set_full_random_state(self, state: Dict[str, Any]) -> None:
        """从完整状态快照恢复"""
        if "pytorch_cpu" in state:
            if isinstance(state["pytorch_cpu"], torch.Tensor):
                torch.set_rng_state(state["pytorch_cpu"])
            else:
                torch.set_rng_state(torch.tensor(state["pytorch_cpu"], dtype=torch.uint8))
        if "generator" in state:
            if isinstance(state["generator"], torch.Tensor):
                self.generator.set_state(state["generator"])
            else:
                self.generator.set_state(torch.tensor(state["generator"], dtype=torch.uint8))

    @staticmethod
    def set_global_seeds(
        seed: int,
        include_numpy: bool = True,
        include_cuda: bool = True,
    ) -> Dict[str, Any]:
        """设置全局随机种子（静态方法）

        同时设置 PyTorch、NumPy 和 CUDA 的全局随机种子。
        这是在训练开始时通常需要执行的操作。

        Args:
            seed: 种子值
            include_numpy: 是否设置 NumPy 种子
            include_cuda: 是否设置 CUDA 种子

        Returns:
            设置前的旧状态字典（可用于恢复）
        """
        old_state = {
            "seed": seed,
            "torch_cpu": torch.get_rng_state(),
        }

        # 设置 PyTorch 种子
        torch.manual_seed(seed)

        # 设置 NumPy 种子
        if include_numpy:
            old_state["numpy"] = np.random.get_state()
            np.random.seed(seed)

        # 设置 CUDA 种子
        if include_cuda and torch.cuda.is_available():
            try:
                old_state["cuda"] = torch.cuda.get_rng_state_all()
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                # 禁用 cudnn 的非确定性算法
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception as e:
                logger.warning(f"无法设置 CUDA 种子: {e}")

        logger.info(f"全局种子已设置为: {seed} (包含 numpy={include_numpy}, cuda={include_cuda})")

        return old_state

    @staticmethod
    def restore_global_seeds(old_state: Dict[str, Any]) -> None:
        """恢复全局随机状态（静态方法）

        Args:
            old_state: set_global_seeds 返回的旧状态
        """
        if "torch_cpu" in old_state:
            torch.set_rng_state(old_state["torch_cpu"])
        if "numpy" in old_state:
            np.random.set_state(old_state["numpy"])
        if "cuda" in old_state and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(old_state["cuda"])
            except Exception as e:
                logger.warning(f"无法恢复 CUDA 状态: {e}")

        logger.info("全局随机状态已恢复")

    def verify_reproducibility(
        self,
        num_trials: int = 3,
        sample_size: int = 1000,
    ) -> Tuple[bool, float]:
        """严格验证可复现性

        多次重置和采样，验证每次都得到相同的结果。

        Args:
            num_trials: 试验次数
            sample_size: 每次采样的样本数

        Returns:
            (all_identical, max_deviation) 元组
        """
        reference_samples = None
        max_deviation = 0.0
        all_identical = True

        for trial in range(num_trials):
            self.reset()
            samples = self.sample(sample_size)

            if reference_samples is None:
                reference_samples = samples.clone()
            else:
                deviation = (samples - reference_samples).abs().max().item()
                max_deviation = max(max_deviation, deviation)

                if deviation > 1e-6:
                    all_identical = False
                    logger.warning(
                        f"试验 {trial+1}: 检测到偏差 {deviation:.2e}"
                    )

        result_str = "通过" if all_identical else f"失败 (最大偏差: {max_deviation:.2e})"
        logger.info(f"可复现性验证 ({num_trials} 次试验): {result_str}")

        return all_identical, max_deviation

    def get_generator(self) -> torch.Generator:
        """获取底层的 PyTorch 随机数生成器

        Returns:
            当前使用的 Generator 对象
        """
        return self.generator

    def to(self, device: Union[str, torch.device]) -> "ReproducibleSampler":
        """移动到指定设备

        注意：移动设备会重新创建生成器，可能影响可复现性。

        Args:
            device: 目标设备

        Returns:
            self（支持链式调用）
        """
        self.base_sampler.to(device)
        new_device = device if isinstance(device, torch.device) else torch.device(device)
        self.generator = torch.Generator(device=new_device)
        self.generator.manual_seed(self.seed)
        return self

    def __repr__(self) -> str:
        return (
            f"ReproducibleSampler(seed={self.seed}, "
            f"sampler={self.base_sampler.__repr__()})"
        )
