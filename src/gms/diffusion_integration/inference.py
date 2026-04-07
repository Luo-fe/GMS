"""GMS 推理 Pipeline - 快速图像生成

实现完整的推理流程，支持从训练好的 GMS-Diffusion 模型快速生成图像样本。
提供多种采样策略、批量生成和输出后处理功能。

核心特性:
    - 多种采样策略（DDPM, DDIM 加速采样）
    - 支持有条件和无条件生成
    - 批量生成和保存
    - 完整的中间结果记录
    - 灵活的输出格式配置

与标准扩散模型推理的对比:
    标准 DDPM 推理:
        - 固定的 1000 步马尔可夫链
        - 仅支持标准高斯噪声
        - 无条件引导

    GMS 推理:
        - 可选的加速采样（DDIM 50-100步）
        - GMM 引导的条件生成
        - 自适应噪声调度

Example:
    >>> from gms.diffusion_integration.inference import (
    ...     GMSInferencePipeline, InferenceConfig, GenerationResult
    ... )
    >>>
    >>> config = InferenceConfig(
    ...     sampling_steps=50,
    ...     batch_size=16,
    ...     method='ddim'
    ... )
    >>>
    >>> pipeline = GMSInferencePipeline(
    ...     model=trained_model,
    ...     noise_scheduler=scheduler,
    ...     backward_process=backward_proc,
    ...     config=config
    ... )
    >>>
    >>> result = pipeline.generate(n_samples=64)
    >>> print(f"生成了 {result.samples.shape[0]} 张图像")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Union, Tuple, Callable,
)
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None


class SamplingMethod(Enum):
    """采样方法枚举

    DDPM: 标准马尔可夫链采样（1000步），质量最高但最慢
    DDIM: 确定性加速采样（50-100步），速度快且质量接近
    DDPM_FAST: 快速 DDPM（250步），质量和速度的折中
    CUSTOM: 自定义时间步序列
    """
    DDPM = "ddpm"
    DDIM = "ddim"
    DDPM_FAST = "ddpm_fast"
    CUSTOM = "custom"


@dataclass
class InferenceConfig:
    """推理配置数据类

    控制生成过程的所有参数。

    Attributes:
        sampling_steps: 采样步数（DDPM 通常 1000，DDIM 通常 50-100）
        method: 采样方法
        batch_size: 每批生成的样本数
        guidance_scale: 分类器自由引导强度（>1 增强条件影响）
        eta: DDIM 的随机性参数（0=完全确定性，1=与DDPM相同）
        num_inference_timesteps: 推理时使用的时间步数
        output_dir: 输出目录
        output_format: 输出格式 ('png', 'jpg', 'npy', 'pt')
        image_size: 输出图像尺寸 (H, W) 或 None（使用模型默认值）
        clamp_output: 是否将输出裁剪到 [0, 1]
        denormalize: 是否反归一化输出
        use_gms_guidance: 是否启用 GMS 引导
        gmm_guidance_strength: GMS 引导强度
        seed: 随机种子（None 表示随机）
        device: 计算设备
        dtype: 数据类型
        verbose: 是否打印详细进度信息

    Example:
        >>> config = InferenceConfig(
        ...     sampling_steps=50,
        ...     method='ddim',
        ...     batch_size=8,
        ...     guidance_scale=7.5
        ... )
    """

    sampling_steps: int = 50
    method: str = "ddim"
    batch_size: int = 1
    guidance_scale: float = 1.0
    eta: float = 0.0
    num_inference_timesteps: Optional[int] = None

    output_dir: str = "./generated"
    output_format: str = "png"
    image_size: Optional[Tuple[int, int]] = None
    clamp_output: bool = True
    denormalize: bool = True

    use_gms_guidance: bool = False
    gmm_guidance_strength: float = 1.0

    seed: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    verbose: bool = True

    def __post_init__(self):
        """初始化后验证"""
        if self.sampling_steps <= 0:
            raise ValueError(f"sampling_steps 必须为正整数，当前值: {self.sampling_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，当前值: {self.batch_size}")
        if self.guidance_scale < 0:
            raise ValueError(f"guidance_scale 不能为负数，当前值: {self.guidance_scale}")
        if self.output_format not in ['png', 'jpg', 'npy', 'pt']:
            raise ValueError(f"不支持的输出格式: {self.output_format}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceConfig":
        """从字典创建"""
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


@dataclass
class GenerationResult:
    """生成结果数据类

    存储生成的样本及其元信息和统计指标。

    Attributes:
        samples: 生成的样本张量，形状 (N, C, H, W)
        generation_time: 总生成耗时（秒）
        samples_per_second: 每秒生成速率
        config: 使用的推理配置
        timestamps: 时间戳列表
        intermediate_results: 中间结果列表（如果记录）
        metadata: 其他元数据字典
        quality_metrics: 质量评估指标（可选）

    Example:
        >>> result = pipeline.generate(n_samples=16)
        >>> print(f"生成 {result.samples.shape[0]} 张图，耗时 {result.generation_time:.2f}s")
        >>> result.save('./output/')
    """

    samples: torch.Tensor
    generation_time: float
    samples_per_second: float
    config: InferenceConfig
    timestamps: List[str] = field(default_factory=list)
    intermediate_results: Optional[List[torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[Dict[str, float]] = None

    def save(
        self,
        save_dir: str,
        filename_prefix: str = "generated",
        indices: Optional[List[int]] = None,
    ) -> List[str]:
        """保存生成的样本到文件

        Args:
            save_dir: 保存目录
            filename_prefix: 文件名前缀
            indices: 要保存的样本索引（None 表示全部）

        Returns:
            保存的文件路径列表
        """
        os.makedirs(save_dir, exist_ok=True)

        saved_paths = []
        samples_to_save = self.samples

        if indices is not None:
            samples_to_save = self.samples[indices]

        for i in range(samples_to_save.shape[0]):
            sample = samples_to_save[i]

            if self.config.output_format in ['png', 'jpg'] and HAS_PIL:
                self._save_image(sample, save_dir, filename_prefix, i, saved_paths)

            elif self.config.output_format == 'npy':
                path = os.path.join(save_dir, f"{filename_prefix}_{i:04d}.npy")
                np.save(path, sample.cpu().numpy())
                saved_paths.append(path)

            elif self.config.output_format == 'pt':
                path = os.path.join(save_dir, f"{filename_prefix}_{i:04d}.pt")
                torch.save(sample.cpu(), path)
                saved_paths.append(path)

        if logger:
            logger.info(f"已保存 {len(saved_paths)} 个文件到 {save_dir}")

        return saved_paths

    def _save_image(
        self,
        sample: torch.Tensor,
        save_dir: str,
        prefix: str,
        index: int,
        saved_paths: List[str],
    ) -> None:
        """保存单个图像文件

        Args:
            sample: 单个样本张量
            save_dir: 保存目录
            prefix: 文件前缀
            index: 样本索引
            saved_paths: 已保存路径列表（追加到此）
        """
        sample_np = sample.cpu().numpy()

        if sample_np.ndim == 3 and sample_np.shape[0] in [1, 3, 4]:
            sample_np = np.transpose(sample_np, (1, 2, 0))

        if sample_np.shape[-1] == 1:
            sample_np = sample_np.squeeze(-1)

        sample_np = np.clip(sample_np * 255, 0, 255).astype(np.uint8)

        filename = f"{prefix}_{index:04d}.{self.config.output_format}"
        filepath = os.path.join(save_dir, filename)

        Image.fromarray(sample_np).save(filepath, quality=95)
        saved_paths.append(filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """获取生成统计信息

        Returns:
            统计信息字典
        """
        samples_np = self.samples.cpu().numpy()

        return {
            'num_samples': self.samples.shape[0],
            'sample_shape': list(self.samples.shape[1:]),
            'generation_time_s': self.generation_time,
            'samples_per_sec': self.samples_per_second,
            'mean': float(np.mean(samples_np)),
            'std': float(np.std(samples_np)),
            'min': float(np.min(samples_np)),
            'max': float(np.max(samples_np)),
            'config': self.config.__dict__,
        }


class GMSInferencePipeline:
    """GMS 推理 Pipeline

    封装完整的图像生成流程，从训练好的 GMS-Diffusion 模型快速生成高质量样本。

    工作流程:
        1. 初始化：加载模型权重和配置
        2. 采样纯噪声 x_T ~ N(0, I)
        3. 迭代去噪：对于 t = T, T-1, ..., 1:
           a. 通过去噪网络预测 ε_θ(x_t, t)
           b. （可选）应用 GMS 引导或 CFG
           c. 计算并采样 x_{t-1}
        4. 后处理：裁剪、反归一化等
        5. 返回生成的样本

    支持的采样策略:
        - DDPM (标准): 1000 步，最高质量
        - DDIM (加速): 50-100 步，速度快
        - DDPM_FAST: 250 步，平衡方案

    Attributes:
        model: 训练好的去噪网络
        noise_scheduler: 噪声调度器
        backward_process: 反向去噪过程
        config: 推理配置
        device: 计算设备
        _timesteps: 自定义时间步序列（用于 CUSTOM 方法）

    Example:
        >>> pipeline = GMSInferencePipeline(
        ...     model=trained_unet,
        ...     noise_scheduler=scheduler,
        ...     backward_process=backward,
        ...     config=InferenceConfig(sampling_steps=50, method='ddim')
        ... )
        >>>
        >>> # 生成单个批次
        >>> samples = pipeline.generate(n_samples=16)
        >>>
        >>> # 批量生成
        >>> results = pipeline.generate_batch(
        ...     total_samples=256,
        ...     save_dir='./outputs'
        ... )
    """

    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: "NoiseScheduler",
        backward_process: "GMSBackwardProcess",
        config: Optional[InferenceConfig] = None,
        condition_encoder: Optional[nn.Module] = None,
        gmm_parameters: Optional["GMMParameters"] = None,
    ):
        """初始化推理 Pipeline

        Args:
            model: 训练好的去噪神经网络
            noise_scheduler: NoiseScheduler 实例
            backward_process: GMSBackwardProcess 实例
            config: InferenceConfig 配置（可选）
            condition_encoder: 条件编码器（可选）
            gmm_parameters: GMM 参数（用于 GMS 引导）
        """
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.backward_process = backward_process
        self.condition_encoder = condition_encoder
        self.gmm_parameters = gmm_parameters

        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        self.dtype = self.config.dtype

        self.model.to(self.device).to(self.dtype).eval()
        self.noise_scheduler.to(self.device)
        self.backward_process.to(self.device)

        if self.condition_encoder is not None:
            self.condition_encoder.to(self.device).to(self.dtype).eval()

        self._timesteps: Optional[List[int]] = None
        self._generator: Optional[torch.Generator] = None

        if self.config.seed is not None:
            self._set_seed(self.config.seed)

        os.makedirs(self.config.output_dir, exist_ok=True)

        if logger:
            logger.info(
                f"GMSInferencePipeline 初始化完成: "
                f"method={self.config.method}, "
                f"steps={self.config.sampling_steps}, "
                f"device={self.device}"
            )

    def _set_seed(self, seed: int) -> None:
        """设置随机种子以确保可重复性

        Args:
            seed: 随机种子值
        """
        import random
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

        if logger:
            logger.debug(f"推理随机种子设置为: {seed}")

    def set_custom_timesteps(self, timesteps: List[int]) -> None:
        """设置自定义时间步序列

        用于 CUSTOM 采样方法。

        Args:
            timesteps: 降序排列的时间步列表，如 [999, 998, ..., 0]

        Example:
            >>> # 使用对数间隔的时间步（更关注早期步骤）
            >>> import numpy as np
            >>> timesteps = np.linspace(0, 999, 50).astype(int)[::-1].tolist()
            >>> pipeline.set_custom_timesteps(timesteps)
        """
        self._timesteps = sorted(timesteps, reverse=True)

        if logger:
            logger.info(f"自定义时间步已设置: {len(timesteps)} 步")

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        condition: Optional[torch.Tensor] = None,
        record_intermediates: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> GenerationResult:
        """生成样本

        从纯噪声开始，通过迭代去噪生成指定数量的样本。

        Args:
            n_samples: 要生成的样本数量
            condition: 条件向量（形状 (B, D) 或 (D,)）
            record_intermediates: 是否记录中间结果
            progress_callback: 进度回调函数 callback(step, total)

        Returns:
            GenerationResult 包含生成的样本和元信息

        Example:
            >>> result = pipeline.generate(n_samples=8)
            >>> images = result.samples  # 形状 (8, C, H, W)
        """
        start_time = time.time()

        effective_batch_size = min(n_samples, self.config.batch_size)
        all_samples = []

        for batch_start in range(0, n_samples, effective_batch_size):
            current_batch_size = min(effective_batch_size, n_samples - batch_start)

            batch_result = self._generate_batch(
                batch_size=current_batch_size,
                condition=condition,
                record_intermediates=record_intermediates,
                progress_callback=progress_callback,
            )

            all_samples.append(batch_result.samples)

        samples = torch.cat(all_samples, dim=0)[:n_samples]

        generation_time = time.time() - start_time
        samples_per_second = n_samples / max(generation_time, 1e-6)

        result = GenerationResult(
            samples=samples,
            generation_time=generation_time,
            samples_per_second=samples_per_second,
            config=self.config,
            timestamps=[datetime.now().isoformat()],
        )

        if self.config.verbose:
            summary = result.get_statistics()
            if logger:
                logger.info(
                    f"生成完成: {n_samples} 个样本, "
                    f"耗时 {generation_time:.2f}s, "
                    f"速度 {samples_per_second:.2f} samples/s"
                )

        return result

    def _get_sampling_timesteps(self) -> List[int]:
        """获取采样用的时间步序列

        Returns:
            降序排列的时间步列表
        """
        method = SamplingMethod(self.config.method)
        num_steps = self.config.sampling_steps
        total_steps = self.noise_scheduler.num_steps

        if method == SamplingMethod.DDPM:
            return list(range(total_steps - 1, -1, -1))

        elif method == SamplingMethod.DDIM or method == SamplingMethod.DDPM_FAST:
            step_ratio = total_steps // num_steps
            timesteps = list(range(0, total_steps, step_ratio))[:num_steps]
            timesteps = sorted(timesteps, reverse=True)

            if len(timesteps) < num_steps:
                missing = num_steps - len(timesteps)
                timesteps.extend([0] * missing)

            return timesteps[:num_steps]

        elif method == SamplingMethod.CUSTOM:
            if self._timesteps is None:
                raise ValueError("CUSTOM 方法需要先调用 set_custom_timesteps()")
            return self._timesteps

        else:
            return list(range(total_steps - 1, -1, -1))

    @torch.no_grad()
    def _generate_batch(
        self,
        batch_size: int,
        condition: Optional[torch.Tensor],
        record_intermediates: bool,
        progress_callback: Optional[Callable],
    ) -> GenerationResult:
        """生成一个批次的样本

        Args:
            batch_size: 批次大小
            condition: 条件向量
            record_intermediates: 是否记录中间结果
            progress_callback: 进度回调

        Returns:
            该批次的 GenerationResult
        """
        shape = self._infer_sample_shape(batch_size)
        x_T = torch.randn(shape, device=self.device, dtype=self.dtype,
                         generator=self._generator)

        timesteps = self._get_sampling_timesteps()
        x_current = x_T.clone()
        intermediates = [] if record_intermediates else None

        method = SamplingMethod(self.config.method)

        for i, t in enumerate(timesteps):
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )

            model_output = self.model(x_current, t_batch)

            if self.config.guidance_scale != 1.0 and condition is not None:
                model_output = self._apply_guidance(
                    x_current, t_batch, model_output, condition
                )

            if method == SamplingMethod.DDIM:
                x_current = self._ddim_step(
                    x_current, t_batch, model_output, t, timesteps
                )
            else:
                x_current = self.backward_process.sample_step(
                    x_current, t_batch, model_output, generator=self._generator
                )

            if record_intermediates and intermediates is not None:
                intermediates.append(x_current.clone())

            if progress_callback is not None:
                progress_callback(i + 1, len(timesteps))

        samples = self._postprocess(x_current)

        return GenerationResult(
            samples=samples,
            generation_time=0.0,
            samples_per_second=0.0,
            config=self.config,
            intermediate_results=intermediates,
        )

    def _infer_sample_shape(self, batch_size: int) -> Tuple[int, ...]:
        """推断样本张量的形状

        尝试从模型或配置中确定输出形状。

        Args:
            batch_size: 批次大小

        Returns:
            样本张量形状
        """
        if self.config.image_size is not None:
            h, w = self.config.image_size
            return (batch_size, 3, h, w)

        try:
            dummy_input = torch.randn(1, 3, 32, 32, device=self.device)
            with torch.no_grad():
                dummy_output = self.model(dummy_input, torch.tensor([0], device=self.device))
            return (batch_size,) + dummy_output.shape[1:]
        except Exception:
            return (batch_size, 3, 32, 32)

    def _apply_guidance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output_cond: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """应用分类器自由引导 (CFG)

        公式: output = output_uncond + scale * (output_cond - output_uncond)

        Args:
            x_t: 当前加噪数据
            t: 时间步
            model_output_cond: 有条件的模型输出
            condition: 条件向量

        Returns:
            引导后的模型输出
        """
        from .backward_process import apply_classifier_free_guidance

        uncond_output = self.model(x_t, t, gms_condition=None)

        guided = apply_classifier_free_guidance(
            model_output_cond=model_output_cond,
            model_output_uncond=uncond_output,
            guidance_scale=self.config.guidance_scale,
        )

        return guided

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        current_t: int,
        all_timesteps: List[int],
    ) -> torch.Tensor:
        """DDIM 单步采样

        使用确定性或随机性 DDIM 更新规则。

        数学公式 (DDIM deterministic, η=0):
            x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}) · direction
            其中 direction 指向 x_t

        Args:
            x_t: 当前状态
            t: 当前时间步张量
            model_output: 模型预测（噪声）
            current_t: 当前时间步整数值
            all_timesteps: 所有时间步列表

        Returns:
            更新后的状态
        """
        sched = self.noise_scheduler

        alpha_prod_t = sched.alphas_cumprod[current_t + 1]

        current_idx = all_timesteps.index(current_t) if current_t in all_timesteps else -1
        if current_idx > 0:
            prev_t = all_timesteps[current_idx - 1]
            alpha_prod_t_prev = sched.alphas_cumprod[prev_t + 1]
        else:
            alpha_prod_t_prev = torch.tensor(1.0, device=x_t.device)

        alpha_prod_t = alpha_prod_t.to(x_t.device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(x_t.device)

        predicted_x0 = (
            (x_t - torch.sqrt(1 - alpha_prod_t) * model_output) /
            torch.sqrt(alpha_prod_t.clamp(min=1e-10))
        )

        sigma_t = (
            self.config.eta *
            torch.sqrt((1 - alpha_prod_t_prev) / (1 - alpha_prod_t.clamp(min=1e-10))) *
            torch.sqrt(1 - alpha_prod_t / alpha_prod_t_prev.clamp(min=1e-10))
        )

        pred_direction = torch.sqrt(1 - alpha_prod_t_prev - sigma_t ** 2) * predicted_x0
        noise = sigma_t * model_output

        x_prev = torch.sqrt(alpha_prod_t_prev) * predicted_x0 + pred_direction + noise

        return x_prev

    def _postprocess(self, x: torch.Tensor) -> torch.Tensor:
        """后处理生成的样本

        包括裁剪到有效范围和反归一化。

        Args:
            x: 原始输出张量

        Returns:
            处理后的张量
        """
        if self.config.clamp_output:
            x = torch.clamp(x, 0.0, 1.0)

        return x

    @torch.no_grad()
    def generate_with_intermediates(
        self,
        n_samples: int,
        condition: Optional[torch.Tensor] = None,
        save_interval: int = 10,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """生成样本并返回中间结果

        用于可视化去噪过程。

        Args:
            n_samples: 样本数量
            condition: 条件向量
            save_interval: 保存中间结果的间隔

        Returns:
            (最终结果, 中间结果列表) 元组

        Example:
            >>> final, intermediates = pipeline.generate_with_intermediates(4)
            >>> # intermediates[i] 是第 i*save_interval 步的状态
        """
        result = self.generate(
            n_samples=n_samples,
            condition=condition,
            record_intermediates=True,
        )

        if result.intermediate_results is not None:
            sampled_intermediates = result.intermediate_results[::save_interval]
            return result.samples, sampled_intermediates

        return result.samples, []

    @torch.no_grad()
    def generate_batch(
        self,
        batch_size: int,
        total_samples: int,
        save_dir: Optional[str] = None,
        filename_prefix: str = "gen",
    ) -> List[GenerationResult]:
        """大批量生成并可选保存

        分批生成大量样本，适合生产环境使用。

        Args:
            batch_size: 每批大小
            total_samples: 总样本数
            save_dir: 保存目录（None 则不保存）
            filename_prefix: 文件名前缀

        Returns:
            每个 GenerationResult 的列表

        Example:
            >>> results = pipeline.generate_batch(
            ...     batch_size=32,
            ...     total_samples=1024,
            ...     save_dir='./large_generation'
            ... )
            >>> print(f"共生成 {sum(r.samples.shape[0] for r in results)} 张")
        """
        all_results = []

        for start in range(0, total_samples, batch_size):
            current_n = min(batch_size, total_samples - start)

            result = self.generate(n_samples=current_n)

            if save_dir:
                result.save(save_dir, filename_prefix=filename_prefix)

            all_results.append(result)

            if self.config.verbose and logger:
                logger.info(
                    f"批量生成进度: {min(start + batch_size, total_samples)}/{total_samples}"
                )

        return all_results

    def load_from_checkpoint(
        self,
        checkpoint_path: str,
        strict: bool = True,
    ) -> None:
        """从检查点加载模型权重

        Args:
            checkpoint_path: 检查点文件路径
            strict: 是否严格匹配参数
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

        if ('condition_encoder_state_dict' in checkpoint and
            self.condition_encoder is not None):
            self.condition_encoder.load_state_dict(
                checkpoint['condition_encoder_state_dict']
            )

        if 'gmm_params' in checkpoint and checkpoint['gmm_params'] is not None:
            from gms.gmm_optimization.gmm_parameters import GMMParameters
            self.gmm_parameters = GMMParameters.from_dict(checkpoint['gmm_params'])

        if logger:
            logger.info(f"模型权重已从检查点加载: {checkpoint_path}")

    def export_config(self) -> Dict[str, Any]:
        """导出当前配置

        Returns:
            配置字典
        """
        return {
            'inference_config': self.config.__dict__,
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'trainable': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            },
            'scheduler_info': self.noise_scheduler.get_schedule_info(),
            'backward_config': self.backward_process.export_state(),
        }

    def benchmark(
        self,
        n_samples: int = 16,
        warmup_runs: int = 2,
        benchmark_runs: int = 5,
    ) -> Dict[str, Any]:
        """性能基准测试

        测量不同配置下的生成速度。

        Args:
            n_samples: 测试用的样本数
            warmup_runs: 预热轮数
            benchmark_runs: 正式测试轮数

        Returns:
            性能统计数据

        Example:
            >>> stats = pipeline.benchmark(n_samples=32)
            >>> print(f"平均速度: {stats['avg_samples_per_sec']:.2f} samples/sec")
        """
        times = []

        for i in range(warmup_runs + benchmark_runs):
            start = time.time()
            self.generate(n_samples=n_samples)
            elapsed = time.time() - start

            if i >= warmup_runs:
                times.append(elapsed)

        avg_time = sum(times) / len(times) if times else 0
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5 if times else 0

        stats = {
            'n_samples': n_samples,
            'method': self.config.method,
            'sampling_steps': self.config.sampling_steps,
            'avg_time_seconds': round(avg_time, 4),
            'std_time_seconds': round(std_time, 4),
            'avg_samples_per_sec': round(n_samples / avg_time, 2),
            'benchmark_runs': benchmark_runs,
            'device': str(self.device),
        }

        if logger:
            logger.info(f"基准测试结果: {stats}")

        return stats


def create_inference_pipeline_from_trainer(
    trainer: "GMSTrainer",
    inference_config: Optional[InferenceConfig] = None,
) -> GMSInferencePipeline:
    """从训练器创建推理 Pipeline

    便捷函数，自动从训练器提取所需组件。

    Args:
        trainer: 已训练的 GMSTrainer 实例
        inference_config: 推理配置（可选）

    Returns:
        配置完成的 GMSInferencePipeline

    Example:
        >>> history = trainer.train_full(epochs=100, dataloaders=loaders)
        >>> pipeline = create_inference_pipeline_from_trainer(trainer)
        >>> result = pipeline.generate(n_samples=16)
    """
    export_state = trainer.export_for_inference()

    config = inference_config or InferenceConfig(
        device=str(trainer.device),
        dtype=trainer.dtype,
    )

    pipeline = GMSInferencePipeline(
        model=trainer.model,
        noise_scheduler=trainer.noise_scheduler,
        backward_process=trainer.backward_process,
        config=config,
        condition_encoder=trainer.condition_encoder,
        gmm_parameters=trainer.gmm_parameters,
    )

    return pipeline


if __name__ == "__main__":
    print("GMS Inference Pipeline - 用于快速图像生成")
    print("\n主要组件:")
    print("  - GMSInferencePipeline: 主推理类")
    print("  - InferenceConfig: 推理配置")
    print("  - GenerationResult: 生成结果")
    print("\n支持的采样方法:")
    print("  - DDPM: 标准 1000 步采样")
    print("  - DDIM: 加速 50-100 步采样")
    print("\n快速开始:")
    print("  from gms.diffusion_integration.inference import GMSInferencePipeline")
    print("  pipeline = GMSInferencePipeline(model, scheduler, backward)")
    print("  result = pipeline.generate(n_samples=16)")
