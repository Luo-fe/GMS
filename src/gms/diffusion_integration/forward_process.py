"""GMS 前向扩散过程 - 噪声调度对齐

实现扩散模型前向过程 (Forward Process / Noising Process) 的 GMS 集成版本。
标准 DDPM 的前向过程定义为:
    q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)

GMS 版本使用 GMM 采样的噪声替代标准高斯噪声，使得噪声的统计特性
由 GMM 参数控制，从而在数据分布建模上提供更强的表达能力。

与标准扩散模型的对比:
    标准: ε ~ N(0, I) 固定的高斯白噪声
    GMS:  ε ~ N(μ_t, σ_t²I) 时间步依赖的、参数化的噪声

Example:
    >>> import torch
    >>> from gms.diffusion_integration.forward_process import (
    ...     GMSForwardProcess, NoiseScheduler
    ... )
    >>>
    >>> scheduler = NoiseScheduler(num_steps=1000, schedule_type='cosine')
    >>> forward_process = GMSForwardProcess(scheduler)
    >>>
    >>> x_0 = torch.randn(4, 3, 32, 32)
    >>> t = torch.tensor([50, 200, 500, 800])
    >>> x_t, noise = forward_process(x_0, t)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Tuple
import math
import torch
import torch.nn as nn

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None


class ScheduleType(Enum):
    """噪声调度类型枚举

    LINEAR: 线性调度 β_t 线性递增（原始DDPM）
    COSINE: 余弦调度（Improved DDPM，更平滑）
    SQRT: 平方根调度
    GMM_ADAPTIVE: 基于 GMM 参数的自适应调度
    CUSTOM: 自定义调度曲线
    """

    LINEAR = "linear"
    COSINE = "cosine"
    SQRT = "sqrt"
    GMM_ADAPTIVE = "gmm_adaptive"
    CUSTOM = "custom"


@dataclass
class SchedulerConfig:
    """噪声调度器配置

    Attributes:
        num_steps: 总时间步数 T
        schedule_type: 调度类型
        beta_start: β 的起始值
        beta_end: β 的结束值
        variance_type: 方差类型 ('fixed_small', 'fixed_large', 'learned')
        clip_min: ᾱ_t 的最小裁剪值
        device: 计算设备
    """

    num_steps: int = 1000
    schedule_type: str = "cosine"
    beta_start: float = 1e-4
    beta_end: float = 0.02
    variance_type: str = "fixed_small"
    clip_min: float = 0.0001
    device: Union[str, torch.device] = "cpu"


class NoiseScheduler(nn.Module):
    """噪声调度管理器

    管理扩散过程中所有时间步的噪声水平参数计算：
    - α_t = 1 - β_t （保留信号比例）
    - ᾱ_t = ∏_{s=1}^{t} α_s （累积保留比例）
    - σ_t² = 1 - ᾱ_t 或其他变体 （噪声方差）

    支持多种调度曲线和与采样控制器的时间步对齐。

    数学公式:
        标准线性调度:
            β_t = β_start + (β_end - β_start) * t / T

        余弦调度 (Nichol & Dhariwal, 2021):
            f(t) = cos((t/T + s)/(1+s) · π/2)²
            ᾱ_t = f(t)/f(0), β_t = 1 - α_t/α_{t-1}

    Attributes:
        config: 调度配置
        betas: 形状为 (T,) 的 β 序列
        alphas: 形状为 (T,) 的 α 序列
        alphas_cumprod: 形状为 (T+1,) 的累积 α（含 ᾱ_0=1）

    Example:
        >>> scheduler = NoiseScheduler(num_steps=1000, schedule_type='cosine')
        >>> # 获取第500步的参数
        >>> alpha_500 = scheduler.alphas[500]
        >>> sqrt_alpha_bar_500 = scheduler.sqrt_alphas_cumprod[500 + 1]
    """

    def __init__(
        self,
        num_steps: int = 1000,
        schedule_type: Union[str, ScheduleType] = ScheduleType.COSINE,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        variance_type: str = "fixed_small",
        custom_betas: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """初始化噪声调度器

        Args:
            num_steps: 扩散总时间步数
            schedule_type: 调度类型字符串或枚举
            beta_start: β 起始值
            beta_end: β 结束值
            variance_type: 方差计算方式
            custom_betas: 自定义 β 序列（schedule_type=CUSTOM 时使用）
            device: 计算设备
            dtype: 数据类型

        Raises:
            ValueError: 如果参数不合法
        """
        super().__init__()

        if isinstance(schedule_type, str):
            schedule_type = ScheduleType(schedule_type)

        if num_steps <= 0:
            raise ValueError(f"num_steps 必须为正整数，当前值: {num_steps}")
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError(f"beta_start 和 beta_end 必须为正数")
        if beta_start >= beta_end:
            raise ValueError(f"beta_start ({beta_start}) 必须 < beta_end ({beta_end})")

        self.config = SchedulerConfig(
            num_steps=num_steps,
            schedule_type=schedule_type.value,
            beta_start=beta_start,
            beta_end=beta_end,
            variance_type=variance_type,
            device=device,
        )
        self.num_steps = num_steps
        self.schedule_type = schedule_type
        self.variance_type = variance_type
        self.dtype = dtype

        _device = device if isinstance(device, torch.device) else torch.device(device)

        if schedule_type == ScheduleType.CUSTOM and custom_betas is not None:
            betas = custom_betas.to(_device).to(dtype)
        else:
            betas = self._build_beta_schedule(
                schedule_type, num_steps, beta_start, beta_end, _device, dtype
            )

        self.register_buffer('betas', betas)

        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)

        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = torch.cat(
            [torch.ones(1, device=_device, dtype=dtype), alphas_cumprod]
        )
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        self.register_buffer(
            'sqrt_alphas_cumprod',
            torch.sqrt(alphas_cumprod),
        )

        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - alphas_cumprod),
        )

        self.register_buffer(
            'log_one_minus_alphas_cumprod',
            torch.log(1.0 - alphas_cumprod.clamp(min=1e-20)),
        )

        self.register_buffer(
            'alphas_cumprod_prev',
            F.pad(alphas_cumprod[:-1], (1, 0), value=1.0),
        )

        self._compute_posterior_variance()

        if logger:
            logger.info(
                f"NoiseScheduler 初始化完成: steps={num_steps}, "
                f"type={schedule_type.value}, "
                f"β_range=[{betas.min():.6f}, {betas.max():.4f}]"
            )

    def _build_beta_schedule(
        self,
        schedule_type: ScheduleType,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """构建 β 调度序列

        Args:
            schedule_type: 调度类型
            num_steps: 步数
            beta_start: 起始β
            beta_end: 结束β
            device: 设备
            dtype: 数据类型

        Returns:
            形状为 (num_steps,) 的 β 张量
        """
        if schedule_type == ScheduleType.LINEAR:
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=dtype)

        elif schedule_type == ScheduleType.COSINE:
            s = 0.008
            steps = torch.linspace(0, 1, num_steps + 1, device=device, dtype=dtype)
            alphas_cumprod = torch.cos(((steps + s) / (1 + s)) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, min=1e-6, max=0.999)

        elif schedule_type == ScheduleType.SQRT:
            steps = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
            betas = beta_start + (beta_end - beta_start) * torch.sqrt(steps)

        elif schedule_type == ScheduleType.GMM_ADAPTIVE:
            steps = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
            betas = beta_start + (beta_end - beta_start) * (steps ** 1.5)

        else:
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device, dtype=dtype)

        return betas

    def _compute_posterior_variance(self) -> None:
        """计算后验方差 β̃_t

        后验方差用于反向过程中的重参数化:
            β̃_t = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)
        """
        posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev[1:]) /
            (1 - self.alphas_cumprod[1:]).clamp(min=1e-20)
        )

        if self.variance_type == "fixed_small":
            posterior_variance = torch.clamp(posterior_variance, min=1e-20)
        elif self.variance_type == "fixed_large":
            posterior_variance = self.betas.clone()
        elif self.variance_type == "learned":
            posterior_variance = torch.zeros_like(self.betas)
        else:
            posterior_variance = torch.clamp(posterior_variance, min=1e-20)

        self.register_buffer('posterior_variance', posterior_variance)

        self.register_buffer(
            'posterior_log_variance_clipped',
            torch.log(posterior_variance.clamp(min=1e-20)),
        )

        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(self.alphas_cumprod_prev[1:]) *
            self.betas / (1 - self.alphas_cumprod[1:]).clamp(min=1e-20),
        )

        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(self.alphas) *
            (1 - self.alphas_cumprod_prev[1:]) /
            (1 - self.alphas_cumprod[1:]).clamp(min=1e-20),
        )

    def get_alpha_and_sigma(
        self,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定时间步的 √ᾱ_t 和 √(1-ᾱ_t)

        Args:
            timestep: 时间步张量，形状 (B,)

        Returns:
            (sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod) 元组，
            形状均为 (B, 1, 1, 1)（可广播到图像形状）
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep + 1].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep + 1].reshape(-1, 1, 1, 1)
        return sqrt_alpha, sqrt_one_minus_alpha

    def get_alpha_and_sigma_flat(
        self,
        timestep: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定时间步的 √ᾱ_t 和 √(1-ᾱ_t)（扁平版本）

        用于非图像数据的处理。

        Args:
            timestep: 时间步张量，形状 (B,)

        Returns:
            (sqrt_alpha_cumprod, sqrt_one_minus_alpha_cumprod) 元组，
            形状均为 (B, 1)
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timestep + 1].reshape(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timestep + 1].reshape(-1, 1)
        return sqrt_alpha, sqrt_one_minus_alpha

    def get_beta(self, timestep: int) -> float:
        """获取指定时间步的 β_t

        Args:
            timestep: 时间步索引

        Returns:
            β_t 值
        """
        return self.betas[timestep].item()

    def get_schedule_info(self) -> Dict[str, Any]:
        """获取调度信息摘要

        Returns:
            包含调度统计信息的字典
        """
        return {
            'type': self.schedule_type.value,
            'num_steps': self.num_steps,
            'beta_range': (self.betas.min().item(), self.betas.max().item()),
            'alpha_final': self.alphas_cumprod[-1].item(),
            'variance_type': self.variance_type,
        }

    def interpolate_to_steps(
        self,
        target_steps: int,
    ) -> "NoiseScheduler":
        """插值到新的时间步数

        Args:
            target_steps: 目标步数

        Returns:
            新的 NoiseScheduler 实例
        """
        new_betas = torch.nn.functional.interpolate(
            self.betas.unsqueeze(0).unsqueeze(0),
            size=target_steps,
            mode='linear',
            align_corners=True,
        ).squeeze()

        return NoiseScheduler(
            num_steps=target_steps,
            schedule_type=self.schedule_type,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            variance_type=self.variance_type,
            custom_betas=new_betas,
            device=self.betas.device,
            dtype=self.betas.dtype,
        )


import torch.nn.functional as F


class GMSForwardProcess(nn.Module):
    """GMS 集成的扩散前向过程

    实现 q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, σ_t²·I) 的 GMS 版本。

    与标准前向过程的区别:
        - 标准: 使用 ε ~ N(0, I) 的固定高斯噪声
        - GMS:  使用由 GMM 参数控制的非均匀噪声分布

    GMS 特性:
        1. 噪声的均值可以是非零的（来自 GMM 混合均值）
        2. 噪声的方差随时间步自适应变化
        3. 支持各向异性噪声（不同维度不同方差）

    Attributes:
        noise_scheduler: 噪声调度器实例
        gmm_noise_enabled: 是否启用 GMM 噪声模式
        noise_offset: 全局噪声偏移量

    Example:
        >>> scheduler = NoiseScheduler(num_steps=1000)
        >>> forward_proc = GMSForwardProcess(scheduler)
        >>>
        >>> x_0 = torch.randn(8, 3, 32, 32)
        >>> t = torch.randint(0, 1000, (8,))
        >>> x_t, eps = forward_proc(x_0, t)
        >>>
        >>> # 获取完整的前向轨迹
        >>> trajectory = forward_proc.forward_full(x_0, [0, 100, 500, 999])
    """

    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        gmm_noise_enabled: bool = True,
        noise_offset: float = 0.0,
        clip_output: bool = True,
        clip_value: float = 10.0,
    ):
        """初始化 GMS 前向过程

        Args:
            noise_scheduler: NoiseScheduler 实例
            gmm_noise_enabled: 是否使用 GMM 噪声模式
            noise_offset: 噪声均值的全局偏移
            clip_output: 是否裁剪输出
            clip_value: 裁剪阈值
        """
        super().__init__()

        self.noise_scheduler = noise_scheduler
        self.gmm_noise_enabled = gmm_noise_enabled
        self.noise_offset = noise_offset
        self.clip_output = clip_output
        self.clip_value = clip_value

        if logger:
            logger.info(
                f"GMSForwardProcess 初始化完成: "
                f"gmm_noise={gmm_noise_enabled}, offset={noise_offset}"
            )

    def forward(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        gmm_noise_params: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步前向扩散

        计算 q(x_t | x_0)，即从干净数据添加噪声到指定时间步。

        数学公式:
            x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
            其中 ε ~ N(μ_t, σ_t²) 在 GMS 模式下

        Args:
            x_0: 干净数据，形状 (B, C, H, W) 或 (B, D)
            t: 时间步，形状 (B,)，每个元素为 [0, num_steps-1] 的整数
            gmm_noise_params: 可选的 GMM 噪声参数字典:
                - 'mean': 噪声均值，形状 (B, ...) 或标量
                - 'variance': 噪声方差，形状 (B, ...) 或标量
            generator: PyTorch 随机数生成器

        Returns:
            (x_t, noise) 元组:
                - x_t: 加噪后的数据，形状与 x_0 相同
                - noise: 实际使用的噪声，形状与 x_0 相同
        """
        scheduler = self.noise_scheduler

        if x_0.dim() == 2:
            sqrt_alpha, sqrt_one_minus_alpha = scheduler.get_alpha_and_sigma_flat(t)
        elif x_0.dim() == 4:
            sqrt_alpha, sqrt_one_minus_alpha = scheduler.get_alpha_and_sigma(t)
        else:
            sqrt_alpha = scheduler.sqrt_alphas_cumprod[t + 1].reshape(-1, *[1]*(x_0.dim()-1))
            sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t + 1].reshape(-1, *[1]*(x_0.dim()-1))

        if self.gmm_noise_enabled and gmm_noise_params is not None:
            noise = self._sample_gmm_noise(
                x_0.shape, gmm_noise_params, generator
            )
        else:
            noise = torch.randn_like(x_0, generator=generator)
            noise = noise + self.noise_offset

        x_t = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

        if self.clip_output:
            x_t = torch.clamp(x_t, -self.clip_value, self.clip_value)

        if logger:
            logger.debug(
                f"前向扩散: t={t.tolist()}, "
                f"x_0 range=[{x_0.min():.3f}, {x_0.max():.3f}], "
                f"x_t range=[{x_t.min():.3f}, {x_t.max():.3f}], "
                f"noise std={noise.std():.4f}"
            )

        return x_t, noise

    def _sample_gmm_noise(
        self,
        shape: torch.Size,
        gmm_params: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """根据 GMM 参数采样噪声

        从 GMM 控制的分布中采样噪声:
            ε ~ w·N(μ₁, Σ₁) + (1-w)·N(μ₂, Σ₂)

        Args:
            shape: 输出张量的形状
            gmm_params: GMM 参数字典
            generator: 随机数生成器

        Returns:
            噪声张量
        """
        mean = gmm_params.get('mean')
        variance = gmm_params.get('variance')

        base_noise = torch.randn(shape, generator=generator)

        if mean is not None:
            mean_expanded = self._expand_param(mean, shape)
            base_noise = base_noise + mean_expanded

        if variance is not None:
            var_expanded = self._expand_param(variance, shape)
            var_expanded = var_expanded.clamp(min=1e-10)
            base_noise = base_noise * torch.sqrt(var_expanded)

        base_noise = base_noise + self.noise_offset

        return base_noise

    @staticmethod
    def _expand_param(param: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """将参数广播到目标形状

        Args:
            param: 参数张量
            target_shape: 目标形状

        Returns:
            广播后的张量
        """
        if param.dim() == 1 and len(target_shape) >= 2:
            param = param.unsqueeze(0).expand(target_shape[0], -1)
        else:
            while param.dim() < len(target_shape):
                param = param.unsqueeze(-1)
            param = param.expand(target_shape)
        return param

    def forward_full(
        self,
        x_0: torch.Tensor,
        all_timesteps: List[int],
        gmm_noise_params: Optional[Dict[str, torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """完整前向轨迹

        计算多个时间步上的前向结果，返回完整的加噪轨迹。

        Args:
            x_0: 干净数据
            all_timesteps: 要计算的时间步列表
            gmm_noise_params: 可选的 GMM 噪声参数
            generator: 随机数生成器

        Returns:
            包含每个时间步 x_t 的列表，顺序与 all_timesteps 一致

        Example:
            >>> trajectory = forward_proc.forward_full(
            ...     x_0, [0, 250, 500, 750, 999]
            ... )
            >>> print(f"轨迹长度: {len(trajectory)}")
            >>> # trajectory[0] 对应 t=0 (几乎无噪声)
            >>> # trajectory[-1] 对应 t=T (几乎纯噪声)
        """
        trajectory = []

        for step in all_timesteps:
            t_tensor = torch.full(
                (x_0.shape[0],),
                step,
                device=x_0.device,
                dtype=torch.long,
            )
            x_t, _ = self.forward(
                x_0, t_tensor, gmm_noise_params, generator
            )
            trajectory.append(x_t.detach())

        if logger:
            logger.debug(
                f"完整前向轨迹: {len(all_timesteps)} 个时间步, "
                f"x_0 range=[{x_0.min():.3f}, {x_0.max():.3f}], "
                f"x_T range=[{trajectory[-1].min():.3f}, {trajectory[-1].max():.3f}]"
            )

        return trajectory

    def compute_loss_weight(
        self,
        t: torch.Tensor,
        weighting_scheme: str = "uniform",
    ) -> torch.Tensor:
        """计算损失权重

        不同时间步可能需要不同的损失权重以优化训练动态。

        Args:
            t: 时间步张量，形状 (B,)
            weighting_scheme: 权重方案:
                - 'uniform': 统一权重 (1.0)
                - 'min_snr': 基于 SNR 的加权 (Ho et al.)
                - 'snr': SNR 反比加权
                - 'truncated_snr': 截断 SNR 加权

        Returns:
            权重张量，形状 (B,)
        """
        if weighting_scheme == "uniform":
            weights = torch.ones_like(t, dtype=torch.float32)

        elif weighting_scheme in ("min_snr", "snr", "truncated_snr"):
            snr = (
                self.noise_scheduler.alphas_cumprod[t + 1] /
                (1 - self.noise_scheduler.alphas_cumprod[t + 1]).clamp(min=1e-10)
            )

            if weighting_scheme == "min_snr":
                gamma = 5.0
                weights = torch.clamp(snr, max=gamma) / snr.clamp(min=1e-6)
            elif weighting_scheme == "snr":
                weights = 1.0 / snr.clamp(min=1e-6)
            elif weighting_scheme == "truncated_snr":
                weights = torch.where(snr < 1.0, torch.ones_like(snr), 1.0 / snr)
            else:
                weights = torch.ones_like(t, dtype=torch.float32)
        else:
            weights = torch.ones_like(t, dtype=torch.float32)

        return weights

    @torch.no_grad()
    def visualize_trajectory(
        self,
        x_0: torch.Tensor,
        num_vis_steps: int = 5,
    ) -> Dict[int, torch.Tensor]:
        """可视化用：获取均匀间隔的前向轨迹点

        Args:
            x_0: 输入数据（单个样本）
            num_vis_steps: 可视化步骤数

        Returns:
            {timestep: x_t} 字典
        """
        T = self.noise_scheduler.num_steps
        vis_timesteps = [int(i * (T - 1) / (num_vis_steps - 1)) for i in range(num_vis_steps)]

        results = {}
        for t in vis_timesteps:
            t_tensor = torch.tensor([t], device=x_0.device, dtype=torch.long)
            x_t, _ = self.forward(x_0.unsqueeze(0), t_tensor)
            results[t] = x_t.squeeze(0).detach().cpu()

        return results

    def extra_repr(self) -> str:
        return (
            f"gmm_noise={self.gmm_noise_enabled}, "
            f"offset={self.noise_offset}, "
            f"scheduler={self.noise_scheduler.schedule_type.value}"
        )
