"""GMS-Diffusion 适配器 - 连接 GMM 求解器与扩散模型

实现适配器模式 (Adapter Pattern)，作为 GMS 模块和标准扩散模型之间的桥梁：
- 将 GMM 参数转换为扩散过程可用的噪声调度
- 协调采样控制器与扩散时间步
- 管理不同模块之间的数据流

与标准扩散模型的对比说明:
    标准DDPM使用固定的线性或余弦噪声调度（β_t）。
    GMS通过GMM参数动态生成噪声分布，使得每个时间步的噪声
    统计特性可以自适应调整，从而在保持生成质量的同时
    提供更灵活的噪声控制。

Example:
    >>> import torch
    >>> from gms.gmm_optimization.gmm_parameters import GMMParameters
    >>> from gms.diffusion_integration.adapter import GMSDiffusionAdapter
    >>>
    >>> gmm_params = GMMParameters(
    ...     weight=0.6,
    ...     mean1=torch.tensor([0.0, 0.0]),
    ...     mean2=torch.tensor([1.0, 1.0]),
    ...     variance1=torch.tensor([0.5, 0.5]),
    ...     variance2=torch.tensor([2.0, 2.0])
    ... )
    >>>
    >>> adapter = GMSDiffusionAdapter(num_diffusion_steps=1000)
    >>> schedule = adapter.adapt_gmm_to_diffusion(gmm_params)
    >>> print(f"噪声调度形状: {schedule.means.shape}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Tuple
import math
import json
import torch
import numpy as np

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None

from gms.gmm_optimization.gmm_parameters import GMMParameters
from gms.sampling.time_step_controller import TimeStepController


@dataclass
class NoiseSchedule:
    """噪声调度数据类

    存储每个扩散时间步的噪声统计特性，这些特性来源于GMM的采样结果。
    支持线性插值和非均匀时间步，提供序列化支持。

    数学定义:
        在时间步 t，噪声 ε_t ~ N(μ_t, σ_t²)
        其中 μ_t 和 σ_t 由 GMM 参数在该时间步的混合分布决定

    Attributes:
        means: 每个时间步的噪声均值，形状 (T,) 或 (T, d)
        variances: 每个时间步的噪声方差，形状 (T,) 或 (T, d)
        stds: 每个时间步的噪声标准差（= sqrt(variances)）
        num_steps: 总时间步数 T
        dimensionality: 特征维度 d（标量噪声时为1）
        source_gmm_params: 来源GMM参数的字典表示（可选）
        interpolation_mode: 插值模式 ('linear', 'none')

    Example:
        >>> schedule = NoiseSchedule(
        ...     means=torch.zeros(100),
        ...     variances=torch.linspace(0.0001, 0.02, 100)
        ... )
        >>> # 获取第50步的噪声参数
        >>> mu_50, sigma_50 = schedule.get_step(50)
    """

    means: torch.Tensor
    variances: torch.Tensor
    source_gmm_params: Optional[Dict[str, Any]] = None
    interpolation_mode: str = "linear"

    def __post_init__(self):
        """初始化后验证"""
        if self.means.shape != self.variances.shape:
            raise ValueError(
                f"means 和 variances 形状不一致: "
                f"means{tuple(self.means.shape)} vs variances{tuple(self.variances.shape)}"
            )

        if self.interpolation_mode not in ["linear", "none"]:
            raise ValueError(f"不支持的插值模式: {self.interpolation_mode}")

        self._stds = torch.sqrt(torch.clamp(self.variances, min=1e-10))

    @property
    def stds(self) -> torch.Tensor:
        """获取标准差张量"""
        return self._stds

    @property
    def num_steps(self) -> int:
        """获取总时间步数"""
        return self.means.shape[0]

    @property
    def dimensionality(self) -> int:
        """获取特征维度"""
        if self.means.dim() == 1:
            return 1
        return self.means.shape[-1]

    def get_step(self, step_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定时间步的噪声参数

        Args:
            step_idx: 时间步索引 [0, num_steps-1]

        Returns:
            (mean, variance) 元组

        Raises:
            IndexError: 如果 step_idx 超出范围
        """
        if not 0 <= step_idx < self.num_steps:
            raise IndexError(
                f"step_idx={step_idx} 超出范围 [0, {self.num_steps-1}]"
            )
        return self.means[step_idx], self.variances[step_idx]

    def get_range(
        self,
        start: int,
        end: int,
        step: int = 1
    ) -> "NoiseSchedule":
        """获取子范围的噪声调度

        Args:
            start: 起始索引（包含）
            end: 结束索引（不包含）
            step: 步长

        Returns:
            新的 NoiseSchedule 实例
        """
        return NoiseSchedule(
            means=self.means[start:end:step].clone(),
            variances=self.variances[start:end:step].clone(),
            source_gmm_params=self.source_gmm_params,
            interpolation_mode=self.interpolation_mode,
        )

    def interpolate(
        self,
        target_steps: int,
        mode: Optional[str] = None
    ) -> "NoiseSchedule":
        """将噪声调度插值到新的时间步数

        支持线性插值以匹配不同的扩散步数配置。

        Args:
            target_steps: 目标时间步数
            mode: 插值模式（覆盖实例默认值）

        Returns:
            插值后的新 NoiseSchedule
        """
        interp_mode = mode or self.interpolation_mode

        if target_steps == self.num_steps:
            return NoiseSchedule(
                means=self.means.clone(),
                variances=self.variances.clone(),
                source_gmm_params=self.source_gmm_params,
                interpolation_mode=interp_mode,
            )

        old_indices = torch.linspace(0, self.num_steps - 1, self.num_steps)
        new_indices = torch.linspace(0, self.num_steps - 1, target_steps)

        new_means = self._interpolate_1d(old_indices, new_indices, self.means)
        new_variances = self._interpolate_1d(old_indices, new_indices, self.variances)
        new_variances = torch.clamp(new_variances, min=1e-10)

        logger.debug(
            f"噪声调度插值: {self.num_steps} -> {target_steps}, "
            f"mode={interp_mode}"
        )

        return NoiseSchedule(
            means=new_means,
            variances=new_variances,
            source_gmm_params=self.source_gmm_params,
            interpolation_mode=interp_mode,
        )

    @staticmethod
    def _interpolate_1d(
        old_x: torch.Tensor,
        new_x: torch.Tensor,
        old_y: torch.Tensor
    ) -> torch.Tensor:
        """一维线性插值的向量化实现

        Args:
            old_x: 原始x坐标
            new_x: 目标x坐标
            old_y: 原始y值

        Returns:
            插值后的y值
        """
        old_x_np = old_x.cpu().numpy()
        new_x_np = new_x.cpu().numpy()
        old_y_np = (old_y.float() if old_y.dim() == 1 else old_y.reshape(-1).float()).cpu().numpy()
        np_interp = np.interp(new_x_np, old_x_np, old_y_np)
        torch_interp = torch.tensor(np_interp, dtype=old_y.dtype, device=old_y.device)
        if old_y.dim() > 1:
            torch_interp = torch_interp.reshape(-1, old_y.shape[-1])
        return torch_interp

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典

        Returns:
            包含所有数据的字典
        """
        return {
            'means': self.means.cpu().numpy().tolist(),
            'variances': self.variances.cpu().numpy().tolist(),
            'num_steps': self.num_steps,
            'dimensionality': self.dimensionality,
            'interpolation_mode': self.interpolation_mode,
            'source_gmm_params': self.source_gmm_params,
        }

    def to_json(self, filepath: str) -> None:
        """保存为JSON文件

        Args:
            filepath: 输出文件路径
        """
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"噪声调度已保存到 {filepath}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NoiseSchedule":
        """从字典反序列化

        Args:
            data: 序列化的字典

        Returns:
            NoiseSchedule 实例
        """
        means = torch.tensor(data['means'])
        variances = torch.tensor(data['variances'])

        return cls(
            means=means,
            variances=variances,
            source_gmm_params=data.get('source_gmm_params'),
            interpolation_mode=data.get('interpolation_mode', 'linear'),
        )

    @classmethod
    def from_json(cls, filepath: str) -> "NoiseSchedule":
        """从JSON文件加载

        Args:
            filepath: JSON 文件路径

        Returns:
            NoiseSchedule 实例
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to(self, device: Union[str, torch.device]) -> "NoiseSchedule":
        """移动到指定设备

        Args:
            device: 目标设备

        Returns:
            新的 NoiseSchedule 实例
        """
        return NoiseSchedule(
            means=self.means.to(device),
            variances=self.variances.to(device),
            source_gmm_params=self.source_gmm_params,
            interpolation_mode=self.interpolation_mode,
        )

    def __repr__(self) -> str:
        return (
            f"NoiseSchedule(steps={self.num_steps}, "
            f"d={self.dimensionality}, "
            f"mean_range=[{self.means.min():.4f}, {self.means.max():.4f}], "
            f"var_range=[{self.variances.min():.6f}, {self.variances.max():.4f}])"
        )


class AdaptationStrategy(Enum):
    """GMM 到扩散过程的适配策略

    VARIANCE_WEIGHTED: 使用加权方差混合作为噪声方差
    MOMENT_MATCHING: 通过矩匹配确定噪声参数
    ENTROPY_BASED: 基于熵的噪声尺度调整
    CUSTOM: 自定义策略（通过回调函数）
    """

    VARIANCE_WEIGHTED = "variance_weighted"
    MOMENT_MATCHING = "moment_matching"
    ENTROPY_BASED = "entropy_based"
    CUSTOM = "custom"


class GMSDiffusionAdapter:
    """GMS 与扩散模型的适配器

    作为 GMM 求解器和标准扩散模型之间的桥梁，负责：
    1. 将 GMM 参数转换为扩散过程可用的噪声调度
    2. 对齐 GMM 时间步与扩散时间步
    3. 转换 GMM 采样结果到扩散空间

    设计原则:
        遵循适配器模式 (Adapter Pattern)，将 GMS 的接口转换为
        扩散模型期望的标准接口形式，使得现有 DDPM/DDIM 等
        框架无需修改即可接入 GMS。

    Attributes:
        num_diffusion_steps: 扩散过程总时间步数 T
        strategy: 适配策略
        time_step_controller: 可选的时间步控制器引用
        _noise_schedule_cache: 缓存的噪声调度

    Example:
        >>> adapter = GMSDiffusionAdapter(
        ...     num_diffusion_steps=1000,
        ...     strategy=AdaptationStrategy.VARIANCE_WEIGHTED
        ... )
        >>> schedule = adapter.adapt_gmm_to_diffusion(gmm_params)
        >>> alignment = adapter.align_time_steps(
        ...     gmm_timesteps=[0, 100, 500, 999],
        ...     diffusion_timesteps=list(range(1000))
        ... )
    """

    def __init__(
        self,
        num_diffusion_steps: int = 1000,
        strategy: AdaptationStrategy = AdaptationStrategy.VARIANCE_WEIGHTED,
        time_step_controller: Optional[TimeStepController] = None,
        custom_adapter_fn: Optional[callable] = None,
        clamp_variance: Tuple[float, float] = (1e-8, 10.0),
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """初始化 GMS-Diffusion 适配器

        Args:
            num_diffusion_steps: 扩散过程总时间步数
            strategy: GMM到扩散的适配策略
            time_step_controller: 可选的采样时间步控制器
            custom_adapter_fn: 当 strategy=CUSTOM 时的自定义转换函数
                                签名: fn(gmm_params, t) -> (mean, variance)
            clamp_variance: 方差裁剪范围 (min, max)
            device: 计算设备
            dtype: 数据类型

        Raises:
            ValueError: 如果参数不合法
        """
        if num_diffusion_steps <= 0:
            raise ValueError(f"num_diffusion_steps 必须为正整数，当前值: {num_diffusion_steps}")

        if clamp_variance[0] <= 0 or clamp_variance[1] <= clamp_variance[0]:
            raise ValueError(
                f"clamp_variance 不合法: {clamp_variance}, "
                f"需要 min>0 且 max>min"
            )

        self.num_diffusion_steps = num_diffusion_steps
        self.strategy = strategy
        self.time_step_controller = time_step_controller
        self.custom_adapter_fn = custom_adapter_fn
        self.clamp_variance = clamp_variance
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype

        self._noise_schedule_cache: Optional[NoiseSchedule] = None

        logger.info(
            f"GMSDiffusionAdapter 初始化完成: steps={num_diffusion_steps}, "
            f"strategy={strategy.value}, device={device}"
        )

    def adapt_gmm_to_diffusion(
        self,
        gmm_params: GMMParameters,
        force_recompute: bool = False,
    ) -> NoiseSchedule:
        """将 GMM 参数转换为扩散噪声调度

        核心转换逻辑：根据选定的策略，将双分量 GMM 的统计特性
        映射为每个扩散时间步上的噪声分布参数。

        不同策略的工作原理:

        VARIANCE_WEIGHTED (默认):
            σ_t² = w·σ₁² + (1-w)·σ₂² + β(t) 的缩放
            其中 β(t) 是随时间递增的基准噪声

        MOMENT_MATCHING:
            匹配 GMM 混合分布的前两阶矩到高斯噪声

        ENTROPY_BASED:
            根据混合熵随时间的变化调整噪声强度

        Args:
            gmm_params: GMMParameters 实例
            force_recompute: 是否强制重新计算（忽略缓存）

        Returns:
            NoiseSchedule 实例，包含每步的噪声均值和方差

        Raises:
            TypeError: 如果 gmm_params 不是 GMMParameters 实例
            ValueError: 如果自定义策略缺少回调函数
        """
        if not isinstance(gmm_params, GMMParameters):
            raise TypeError(
                f"gmm_params 必须是 GMMParameters 实例，"
                f"得到 {type(gmm_params).__name__}"
            )

        if self._noise_schedule_cache is not None and not force_recompute:
            logger.debug("使用缓存的噪声调度")
            return self._noise_schedule_cache

        gmm_params_device = gmm_params.to_device(self.device)
        T = self.num_diffusion_steps
        d = gmm_params_device.dimensionality

        if self.strategy == AdaptationStrategy.VARIANCE_WEIGHTED:
            means, variances = self._adapt_variance_weighted(gmm_params_device, T, d)
        elif self.strategy == AdaptationStrategy.MOMENT_MATCHING:
            means, variances = self._adapt_moment_matching(gmm_params_device, T, d)
        elif self.strategy == AdaptationStrategy.ENTROPY_BASED:
            means, variances = self._adapt_entropy_based(gmm_params_device, T, d)
        elif self.strategy == AdaptationStrategy.CUSTOM:
            if self.custom_adapter_fn is None:
                raise ValueError("CUSTOM 策略需要提供 custom_adapter_fn")
            means, variances = self._adapt_custom(gmm_params_device, T, d)
        else:
            raise ValueError(f"不支持的适配策略: {self.strategy}")

        variances = variances.clamp(self.clamp_variance[0], self.clamp_variance[1])

        source_dict = {
            'weight': gmm_params.weight,
            'dimensionality': d,
            'is_diagonal': gmm_params_device.is_diagonal,
        }

        schedule = NoiseSchedule(
            means=means,
            variances=variances,
            source_gmm_params=source_dict,
            interpolation_mode="linear",
        )

        self._noise_schedule_cache = schedule

        logger.info(
            f"GMM->Diffusion 适配完成: strategy={self.strategy.value}, "
            f"steps={T}, variance_range=[{variances.min():.6f}, {variances.max():.4f}]"
        )

        return schedule

    def _adapt_variance_weighted(
        self,
        params: GMMParameters,
        T: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于加权方差的适配策略

        计算公式:
            σ_t²(t) = base_β(t) · [w·Σ₁ + (1-w)·Σ₂]
            μ_t(t) = w·μ₁ + (1-w)·μ₂ · scale(t)

        其中 base_β(t) 从小到大递增（模拟扩散过程中噪声增强）。
        """
        t_norm = torch.linspace(0, 1, T, device=self.device, dtype=self.dtype)
        base_beta = 1e-4 + (0.02 - 1e-4) * t_norm ** 2

        w = params.weight
        w2 = params.weight2

        if params.is_diagonal:
            mixed_var = w * params.variance1 + w2 * params.variance2
            mixed_mean = w * params.mean1 + w2 * params.mean2
        else:
            mixed_var = w * params.variance1 + w2 * params.variance2
            mixed_mean = w * params.mean1 + w2 * params.mean2

        scale_factor = 0.1 * (1 - t_norm.unsqueeze(-1))

        variances = base_beta.unsqueeze(-1) * (1.0 + mixed_var.unsqueeze(0))
        means = mixed_mean.unsqueeze(0) * scale_factor

        return means, variances

    def _adapt_moment_matching(
        self,
        params: GMMParameters,
        T: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于矩匹配的适配策略

        将 GMM 混合分布的前两阶矩匹配到等价的高斯分布，
        并根据时间步进行调制。

        混合分布矩:
            E[X] = w·μ₁ + (1-w)·μ₂
            Var[X] = w(Σ₁+μ₁²) + (1-w)(Σ₂+μ₂²) - E[X]²
        """
        t_norm = torch.linspace(0, 1, T, device=self.device, dtype=self.dtype)
        beta_schedule = torch.cos(t_norm * math.pi / 2) ** 2
        beta_schedule = 1e-4 + (0.02 - 1e-4) * (1 - beta_schedule)

        w = params.weight
        w2 = params.weight2
        mu1, mu2 = params.mean1, params.mean2
        s1, s2 = params.variance1, params.variance2

        mixture_mean = w * mu1 + w2 * mu2

        if params.is_diagonal:
            mixture_var = (
                w * (s1 + mu1 ** 2) +
                w2 * (s2 + mu2 ** 2) -
                mixture_mean ** 2
            )
        else:
            diff1 = (mu1 - mixture_mean).unsqueeze(1)
            diff2 = (mu2 - mixture_mean).unsqueeze(1)
            mixture_var = (
                w * (s1 + diff1 @ diff1.T) +
                w2 * (s2 + diff2 @ diff2.T)
            )

        variances = beta_schedule.unsqueeze(-1) * (mixture_var.unsqueeze(0) + 1.0)
        means = mixture_mean.unsqueeze(0) * (1 - t_norm).unsqueeze(-1) * 0.01

        return means, variances

    def _adapt_entropy_based(
        self,
        params: GMMParameters,
        T: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于熵的适配策略

        利用 GMM 的混合熵来调节噪声强度。
        高熵区域对应更强的噪声注入。

        近似熵计算:
            H ≈ -Σ_k w_k log(w_k) + Σ_k w_k H(N(μ_k, Σ_k))
        """
        t_norm = torch.linspace(0, 1, T, device=self.device, dtype=self.dtype)
        w = params.weight
        w2 = params.weight2

        entropy_weight = -(w * math.log(w + 1e-10) + w2 * math.log(w2 + 1e-10))

        if params.is_diagonal:
            entropy_comp1 = 0.5 * d * (1 + math.log(2 * math.pi)) + \
                           0.5 * torch.log(params.variance1).sum()
            entropy_comp2 = 0.5 * d * (1 + math.log(2 * math.pi)) + \
                           0.5 * torch.log(params.variance2).sum()
            total_entropy = entropy_weight + w * entropy_comp1.item() + w2 * entropy_comp2.item()
        else:
            total_entropy = entropy_weight + d * (1 + math.log(2 * math.pi)) / 2

        entropy_scale = total_entropy / max(d * (1 + math.log(2 * math.pi)) / 2, 1e-10)
        entropy_scale = min(max(entropy_scale, 0.5), 3.0)

        base_noise = 1e-4 + (0.02 * entropy_scale - 1e-4) * t_norm ** 2

        variances = base_noise.unsqueeze(-1).expand(T, d).clone()
        means = torch.zeros(T, d, device=self.device, dtype=self.dtype)

        return means, variances

    def _adapt_custom(
        self,
        params: GMMParameters,
        T: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """自定义适配策略

        使用用户提供的回调函数进行转换。

        回调函数签名:
            fn(params: GMMParameters, t: Tensor) -> Tuple[Tensor, Tensor]
            返回 (means, variances)，形状均为 (T,) 或 (T, d)
        """
        timesteps = torch.linspace(0, 1, T, device=self.device, dtype=self.dtype)

        try:
            means, variances = self.custom_adapter_fn(params, timesteps)
        except Exception as e:
            logger.error(f"自定义适配函数执行失败: {e}")
            raise RuntimeError(f"custom_adapter_fn 执行错误: {e}") from e

        if means.shape[0] != T:
            raise ValueError(
                f"自定义函数返回的均值形状不正确: "
                f"期望第一维为 {T}, 得到 {means.shape[0]}"
            )
        if variances.shape != means.shape:
            raise ValueError(
                f"自定义函数返回的方差形状与均值不匹配: "
                f"variances{tuple(variances.shape)} vs means{tuple(means.shape)}"
            )

        return means.to(self.device), variances.to(self.device)

    def align_time_steps(
        self,
        gmm_timesteps: List[int],
        diffusion_timesteps: List[int],
        method: str = "nearest",
    ) -> Dict[int, int]:
        """对齐 GMM 时间步与扩散时间步

        建立两个时间轴之间的映射关系，用于协调采样控制器
        和扩散过程的时间同步。

        Args:
            gmm_timesteps: GMM 采样的时间步列表
            diffusion_timesteps: 扩散过程的时间步列表
            method: 对齐方法 ('nearest', 'linear', 'uniform')

        Returns:
            字典 {gmm_step: diffusion_step}

        Example:
            >>> alignment = adapter.align_time_steps(
            ...     gmm_timesteps=[0, 50, 100],
            ...     diffusion_timesteps=list(range(1000)),
            ...     method='linear'
            ... )
            >>> # alignment[50] 返回对应的扩散时间步
        """
        if not gmm_timesteps:
            logger.warning("gmm_timesteps 为空，返回空映射")
            return {}

        if not diffusion_timesteps:
            raise ValueError("diffusion_timesteps 不能为空")

        gmm_arr = np.array(gmm_timesteps, dtype=np.float64)
        diff_arr = np.array(diffusion_timesteps, dtype=np.float64)

        gmm_min, gmm_max = gmm_arr.min(), gmm_arr.max()
        diff_min, diff_max = diff_arr.min(), diff_arr.max()

        gmm_range = gmm_max - gmm_min if gmm_max > gmm_min else 1.0
        diff_range = diff_max - diff_min if diff_max > diff_min else 1.0

        alignment: Dict[int, int] = {}

        if method == "nearest":
            for gt in gmm_timesteps:
                normalized_gt = (gt - gmm_min) / gmm_range
                target_val = diff_min + normalized_gt * diff_range
                nearest_idx = int(np.argmin(np.abs(diff_arr - target_val)))
                alignment[gt] = int(diffusion_timesteps[nearest_idx])

        elif method == "linear":
            for gt in gmm_timesteps:
                normalized_gt = (gt - gmm_min) / gmm_range
                target_val = diff_min + normalized_gt * diff_range
                aligned_val = int(np.round(target_val))
                aligned_val = max(diff_min, min(diff_max, aligned_val))
                alignment[gt] = int(aligned_val)

        elif method == "uniform":
            n_gmm = len(gmm_timesteps)
            n_diff = len(diffusion_timesteps)
            for i, gt in enumerate(gmm_timesteps):
                ratio = i / max(n_gmm - 1, 1)
                idx = int(ratio * (n_diff - 1))
                alignment[gt] = diffusion_timesteps[idx]

        else:
            raise ValueError(f"不支持的对齐方法: {method}")

        if self.time_step_controller is not None:
            logger.debug(
                f"时间步对齐完成: {len(alignment)} 个映射, "
                f"method={method}, controller已连接"
            )
        else:
            logger.debug(
                f"时间步对齐完成: {len(alignment)} 个映射, method={method}"
            )

        return alignment

    def transform_samples(
        self,
        gmm_samples: torch.Tensor,
        timestep: int,
        noise_schedule: Optional[NoiseSchedule] = None,
    ) -> torch.Tensor:
        """将 GMM 采样结果转换到扩散空间

        将从 GMM 中采样的样本转换为扩散模型可用的格式，
        包括维度对齐、尺度变换和时间步特定的偏移。

        数学变换:
            x_diffused = α_t · x_gmm + σ_t · ε
            其中 ε ~ N(0, I), α_t = √ᾱ_t, σ_t 来自噪声调度

        Args:
            gmm_samples: GMM 采样结果，形状 (n, d) 或 (d,)
            timestep: 当前扩散时间步
            noise_schedule: 可选的噪声调度（未提供则使用缓存）

        Returns:
            转换后的张量，形状与输入相同

        Raises:
            IndexError: 如果 timestep 超出范围
        """
        schedule = noise_schedule or self._noise_schedule_cache
        if schedule is None:
            raise RuntimeError(
                "没有可用的噪声调度。请先调用 adapt_gmm_to_diffusion() "
                "或显式传入 noise_schedule"
            )

        if not 0 <= timestep < schedule.num_steps:
            raise IndexError(
                f"timestep={timestep} 超出范围 [0, {schedule.num_steps-1}]"
            )

        original_shape = gmm_samples.shape
        if gmm_samples.dim() == 1:
            gmm_samples = gmm_samples.unsqueeze(0)

        mean_t, var_t = schedule.get_step(timestep)
        std_t = torch.sqrt(var_t.clamp(min=1e-10))

        alpha_t = 1.0 - (timestep / schedule.num_steps)
        alpha_t = float(alpha_t)

        noise = torch.randn_like(gmm_samples)

        transformed = alpha_t * gmm_samples + std_t * noise

        if len(original_shape) == 1:
            transformed = transformed.squeeze(0)

        logger.debug(
            f"样本转换: timestep={timestep}, "
            f"α_t={alpha_t:.4f}, σ_t={std_t.mean():.6f}"
        )

        return transformed

    def compute_alpha_schedule(
        self,
        noise_schedule: Optional[NoiseSchedule] = None,
    ) -> Dict[str, torch.Tensor]:
        """计算完整的 α/β 调度表

        从噪声方差推导标准的扩散调度参数:
            β_t = σ_t² / (1 - ᾱ_{t-1})  （近似）
            α_t = 1 - β_t
            ᾱ_t = ∏_{s=1}^{t} α_s

        Args:
            noise_schedule: 噪声调度（可选）

        Returns:
            包含 'beta', 'alpha', 'alpha_cumprod', 'sqrt_alpha_cumprod',
            'sqrt_one_minus_alpha_cumprod' 的字典
        """
        schedule = noise_schedule or self._noise_schedule_cache
        if schedule is None:
            raise RuntimeError("没有可用的噪声调度")

        T = schedule.num_steps
        betas = schedule.variances.clone()

        if betas.dim() > 1:
            betas = betas.mean(dim=-1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod = torch.cat([torch.ones(1, device=schedule.means.device), alphas_cumprod[:-1]])

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        result = {
            'betas': betas,
            'alphas': alphas,
            'alphas_cumprod': alphas_cumprod,
            'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
            'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        }

        logger.debug(f"α/β 调度计算完成: steps={T}")
        return result

    def clear_cache(self) -> None:
        """清除缓存的噪声调度"""
        self._noise_schedule_cache = None
        logger.debug("噪声调度缓存已清除")

    def export_state(self) -> Dict[str, Any]:
        """导出适配器状态

        Returns:
            包含配置和状态的字典
        """
        state = {
            'num_diffusion_steps': self.num_diffusion_steps,
            'strategy': self.strategy.value,
            'clamp_variance': list(self.clamp_variance),
            'device': str(self.device),
            'dtype': str(self.dtype),
        }

        if self._noise_schedule_cache is not None:
            state['cached_schedule'] = self._noise_schedule_cache.to_dict()

        return state

    def to(self, device: Union[str, torch.device]) -> "GMSDiffusionAdapter":
        """移动适配器到指定设备

        Args:
            device: 目标设备

        Returns:
            self（支持链式调用）
        """
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        if self._noise_schedule_cache is not None:
            self._noise_schedule_cache = self._noise_schedule_cache.to(device)
        return self

    def __repr__(self) -> str:
        cached = "有缓存" if self._noise_schedule_cache is not None else "无缓存"
        return (
            f"GMSDiffusionAdapter(steps={self.num_diffusion_steps}, "
            f"strategy={self.strategy.value}, cache={cached})"
        )
