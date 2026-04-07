"""GMS 反向扩散过程 - 梯度流保持

实现扩散模型反向去噪过程 (Backward Process / Denoising Process) 的 GMS 集成版本。
标准 DDPM 的反向过程定义为:
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²·I)

GMS 版本在反向过程中注入 GMM 条件信息，引导去噪网络的预测目标，
同时确保所有操作可微分以保持梯度流的完整性。

与标准扩散模型的对比:
    标准: 去噪网络仅依赖 (x_t, t)
    GMS:  去噪网络额外接收 GMS 条件信息，预测受 GMS 引导

Example:
    >>> import torch
    >>> import torch.nn as nn
    >>> from gms.diffusion_integration.backward_process import (
    ...     GMSBackwardProcess, DenoisingNetworkWrapper
    ... )
    >>>
    >>> # 创建模拟的去噪网络
    >>> dummy_model = nn.Sequential(nn.Linear(10, 10))
    >>> wrapped = DenoisingNetworkWrapper(dummy_model, condition_dim=4)
    >>>
    >>> scheduler = NoiseScheduler(num_steps=1000)
    >>> backward = GMSBackwardProcess(scheduler)
    >>>
    >>> x_t = torch.randn(2, 10)
    >>> t = torch.tensor([500, 500])
    >>> model_output = wrapped(x_t, t, gms_condition=torch.randn(2, 4))
    >>> x_prev = backward.sample_step(x_t, t, model_output)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Tuple, Callable
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None


from .forward_process import NoiseScheduler


class PredictionType(Enum):
    """去噪网络的预测类型

    EPSILON: 预测噪声 ε（原始DDPM）
    SAMPLE: 直接预测 x_0
    VELOCITY: 预测速度 v = α_t·ε - σ_t·x_0（v-prediction）
    V: v-prediction 的别名
    """

    EPSILON = "epsilon"
    SAMPLE = "sample"
    VELOCITY = "velocity"
    V = "v"


@dataclass
class BackwardConfig:
    """反向过程配置

    Attributes:
        prediction_type: 预测类型
        clip_denoised: 是否裁剪去噪输出
        clip_range: 裁剪范围
        gradient_clip: 梯度裁剪值（0 表示不裁剪）
        clamp_samples: 是否限制采样范围
        sample_clamp_range: 采样范围限制
    """

    prediction_type: str = "epsilon"
    clip_denoised: bool = True
    clip_range: Tuple[float, float] = (-10.0, 10.0)
    gradient_clip: float = 1.0
    clamp_samples: bool = True
    sample_clamp_range: Tuple[float, float] = (-10.0, 10.0)


class DenoisingNetworkWrapper(nn.Module):
    """去噪网络包装器

    包装标准的去噪网络（如 UNet），注入 GMS 条件信息。
    处理网络输出并转换为均值/方差格式。

    设计模式:
        使用装饰器/包装器模式 (Wrapper Pattern)，在不修改原有网络结构
        的前提下扩展其功能。

    Attributes:
        base_network: 底层去噪网络
        condition_dim: 条件向量维度
        condition_projection: 条件投影层
        output_dim: 输出维度

    Example:
        >>> unet = MyUNet(in_channels=3, out_channels=6)
        >>> wrapper = DenoisingNetworkWrapper(
        ...     unet,
        ...     condition_dim=8,
        ...     output_dim=3
        ... )
        >>> output = wrapper(
        ...     x_t, t,
        ...     gms_condition=condition_vector
        ... )
    """

    def __init__(
        self,
        base_network: nn.Module,
        condition_dim: int = 0,
        output_dim: Optional[int] = None,
        condition_injection: str = "concat",
        use_condition_encoder: bool = False,
        condition_hidden_dim: int = 64,
    ):
        """初始化去噪网络包装器

        Args:
            base_network: 底层去噪网络模块
            condition_dim: GMS 条件向量的维度
            output_dim: 网络输出的有效维度（用于分割 mean/variance）
            condition_injection: 条件注入方式 ('concat', 'adagn', 'film', 'cross_attn')
            use_condition_encoder: 是否使用条件编码器
            condition_hidden_dim: 条件编码器的隐藏维度

        Raises:
            ValueError: 如果条件注入方式不支持
        """
        super().__init__()

        self.base_network = base_network
        self.condition_dim = condition_dim
        self.output_dim = output_dim
        self.condition_injection = condition_injection
        self.use_condition_encoder = use_condition_encoder

        valid_injections = ["concat", "adagn", "film", "cross_attn", "none"]
        if condition_injection not in valid_injections:
            raise ValueError(
                f"condition_injection 必须是 {valid_injections} 之一，"
                f"得到 {condition_injection}"
            )

        if condition_dim > 0 and condition_injection != "none":
            if use_condition_encoder:
                self.condition_encoder = nn.Sequential(
                    nn.Linear(condition_dim, condition_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(condition_hidden_dim, condition_hidden_dim),
                    nn.SiLU(),
                    nn.Linear(condition_hidden_dim, condition_hidden_dim),
                )
                encoded_dim = condition_hidden_dim
            else:
                self.condition_encoder = None
                encoded_dim = condition_dim

            if condition_injection == "concat":
                self.condition_projection = nn.Identity()
            elif condition_injection == "film":
                self.condition_projection = nn.Sequential(
                    nn.Linear(encoded_dim, output_dim * 2),
                )
            elif condition_injection == "adagn":
                self.condition_projection = nn.Sequential(
                    nn.Linear(encoded_dim, output_dim * 2),
                )

        if logger:
            logger.info(
                f"DenoisingNetworkWrapper 初始化完成: "
                f"injection={condition_injection}, cond_dim={condition_dim}"
            )

    def forward(
        self,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        gms_condition: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """前向传播

        将 GMS 条件信息注入后调用底层网络。

        Args:
            x_t: 加噪输入，形状 (B, C, H, W) 或 (B, D)
            timestep: 时间步，形状 (B,)
            gms_condition: GMS 条件向量，形状 (B, condition_dim)
            **kwargs: 传递给底层网络的额外参数

        Returns:
            网络输出张量
        """
        if gms_condition is not None and self.condition_dim > 0:
            if self.use_condition_encoder and self.condition_encoder is not None:
                encoded_cond = self.condition_encoder(gms_condition)
            else:
                encoded_cond = gms_condition

            if self.condition_injection == "concat":
                if x_t.dim() == 4:
                    cond_expanded = encoded_cond.unsqueeze(-1).unsqueeze(-1).expand_as(
                        x_t[:, :self.condition_dim, :, :]
                    )
                    x_t_augmented = torch.cat([x_t, cond_expanded], dim=1)
                else:
                    x_t_augmented = torch.cat([x_t, encoded_cond], dim=-1)

                output = self._call_base_network(x_t_augmented, timestep, **kwargs)
            elif self.condition_injection in ("film", "adagn"):
                raw_output = self._call_base_network(x_t, timestep, **kwargs)
                modulation_params = self.condition_projection(encoded_cond)

                if raw_output.dim() == 4:
                    mod_shape = (modulation_params.shape[0], -1, 1, 1)
                else:
                    mod_shape = (modulation_params.shape[0], -1)

                gamma, beta = modulation_params.reshape(mod_shape).chunk(2, dim=1)
                output = (1 + gamma) * raw_output + beta
            elif self.condition_injection == "cross_attn":
                raw_output = self._call_base_network(x_t, timestep, **kwargs)
                output = self._cross_attention_inject(raw_output, encoded_cond)
            else:
                output = self._call_base_network(x_t, timestep, **kwargs)
        else:
            output = self._call_base_network(x_t, timestep, **kwargs)

        return output

    def _call_base_network(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """调用底层网络，处理不同的参数签名

        Args:
            x: 输入张量
            timestep: 时间步
            **kwargs: 额外参数

        Returns:
            网络输出
        """
        import inspect
        sig = inspect.signature(self.base_network.forward)
        params = list(sig.parameters.keys())

        if 't' in params or 'timestep' in params:
            return self.base_network(x, timestep, **kwargs)
        else:
            return self.base_network(x)

    def _cross_attention_inject(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """交叉注意力注入条件信息

        Args:
            features: 特征图
            condition: 条件向量

        Returns:
            注入后的特征
        """
        B = features.shape[0]
        C = features.shape[1]

        query = features.reshape(B, C, -1).transpose(1, 2)
        key = condition.unsqueeze(1)
        value = condition.unsqueeze(1)

        scale = C ** -0.5
        attn_scores = torch.bmm(query, key.transpose(1, 2)) * scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, value)

        injected = features + attn_output.transpose(1, 2).reshape_as(features)
        return injected

    def split_output_to_mean_var(
        self,
        output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将网络输出分割为均值和方差

        Args:
            output: 网络输出张量

        Returns:
            (mean, log_variance) 元组
        """
        if self.output_dim is None:
            return output, torch.zeros_like(output)

        if output.shape[-1] == self.output_dim * 2:
            mean, log_var = output.chunk(2, dim=-1)
        elif output.shape[1] == self.output_dim * 2:
            mean, log_var = output.chunk(2, dim=1)
        else:
            mean = output
            log_var = torch.zeros_like(output)

        return mean, log_var.clamp(min=-20, max=20)


class GMSBackwardProcess(nn.Module):
    """GMS 集成的反向去噪过程

    实现 p_θ(x_{t-1} | x_t) 的 GMS 版本，
    在去噪过程中注入 GMM 引导信号。

    核心公式 (epsilon prediction):
        μ_θ = (1/√α_t) · [x_t - (β_t / √(1-ᾱ_t)) · ε_θ(x_t, t)]
        x_{t-1} = μ_θ + σ_t · z,  z ~ N(0, I) 当 t > 1

    GMS 增强:
        - 去噪网络的预测受 GMS 条件调制
        - 支持多种预测类型 (epsilon/sample/v-prediction)
        - 完整的梯度追踪和可选的梯度裁剪

    Attributes:
        noise_scheduler: 噪声调度器
        config: 反向过程配置
        prediction_type: 当前使用的预测类型

    Example:
        >>> scheduler = NoiseScheduler(num_steps=1000)
        >>> backward = GMSBackwardProcess(scheduler)
        >>>
        >>> x_T = torch.randn(4, 3, 32, 32)
        >>> for t in reversed(range(1000)):
        ...     t_batch = torch.full((4,), t, dtype=torch.long)
        ...     model_out = denoise_model(x_T, t_batch)
        ...     x_T = backward.sample_step(x_T, t_batch, model_out)
        >>> # x_T 现在是生成的样本
    """

    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        prediction_type: Union[str, PredictionType] = PredictionType.EPSILON,
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
        gradient_clip_value: float = 1.0,
        enable_gradient_norm_tracking: bool = False,
    ):
        """初始化 GMS 反向过程

        Args:
            noise_scheduler: NoiseScheduler 实例
            prediction_type: 预测类型
            clip_denoised: 是否裁剪去噪结果
            clip_range: 裁剪范围
            gradient_clip_value: 梯度裁剪阈值（0表示不裁剪）
            enable_gradient_norm_tracking: 是否启用梯度范数跟踪

        Raises:
            ValueError: 如果参数不合法
        """
        super().__init__()

        if isinstance(prediction_type, str):
            prediction_type = PredictionType(prediction_type)

        self.noise_scheduler = noise_scheduler
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
        self.gradient_clip_value = gradient_clip_value
        self.enable_gradient_norm_tracking = enable_gradient_norm_tracking

        self._gradient_norms: List[float] = []

        if logger:
            logger.info(
                f"GMSBackwardProcess 初始化完成: "
                f"pred={prediction_type.value}, "
                f"clip={clip_denoised}, grad_clip={gradient_clip_value}"
            )

    @property
    def last_gradient_norm(self) -> Optional[float]:
        """获取最近一次的梯度范数"""
        return self._gradient_norms[-1] if self._gradient_norms else None

    @property
    def gradient_stats(self) -> Dict[str, float]:
        """获取梯度统计信息"""
        if not self._gradient_norms:
            return {'mean': 0.0, 'max': 0.0, 'count': 0}
        grads = torch.tensor(self._gradient_norms)
        return {
            'mean': grads.mean().item(),
            'max': grads.max().item(),
            'min': grads.min().item(),
            'count': len(grads),
        }

    def reset_gradient_tracking(self) -> None:
        """重置梯度跟踪"""
        self._gradient_norms.clear()

    def sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """单步反向采样

        从时间步 t 到 t-1 的去噪步骤。

        数学推导:

        Epsilon prediction (Ho et al., 2020):
            μ_θ = (1/√ᾱ_{t-1}) · β̃_t · √ᾱ_t · x_0_pred +
                  (1/√α_t) · (1-β̃_t) · x_t
            其中 x_0_pred = (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t

        Sample prediction:
            μ_θ = posterior_mean_coef1 · x_0_pred + posterior_mean_coef2 · x_t

        V-prediction (Song et al., 2023):
            ε_θ = (√ᾱ_t · v_θ - α_t · x_t) / √(1-ᾱ_t)
            x_0_pred = (√ᾱ_t · x_t - √(1-ᾱ_t) · v_θ) / α_t

        Args:
            x_t: 当前时刻的加噪数据，形状 (B, ...)
            t: 当前时间步，形状 (B,)，整数
            model_output: 去噪网络的输出，形状与 x_t 相同
            generator: 随机数生成器（用于采样阶段的随机性）

        Returns:
            x_{t-1}: 去噪一步后的数据，形状与 x_t 相同
        """
        sched = self.noise_scheduler

        if x_t.requires_grad or self.enable_gradient_norm_tracking:
            x_t_for_compute = x_t.clone().requires_grad_(True)
        else:
            x_t_for_compute = x_t

        if self.prediction_type == PredictionType.EPSILON:
            predicted_x0, eps = self._predict_x0_from_eps(x_t_for_compute, t, model_output)
        elif self.prediction_type == PredictionType.SAMPLE:
            predicted_x0 = model_output
            eps = self._compute_eps_from_x0(x_t_for_compute, t, predicted_x0)
        elif self.prediction_type in (PredictionType.VELOCITY, PredictionType.V):
            predicted_x0, eps = self._predict_x0_from_v(x_t_for_compute, t, model_output)
        else:
            raise ValueError(f"不支持的预测类型: {self.prediction_type}")

        if self.clip_denoised:
            predicted_x0 = torch.clamp(predicted_x0, *self.clip_range)

        mean_coef1 = sched.posterior_mean_coef1[t]
        mean_coef2 = sched.posterior_mean_coef2[t]

        while mean_coef1.dim() < x_t.dim():
            mean_coef1 = mean_coef1.unsqueeze(-1)
            mean_coef2 = mean_coef2.unsqueeze(-1)

        pred_mean = mean_coef1 * predicted_x0 + mean_coef2 * x_t

        variance = sched.posterior_variance[t]
        while variance.dim() < x_t.dim():
            variance = variance.unsqueeze(-1)

        nonzero_mask = (t > 0).float().reshape(-1, *[1]*(x_t.dim()-1))

        noise_sample = torch.randn_like(x_t, generator=generator)

        x_prev = pred_mean + nonzero_mask * torch.sqrt(variance) * noise_sample

        if x_t_for_compute.requires_grad:
            grad = torch.autograd.grad(outputs=x_prev.sum(), inputs=x_t_for_compute, create_graph=False)[0]
            grad_norm = grad.norm().item()
            self._gradient_norms.append(grad_norm)

            if self.gradient_clip_value > 0:
                x_prev = torch.clamp(
                    x_prev,
                    self.clip_range[0],
                    self.clip_range[1],
                )

        if logger:
            logger.debug(
                f"反向采样 step: t={t.tolist()}, "
                f"x_t range=[{x_t.min():.3f}, {x_t.max():.3f}], "
                f"x_prev range=[{x_prev.min():.3f}, {x_prev.max():.3f}], "
                f"x0_pred range=[{predicted_x0.min():.3f}, {predicted_x0.max():.3f}]"
            )

        return x_prev

    def _predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 epsilon 预测计算 x_0

        公式: x_0 = (x_t - √(1-ᾱ_t) · ε) / √ᾱ_t

        Args:
            x_t: 加噪数据
            t: 时间步
            eps: 预测的噪声

        Returns:
            (predicted_x0, eps) 元组
        """
        sched = self.noise_scheduler

        sqrt_alpha_prod = sched.sqrt_alphas_cumprod[t + 1]
        sqrt_one_minus_alpha_prod = sched.sqrt_one_minus_alphas_cumprod[t + 1]

        while sqrt_alpha_prod.dim() < x_t.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        predicted_x0 = (x_t - sqrt_one_minus_alpha_prod * eps) / sqrt_alpha_prod.clamp(min=1e-10)

        return predicted_x0, eps

    def _compute_eps_from_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        x0_pred: torch.Tensor,
    ) -> torch.Tensor:
        """从 x_0 预测反推噪声

        公式: ε = (x_t - √ᾱ_t · x_0) / √(1-ᾱ_t)

        Args:
            x_t: 加噪数据
            t: 时间步
            x0_pred: 预测的 x_0

        Returns:
            计算出的噪声
        """
        sched = self.noise_scheduler

        sqrt_alpha_prod = sched.sqrt_alphas_cumprod[t + 1]
        sqrt_one_minus_alpha_prod = sched.sqrt_one_minus_alphas_cumprod[t + 1]

        while sqrt_alpha_prod.dim() < x_t.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        eps = (x_t - sqrt_alpha_prod * x0_pred) / sqrt_one_minus_alpha_prod.clamp(min=1e-10)

        return eps

    def _predict_x0_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 v-prediction 计算 x_0 和 ε

        v = α_t · x_t_target - σ_t · ε

        Args:
            x_t: 加噪数据
            t: 时间步
            v_pred: 预测的速度

        Returns:
            (predicted_x0, eps) 元组
        """
        sched = self.noise_scheduler

        sqrt_alpha_prod = sched.sqrt_alphas_cumprod[t + 1]
        sqrt_one_minus_alpha_prod = sched.sqrt_one_minus_alphas_cumprod[t + 1]

        while sqrt_alpha_prod.dim() < x_t.dim():
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        predicted_x0 = (
            sqrt_alpha_prod * x_t - sqrt_one_minus_alpha_prod * v_pred
        )
        eps = (
            sqrt_one_minus_alpha_prod * x_t + sqrt_alpha_prod * v_pred
        )

        return predicted_x0, eps

    def sample_full(
        self,
        x_T: torch.Tensor,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        all_timesteps: Optional[List[int]] = None,
        generator: Optional[torch.Generator] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """完整反向轨迹

        从纯噪声 x_T 开始，逐步去噪到 x_0。

        Args:
            x_T: 初始噪声张量，形状 (B, ...)
            model_fn: 去噪函数，签名 fn(x_t, t) -> model_output
            all_timesteps: 要遍历的时间步列表（降序）。
                          默认为 range(num_steps-1, -1, -1)
            generator: 随机数生成器
            progress_callback: 进度回调函数 callback(step, total)

        Returns:
            最终生成的样本 x_0，形状与 x_T 相同

        Example:
            >>> def my_denoiser(x_t, t):
            ...     return simple_unet(x_t, t)
            >>>
            >>> x_T = torch.randn(4, 3, 32, 32)
            >>> generated = backward.sample_full(x_T, my_denoiser)
        """
        if all_timesteps is None:
            all_timesteps = list(range(self.noise_scheduler.num_steps - 1, -1, -1))

        x_current = x_T
        total_steps = len(all_timesteps)

        for i, t in enumerate(all_timesteps):
            t_batch = torch.full(
                (x_T.shape[0],),
                t,
                device=x_T.device,
                dtype=torch.long,
            )

            with torch.set_grad_enabled(x_T.requires_grad):
                model_output = model_fn(x_current, t_batch)

            x_current = self.sample_step(x_current, t_batch, model_output, generator)

            if progress_callback is not None:
                progress_callback(i + 1, total_steps)

        if logger:
            logger.info(
                f"完整反向采样完成: {total_steps} 步, "
                f"最终范围=[{x_current.min():.3f}, {x_current.max():.3f}]"
            )

        return x_current

    def apply_gradient_normalization(
        self,
        parameters: Any,
        norm_type: float = 2.0,
    ) -> float:
        """应用梯度归一化

        对指定参数的梯度进行归一化处理。

        Args:
            parameters: 模型参数或参数迭代器
            norm_type: 归一化范数类型（通常为2.0）

        Returns:
            总梯度的全局范数
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        parameters = list(filter(lambda p: p.grad is not None, parameters))

        if len(parameters) == 0:
            return 0.0

        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type)
                for p in parameters
            ]),
            norm_type,
        ).item()

        if self.gradient_clip_value > 0:
            clip_coef = self.gradient_clip_value / (total_norm + 1e-6)
            if clip_coef < 1:
                for p in parameters:
                    p.grad.detach().mul_(clip_coef)

        self._gradient_norms.append(total_norm)

        return total_norm

    def set_prediction_type(
        self,
        new_type: Union[str, PredictionType],
    ) -> None:
        """切换预测类型

        Args:
            new_type: 新的预测类型
        """
        if isinstance(new_type, str):
            new_type = PredictionType(new_type)
        old_type = self.prediction_type
        self.prediction_type = new_type
        if logger:
            logger.info(f"预测类型从 {old_type.value} 切换为 {new_type.value}")

    def export_state(self) -> Dict[str, Any]:
        """导出状态

        Returns:
            包含配置和状态的字典
        """
        return {
            'prediction_type': self.prediction_type.value,
            'clip_denoised': self.clip_denoised,
            'clip_range': list(self.clip_range),
            'gradient_clip_value': self.gradient_clip_value,
            'gradient_stats': self.gradient_stats,
        }

    def extra_repr(self) -> str:
        return (
            f"pred={self.prediction_type.value}, "
            f"clip={self.clip_denoised}, "
            f"grad_clip={self.gradient_clip_value}"
        )


def compute_gms_guidance_scale(
    base_scale: float,
    gmm_weight: float,
    timestep_ratio: float,
    min_scale: float = 1.0,
    max_scale: float = 20.0,
) -> float:
    """计算 GMS 引导尺度

    根据 GMM 权重和时间步比例动态调整引导强度。

    Args:
        base_scale: 基础引导尺度
        gmm_weight: GMM 第一个分量的权重
        timestep_ratio: 当前时间步比例 t/T ∈ [0, 1]
        min_scale: 最小尺度
        max_scale: 最大尺度

    Returns:
        调整后的引导尺度
    """
    weight_factor = 1.0 + abs(gmm_weight - 0.5) * 2.0
    time_factor = 1.0 + (1.0 - timestep_ratio) * 3.0

    adjusted = base_scale * weight_factor * time_factor
    adjusted = max(min_scale, min(max_scale, adjusted))

    return adjusted


def apply_classifier_free_guidance(
    model_output_cond: torch.Tensor,
    model_output_uncond: torch.Tensor,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    """应用无分类器引导 (Classifier-Free Guidance)

    CFG 公式:
        output = output_uncond + guidance_scale * (output_cond - output_uncond)

    Args:
        model_output_cond: 有条件时的模型输出
        model_output_uncond: 无条件时的模型输出
        guidance_scale: 引导尺度（>1 增强条件影响）

    Returns:
        引导后的输出
    """
    return model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
