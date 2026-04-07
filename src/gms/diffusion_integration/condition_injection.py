"""GMS 条件注入机制 - 将 GMM 参数作为条件注入扩散模型

实现多种条件注入方式，将 GMS 求解器的参数信息编码后
注入到扩散模型的去噪网络中，增强生成过程的可控性。

支持的注入方式:
    - AdaGN (Adaptive Group Normalization): 用 GMS 参数调制归一化层
    - Cross-Attention: GMS 参数作为交叉注意力的条件
    - FiLM (Feature-wise Linear Modulation): 线性调制特征图
    - Concatenation: 直接拼接 GMS 特征

Example:
    >>> import torch
    >>> from gms.diffusion_integration.condition_injection import (
    ...     GMSConditionInjector, ConditionType, GMSEncoder
    ... )
    >>>
    >>> encoder = GMSEncoder(condition_dim=8, hidden_dim=64)
    >>> injector = GMSConditionInjector(
    ...     feature_dim=128,
    ...     condition_type=ConditionType.FILM,
    ... )
    >>>
    >>> gmm_params_dict = {'weight': 0.6, 'mean': torch.randn(2), 'var': torch.randn(2)}
    >>> cond = encoder(gmm_params_dict)
    >>> features = torch.randn(4, 128, 32, 32)
    >>> injected = injector(features, cond)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import logging
    logger = logging.getLogger(__name__)
except ImportError:
    logger = None


class ConditionType(Enum):
    """条件注入类型枚举

    ADAGN: 自适应组归一化，用条件向量调制 BN/GN 的均值和方差参数
    CROSS_ATTENTION: 交叉注意力机制，将条件作为 key/value 注入
    FILM: 特征级线性调制，对特征图进行仿射变换
    CONCAT: 直接拼接，将条件拼接到通道/特征维度
    ADAPTIVE_NORM: 自适应归一化（AdaGN 的简化版）
    NONE: 不注入（用于对比实验）
    """

    ADAGN = "adagn"
    CROSS_ATTENTION = "cross_attention"
    FILM = "film"
    CONCAT = "concat"
    ADAPTIVE_NORM = "adaptive_norm"
    NONE = "none"


@dataclass
class InjectorConfig:
    """条件注入器配置

    Attributes:
        feature_dim: 输入特征维度
        condition_dim: 条件向量维度
        condition_type: 注入类型
        temperature: 条件强度温度参数 τ
        num_heads: 交叉注意力头数（仅 CROSS_ATTENTION）
        dropout: Dropout 比率
        use_residual: 是否使用残差连接
    """

    feature_dim: int = 256
    condition_dim: int = 64
    condition_type: str = "film"
    temperature: float = 1.0
    num_heads: int = 4
    dropout: float = 0.1
    use_residual: bool = True


class GMSEncoder(nn.Module):
    """GMM 参数条件编码器

    将 GMM 参数字典编码为固定维度的条件向量，
    支持多种表示形式（嵌入向量、特征图等）。

    编码流程:
        1. 提取原始 GMM 参数（权重、均值、方差）
        2. 归一化到合理范围
        3. 通过 MLP 映射到目标维度
        4. 可选的位置编码和时间嵌入融合

    Attributes:
        input_dim: 输入维度（由 GMM 参数决定）
        output_dim: 输出条件向量维度
        hidden_dims: 隐藏层维度列表
        use_layer_norm: 是否使用 LayerNorm

    Example:
        >>> encoder = GMSEncoder(
        ...     input_dim=7,
        ...     output_dim=32,
        ...     hidden_dims=[64, 64]
        ... )
        >>> params = {
        ...     'weight': 0.6,
        ...     'mean1': torch.tensor([1.0, -0.5]),
        ...     'mean2': torch.tensor([-1.0, 0.5]),
        ...     'variance1': torch.tensor([0.3, 0.3]),
        ...     'variance2': torch.tensor([0.8, 0.8])
        ... }
        >>> cond_vector = encoder(params)
        >>> print(cond_vector.shape)  # torch.Size([32])
    """

    def __init__(
        self,
        output_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        input_dim: int = 7,
        activation: str = "silu",
        use_layer_norm: bool = True,
        dropout: float = 0.0,
        time_embedding_dim: int = 0,
    ):
        """初始化 GMM 条件编码器

        Args:
            output_dim: 输出条件向量的维度
            hidden_dims: 隐藏层维度列表，默认 [output_dim*2, output_dim]
            input_dim: 输入维度（GMM 参数的展平维度）
            activation: 激活函数 ('relu', 'silu', 'gelu', 'tanh')
            use_layer_norm: 是否在输出前使用 LayerNorm
            dropout: Dropout 比率
            time_embedding_dim: 时间嵌入的额外维度（0 表示不使用）
        """
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.time_embedding_dim = time_embedding_dim

        if hidden_dims is None:
            hidden_dims = [output_dim * 2, output_dim]

        act_fn = self._get_activation(activation)

        layers = []
        prev_dim = input_dim + (time_embedding_dim if time_embedding_dim > 0 else 0)

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                act_fn,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

        if time_embedding_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, time_embedding_dim),
                act_fn,
                nn.Linear(time_embedding_dim, time_embedding_dim),
            )

        if logger:
            logger.info(
                f"GMSEncoder 初始化完成: "
                f"in={input_dim}, out={output_dim}, "
                f"hidden={hidden_dims}, time_emb={time_embedding_dim}"
            )

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """获取激活函数模块"""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'silu': nn.SiLU(inplace=True),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
        }
        return activations.get(name.lower(), nn.SiLU(inplace=True))

    def _extract_gmm_features(
        self,
        gmm_params: Union[Dict[str, Any], torch.Tensor],
    ) -> torch.Tensor:
        """从 GMM 参数中提取原始特征

        支持两种输入格式:
        1. 字典格式: 包含 weight, mean1, mean2, variance1, variance2 等
        2. 张量格式: 已经预处理好的特征张量

        Args:
            gmm_params: GMM 参数或特征张量

        Returns:
            展平的特征张量
        """
        if isinstance(gmm_params, torch.Tensor):
            return gmm_params.flatten()

        features = []

        weight = gmm_params.get('weight', 0.5)
        features.append(torch.tensor([float(weight)]))

        mean1 = gmm_params.get('mean1', gmm_params.get('mean', torch.zeros(1)))
        if isinstance(mean1, torch.Tensor):
            features.append(mean1.flatten())
        else:
            features.append(torch.tensor([float(mean1)]))

        mean2 = gmm_params.get('mean2', torch.zeros_like(mean1))
        if isinstance(mean2, torch.Tensor):
            features.append(mean2.flatten())
        else:
            features.append(torch.tensor([float(mean2)]))

        var1 = gmm_params.get('variance1', gmm_params.get('variance', torch.ones(1)))
        if isinstance(var1, torch.Tensor):
            features.append(torch.log(torch.clamp(var1, min=1e-6)).flatten())
        else:
            features.append(torch.tensor([math.log(max(float(var1), 1e-6))]))

        var2 = gmm_params.get('variance2', torch.ones_like(var1))
        if isinstance(var2, torch.Tensor):
            features.append(torch.log(torch.clamp(var2, min=1e-6)).flatten())
        else:
            features.append(torch.tensor([math.log(max(float(var2), 1e-6))]))

        result = torch.cat([f if f.dim() > 0 else f.unsqueeze(0) for f in features])

        return result.float()

    def forward(
        self,
        gmm_params: Union[Dict[str, Any], torch.Tensor],
        timestep_ratio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """编码 GMM 参数为条件向量

        Args:
            gmm_params: GMM 参数字典或预处理的特征张量
            timestep_ratio: 可选的时间步比例张量 (B,) ∈ [0, 1]，
                           用于时间相关的条件编码

        Returns:
            条件向量，形状 (B, output_dim) 或 (output_dim,)
        """
        raw_features = self._extract_gmm_features(gmm_params)

        if raw_features.device != next(self.parameters()).device:
            raw_features = raw_features.to(next(self.parameters()).device)

        if timestep_ratio is not None and self.time_embedding_dim > 0:
            t_emb = self.time_mlp(timestep_ratio.unsqueeze(-1).float())
            if raw_features.dim() == 1:
                combined = torch.cat([raw_features, t_emb.squeeze(0)], dim=-1)
            else:
                combined = torch.cat([raw_features, t_emb], dim=-1)
        else:
            combined = raw_features

        encoded = self.mlp(combined)

        if self.layer_norm is not None:
            encoded = self.layer_norm(encoded)

        return encoded

    def encode_batch(
        self,
        batch_gmm_params: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """批量编码多个 GMM 参数

        Args:
            batch_gmm_params: GMM 参数字典的列表

        Returns:
            批量条件向量，形状 (B, output_dim)
        """
        all_encoded = []
        for params in batch_gmm_params:
            enc = self.forward(params)
            all_encoded.append(enc)

        return torch.stack(all_encoded, dim=0)


class CrossAttentionInjector(nn.Module):
    """交叉注意力条件注入模块

    使用多头交叉注意力将条件信息注入特征图。

    架构:
        Q = Linear(features)      -- 来自主特征
        K = Linear(condition)      -- 来自条件向量
        V = Linear(condition)      -- 来自条件向量
        output = Softmax(QK^T / √d) · V
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """初始化交叉注意力注入器

        Args:
            feature_dim: 特征维度
            condition_dim: 条件维度
            num_heads: 注意力头数
            dropout: Dropout 比率
        """
        super().__init__()

        assert feature_dim % num_heads == 0, \
            f"feature_dim ({feature_dim}) 必须能被 num_heads ({num_heads}) 整除"

        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.q_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.k_proj = nn.Linear(condition_dim, feature_dim, bias=False)
        self.v_proj = nn.Linear(condition_dim, feature_dim, bias=False)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """应用交叉注意力注入

        Args:
            features: 输入特征，形状 (B, C, H, W) 或 (B, N, C)
            condition: 条件向量，形状 (B, D)

        Returns:
            注入后的特征
        """
        B = features.shape[0]
        original_shape = features.shape

        if features.dim() == 4:
            C, H, W = features.shape[1:]
            features_flat = features.reshape(B, C, H * W).transpose(1, 2)
        else:
            features_flat = features

        q = self.q_proj(features_flat).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(condition).unsqueeze(1).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(condition).unsqueeze(1).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(B, -1, self.feature_dim)

        output = self.out_proj(attn_output)

        if original_shape[-2:] == features.shape[-2:]:
            output = output.transpose(1, 2).reshape(original_shape)

        return output


class FiLMLayer(nn.Module):
    """FiLM (Feature-wise Linear Modulation) 调制层

    对输入特征进行条件化的仿射变换:
        FiLM(x; γ, β) = γ ⊙ x + β

    其中 γ 和 β 由条件向量通过线性层生成。
    """

    def __init__(
        self,
        feature_dim: int,
        condition_dim: int,
        init_bias: float = 0.0,
    ):
        """初始化 FiLM 层

        Args:
            feature_dim: 特征维度
            condition_dim: 条件维度
            init_bias: β 的初始偏置值
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim

        self.modulation_net = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
        )

        nn.init.zeros_(self.modulation_net[0].bias[:feature_dim])
        nn.init.zeros_(self.modulation_net[0].bias[feature_dim:])
        with torch.no_grad():
            self.modulation_net[0].weight[:, :feature_dim].fill_(0.0)
            self.modulation_net[0].weight[:, feature_dim:].fill_(0.0)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 FiLM 调制

        Args:
            features: 输入特征，形状 (..., feature_dim)
            condition: 条件向量，形状 (..., condition_dim)

        Returns:
            (modulated_features, (gamma, beta)) 元组
        """
        modulation = self.modulation_net(condition)
        gamma, beta = modulation.chunk(2, dim=-1)

        while gamma.dim() < features.dim():
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        modulated = gamma * features + beta

        return modulated, (gamma.detach(), beta.detach())


class AdaptiveGroupNorm(nn.Module):
    """自适应组归一化 (Adaptive Group Normalization / AdaGN)

    用条件向量动态计算 GroupNorm 的均值和方差偏置。

    公式:
        AdaGN(x; γ, β) = γ · GN(x) + β
        其中 (γ, β) = MLP(condition)
    """

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 8,
        condition_dim: int = 64,
        eps: float = 1e-5,
    ):
        """初始化 AdaGN

        Args:
            num_channels: 通道数
            num_groups: 组数（必须整除 num_channels）
            condition_dim: 条件维度
            eps: 数值稳定性常数
        """
        super().__init__()
        assert num_channels % num_groups == 0

        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, num_channels * 2),
        )

        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """应用自适应组归一化

        Args:
            x: 输入特征，形状 (B, C, ...)
            condition: 条件向量，形状 (B, D)

        Returns:
            归一化并调制的特征
        """
        normalized = self.norm(x)
        modulation = self.modulation(condition)

        shape = [x.shape[0], -1] + [1] * (x.dim() - 2)
        gamma, beta = modulation.reshape(shape).chunk(2, dim=1)

        return gamma * normalized + beta


class GMSConditionInjector(nn.Module):
    """GMS 条件注入管理器

    统一管理多种条件注入方式的选择和执行。
    作为扩散模型与 GMS 之间的条件接口。

    Attributes:
        condition_type: 当前使用的注入类型
        temperature: 条件强度温度参数 τ
        injectors: 各类型的注入子模块
        encoder: GMM 参数编码器

    Example:
        >>> injector = GMSConditionInjector(
        ...     feature_dim=128,
        ...     condition_type=ConditionType.FILM,
        ...     temperature=1.5,
        ... )
        >>>
        >>> features = torch.randn(4, 128, 16, 16)
        >>> condition = torch.randn(4, 64)
        >>> out = injector(features, condition)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        condition_dim: int = 64,
        condition_type: Union[str, ConditionType] = ConditionType.FILM,
        temperature: float = 1.0,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_residual: bool = True,
        num_groups: int = 8,
    ):
        """初始化 GMS 条件注入器

        Args:
            feature_dim: 输入特征的维度（通道数或特征维度）
            condition_dim: 条件向量的维度
            condition_type: 条件注入类型
            temperature: 温度参数 τ，控制条件强度:
                         τ=1 标准强度, τ>1 增强, τ<1 减弱
            num_heads: 交叉注意力的头数
            dropout: Dropout 比率
            use_residual: 是否使用残差连接
            num_groups: GroupNorm 的分组数

        Raises:
            ValueError: 如果参数不合法
        """
        super().__init__()

        if isinstance(condition_type, str):
            condition_type = ConditionType(condition_type)

        if temperature <= 0:
            raise ValueError(f"temperature 必须为正数，当前值: {temperature}")

        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.condition_type = condition_type
        self.temperature = temperature
        self.use_residual = use_residual

        self.film_layer: Optional[FiLMLayer] = None
        self.cross_attn: Optional[CrossAttentionInjector] = None
        self.adagn: Optional[AdaptiveGroupNorm] = None
        self.concat_projection: Optional[nn.Module] = None

        if condition_type == ConditionType.FILM:
            self.film_layer = FiLMLayer(feature_dim, condition_dim)

        elif condition_type == ConditionType.CROSS_ATTENTION:
            self.cross_attn = CrossAttentionInjector(
                feature_dim=feature_dim,
                condition_dim=condition_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        elif condition_type == ConditionType.ADAGN or \
             condition_type == ConditionType.ADAPTIVE_NORM:
            self.adagn = AdaptiveGroupNorm(
                num_channels=feature_dim,
                num_groups=num_groups,
                condition_dim=condition_dim,
            )

        elif condition_type == ConditionType.CONCAT:
            self.concat_projection = nn.Conv2d(
                feature_dim + condition_dim,
                feature_dim,
                kernel_size=1,
            )

        if logger:
            logger.info(
                f"GMSConditionInjector 初始化完成: "
                f"type={condition_type.value}, "
                f"feat={feature_dim}, cond={condition_dim}, "
                f"τ={temperature}"
            )

    def set_temperature(self, new_temperature: float) -> None:
        """设置新的温度参数

        Args:
            new_temperature: 新的温度值（必须 > 0）

        Raises:
            ValueError: 如果温度值不合法
        """
        if new_temperature <= 0:
            raise ValueError(f"temperature 必须为正数，得到 {new_temperature}")
        old_temp = self.temperature
        self.temperature = new_temperature
        if logger:
            logger.debug(f"温度更新: {old_temp} -> {new_temperature}")

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
        return_modulation_params: bool = False,
    ) -> torch.Tensor:
        """执行条件注入

        Args:
            features: 输入特征张量:
                      - 图像数据: (B, C, H, W)
                      - 序列数据: (B, N, D)
                      - 向量数据: (B, D)
            condition: GMS 条件向量，形状 (B, condition_dim)
            return_modulation_params: 是否返回调制参数（用于分析/可视化）

        Returns:
            注入后的特征张量，形状与 features 相同。
            如果 return_modulation_params=True，额外返回调制参数。

        Raises:
            RuntimeError: 如果特征维度不匹配配置
        """
        actual_feature_dim = features.shape[1]
        if actual_feature_dim != self.feature_dim and \
           self.condition_type not in (ConditionType.NONE,):
            raise RuntimeError(
                f"特征维度不匹配: 配置 {self.feature_dim}, "
                f"实际 {actual_feature_dim}"
            )

        scaled_condition = condition * self.temperature

        if self.condition_type == ConditionType.NONE:
            output = features
            mod_params = None

        elif self.condition_type == ConditionType.FILM and self.film_layer is not None:
            output, mod_params = self.film_layer(features, scaled_condition)

        elif self.condition_type == ConditionType.CROSS_ATTENTION and \
             self.cross_attn is not None:
            output = self.cross_attn(features, scaled_condition)
            mod_params = None

        elif self.condition_type in (ConditionType.ADAGN, ConditionType.ADAPTIVE_NORM) and \
             self.adagn is not None:
            output = self.adagn(features, scaled_condition)
            mod_params = None

        elif self.condition_type == ConditionType.CONCAT and \
             self.concat_projection is not None:
            if features.dim() == 4:
                B, C, H, W = features.shape
                cond_expanded = scaled_condition.unsqueeze(-1).unsqueeze(-1).expand(B, -1, H, W)
                concat_feat = torch.cat([features, cond_expanded], dim=1)
            elif features.dim() == 3:
                B, N, D = features.shape
                cond_expanded = scaled_condition.unsqueeze(1).expand(B, N, -1)
                concat_feat = torch.cat([features, cond_expanded], dim=-1)
            elif features.dim() == 2:
                concat_feat = torch.cat([features, scaled_condition], dim=-1)
            else:
                concat_feat = torch.cat([features, scaled_condition.expand_as(features)], dim=-1)

            output = self.concat_projection(concat_feat)
            mod_params = None

        else:
            output = features
            mod_params = None

        if self.use_residual and self.condition_type != ConditionType.NONE:
            output = output + features

        if logger:
            logger.debug(
                f"条件注入完成: type={self.condition_type.value}, "
                f"τ={self.temperature:.3f}, "
                f"input_range=[{features.min():.3f}, {features.max():.3f}], "
                f"output_range=[{output.min():.3f}, {output.max():.3f}]"
            )

        if return_modulation_params:
            return output, mod_params
        return output

    def get_injection_stats(self) -> Dict[str, Any]:
        """获取注入器的统计信息

        Returns:
            包含配置和状态信息的字典
        """
        total_params = sum(p.numel() for p in self.parameters())

        stats = {
            'type': self.condition_type.value,
            'feature_dim': self.feature_dim,
            'condition_dim': self.condition_dim,
            'temperature': self.temperature,
            'total_parameters': total_params,
            'use_residual': self.use_residual,
            'has_film': self.film_layer is not None,
            'has_cross_attn': self.cross_attn is not None,
            'has_adagn': self.adagn is not None,
            'has_concat': self.concat_projection is not None,
        }

        return stats

    def switch_mode(
        self,
        new_type: Union[str, ConditionType],
        **kwargs,
    ) -> None:
        """切换条件注入模式

        注意：这会重新创建内部模块，可能影响已学习的参数。

        Args:
            new_type: 新的注入类型
            **kwargs: 传递给新模式的额外参数
        """
        old_type = self.condition_type.value

        if isinstance(new_type, str):
            new_type = ConditionType(new_type)

        self.__init__(
            feature_dim=self.feature_dim,
            condition_dim=self.condition_dim,
            condition_type=new_type,
            temperature=self.temperature,
            **kwargs,
        )

        if logger:
            logger.info(f"注入模式从 {old_type} 切换为 {new_type.value}")

    def extra_repr(self) -> str:
        return (
            f"type={self.condition_type.value}, "
            f"τ={self.temperature:.2f}, "
            f"residual={self.use_residual}"
        )


def build_full_conditioning_pipeline(
    feature_dim: int,
    condition_dim: int,
    injection_type: str = "film",
    encoder_hidden_dims: Optional[List[int]] = None,
    gmm_input_dim: int = 7,
    temperature: float = 1.0,
) -> Tuple[GMSEncoder, GMSConditionInjector]:
    """构建完整的条件注入管线

    快速创建编码器和注入器的便捷函数。

    Args:
        feature_dim: 特征维度
        condition_dim: 条件维度
        injection_type: 注入类型
        encoder_hidden_dims: 编码器隐藏层维度
        gmm_input_dim: GMM 输入维度
        temperature: 温度参数

    Returns:
        (encoder, injector) 元组

    Example:
        >>> encoder, injector = build_full_conditioning_pipeline(
        ...     feature_dim=128, condition_dim=32, injection_type='film'
        ... )
        >>> gmm_params = {'weight': 0.5, 'mean': torch.randn(2)}
        >>> cond = encoder(gmm_params)
        >>> features = torch.randn(4, 128, 16, 16)
        >>> out = injector(features, cond)
    """
    encoder = GMSEncoder(
        output_dim=condition_dim,
        hidden_dims=encoder_hidden_dims,
        input_dim=gmm_input_dim,
    )

    injector = GMSConditionInjector(
        feature_dim=feature_dim,
        condition_dim=condition_dim,
        condition_type=injection_type,
        temperature=temperature,
    )

    return encoder, injector
