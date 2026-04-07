"""三阶矩输出头模块

用于估计高斯混合模型的三阶矩（偏度系数）。
偏度衡量分布的不对称性，是高斯混合模型的重要特征。
"""

from typing import Dict, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SkewnessHead(nn.Module):
    """三阶矩输出头（偏度估计器）

    从骨干网络特征中预测分布的偏度系数 γ。
    偏度公式: γ = E[(X-μ)³] / σ³

    偏度的含义：
    - γ > 0: 正偏斜（右尾较长）
    - γ = 0: 对称分布
    - γ < 0: 负偏斜（左尾较长）

    Attributes:
        feature_dim: 输入特征维度
        output_dim: 输出偏度向量的维度
        hidden_dims: 隐藏层维度配置
        activation: 激活函数类型

    Example:
        >>> head = SkewnessHead(feature_dim=1024, output_dim=10)
        >>> features = torch.randn(32, 1024)
        >>> skewness = head(features)
        >>> print(skewness.shape)
        torch.Size([32, 10])
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_dims: Optional[Union[int, Tuple[int, ...]]] = None,
        activation: str = "tanh",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        clamp_range: Tuple[float, float] = (-5.0, 5.0),
    ) -> None:
        """初始化偏度输出头

        Args:
            feature_dim: 骨干网络特征维度
            output_dim: 输出偏度向量的维度
            hidden_dims: 隐藏层维度配置，可以是单个整数或元组。
                        如果为None，则使用单层全连接网络
            activation: 激活函数类型，支持 'relu', 'gelu', 'tanh', 'none'
                       默认使用tanh，因为偏度通常在有限范围内
            dropout: Dropout比率，0表示不使用dropout
            use_batch_norm: 是否在隐藏层后使用批归一化
            clamp_range: 将输出限制在此范围内，防止极端值。
                        典型偏度值通常在 [-3, 3] 范围内

        Raises:
            ValueError: 如果参数值无效
        """
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim必须是正整数，当前值: {feature_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim必须是正整数，当前值: {output_dim}")
        if clamp_range[0] >= clamp_range[1]:
            raise ValueError(f"clamp_range无效: {clamp_range}，下限必须小于上限")

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.clamp_min = clamp_range[0]
        self.clamp_max = clamp_range[1]
        self.activation_name = activation.lower()

        layers = []
        current_dim = feature_dim

        if hidden_dims is not None:
            if isinstance(hidden_dims, int):
                hidden_dims = (hidden_dims,)

            for i, h_dim in enumerate(hidden_dims):
                if h_dim <= 0:
                    raise ValueError(f"hidden_dims中的所有值必须是正整数")

                layers.append(nn.Linear(current_dim, h_dim))

                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(h_dim))

                if self.activation_name != "none":
                    layers.append(self._get_activation(activation))

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                current_dim = h_dim

        self.fc = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_dim, output_dim)

        logger.info(
            f"SkewnessHead初始化完成: "
            f"输入维度={feature_dim}, "
            f"输出维度={output_dim}"
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数实例

        Args:
            activation: 激活函数名称

        Returns:
            对应的nn.Module激活函数实例
        """
        activation_map = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1, inplace=True),
            "silu": nn.SiLU(inplace=True),
            "none": nn.Identity(),
        }

        act_lower = activation.lower()
        if act_lower not in activation_map:
            raise ValueError(
                f"不支持的激活函数: {activation}。"
                f"支持的激活函数: {list(activation_map.keys())}"
            )

        return activation_map[act_lower]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播：从特征预测偏度系数

        Args:
            features: 骨干网络提取的特征张量，
                     形状为 (batch_size, feature_dim) 或 (feature_dim,)

        Returns:
            偏度系数张量，形状为 (batch_size, output_dim) 或 (output_dim,)
            值被限制在 [clamp_min, clamp_max] 范围内
        """
        x = features
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.fc(x)
        skewness = self.output_layer(x)

        skewness = torch.clamp(skewness, min=self.clamp_min, max=self.clamp_max)

        if features.dim() == 1:
            skewness = skewness.squeeze(0)

        return skewness

    def compute_skewness_error(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算偏度系数误差

        使用绝对误差和符号误差的组合来评估。

        Args:
            predicted: 预测的偏度系数
            target: 真实的偏度系数
            reduction: 归约方式，'mean' | 'sum' | 'none'

        Returns:
            误差值
        """
        abs_error = torch.abs(predicted - target)

        if reduction == "mean":
            return abs_error.mean()
        elif reduction == "sum":
            return abs_error.sum()
        else:
            return abs_error

    def check_sign_correctness(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        tolerance: float = 0.1,
    ) -> Dict[str, float]:
        """检查偏度符号的正确性

        判断预测的偏度方向（正/负/零）是否与真实值一致。

        Args:
            predicted: 预测的偏度系数
            target: 真实的偏度系数
            tolerance: 判断为零的容差范围

        Returns:
            包含符号正确性指标的字典：
            - sign_accuracy: 符号正确率（0到1之间）
            - positive_correct: 正偏斜正确数
            - negative_correct: 负偏斜正确数
            - zero_correct: 接近零的正确数
        """
        with torch.no_grad():
            pred_signs = torch.sign(predicted)
            true_signs = torch.sign(target)

            near_zero_mask = torch.abs(target) < tolerance
            pred_near_zero = torch.abs(predicted) < tolerance

            exact_match = (pred_signs == true_signs).float()
            zero_match = (near_zero_mask & pred_near_zero).float()
            combined_correct = torch.max(exact_match, zero_match)

            pos_true = (true_signs > 0).float().sum()
            neg_true = (true_signs < 0).float().sum()

            pos_correct = ((pred_signs > 0) & (true_signs > 0)).float().sum()
            neg_correct = ((pred_signs < 0) & (true_signs < 0)).float().sum()
            zero_correct = zero_match.sum()

            total = predicted.numel()
            sign_accuracy = combined_correct.sum().item() / max(total, 1)

            metrics = {
                "sign_accuracy": sign_accuracy,
                "positive_correct": pos_correct.item(),
                "negative_correct": neg_correct.item(),
                "zero_correct": zero_correct.item(),
                "total_positive": pos_true.item(),
                "total_negative": neg_true.item(),
            }

        return metrics

    def evaluate_accuracy(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """评估偏度预测的精度指标

        计算多个精度指标并返回字典。

        Args:
            predicted: 预测的偏度系数
            target: 真实的偏度系数

        Returns:
            包含以下指标的字典：
            - mae: 平均绝对误差
            - mse: 均方误差
            - rmse: 均方根误差
            - max_error: 最大绝对误差
            - sign_accuracy: 符号正确率
            - mean_predicted: 预测值的均值
            - mean_target: 目标值的均值
        """
        with torch.no_grad():
            metrics = {}

            mae = torch.mean(torch.abs(predicted - target)).item()
            mse = torch.mean((predicted - target) ** 2).item()
            rmse = torch.sqrt(torch.tensor(mse + 1e-8))
            max_error = torch.max(torch.abs(predicted - target)).item()

            metrics["mae"] = mae
            metrics["mse"] = mse
            metrics["rmse"] = rmse
            metrics["max_error"] = max_error
            metrics["mean_predicted"] = predicted.mean().item()
            metrics["mean_target"] = target.mean().item()

            sign_metrics = self.check_sign_correctness(predicted, target)
            metrics.update(sign_metrics)

        return metrics

    @staticmethod
    def compute_skewness_from_samples(
        samples: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """从样本计算偏度系数

        根据公式: γ = E[(X-μ)³] / σ³

        Args:
            samples: 样本张量，形状为 (n_samples,) 或 (n_samples, dim)
            mean: 可选的已知均值，如果为None则从样本计算
            std: 可选的标准差，如果为None则从样本计算

        Returns:
            偏度系数，形状为 () 或 (dim,)
        """
        if mean is None:
            mean = samples.mean(dim=0)

        centered = samples - mean

        if std is None:
            std = torch.std(samples, dim=0, unbiased=True) + 1e-8

        third_moment = torch.mean(centered ** 3, dim=0)
        skewness = third_moment / (std ** 3 + 1e-8)

        return skewness

    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """获取参数数量

        Args:
            trainable_only: 是否只统计可训练参数

        Returns:
            参数总数或可训练参数数
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """返回模型的字符串表示"""
        num_params = self.get_num_parameters()
        return (
            f"SkewnessHead("
            f"input_dim={self.feature_dim}, "
            f"output_dim={self.output_dim}, "
            f"clamp=[{self.clamp_min}, {self.clamp_max}], "
            f"params={num_params})"
        )
