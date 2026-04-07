"""二阶矩输出头模块

用于估计高斯混合模型的二阶矩（方差/协方差矩阵）。
支持对角协方差和全协方差矩阵两种模式。
"""

from typing import Dict, Literal, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VarianceHead(nn.Module):
    """二阶矩输出头（方差/协方差估计器）

    从骨干网络特征中预测分布的方差 σ² 或协方差矩阵 Σ。
    支持对角协方差和全协方差矩阵两种模式。

    Attributes:
        feature_dim: 输入特征维度
        output_dim: 输出维度（对应随机变量的维度）
        mode: 协方差模式，'diagonal' 或 'full'
        hidden_dims: 隐藏层维度配置
        activation: 激活函数类型
        variance_activation: 方差激活函数，确保输出为正数

    Example:
        >>> head = VarianceHead(feature_dim=1024, output_dim=10, mode='diagonal')
        >>> features = torch.randn(32, 1024)
        >>> variance = head(features)
        >>> print(variance.shape)
        torch.Size([32, 10])
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        mode: Literal["diagonal", "full"] = "diagonal",
        hidden_dims: Optional[Union[int, Tuple[int, ...]]] = None,
        activation: str = "relu",
        variance_activation: str = "softplus",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
        min_variance: float = 1e-6,
    ) -> None:
        """初始化方差输出头

        Args:
            feature_dim: 骨干网络特征维度
            output_dim: 输出维度（随机变量的维度）
            mode: 协方差模式
                  - 'diagonal': 输出对角协方差矩阵 (batch, output_dim)
                  - 'full': 输出全协方差矩阵 (batch, output_dim, output_dim)
            hidden_dims: 隐藏层维度配置
            activation: 隐藏层激活函数类型
            variance_activation: 方差激活函数，确保输出为正数
                                 支持 'softplus', 'exp', 'relu', 'elu'
            dropout: Dropout比率
            use_batch_norm: 是否使用批归一化
            min_variance: 方差的最小值（数值稳定性）

        Raises:
            ValueError: 如果参数值无效
        """
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim必须是正整数，当前值: {feature_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim必须是正整数，当前值: {output_dim}")
        if mode not in ["diagonal", "full"]:
            raise ValueError(f"mode必须是'diagonal'或'full'，当前值: {mode}")

        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.mode = mode
        self.min_variance = min_variance
        self.variance_activation_name = variance_activation.lower()

        layers = []
        current_dim = feature_dim

        if hidden_dims is not None:
            if isinstance(hidden_dims, int):
                hidden_dims = (hidden_dims,)

            for h_dim in hidden_dims:
                if h_dim <= 0:
                    raise ValueError(f"hidden_dims中的所有值必须是正整数")

                layers.append(nn.Linear(current_dim, h_dim))

                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(h_dim))

                if activation.lower() != "none":
                    layers.append(self._get_activation(activation))

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                current_dim = h_dim

        self.fc = nn.Sequential(*layers)

        if mode == "diagonal":
            self.output_layer = nn.Linear(current_dim, output_dim)
        else:
            self.output_layer = nn.Linear(current_dim, output_dim * output_dim)

        logger.info(
            f"VarianceHead初始化完成: "
            f"输入维度={feature_dim}, "
            f"输出维度={output_dim}, "
            f"模式={mode}"
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

    def _apply_variance_activation(self, x: torch.Tensor) -> torch.Tensor:
        """应用方差激活函数，确保输出为正数

        Args:
            x: 原始输出

        Returns:
            激活后的正数张量
        """
        if self.variance_activation_name == "softplus":
            return F.softplus(x) + self.min_variance
        elif self.variance_activation_name == "exp":
            return torch.exp(x) + self.min_variance
        elif self.variance_activation_name == "relu":
            return F.relu(x) + self.min_variance
        elif self.variance_activation_name == "elu":
            return F.elu(x) + 1.0 + self.min_variance
        else:
            raise ValueError(
                f"不支持的方差激活函数: {self.variance_activation_name}"
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播：从特征预测方差/协方差

        Args:
            features: 骨干网络提取的特征张量，
                     形状为 (batch_size, feature_dim) 或 (feature_dim,)

        Returns:
            如果mode='diagonal': 方差向量，形状为 (batch_size, output_dim) 或 (output_dim,)
            如果mode='full': 协方差矩阵，形状为 (batch_size, output_dim, output_dim) 或 (output_dim, output_dim,)
        """
        x = features
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.fc(x)
        output = self.output_layer(x)

        if self.mode == "diagonal":
            variance = self._apply_variance_activation(output)
        else:
            batch_size = x.shape[0]
            output = output.view(batch_size, self.output_dim, self.output_dim)

            L = torch.tril(output)
            diag_indices = torch.arange(self.output_dim, device=output.device)

            diag_values = self._apply_variance_activation(
                L[:, diag_indices, diag_indices]
            )
            L[:, diag_indices, diag_indices] = diag_values

            variance = torch.bmm(L, L.transpose(-2, -1))

            eye = torch.eye(
                self.output_dim,
                device=variance.device,
                dtype=variance.dtype
            ).unsqueeze(0)
            variance = variance + self.min_variance * eye

        if features.dim() == 1:
            variance = variance.squeeze(0)

        return variance

    def compute_relative_error(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算相对误差

        公式: |predicted - target| / (|target| + epsilon)

        Args:
            predicted: 预测的方差/协方差
            target: 真实的方差/协方差
            reduction: 归约方式，'mean' | 'sum' | 'none'

        Returns:
            相对误差值
        """
        epsilon = 1e-8
        abs_diff = torch.abs(predicted - target)
        abs_target = torch.abs(target) + epsilon
        relative_error = abs_diff / abs_target

        if reduction == "mean":
            return relative_error.mean()
        elif reduction == "sum":
            return relative_error.sum()
        else:
            return relative_error

    def compute_frobenius_norm(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算Frobenius范数（适用于矩阵）

        公式: ||predicted - target||_F

        Args:
            predicted: 预测的协方差矩阵
            target: 真实的协方差矩阵
            reduction: 归约方式，'mean' | 'sum' | 'none'

        Returns:
            Frobenius范数值
        """
        diff = predicted - target
        frob_norm = torch.norm(diff, p='fro', dim=(-2, -1))

        if reduction == "mean":
            return frob_norm.mean()
        elif reduction == "sum":
            return frob_norm.sum()
        else:
            return frob_norm

    def compute_diagonal_accuracy(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """计算对角元素的准确性

        对于对角协方差模式，计算所有元素的精度。
        对于全协方差模式，只计算对角线元素的精度。

        Args:
            predicted: 预测的方差/协方差
            target: 真实的方差/协方差

        Returns:
            包含对角元素精度指标的字典
        """
        with torch.no_grad():
            if self.mode == "diagonal":
                pred_diag = predicted
                target_diag = target
            else:
                pred_diag = torch.diagonal(predicted, dim1=-2, dim2=-1)
                target_diag = torch.diagonal(target, dim1=-2, dim2=-1)

            abs_error = torch.abs(pred_diag - target_diag)
            rel_error = abs_error / (torch.abs(target_diag) + 1e-8)

            metrics = {
                "diagonal_mae": abs_error.mean().item(),
                "diagonal_relative_error": rel_error.mean().item(),
                "diagonal_max_error": abs_error.max().item(),
            }

        return metrics

    def evaluate_accuracy(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """评估方差/协方差预测的精度指标

        计算多个精度指标并返回字典。

        Args:
            predicted: 预测的方差/协方差
            target: 真实的方差/协方差

        Returns:
            包含精度指标的字典
        """
        with torch.no_grad():
            metrics = {}

            relative_error = self.compute_relative_error(
                predicted, target, reduction="mean"
            ).item()
            metrics["relative_error"] = relative_error

            if self.mode == "full":
                frob_norm = self.compute_frobenius_norm(
                    predicted, target, reduction="mean"
                ).item()
                metrics["frobenius_norm"] = frob_norm

            diag_metrics = self.compute_diagonal_accuracy(predicted, target)
            metrics.update(diag_metrics)

            mse = torch.mean((predicted - target) ** 2).item()
            metrics["mse"] = mse

        return metrics

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
            f"VarianceHead("
            f"input_dim={self.feature_dim}, "
            f"output_dim={self.output_dim}, "
            f"mode={self.mode}, "
            f"params={num_params})"
        )
