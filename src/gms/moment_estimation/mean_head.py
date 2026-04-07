"""一阶矩输出头模块

用于估计高斯混合模型的一阶矩（均值向量）。
支持多维输出、精度评估和与骨干网络的集成。
"""

from typing import Dict, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MeanHead(nn.Module):
    """一阶矩输出头（均值估计器）

    从骨干网络特征中预测分布的均值向量 μ。
    支持全连接层网络结构和可选的激活函数。

    Attributes:
        feature_dim: 输入特征维度
        output_dim: 输出均值向量维度
        hidden_dim: 隐藏层维度（如果使用多层网络）
        activation: 激活函数类型
        fc: 全连接层序列

    Example:
        >>> head = MeanHead(feature_dim=1024, output_dim=10)
        >>> features = torch.randn(32, 1024)
        >>> mean = head(features)
        >>> print(mean.shape)
        torch.Size([32, 10])
    """

    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        hidden_dims: Optional[Union[int, Tuple[int, ...]]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        """初始化均值输出头

        Args:
            feature_dim: 骨干网络特征维度
            output_dim: 输出均值向量的维度
            hidden_dims: 隐藏层维度，可以是单个整数或元组。
                        如果为None，则使用单层全连接网络
            activation: 激活函数类型，支持 'relu', 'gelu', 'tanh', 'none'
            dropout: Dropout比率，0表示不使用dropout
            use_batch_norm: 是否在隐藏层后使用批归一化

        Raises:
            ValueError: 如果feature_dim或output_dim不是正整数
        """
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim必须是正整数，当前值: {feature_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim必须是正整数，当前值: {output_dim}")

        self.feature_dim = feature_dim
        self.output_dim = output_dim
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
            f"MeanHead初始化完成: "
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
        """前向传播：从特征预测均值向量

        Args:
            features: 骨干网络提取的特征张量，
                     形状为 (batch_size, feature_dim) 或 (feature_dim,)

        Returns:
            均值向量张量，形状为 (batch_size, output_dim) 或 (output_dim,)
        """
        x = features
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.fc(x)
        mean = self.output_layer(x)

        if features.dim() == 1:
            mean = mean.squeeze(0)

        return mean

    def compute_mse(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算均方误差（MSE）

        Args:
            predicted: 预测的均值向量
            target: 真实的均值向量
            reduction: 归约方式，'mean' | 'sum' | 'none'

        Returns:
            MSE值，根据reduction参数可能是标量或张量
        """
        diff = predicted - target
        mse = torch.mean(diff ** 2, dim=-1)

        if reduction == "mean":
            return mse.mean()
        elif reduction == "sum":
            return mse.sum()
        else:
            return mse

    def compute_rmse(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """计算均方根误差（RMSE）

        Args:
            predicted: 预测的均值向量
            target: 真实的均值向量
            reduction: 归约方式，'mean' | 'sum' | 'none'

        Returns:
            RMSE值
        """
        mse = self.compute_mse(predicted, target, reduction="none")
        rmse = torch.sqrt(mse + 1e-8)

        if reduction == "mean":
            return rmse.mean()
        elif reduction == "sum":
            return rmse.sum()
        else:
            return rmse

    def evaluate_accuracy(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """评估均值预测的精度指标

        计算多个精度指标并返回字典。

        Args:
            predicted: 预测的均值向量
            target: 真实的均值向量

        Returns:
            包含以下指标的字典：
            - mse: 均方误差
            - rmse: 均方根误差
            - mae: 平均绝对误差
            - max_error: 最大绝对误差
            - r_squared: 决定系数 R²
        """
        with torch.no_grad():
            metrics = {}

            mse = self.compute_mse(predicted, target).item()
            rmse = self.compute_rmse(predicted, target).item()

            mae = torch.mean(torch.abs(predicted - target)).item()
            max_error = torch.max(torch.abs(predicted - target)).item()

            ss_res = torch.sum((target - predicted) ** 2).item()
            ss_tot = torch.sum((target - torch.mean(target)) ** 2).item()
            r_squared = 1 - (ss_res / (ss_tot + 1e-8))

            metrics["mse"] = mse
            metrics["rmse"] = rmse
            metrics["mae"] = mae
            metrics["max_error"] = max_error
            metrics["r_squared"] = r_squared

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
            f"MeanHead("
            f"input_dim={self.feature_dim}, "
            f"output_dim={self.output_dim}, "
            f"params={num_params})"
        )
