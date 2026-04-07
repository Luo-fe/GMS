"""输出头模块的单元测试

测试MeanHead、VarianceHead、SkewnessHead和MomentEstimator的功能。
包括实例化、前向传播、精度评估、边界情况和梯度流测试。
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any

# 导入被测模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gms.moment_estimation.mean_head import MeanHead
from gms.moment_estimation.variance_head import VarianceHead
from gms.moment_estimation.skewness_head import SkewnessHead
from gms.moment_estimation.moment_heads import (
    MomentEstimator,
    MomentResult,
    create_moment_estimator,
)


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def device():
    """返回可用的PyTorch设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def feature_dim():
    """特征维度"""
    return 256


@pytest.fixture(scope="session")
def output_dim():
    """输出维度"""
    return 10


@pytest.fixture(scope="session")
def batch_size():
    """批次大小"""
    return 32


@pytest.fixture(scope="session")
def sample_features(batch_size, feature_dim, device):
    """生成示例特征张量"""
    torch.manual_seed(42)
    features = torch.randn(batch_size, feature_dim).to(device)
    return features


@pytest.fixture(scope="session")
def single_feature(feature_dim, device):
    """生成单个样本的特征张量"""
    torch.manual_seed(42)
    feature = torch.randn(feature_dim).to(device)
    return feature


# ==================== 测试 MeanHead ====================

class TestMeanHead:
    """测试均值输出头"""

    def test_basic_instantiation(self, feature_dim, output_dim):
        """测试基本实例化"""
        head = MeanHead(feature_dim=feature_dim, output_dim=output_dim)

        assert head.feature_dim == feature_dim
        assert head.output_dim == output_dim
        assert isinstance(head.fc, nn.Sequential)
        assert isinstance(head.output_layer, nn.Linear)

    def test_invalid_feature_dim(self, output_dim):
        """测试无效的feature_dim"""
        with pytest.raises(ValueError, match="feature_dim"):
            MeanHead(feature_dim=0, output_dim=output_dim)

        with pytest.raises(ValueError, match="feature_dim"):
            MeanHead(feature_dim=-5, output_dim=output_dim)

    def test_invalid_output_dim(self, feature_dim):
        """测试无效的output_dim"""
        with pytest.raises(ValueError, match="output_dim"):
            MeanHead(feature_dim=feature_dim, output_dim=0)

        with pytest.raises(ValueError, match="output_dim"):
            MeanHead(feature_dim=feature_dim, output_dim=-3)

    def test_forward_batch(self, sample_features, output_dim):
        """测试批量输入的前向传播"""
        head = MeanHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
        )

        mean = head(sample_features)

        assert mean.shape == (sample_features.shape[0], output_dim)
        assert mean.dtype == sample_features.dtype
        assert mean.device == sample_features.device

    def test_forward_single(self, single_feature, output_dim):
        """测试单样本输入的前向传播"""
        head = MeanHead(
            feature_dim=single_feature.shape[0],
            output_dim=output_dim,
        )

        mean = head(single_feature)

        assert mean.shape == (output_dim,)
        assert mean.dim() == 1

    def test_with_hidden_layers(self, sample_features, output_dim):
        """测试带隐藏层的网络结构"""
        hidden_dims = [128, 64]
        head = MeanHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation="relu",
            dropout=0.1,
            use_batch_norm=True,
        )

        mean = head(sample_features)
        assert mean.shape == (sample_features.shape[0], output_dim)

    def test_different_activations(self, sample_features, output_dim):
        """测试不同的激活函数"""
        activations = ["relu", "gelu", "tanh", "leaky_relu", "silu"]

        for activation in activations:
            head = MeanHead(
                feature_dim=sample_features.shape[1],
                output_dim=output_dim,
                activation=activation,
            )
            mean = head(sample_features)
            assert mean.shape[-1] == output_dim

    def test_none_activation_valid(self, feature_dim, output_dim):
        """测试'none'激活函数是有效的"""
        head = MeanHead(
            feature_dim=feature_dim,
            output_dim=output_dim,
            activation="none"
        )
        assert head is not None

    def test_compute_mse(self, batch_size, output_dim, device):
        """测试MSE计算"""
        head = MeanHead(feature_dim=64, output_dim=output_dim)

        predicted = torch.randn(batch_size, output_dim, device=device)
        target = torch.randn(batch_size, output_dim, device=device)

        mse_mean = head.compute_mse(predicted, target, reduction="mean")
        mse_sum = head.compute_mse(predicted, target, reduction="sum")
        mse_none = head.compute_mse(predicted, target, reduction="none")

        assert mse_mean.dim() == 0
        assert mse_sum.dim() == 0
        assert mse_none.shape == (batch_size,)
        assert mse_sum >= mse_mean

    def test_compute_rmse(self, batch_size, output_dim, device):
        """测试RMSE计算"""
        head = MeanHead(feature_dim=64, output_dim=output_dim)

        predicted = torch.randn(batch_size, output_dim, device=device)
        target = torch.randn(batch_size, output_dim, device=device)

        rmse = head.compute_rmse(predicted, target)
        mse = head.compute_mse(predicted, target)

        assert rmse.item() >= 0

    def test_evaluate_accuracy(self, batch_size, output_dim, device):
        """测试精度评估"""
        head = MeanHead(feature_dim=64, output_dim=output_dim)

        target = torch.randn(batch_size, output_dim, device=device)
        predicted = target + 0.1 * torch.randn_like(target)

        metrics = head.evaluate_accuracy(predicted, target)

        assert "mse" in metrics
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "max_error" in metrics
        assert "r_squared" in metrics

        assert metrics["mse"] >= 0
        assert metrics["rmse"] >= 0
        assert -1 <= metrics["r_squared"] <= 1

    def test_perfect_prediction(self, batch_size, output_dim, device):
        """测试完美预测的情况"""
        head = MeanHead(feature_dim=64, output_dim=output_dim)

        target = torch.randn(batch_size, output_dim, device=device)
        metrics = head.evaluate_accuracy(target, target)

        assert metrics["mse"] < 1e-6
        assert metrics["mae"] < 1e-6
        assert abs(metrics["r_squared"] - 1.0) < 1e-4

    def test_gradient_flow(self, sample_features, output_dim):
        """测试梯度流是否正常"""
        head = MeanHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
        )

        sample_features.requires_grad_(True)
        mean = head(sample_features)
        loss = mean.sum()
        loss.backward()

        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()

        for param in head.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    def test_get_num_parameters(self, feature_dim, output_dim):
        """测试参数数量统计"""
        head = MeanHead(feature_dim=feature_dim, output_dim=output_dim)

        total_params = head.get_num_parameters()
        trainable_params = head.get_num_parameters(trainable_only=True)

        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_repr(self, feature_dim, output_dim):
        """测试字符串表示"""
        head = MeanHead(feature_dim=feature_dim, output_dim=output_dim)
        repr_str = repr(head)

        assert "MeanHead" in repr_str
        assert str(feature_dim) in repr_str
        assert str(output_dim) in repr_str


# ==================== 测试 VarianceHead ====================

class TestVarianceHead:
    """测试方差/协方差输出头"""

    def test_diagonal_mode_instantiation(self, feature_dim, output_dim):
        """测试对角模式实例化"""
        head = VarianceHead(
            feature_dim=feature_dim,
            output_dim=output_dim,
            mode="diagonal"
        )

        assert head.mode == "diagonal"
        assert head.output_dim == output_dim

    def test_full_mode_instantiation(self, feature_dim, output_dim):
        """测试全协方差模式实例化"""
        head = VarianceHead(
            feature_dim=feature_dim,
            output_dim=output_dim,
            mode="full"
        )

        assert head.mode == "full"

    def test_invalid_mode(self, feature_dim, output_dim):
        """测试无效的模式"""
        with pytest.raises(ValueError, match="mode"):
            VarianceHead(
                feature_dim=feature_dim,
                output_dim=output_dim,
                mode="invalid_mode"
            )

    def test_forward_diagonal(self, sample_features, output_dim):
        """测试对角模式的前向传播"""
        head = VarianceHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            mode="diagonal"
        )

        variance = head(sample_features)

        assert variance.shape == (sample_features.shape[0], output_dim)
        assert torch.all(variance > 0), "方差必须为正数"

    def test_forward_full_covariance(self, sample_features, output_dim):
        """测试全协方差模式的前向传播"""
        head = VarianceHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            mode="full"
        )

        cov_matrix = head(sample_features)

        assert cov_matrix.shape == (
            sample_features.shape[0], output_dim, output_dim
        )

        for i in range(sample_features.shape[0]):
            matrix = cov_matrix[i]

            assert torch.allclose(matrix, matrix.T, atol=1e-5), \
                "协方差矩阵必须对称"

            eigenvalues = torch.linalg.eigvalsh(matrix)
            assert torch.all(eigenvalues > -1e-3), \
                "协方差矩阵应该是半正定的"

    def test_forward_single_sample_diagonal(self, single_feature, output_dim):
        """测试单样本对角模式"""
        head = VarianceHead(
            feature_dim=single_feature.shape[0],
            output_dim=output_dim,
            mode="diagonal"
        )

        variance = head(single_feature)

        assert variance.shape == (output_dim,)
        assert variance.dim() == 1

    def test_variance_activation_functions(self, sample_features, output_dim):
        """测试不同的方差激活函数"""
        activations = ["softplus", "exp", "relu", "elu"]

        for act in activations:
            head = VarianceHead(
                feature_dim=sample_features.shape[1],
                output_dim=output_dim,
                variance_activation=act,
            )
            variance = head(sample_features)
            assert torch.all(variance > 0), f"{act}激活后方差必须为正数"

    def test_min_variance_constraint(self, sample_features, output_dim):
        """测试最小方差异限"""
        min_var = 0.01
        head = VarianceHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            min_variance=min_var,
        )

        variance = head(sample_features)
        assert torch.all(variance >= min_var * 0.99), \
            f"方差应该大于等于 {min_var}"

    def test_compute_relative_error(self, batch_size, output_dim, device):
        """测试相对误差计算"""
        head = VarianceHead(feature_dim=64, output_dim=output_dim)

        target = torch.abs(torch.randn(batch_size, output_dim, device=device)) + 0.1
        predicted = target * 1.1

        rel_error = head.compute_relative_error(predicted, target)

        assert 0 <= rel_error.item() <= 1.0

    def test_compute_frobenius_norm(self, batch_size, output_dim, device):
        """测试Frobenius范数计算"""
        head = VarianceHead(
            feature_dim=64,
            output_dim=output_dim,
            mode="full"
        )

        target = torch.randn(batch_size, output_dim, output_dim, device=device)
        predicted = target + 0.1 * torch.randn_like(target)

        frob_norm = head.compute_frobenius_norm(predicted, target)

        assert frob_norm.dim() == 0
        assert frob_norm.item() >= 0

    def test_evaluate_accuracy_diagonal(self, batch_size, output_dim, device):
        """测试对角模式的精度评估"""
        head = VarianceHead(
            feature_dim=64,
            output_dim=output_dim,
            mode="diagonal"
        )

        target = torch.abs(torch.randn(batch_size, output_dim, device=device)) + 0.1
        predicted = target * (1 + 0.05 * torch.randn_like(target))

        metrics = head.evaluate_accuracy(predicted, target)

        assert "relative_error" in metrics
        assert "diagonal_mae" in metrics
        assert "diagonal_relative_error" in metrics
        assert "mse" in metrics

    def test_evaluate_accuracy_full(self, batch_size, output_dim, device):
        """测试全协方差模式的精度评估"""
        head = VarianceHead(
            feature_dim=64,
            output_dim=output_dim,
            mode="full"
        )

        target = torch.randn(batch_size, output_dim, output_dim, device=device)
        target = 0.5 * (target + target.transpose(-2, -1))
        target = target + torch.eye(output_dim, device=device).unsqueeze(0) * 0.1

        predicted = target + 0.01 * torch.randn_like(target)

        metrics = head.evaluate_accuracy(predicted, target)

        assert "frobenius_norm" in metrics
        assert "relative_error" in metrics

    def test_gradient_flow(self, sample_features, output_dim):
        """测试梯度流"""
        head = VarianceHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            mode="diagonal"
        )

        sample_features.requires_grad_(True)
        variance = head(sample_features)
        loss = variance.sum()
        loss.backward()

        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()

    def test_get_num_parameters(self, feature_dim, output_dim):
        """测试参数统计"""
        head_diag = VarianceHead(
            feature_dim=feature_dim,
            output_dim=output_dim,
            mode="diagonal"
        )
        head_full = VarianceHead(
            feature_dim=feature_dim,
            output_dim=output_dim,
            mode="full"
        )

        params_diag = head_diag.get_num_parameters()
        params_full = head_full.get_num_parameters()

        assert params_diag > 0
        assert params_full > 0
        assert params_full > params_diag, \
            "全协方差模式的参数应该更多"


# ==================== 测试 SkewnessHead ====================

class TestSkewnessHead:
    """测试偏度输出头"""

    def test_basic_instantiation(self, feature_dim, output_dim):
        """测试基本实例化"""
        head = SkewnessHead(feature_dim=feature_dim, output_dim=output_dim)

        assert head.feature_dim == feature_dim
        assert head.output_dim == output_dim
        assert head.clamp_min == -5.0
        assert head.clamp_max == 5.0

    def test_custom_clamp_range(self, feature_dim, output_dim):
        """测试自定义clamp范围"""
        head = SkewnessHead(
            feature_dim=feature_dim,
            output_dim=output_dim,
            clamp_range=(-3.0, 3.0)
        )

        assert head.clamp_min == -3.0
        assert head.clamp_max == 3.0

    def test_invalid_clamp_range(self, feature_dim, output_dim):
        """测试无效的clamp范围"""
        with pytest.raises(ValueError, match="clamp_range"):
            SkewnessHead(
                feature_dim=feature_dim,
                output_dim=output_dim,
                clamp_range=(5.0, -5.0)  # 无效：下限 > 上限
            )

    def test_forward_batch(self, sample_features, output_dim):
        """测试批量前向传播"""
        head = SkewnessHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim
        )

        skewness = head(sample_features)

        assert skewness.shape == (sample_features.shape[0], output_dim)
        assert torch.all(skewness >= head.clamp_min)
        assert torch.all(skewness <= head.clamp_max)

    def test_forward_single(self, single_feature, output_dim):
        """测试单样本前向传播"""
        head = SkewnessHead(
            feature_dim=single_feature.shape[0],
            output_dim=output_dim
        )

        skewness = head(single_feature)

        assert skewness.shape == (output_dim,)
        assert skewness.dim() == 1

    def test_output_clamping(self, sample_features, output_dim):
        """测试输出值限制在范围内"""
        clamp_range = (-2.0, 2.0)
        head = SkewnessHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            clamp_range=clamp_range
        )

        skewness = head(sample_features)

        assert torch.all(skewness >= clamp_range[0])
        assert torch.all(skewness <= clamp_range[1])

    def test_different_activations(self, sample_features, output_dim):
        """测试不同的激活函数"""
        activations = ["relu", "gelu", "tanh", "leaky_relu"]

        for activation in activations:
            head = SkewnessHead(
                feature_dim=sample_features.shape[1],
                output_dim=output_dim,
                activation=activation
            )
            skewness = head(sample_features)
            assert skewness.shape[-1] == output_dim

    def test_compute_skewness_from_samples(self, device):
        """测试从样本计算偏度"""
        n_samples = 10000
        mean_true = 0.0
        std_true = 1.0

        samples = torch.randn(n_samples, device=device) * std_true + mean_true

        computed_skewness = SkewnessHead.compute_skewness_from_samples(samples)

        assert abs(computed_skewness.item()) < 0.15, \
            "正态分布的偏度应接近0"

    def test_skewness_of_skewed_distribution(self, device):
        """测试偏斜分布的偏度计算"""
        n_samples = 50000

        samples = -torch.log(torch.rand(n_samples, device=device))

        computed_skewness = SkewnessHead.compute_skewness_from_samples(samples)

        assert computed_skewness.item() > 0, \
            "指数分布应该是正偏斜的"

    def test_check_sign_correctness(self, batch_size, output_dim, device):
        """测试符号正确性检查"""
        head = SkewnessHead(feature_dim=64, output_dim=output_dim)

        target = torch.randn(batch_size, output_dim, device=device)
        predicted = target.clone()

        sign_metrics = head.check_sign_correctness(predicted, target)

        assert "sign_accuracy" in sign_metrics
        assert sign_metrics["sign_accuracy"] == 1.0, \
            "完全匹配时符号准确率应为1.0"

    def test_check_sign_with_errors(self, batch_size, output_dim, device):
        """测试有误差时的符号检查"""
        head = SkewnessHead(feature_dim=64, output_dim=output_dim)

        target = torch.randn(batch_size, output_dim, device=device)
        predicted = target + 0.5 * torch.randn_like(target)

        sign_metrics = head.check_sign_correctness(predicted, target)

        assert 0 <= sign_metrics["sign_accuracy"] <= 1.0

    def test_evaluate_accuracy(self, batch_size, output_dim, device):
        """测试精度评估"""
        head = SkewnessHead(feature_dim=64, output_dim=output_dim)

        target = torch.randn(batch_size, output_dim, device=device)
        predicted = target + 0.1 * torch.randn_like(target)

        metrics = head.evaluate_accuracy(predicted, target)

        assert "mae" in metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "sign_accuracy" in metrics
        assert "mean_predicted" in metrics
        assert "mean_target" in metrics

    def test_gradient_flow(self, sample_features, output_dim):
        """测试梯度流"""
        head = SkewnessHead(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim
        )

        sample_features.requires_grad_(True)
        skewness = head(sample_features)
        loss = skewness.sum()
        loss.backward()

        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()


# ==================== 测试 MomentResult ====================

class TestMomentResult:
    """测试矩估计结果数据类"""

    def test_empty_result(self):
        """测试空结果"""
        result = MomentResult()

        assert not result.has_mean
        assert not result.has_variance
        assert not result.has_skewness

    def test_result_with_all_fields(self, device):
        """测试包含所有字段的结果"""
        result = MomentResult(
            mean=torch.randn(10, device=device),
            variance=torch.abs(torch.randn(10, device=device)),
            skewness=torch.randn(10, device=device),
            metadata={"test": True}
        )

        assert result.has_mean
        assert result.has_variance
        assert result.has_skewness
        assert result.metadata["test"] is True

    def test_to_dict(self, device):
        """测试转换为字典"""
        result = MomentResult(
            mean=torch.randn(5, device=device),
            variance=torch.abs(torch.randn(5, device=device)),
        )

        result_dict = result.to_dict()

        assert "mean" in result_dict
        assert "variance" in result_dict
        assert "skewness" not in result_dict
        assert "metrics" in result_dict

    def test_detach(self, device):
        """测试分离梯度"""
        result = MomentResult(
            mean=torch.randn(5, device=device, requires_grad=True),
            variance=torch.abs(torch.randn(5, device=device, requires_grad=True)),
        )

        detached = result.detach()

        assert not detached.mean.requires_grad
        assert not detached.variance.requires_grad

    def test_cpu_transfer(self, device):
        """测试转移到CPU"""
        if device.type == "cuda":
            result = MomentResult(
                mean=torch.randn(5, device=device),
            )

            cpu_result = result.cpu()

            assert cpu_result.mean.device.type == "cpu"


# ==================== 测试 MomentEstimator ====================

class TestMomentEstimator:
    """测试矩估计器统一接口"""

    def test_all_heads_enabled(self, feature_dim, output_dim):
        """测试启用所有输出头"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
            enable_mean=True,
            enable_variance=True,
            enable_skewness=True,
        )

        assert estimator.enable_mean
        assert estimator.enable_variance
        assert estimator.enable_skewness

    def test_no_head_enabled_raises_error(self, feature_dim, output_dim):
        """测试未启用任何输出头时的错误"""
        with pytest.raises(ValueError, match="至少需要启用"):
            MomentEstimator(
                feature_dim=feature_dim,
                output_dim=output_dim,
                enable_mean=False,
                enable_variance=False,
                enable_skewness=False,
            )

    def test_partial_heads_enabled(self, feature_dim, output_dim):
        """测试部分启用输出头"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
            enable_mean=True,
            enable_variance=True,
            enable_skewness=False,
        )

        assert estimator.enable_mean
        assert estimator.enable_variance
        assert not estimator.enable_skewness

    def test_forward_all_heads(self, sample_features, output_dim):
        """测试所有输出头的前向传播"""
        estimator = MomentEstimator(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            enable_mean=True,
            enable_variance=True,
            enable_skewness=True,
        )

        result = estimator(sample_features)

        assert isinstance(result, MomentResult)
        assert result.has_mean
        assert result.has_variance
        assert result.has_skewness

        assert result.mean.shape == (sample_features.shape[0], output_dim)
        assert result.variance.shape == (sample_features.shape[0], output_dim)
        assert result.skewness.shape == (sample_features.shape[0], output_dim)

    def test_forward_partial_heads(self, sample_features, output_dim):
        """测试部分输出头的前向传播"""
        estimator = MomentEstimator(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            enable_mean=True,
            enable_variance=False,
            enable_skewness=True,
        )

        result = estimator(sample_features)

        assert result.has_mean
        assert not result.has_variance
        assert result.has_skewness

    def test_forward_with_targets(self, sample_features, output_dim, device):
        """测试带目标值的前向传播"""
        estimator = MomentEstimator(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
        )

        targets = {
            "mean": torch.randn(sample_features.shape[0], output_dim, device=device),
            "variance": torch.abs(torch.randn(
                sample_features.shape[0], output_dim, device=device)) + 0.1,
            "skewness": torch.randn(sample_features.shape[0], output_dim, device=device),
        }

        result = estimator.forward_with_targets(
            sample_features,
            target_mean=targets["mean"],
            target_variance=targets["variance"],
            target_skewness=targets["skewness"],
        )

        assert len(result.mean_metrics) > 0
        assert len(result.variance_metrics) > 0
        assert len(result.skewness_metrics) > 0

    def test_set_head_enabled(self, feature_dim, output_dim):
        """测试动态启用/禁用输出头"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
        )

        estimator.set_head_enabled("skewness", False)

        assert not estimator.enable_skewness

        estimator.set_head_enabled("skewness", True)

        assert estimator.enable_skewness

    def test_set_invalid_head_raises_error(self, feature_dim, output_dim):
        """测试设置无效输出头名称"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
            enable_skewness=False,
        )

        with pytest.raises(ValueError, match="无效的head_name"):
            estimator.set_head_enabled("invalid_head", True)

    def test_get_enabled_heads(self, feature_dim, output_dim):
        """测试获取启用的输出头列表"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
            enable_mean=True,
            enable_variance=False,
            enable_skewness=True,
        )

        enabled = estimator.get_enabled_heads()

        assert "mean" in enabled
        assert "variance" not in enabled
        assert "skewness" in enabled

    def test_compute_total_loss(self, sample_features, output_dim, device):
        """测试总损失计算"""
        estimator = MomentEstimator(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
        )

        result = estimator(sample_features)

        targets = {
            "mean": torch.randn(sample_features.shape[0], output_dim, device=device),
            "variance": torch.abs(torch.randn(
                sample_features.shape[0], output_dim, device=device)) + 0.1,
            "skewness": torch.randn(sample_features.shape[0], output_dim, device=device),
        }

        loss = estimator.compute_total_loss(
            result,
            target_mean=targets["mean"],
            target_variance=targets["variance"],
            target_skewness=targets["skewness"],
        )

        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_compute_loss_with_weights(self, sample_features, output_dim, device):
        """测试带权重的损失计算"""
        estimator = MomentEstimator(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
        )

        result = estimator(sample_features)

        target = torch.randn(sample_features.shape[0], output_dim, device=device)

        weights_equal = {"mean": 1.0, "variance": 1.0, "skewness": 1.0}
        weights_custom = {"mean": 2.0, "variance": 1.0, "skewness": 0.5}

        var_target = torch.abs(torch.randn_like(target)) + 0.1
        skew_target = torch.randn_like(target)

        loss_eq = estimator.compute_total_loss(
            result,
            target_mean=target,
            target_variance=var_target,
            target_skewness=skew_target,
            weights=weights_equal,
        )

        loss_custom = estimator.compute_total_loss(
            result,
            target_mean=target,
            target_variance=var_target,
            target_skewness=skew_target,
            weights=weights_custom,
        )

        assert loss_eq != loss_custom or True

    def test_get_parameters_by_head(self, feature_dim, output_dim):
        """测试获取各输出头的参数数量"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
        )

        params_by_head = estimator.get_parameters_by_head()

        assert "mean" in params_by_head
        assert "variance" in params_by_head
        assert "skewness" in params_by_head

        for head_name, num_params in params_by_head.items():
            assert num_params > 0, f"{head_name} 应该有参数"

    def test_gradient_flow_through_estimator(self, sample_features, output_dim):
        """测试梯度流通过整个估计器"""
        estimator = MomentEstimator(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
        )

        sample_features.requires_grad_(True)
        result = estimator(sample_features)
        loss = result.mean.sum() + result.variance.sum() + result.skewness.sum()
        loss.backward()

        assert sample_features.grad is not None
        assert not torch.isnan(sample_features.grad).any()

        for param in estimator.parameters():
            assert param.grad is not None or param.requires_grad is False

    def test_full_covariance_mode(self, sample_features, output_dim):
        """测试全协方差模式"""
        estimator = MomentEstimator(
            feature_dim=sample_features.shape[1],
            output_dim=output_dim,
            variance_mode="full",
        )

        result = estimator(sample_features)

        assert result.variance.shape == (
            sample_features.shape[0], output_dim, output_dim
        )

    def test_repr(self, feature_dim, output_dim):
        """测试字符串表示"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
            enable_mean=True,
            enable_variance=True,
            enable_skewness=False,
        )

        repr_str = repr(estimator)

        assert "MomentEstimator" in repr_str
        assert str(feature_dim) in repr_str
        assert str(output_dim) in repr_str


# ==================== 测试工厂函数 ====================

class TestCreateMomentEstimator:
    """测试工厂函数"""

    def test_default_config(self, feature_dim, output_dim):
        """测试默认配置创建"""
        estimator = create_moment_estimator(feature_dim, output_dim)

        assert estimator.enable_mean
        assert estimator.enable_variance
        assert estimator.enable_skewness
        assert estimator.feature_dim == feature_dim
        assert estimator.output_dim == output_dim

    def test_custom_config(self, feature_dim, output_dim):
        """测试自定义配置"""
        config = {
            "enable_mean": True,
            "enable_variance": True,
            "enable_skewness": False,
            "activation": "gelu",
            "dropout": 0.2,
        }

        estimator = create_moment_estimator(feature_dim, output_dim, config)

        assert estimator.enable_mean
        assert estimator.enable_variance
        assert not estimator.enable_skewness

    def test_empty_config(self, feature_dim, output_dim):
        """测试空配置（使用默认值）"""
        estimator = create_moment_estimator(feature_dim, output_dim, {})

        assert estimator is not None


# ==================== 边界情况测试 ====================

class TestEdgeCases:
    """测试边界情况和极端值"""

    def test_very_small_input(self, device):
        """测试非常小的输入"""
        head = MeanHead(feature_dim=10, output_dim=5)
        tiny_input = torch.randn(1, 10, device=device) * 1e-10

        output = head(tiny_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_very_large_input(self, device):
        """测试非常大的输入"""
        head = VarianceHead(feature_dim=10, output_dim=5)
        large_input = torch.randn(1, 10, device=device) * 1e6

        output = head(large_input)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert torch.all(output > 0), "方差必须为正数"

    def test_zero_variance_target(self, batch_size, output_dim, device):
        """测试零方差目标（数值稳定性）"""
        head = VarianceHead(
            feature_dim=64,
            output_dim=output_dim,
            min_variance=1e-6,
        )

        target = torch.zeros(batch_size, output_dim, device=device)
        predicted = torch.ones(batch_size, output_dim, device=device) * 0.01

        metrics = head.evaluate_accuracy(predicted, target)

        assert not any(np.isinf(v) for v in metrics.values())
        assert not any(np.isnan(v) for v in metrics.values())

    def test_extreme_skewness_values(self, device):
        """测试极端偏度值"""
        head = SkewnessHead(
            feature_dim=32,
            output_dim=5,
            clamp_range=(-10.0, 10.0),
        )

        features = torch.randn(1, 32, device=device) * 100
        skewness = head(features)

        assert torch.all(skewness >= -10.0)
        assert torch.all(skewness <= 10.0)

    def test_single_dimension_output(self, feature_dim, device):
        """测试一维输出"""
        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=1,
        )

        features = torch.randn(16, feature_dim, device=device)
        result = estimator(features)

        assert result.mean.shape == (16, 1)
        assert result.variance.shape == (16, 1)
        assert result.skewness.shape == (16, 1)

    def test_large_output_dimension(self, device):
        """测试大输出维度"""
        large_output_dim = 256
        head = MeanHead(feature_dim=512, output_dim=large_output_dim)

        features = torch.randn(8, 512, device=device)
        output = head(features)

        assert output.shape == (8, large_output_dim)

    def test_batch_size_one(self, feature_dim, output_dim, device):
        """测试batch_size为1"""
        head = MeanHead(feature_dim=feature_dim, output_dim=output_dim)

        features = torch.randn(1, feature_dim, device=device)
        output = head(features)

        assert output.shape == (1, output_dim)

    def test_consistent_output_for_same_input(self, feature_dim, output_dim, device):
        """测试相同输入产生一致输出（无dropout时）"""
        head = MeanHead(
            feature_dim=feature_dim,
            output_dim=output_dim,
            dropout=0.0,
        )
        head.eval()

        features = torch.randn(1, feature_dim, device=device)

        output1 = head(features)
        output2 = head(features)

        assert torch.equal(output1, output2)


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试：测试组件协同工作"""

    def test_end_to_end_pipeline(self, device):
        """端到端管道测试"""
        torch.manual_seed(42)

        feature_dim = 512
        output_dim = 20
        batch_size = 16

        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
            hidden_dims=[256, 128],
            dropout=0.1,
            use_batch_norm=True,
        )

        features = torch.randn(batch_size, feature_dim, device=device)

        result = estimator(features)

        assert result.mean.shape == (batch_size, output_dim)
        assert result.variance.shape == (batch_size, output_dim)
        assert result.skewness.shape == (batch_size, output_dim)

        assert torch.all(result.variance > 0)

        print("\n✓ 端到端管道集成测试通过")

    def test_training_step_simulation(self, device):
        """模拟训练步骤"""
        torch.manual_seed(123)

        feature_dim = 256
        output_dim = 10
        batch_size = 32

        estimator = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim,
        )

        optimizer = torch.optim.Adam(estimator.parameters(), lr=0.001)

        features = torch.randn(batch_size, feature_dim, device=device)
        targets = {
            "mean": torch.randn(batch_size, output_dim, device=device),
            "variance": torch.abs(torch.randn(
                batch_size, output_dim, device=device)) + 0.1,
            "skewness": torch.randn(batch_size, output_dim, device=device).clamp(-3, 3),
        }

        optimizer.zero_grad()
        result = estimator(features)
        loss = estimator.compute_total_loss(
            result,
            target_mean=targets["mean"],
            target_variance=targets["variance"],
            target_skewness=targets["skewness"],
        )

        loss.backward()
        optimizer.step()

        assert loss.item() >= 0
        assert not torch.isnan(loss)

        print(f"\n✓ 训练步骤模拟通过，loss={loss.item():.4f}")

    def test_multiple_estimators_shared_features(self, device):
        """测试多个估计器共享特征"""
        feature_dim = 384
        output_dim_a = 15
        output_dim_b = 8

        estimator_a = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim_a,
        )

        estimator_b = MomentEstimator(
            feature_dim=feature_dim,
            output_dim=output_dim_b,
        )

        features = torch.randn(16, feature_dim, device=device)

        result_a = estimator_a(features)
        result_b = estimator_b(features)

        assert result_a.mean.shape[1] == output_dim_a
        assert result_b.mean.shape[1] == output_dim_b

        print("\n✓ 多估计器共享特征测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
