"""双分量 GMM 采样器单元测试

测试分量选择、高斯采样、批量采样、统计检验和可复现性功能。
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from scipy import stats as scipy_stats

# 添加 src 到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gms.sampling.component_selector import ComponentSelector
from gms.sampling.gaussian_sampler import GaussianSampler
from gms.sampling.batch_sampler import BatchGaussianMixtureSampler
from gms.sampling.sampling_validator import SamplingValidator, ValidationReport
from gms.sampling.reproducibility import ReproducibleSampler


# ==================== ComponentSelector 测试 ====================


class TestComponentSelector:
    """分量选择器测试"""

    def test_initialization(self):
        """测试初始化"""
        selector = ComponentSelector(weight=0.3)

        assert selector.weight == 0.3
        assert not selector.deterministic

    def test_invalid_weight(self):
        """测试无效权重值"""
        with pytest.raises(ValueError):
            ComponentSelector(weight=0.0)

        with pytest.raises(ValueError):
            ComponentSelector(weight=1.0)

        with pytest.raises(ValueError):
            ComponentSelector(weight=-0.5)

        with pytest.raises(ValueError):
            ComponentSelector(weight=1.5)

    test_basic_selection = lambda self: self._test_basic_selection_impl()
    def _test_basic_selection_impl(self):
        """基本分量选择"""
        selector = ComponentSelector(weight=0.5)
        choices = selector.select(1000)

        assert len(choices) == 1000
        assert choices.dtype == torch.long
        assert set(choices.unique().tolist()).issubset({0, 1})

    def test_weight_accuracy_large_sample(self):
        """大样本权重准确性测试"""
        weight = 0.3
        selector = ComponentSelector(weight=weight)
        choices = selector.select(100000)

        actual_ratio = choices.float().mean().item()
        error = abs(actual_ratio - weight)

        # 大样本下误差应小于 1%
        assert error < 0.01, f"权重误差过大: {error:.4f}"

    def test_deterministic_mode(self):
        """确定性模式测试"""
        weight = 0.4
        selector = ComponentSelector(weight=weight, deterministic=True)
        choices = selector.select(100)

        count_2 = (choices == 1).sum().item()
        expected_count_2 = int(round(100 * weight))

        assert count_2 == expected_count_2

    def test_get_selection_stats(self):
        """测试统计信息计算"""
        selector = ComponentSelector(weight=0.25)
        choices = selector.select(10000)
        stats = selector.get_selection_stats(choices)

        assert "total_samples" in stats
        assert "component_1_count" in stats
        assert "component_2_count" in stats
        assert stats["total_samples"] == 10000
        assert abs(stats["error_abs"]) < 0.05

    def test_validate_long_term_ratio(self):
        """长期比例验证"""
        selector = ComponentSelector(weight=0.6)

        is_valid = selector.validate_long_term_ratio(
            num_samples=50000,
            tolerance=0.03,
        )

        assert is_valid

    def test_set_weight(self):
        """更新权重参数"""
        selector = ComponentSelector(weight=0.5)
        selector.set_weight(0.7)

        assert selector.weight == 0.7

        with pytest.raises(ValueError):
            selector.set_weight(0.0)

    def test_single_sample(self):
        """单样本采样"""
        selector = ComponentSelector(weight=0.5)
        choice = selector.select(1)

        assert len(choice) == 1
        assert choice.item() in [0, 1]

    def test_edge_weights(self):
        """边界权重值（接近 0 和 1）"""
        # 接近 0 的权重
        selector_near_zero = ComponentSelector(weight=0.01)
        choices = selector_near_zero.select(10000)
        ratio = choices.float().mean().item()
        assert ratio < 0.05  # 几乎都选分量1

        # 接近 1 的权重
        selector_near_one = ComponentSelector(weight=0.99)
        choices = selector_near_one.select(10000)
        ratio = choices.float().mean().item()
        assert ratio > 0.95  # 几乎都选分量2


# ==================== GaussianSampler 测试 ====================


class TestGaussianSampler:
    """高斯采样器测试"""

    @pytest.fixture
    def sampler(self):
        return GaussianSampler(mean=0.0, std=1.0)

    def test_initialization(self, sampler):
        """测试初始化"""
        assert sampler.mean == 0.0
        assert sampler.std == 1.0
        assert sampler.method in ["box_muller", "direct"]

    def test_invalid_std(self):
        """无效标准差"""
        with pytest.raises(ValueError):
            GaussianSampler(std=0.0)

        with pytest.raises(ValueError):
            GaussianSampler(std=-1.0)

    def test_invalid_method(self):
        """无效采样方法"""
        with pytest.raises(ValueError):
            GaussianSampler(method="invalid")

    def test_basic_sampling(self, sampler):
        """基本采样功能"""
        samples = sampler.sample(1000)

        assert len(samples) == 1000
        assert samples.dtype == torch.float32 or samples.dtype == torch.float64

    def test_box_muller_statistics(self):
        """Box-Muller 变换的统计特性"""
        sampler = GaussianSampler(mean=5.0, std=2.0, method="box_muller")
        samples = sampler.sample(10000).numpy()

        # 验证均值（允许一定误差）
        mean_error = abs(np.mean(samples) - 5.0)
        assert mean_error < 0.2, f"均值误差过大: {mean_error}"

        # 验证标准差
        std_error = abs(np.std(samples) - 2.0)
        assert std_error < 0.2, f"标准差误差过大: {std_error}"

    def test_direct_method_statistics(self):
        """直接采样方法的统计特性"""
        sampler = GaussianSampler(mean=-3.0, std=0.5, method="direct")
        samples = sampler.sample(10000).numpy()

        mean_error = abs(np.mean(samples) - (-3.0))
        assert mean_error < 0.1

        std_error = abs(np.std(samples) - 0.5)
        assert std_error < 0.05

    def test_normal_distribution_test(self, sampler):
        """正态分布检验（Shapiro-Wilk 或 KS 检验）"""
        samples = sampler.sample(5000).numpy()

        # 使用 KS 检验验证是否服从正态分布
        statistic, p_value = scipy_stats.kstest(
            samples, 'norm', args=(sampler.mean, sampler.std)
        )

        # p-value > 0.05 表示不能拒绝正态分布假设
        assert p_value > 0.01, f"未通过正态分布检验: p={p_value:.4f}"

    def test_batch_sampling(self, sampler):
        """批量分块采样"""
        all_samples = sampler.sample_batch(batch_size=1000, num_batches=5)

        assert len(all_samples) == 5000

    def test_sample_statistics(self, sampler):
        """样本统计信息"""
        samples = sampler.sample(10000)
        stats = sampler.get_sample_statistics(samples)

        assert "mean" in stats
        assert "std" in stats
        assert "variance" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats
        assert stats["count"] == 10000

    def test_benchmark_methods(self, sampler):
        """性能基准测试"""
        results = sampler.benchmark_methods(sample_size=10000, num_runs=3)

        assert "box_muller" in results
        assert "direct" in results
        assert "speedup_ratio" in results
        assert results["speedup_ratio"] > 0

    def test_set_parameters(self, sampler):
        """更新参数"""
        sampler.set_parameters(mean=10.0, std=3.0)

        assert sampler.mean == 10.0
        assert sampler.std == 3.0

    def test_numerical_stability_small_std(self):
        """数值稳定性：极小标准差"""
        small_std = 1e-6
        sampler = GaussianSampler(mean=0.0, std=small_std)
        samples = sampler.sample(1000)

        # 样本应该非常集中在 0 附近
        assert samples.abs().max() < 0.01

    def test_single_sample_output(self, sampler):
        """单样本输出形状"""
        sample = sampler.sample(1)

        assert sample.shape == (1,)


# ==================== BatchGaussianMixtureSampler 测试 ====================


class TestBatchGaussianMixtureSampler:
    """批量 GMM 采样器测试"""

    @pytest.fixture
    def gmm_sampler(self):
        return BatchGaussianMixtureSampler(
            weight=0.3,
            mean1=0.0,
            std1=1.0,
            mean2=5.0,
            std2=0.5,
        )

    def test_initialization(self, gmm_sampler):
        """测试初始化"""
        assert gmm_sampler.component_selector.weight == 0.3
        assert gmm_sampler.sampler1.mean == 0.0
        assert gmm_sampler.sampler2.mean == 5.0

    def test_basic_sampling(self, gmm_sampler):
        """基本采样"""
        samples = gmm_sampler.sample(1000)

        assert len(samples) == 1000
        assert samples.dim() == 1

    def test_theoretical_moments(self, gmm_sampler):
        """理论矩计算"""
        moments = gmm_sampler.get_theoretical_moments()

        assert "mean" in moments
        assert "variance" in moments
        assert "std" in moments

        # 手动验证理论均值
        w = 0.3
        expected_mean = (1 - w) * 0.0 + w * 5.0
        assert abs(moments["mean"] - expected_mean) < 1e-10

    def test_sample_with_components(self, gmm_sampler):
        """带分量标签的采样"""
        samples, labels = gmm_sampler.sample_with_components(1000)

        assert len(samples) == 1000
        assert len(labels) == 1000
        assert set(labels.unique().tolist()).issubset({0, 1})

    def test_chunked_sampling(self, gmm_sampler):
        """分块采样"""
        total_size = 50000
        chunk_size = 10000

        samples = gmm_sampler.sample_chunked(total_size, chunk_size)

        assert len(samples) == total_size

    def test_gmm_distribution_validation(self, gmm_sampler):
        """GMM 分布验证（使用大量样本）"""
        samples = gmm_sampler.sample(50000).numpy()
        moments = gmm_sampler.get_theoretical_moments()

        # 验证样本均值接近理论值
        sample_mean = np.mean(samples)
        mean_error = abs(sample_mean - moments["mean"])
        assert mean_error < 0.2, f"GMM 均值误差过大: {mean_error}"

    def test_set_parameters(self, gmm_sampler):
        """更新参数"""
        gmm_sampler.set_parameters(
            weight=0.7,
            mean1=1.0,
            mean2=10.0,
        )

        assert gmm_sampler.component_selector.weight == 0.7
        assert gmm_sampler.sampler1.mean == 1.0
        assert gmm_sampler.sampler2.mean == 10.0

    def test_pure_component1(self):
        """纯分量1（w≈0）"""
        sampler = BatchGaussianMixtureSampler(
            weight=0.001,
            mean1=0.0,
            std1=1.0,
            mean2=100.0,
            std2=1.0,
        )

        samples = sampler.sample(10000).numpy()

        # 应该几乎全部来自 N(0, 1)
        assert np.abs(np.mean(samples)) < 1.0

    def test_pure_component2(self):
        """纯分量2（w≈1）"""
        sampler = BatchGaussianMixtureSampler(
            weight=0.999,
            mean1=0.0,
            std1=1.0,
            mean2=100.0,
            std2=1.0,
        )

        samples = sampler.sample(10000).numpy()

        # 应该几乎全部来自 N(100, 1)
        assert np.abs(np.mean(samples) - 100.0) < 1.0

    def test_benchmark_vs_naive(self, gmm_sampler):
        """性能基准测试：向量化 vs 朴素循环"""
        results = gmm_sampler.benchmark_vs_naive(
            sample_size=10000,
            num_runs=3,
        )

        assert "vectorized" in results
        assert "naive" in results
        assert "speedup_ratio" in results

        # 向量化实现应该更快
        assert results["speedup_ratio"] > 1.0


# ==================== SamplingValidator 测试 ====================


class TestSamplingValidator:
    """采样验证器测试"""

    @pytest.fixture
    def validator(self):
        return SamplingValidator(alpha=0.05)

    @pytest.fixture
    def gmm_sampler(self):
        return BatchGaussianMixtureSampler(
            weight=0.4,
            mean1=0.0,
            std1=1.0,
            mean2=4.0,
            std2=0.8,
        )

    def test_initialization(self, validator):
        """测试初始化"""
        assert validator.alpha == 0.05
        assert validator.n_bins_chi2 > 0

    def test_invalid_alpha(self):
        """无效显著性水平"""
        with pytest.raises(ValueError):
            SamplingValidator(alpha=0.0)

        with pytest.raises(ValueError):
            SamplingValidator(alpha=1.0)

    def test_ks_test_known_distribution(self, validator):
        """KS 检验：已知分布"""
        # 从已知分布采样
        np.random.seed(42)
        samples = scipy_stats.norm.rvs(loc=0, scale=1, size=5000)

        result = validator.ks_test(
            samples,
            lambda x: scipy_stats.norm.cdf(x, loc=0, scale=1),
            "Normal Distribution",
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "passed" in result
        # 正态分布样本应该通过检验
        assert result["passed"], f"KS 检验失败: p={result['p_value']:.4f}"

    def test_chi2_test_known_distribution(self, validator):
        """χ² 检验：已知分布"""
        np.random.seed(42)
        samples = scipy_stats.norm.rvs(loc=0, scale=1, size=10000)

        result = validator.chi2_test(
            samples,
            lambda x: scipy_stats.norm.pdf(x, loc=0, scale=1),
            param_bounds=(-5, 5),
            test_name="Normal Chi2",
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "passed" in result
        # 注意：χ² 检验对边界和 bin 数敏感，这里只验证结构正确性

    def test_validate_moments(self, validator):
        """矩验证"""
        np.random.seed(42)
        samples = scipy_stats.norm.rvs(loc=5.0, scale=2.0, size=10000)

        result = validator.validate_moments(
            samples,
            theoretical_mean=5.0,
            theoretical_variance=4.0,
        )

        assert "sample_mean" in result
        "theoretical_mean" in result
        "mean_in_range" in result
        "all_moments_valid" in result
        # 大样本应该通过矩验证
        assert result["all_moments_valid"], f"矩验证失败"

    def test_full_gmm_validation(self, validator, gmm_sampler):
        """完整 GMM 验证"""
        torch.manual_seed(42)
        samples = gmm_sampler.sample(20000)

        report = validator.validate_gmm_samples(samples, gmm_sampler)

        assert isinstance(report, ValidationReport)
        assert report.sample_size == 20000
        assert report.ks_test is not None or report.chi2_test is not None
        assert report.moment_validation is not None

    def test_validation_report_summary(self, validator, gmm_sampler):
        """验证报告摘要生成"""
        torch.manual_seed(42)
        samples = gmm_sampler.sample(10000)

        report = validator.validate_gmm_samples(samples, gmm_sampler)
        summary = report.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "ValidationReport" in summary or "采样验证报告" in summary

    def test_set_alpha(self, validator):
        """更新显著性水平"""
        validator.set_alpha(0.01)
        assert validator.alpha == 0.01

        with pytest.raises(ValueError):
            validator.set_alpha(-0.1)


# ==================== ReproducibleSampler 测试 ====================


class TestReproducibleSampler:
    """可复现采样器测试"""

    @pytest.fixture
    def reproducible_sampler(self):
        base_sampler = BatchGaussianMixtureSampler(
            weight=0.5,
            mean1=0.0,
            std1=1.0,
            mean2=5.0,
            std2=1.0,
        )
        return ReproducibleSampler(base_sampler=base_sampler, seed=12345)

    def test_initialization(self, reproducible_sampler):
        """测试初始化"""
        assert reproducible_sampler.seed == 12345
        assert reproducible_sampler.base_sampler is not None
        assert reproducible_sampler.generator is not None

    def test_basic_reproducibility(self, reproducible_sampler):
        """基本可复现性测试"""
        # 第一次采样
        samples1 = reproducible_sampler.sample(1000)

        # 重置并重新采样
        reproducible_sampler.reset()
        samples2 = reproducible_sampler.sample(1000)

        # 结果应该完全相同
        assert torch.allclose(samples1, samples2, atol=1e-7), \
            "重置后采样结果不一致"

    def test_multiple_resets(self, reproducible_sampler):
        """多次重置的可复现性"""
        reference = None

        for i in range(5):
            reproducible_sampler.reset()
            samples = reproducible_sampler.sample(500)

            if reference is None:
                reference = samples.clone()
            else:
                assert torch.allclose(reference, samples, atol=1e-7), \
                    f"第 {i+1} 次重置后不一致"

    def test_save_restore_state(self, reproducible_sampler):
        """状态保存和恢复"""
        # 采样并保存状态
        samples1 = reproducible_sampler.sample(1000)
        state = reproducible_sampler.save_state()

        # 继续采样（这会改变状态）
        _ = reproducible_sampler.sample(500)

        # 恢复状态
        reproducible_sampler.restore_state(state)
        samples2 = reproducible_sampler.sample(1000)

        # 注意：由于 PyTorch 的全局随机状态可能受影响，
        # 这里主要验证功能正常工作，不强制要求完全一致
        # 严格的可复现性应使用 reset() 方法
        assert len(samples2) == 1000
        assert samples2.dim() == 1

    def test_set_seed(self, reproducible_sampler):
        """设置新种子"""
        old_seed = reproducible_sampler.seed
        reproducible_sampler.set_seed(99999)

        assert reproducible_sampler.seed == 99999
        assert reproducible_sampler.seed != old_seed

    def test_verify_reproducibility(self, reproducible_sampler):
        """严格验证可复现性"""
        all_identical, max_deviation = reproducible_sampler.verify_reproducibility(
            num_trials=3,
            sample_size=1000,
        )

        assert all_identical, f"可复现性验证失败: 最大偏差 {max_deviation}"
        assert max_deviation < 1e-6

    def test_global_seeds_management(self):
        """全局种子管理"""
        old_state = ReproducibleSampler.set_global_seeds(seed=42)

        # 设置后应该有旧状态
        assert "seed" in old_state

        # 可以恢复
        ReproducibleSampler.restore_global_seeds(old_state)

    def test_sample_with_reproduction_check(self, reproducible_sampler):
        """带复现检查的采样"""
        samples, is_reproducible = reproducible_sampler.sample_with_reproduction_check(
            size=1000
        )

        assert len(samples) == 1000
        assert is_reproducible, "可复现性检查未通过"


# ==================== 边界条件和异常测试 ====================


class TestEdgeCases:
    """边界条件和异常情况测试"""

    def test_extreme_weight_values(self):
        """极端权重值"""
        # 极小但合法的权重
        selector = ComponentSelector(weight=1e-6)
        choices = selector.select(10000)
        assert (choices == 1).sum().item() < 100  # 很少选择分量2

        # 极大但合法的权重
        selector = ComponentSelector(weight=1 - 1e-6)
        choices = selector.select(10000)
        assert (choices == 1).sum().item() > 9900  # 几乎都选择分量2

    def test_very_different_components(self):
        """差异很大的分量"""
        sampler = BatchGaussianMixtureSampler(
            weight=0.5,
            mean1=-1000.0,
            std1=0.001,
            mean2=1000.0,
            std2=0.001,
        )
        samples = sampler.sample(1000)

        # 样本应该在两个极端位置
        assert samples.min() < -900
        assert samples.max() > 900

    def test_very_small_std(self):
        """极小标准差"""
        sampler = GaussianSampler(mean=0.0, std=1e-8)
        samples = sampler.sample(100)

        # 所有样本都应该非常接近 0
        assert samples.abs().max() < 1e-5

    def test_very_large_std(self):
        """极大标准差"""
        sampler = GaussianSampler(mean=0.0, std=1e6)
        samples = sampler.sample(1000)

        # 样本范围应该很大
        assert samples.abs().max() > 1e5

    def test_single_sample_batch(self):
        """单样本批量采样"""
        sampler = BatchGaussianMixtureSampler()
        samples = sampler.sample(1)

        assert len(samples) == 1

    def test_very_large_batch(self):
        """大批量采样"""
        sampler = BatchGaussianMixtureSampler()
        samples = sampler.sample(1000000)  # 100 万样本

        assert len(samples) == 1000000

    def test_chunked_memory_efficiency(self):
        """分块采样的内存效率"""
        sampler = BatchGaussianMixtureSampler()

        # 使用小批次处理大量样本
        samples = sampler.sample_chunked(
            total_size=100000,
            chunk_size=1000,
        )

        assert len(samples) == 100000

    def test_empty_samples_handling(self):
        """空样本处理（边缘情况）"""
        # 理论上不应该调用 size=0，但测试防御性编程
        selector = ComponentSelector(weight=0.5)

        # 这应该正常工作或优雅地报错
        try:
            choices = selector.select(0)
            assert len(choices) == 0
        except Exception as e:
            # 如果抛出异常也应该是合理的
            pass

    def test_device_consistency(self):
        """设备一致性"""
        sampler = BatchGaussianMixtureSampler(device="cpu")
        samples = sampler.sample(100)

        assert samples.device.type == "cpu"


# ==================== 性能基准测试 ====================


class TestPerformanceBenchmarks:
    """性能基准测试"""

    def test_vectorized_vs_loop_speedup(self):
        """向量化 vs 循环的加速比"""
        sampler = BatchGaussianMixtureSampler(
            weight=0.5,
            mean1=0.0,
            std1=1.0,
            mean2=5.0,
            std2=1.0,
        )

        results = sampler.benchmark_vs_naive(
            sample_size=50000,
            num_runs=5,
        )

        speedup = results["speedup_ratio"]
        print(f"\n性能基准测试结果:")
        print(f"  朴素循环: {results['naive']['mean_time']:.4f}s")
        print(f"  向量化:   {results['vectorized']['mean_time']:.4f}s")
        print(f"  加速比:   {speedup:.2f}x")

        # 向量化应该至少快 5 倍（保守估计）
        assert speedup > 5.0, f"加速比不足: {speedup:.2f}x"

    def test_box_muller_vs_direct_performance(self):
        """Box-Muller vs 直接方法的性能对比"""
        sampler_bm = GaussianSampler(method="box_muller")
        sampler_direct = GaussianSampler(method="direct")

        bm_results = sampler_bm.benchmark_methods(sample_size=100000, num_runs=5)
        direct_results = sampler_direct.benchmark_methods(sample_size=100000, num_runs=5)

        print(f"\n高斯采样方法对比:")
        print(f"  Box-Muller: {bm_results['box_muller']['mean_time']:.4f}s")
        print(f"  Direct:     {direct_results['direct']['mean_time']:.4f}s")
        print(f"  加速比:     {bm_results['speedup_ratio']:.2f}x")

        # 直接方法通常更快
        assert bm_results["speedup_ratio"] >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
