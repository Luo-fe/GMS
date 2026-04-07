"""采样统计检验验证器 - 验证采样分布的正确性

实现多种统计检验方法来验证采样结果与理论分布的一致性。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import logging
import numpy as np
import torch
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """采样验证报告数据类

    存储所有统计检验的结果和汇总信息。

    Attributes:
        sample_size: 样本数量
        ks_test: KS 检验结果字典
        chi2_test: χ² 检验结果字典
        moment_validation: 矩验证结果字典
        overall_passed: 是否通过所有检验
        details: 详细的检验信息列表
    """

    sample_size: int = 0
    ks_test: Optional[Dict[str, Any]] = None
    chi2_test: Optional[Dict[str, Any]] = None
    moment_validation: Optional[Dict[str, Any]] = None
    overall_passed: bool = False
    details: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """生成验证报告摘要

        Returns:
            格式化的摘要字符串
        """
        lines = [
            f"=" * 60,
            f"采样验证报告 (样本数: {self.sample_size})",
            f"=" * 60,
            f"总体结果: {'✓ 通过' if self.overall_passed else '✗ 未通过'}",
            "-" * 40,
        ]

        if self.ks_test:
            lines.append(
                f"KS 检验: statistic={self.ks_test['statistic']:.6f}, "
                f"p-value={self.ks_test['p_value']:.6f}, "
                f"{'✓' if self.ks_test['passed'] else '✗'}"
            )

        if self.chi2_test:
            lines.append(
                f"χ² 检验: statistic={self.chi2_test['statistic']:.4f}, "
                f"p-value={self.chi2_test['p_value']:.6f}, "
                f"{'✓' if self.chi2_test['passed'] else '✗'}"
            )

        if self.moment_validation:
            mv = self.moment_validation
            lines.append(f"矩验证:")
            lines.append(f"  均值: 误差={mv['mean_error']:.6f} {'✓' if mv['mean_in_range'] else '✗'}")
            lines.append(f"  方差: 误差={mv['variance_error']:.6f} {'✓' if mv['variance_in_range'] else '✗'}")
            if 'skewness_error' in mv:
                lines.append(f"  偏度: 误差={mv['skewness_error']:.6f}")

        if self.details:
            lines.extend(["-" * 40, "详细信息:"])
            for detail in self.details:
                lines.append(f"  • {detail}")

        lines.append("=" * 60)

        return "\n".join(lines)


class SamplingValidator:
    """采样统计检验验证器

    提供多种统计检验方法来验证采样结果的正确性：
    - Kolmogorov-Smirnov (KS) 检验：比较经验分布函数
    - Chi-squared (χ²) 拟合优度检验：直方图拟合检验
    - 矩验证：均值、方差、偏度的置信区间判断

    使用示例：
        >>> validator = SamplingValidator(alpha=0.05)
        >>> samples = sampler.sample(10000)
        >>> report = validator.validate_gmm(samples, gmm_sampler)
        >>> print(report.summary())
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bins_chi2: int = 50,
        confidence_level: float = 0.95,
    ) -> None:
        """初始化验证器

        Args:
            alpha: 显著性水平（默认 0.05，对应 95% 置信度）
            n_bins_chi2: χ² 检验的直方图 bin 数量
            confidence_level: 矩验证的置信水平（默认 95%）
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha 必须在 (0, 1) 范围内，得到 {alpha}")
        if not 0 < confidence_level < 1:
            raise ValueError(f"confidence_level 必须在 (0, 1) 范围内，得到 {confidence_level}")

        self.alpha = alpha
        self.n_bins_chi2 = n_bins_chi2
        self.confidence_level = confidence_level

        logger.info(
            f"初始化 SamplingValidator: α={alpha}, "
            f"chi2_bins={n_bins_chi2}, confidence={confidence_level}"
        )

    def ks_test(
        self,
        samples: np.ndarray,
        cdf_func,
        test_name: str = "KS Test",
    ) -> Dict[str, Any]:
        """Kolmogorov-Smirnov 检验

        比较样本的经验分布函数（EDF）与理论累积分布函数（CDF）。

        原假设 H0：样本来自由 cdf_func 定义的分布
        - p-value > alpha：不能拒绝 H0（检验通过）
        - p-value <= alpha：拒绝 H0（检验未通过）

        Args:
            samples: 样本数组（numpy 数组）
            cdf_func: 理论 CDF 函数，接受一个参数返回 CDF 值
            test_name: 测试名称（用于日志）

        Returns:
            包含检验结果的字典：
            - statistic: KS 统计量（最大偏差）
            - p_value: p 值
            - passed: 是否通过检验（p > alpha）
            - critical_value: 临界值
        """
        # 执行 KS 检验
        statistic, p_value = scipy_stats.kstest(samples, cdf_func)

        # 计算临界值（近似公式）
        n = len(samples)
        critical_value = np.sqrt(-0.5 * np.log(self.alpha / 2)) / np.sqrt(n)

        passed = p_value > self.alpha

        result = {
            "test_name": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "critical_value": float(critical_value),
            "passed": passed,
            "alpha": self.alpha,
            "sample_size": n,
        }

        logger.info(
            f"{test_name}: D={statistic:.6f}, p={p_value:.6f}, "
            f"临界值={critical_value:.6f}, {'通过' if passed else '未通过'}"
        )

        return result

    def chi2_test(
        self,
        samples: np.ndarray,
        pdf_func,
        param_bounds: Optional[Tuple[float, float]] = None,
        test_name: str = "Chi2 Test",
    ) -> Dict[str, Any]:
        """Chi-squared 拟合优度检验

        将连续分布离散化为直方图，然后进行 χ² 拟合优度检验。

        步骤：
        1. 将样本范围划分为 n_bins 个区间
        2. 计算每个区间的观测频数
        3. 计算每个区间的理论频数（基于 PDF 积分）
        4. 计算 χ² 统计量并判断是否拒绝原假设

        Args:
            samples: 样本数组
            pdf_func: 理论概率密度函数
            param_bounds: 参数范围 (min, max)，用于确定直方图边界
            test_name: 测试名称

        Returns:
            包含检验结果的字典
        """
        n = len(samples)

        # 确定直方图边界
        if param_bounds is None:
            data_min = float(np.min(samples))
            data_max = float(np.max(samples))
            padding = (data_max - data_min) * 0.1
            bin_min = data_min - padding
            bin_max = data_max + padding
        else:
            bin_min, bin_max = param_bounds

        # 创建直方图
        bin_edges = np.linspace(bin_min, bin_max, self.n_bins_chi2 + 1)
        observed_counts, _ = np.histogram(samples, bins=bin_edges)

        # 计算每个 bin 的理论概率
        bin_width = (bin_max - bin_min) / self.n_bins_chi2
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        theoretical_probs = pdf_func(bin_centers) * bin_width
        theoretical_probs = theoretical_probs / theoretical_probs.sum()  # 归一化

        # 计算理论频数
        expected_counts = theoretical_probs * n

        # 合并频数过小的 bin（要求每个 bin 的期望频数 >= 5）
        mask = expected_counts >= 5
        if mask.sum() < 2:
            logger.warning("χ² 检验：有效 bin 数不足，调整 bin 数量")
            return self._chi2_test_adjusted(samples, pdf_func, param_bounds, test_name)

        observed_filtered = observed_counts[mask]
        expected_filtered = expected_counts[mask]

        # 重新归一化期望频数以匹配观测频数总和
        expected_filtered = expected_filtered * (observed_filtered.sum() / expected_filtered.sum())

        # 执行 χ² 检验
        statistic, p_value = scipy_stats.chisquare(
            observed_filtered,
            f_exp=expected_filtered,
        )

        degrees_of_freedom = len(observed_filtered) - 1
        passed = p_value > self.alpha

        result = {
            "test_name": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "degrees_of_freedom": int(degrees_of_freedom),
            "passed": passed,
            "alpha": self.alpha,
            "n_bins_used": int(mask.sum()),
            "sample_size": n,
        }

        logger.info(
            f"{test_name}: χ²={statistic:.4f}, p={p_value:.6f}, "
            f"df={degrees_of_freedom}, {'通过' if passed else '未通过'}"
        )

        return result

    def _chi2_test_adjusted(
        self,
        samples: np.ndarray,
        pdf_func,
        param_bounds: Optional[Tuple[float, float]],
        test_name: str,
    ) -> Dict[str, Any]:
        """使用调整后的 bin 数重新执行 χ² 检验"""
        original_bins = self.n_bins_chi2
        self.n_bins_chi2 = max(10, original_bins // 2)
        result = self.chi2_test(samples, pdf_func, param_bounds, test_name)
        self.n_bins_chi2 = original_bins
        result["details"] = f"自动调整 bin 数从 {original_bins} 到 {self.n_bins_chi2}"
        return result

    def validate_moments(
        self,
        samples: np.ndarray,
        theoretical_mean: float,
        theoretical_variance: float,
        theoretical_skewness: Optional[float] = None,
    ) -> Dict[str, Any]:
        """验证样本的矩（均值、方差、偏度）

        使用大样本性质：对于足够大的样本，
        样本均值服从渐近正态分布，可以构建置信区间。

        Args:
            samples: 样本数组
            theoretical_mean: 理论均值
            theoretical_variance: 理论方差
            theoretical_skewness: 理论偏度（可选）

        Returns:
            包含矩验证结果的字典
        """
        n = len(samples)
        sample_mean = float(np.mean(samples))
        sample_var = float(np.var(samples, ddof=1))  # 无偏估计
        sample_std = np.sqrt(sample_var)

        from scipy.stats import skew as calc_skew
        sample_skewness = float(calc_skew(samples))

        # 均值的置信区间（基于中心极限定理）
        se_mean = sample_std / np.sqrt(n)
        z_critical = scipy_stats.norm.ppf((1 + self.confidence_level) / 2)
        mean_ci_lower = sample_mean - z_critical * se_mean
        mean_ci_upper = sample_mean + z_critical * se_mean

        # 判断理论值是否在置信区间内
        mean_in_range = mean_ci_lower <= theoretical_mean <= mean_ci_upper
        mean_error = abs(sample_mean - theoretical_mean)

        # 方差的置信区间（使用卡方分布）
        chi2_lower = scipy_stats.chi2.ppf((1 - self.confidence_level) / 2, n - 1)
        chi2_upper = scipy_stats.chi2.ppf((1 + self.confidence_level) / 2, n - 1)
        var_ci_lower = (n - 1) * sample_var / chi2_upper
        var_ci_upper = (n - 1) * sample_var / chi2_lower

        variance_in_range = var_ci_lower <= theoretical_variance <= var_ci_upper
        variance_error = abs(sample_var - theoretical_variance)

        result = {
            "sample_mean": sample_mean,
            "theoretical_mean": theoretical_mean,
            "mean_error": mean_error,
            "mean_ci": (mean_ci_lower, mean_ci_upper),
            "mean_in_range": mean_in_range,
            "sample_variance": sample_var,
            "theoretical_variance": theoretical_variance,
            "variance_error": variance_error,
            "variance_ci": (var_ci_lower, var_ci_upper),
            "variance_in_range": variance_in_range,
            "sample_skewness": sample_skewness,
            "confidence_level": self.confidence_level,
            "sample_size": n,
        }

        if theoretical_skewness is not None:
            skewness_error = abs(sample_skewness - theoretical_skewness)
            result["theoretical_skewness"] = theoretical_skewness
            result["skewness_error"] = skewness_error

        all_passed = mean_in_range and variance_in_range
        result["all_moments_valid"] = all_passed

        logger.info(
            f"矩验证: 均值误差={mean_error:.6f} {'✓' if mean_in_range else '✗'}, "
            f"方差误差={variance_error:.6f} {'✓' if variance_in_range else '✗'}"
        )

        return result

    def validate_gmm_samples(
        self,
        samples: torch.Tensor,
        gmm_sampler,
    ) -> ValidationReport:
        """完整的 GMM 采样验证

        对 GMM 采样结果执行所有可用的统计检验。

        Args:
            samples: GMM 采样的张量
            gmm_sampler: BatchGaussianMixtureSampler 实例

        Returns:
            完整的 ValidationReport
        """
        samples_np = samples.detach().cpu().numpy()
        n = len(samples_np)

        report = ValidationReport(sample_size=n)
        details = []

        # 获取理论矩
        moments = gmm_sampler.get_theoretical_moments()
        theoretical_mean = moments["mean"]
        theoretical_variance = moments["variance"]

        # 1. KS 检验
        try:
            def gmm_cdf(x):
                w = gmm_sampler.component_selector.weight
                m1, s1 = gmm_sampler.sampler1.mean, gmm_sampler.sampler1.std
                m2, s2 = gmm_sampler.sampler2.mean, gmm_sampler.sampler2.std

                cdf = (
                    (1 - w) * scipy_stats.norm.cdf(x, loc=m1, scale=s1) +
                    w * scipy_stats.norm.cdf(x, loc=m2, scale=s2)
                )
                return cdf

            ks_result = self.ks_test(samples_np, gmm_cdf, "GMM KS Test")
            report.ks_test = ks_result
            details.append(
                f"KS 检验: D={ks_result['statistic']:.6f}, "
                f"p={ks_result['p_value']:.6f}"
            )
        except Exception as e:
            details.append(f"KS 检验失败: {str(e)}")
            logger.error(f"KS 检验执行失败: {e}")

        # 2. Chi-squared 检验
        try:
            def gmm_pdf(x):
                w = gmm_sampler.component_selector.weight
                m1, s1 = gmm_sampler.sampler1.mean, gmm_sampler.sampler1.std
                m2, s2 = gmm_sampler.sampler2.mean, gmm_sampler.sampler2.std

                pdf = (
                    (1 - w) * scipy_stats.norm.pdf(x, loc=m1, scale=s1) +
                    w * scipy_stats.norm.pdf(x, loc=m2, scale=s2)
                )
                return pdf

            # 设置合理的参数边界
            margin = 5 * max(gmm_sampler.sampler1.std, gmm_sampler.sampler2.std)
            bounds = (
                min(gmm_sampler.sampler1.mean, gmm_sampler.sampler2.mean) - margin,
                max(gmm_sampler.sampler1.mean, gmm_sampler.sampler2.mean) + margin,
            )

            chi2_result = self.chi2_test(
                samples_np, gmm_pdf, bounds, "GMM Chi2 Test"
            )
            report.chi2_test = chi2_result
            details.append(
                f"χ² 检验: stat={chi2_result['statistic']:.4f}, "
                f"p={chi2_result['p_value']:.6f}"
            )
        except Exception as e:
            details.append(f"χ² 检验失败: {str(e)}")
            logger.error(f"χ² 检验执行失败: {e}")

        # 3. 矩验证
        try:
            moment_result = self.validate_moments(
                samples_np,
                theoretical_mean=theoretical_mean,
                theoretical_variance=theoretical_variance,
            )
            report.moment_validation = moment_result
            details.append(
                f"矩验证: 均值误差={moment_result['mean_error']:.6f}, "
                f"方差误差={moment_result['variance_error']:.6f}"
            )
        except Exception as e:
            details.append(f"矩验证失败: {str(e)}")
            logger.error(f"矩验证执行失败: {e}")

        # 判断总体是否通过
        tests_results = []
        if report.ks_test:
            tests_results.append(report.ks_test.get("passed", False))
        if report.chi2_test:
            tests_results.append(report.chi2_test.get("passed", False))
        if report.moment_validation:
            tests_results.append(report.moment_validation.get("all_moments_valid", False))

        report.overall_passed = all(tests_results) if tests_results else False
        report.details = details

        logger.info(
            f"GMM 验证完成: 总体={'通过' if report.overall_passed else '未通过'}, "
            f"样本数={n}, 通过检验数={sum(tests_results)}/{len(tests_results)}"
        )

        return report

    def set_alpha(self, new_alpha: float) -> None:
        """更新显著性水平

        Args:
            new_alpha: 新的显著性水平 (0, 1)
        """
        if not 0 < new_alpha < 1:
            raise ValueError(f"new_alpha 必须在 (0, 1) 范围内，得到 {new_alpha}")
        self.alpha = new_alpha
        logger.info(f"显著性水平更新为: {new_alpha}")

    def __repr__(self) -> str:
        return (
            f"SamplingValidator(α={self.alpha}, "
            f"chi2_bins={self.n_bins_chi2}, "
            f"confidence={self.confidence_level})"
        )
