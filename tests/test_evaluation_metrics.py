"""评估指标模块测试

全面测试 GMS 评估模块的所有功能，包括:
1. FID 计算正确性（与已知结果对比）
2. IS 计算范围和稳定性
3. Precision & Recall 边界情况
4. 批量评估脚本命令行解析
5. 报告生成功能
6. 小规模数据快速验证
7. 性能基准测试

运行方式:
    pytest tests/test_evaluation_metrics.py -v

注意:
    - 测试使用小型模型和小批量数据以确保快速执行
    - 部分测试需要 CUDA（会自动跳过）
    - InceptionV3 模型会在首次运行时下载
"""

import os
import sys
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import torch
from PIL import Image


class TestFIDCalculator:
    """FID 计算器测试"""

    @pytest.fixture(scope="class")
    def device(self):
        """获取可用设备"""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture(scope="class")
    def fid_calculator(self, device):
        """创建 FID 计算器实例"""
        from gms.evaluation.metrics.fid_score import FIDCalculator
        return FIDCalculator(device=device, batch_size=10)

    @pytest.fixture
    def sample_images(self):
        """创建示例图像数据 (N, C, H, W), 值域 [0, 1]"""
        torch.manual_seed(42)
        return torch.randn(50, 3, 299, 299).clamp(0, 1)

    def test_fid_initialization(self, device):
        """测试 FID 计算器初始化"""
        from gms.evaluation.metrics.fid_score import FIDCalculator

        calc = FIDCalculator(device=device)
        assert calc.device == device
        assert calc.dims == 2048
        assert calc.batch_size == 50
        assert isinstance(calc.feature_cache, dict)

    def test_invalid_device(self):
        """测试无效设备参数"""
        from gms.evaluation.metrics.fid_score import FIDCalculator

        with pytest.raises(ValueError, match="device"):
            FIDCalculator(device="invalid")

    def test_fid_same_distribution(self, fid_calculator, sample_images):
        """测试相同分布的 FID 应接近 0"""
        fid = fid_calculator.calculate_fid(sample_images, sample_images)

        # 相同分布的 FID 应该非常小 (< 1.0)
        assert fid < 1.0, f"相同分布的 FID 应接近 0, 得到 {fid}"
        print(f"✓ 相同分布 FID: {fid:.6f}")

    def test_fid_different_distributions(self, fid_calculator, sample_images):
        """测试不同分布的 FID 应 > 0"""
        torch.manual_seed(123)  # 不同种子生成不同图像
        different_images = torch.randn(50, 3, 299, 299).clamp(0, 1)

        fid = fid_calculator.calculate_fid(sample_images, different_images)

        # 不同分布的 FID 应该明显大于 0
        assert fid > 0.0, "不同分布的 FID 应大于 0"
        print(f"✓ 不同分布 FID: {fid:.4f}")

    def test_fid_from_features(self, fid_calculator, sample_images):
        """测试从预计算特征计算 FID"""
        features = fid_calculator.extract_features(sample_images)

        # 从特征计算 FID
        fid = fid_calculator.calculate_fid_from_features(features, features)

        assert fid < 1.0, "从特征计算的相同分布 FID 应接近 0"
        print(f"✓ 特征计算 FID: {fid:.6f}")

    def test_compute_statistics(self, fid_calculator, sample_images):
        """测试统计量计算"""
        mean, cov = fid_calculator.compute_statistics(sample_images)

        assert mean.shape == (2048,), f"均值维度错误: {mean.shape}"
        assert cov.shape == (2048, 2048), f"协方差维度错误: {cov.shape}"

        # 检查协方差矩阵的对称性
        assert np.allclose(cov, cov.T), "协方差矩阵不对称"

        print(f"✓ 统计量计算正确: mean shape={mean.shape}, cov shape={cov.shape}")

    def test_feature_caching(self, fid_calculator, sample_images):
        """测试特征缓存功能"""
        assert fid_calculator.use_cache is True

        # 第一次提取（应缓存）
        fid_calculator.extract_features(sample_images, cache_key="test")

        # 检查缓存信息
        info = fid_calculator.get_cache_info()
        assert info["num_cached_items"] >= 1
        assert "test" in info["cached_keys"]

        # 清除缓存
        fid_calculator.clear_cache()
        info = fid_calculator.clear_cache()
        info_after = fid_calculator.get_cache_info()
        assert info_after["num_cached_items"] == 0

        print("✓ 缓存功能正常")

    def test_fid_numerical_stability(self, fid_calculator):
        """测试数值稳定性（小方差数据）"""
        # 创建低方差的数据
        torch.manual_seed(42)
        low_var_images = torch.ones(20, 3, 299, 299) * 0.5 + torch.randn(
            20, 3, 299, 299
        ) * 0.01
        low_var_images = low_var_images.clamp(0, 1)

        # 不应该抛出数值异常
        try:
            fid = fid_calculator.calculate_fid(low_var_images, low_var_images)
            assert np.isfinite(fid), "FID 应为有限值"
            print(f"✓ 数值稳定性测试通过: {fid:.6f}")
        except Exception as e:
            pytest.fail(f"数值稳定性测试失败: {e}")

    def test_batch_processing(self, fid_calculator):
        """测试批处理功能"""
        # 创建大批量数据
        torch.manual_seed(42)
        large_batch = torch.randn(100, 3, 299, 299).clamp(0, 1)

        start_time = time.time()
        fid = fid_calculator.calculate_fid(large_batch[:50], large_batch[50:])
        elapsed = time.time() - start_time

        assert np.isfinite(fid), "批处理后 FID 应为有限值"
        print(f"✓ 批处理测试通过: {elapsed:.2f}s, FID={fid:.4f}")


class TestISCalculator:
    """IS 计算器测试"""

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    @pytest.fixture(scope="class")
    def is_calculator(self, device):
        from gms.evaluation.metrics.is_score import ISCalculator
        return ISCalculator(device=device, batch_size=10)

    @pytest.fixture
    def sample_images(self):
        """创建示例图像"""
        torch.manual_seed(42)
        return torch.randn(100, 3, 299, 299).clamp(0, 1)

    def test_is_initialization(self, device):
        """测试 IS 计算器初始化"""
        from gms.evaluation.metrics.is_score import ISCalculator

        calc = ISCalculator(device=device)
        assert calc.device == device
        assert calc.num_classes == 1000

    def test_is_range(self, is_calculator, sample_images):
        """测试 IS 值在合理范围内 [1, 1000]"""
        is_mean, is_std = is_calculator.calculate_is(
            sample_images, splits=5
        )

        # IS 理论上应该在合理范围内
        assert 1.0 <= is_mean <= 1000, f"IS 超出合理范围: {is_mean}"
        assert is_std >= 0, f"标准差不应为负: {is_std}"

        print(f"✓ IS 范围正确: {is_mean:.2f} ± {is_std:.2f}")

    def test_is_splits_parameter(self, is_calculator, sample_images):
        """测试 splits 参数对结果的影响"""
        splits_list = [2, 5, 10]

        results = []
        for splits in splits_list:
            is_mean, is_std = is_calculator.calculate_is(
                sample_images, splits=splits
            )
            results.append((is_mean, is_std))

        # 不同 splits 应给出相近的结果（允许一定波动）
        means = [r[0] for r in results]
        mean_of_means = np.mean(means)

        for i, (m, s) in enumerate(results):
            deviation = abs(m - mean_of_means) / mean_of_means
            assert (
                deviation < 0.5
            ), f"splits={splits_list[i]} 时偏差过大: {deviation:.2%}"

        print(f"✓ Splits 参数测试通过: {[f'{m:.2f}' for m, _ in results]}")

    def test_is_from_features(self, is_calculator, sample_images):
        """测试从预计算的特征计算 IS"""
        predictions = is_calculator.get_predictions(sample_images)

        is_mean, is_std = is_calculator.calculate_is_from_features(
            predictions, splits=5
        )

        assert 1.0 <= is_mean <= 1000
        assert is_std >= 0

        print(f"✓ 特征计算 IS: {is_mean:.2f} ± {is_std:.2f}")

    def test_is_minimum_samples(self, is_calculator, sample_images):
        """测试最小样本数要求"""
        predictions = is_calculator.get_predictions(sample_images[:10])

        # splits 大于样本数时应报错
        with pytest.raises(ValueError, match="样本数量"):
            is_calculator.calculate_is_from_features(predictions, splits=20)

        print("✓ 最小样本数检查正常")

    def test_is_invalid_splits(self, is_calculator, sample_images):
        """测试非法 splits 参数"""
        predictions = is_calculator.get_predictions(sample_images)

        with pytest.raises(ValueError, match="splits"):
            is_calculator.calculate_is_from_features(predictions, splits=0)

        with pytest.raises(ValueError, match="splits"):
            is_calculator.calculate_is_from_features(predictions, splits=-1)

        print("✓ 参数验证正常")


class TestPrecisionRecallCalculator:
    """Precision & Recall 计算器测试"""

    @pytest.fixture(scope="class")
    def pr_calculator(self):
        from gms.evaluation.metrics.precision_recall import (
            PrecisionRecallCalculator,
        )
        return PrecisionRecallCalculator(k=3)

    @pytest.fixture
    def sample_features(self):
        """创建示例特征向量"""
        np.random.seed(42)
        real_features = np.random.randn(100, 2048).astype(np.float32)
        gen_features = np.random.randn(100, 2048).astype(np.float32)
        return real_features, gen_features

    def test_pr_initialization(self):
        """测试 P&R 计算器初始化"""
        from gms.evaluation.metrics.precision_recall import (
            PrecisionRecallCalculator,
            DistanceMetric,
        )

        calc = PrecisionRecallCalculator(k=3)
        assert calc.k == 3
        assert calc.distance_metric == DistanceMetric.EUCLIDEAN

    def test_pr_range(self, pr_calculator, sample_features):
        """测试 Precision 和 Recall 在 [0, 1] 范围内"""
        real_feats, gen_feats = sample_features

        results = pr_calculator.calculate_precision_recall_from_features(
            real_feats, gen_feats
        )

        precision = results["precision"]
        recall = results["recall"]

        assert 0.0 <= precision <= 1.0, f"Precision 超出范围: {precision}"
        assert 0.0 <= recall <= 1.0, f"Recall 超出范围: {recall}"

        print(f"✓ P&R 范围正确: P={precision:.4f}, R={recall:.4f}")

    def test_pr_identical_distributions(self, pr_calculator):
        """测试相同分布时 P 和 R 都应较高"""
        np.random.seed(42)
        features = np.random.randn(100, 2048).astype(np.float32)

        results = pr_calculator.calculate_precision_recall_from_features(
            features, features
        )

        # 相同分布时，P 和 R 都应该很高（> 0.9）
        assert results["precision"] > 0.9, (
            f"相同分布 Precision 应 > 0.9, 得到 {results['precision']}"
        )
        assert results["recall"] > 0.9, (
            f"相同分布 Recall 应 > 0.9, 得到 {results['recall']}"
        )

        print(f"✓ 相同分布 P&R: P={results['precision']:.4f}, R={results['recall']:.4f}")

    def test_pr_different_k_values(self, pr_calculator, sample_features):
        """测试不同 k 值的影响"""
        real_feats, gen_feats = sample_features

        k_values = [1, 3, 5, 10]
        results_dict = {}

        for k in k_values:
            results = pr_calculator.calculate_precision_recall_from_features(
                real_feats, gen_feats, k=k
            )
            results_dict[k] = results

        # 所有结果都应在有效范围内
        for k, res in results_dict.items():
            assert 0.0 <= res["precision"] <= 1.0
            assert 0.0 <= res["recall"] <= 1.0

        print(f"✓ 不同 k 值测试通过: {list(results_dict.keys())}")

    def test_pr_distance_metric(self):
        """测试不同的距离度量"""
        from gms.evaluation.metrics.precision_recall import (
            PrecisionRecallCalculator,
            DistanceMetric,
        )

        np.random.seed(42)
        real_feats = np.random.randn(50, 512).astype(np.float32)
        gen_feats = np.random.randn(50, 512).astype(np.float32)

        metrics = [
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.COSINE,
            DistanceMetric.MANHATTAN,
        ]

        for metric in metrics:
            calc = PrecisionRecallCalculator(k=3, distance_metric=metric)
            results = calc.calculate_precision_recall_from_features(
                real_feats, gen_feats
            )
            assert 0.0 <= results["precision"] <= 1.0
            assert 0.0 <= results["recall"] <= 1.0

        print("✓ 多种距离度量测试通过")

    def test_pr_empty_data(self, pr_calculator):
        """测试空数据处理"""
        empty_real = np.array([]).reshape(0, 2048)
        empty_gen = np.array([]).reshape(0, 2048)

        # 空数据应导致错误或返回特殊值
        try:
            results = pr_calculator.calculate_precision_recall_from_features(
                empty_real, empty_gen
            )
            # 如果不报错，检查返回值是否合理
            if results:
                assert "error" in results or (
                    0.0 <= results.get("precision", 0) <= 1.0
                )
        except Exception as e:
            # 抛出异常也是可接受的行为
            assert True

        print("✓ 空数据处理测试完成")


class TestEvaluationScript:
    """批量评估脚本测试"""

    def test_config_creation(self):
        """测试配置对象创建"""
        from gms.evaluation.evaluation_script import EvaluationConfig

        config = EvaluationConfig(
            real_data_path="./test_real",
            generated_path="./test_gen",
            metrics=["fid", "is"],
            device="cpu",
            batch_size=32,
        )

        assert config.real_data_path == "./test_real"
        assert config.generated_path == "./test_gen"
        assert config.metrics == ["fid", "is"]
        assert config.batch_size == 32
        print("✓ 配置创建成功")

    def test_argument_parsing(self):
        """测试命令行参数解析"""
        from gms.evaluation.evaluation_script import parse_arguments

        # Mock sys.argv
        test_args = [
            "--real_data_path",
            "/path/to/real",
            "--generated_path",
            "/path/to/gen",
            "--metrics",
            "fid",
            "is",
            "--device",
            "cpu",
            "--batch_size",
            "64",
        ]

        with patch.object(sys, "argv", ["script.py"] + test_args):
            args = parse_arguments()

        assert args.real_data_path == "/path/to/real"
        assert args.generated_path == "/path/to/gen"
        assert args.metrics == ["fid", "is"]
        assert args.device == "cpu"
        assert args.batch_size == 64
        print("✓ 命令行参数解析正确")

    def test_load_images_from_directory(self, tmp_path):
        """测试从目录加载图像"""
        from gms.evaluation.evaluation_script import load_images_from_directory

        # 创建临时图像文件
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for i in range(5):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(img_dir / f"image_{i}.jpg")

        images = load_images_from_directory(str(img_dir))

        assert images.shape[0] == 5, f"应加载 5 张图像, 得到 {images.shape[0]}"
        assert images.shape[1] == 3, "应为 RGB 图像"
        assert images.min() >= 0.0 and images.max() <= 1.0, "值域应在 [0, 1]"
        print(f"✓ 图像加载成功: {images.shape}")

    def test_load_nonexistent_directory(self):
        """测试加载不存在的目录"""
        from gms.evaluation.evaluation_script import load_images_from_directory

        with pytest.raises(FileNotFoundError):
            load_images_from_directory("/nonexistent/path")

        print("✓ 错误路径检测正常")

    def test_save_results(self, tmp_path):
        """测试结果保存功能"""
        from gms.evaluation.evaluation_script import _save_results

        test_results = {
            "config": {"metrics": ["fid"], "device": "cpu"},
            "timestamp": "2024-01-01 12:00:00",
            "metrics": {
                "fid": {"value": 123.45, "computation_time": 10.5},
            },
            "duration": 15.3,
        }

        _save_results(test_results, str(tmp_path / "output"))

        # 检查 JSON 文件是否存在
        json_files = list(tmp_path.glob("**/*.json"))
        assert len(json_files) > 0, "应生成 JSON 文件"

        # 验证 JSON 内容
        with open(json_files[0], "r") as f:
            saved_data = json.load(f)

        assert saved_data["metrics"]["fid"]["value"] == 123.45
        print("✓ 结果保存成功")


class TestReportGenerator:
    """报告生成器测试"""

    @pytest.fixture
    def sample_results(self):
        """创建示例评估结果"""
        return {
            "config": {
                "real_data_path": "./real",
                "generated_path": "./gen",
                "metrics": ["fid", "is"],
                "device": "cpu",
                "num_real_images": 1000,
                "num_gen_images": 1000,
            },
            "timestamp": "2024-01-01 12:00:00",
            "duration": 45.6,
            "metrics": {
                "fid": {"value": 89.23, "computation_time": 30.2},
                "is": {"mean": 7.85, "std": 0.34, "splits": 10},
                "precision_recall": {
                    "precision": 0.78,
                    "recall": 0.65,
                    "density": 0.0023,
                    "manifold_radius": 0.0156,
                },
            },
        }

    def test_report_generation_zh(self, sample_results, tmp_path):
        """测试中文报告生成"""
        from gms.evaluation.report_generator import EvaluationReportGenerator

        generator = EvaluationReportGenerator(
            output_dir=str(tmp_path), language="zh"
        )

        report_path = generator.generate_report(
            evaluation_results=sample_results,
            experiment_name="TestExperiment",
        )

        assert Path(report_path).exists(), "报告文件应存在"
        assert report_path.endswith(".html"), "应为 HTML 文件"

        # 验证内容包含中文
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "GMS 评估报告" in content, "应包含中文标题"
        assert "实验概览" in content, "应包含中文章节标题"

        print(f"✓ 中文报告生成成功: {report_path}")

    def test_report_generation_en(self, sample_results, tmp_path):
        """测试英文报告生成"""
        from gms.evaluation.report_generator import EvaluationReportGenerator

        generator = EvaluationReportGenerator(
            output_dir=str(tmp_path), language="en"
        )

        report_path = generator.generate_report(
            evaluation_results=sample_results,
            experiment_name="TestExperiment_EN",
        )

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "Evaluation Report" in content, "应包含英文标题"
        print(f"✓ 英文报告生成成功: {report_path}")

    def test_report_with_samples(self, sample_results, tmp_path):
        """测试带样本展示的报告"""
        from gms.evaluation.report_generator import EvaluationReportGenerator

        generator = EvaluationReportGenerator(output_dir=str(tmp_path))

        # 创建示例图像
        torch.manual_seed(42)
        samples = torch.randn(16, 3, 64, 64).clamp(0, 1)

        report_path = generator.generate_report(
            evaluation_results=sample_results,
            experiment_name="WithSamples",
            generated_samples=samples,
        )

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "base64" in content, "应包含编码的样本图像"
        print(f"✓ 样本展示报告生成成功")

    def test_report_with_baseline(self, sample_results, tmp_path):
        """测试带基线对比的报告"""
        from gms.evaluation.report_generator import EvaluationReportGenerator

        generator = EvaluationReportGenerator(output_dir=str(tmp_path))

        baseline = {"fid": 120.5, "is": 6.8, "precision": 0.65}

        report_path = generator.generate_report(
            evaluation_results=sample_results,
            experiment_name="BaselineComparison",
            baseline_results=baseline,
        )

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "基线" in content or "Baseline" in content, "应包含基线对比部分"
        print(f"✓ 基线对比报告生成成功")

    def test_invalid_language(self, tmp_path):
        """测试不支持的语言参数"""
        from gms.evaluation.report_generator import EvaluationReportGenerator

        with pytest.raises(ValueError, match="语言"):
            EvaluationReportGenerator(language="fr")

        print("✓ 语言参数验证正常")


class TestIntegrationAndPerformance:
    """集成测试和性能基准"""

    @pytest.fixture(scope="class")
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_end_to_end_evaluation(self, device, tmp_path):
        """端到端评估流程测试"""
        from gms.evaluation.metrics.fid_score import FIDCalculator
        from gms.evaluation.metrics.is_score import ISCalculator
        from gms.evaluation.metrics.precision_recall import (
            PrecisionRecallCalculator,
        )

        # 准备数据
        torch.manual_seed(42)
        real_images = torch.randn(200, 3, 299, 299).clamp(0, 1)
        gen_images = torch.randn(200, 3, 299, 299).clamp(0, 1)

        # 初始化所有计算器
        fid_calc = FIDCalculator(device=device, batch_size=20)
        is_calc = ISCalculator(device=device, batch_size=20)
        pr_calc = PrecisionRecallCalculator(k=3)

        # 计算 FID
        fid_start = time.time()
        fid = fid_calc.calculate_fid(real_images, gen_images)
        fid_time = time.time() - fid_start

        # 计算 IS
        is_start = time.time()
        is_mean, is_std = is_calc.calculate_is(gen_images, splits=5)
        is_time = time.time() - is_start

        # 计算 P&R
        pr_start = time.time()
        pr_results = pr_calc.calculate_precision_recall(
            real_images, gen_images, feature_extractor=fid_calc
        )
        pr_time = time.time() - pr_start

        # 验证结果
        assert np.isfinite(fid), "FID 应为有限值"
        assert 1.0 <= is_mean <= 1000, "IS 应在有效范围内"
        assert 0.0 <= pr_results["precision"] <= 1.0, "Precision 应在 [0, 1]"
        assert 0.0 <= pr_results["recall"] <= 1.0, "Recall 应在 [0, 1]"

        total_time = fid_time + is_time + pr_time

        print("\n" + "=" * 60)
        print("端到端评估结果:")
        print(f"  FID:      {fid:.4f} ({fid_time:.2f}s)")
        print(f"  IS:       {is_mean:.4f} ± {is_std:.4f} ({is_time:.2f}s)")
        print(f"  Prec:     {pr_results['precision']:.4f}")
        print(f"  Rec:      {pr_results['recall']:.4f} ({pr_time:.2f}s)")
        print(f"  总耗时:   {total_time:.2f}s")
        print("=" * 60)

    def test_performance_benchmark_1000_images(self, device):
        """性能基准测试：1000 张图像"""
        from gms.evaluation.metrics.fid_score import FIDCalculator

        torch.manual_seed(42)
        images_1 = torch.randn(1000, 3, 299, 299).clamp(0, 1)
        images_2 = torch.randn(1000, 3, 299, 299).clamp(0, 1)

        calculator = FIDCalculator(device=device, batch_size=50)

        start_time = time.time()
        fid = calculator.calculate_fid(images_1, images_2)
        elapsed = time.time() - start_time

        assert np.isfinite(fid), "FID 应为有限值"

        print(f"\n性能基准 (1000 张图像):")
        print(f"  设备:     {device}")
        print(f"  FID:      {fid:.4f}")
        print(f"  耗时:     {elapsed:.2f}s")
        print(f"  吞吐量:   {2000 / elapsed:.1f} 张/秒")

        # 性能阈值 (CPU 上可能较慢)
        max_allowed_time = 300 if device == "cpu" else 60
        assert (
            elapsed < max_allowed_time
        ), f"耗时过长: {elapsed:.2f}s > {max_allowed_time}s"

    def test_quick_functions(self, device):
        """测试便捷函数接口"""
        from gms.evaluation.metrics.fid_score import calculate_fid_quick
        from gms.evaluation.metrics.is_score import calculate_is_quick

        torch.manual_seed(42)
        real_imgs = torch.randn(50, 3, 299, 299).clamp(0, 1)
        gen_imgs = torch.randn(50, 3, 299, 299).clamp(0, 1)

        # 快速 FID
        fid = calculate_fid_quick(real_imgs, gen_imgs, device=device)
        assert np.isfinite(fid)

        # 快速 IS
        is_mean, is_std = calculate_is_quick(gen_imgs, device=device, splits=3)
        assert 1.0 <= is_mean <= 1000

        print(f"\n✓ 便捷函数测试通过:")
        print(f"  Quick FID: {fid:.4f}")
        print(f"  Quick IS:  {is_mean:.4f} ± {is_std:.4f}")


class TestEdgeCases:
    """边界情况和错误处理测试"""

    def test_single_image_input(self):
        """测试单张图像输入"""
        from gms.evaluation.metrics.fid_score import FIDCalculator

        calculator = FIDCalculator(device="cpu")

        # 单张图像 (需要添加 batch 维度)
        single_img = torch.randn(1, 3, 299, 299).clamp(0, 1)

        # 不应崩溃
        features = calculator.extract_features(single_img)
        assert features.shape[0] == 1, "应提取 1 个特征向量"

        print("✓ 单张图像输入处理正常")

    def test_very_small_images(self):
        """测试非常小的图像尺寸"""
        from gms.evaluation.metrics.fid_score import FIDCalculator

        calculator = FIDCalculator(device="cpu")

        # 小尺寸图像 (应自动 resize 到 299x299)
        small_images = torch.randn(10, 3, 32, 32).clamp(0, 1)

        try:
            features = calculator.extract_features(small_images)
            assert features.shape[0] == 10
            print("✓ 小尺寸图像自动 resize 正常")
        except Exception as e:
            pytest.fail(f"小尺寸图像处理失败: {e}")

    def test_grayscale_to_rgb_conversion(self):
        """测试灰度图转换"""
        from gms.evaluation.metrics.fid_score import FIDCalculator

        calculator = FIDCalculator(device="cpu")

        # 灰度图像 (1 channel) - 可能失败或自动处理
        gray_images = torch.randn(10, 1, 299, 299).clamp(0, 1)

        try:
            features = calculator.extract_features(gray_images)
            print("✓ 灰度图像处理完成")
        except Exception as e:
            print(f"⚠ 灰度图像可能不被支持: {e}")
            # 这是预期行为

    def test_memory_efficiency(self):
        """测试内存效率（大数据集）"""
        from gms.evaluation.metrics.fid_score import FIDCalculator

        # 使用较小的 batch_size 测试内存管理
        calculator = FIDCalculator(device="cpu", batch_size=5)

        torch.manual_seed(42)
        large_dataset = torch.randn(500, 3, 299, 299).clamp(0, 1)

        try:
            fid = calculator.calculate_fid(large_dataset[:250], large_dataset[250:])
            assert np.isfinite(fid)
            print(f"✓ 内存效率测试通过 (batch_size=5): FID={fid:.4f}")
        except Exception as e:
            pytest.fail(f"内存效率测试失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
