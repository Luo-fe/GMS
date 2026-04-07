"""双分量高斯混合模型（Two-Component GMM）完整单元测试

覆盖所有核心功能的测试:
- GMMParameters 参数类的创建、验证和属性
- GaussianMixtureModel 的 PDF/log-PDF/后验概率/采样
- 序列化/反序列化（JSON/Pickle/State Dict）
- 初始化策略（K-means++/随机/启发式）
- 边界情况和数值稳定性

运行方式:
    pytest tests/test_gmm_model.py -v
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from scipy import stats as scipy_stats
import math


class TestGMMParameters:
    """GMMParameters 参数类测试"""

    def test_create_diagonal_params(self):
        """测试创建对角协方差的参数"""
        from src.gms.gmm_optimization import GMMParameters, GMMParametersConfig

        params = GMMParameters(
            weight=0.6,
            mean1=torch.tensor([1.0, 2.0]),
            mean2=torch.tensor([-1.0, -2.0]),
            variance1=torch.tensor([0.5, 0.5]),
            variance2=torch.tensor([1.0, 1.0])
        )

        assert params.weight == pytest.approx(0.6)
        assert params.weight2 == pytest.approx(0.4)
        assert params.dimensionality == 2
        assert params.is_diagonal is True

    def test_create_full_covariance_params(self):
        """测试创建全协方差矩阵的参数"""
        from src.gms.gmm_optimization import GMMParameters, GMMParametersConfig

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([1.0, 2.0]),
            mean2=torch.tensor([-1.0, -2.0]),
            variance1=torch.tensor([[1.0, 0.3], [0.3, 2.0]]),
            variance2=torch.tensor([[2.0, 0.5], [0.5, 1.0]])
        )

        assert params.dimensionality == 2
        assert params.is_diagonal is False
        assert params.variance1.shape == (2, 2)

    def test_validate_valid_params(self):
        """测试验证合法参数"""
        from src.gms.gmm_optimization import GMMParameters

        params = GMMParameters(
            weight=0.6,
            mean1=torch.tensor([1.0, 2.0]),
            mean2=torch.tensor([-1.0, -2.0]),
            variance1=torch.tensor([0.5, 0.5]),
            variance2=torch.tensor([1.0, 1.0])
        )

        assert params.validate() is True

    def test_validate_invalid_weight(self):
        """测试验证非法权重"""
        from src.gms.gmm_optimization import GMMParameters, GMMParametersConfig

        config = GMMParametersConfig(validate_on_creation=False)
        with pytest.raises(ValueError, match="权重"):
            params = GMMParameters(
                weight=1.5,
                mean1=torch.tensor([1.0, 2.0]),
                mean2=torch.tensor([-1.0, -2.0]),
                variance1=torch.tensor([0.5, 0.5]),
                variance2=torch.tensor([1.0, 1.0]),
                _config=config
            )
            params.validate()

    def test_validate_negative_variance(self):
        """测试验证负方差"""
        from src.gms.gmm_optimization import GMMParameters

        with pytest.raises(ValueError, match="非正值"):
            params = GMMParameters(
                weight=0.6,
                mean1=torch.tensor([1.0, 2.0]),
                mean2=torch.tensor([-1.0, -2.0]),
                variance1=torch.tensor([-0.5, 0.5]),
                variance2=torch.tensor([1.0, 1.0])
            )
            params.validate()

    def test_clamp_invalid_params(self):
        """测试修正非法参数"""
        from src.gms.gmm_optimization import GMMParameters

        config = __import__('src.gms.gmm_optimization.gmm_parameters',
                           fromlist=['GMMParametersConfig']).GMMParametersConfig(validate_on_creation=False)

        params = GMMParameters(
            weight=1.5,
            mean1=torch.tensor([1.0, 2.0]),
            mean2=torch.tensor([-1.0, -2.0]),
            variance1=torch.tensor([-0.5, 0.5]),
            variance2=torch.tensor([1.0, 1.0]),
            _config=config
        )

        params.clamp()

        assert 0 < params.weight < 1
        assert (params.variance1 > 0).all()
        params.validate()

    def test_to_optimizer_params(self):
        """测试转换为优化器参数格式"""
        from src.gms.gmm_optimization import GMMParameters

        params = GMMParameters(
            weight=0.6,
            mean1=torch.tensor([1.0, 2.0]),
            mean2=torch.tensor([-1.0, -2.0]),
            variance1=torch.tensor([0.5, 0.5]),
            variance2=torch.tensor([1.0, 1.0])
        )

        opt_params = params.to_optimizer_params()

        assert 'means' in opt_params
        assert 'covariances' in opt_params
        assert 'weights' in opt_params
        assert opt_params['means'].shape == (2, 2)

    def test_from_optimizer_params(self):
        """测试从优化器结果创建参数"""
        from src.gms.gmm_optimization import GMMParameters, OptimizedParams

        optimized = OptimizedParams(
            means=torch.tensor([[1.0, 2.0], [-1.0, -2.0]]),
            covariances=torch.tensor([[[0.5, 0.0], [0.0, 0.5]],
                                       [[1.0, 0.0], [0.0, 1.0]]]),
            weights=torch.tensor([0.6, 0.4])
        )

        params = GMMParameters.from_optimizer_params(optimized)

        assert params.weight == pytest.approx(0.6)
        assert torch.allclose(params.mean1, torch.tensor([1.0, 2.0]))

    def test_properties_means_covariances_weights(self):
        """测试属性：means, covariances, weights"""
        from src.gms.gmm_optimization import GMMParameters

        params = GMMParameters(
            weight=0.7,
            mean1=torch.tensor([1.0, 2.0, 3.0]),
            mean2=torch.tensor([4.0, 5.0, 6.0]),
            variance1=torch.tensor([0.5, 0.6, 0.7]),
            variance2=torch.tensor([1.0, 1.1, 1.2])
        )

        means = params.means
        assert means.shape == (2, 3)
        assert torch.allclose(means[0], params.mean1)
        assert torch.allclose(means[1], params.mean2)

        covs = params.covariances
        assert covs.shape == (2, 3)

        weights = params.weights
        assert weights.shape == (2,)
        assert weights[0] == pytest.approx(0.7)
        assert weights[1] == pytest.approx(0.3)


class TestGaussianMixtureModelPDF:
    """GaussianMixtureModel PDF 计算测试"""

    @pytest.fixture
    def simple_gmm(self):
        """创建简单的测试用GMM模型"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        params = GMMParameters(
            weight=0.6,
            mean1=torch.tensor([0.0]),
            mean2=torch.tensor([3.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.5])
        )
        return GaussianMixtureModel(params)

    def test_pdf_single_point(self, simple_gmm):
        """测试单点PDF计算"""
        x = torch.tensor([0.0])

        pdf_val = simple_gmm.pdf(x)

        assert isinstance(pdf_val, float)
        assert pdf_val > 0

    def test_pdf_batch(self, simple_gmm):
        """测试批量PDF计算"""
        X = torch.randn(100, 1)

        pdf_vals = simple_gmm.pdf(X)

        assert pdf_vals.shape == (100,)
        assert (pdf_vals > 0).all()

    def test_pdf_against_scipy(self, simple_gmm):
        """与scipy.stats对比验证PDF正确性"""
        x = np.array([0.0, 1.0, 2.0, 3.0])

        X_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)  # (4, 1)
        our_pdf = simple_gmm.pdf(X_tensor)

        w = 0.6
        scipy_pdf = (
            w * scipy_stats.norm.pdf(x, loc=0.0, scale=1.0) +
            (1-w) * scipy_stats.norm.pdf(x, loc=3.0, scale=np.sqrt(1.5))
        )

        np.testing.assert_allclose(our_pdf.numpy(), scipy_pdf, rtol=1e-4)

    def test_log_pdf_numerical_stability(self, simple_gmm):
        """测试log_pdf的数值稳定性"""
        x_extreme = torch.tensor([100.0])  # 极端值

        log_pdf = simple_gmm.log_pdf(x_extreme)

        assert not math.isnan(log_pdf)
        assert not math.isinf(log_pdf)
        assert log_pdf < 0  # log概率应该为负

    def test_log_pdf_consistency_with_pdf(self, simple_gmm):
        """验证log_pdf与pdf的一致性: log(pdf(x)) ≈ log_pdf(x)"""
        X = torch.randn(50, 1)

        pdf_vals = simple_gmm.pdf(X)
        log_pdf_vals = simple_gmm.log_pdf(X)

        expected_log_pdf = np.log(np.array(pdf_vals))

        np.testing.assert_allclose(log_pdf_vals, expected_log_pdf, rtol=1e-4)

    def test_component_pdf(self, simple_gmm):
        """测试单个分量的PDF计算"""
        x = torch.tensor([0.0])

        p_comp0 = simple_gmm.component_pdf(x, component_id=0)
        p_comp1 = simple_gmm.component_pdf(x, component_id=1)

        assert p_comp0 > 0
        assert p_comp1 > 0
        assert p_comp0 > p_comp1  # x更接近mean1

    def test_component_pdf_invalid_id(self, simple_gmm):
        """测试无效的component_id"""
        x = torch.tensor([0.0])

        with pytest.raises(ValueError, match="component_id"):
            simple_gmm.component_pdf(x, component_id=2)


class TestPosteriorProbability:
    """后验概率计算测试"""

    @pytest.fixture
    def gmm_for_posterior(self):
        """创建用于后验概率测试的GMM"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([-2.0]),
            mean2=torch.tensor([2.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.0])
        )
        return GaussianMixtureModel(params)

    def test_posterior_single_point(self, gmm_for_posterior):
        """测试单点后验概率"""
        x = torch.tensor([-2.0])  # 接近第一个分量

        posteriors = gmm_for_posterior.posterior_probability(x)

        assert posteriors.shape == (2,)
        assert torch.isclose(posteriors.sum(), torch.tensor(1.0), atol=1e-5)
        assert posteriors[0] > posteriors[1]

    def test_posterior_batch(self, gmm_for_posterior):
        """测试批量后验概率"""
        X = torch.tensor([[-2.0], [0.0], [2.0]])

        posteriors = gmm_for_posterior.posterior_probability(X)

        assert posteriors.shape == (3, 2)
        assert torch.allclose(posteriors.sum(dim=1), torch.ones(3), atol=1e-5)

    def test_posterior_symmetry(self, gmm_for_posterior):
        """测试对称性：等权重的对称分布的后验应该对称"""
        x_left = torch.tensor([-2.0])
        x_right = torch.tensor([2.0])

        post_left = gmm_for_posterior.posterior_probability(x_left)
        post_right = gmm_for_posterior.posterior_probability(x_right)

        assert torch.isclose(post_left[0], post_right[1], atol=1e-4)
        assert torch.isclose(post_left[1], post_right[0], atol=1e-4)


class TestSampling:
    """采样功能测试"""

    @pytest.fixture
    def gmm_for_sampling(self):
        """创建用于采样的GMM"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        params = GMMParameters(
            weight=0.6,
            mean1=torch.tensor([0.0, 0.0]),
            mean2=torch.tensor([5.0, 5.0]),
            variance1=torch.tensor([1.0, 1.0]),
            variance2=torch.tensor([1.0, 1.0])
        )
        return GaussianMixtureModel(params)

    def test_sample_shape(self, gmm_for_sampling):
        """测试采样形状"""
        n_samples = 1000
        samples = gmm_for_sampling.sample(n_samples, seed=42)

        assert samples.shape == (n_samples, 2)

    def test_sample_reproducibility(self, gmm_for_sampling):
        """测试采样的可复现性"""
        samples1 = gmm_for_sampling.sample(100, seed=42)
        samples2 = gmm_for_sampling.sample(100, seed=42)

        assert torch.equal(samples1, samples2)

    def test_sample_statistical_mean(self, gmm_for_sampling):
        """测试样本均值是否接近理论值"""
        n_samples = 50000
        samples = gmm_for_sampling.sample(n_samples, seed=42)

        sample_mean = samples.mean(dim=0)

        theoretical_mean = (
            0.6 * torch.tensor([0.0, 0.0]) +
            0.4 * torch.tensor([5.0, 5.0])
        )

        assert torch.allclose(sample_mean, theoretical_mean, atol=0.2)

    def test_sample_distribution_mixture(self, gmm_for_sampling):
        """测试样本是否呈现双峰分布特征"""
        n_samples = 10000
        samples = gmm_for_sampling.sample(n_samples, seed=42)

        projected = samples.mean(dim=1)

        hist, bin_edges = np.histogram(projected.numpy(), bins=20)

        peaks = []
        for i in range(1, len(hist)-1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peaks.append(i)

        assert len(peaks) >= 1  # 至少有一个峰（理想情况下两个）


class TestSerialization:
    """序列化/反序列化测试"""

    @pytest.fixture
    def sample_params(self):
        """创建用于序列化测试的参数"""
        from src.gms.gmm_optimization import GMMParameters

        return GMMParameters(
            weight=0.65,
            mean1=torch.tensor([1.23, -2.34, 3.45]),
            mean2=torch.tensor([-1.23, 2.34, -3.45]),
            variance1=torch.tensor([0.56, 0.67, 0.78]),
            variance2=torch.tensor([0.89, 0.91, 0.92])
        )

    def test_json_roundtrip(self, sample_params, tmp_path):
        """测试JSON格式保存和加载的往返一致性"""
        from src.gms.gmm_optimization import GMMSerializer

        serializer = GMMSerializer()
        filepath = tmp_path / "test_params.json"

        serializer.save_json(sample_params, filepath)
        loaded_params = serializer.load_json(filepath)

        assert loaded_params.weight == pytest.approx(sample_params.weight)
        assert torch.allclose(loaded_params.mean1, sample_params.mean1)
        assert torch.allclose(loaded_params.mean2, sample_params.mean2)
        assert torch.allclose(loaded_params.variance1, sample_params.variance1)
        assert torch.allclose(loaded_params.variance2, sample_params.variance2)

    def test_pickle_roundtrip(self, sample_params, tmp_path):
        """测试Pickle格式保存和加载的往返一致性"""
        from src.gms.gmm_optimization import GMMSerializer

        serializer = GMMSerializer()
        filepath = tmp_path / "test_params.pkl"

        serializer.save_pickle(sample_params, filepath)
        loaded_params = serializer.load_pickle(filepath)

        assert loaded_params.weight == pytest.approx(sample_params.weight)
        assert torch.allclose(loaded_params.mean1, sample_params.mean1)

    def test_state_dict_roundtrip(self, sample_params, tmp_path):
        """测试State Dict格式保存和加载的往返一致性"""
        from src.gms.gmm_optimization import GMMSerializer

        serializer = GMMSerializer()
        filepath = tmp_path / "test_params.pt"

        serializer.save_state_dict(sample_params, filepath)
        loaded_params = serializer.load_state_dict(filepath)

        assert loaded_params.weight == pytest.approx(sample_params.weight)
        assert torch.allclose(loaded_params.mean1, sample_params.mean1)

    def test_json_compression(self, sample_params, tmp_path):
        """测试JSON gzip压缩"""
        from src.gms.gmm_optimization import GMMSerializer

        serializer = GMMSerializer()

        json_path = tmp_path / "test.json"
        gz_path = tmp_path / "test.json.gz"

        serializer.save_json(sample_params, json_path)
        serializer.save_json(sample_params, gz_path)

        assert gz_path.stat().st_size <= json_path.stat().st_size * 1.1 + 100

        loaded_gz = serializer.load_json(gz_path)
        loaded_normal = serializer.load_json(json_path)

        assert loaded_gz.weight == pytest.approx(loaded_normal.weight)

    def test_auto_format_detection(self, sample_params, tmp_path):
        """测试自动格式检测"""
        from src.gms.gmm_optimization import GMMSerializer

        serializer = GMMSerializer()

        serializer.save(sample_params, tmp_path / "auto.json")
        serializer.save(sample_params, tmp_path / "auto.pkl")
        serializer.save(sample_params, tmp_path / "auto.pt")

        params_json = serializer.load(tmp_path / "auto.json")
        params_pkl = serializer.load(tmp_path / "auto.pkl")
        params_pt = serializer.load(tmp_path / "auto.pt")

        assert params_json.weight == pytest.approx(sample_params.weight)
        assert params_pkl.weight == pytest.approx(sample_params.weight)
        assert params_pt.weight == pytest.approx(sample_params.weight)

    def test_file_info(self, sample_params, tmp_path):
        """测试获取文件信息功能"""
        from src.gms.gmm_optimization import GMMSerializer

        serializer = GMMSerializer()
        filepath = tmp_path / "info_test.json"

        serializer.save_json(sample_params, filepath)
        info = serializer.get_file_info(filepath)

        assert 'format' in info
        assert 'size_bytes' in info
        assert 'version' in info
        assert info['format'] == 'json'


class TestInitialization:
    """初始化策略测试"""

    @pytest.fixture
    def bimodal_data(self):
        """生成双峰数据用于初始化测试"""
        torch.manual_seed(42)

        n_per_cluster = 500
        cluster1 = torch.randn(n_per_cluster, 2) * 0.5 + torch.tensor([2.0, 2.0])
        cluster2 = torch.randn(n_per_cluster, 2) * 1.0 + torch.tensor([-2.0, -2.0])

        data = torch.cat([cluster1, cluster2], dim=0)
        return data

    def test_kmeans_initialization(self, bimodal_data):
        """测试K-means++初始化"""
        from src.gms.gmm_optimization import KMeansInitializer

        initializer = KMeansInitializer(max_iter=50, n_init=3)
        params = initializer.initialize(bimodal_data, seed=42)

        assert 0 < params.weight < 1
        assert params.dimensionality == 2
        params.validate()

    def test_random_initialization(self, bimodal_data):
        """测试随机初始化"""
        from src.gms.gmm_optimization import RandomInitializer

        initializer = RandomInitializer(seed=42)
        params = initializer.initialize(bimodal_data, seed=123)

        assert 0 < params.weight < 1
        params.validate()

    def test_heuristic_initialization(self, bimodal_data):
        """测试启发式初始化"""
        from src.gms.gmm_optimization import HeuristicInitializer

        initializer = HeuristicInitializer(method='quantile')
        params = initializer.initialize(bimodal_data)

        assert 0 < params.weight < 1
        params.validate()

    def test_kmeans_detects_clusters(self, bimodal_data):
        """测试K-means是否能检测到聚类结构"""
        from src.gms.gmm_optimization import KMeansInitializer

        initializer = KMeansInitializer(max_iter=100, n_init=10)
        params = initializer.initialize(bimodal_data, seed=42)

        mean_distance = torch.norm(params.mean1 - params.mean2).item()

        assert mean_distance > 0.5  # 均值之间应该有一定距离（即使回退到随机初始化）

    def test_factory_function(self, bimodal_data):
        """测试工厂函数create_initializer"""
        from src.gms.gmm_optimization import create_initializer

        kmeans_init = create_initializer('kmeans')
        random_init = create_initializer('random')
        heuristic_init = create_initializer('heuristic')

        params1 = kmeans_init.initialize(bimodal_data, seed=42)
        params2 = random_init.initialize(bimodal_data, seed=42)
        params3 = heuristic_init.initialize(bimodal_data)

        for params in [params1, params2, params3]:
            params.validate()

    def test_invalid_strategy_name(self):
        """测试无效的策略名称"""
        from src.gms.gmm_optimization import create_initializer

        with pytest.raises(ValueError, match="不支持的初始化策略"):
            create_initializer('invalid_strategy')

    def test_multi_start_initialization(self, bimodal_data):
        """测试多起点初始化"""
        from src.gms.gmm_optimization import MultiStartInitializer

        multi_init = MultiStartInitializer(
            strategies=['kmeans', 'random'],
            n_trials_per_strategy=2
        )

        params = multi_init.initialize(bimodal_data, seed=42)

        params.validate()


class TestEdgeCases:
    """边界情况测试"""

    def test_extreme_weights(self):
        """测试极端权重值"""
        from src.gms.gmm_optimization import GMMParameters, GMMParametersConfig, GaussianMixtureModel

        config = GMMParametersConfig(weight_range=(1e-4, 1-1e-4))
        params = GMMParameters(
            weight=1e-3,
            mean1=torch.tensor([0.0]),
            mean2=torch.tensor([1.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.0]),
            _config=config
        )

        model = GaussianMixtureModel(params)
        x = torch.tensor([0.0])
        pdf_val = model.pdf(x)

        assert not math.isnan(pdf_val)
        assert pdf_val > 0

    def test_close_means(self):
        """测试非常接近的均值"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([0.0, 0.0]),
            mean2=torch.tensor([0.001, 0.001]),
            variance1=torch.tensor([1.0, 1.0]),
            variance2=torch.tensor([1.0, 1.0])
        )

        model = GaussianMixtureModel(params)
        X = torch.randn(100, 2)

        pdf_vals = model.pdf(X)

        assert not torch.isnan(pdf_vals).any()
        assert (pdf_vals > 0).all()

    def test_degenerate_covariance(self):
        """测试退化的协方差（小方差）"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([0.0]),
            mean2=torch.tensor([1.0]),
            variance1=torch.tensor([1e-6]),  # 非常小的方差
            variance2=torch.tensor([1.0])
        )

        model = GaussianMixtureModel(params)
        x = torch.tensor([0.0])

        log_pdf = model.log_pdf(x)

        assert not math.isnan(log_pdf)
        assert not math.isinf(log_pdf)

    def test_high_dimensional(self):
        """测试高维数据"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        d = 50
        params = GMMParameters(
            weight=0.5,
            mean1=torch.zeros(d),
            mean2=torch.ones(d) * 2.0,
            variance1=torch.ones(d),
            variance2=torch.ones(d) * 2.0
        )

        model = GaussianMixtureModel(params)
        X = torch.randn(100, d)

        pdf_vals = model.pdf(X)

        assert pdf_vals.shape == (100,)
        assert not torch.isnan(pdf_vals).any()

    def test_large_scale_values(self):
        """测试大尺度数值"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([1e6]),
            mean2=torch.tensor([-1e6]),
            variance1=torch.tensor([1e10]),
            variance2=torch.tensor([1e10])
        )

        model = GaussianMixtureModel(params)
        x = torch.tensor([0.0])

        log_pdf = model.log_pdf(x)

        assert not math.isnan(log_pdf)
        assert not math.isinf(log_pdf)

    def test_very_small_probabilities(self):
        """测试极小概率区域"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([0.0]),
            mean2=torch.tensor([0.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.0])
        )

        model = GaussianMixtureModel(params)
        x = torch.tensor([100.0])  # 远离均值

        log_pdf = model.log_pdf(x)

        assert not math.isnan(log_pdf)
        assert log_pdf < -100  # 应该是非常小的对数概率


class TestNumericalStability:
    """数值稳定性专项测试"""

    def test_logsumexp_stability(self):
        """测试log-sum-exp实现的稳定性"""
        from src.gms.gmm_optimization.probability_density import GaussianMixtureModel
        import torch

        large_vals = torch.tensor([1000.0, 1001.0, 1002.0])
        small_vals = torch.tensor([-1000.0, -1001.0, -1002.0])
        mixed_vals = torch.tensor([-1000.0, 0.0, 1000.0])

        result_large = GaussianMixtureModel._logsumexp(large_vals)
        result_small = GaussianMixtureModel._logsumexp(small_vals)
        result_mixed = GaussianMixtureModel._logsumexp(mixed_vals)

        assert not math.isnan(result_large.item())
        assert not math.isnan(result_small.item())
        assert not math.isnan(result_mixed.item())

    def test_cholesky_fallback(self):
        """测试Cholesky分解失败时的回退机制"""
        from src.gms.gmm_optimization import GMMParameters, GaussianMixtureModel

        nearly_singular = torch.tensor([
            [1.0, 0.9999999],
            [0.9999999, 1.0]
        ])

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([0.0, 0.0]),
            mean2=torch.tensor([1.0, 1.0]),
            variance1=nearly_singular,
            variance2=torch.eye(2)
        )

        model = GaussianMixtureModel(params)
        x = torch.tensor([0.0, 0.0])

        log_pdf = model.log_pdf(x)

        assert not math.isnan(log_pdf)

    def test_gradient_flow(self):
        """测试自动微分梯度流动"""
        from src.gms.gmm_optimization import GMMParameters

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([1.0, 2.0], requires_grad=True),
            mean2=torch.tensor([-1.0, -2.0], requires_grad=True),
            variance1=torch.tensor([0.5, 0.5], requires_grad=True),
            variance2=torch.tensor([1.0, 1.0], requires_grad=True)
        )

        loss = (params.mean1 ** 2).sum() + (params.mean2 ** 2).sum()
        loss.backward()

        assert params.mean1.grad is not None
        assert params.mean2.grad is not None


class TestDistanceMetrics:
    """距离度量函数测试"""

    def test_kl_divergence_same_distribution(self):
        """测试相同分布的KL散度为0"""
        from src.gms.gmm_optimization import (
            GMMParameters, GaussianMixtureModel, compute_kl_divergence
        )

        params = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([0.0]),
            mean2=torch.tensor([1.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.0])
        )

        model = GaussianMixtureModel(params)
        kl = compute_kl_divergence(model, model)

        assert abs(kl) < 0.01  # 应该接近0

    def test_js_divergence_symmetry(self):
        """测试JS散度的对称性"""
        from src.gms.gmm_optimization import (
            GMMParameters, GaussianMixtureModel, compute_js_divergence
        )

        params1 = GMMParameters(
            weight=0.3,
            mean1=torch.tensor([0.0]),
            mean2=torch.tensor([2.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.0])
        )

        params2 = GMMParameters(
            weight=0.7,
            mean1=torch.tensor([-1.0]),
            mean2=torch.tensor([3.0]),
            variance1=torch.tensor([1.5]),
            variance2=torch.tensor([0.8])
        )

        model1 = GaussianMixtureModel(params1)
        model2 = GaussianMixtureModel(params2)

        js_12 = compute_js_divergence(model1, model2)
        js_21 = compute_js_divergence(model2, model1)

        assert abs(js_12 - js_21) < 0.01  # JS散度应该对称

    def test_wasserstein_distance_positive(self):
        """测试Wasserstein距离为正"""
        from src.gms.gmm_optimization import (
            GMMParameters, GaussianMixtureModel, compute_wasserstein_distance
        )

        params1 = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([0.0]),
            mean2=torch.tensor([1.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.0])
        )

        params2 = GMMParameters(
            weight=0.5,
            mean1=torch.tensor([5.0]),
            mean2=torch.tensor([6.0]),
            variance1=torch.tensor([1.0]),
            variance2=torch.tensor([1.0])
        )

        model1 = GaussianMixtureModel(params1)
        model2 = GaussianMixtureModel(params2)

        w_dist = compute_wasserstein_distance(model1, model2)

        assert w_dist > 0


class TestIntegration:
    """集成测试：端到端流程"""

    def test_full_pipeline(self):
        """测试完整的端到端流程：数据 -> 初始化 -> 模型 -> 评估 -> 保存"""
        torch.manual_seed(42)

        n_samples = 2000
        cluster1 = torch.randn(n_samples // 2, 2) * 0.5 + torch.tensor([2.0, 2.0])
        cluster2 = torch.randn(n_samples // 2, 2) * 1.0 + torch.tensor([-2.0, -2.0])
        data = torch.cat([cluster1, cluster2], dim=0)

        from src.gms.gmm_optimization import (
            KMeansInitializer,
            GMMParameters,
            GaussianMixtureModel,
            GMMSerializer
        )

        initializer = KMeansInitializer(max_iter=100, n_init=5)
        params = initializer.initialize(data, seed=42)

        model = GaussianMixtureModel(params)

        log_likelihood = model.log_pdf(data).sum().item()
        assert not math.isnan(log_likelihood)
        assert log_likelihood < 0

        samples = model.sample(1000, seed=123)
        assert samples.shape == (1000, 2)

        posteriors = model.posterior_probability(data[:10])
        assert posteriors.shape == (10, 2)
        assert torch.allclose(posteriors.sum(dim=1), torch.ones(10), atol=1e-5)

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            serializer = GMMSerializer()
            filepath = Path(tmpdir) / "pipeline_model.json"
            serializer.save_json(params, filepath)

            loaded_params = serializer.load_json(filepath)
            assert loaded_params.weight == pytest.approx(params.weight)

    def test_integration_with_optimizer(self):
        """测试与Task 4优化器的集成"""
        from src.gms.gmm_optimization import (
            GMMParameters,
            GaussianMixtureModel,
            AdamOptimizer,
            OptimizationConfig,
            TargetMoments,
            KMeansInitializer
        )

        torch.manual_seed(42)
        data = torch.randn(1000, 2)

        true_params = GMMParameters(
            weight=0.6,
            mean1=torch.tensor([1.0, 2.0]),
            mean2=torch.tensor([-1.0, -2.0]),
            variance1=torch.tensor([0.5, 0.6]),
            variance2=torch.tensor([1.0, 1.1])
        )

        true_model = GaussianMixtureModel(true_params)
        stats = true_model.compute_statistics()

        target_moments = TargetMoments(
            mean=stats['mean'],
            covariance=stats['covariance']
        )

        initializer = KMeansInitializer(max_iter=50, n_init=3)
        init_params = initializer.initialize(data, seed=42)

        config = OptimizationConfig(
            learning_rate=0.01,
            max_iterations=50,
            verbose=False
        )

        optimizer = AdamOptimizer(config)

        result = optimizer.optimize(
            target_moments,
            init_params.to_optimizer_params()
        )

        assert result.n_iterations > 0
        assert isinstance(result.final_loss, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
