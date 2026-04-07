"""扩散集成层单元测试

覆盖以下模块的全面测试:
- adapter.py: GMSDiffusionAdapter, NoiseSchedule
- forward_process.py: GMSForwardProcess, NoiseScheduler
- backward_process.py: GMSBackwardProcess, DenoisingNetworkWrapper
- condition_injection.py: GMSConditionInjector, GMSEncoder

测试类别:
1. 适配器参数转换和时间步对齐
2. 前向过程噪声添加正确性
3. 反向过程梯度流完整性
4. 条件注入各种模式
5. 端到端完整流程（模拟去噪网络）
6. 数值稳定性（极端时间步、极端参数）
7. 性能基准测试
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import time
import math
from pathlib import Path
from typing import Dict, Any

sys_path = str(Path(__file__).parent.parent / "src")
if sys_path not in __import__('sys').path:
    __import__('sys').path.insert(0, sys_path)

import logging
logger = logging.getLogger(__name__)

from gms.gmm_optimization.gmm_parameters import GMMParameters
from gms.diffusion_integration.adapter import (
    GMSDiffusionAdapter,
    NoiseSchedule,
    AdaptationStrategy,
)
from gms.diffusion_integration.forward_process import (
    GMSForwardProcess,
    NoiseScheduler,
    ScheduleType,
)
from gms.diffusion_integration.backward_process import (
    GMSBackwardProcess,
    DenoisingNetworkWrapper,
    PredictionType,
    BackwardConfig,
    compute_gms_guidance_scale,
    apply_classifier_free_guidance,
)
from gms.diffusion_integration.condition_injection import (
    GMSConditionInjector,
    GMSEncoder,
    ConditionType,
    FiLMLayer,
    AdaptiveGroupNorm,
    CrossAttentionInjector,
    build_full_conditioning_pipeline,
)


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def sample_gmm_params(device):
    """创建测试用的 GMM 参数"""
    return GMMParameters(
        weight=0.6,
        mean1=torch.tensor([0.5, -0.3], device=device),
        mean2=torch.tensor([-0.8, 0.7], device=device),
        variance1=torch.tensor([0.4, 0.4], device=device),
        variance2=torch.tensor([1.2, 1.2], device=device),
    )


class TestNoiseSchedule:
    """NoiseSchedule 数据类测试"""

    def test_creation(self, device):
        means = torch.zeros(100, device=device)
        variances = torch.linspace(0.0001, 0.02, 100, device=device)

        schedule = NoiseSchedule(means=means, variances=variances)

        assert schedule.num_steps == 100
        assert schedule.dimensionality == 1
        assert schedule.means.shape == (100,)
        assert schedule.variances.shape == (100,)
        assert schedule.stds.shape == (100,)

    def test_multidimensional(self, device):
        d = 8
        T = 50
        means = torch.randn(T, d, device=device) * 0.01
        variances = torch.rand(T, d, device=device) * 0.02 + 0.0001

        schedule = NoiseSchedule(means=means, variances=variances)

        assert schedule.num_steps == 50
        assert schedule.dimensionality == d

    def test_get_step(self, device):
        T = 200
        means = torch.zeros(T, device=device)
        variances = torch.linspace(0.001, 0.02, T, device=device)
        schedule = NoiseSchedule(means=means, variances=variances)

        mean_0, var_0 = schedule.get_step(0)
        assert mean_0.dim() == 0
        assert var_0.item() < 0.002

        mean_last, var_last = schedule.get_step(T - 1)
        assert var_last.item() > 0.019

        with pytest.raises(IndexError):
            schedule.get_step(T)
        with pytest.raises(IndexError):
            schedule.get_step(-1)

    def test_interpolation(self, device):
        T_orig = 100
        T_target = 250
        means = torch.sin(torch.linspace(0, 2 * np.pi, T_orig, device=device)) * 0.05
        variances = torch.linspace(0.0001, 0.02, T_orig, device=device)

        schedule = NoiseSchedule(means=means, variances=variances)
        interpolated = schedule.interpolate(T_target)

        assert interpolated.num_steps == T_target
        assert interpolated.means.shape[0] == T_target
        assert interpolated.variances.min() >= 0

    def test_get_range(self, device):
        T = 500
        means = torch.zeros(T, device=device)
        variances = torch.linspace(0.0001, 0.02, T, device=device)
        schedule = NoiseSchedule(means=means, variances=variances)

        sub_schedule = schedule.get_range(start=100, end=300, step=2)

        assert sub_schedule.num_steps == 100
        assert sub_schedule.means.shape[0] == 100

    def test_serialization_roundtrip(self, device, tmp_path):
        T = 50
        means = torch.randn(T, device=device) * 0.01
        variances = torch.rand(T, device=device) * 0.02 + 0.0001

        original = NoiseSchedule(
            means=means,
            variances=variances,
            source_gmm_params={'weight': 0.7},
        )

        data = original.to_dict()
        loaded = NoiseSchedule.from_dict(data)

        assert loaded.num_steps == original.num_steps
        assert torch.allclose(loaded.means, original.means, atol=1e-6)
        assert torch.allclose(loaded.variances, original.variances, atol=1e-6)

        json_file = tmp_path / "schedule.json"
        original.to_json(str(json_file))
        loaded_from_json = NoiseSchedule.from_json(str(json_file))

        assert loaded_from_json.num_steps == original.num_steps

    def test_device_transfer(self, device):
        schedule = NoiseSchedule(
            means=torch.zeros(10),
            variances=torch.ones(10) * 0.01,
        )
        moved = schedule.to(device)
        assert moved.means.device.type == device.type


class TestGMSDiffusionAdapter:
    """GMSDiffusionAdapter 测试"""

    def test_initialization(self):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=1000)
        assert adapter.num_diffusion_steps == 1000
        assert adapter.strategy == AdaptationStrategy.VARIANCE_WEIGHTED

        with pytest.raises(ValueError):
            GMSDiffusionAdapter(num_diffusion_steps=-1)
        with pytest.raises(ValueError):
            GMSDiffusionAdapter(clamp_variance=(0, 10))

    def test_variance_weighted_adaptation(self, sample_gmm_params):
        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=100,
            strategy=AdaptationStrategy.VARIANCE_WEIGHTED,
        )

        schedule = adapter.adapt_gmm_to_diffusion(sample_gmm_params)

        assert isinstance(schedule, NoiseSchedule)
        assert schedule.num_steps == 100
        assert schedule.variances.min() >= 1e-8
        assert schedule.source_gmm_params is not None

    def test_moment_matching_adaptation(self, sample_gmm_params):
        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=100,
            strategy=AdaptationStrategy.MOMENT_MATCHING,
        )

        schedule = adapter.adapt_gmm_to_diffusion(sample_gmm_params)

        assert schedule.num_steps == 100
        assert schedule.variances.min() > 0

    def test_entropy_based_adaptation(self, sample_gmm_params):
        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=100,
            strategy=AdaptationStrategy.ENTROPY_BASED,
        )

        schedule = adapter.adapt_gmm_to_diffusion(sample_gmm_params)

        assert schedule.num_steps == 100
        assert schedule.variances.min() > 0

    def test_custom_adapter_fn(self, sample_gmm_params, device):
        def custom_fn(params, t):
            T = len(t)
            d = params.dimensionality
            means = torch.zeros(T, d, device=device)
            variances = 0.01 * (t ** 2).unsqueeze(-1).expand(T, d)
            return means, variances

        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=50,
            strategy=AdaptationStrategy.CUSTOM,
            custom_adapter_fn=custom_fn,
        )

        schedule = adapter.adapt_gmm_to_diffusion(sample_gmm_params)

        assert schedule.num_steps == 50
        assert schedule.variances[-1].mean().item() > schedule.variances[0].mean().item()

    def test_invalid_custom_fn_raises(self, sample_gmm_params):
        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=10,
            strategy=AdaptationStrategy.CUSTOM,
            custom_adapter_fn=None,
        )

        with pytest.raises(ValueError, match="custom_adapter_fn"):
            adapter.adapt_gmm_to_diffusion(sample_gmm_params)

    def test_invalid_params_type(self):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=10)

        with pytest.raises(TypeError):
            adapter.adapt_gmm_to_diffusion("not a gmm params")

    def test_caching(self, sample_gmm_params):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=20)

        schedule1 = adapter.adapt_gmm_to_diffusion(sample_gmm_params)
        schedule2 = adapter.adapt_gmm_to_diffusion(sample_gmm_params)

        assert schedule1 is schedule2

        schedule3 = adapter.adapt_gmm_to_diffusion(sample_gmm_params, force_recompute=True)
        assert schedule3 is not schedule1

    def test_align_time_steps_nearest(self):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=1000)

        alignment = adapter.align_time_steps(
            gmm_timesteps=[0, 50, 100],
            diffusion_timesteps=list(range(1000)),
            method="nearest",
        )

        assert len(alignment) == 3
        assert 0 in alignment
        assert alignment[0] == 0

    def test_align_time_steps_linear(self):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=1000)

        alignment = adapter.align_time_steps(
            gmm_timesteps=[0, 500, 999],
            diffusion_timesteps=list(range(1000)),
            method="linear",
        )

        assert len(alignment) == 3
        assert alignment[0] == 0
        assert alignment[999] == 999

    def test_align_time_steps_uniform(self):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=1000)

        alignment = adapter.align_time_steps(
            gmm_timesteps=[0, 1, 2],
            diffusion_timesteps=list(range(1000)),
            method="uniform",
        )

        assert len(alignment) == 3

    def test_align_empty_timesteps(self):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=100)

        result = adapter.align_time_steps([], list(range(100)))
        assert result == {}

    def test_transform_samples(self, sample_gmm_params, device):
        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=100,
            device=device,
        )
        schedule = adapter.adapt_gmm_to_diffusion(sample_gmm_params)

        samples = torch.randn(16, 2, device=device)
        transformed = adapter.transform_samples(samples, timestep=50, noise_schedule=schedule)

        assert transformed.shape == samples.shape
        assert not torch.allclose(transformed, samples)

    def test_transform_samples_no_schedule(self):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=10)

        with pytest.raises(RuntimeError, match="噪声调度"):
            adapter.transform_samples(torch.randn(4), timestep=5)

    def test_compute_alpha_schedule(self, sample_gmm_params, device):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=100, device=device)
        schedule = adapter.adapt_gmm_to_diffusion(sample_gmm_params)

        alpha_dict = adapter.compute_alpha_schedule(schedule)

        assert 'betas' in alpha_dict
        assert 'alphas' in alpha_dict
        assert 'alphas_cumprod' in alpha_dict
        assert 'sqrt_alphas_cumprod' in alpha_dict
        assert 'sqrt_one_minus_alphas_cumprod' in alpha_dict

        assert alpha_dict['betas'].shape[0] == 100
        assert alpha_dict['alphas'].min() > 0
        assert alpha_dict['alphas'].max() <= 1.0

    def test_clear_cache(self, sample_gmm_params):
        adapter = GMSDiffusionAdapter(num_diffusion_steps=20)
        adapter.adapt_gmm_to_diffusion(sample_gmm_params)
        assert adapter._noise_schedule_cache is not None

        adapter.clear_cache()
        assert adapter._noise_schedule_cache is None

    def test_export_state(self, sample_gmm_params, device):
        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=500,
            device=device,
        )
        state = adapter.export_state()

        assert state['num_diffusion_steps'] == 500
        assert 'strategy' in state
        assert 'device' in state


class TestNoiseScheduler:
    """NoiseScheduler 测试"""

    @pytest.mark.parametrize("schedule_type", ["linear", "cosine", "sqrt"])
    def test_scheduler_types(self, schedule_type, device):
        scheduler = NoiseScheduler(
            num_steps=1000,
            schedule_type=schedule_type,
            device=device,
        )

        assert scheduler.betas.shape[0] == 1000
        assert scheduler.alphas.shape[0] == 1000
        assert scheduler.alphas_cumprod.shape[0] == 1001

        assert scheduler.betas.min() > 0
        assert scheduler.betas.max() < 1.0
        assert scheduler.alphas.min() > 0
        assert scheduler.alphas.max() < 1.0

    def test_beta_monotonicity_cosine(self, device):
        scheduler = NoiseScheduler(
            num_steps=1000,
            schedule_type="cosine",
            device=device,
        )

        diffs = torch.diff(scheduler.betas)
        assert (diffs >= -1e-6).all(), "Cosine beta should be non-decreasing"

    def test_alpha_cumprod_properties(self, device):
        scheduler = NoiseScheduler(
            num_steps=1000,
            schedule_type="linear",
            device=device,
        )

        assert scheduler.alphas_cumprod[0].item() == pytest.approx(1.0, abs=1e-6)
        assert scheduler.alphas_cumprod[-1].item() > 0
        assert scheduler.alphas_cumprod[-1].item() < 1.0

        for i in range(1, len(scheduler.alphas_cumprod)):
            assert scheduler.alphas_cumprod[i] <= scheduler.alphas_cumprod[i-1] + 1e-6

    def test_posterior_variance(self, device):
        scheduler = NoiseScheduler(num_steps=100, device=device)

        assert scheduler.posterior_variance.shape[0] == 100
        assert scheduler.posterior_variance.min() > 0

    def test_get_alpha_and_sigma_image(self, device):
        scheduler = NoiseScheduler(num_steps=1000, device=device)
        t = torch.tensor([0, 500, 999], device=device)

        sqrt_a, sqrt_om = scheduler.get_alpha_and_sigma(t)

        assert sqrt_a.shape == (3, 1, 1, 1)
        assert sqrt_om.shape == (3, 1, 1, 1)
        assert (sqrt_a > 0).all()
        assert (sqrt_om >= 0).all()
        assert (sqrt_a**2 + sqrt_om**2).allclose(
            torch.ones_like(sqrt_a), atol=1e-4
        )

    def test_get_alpha_and_sigma_flat(self, device):
        scheduler = NoiseScheduler(num_steps=1000, device=device)
        t = torch.tensor([10, 50, 100], device=device)

        sqrt_a, sqrt_om = scheduler.get_alpha_and_sigma_flat(t)

        assert sqrt_a.shape == (3, 1)
        assert sqrt_om.shape == (3, 1)

    def test_custom_betas(self, device):
        custom_betas = torch.logspace(-4, -1, 200, device=device)
        scheduler = NoiseScheduler(
            num_steps=200,
            schedule_type="custom",
            custom_betas=custom_betas,
            device=device,
        )

        assert torch.allclose(scheduler.betas, custom_betas, atol=1e-6)

    def test_get_schedule_info(self, device):
        scheduler = NoiseScheduler(num_steps=100, schedule_type="cosine", device=device)
        info = scheduler.get_schedule_info()

        assert info['type'] == 'cosine'
        assert info['num_steps'] == 100
        assert len(info['beta_range']) == 2

    def test_interpolate_to_steps(self, device):
        scheduler = NoiseScheduler(num_steps=1000, device=device)
        new_scheduler = scheduler.interpolate_to_steps(250)

        assert new_scheduler.num_steps == 250
        assert new_scheduler.betas.shape[0] == 250

    def test_invalid_beta_range(self):
        with pytest.raises(ValueError):
            NoiseScheduler(beta_start=-0.1, beta_end=0.02)
        with pytest.raises(ValueError):
            NoiseScheduler(beta_start=0.01, beta_end=0.005)
        with pytest.raises(ValueError):
            NoiseScheduler(num_steps=-1)


class TestGMSForwardProcess:
    """GMSForwardProcess 测试"""

    def _make_scheduler(self, steps=1000, device="cpu"):
        return NoiseScheduler(num_steps=steps, schedule_type="cosine", device=device)

    def test_forward_basic(self, device):
        scheduler = self._make_scheduler(1000, device)
        forward_proc = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        B, C, H, W = 4, 3, 16, 16
        x_0 = torch.randn(B, C, H, W, device=device)
        t = torch.randint(0, 1000, (B,), device=device)

        x_t, noise = forward_proc(x_0, t)

        assert x_t.shape == x_0.shape
        assert noise.shape == x_0.shape
        assert not torch.isnan(x_t).any()
        assert not torch.isinf(x_t).any()

    def test_forward_with_gmm_noise(self, device):
        scheduler = self._make_scheduler(1000, device)
        forward_proc = GMSForwardProcess(scheduler, gmm_noise_enabled=True)

        B, D = 4, 16
        x_0 = torch.randn(B, D, device=device)
        t = torch.randint(0, 1000, (B,), device=device)

        gmm_params = {
            'mean': torch.zeros(D, device=device),
            'variance': torch.ones(D, device=device) * 0.5,
        }

        x_t, noise = forward_proc(x_0, t, gmm_noise_params=gmm_params)

        assert x_t.shape == x_0.shape
        assert not torch.isnan(x_t).any()

    def test_forward_deterministic_at_t0(self, device):
        scheduler = self._make_scheduler(1000, device)
        forward_proc = GMSForwardProcess(scheduler, gmm_noise_enabled=False, noise_offset=0.0)

        x_0 = torch.randn(2, 4, device=device)
        t_zero = torch.zeros(2, dtype=torch.long, device=device)

        x_t, _ = forward_proc(x_0, t_zero)

        assert torch.allclose(x_t, x_0, atol=5e-2)

    def test_forward_noisy_at_t_max(self, device):
        scheduler = self._make_scheduler(1000, device)
        forward_proc = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        x_0 = torch.randn(4, 8, device=device)
        t_max = torch.full((4,), 999, dtype=torch.long, device=device)

        x_t, noise = forward_proc(x_0, t_max)

        signal_component = (
            scheduler.sqrt_alphas_cumprod[1000] * x_0
        ).abs().mean().item()
        noise_component = scheduler.sqrt_one_minus_alphas_cumprod[1000].item()

        assert noise_component > signal_component * 10, \
            f"At max timestep, noise should dominate: signal={signal_component}, noise_scale={noise_component}"

    def test_forward_full_trajectory(self, device):
        scheduler = self._make_scheduler(100, device)
        forward_proc = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        x_0 = torch.randn(2, 4, device=device)
        trajectory = forward_proc.forward_full(x_0, [0, 25, 50, 75, 99])

        assert len(trajectory) == 5
        for i, x_t in enumerate(trajectory):
            assert x_t.shape == x_0.shape
            assert not torch.isnan(x_t).any()

        variance_should_increase = True
        for i in range(len(trajectory) - 1):
            if trajectory[i].std() > trajectory[i+1].std():
                variance_should_increase = False
                break

    def test_loss_weight_uniform(self, device):
        scheduler = self._make_scheduler(1000, device)
        forward_proc = GMSForwardProcess(scheduler)

        t = torch.randint(0, 1000, (32,), device=device)
        weights = forward_proc.compute_loss_weight(t, "uniform")

        assert weights.shape == (32,)
        assert torch.allclose(weights, torch.ones(32, device=device))

    def test_loss_weight_min_snr(self, device):
        scheduler = self._make_scheduler(1000, device)
        forward_proc = GMSForwardProcess(scheduler)

        t = torch.tensor([0, 100, 500, 900, 999], device=device)
        weights = forward_proc.compute_loss_weight(t, "min_snr")

        assert weights.shape == (5,)
        assert (weights > 0).all()

        early_weight = weights[0]
        late_weight = weights[-1]

        assert late_weight >= early_weight, \
            "Later timesteps should have higher or equal min-SNR weight"

    def test_clip_output(self, device):
        scheduler = self._make_scheduler(100, device)
        forward_proc = GMSForwardProcess(
            scheduler,
            clip_output=True,
            clip_value=5.0,
        )

        x_0 = torch.randn(2, 4, device=device) * 100
        t = torch.full((2,), 99, dtype=torch.long, device=device)

        x_t, _ = forward_proc(x_0, t)

        assert x_t.max() <= 5.0 + 1e-4
        assert x_t.min() >= -5.0 - 1e-4

    def test_visualize_trajectory(self, device):
        scheduler = self._make_scheduler(100, device)
        forward_proc = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        x_0 = torch.randn(3, 8, device=device)
        vis_result = forward_proc.visualize_trajectory(x_0, num_vis_steps=5)

        assert len(vis_result) == 5
        for t, x_t in vis_result.items():
            assert x_t.shape == (3, 8)
            assert 0 <= t < 100


class TestGMSBackwardProcess:
    """GMSBackwardProcess 测试"""

    def _make_components(self, device="cpu", **backward_kwargs):
        scheduler = NoiseScheduler(num_steps=100, schedule_type="cosine", device=device)
        backward = GMSBackwardProcess(
            scheduler,
            prediction_type=PredictionType.EPSILON,
            clip_denoised=True,
            **backward_kwargs,
        )
        return scheduler, backward

    class DummyDenoiseNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.SiLU(),
                nn.Linear(input_dim * 2, input_dim),
            )

        def forward(self, x, t):
            return self.net(x)

    def test_sample_step_epsilon(self, device):
        sched, backward = self._make_components(device)
        model = self.DummyDenoiseNet(8).to(device)

        B, D = 4, 8
        x_t = torch.randn(B, D, device=device)
        t = torch.tensor([50, 50, 50, 50], dtype=torch.long, device=device)

        model_output = model(x_t, t)
        x_prev = backward.sample_step(x_t, t, model_output)

        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
        assert not torch.isinf(x_prev).any()

    def test_sample_step_sample_prediction(self, device):
        sched, backward = self._make_components(device)
        backward.set_prediction_type(PredictionType.SAMPLE)

        B, D = 2, 8
        x_t = torch.randn(B, D, device=device)
        t = torch.tensor([30, 30], dtype=torch.long, device=device)
        x0_pred = torch.randn(B, D, device=device)

        x_prev = backward.sample_step(x_t, t, x0_pred)

        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()

    def test_sample_step_v_prediction(self, device):
        sched, backward = self._make_components(device)
        backward.set_prediction_type(PredictionType.VELOCITY)

        B, D = 2, 8
        x_t = torch.randn(B, D, device=device)
        t = torch.tensor([70, 70], dtype=torch.long, device=device)
        v_pred = torch.randn(B, D, device=device)

        x_prev = backward.sample_step(x_t, t, v_pred)

        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()

    def test_sample_full(self, device):
        sched, backward = self._make_components(device)
        model = self.DummyDenoiseNet(8).to(device)

        x_T = torch.randn(2, 8, device=device)

        def denoise_fn(x_t, t_batch):
            return model(x_t, t_batch)

        x_0 = backward.sample_full(x_T, denoise_fn, all_timesteps=list(range(99, -1, -1)))

        assert x_0.shape == x_T.shape
        assert not torch.isnan(x_0).any()

    def test_gradient_tracking(self, device):
        sched, backward = self._make_components(
            device,
            enable_gradient_norm_tracking=True,
        )
        backward.gradient_clip_value = 0.0

        x_t = torch.randn(2, 4, device=device, requires_grad=True)
        t = torch.tensor([50, 50], dtype=torch.long, device=device)
        model_out = torch.randn(2, 4, device=device)

        x_prev = backward.sample_step(x_t, t, model_out)

        assert backward.last_gradient_norm is not None
        assert backward.last_gradient_norm > 0

        stats = backward.gradient_stats
        assert stats['count'] >= 1

    def test_gradient_clipping(self, device):
        sched, backward = self._make_components(device)
        backward.gradient_clip_value = 0.5

        x_t = torch.randn(2, 4, device=device, requires_grad=True)
        t = torch.tensor([80, 80], dtype=torch.long, device=device)
        model_out = torch.randn(2, 4, device=device) * 10

        x_prev = backward.sample_step(x_t, t, model_out)

        assert x_prev.max() <= backward.clip_range[1] + 1e-4
        assert x_prev.min() >= backward.clip_range[0] - 1e-4

    def test_reset_gradient_tracking(self, device):
        sched, backward = self._make_components(
            device,
            enable_gradient_norm_tracking=True,
        )

        x_t = torch.randn(1, 2, device=device, requires_grad=True)
        t = torch.tensor([50], dtype=torch.long, device=device)
        backward.sample_step(x_t, t, torch.randn(1, 2, device=device))

        assert backward.gradient_stats['count'] == 1

        backward.reset_gradient_tracking()
        assert backward.gradient_stats['count'] == 0
        assert backward.last_gradient_norm is None

    def test_switch_prediction_type(self, device):
        _, backward = self._make_components(device)

        assert backward.prediction_type == PredictionType.EPSILON
        backward.set_prediction_type("sample")
        assert backward.prediction_type == PredictionType.SAMPLE
        backward.set_prediction_type(PredictionType.V)
        assert backward.prediction_type == PredictionType.V

    def test_export_state(self, device):
        _, backward = self._make_components(device)
        state = backward.export_state()

        assert 'prediction_type' in state
        assert 'clip_denoised' in state
        assert 'gradient_stats' in state

    def test_final_step_no_noise(self, device):
        sched, backward = self._make_components(device)

        x_t = torch.randn(2, 4, device=device)
        t_zero = torch.zeros(2, dtype=torch.long, device=device)
        model_out = torch.randn(2, 4, device=device)

        x_prev = backward.sample_step(x_t, t_zero, model_out)

        assert x_prev.shape == x_t.shape


class TestDenoisingNetworkWrapper:
    """DenoisingNetworkWrapper 测试"""

    def test_concat_injection(self, device):
        base_net = nn.Linear(12, 8)
        wrapper = DenoisingNetworkWrapper(
            base_net,
            condition_dim=4,
            output_dim=8,
            condition_injection="concat",
        ).to(device)

        x = torch.randn(4, 8, device=device)
        t = torch.zeros(4, dtype=torch.long, device=device)
        cond = torch.randn(4, 4, device=device)

        output = wrapper(x, t, gms_condition=cond)

        assert output.shape == (4, 8)

    def test_film_injection(self, device):
        base_net = nn.Linear(8, 8)
        wrapper = DenoisingNetworkWrapper(
            base_net,
            condition_dim=16,
            output_dim=8,
            condition_injection="film",
            use_condition_encoder=True,
            condition_hidden_dim=32,
        ).to(device)

        x = torch.randn(4, 8, device=device)
        t = torch.zeros(4, dtype=torch.long, device=device)
        cond = torch.randn(4, 16, device=device)

        output = wrapper(x, t, gms_condition=cond)

        assert output.shape == (4, 8)

    def test_no_condition(self, device):
        base_net = nn.Linear(8, 8)
        wrapper = DenoisingNetworkWrapper(base_net, output_dim=8).to(device)

        x = torch.randn(4, 8, device=device)
        t = torch.zeros(4, dtype=torch.long, device=device)

        output = wrapper(x, t)

        assert output.shape == (4, 8)

    def test_split_output_mean_var(self, device):
        base_net = nn.Linear(8, 16)
        wrapper = DenoisingNetworkWrapper(base_net, output_dim=8).to(device)

        x = torch.randn(2, 8, device=device)
        t = torch.zeros(2, dtype=torch.long, device=device)

        output = wrapper(x, t)
        mean, log_var = wrapper.split_output_to_mean_var(output)

        assert mean.shape == (2, 8)
        assert log_var.shape == (2, 8)

    def test_invalid_injection_type(self):
        base_net = nn.Linear(8, 8)
        with pytest.raises(ValueError):
            DenoisingNetworkWrapper(base_net, condition_injection="invalid_type")

    def test_none_injection(self, device):
        base_net = nn.Linear(8, 8)
        wrapper = DenoisingNetworkWrapper(
            base_net,
            condition_dim=4,
            condition_injection="none",
        ).to(device)

        x = torch.randn(2, 8, device=device)
        t = torch.zeros(2, dtype=torch.long, device=device)
        cond = torch.randn(2, 4, device=device)

        output = wrapper(x, t, gms_condition=cond)
        assert output.shape == (2, 8)


class TestGMSEncoder:
    """GMSEncoder 测试"""

    def test_encode_from_dict(self, device):
        encoder = GMSEncoder(output_dim=32, input_dim=9).to(device)

        params = {
            'weight': 0.6,
            'mean1': torch.tensor([0.5, -0.3], device=device),
            'mean2': torch.tensor([-0.8, 0.7], device=device),
            'variance1': torch.tensor([0.4, 0.4], device=device),
            'variance2': torch.tensor([1.2, 1.2], device=device),
        }

        cond = encoder(params)

        assert cond.shape == (32,)
        assert cond.dtype == torch.float32
        assert not torch.isnan(cond).any()

    def test_encode_from_tensor(self, device):
        encoder = GMSEncoder(output_dim=16, input_dim=8).to(device)

        raw_features = torch.randn(8, device=device)
        cond = encoder(raw_features)

        assert cond.shape == (16,)

    def test_with_time_embedding(self, device):
        encoder = GMSEncoder(
            output_dim=24,
            input_dim=5,
            time_embedding_dim=16,
        ).to(device)

        params = {
            'weight': 0.5,
            'mean1': torch.tensor([1.0]),
            'mean2': torch.tensor([-1.0]),
            'variance1': torch.tensor([0.5]),
            'variance2': torch.tensor([0.5]),
        }
        t_ratio = torch.tensor([0.5], device=device)

        cond = encoder(params, timestep_ratio=t_ratio)

        assert cond.shape == (24,)

    def test_encode_batch(self, device):
        encoder = GMSEncoder(output_dim=16, input_dim=9).to(device)

        batch_params = [
            {'weight': 0.3, 'mean1': torch.zeros(2, device=device),
             'mean2': torch.ones(2, device=device),
             'variance1': torch.ones(2, device=device) * 0.5,
             'variance2': torch.ones(2, device=device)},
            {'weight': 0.7, 'mean1': torch.ones(2, device=device),
             'mean2': torch.zeros(2, device=device),
             'variance1': torch.ones(2, device=device),
             'variance2': torch.ones(2, device=device) * 0.3},
        ]

        batch_cond = encoder.encode_batch(batch_params)

        assert batch_cond.shape == (2, 16)

    def test_different_activations(self, device):
        for act in ['relu', 'silu', 'gelu', 'tanh']:
            encoder = GMSEncoder(output_dim=8, input_dim=5, activation=act).to(device)
            params = {'weight': 0.5, 'mean1': torch.zeros(1, device=device),
                      'mean2': torch.ones(1, device=device),
                      'variance1': torch.ones(1, device=device),
                      'variance2': torch.ones(1, device=device)}
            cond = encoder(params)
            assert cond.shape == (8,)


class TestConditionInjectionModes:
    """条件注入模式全面测试"""

    @pytest.fixture(params=["film", "adagn", "concat"])
    def injector(self, request, device):
        return GMSConditionInjector(
            feature_dim=64,
            condition_dim=32,
            condition_type=request.param,
            temperature=1.0,
            device=device,
        )

    def test_film_injection(self, device):
        injector = GMSConditionInjector(
            feature_dim=32,
            condition_dim=16,
            condition_type=ConditionType.FILM,
            temperature=1.0,
            use_residual=False,
        )

        features = torch.randn(4, 32, 8, 8, device=device)
        condition = torch.randn(4, 16, device=device)

        output = injector(features, condition)

        assert output.shape == features.shape
        assert not torch.allclose(output, features)

    def test_cross_attention_injection(self, device):
        injector = GMSConditionInjector(
            feature_dim=32,
            condition_dim=16,
            condition_type=ConditionType.CROSS_ATTENTION,
            num_heads=4,
        )

        features = torch.randn(2, 32, 16, 16, device=device)
        condition = torch.randn(2, 16, device=device)

        output = injector(features, condition)

        assert output.shape == features.shape

    def test_adagn_injection(self, device):
        injector = GMSConditionInjector(
            feature_dim=16,
            condition_dim=32,
            condition_type=ConditionType.ADAGN,
            num_groups=4,
        )

        features = torch.randn(4, 16, 8, 8, device=device)
        condition = torch.randn(4, 32, device=device)

        output = injector(features, condition)

        assert output.shape == features.shape

    def test_concat_injection(self, device):
        injector = GMSConditionInjector(
            feature_dim=16,
            condition_dim=8,
            condition_type=ConditionType.CONCAT,
        )

        features = torch.randn(4, 16, 4, 4, device=device)
        condition = torch.randn(4, 8, device=device)

        output = injector(features, condition)

        assert output.shape == features.shape

    def test_none_injection_passthrough(self, device):
        injector = GMSConditionInjector(
            feature_dim=16,
            condition_dim=8,
            condition_type=ConditionType.NONE,
        ).to(device)

        features = torch.randn(4, 16, device=device)
        condition = torch.randn(4, 8, device=device)

        output = injector(features, condition)

        assert torch.equal(output, features)

    def test_residual_connection(self, device):
        injector_with_residual = GMSConditionInjector(
            feature_dim=16,
            condition_dim=8,
            condition_type=ConditionType.FILM,
            use_residual=True,
        ).to(device)

        injector_no_residual = GMSConditionInjector(
            feature_dim=16,
            condition_dim=8,
            condition_type=ConditionType.FILM,
            use_residual=False,
        ).to(device)

        features = torch.randn(4, 16, device=device)
        condition = torch.randn(4, 8, device=device)

        out_resid = injector_with_residual(features, condition)
        out_no_resid = injector_no_residual(features, condition)

        assert not torch.equal(out_resid, out_no_resid)

    def test_temperature_scaling(self, device):
        injector = GMSConditionInjector(
            feature_dim=16,
            condition_dim=8,
            condition_type=ConditionType.FILM,
            temperature=1.0,
        )

        features = torch.randn(4, 16, device=device)
        condition = torch.randn(4, 8, device=device)

        output_normal = injector(features, condition)

        injector.set_temperature(2.0)
        output_high_temp = injector(features, condition)

        assert output_high_temp.shape == output_normal.shape

        injector.set_temperature(0.1)
        output_low_temp = injector(features, condition)

        assert output_low_temp.shape == output_normal.shape

    def test_invalid_temperature(self, device):
        injector = GMSConditionInjector(feature_dim=16, condition_dim=8).to(device)

        with pytest.raises(ValueError):
            injector.set_temperature(0.0)
        with pytest.raises(ValueError):
            injector.set_temperature(-1.0)

    def test_return_modulation_params_film(self, device):
        injector = GMSConditionInjector(
            feature_dim=16,
            condition_dim=8,
            condition_type=ConditionType.FILM,
        ).to(device)

        features = torch.randn(2, 16, device=device)
        condition = torch.randn(2, 8, device=device)

        output, mod_params = injector(features, condition, return_modulation_params=True)

        assert mod_params is not None
        gamma, beta = mod_params
        assert gamma.shape == features.shape
        assert beta.shape == features.shape

    def test_get_stats(self, device):
        injector = GMSConditionInjector(
            feature_dim=32,
            condition_dim=16,
            condition_type=ConditionType.FILM,
            temperature=1.5,
        ).to(device)

        stats = injector.get_injection_stats()

        assert stats['type'] == 'film'
        assert stats['feature_dim'] == 32
        assert stats['condition_dim'] == 16
        assert stats['temperature'] == 1.5
        assert stats['has_film'] is True
        assert stats['has_cross_attn'] is False

    def test_switch_mode(self, device):
        injector = GMSConditionInjector(
            feature_dim=16,
            condition_dim=8,
            condition_type=ConditionType.FILM,
        ).to(device)

        assert injector.condition_type == ConditionType.FILM

        injector.switch_mode("concat")
        assert injector.condition_type == ConditionType.CONCAT

    def test_dimension_mismatch_error(self, device):
        injector = GMSConditionInjector(
            feature_dim=32,
            condition_dim=8,
            condition_type=ConditionType.FILM)
        injector = injector.to(device)

        wrong_features = torch.randn(4, 64, device=device)
        condition = torch.randn(4, 8, device=device)

        with pytest.raises(RuntimeError, match="特征维度不匹配"):
            injector(wrong_features, condition)


class TestFiLMLayer:
    """FiLM 层单独测试"""

    def test_film_basic(self, device):
        film = FiLMLayer(feature_dim=16, condition_dim=8).to(device)

        features = torch.randn(4, 16, device=device)
        condition = torch.randn(4, 8, device=device)

        output, (gamma, beta) = film(features, condition)

        assert output.shape == features.shape
        assert gamma.shape == features.shape
        assert beta.shape == features.shape

    def test_film_identity_init(self, device):
        film = FiLMLayer(feature_dim=8, condition_dim=4).to(device)

        zero_cond = torch.zeros(2, 4, device=device)
        features = torch.randn(2, 8, device=device)

        output, _ = film(features, zero_cond)

        assert torch.allclose(output, torch.zeros_like(features), atol=1e-3)


class TestAdaptiveGroupNorm:
    """AdaGN 单独测试"""

    def test_adagn_basic(self, device):
        adagn = AdaptiveGroupNorm(
            num_channels=16,
            num_groups=4,
            condition_dim=32,
        ).to(device)

        x = torch.randn(4, 16, 8, 8, device=device)
        cond = torch.randn(4, 32, device=device)

        output = adagn(x, cond)

        assert output.shape == x.shape

    def test_adagn_channels_divisible(self):
        with pytest.raises(AssertionError):
            AdaptiveGroupNorm(num_channels=15, num_groups=4, condition_dim=8)


class TestCrossAttentionInjector:
    """交叉注意力注入器测试"""

    def test_cross_attn_basic(self, device):
        ca = CrossAttentionInjector(
            feature_dim=32,
            condition_dim=16,
            num_heads=4,
        ).to(device)

        features = torch.randn(2, 32, 16, 16, device=device)
        condition = torch.randn(2, 16, device=device)

        output = ca(features, condition)

        assert output.shape == features.shape

    def test_cross_attn_heads_divisible(self):
        with pytest.raises(AssertionError):
            CrossAttentionInjector(feature_dim=17, condition_dim=8, num_heads=4)


class TestBuildFullPipeline:
    """完整管线构建测试"""

    def test_pipeline_construction(self, device):
        encoder, injector = build_full_conditioning_pipeline(
            feature_dim=128,
            condition_dim=32,
            injection_type='film',
            gmm_input_dim=9,
            temperature=1.0,
        )

        encoder = encoder.to(device)
        injector = injector.to(device)

        params = {
            'weight': 0.5,
            'mean1': torch.zeros(2, device=device),
            'mean2': torch.ones(2, device=device),
            'variance1': torch.ones(2, device=device),
            'variance2': torch.ones(2, device=device),
        }

        cond = encoder(params).unsqueeze(0).expand(4, -1)
        features = torch.randn(4, 128, 8, 8, device=device)

        output = injector(features, cond)

        assert output.shape == features.shape


class TestGuidanceFunctions:
    """引导函数测试"""

    def test_guidance_scale_computation(self):
        scale = compute_gms_guidance_scale(
            base_scale=7.5,
            gmm_weight=0.9,
            timestep_ratio=0.1,
        )
        assert scale > 7.5

        scale_balanced = compute_gms_guidance_scale(
            base_scale=7.5,
            gmm_weight=0.5,
            timestep_ratio=0.5,
        )
        assert scale_balanced >= 1.0

    def test_classifier_free_guidance(self, device):
        cond_out = torch.randn(4, 8, device=device)
        uncond_out = torch.randn(4, 8, device=device)

        guided = apply_classifier_free_guidance(
            cond_out, uncond_out, guidance_scale=7.5
        )

        expected = uncond_out + 7.5 * (cond_out - uncond_out)
        assert torch.allclose(guided, expected, atol=1e-6)


class TestNumericalStability:
    """数值稳定性测试"""

    def test_extreme_timestep_small(self, device):
        scheduler = NoiseScheduler(num_steps=1000, device=device)
        fp = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        x_0 = torch.randn(2, 4, device=device)
        t = torch.zeros(2, dtype=torch.long, device=device)

        x_t, noise = fp(x_0, t)

        assert not torch.isnan(x_t).any()
        assert not torch.isinf(x_t).any()

    def test_extreme_timestep_large(self, device):
        scheduler = NoiseScheduler(num_steps=1000, device=device)
        fp = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        x_0 = torch.randn(2, 4, device=device)
        t = torch.full((2,), 999, dtype=torch.long, device=device)

        x_t, noise = fp(x_0, t)

        assert not torch.isnan(x_t).any()
        assert not torch.isinf(x_t).any()

    def test_extreme_input_values(self, device):
        scheduler = NoiseScheduler(num_steps=100, device=device)
        fp = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        x_0 = torch.randn(2, 4, device=device) * 1e6
        t = torch.tensor([50, 50], dtype=torch.long, device=device)

        x_t, _ = fp(x_0, t)

        assert not torch.isnan(x_t).any()
        assert not torch.isinf(x_t).any()

    def test_near_zero_variance(self, device):
        near_zero_var = torch.tensor([1e-10, 1e-10], device=device)
        schedule = NoiseSchedule(
            means=torch.zeros(10, 2, device=device),
            variances=near_zero_var.unsqueeze(0).expand(10, -1).clone(),
        )

        stds = schedule.stds
        assert (stds > 0).all()
        assert not torch.isnan(stds).any()

    def test_large_variance(self, device):
        large_var = torch.tensor([100.0, 100.0], device=device)
        schedule = NoiseSchedule(
            means=torch.zeros(5, 2, device=device),
            variances=large_var.unsqueeze(0).expand(5, -1).clone(),
        )

        stds = schedule.stds
        assert not torch.isnan(stds).any()
        assert not torch.isinf(stds).any()


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_complete_forward_backward_cycle(self, device):
        gmm_params = GMMParameters(
            weight=0.55,
            mean1=torch.tensor([0.3, -0.2], device=device),
            mean2=torch.tensor([-0.6, 0.8], device=device),
            variance1=torch.tensor([0.35, 0.35], device=device),
            variance2=torch.tensor([1.1, 1.1], device=device),
        )

        adapter = GMSDiffusionAdapter(
            num_diffusion_steps=100,
            strategy=AdaptationStrategy.VARIANCE_WEIGHTED,
            device=device,
        )
        noise_schedule = adapter.adapt_gmm_to_diffusion(gmm_params)

        scheduler = NoiseScheduler(
            num_steps=100,
            schedule_type="cosine",
            device=device,
        )
        forward_proc = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        backward_proc = GMSBackwardProcess(
            scheduler,
            prediction_type=PredictionType.EPSILON,
        )

        encoder = GMSEncoder(output_dim=16, input_dim=9).to(device)
        injector = GMSConditionInjector(
            feature_dim=8,
            condition_dim=16,
            condition_type=ConditionType.FILM,
        ).to(device)

        class SimpleDenoiser(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 8))
            def forward(self, x, t):
                return self.net(x)

        denoiser = SimpleDenoiser().to(device)

        x_clean = torch.randn(4, 8, device=device)
        t_forward = torch.tensor([20, 40, 60, 80], dtype=torch.long, device=device)

        x_noisy, noise_added = forward_proc(x_clean, t_forward)

        assert x_noisy.shape == x_clean.shape
        assert not torch.isnan(x_noisy).any()

        gmm_dict = {
            'weight': gmm_params.weight,
            'mean1': gmm_params.mean1,
            'mean2': gmm_params.mean2,
            'variance1': gmm_params.variance1,
            'variance2': gmm_params.variance2,
        }
        cond_vector = encoder(gmm_dict)

        cond_expanded = cond_vector.unsqueeze(0).expand(4, -1)
        conditioned_x = injector(x_noisy, cond_expanded)

        assert conditioned_x.shape[0] == 4

        model_output = denoiser(conditioned_x, t_forward)
        x_denoised = backward_proc.sample_step(x_noisy, t_forward, model_output)

        assert x_denoised.shape == x_clean.shape
        assert not torch.isnan(x_denoised).any()

    def test_multi_step_generation(self, device):
        scheduler = NoiseScheduler(num_steps=50, device=device)
        backward = GMSBackwardProcess(scheduler, prediction_type=PredictionType.EPSILON)

        class SimpleModel(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.fc = nn.Linear(dim, dim)
            def forward(self, x, t):
                return self.fc(x) * 0.1

        model = SimpleModel(4).to(device)
        x_T = torch.randn(2, 4, device=device)

        x_generated = backward.sample_full(
            x_T,
            lambda x, t: model(x, t),
            all_timesteps=list(range(49, -1, -1)),
        )

        assert x_generated.shape == x_T.shape
        assert not torch.isnan(x_generated).any()


class TestPerformanceBenchmarks:
    """性能基准测试"""

    def benchmark_forward_process(self, device):
        scheduler = NoiseScheduler(num_steps=1000, device=device)
        fp = GMSForwardProcess(scheduler, gmm_noise_enabled=False)

        x_0 = torch.randn(32, 3, 32, 32, device=device)
        t = torch.randint(0, 1000, (32,), device=device)

        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = fp(x_0, t)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        logger.info(f"Forward process benchmark (batch=32, 3x32x32): {avg_time*1000:.2f}ms")
        return avg_time

    def benchmark_backward_single_step(self, device):
        scheduler = NoiseScheduler(num_steps=1000, device=device)
        backward = GMSBackwardProcess(scheduler)

        model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to(device)

        x_t = torch.randn(8, 64, device=device)
        t = torch.full((8,), 500, dtype=torch.long, device=device)

        times = []
        for _ in range(20):
            start = time.perf_counter()
            with torch.no_grad():
                model_out = model(x_t)
                _ = backward.sample_step(x_t, t, model_out)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        logger.info(f"Backward single-step benchmark (batch=8, dim=64): {avg_time*1000:.2f}ms")
        return avg_time

    def benchmark_condition_injection(self, device):
        injector = GMSConditionInjector(
            feature_dim=256,
            condition_dim=64,
            condition_type=ConditionType.FILM,
        ).to(device)

        features = torch.randn(16, 256, 16, 16, device=device)
        condition = torch.randn(16, 64, device=device)

        times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = injector(features, condition)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        logger.info(f"Condition injection benchmark (batch=16, 256x16x16): {avg_time*1000:.2f}ms")
        return avg_time

    def test_performance_benchmarks_run(self, device):
        fwd_time = self.benchmark_forward_process(device)
        bwd_time = self.benchmark_backward_single_step(device)
        inj_time = self.benchmark_condition_injection(device)

        assert fwd_time > 0
        assert bwd_time > 0
        assert inj_time > 0

        assert fwd_time < 10.0, f"Forward too slow: {fwd_time}s"
        assert bwd_time < 5.0, f"Backward too slow: {bwd_time}s"
        assert inj_time < 1.0, f"Injection too slow: {inj_time}s"
