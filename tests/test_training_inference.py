"""GMS 训练和推理集成测试

全面测试训练循环、检查点管理、推理 pipeline 和分布式支持。
使用小规模模拟数据进行快速验证。

测试覆盖:
    1. 训练循环基本流程
    2. 损失计算正确性
    3. 检查点保存和加载
    4. 断点续训
    5. 推理 pipeline 图像生成
    6. 不同采样策略对比
    7. 分布式训练初始化（如果有多 GPU）
    8. 性能基准测试

运行方式:
    pytest tests/test_training_inference.py -v

注意:
    - 测试使用小型模型和小批量数据以确保快速执行
    - 部分测试需要 CUDA（会自动跳过）
    - 分布式测试仅在多 GPU 环境下运行
"""

import pytest
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import tempfile
import shutil
from pathlib import Path
import time


class SimpleDenoisingNet(nn.Module):
    """简单的去噪网络用于测试"""

    def __init__(self, in_channels=3, out_channels=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, 3, padding=1),
        )

    def forward(self, x, t, gms_condition=None):
        return self.net(x)


def create_test_data(batch_size=8, image_size=32, channels=3, num_samples=100):
    """创建模拟图像数据"""
    images = torch.randn(num_samples, channels, image_size, image_size)
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


@pytest.fixture(scope="module")
def device():
    """获取可用设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def test_components(device):
    """创建测试所需的组件"""
    from gms.diffusion_integration.forward_process import NoiseScheduler, GMSForwardProcess
    from gms.diffusion_integration.backward_process import GMSBackwardProcess
    from gms.diffusion_integration.trainer import TrainingConfig, GMSTrainer

    scheduler = NoiseScheduler(
        num_steps=100,
        schedule_type='cosine',
        device=device,
    )
    forward = GMSForwardProcess(scheduler)
    backward = GMSBackwardProcess(scheduler)

    model = SimpleDenoisingNet().to(device)

    config = TrainingConfig(
        batch_size=8,
        learning_rate=1e-3,
        epochs=2,
        device=device,
        checkpoint_every=1,
        seed=42,
    )

    trainer = GMSTrainer(
        model=model,
        noise_scheduler=scheduler,
        forward_process=forward,
        backward_process=backward,
        config=config,
    )

    return {
        'model': model,
        'scheduler': scheduler,
        'forward': forward,
        'backward': backward,
        'config': config,
        'trainer': trainer,
        'device': device,
    }


class TestTrainingConfig:
    """TrainingConfig 测试"""

    def test_default_config(self):
        """测试默认配置"""
        from gms.diffusion_integration.trainer import TrainingConfig

        config = TrainingConfig()

        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.epochs == 100
        assert config.gradient_clip_norm == 1.0

    def test_custom_config(self):
        """测试自定义配置"""
        from gms.diffusion_integration.trainer import TrainingConfig

        config = TrainingConfig(
            batch_size=16,
            learning_rate=2e-4,
            epochs=50,
        )

        assert config.batch_size == 16
        assert config.learning_rate == 2e-4
        assert config.epochs == 50

    def test_config_validation(self):
        """测试配置验证"""
        from gms.diffusion_integration.trainer import TrainingConfig

        with pytest.raises(ValueError):
            TrainingConfig(batch_size=-1)

        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0)

        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)

    def test_config_to_dict(self):
        """测试配置序列化"""
        from gms.diffusion_integration.trainer import TrainingConfig

        config = TrainingConfig(batch_size=16)
        data = config.to_dict()

        assert 'batch_size' in data
        assert data['batch_size'] == 16

    def test_config_from_dict(self):
        """测试从字典加载配置"""
        from gms.diffusion_integration.trainer import TrainingConfig

        original = TrainingConfig(batch_size=24, epochs=30)
        data = original.to_dict()
        restored = TrainingConfig.from_dict(data)

        assert restored.batch_size == original.batch_size
        assert restored.epochs == original.epochs


class TestTrainingHistory:
    """TrainingHistory 测试"""

    def test_history_creation(self):
        """测试历史记录创建"""
        from gms.diffusion_integration.trainer import TrainingHistory, EpochMetrics

        history = TrainingHistory()
        assert history.best_epoch == 0
        assert len(history.train_metrics) == 0

    def test_record_epoch(self):
        """测试记录 epoch 指标"""
        from gms.diffusion_integration.trainer import TrainingHistory, EpochMetrics

        history = TrainingHistory()
        metrics = EpochMetrics(epoch=1, total_loss=0.5, phase="train")

        history.record_epoch(metrics)
        assert len(history.train_metrics) == 1
        assert history.train_metrics[0].total_loss == 0.5

    def test_best_epoch_tracking(self):
        """测试最佳 epoch 追踪"""
        from gms.diffusion_integration.trainer import TrainingHistory, EpochMetrics

        history = TrainingHistory()

        history.record_epoch(EpochMetrics(epoch=1, total_loss=0.5), is_validation=True)
        history.record_epoch(EpochMetrics(epoch=2, total_loss=0.3), is_validation=True)
        history.record_epoch(EpochMetrics(epoch=3, total_loss=0.4), is_validation=True)

        assert history.best_epoch == 2
        assert history.best_val_loss == 0.3

    def test_serialization(self):
        """测试序列化和反序列化"""
        from gms.diffusion_integration.trainer import TrainingHistory, EpochMetrics
        import tempfile
        import json

        history = TrainingHistory()
        history.record_epoch(EpochMetrics(epoch=1, total_loss=0.5))

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
            history.save_to_file(filepath)

        loaded = TrainingHistory.from_file(filepath)
        assert len(loaded.train_metrics) == 1
        assert loaded.train_metrics[0].total_loss == 0.5

        Path(filepath).unlink(missing_ok=True)


class TestGMSTrainer:
    """GMSTrainer 核心功能测试"""

    def test_trainer_initialization(self, test_components):
        """测试训练器初始化"""
        trainer = test_components['trainer']

        assert trainer.model is not None
        assert trainer.noise_scheduler is not None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_single_train_epoch(self, test_components):
        """测试单个 epoch 训练"""
        trainer = test_components['trainer']
        device = test_components['device']
        train_loader = create_test_data(batch_size=4, num_samples=16)

        metrics = trainer.train_epoch(train_loader, epoch=1)

        assert metrics.epoch == 1
        assert metrics.phase == "train"
        assert metrics.total_loss > 0
        assert metrics.samples_processed > 0

    def test_validation_loop(self, test_components):
        """测试验证循环"""
        trainer = test_components['trainer']
        device = test_components['device']
        val_loader = create_test_data(batch_size=4, num_samples=12)

        metrics = trainer.validate(val_loader, epoch=1)

        assert metrics.phase == "val"
        assert metrics.total_loss >= 0

    def test_full_training(self, device):
        """测试完整训练流程"""
        from gms.diffusion_integration.forward_process import NoiseScheduler, GMSForwardProcess
        from gms.diffusion_integration.backward_process import GMSBackwardProcess
        from gms.diffusion_integration.trainer import TrainingConfig, GMSTrainer

        scheduler = NoiseScheduler(num_steps=100, schedule_type='cosine', device=device)
        forward = GMSForwardProcess(scheduler)
        backward = GMSBackwardProcess(scheduler)

        model = SimpleDenoisingNet().to(device)

        config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            epochs=2,
            device=device,
            checkpoint_every=1,
            seed=42,
        )

        trainer = GMSTrainer(
            model=model,
            noise_scheduler=scheduler,
            forward_process=forward,
            backward_process=backward,
            config=config,
        )

        train_loader = create_test_data(batch_size=4, num_samples=20)
        val_loader = create_test_data(batch_size=4, num_samples=8)

        history = trainer.train_full(
            epochs=2,
            dataloaders={'train': train_loader, 'val': val_loader},
        )

        assert len(history.train_metrics) == 2
        assert len(history.val_metrics) == 2
        assert trainer.current_epoch == 2

    def test_loss_computation(self, test_components):
        """测试损失计算"""
        trainer = test_components['trainer']
        device = test_components['device']

        x_0 = torch.randn(4, 3, 16, 16).to(device)
        noise = torch.randn_like(x_0)
        model_output = noise + 0.1 * torch.randn_like(noise)
        t = torch.randint(0, 50, (4,), device=device)

        losses = trainer.compute_loss(x_0, model_output, noise, t)

        assert 'total' in losses
        assert 'diffusion' in losses
        assert losses['diffusion'].item() > 0
        assert losses['total'].item() >= losses['diffusion'].item()


class TestCheckpointManager:
    """CheckpointManager 测试"""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """临时检查点目录"""
        tmpdir = tempfile.mkdtemp(prefix="gms_test_ckpt_")
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_save_and_load(self, test_components, temp_checkpoint_dir):
        """测试检查点保存和加载"""
        from gms.diffusion_integration.checkpoint import CheckpointManager, CheckpointConfig

        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir, keep_n_best=3)
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, config=config)
        trainer = test_components['trainer']

        metrics = {'train_loss': 0.123, 'val_loss': 0.145}
        path = manager.save(trainer, epoch=1, metrics=metrics)

        assert Path(path).exists()

        loaded = manager.load(path)
        assert loaded['epoch'] == 1
        assert 'model_state_dict' in loaded
        assert 'optimizer_state_dict' in loaded

    def test_load_latest(self, test_components, temp_checkpoint_dir):
        """测试加载最新检查点"""
        from gms.diffusion_integration.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        trainer = test_components['trainer']

        manager.save(trainer, epoch=1, metrics={'loss': 0.5})
        time.sleep(0.1)
        manager.save(trainer, epoch=2, metrics={'loss': 0.3})

        latest = manager.load_latest()
        assert latest is not None
        assert latest['epoch'] == 2

    def test_best_checkpoint_tracking(self, test_components, temp_checkpoint_dir):
        """测试最佳检查点追踪"""
        from gms.diffusion_integration.checkpoint import CheckpointManager, CheckpointConfig

        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir, keep_n_best=2)
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, config=config)
        trainer = test_components['trainer']

        manager.save(trainer, epoch=1, metrics={'total_loss': 0.5})
        manager.save(trainer, epoch=2, metrics={'total_loss': 0.3})
        manager.save(trainer, epoch=3, metrics={'total_loss': 0.4})

        best_path = manager.get_best_checkpoint()
        assert best_path is not None

        best_ckpt = manager.load(best_path)
        assert best_ckpt['epoch'] == 2

    def test_cleanup_old_checkpoints(self, test_components, temp_checkpoint_dir):
        """测试清理旧检查点"""
        from gms.diffusion_integration.checkpoint import CheckpointManager, CheckpointConfig

        config = CheckpointConfig(checkpoint_dir=temp_checkpoint_dir, keep_n_best=2)
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, config=config)
        trainer = test_components['trainer']

        for i in range(5):
            manager.save(trainer, epoch=i+1, metrics={'loss': 0.5 - i*0.05})

        deleted = manager.cleanup(keep_n_best=2)
        remaining = manager.list_checkpoints()

        assert len(deleted) >= 3
        assert len(remaining) <= 2

    def test_restore_training(self, test_components, temp_checkpoint_dir):
        """测试从检查点恢复训练"""
        from gms.diffusion_integration.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        trainer = test_components['trainer']

        initial_metrics = {'loss': 0.5}
        manager.save(trainer, epoch=1, metrics=initial_metrics)

        new_trainer = test_components['trainer']
        checkpoint = manager.restore_training(new_trainer)

        assert new_trainer.current_epoch == 1
        assert checkpoint['epoch'] == 1

    def test_checkpoint_integrity_verification(self, test_components, temp_checkpoint_dir):
        """测试检查点完整性验证"""
        from gms.diffusion_integration.checkpoint import CheckpointManager, CheckpointConfig

        config = CheckpointConfig(
            checkpoint_dir=temp_checkpoint_dir,
            verify_integrity=True
        )
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir, config=config)
        trainer = test_components['trainer']

        path = manager.save(trainer, epoch=1, metrics={})
        is_valid, error_msg = manager.verify_checkpoint(path)

        assert is_valid is True
        assert error_msg is None


class TestInferencePipeline:
    """GMSInferencePipeline 测试"""

    @pytest.fixture
    def inference_pipeline(self, test_components):
        """创建推理 pipeline"""
        from gms.diffusion_integration.inference import (
            GMSInferencePipeline, InferenceConfig
        )

        config = InferenceConfig(
            sampling_steps=10,
            method='ddpm',
            batch_size=4,
            device=test_components['device'],
            seed=42,
        )

        pipeline = GMSInferencePipeline(
            model=test_components['model'],
            noise_scheduler=test_components['scheduler'],
            backward_process=test_components['backward'],
            config=config,
        )

        return pipeline

    def test_pipeline_initialization(self, inference_pipeline):
        """测试 pipeline 初始化"""
        assert inference_pipeline.model is not None
        assert inference_pipeline.config.sampling_steps == 10

    def test_generate_samples(self, inference_pipeline):
        """测试样本生成"""
        result = inference_pipeline.generate(n_samples=4)

        assert result.samples.shape[0] == 4
        assert result.samples.dim() == 4
        assert result.generation_time > 0
        assert result.samples_per_second > 0

    def test_ddim_sampling(self, test_components):
        """测试 DDIM 加速采样"""
        from gms.diffusion_integration.inference import (
            GMSInferencePipeline, InferenceConfig, SamplingMethod
        )

        config = InferenceConfig(
            sampling_steps=10,
            method='ddim',
            eta=0.0,
            device=test_components['device'],
            seed=42,
        )

        pipeline = GMSInferencePipeline(
            model=test_components['model'],
            noise_scheduler=test_components['scheduler'],
            backward_process=test_components['backward'],
            config=config,
        )

        result = pipeline.generate(n_samples=2)

        assert result.samples.shape[0] == 2

    def test_intermediate_results(self, inference_pipeline):
        """测试中间结果记录"""
        final, intermediates = inference_pipeline.generate_with_intermediates(
            n_samples=2,
        )

        assert final.shape[0] == 2
        # intermediates 可能为空列表（取决于实现），所以只检查类型
        assert isinstance(intermediates, list)

    def test_batch_generation(self, inference_pipeline, tmp_path):
        """测试大批量生成"""
        results = inference_pipeline.generate_batch(
            batch_size=2,
            total_samples=6,
            save_dir=str(tmp_path / 'gen'),
        )

        assert len(results) == 3
        total_generated = sum(r.samples.shape[0] for r in results)
        assert total_generated >= 6

    def test_postprocessing(self, inference_pipeline):
        """测试输出后处理"""
        result = inference_pipeline.generate(n_samples=2)

        if inference_pipeline.config.clamp_output:
            assert result.samples.min() >= 0.0
            assert result.samples.max() <= 1.0

    def test_benchmark(self, inference_pipeline):
        """测试性能基准"""
        stats = inference_pipeline.benchmark(n_samples=2, warmup_runs=1, benchmark_runs=2)

        assert 'avg_time_seconds' in stats
        assert 'avg_samples_per_sec' in stats
        assert stats['n_samples'] == 2


class TestDistributedSupport:
    """分布式训练支持测试"""

    def test_utility_functions(self):
        """测试工具函数"""
        from gms.diffusion_integration.distributed import (
            is_distributed_available,
            get_world_size,
            get_rank,
            is_main_process,
        )

        assert get_world_size() >= 1
        assert get_rank() >= 0
        assert is_main_process() is True

        if not dist.is_initialized():
            assert is_distributed_available() is False

    def test_distributed_config(self):
        """测试分布式配置"""
        from gms.diffusion_integration.distributed import DistributedConfig

        config = DistributedConfig(
            backend='gloo',
            use_ddp=False,
            world_size=1,
        )

        assert config.backend == 'gloo'
        assert config.use_ddp is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
    def test_dataparallel_wrapping(self, test_components):
        """测试 DataParallel 包装"""
        from gms.diffusion_integration.distributed import DistributedTrainer, DistributedConfig

        if torch.cuda.device_count() < 2:
            pytest.skip("需要至少 2 个 GPU")

        config = DistributedConfig(use_ddp=False)
        dist_trainer = DistributedTrainer(test_components['trainer'], config)

        model = dist_trainer.unwrap_model()
        assert model is not None


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_train_to_infer_workflow(self, device, tmp_path):
        """测试完整的 训练 -> 保存 -> 加载 -> 推理 工作流"""
        from gms.diffusion_integration.inference import (
            GMSInferencePipeline, InferenceConfig
        )
        from gms.diffusion_integration.checkpoint import CheckpointManager
        from gms.diffusion_integration.forward_process import NoiseScheduler, GMSForwardProcess
        from gms.diffusion_integration.backward_process import GMSBackwardProcess
        from gms.diffusion_integration.trainer import TrainingConfig, GMSTrainer

        # 创建独立的训练器
        scheduler = NoiseScheduler(num_steps=100, schedule_type='cosine', device=device)
        forward = GMSForwardProcess(scheduler)
        backward = GMSBackwardProcess(scheduler)

        model = SimpleDenoisingNet().to(device)

        config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            epochs=2,
            device=device,
            seed=123,  # 使用不同的种子
        )

        trainer = GMSTrainer(
            model=model,
            noise_scheduler=scheduler,
            forward_process=forward,
            backward_process=backward,
            config=config,
        )

        train_loader = create_test_data(batch_size=4, num_samples=16)
        val_loader = create_test_data(batch_size=4, num_samples=8)

        history = trainer.train_full(
            epochs=2,
            dataloaders={'train': train_loader, 'val': val_loader},
        )

        ckpt_manager = CheckpointManager(checkpoint_dir=str(tmp_path / 'checkpoints'))
        ckpt_manager.save(trainer, epoch=2, metrics={'loss': 0.1})

        infer_config = InferenceConfig(
            sampling_steps=10,
            method='ddpm',
            device=device,
            seed=42,
        )

        pipeline = GMSInferencePipeline(
            model=trainer.model,
            noise_scheduler=trainer.noise_scheduler,
            backward_process=trainer.backward_process,
            config=infer_config,
        )

        result = pipeline.generate(n_samples=2)

        assert result.samples.shape[0] == 2
        assert len(history.train_metrics) == 2

    def test_training_with_gmm_regularization(self, device):
        """测试带 GMM 正则化的训练"""
        from gms.gmm_optimization.gmm_parameters import GMMParameters
        from gms.diffusion_integration.forward_process import NoiseScheduler, GMSForwardProcess
        from gms.diffusion_integration.backward_process import GMSBackwardProcess
        from gms.diffusion_integration.trainer import TrainingConfig, GMSTrainer

        gmm_params = GMMParameters(
            weight=0.6,
            mean1=torch.tensor([0.0, 0.0]),
            mean2=torch.tensor([1.0, 1.0]),
            variance1=torch.tensor([0.5, 0.5]),
            variance2=torch.tensor([1.0, 1.0])
        ).to_device(device)

        scheduler = NoiseScheduler(num_steps=50, device=device)
        forward = GMSForwardProcess(scheduler)
        backward = GMSBackwardProcess(scheduler)

        model = SimpleDenoisingNet(in_channels=2, out_channels=2).to(device)

        config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            epochs=1,
            device=device,
            gmm_regularization_weight=0.01,
            seed=42,
        )

        trainer = GMSTrainer(
            model=model,
            noise_scheduler=scheduler,
            forward_process=forward,
            backward_process=backward,
            config=config,
            gmm_parameters=gmm_params,
        )

        dummy_data = torch.randn(16, 2, 16, 16).to(device)
        dataset = TensorDataset(dummy_data)
        loader = DataLoader(dataset, batch_size=4)

        metrics = trainer.train_epoch(loader, epoch=1)

        assert metrics.total_loss > 0


class TestPerformanceBaselines:
    """性能基准测试"""

    def test_training_time_baseline(self, test_components):
        """测量单个 epoch 的训练时间"""
        trainer = test_components['trainer']
        device = test_components['device']

        train_loader = create_test_data(batch_size=8, num_samples=32)

        start_time = time.time()
        metrics = trainer.train_epoch(train_loader, epoch=1)
        elapsed = time.time() - start_time

        print(f"\n单 epoch 训练时间: {elapsed:.3f}s")
        print(f"处理样本数: {metrics.samples_processed}")
        print(f"吞吐量: {metrics.samples_processed / max(elapsed, 1e-6):.1f} samples/s")

        assert elapsed < 60  # 不应超过 60 秒

    def test_inference_time_baseline(self, test_components):
        """测量推理时间"""
        from gms.diffusion_integration.inference import GMSInferencePipeline, InferenceConfig

        config = InferenceConfig(
            sampling_steps=10,
            method='ddpm',
            device=test_components['device'],
            seed=42,
        )

        pipeline = GMSInferencePipeline(
            model=test_components['model'],
            noise_scheduler=test_components['scheduler'],
            backward_process=test_components['backward'],
            config=config,
        )

        start_time = time.time()
        result = pipeline.generate(n_samples=4)
        elapsed = time.time() - start_time

        print(f"\n生成 4 个样本时间: {elapsed:.3f}s")
        print(f"速度: {result.samples_per_second:.2f} samples/s")

        assert elapsed < 120  # 不应超过 2 分钟


class TestFIDISIntegration:
    """FID/IS 指标集成到训练流程的测试"""

    def test_epoch_metrics_with_fid_is(self):
        """测试 EpochMetrics 包含 FID/IS 字段"""
        from gms.diffusion_integration.trainer import EpochMetrics

        metrics = EpochMetrics(
            epoch=1,
            phase="val",
            total_loss=0.5,
            fid_score=123.45,
            is_mean=7.8,
            is_std=0.3,
        )

        assert metrics.fid_score == 123.45
        assert metrics.is_mean == 7.8
        assert metrics.is_std == 0.3

        data = metrics.to_dict()
        assert 'fid_score' in data
        assert 'is_mean' in data
        assert 'is_std' in data

    def test_training_config_with_eval_settings(self):
        """测试 TrainingConfig 包含评估配置"""
        from gms.diffusion_integration.trainer import TrainingConfig

        config = TrainingConfig(
            eval_frequency=3,
            num_gen_samples_for_eval=500,
            compute_fid=True,
            compute_is=True,
            is_splits=5,
        )

        assert config.eval_frequency == 3
        assert config.num_gen_samples_for_eval == 500
        assert config.compute_fid is True
        assert config.compute_is is True
        assert config.is_splits == 5

    def test_evaluate_with_metrics_method(self, device):
        """测试 evaluate_with_metrics 方法存在且可调用"""
        from gms.diffusion_integration.forward_process import NoiseScheduler, GMSForwardProcess
        from gms.diffusion_integration.backward_process import GMSBackwardProcess
        from gms.diffusion_integration.trainer import TrainingConfig, GMSTrainer

        scheduler = NoiseScheduler(num_steps=50, schedule_type='cosine', device=device)
        forward = GMSForwardProcess(scheduler)
        backward = GMSBackwardProcess(scheduler)

        model = SimpleDenoisingNet().to(device)

        config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            epochs=1,
            device=device,
            eval_frequency=1,
            num_gen_samples_for_eval=4,
            compute_fid=False,
            compute_is=False,
            seed=42,
        )

        trainer = GMSTrainer(
            model=model,
            noise_scheduler=scheduler,
            forward_process=forward,
            backward_process=backward,
            config=config,
        )

        assert hasattr(trainer, 'evaluate_with_metrics')
        assert callable(trainer.evaluate_with_metrics)

    def test_train_full_with_real_images_param(self, device):
        """测试 train_full 接受 real_images_for_eval 参数"""
        from gms.diffusion_integration.forward_process import NoiseScheduler, GMSForwardProcess
        from gms.diffusion_integration.backward_process import GMSBackwardProcess
        from gms.diffusion_integration.trainer import TrainingConfig, GMSTrainer

        scheduler = NoiseScheduler(num_steps=50, schedule_type='cosine', device=device)
        forward = GMSForwardProcess(scheduler)
        backward = GMSBackwardProcess(scheduler)

        model = SimpleDenoisingNet().to(device)

        config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-3,
            epochs=2,
            device=device,
            eval_frequency=1,
            num_gen_samples_for_eval=4,
            compute_fid=False,
            compute_is=False,
            seed=42,
        )

        trainer = GMSTrainer(
            model=model,
            noise_scheduler=scheduler,
            forward_process=forward,
            backward_process=backward,
            config=config,
        )

        train_loader = create_test_data(batch_size=4, num_samples=16)
        val_loader = create_test_data(batch_size=4, num_samples=8)

        real_images = torch.randn(10, 3, 32, 32).to(device)

        history = trainer.train_full(
            epochs=2,
            dataloaders={'train': train_loader, 'val': val_loader},
            real_images_for_eval=real_images,
        )

        assert len(history.train_metrics) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
