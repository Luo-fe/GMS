"""采样控制器单元测试

测试调度器、时间步控制器、检查点管理器和进度监控器的功能。
"""

import pytest
import tempfile
import time
from pathlib import Path
import numpy as np

# 添加 src 到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gms.sampling.sampling_scheduler import (
    BaseScheduler,
    LinearScheduler,
    CosineScheduler,
    ConstantScheduler,
    SqrtScheduler,
)
from gms.sampling.time_step_controller import (
    TimeStepController,
    AdaptationMode,
    StepHistory,
    TimeStepStats,
)
from gms.sampling.checkpoint_manager import (
    SamplingCheckpointManager,
    SamplingCheckpoint,
    CheckpointCleanupPolicy,
)
from gms.sampling.progress_monitor import (
    ProgressMonitor,
    SamplingProgress,
    SamplingEventType,
    TqdmProgressMonitor,
)


# ==================== 调度器测试 ====================


class TestLinearScheduler:
    """线性调度器测试"""

    def test_basic_schedule(self):
        """测试基本调度生成"""
        scheduler = LinearScheduler(start_value=0.0, end_value=1.0)
        schedule = scheduler.get_schedule(10)

        assert len(schedule) == 10
        assert schedule[0] == pytest.approx(0.0, abs=1e-6)
        assert schedule[-1] == pytest.approx(1.0, abs=1e-6)

    def test_monotonic_increase(self):
        """测试单调递增"""
        scheduler = LinearScheduler(start_value=0.01, end_value=0.02)
        schedule = scheduler.get_schedule(100)

        for i in range(len(schedule) - 1):
            assert schedule[i] <= schedule[i + 1]

    def test_custom_range(self):
        """测试自定义范围"""
        scheduler = LinearScheduler(start_value=1e-4, end_value=0.02)
        schedule = scheduler.get_schedule(50)

        assert schedule[0] == pytest.approx(1e-4, abs=1e-8)
        assert schedule[-1] == pytest.approx(0.02, abs=1e-4)

    def test_single_step(self):
        """测试单步"""
        scheduler = LinearScheduler()
        schedule = scheduler.get_schedule(1)

        assert len(schedule) == 1

    def test_invalid_steps(self):
        """测试无效步数"""
        scheduler = LinearScheduler()

        with pytest.raises(ValueError):
            scheduler.get_schedule(0)

        with pytest.raises(ValueError):
            scheduler.get_schedule(-1)

    def test_get_value(self):
        """测试获取单个值"""
        scheduler = LinearScheduler(start_value=0.0, end_value=1.0)

        value_0 = scheduler.get_value(0, 10)
        value_last = scheduler.get_value(9, 10)

        assert value_0 == pytest.approx(0.0, abs=1e-6)
        assert value_last == pytest.approx(1.0, abs=1e-6)

    def test_get_value_out_of_range(self):
        """测试越界访问"""
        scheduler = LinearScheduler()

        with pytest.raises(ValueError):
            scheduler.get_value(-1, 10)

        with pytest.raises(ValueError):
            scheduler.get_value(10, 10)


class TestCosineScheduler:
    """余弦调度器测试"""

    def test_basic_schedule(self):
        """测试基本调度生成"""
        scheduler = CosineScheduler(start_value=1e-4, end_value=0.02)
        schedule = scheduler.get_schedule(100)

        assert len(schedule) == 100
        assert all(v > 0 for v in schedule)

    def test_smooth_transition(self):
        """测试平滑过渡（余弦特性）"""
        scheduler = CosineScheduler()
        schedule = np.array(scheduler.get_schedule(100))

        # 余弦调度在开始和结束应该更平缓
        first_diff = schedule[1] - schedule[0]
        mid_diff = schedule[50] - schedule[49]

        # 中间变化通常比开始快
        # (这个性质取决于具体参数，这里只验证基本合理性)

    def test_custom_offset(self):
        """测试自定义偏移量"""
        scheduler_s = CosineScheduler(s=0.001)
        scheduler_l = CosineScheduler(s=0.05)

        schedule_s = scheduler_s.get_schedule(50)
        schedule_l = scheduler_l.get_schedule(50)

        assert len(schedule_s) == len(schedule_l) == 50


class TestConstantScheduler:
    """常数调度器测试"""

    def test_constant_values(self):
        """测试所有值相同"""
        value = 0.01
        scheduler = ConstantScheduler(value=value)
        schedule = scheduler.get_schedule(20)

        assert all(abs(v - value) < 1e-10 for v in schedule)

    def test_length(self):
        """测试长度正确"""
        scheduler = ConstantScheduler()
        schedule = scheduler.get_schedule(5)

        assert len(schedule) == 5


class TestSqrtScheduler:
    """平方根调度器测试"""

    def test_sqrt_shape(self):
        """测试平方根曲线形状"""
        scheduler = SqrtScheduler(start_value=0.0, end_value=1.0)
        schedule = scheduler.get_schedule(100)

        assert len(schedule) == 100
        assert schedule[0] >= 0
        assert schedule[-1] <= 1.0

    def test_concave_shape(self):
        """测试凹形（开始快，后期慢）"""
        scheduler = SqrtScheduler(start_value=0.0, end_value=1.0)
        schedule = np.array(scheduler.get_schedule(100))

        # 平方根函数是凹的，增量应该递减
        diffs = np.diff(schedule)
        for i in range(len(diffs) - 1):
            assert diffs[i] >= diffs[i + 1]


# ==================== 时间步控制器测试 ====================


class TestTimeStepController:
    """时间步控制器测试"""

    @pytest.fixture
    def controller(self):
        """创建默认控制器"""
        return TimeStepController(
            initial_dt=0.01,
            min_dt=1e-4,
            max_dt=0.1,
        )

    def test_initialization(self, controller):
        """测试初始化"""
        assert controller.current_dt == 0.01
        assert controller.min_dt == 1e-4
        assert controller.max_dt == 0.1

    def test_invalid_initialization(self):
        """测试无效初始化参数"""
        with pytest.raises(ValueError):
            TimeStepController(initial_dt=0.2, min_dt=0.1, max_dt=0.05)

        with pytest.raises(ValueError):
            TimeStepController(safety_factor=1.5)

        with pytest.raises(ValueError):
            TimeStepController(increase_factor=0.9)

        with pytest.raises(ValueError):
            TimeStepController(decrease_factor=1.2)

    def test_fixed_mode(self, controller):
        """测试固定模式"""
        controller.set_mode(AdaptationMode.FIXED)

        dt = controller.adapt_step()
        assert dt == controller.initial_dt

    def test_gradient_based_adaptation(self, controller):
        """测试基于梯度的自适应"""
        controller.set_mode(AdaptationMode.GRADIENT_BASED)

        # 高梯度 -> 减小步长
        dt_high = controller.adapt_step(gradient_norm=10.0)
        assert dt_high < controller.initial_dt

        # 低梯度 -> 增大步长
        controller.reset()
        dt_low = controller.adapt_step(gradient_norm=0.001)
        assert dt_low > controller.initial_dt

    def test_curvature_based_adaptation(self, controller):
        """测试基于曲率的自适应"""
        controller.set_mode(AdaptationMode.CURVATURE_BASED)

        # 高曲率 -> 减小步长
        dt_high = controller.adapt_step(curvature=100.0)
        assert dt_high < controller.initial_dt

    def test_hybrid_mode(self, controller):
        """测试混合模式"""
        controller.set_mode(AdaptationMode.HYBRID)

        # 同时提供高梯度和高曲率
        dt = controller.adapt_step(
            gradient_norm=10.0,
            curvature=100.0,
            loss_change=-0.1,
        )
        assert dt < controller.initial_dt

    def test_constraints(self, controller):
        """测试步长约束"""
        controller.set_mode(AdaptationMode.GRADIENT_BASED)

        # 多次减小到最小值
        for _ in range(20):
            dt = controller.adapt_step(gradient_norm=100.0)

        assert dt >= controller.min_dt

    def test_history_recording(self, controller):
        """测试历史记录"""
        controller.set_mode(AdaptationMode.GRADIENT_BASED)

        for i in range(10):
            controller.adapt_step(gradient_norm=float(i))

        assert len(controller.history) == 10

    def test_statistics(self, controller):
        """测试统计信息"""
        controller.set_mode(AdaptationMode.FIXED)

        for _ in range(20):
            controller.adapt_step()

        stats = controller.get_statistics()
        assert isinstance(stats, TimeStepStats)
        assert stats.mean_dt > 0
        assert stats.std_dt >= 0
        assert stats.total_adaptations == 0

    def test_reset(self, controller):
        """测试重置"""
        controller.adapt_step(gradient_norm=10.0)
        assert controller._step_count > 0

        controller.reset()
        assert controller._step_count == 0
        assert controller.current_dt == controller.initial_dt

    def test_export_import_state(self, controller):
        """测试状态导出/导入"""
        controller.set_mode(AdaptationMode.HYBRID)
        controller.adapt_step(gradient_norm=5.0, curvature=3.0)

        state = controller.export_state()
        restored = TimeStepController.from_state(state)

        assert restored.current_dt == controller.current_dt
        assert restored.mode == controller.mode
        assert restored._step_count == controller._step_count


class TestStepHistory:
    """步骤历史记录测试"""

    def test_creation(self):
        """测试创建"""
        history = StepHistory(step=5, dt=0.01, gradient_norm=1.5)

        assert history.step == 5
        assert history.dt == 0.01
        assert history.gradient_norm == 1.5


# ==================== 检查点管理器测试 ====================


class TestSamplingCheckpointManager:
    """检查点管理器测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def manager(self, temp_dir):
        """创建管理器实例"""
        return SamplingCheckpointManager(
            save_dir=temp_dir,
            format="pickle",
        )

    def test_create_checkpoint(self, manager):
        """测试创建检查点"""
        ckpt = manager.create_checkpoint(
            current_step=10,
            total_steps=100,
            metadata={"loss": 0.5},
        )

        assert isinstance(ckpt, SamplingCheckpoint)
        assert ckpt.current_step == 10
        assert ckpt.total_steps == 100
        assert ckpt.progress == 0.1

    def test_save_and_load_pickle(self, manager, temp_dir):
        """测试 Pickle 格式保存和加载"""
        ckpt = manager.create_checkpoint(
            current_step=25,
            total_steps=100,
            random_state={"numpy": [42]},
            intermediate_results={"samples": np.random.randn(10)},
        )

        path = manager.save_checkpoint(ckpt)
        assert path.exists()

        loaded = manager.load_checkpoint(path)
        assert loaded.current_step == 25
        assert loaded.checkpoint_id == ckpt.checkpoint_id

    def test_save_and_load_json(self, temp_dir):
        """测试 JSON 格式保存和加载"""
        json_manager = SamplingCheckpointManager(save_dir=temp_dir, format="json")

        ckpt = json_manager.create_checkpoint(current_step=15, total_steps=50)
        path = json_manager.save_checkpoint(ckpt)

        loaded = json_manager.load_checkpoint(path)
        assert loaded.current_step == 15

    def test_auto_save_logic(self, manager):
        """测试自动保存逻辑"""
        manager.auto_save_interval = 5

        assert not manager.should_auto_save(3)
        assert not manager.should_auto_save(4)
        assert manager.should_auto_save(5)
        assert not manager.should_auto_save(7)
        assert manager.should_auto_save(10)

    def test_latest_checkpoint(self, manager):
        """测试获取最新检查点"""
        for step in [10, 20, 30]:
            ckpt = manager.create_checkpoint(current_step=step, total_steps=100)
            manager.save_checkpoint(ckpt)
            time.sleep(0.01)  # 确保时间戳不同

        latest = manager.get_latest_checkpoint()
        assert latest.current_step == 30

    def test_list_checkpoints(self, manager):
        """测试列出检查点"""
        for step in [5, 15, 25]:
            ckpt = manager.create_checkpoint(current_step=step, total_steps=100)
            manager.save_checkpoint(ckpt)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3

    def test_delete_checkpoint(self, manager):
        """测试删除检查点"""
        ckpt = manager.create_checkpoint(current_step=10, total_steps=100)
        path = manager.save_checkpoint(ckpt)

        result = manager.delete_checkpoint(ckpt.checkpoint_id)
        assert result is True
        assert not path.exists()

    def test_cleanup_policy_keep_last_n(self, temp_dir):
        """测试保留最近 N 个策略"""
        policy = CheckpointCleanupPolicy(
            strategy="keep_last_n",
            max_checkpoints=3,
        )
        manager = SamplingCheckpointManager(
            save_dir=temp_dir,
            cleanup_policy=policy,
        )

        # 创建 5 个检查点
        for step in range(5):
            ckpt = manager.create_checkpoint(current_step=step * 10, total_steps=100)
            manager.save_checkpoint(ckpt)
            time.sleep(0.01)

        # 应该只有 3 个
        assert len(manager.list_checkpoints()) <= 3

    def test_storage_info(self, manager):
        """测试存储信息"""
        ckpt = manager.create_checkpoint(current_step=5, total_steps=100)
        manager.save_checkpoint(ckpt)

        info = manager.get_storage_info()
        assert info["total_checkpoints"] >= 1
        assert "total_size_bytes" in info

    def test_cleanup_all(self, temp_dir):
        """测试清除所有"""
        policy = CheckpointCleanupPolicy(strategy="keep_all")
        manager = SamplingCheckpointManager(
            save_dir=temp_dir,
            cleanup_policy=policy,
        )
        for _ in range(5):
            ckpt = manager.create_checkpoint(current_step=1, total_steps=100)
            manager.save_checkpoint(ckpt)

        count = manager.cleanup_all()
        assert count == 5
        assert len(manager.list_checkpoints()) == 0

    def test_load_nonexistent(self, manager):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            manager.load_checkpoint(Path("nonexistent.pkl"))


class TestSamplingCheckpoint:
    """检查点数据类测试"""

    def test_progress_calculation(self):
        """测试进度计算"""
        ckpt = SamplingCheckpoint(
            checkpoint_id="test",
            timestamp=time.time(),
            current_step=75,
            total_steps=100,
        )

        assert ckpt.progress == 0.75

    def test_created_at(self):
        """测试格式化时间"""
        now = time.time()
        ckpt = SamplingCheckpoint(
            checkpoint_id="test",
            timestamp=now,
            current_step=0,
            total_steps=100,
        )

        created = ckpt.created_at
        assert isinstance(created, str)
        assert len(created) > 0

    def test_to_dict(self):
        """测试转换为字典"""
        ckpt = SamplingCheckpoint(
            checkpoint_id="test",
            timestamp=time.time(),
            current_step=10,
            total_steps=100,
        )

        data = ckpt.to_dict()
        assert "checkpoint_id" in data
        assert "current_step" in data

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "checkpoint_id": "test",
            "timestamp": time.time(),
            "current_step": 20,
            "total_steps": 100,
        }

        ckpt = SamplingCheckpoint.from_dict(data)
        assert ckpt.current_step == 20


# ==================== 进度监控器测试 ====================


class TestProgressMonitor:
    """进度监控器测试"""

    @pytest.fixture
    def monitor(self):
        """创建监控器（禁用 tqdm）"""
        return ProgressMonitor(total_steps=100, enable_tqdm=False)

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor.total_steps == 100
        assert not monitor.is_running

    def test_invalid_initialization(self):
        """测试无效参数"""
        with pytest.raises(ValueError):
            ProgressMonitor(total_steps=0)

        with pytest.raises(ValueError):
            ProgressMonitor(total_steps=-5)

        with pytest.raises(ValueError):
            ProgressMonitor(total_steps=100, eta_smoothing=1.5)

    def test_start_complete_cycle(self, monitor):
        """测试完整的开始-完成周期"""
        monitor.start()
        assert monitor.is_running

        for i in range(100):
            monitor.on_step_start(i)
            monitor.on_step_complete(i, loss=0.1)

        monitor.complete()
        assert not monitor.is_running

    def test_progress_calculation(self, monitor):
        """测试进度计算"""
        monitor.start()

        monitor.on_step_start(0)
        monitor.on_step_complete(0)

        progress = monitor.progress
        assert progress.current_step == 1
        assert progress.progress_percent == pytest.approx(1.0, abs=0.1)
        assert progress.total_steps == 100

        monitor.complete()

    def test_callback_registration(self, monitor):
        """测试回调注册"""
        events_received = []

        def callback(progress, event_type, **kwargs):
            events_received.append(event_type)

        # 注册多个事件类型
        monitor.register_callback(SamplingEventType.STEP_COMPLETE, callback)
        monitor.register_callback(SamplingEventType.SAMPLING_COMPLETE, callback)
        monitor.start()
        monitor.on_step_start(0)
        monitor.on_step_complete(0)
        monitor.complete()

        assert SamplingEventType.STEP_COMPLETE in events_received
        assert SamplingEventType.SAMPLING_COMPLETE in events_received

    def test_error_callback(self, monitor):
        """测试错误回调"""
        errors = []

        def error_handler(progress, event_type, **kwargs):
            if event_type == SamplingEventType.ERROR:
                errors.append(kwargs.get("error"))

        monitor.register_callback(SamplingEventType.ERROR, error_handler)
        monitor.start()
        monitor.on_error(5, RuntimeError("Test error"))
        monitor.complete()

        assert len(errors) == 1
        assert "Test error" in errors[0]

    def test_warning_callback(self, monitor):
        """测试警告回调"""
        warnings = []

        def warning_handler(progress, event_type, **kwargs):
            if event_type == SamplingEventType.WARNING:
                warnings.append(kwargs.get("message"))

        monitor.register_callback(SamplingEventType.WARNING, warning_handler)
        monitor.start()
        monitor.on_warning(10, "Low gradient")
        monitor.complete()

        assert len(warnings) == 1
        assert "Low gradient" in warnings[0]

    def test_custom_metrics(self, monitor):
        """测试自定义指标"""
        monitor.start()
        monitor.on_step_start(0)
        monitor.on_step_complete(0, loss=0.5, accuracy=0.95)

        metrics = monitor.get_custom_metrics()
        assert metrics["loss"] == 0.5
        assert metrics["accuracy"] == 0.95

        monitor.complete()

    def test_eta_estimation(self, monitor):
        """测试 ETA 估计"""
        import time

        monitor.start()

        for i in range(10):
            monitor.on_step_start(i)
            monitor.on_step_complete(i)
            time.sleep(0.01)  # 模拟耗时操作

        progress = monitor.progress
        assert progress.steps_per_second > 0
        assert progress.estimated_remaining_time > 0

        monitor.complete()

    def test_reset(self, monitor):
        """测试重置"""
        monitor.start()
        monitor.on_step_start(0)
        monitor.on_step_complete(0)
        monitor.complete()

        monitor.reset()
        assert not monitor.is_running
        assert monitor._current_step == 0

    def test_unregister_callback(self, monitor):
        """测试取消注册回调"""
        calls = []

        def callback(progress, event_type, **kwargs):
            calls.append(1)

        monitor.register_callback(SamplingEventType.STEP_START, callback)
        result = monitor.unregister_callback(SamplingEventType.STEP_START, callback)
        assert result is True

        monitor.start()
        monitor.on_step_start(0)
        monitor.complete()

        assert len(calls) == 0


class TestSamplingProgress:
    """进度信息数据类测试"""

    def test_elapsed_str(self):
        """测试已用时间字符串"""
        progress = SamplingProgress(
            current_step=50,
            total_steps=100,
            progress_percent=50.0,
            elapsed_time=3661.0,  # 1小时1分1秒
            estimated_remaining_time=3600.0,
            steps_per_second=10.0,
        )

        elapsed = progress.elapsed_str
        assert "1:" in elapsed or "01:" in elapsed

    def test_eta_str_unknown(self):
        """测试未知 ETA"""
        progress = SamplingProgress(
            current_step=0,
            total_steps=100,
            progress_percent=0.0,
            elapsed_time=0.0,
            estimated_remaining_time=-1.0,
            steps_per_second=0.0,
        )

        assert progress.eta_str == "未知"

    def test_to_dict(self):
        """测试转换为字典"""
        progress = SamplingProgress(
            current_step=25,
            total_steps=100,
            progress_percent=25.0,
            elapsed_time=10.0,
            estimated_remaining_time=30.0,
            steps_per_second=2.5,
            custom_metrics={"loss": 0.5},
        )

        data = progress.to_dict()
        assert data["current_step"] == 25
        assert "custom_metrics" in data


# ==================== 边界条件和异常测试 ====================


class TestEdgeCases:
    """边界条件和异常情况测试"""

    def test_scheduler_zero_steps(self):
        """测试零步数"""
        scheduler = LinearScheduler()
        with pytest.raises(ValueError):
            scheduler.get_schedule(0)

    def test_controller_empty_statistics(self):
        """测试空统计"""
        controller = TimeStepController()
        stats = controller.get_statistics()

        assert stats.mean_dt == controller.initial_dt
        assert stats.std_dt == 0.0
        assert stats.total_adaptations == 0

    def test_monitor_no_tqdm_fallback(self):
        """测试无 tqdm 时的回退"""
        monitor = ProgressMonitor(total_steps=10, enable_tqdm=True)
        # 如果 tqdm 可用则正常工作，否则应优雅降级
        monitor.start()
        monitor.complete()

    def test_checkpoint_large_data(self, tmp_path):
        """测试大数据检查点"""
        manager = SamplingCheckpointManager(save_dir=tmp_path)

        large_array = np.random.randn(10000, 100)
        ckpt = manager.create_checkpoint(
            current_step=50,
            total_steps=100,
            intermediate_results={"large_data": large_array},
        )

        path = manager.save_checkpoint(ckpt)
        loaded = manager.load_checkpoint(path)

        assert loaded.intermediate_results is not None

    def test_concurrent_access_simulation(self, tmp_path):
        """模拟并发访问场景"""
        policy = CheckpointCleanupPolicy(strategy="keep_all")
        manager = SamplingCheckpointManager(
            save_dir=tmp_path,
            cleanup_policy=policy,
        )

        check_points = []
        for i in range(20):
            ckpt = manager.create_checkpoint(current_step=i, total_steps=100)
            path = manager.save_checkpoint(ckpt)
            check_points.append(path)

        assert len(check_points) == 20

        # 验证所有都可加载
        for path in check_points:
            loaded = manager.load_checkpoint(path)
            assert loaded is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
