"""GMM优化器可视化监控接口"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import os
import json
from datetime import datetime
import torch

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib未安装，可视化功能将不可用")


@dataclass
class MonitoringData:
    """监控数据存储类

    存储训练过程中的所有监控信息。

    Attributes:
        epochs: 迭代次数列表
        losses: 损失值列表
        mean_losses: 均值分量损失列表
        variance_losses: 方差分量损失列表
        skewness_losses: 偏度分量损失列表
        gradient_norms: 梯度范数列表（按参数名）
        param_history: 参数历史记录（按参数名）
        learning_rates: 学习率历史
        timestamps: 时间戳列表
        metadata: 额外元数据

    Example:
        >>> monitoring_data = MonitoringData()
        >>> monitoring_data.record(epoch=1, loss=0.5, params={'means': ...})
    """

    epochs: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)
    mean_losses: List[float] = field(default_factory=list)
    variance_losses: List[float] = field(default_factory=list)
    skewness_losses: List[float] = field(default_factory=list)
    gradient_norms: Dict[str, List[float]] = field(default_factory=dict)
    param_history: Dict[str, List[torch.Tensor]] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record(
        self,
        epoch: int,
        loss: float,
        params: Optional[Dict[str, torch.Tensor]] = None,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        learning_rate: float = 0.01,
        loss_components: Optional[Dict[str, float]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """记录单次迭代的数据

        Args:
            epoch: 当前迭代次数
            loss: 总损失值
            params: 当前参数字典
            gradients: 当前梯度字典
            learning_rate: 当前学习率
            loss_components: 各分量的损失字典
            timestamp: 时间戳（秒）
        """
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.learning_rates.append(learning_rate)

        if timestamp is not None:
            self.timestamps.append(timestamp)

        if loss_components:
            if 'mean' in loss_components:
                self.mean_losses.append(loss_components['mean'])
            if 'variance' in loss_components:
                self.variance_losses.append(loss_components['variance'])
            if 'skewness' in loss_components:
                self.skewness_losses.append(loss_components['skewness'])

        if params is not None:
            for name, param in params.items():
                if name not in self.param_history:
                    self.param_history[name] = []
                self.param_history[name].append(param.detach().cpu())

        if gradients is not None:
            for name, grad in gradients.items():
                if name not in self.gradient_norms:
                    self.gradient_norms[name] = []
                self.gradient_norms[name].append(grad.norm().item())

    @property
    def n_records(self) -> int:
        """获取记录数量"""
        return len(self.epochs)

    @property
    def best_loss(self) -> float:
        """获取最佳损失值"""
        if not self.losses:
            return float('inf')
        return min(self.losses)

    @property
    def best_epoch(self) -> int:
        """获取最佳损失的迭代次数"""
        if not self.losses:
            return 0
        min_idx = self.losses.index(self.best_loss)
        return self.epochs[min_idx]

    def get_param_trajectory(
        self,
        param_name: str,
        component_idx: Optional[Tuple[int, ...]] = None,
    ) -> List[float]:
        """获取特定参数的变化轨迹

        Args:
            param_name: 参数名称
            component_idx: 要提取的分量索引，如 (0,) 表示第一个均值向量

        Returns:
            参数值列表
        """
        if param_name not in self.param_history:
            return []

        trajectory = []
        for param in self.param_history[param_name]:
            if component_idx is not None:
                value = param[component_idx].item()
            else:
                value = param.norm().item()
            trajectory.append(value)

        return trajectory

    def to_dict(self) -> Dict[str, Any]:
        """将监控数据转换为可序列化的字典

        Returns:
            字典格式的监控数据
        """
        data = {
            "epochs": self.epochs,
            "losses": self.losses,
            "mean_losses": self.mean_losses,
            "variance_losses": self.variance_losses,
            "skewness_losses": self.skewness_losses,
            "gradient_norms": {k: v for k, v in self.gradient_norms.items()},
            "learning_rates": self.learning_rates,
            "timestamps": self.timestamps,
            "n_records": self.n_records,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "metadata": self.metadata,
        }

        param_summary = {}
        for name, history in self.param_history.items():
            if history:
                first_shape = list(history[0].shape)
                param_summary[name] = {
                    "shape": first_shape,
                    "n_records": len(history),
                    "final_value": history[-1].tolist() if history[-1].numel() < 100 else None,
                }
        data["param_summary"] = param_summary

        return data

    def save_to_json(self, filepath: str) -> None:
        """保存监控数据到JSON文件

        Args:
            filepath: 输出文件路径
        """
        data = self.to_dict()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"监控数据已保存到: {filepath}")

    def generate_report(self) -> str:
        """生成简单的监控数据报告

        Returns:
            格式化的报告字符串
        """
        report_lines = [
            "=" * 60,
            "MONITORING DATA REPORT",
            "=" * 60,
            "",
            f"Total Records: {self.n_records}",
            f"Best Loss: {self.best_loss:.6f}" if self.losses else "No data",
            f"Best Epoch: {self.best_epoch}" if self.epochs else "N/A",
            "",
        ]

        if self.losses:
            report_lines.extend([
                "-" * 40,
                "LOSS STATISTICS:",
                f"  Initial Loss: {self.losses[0]:.6f}",
                f"  Final Loss:   {self.losses[-1]:.6f}" if self.losses else "",
                "",
            ])

        report_lines.extend([
            "=" * 60,
        ])

        return "\n".join(report_lines)


class TrainingMonitor:
    """训练过程监控器

    记录和可视化GMM优化过程的各项指标。

    功能:
        - 记录每轮的损失值、参数变化、梯度范数
        - 绘制训练曲线 (loss vs iteration)
        - 参数轨迹可视化
        - 日志记录到文件
        - TensorBoard集成（可选）

    Attributes:
        data: MonitoringData实例，存储所有监控数据
        log_dir: 日志目录
        enable_plotting: 是否启用绑图

    Example:
        >>> monitor = TrainingMonitor(log_dir="./logs")
        >>> optimizer.add_callback("on_epoch_end", monitor.callback)
        >>> result = optimizer.optimize(target_moments, initial_params)
        >>> monitor.plot_training_curves()
        >>> monitor.plot_parameter_trajectories()
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        enable_plotting: bool = True,
        save_interval: int = 10,
        plot_dpi: int = 100,
    ):
        """初始化训练监控器

        Args:
            log_dir: 日志保存目录
            enable_plotting: 是否启用绑图功能
            save_interval: 数据保存间隔（迭代次数）
            plot_dpi: 图像DPI
        """
        self.data = MonitoringData()
        self.log_dir = log_dir
        self.enable_plotting = enable_plotting and MATPLOTLIB_AVAILABLE
        self.save_interval = save_interval
        self.plot_dpi = plot_dpi
        self._start_time = None
        self._writer = None

        os.makedirs(log_dir, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard日志目录: {log_dir}")
        except ImportError:
            logger.info("TensorBoard不可用，使用文件日志")

    def callback(self, callback_data) -> None:
        """回调函数：在每次迭代结束时调用

        可以直接作为optimizer的回调函数使用。

        Args:
            callback_data: EpochCallbackData对象
        """
        epoch = callback_data.epoch
        loss = callback_data.loss
        params = callback_data.params
        gradients = callback_data.gradients
        lr = callback_data.learning_rate
        elapsed = callback_data.elapsed_time

        if self._start_time is None:
            self._start_time = elapsed

        timestamp = elapsed if elapsed > 0 else None

        self.data.record(
            epoch=epoch,
            loss=loss,
            params=params,
            gradients=gradients,
            learning_rate=lr,
            timestamp=timestamp,
        )

        if self._writer is not None:
            self._log_to_tensorboard(epoch, loss, params, gradients, lr)

        if epoch % self.save_interval == 0:
            self._save_log_entry(epoch, loss, lr, elapsed)

    def _log_to_tensorboard(
        self,
        epoch: int,
        loss: float,
        params: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]],
        lr: float,
    ) -> None:
        """记录数据到TensorBoard

        Args:
            epoch: 迭代次数
            loss: 损失值
            params: 参数
            gradients: 梯度
            lr: 学习率
        """
        self._writer.add_scalar('Loss/total', loss, epoch)
        self._writer.add_scalar('Training/learning_rate', lr, epoch)

        if gradients:
            for name, grad in gradients.items():
                self._writer.add_scalar(f'Gradients/{name}_norm', grad.norm().item(), epoch)

        if params and 'means' in params:
            means = params['means']
            n_components = means.shape[0]
            for k in range(min(n_components, 10)):
                mean_norm = means[k].norm().item()
                self._writer.add_scalar(f'Parameters/mean_{k}_norm', mean_norm, epoch)

        self._writer.flush()

    def _save_log_entry(
        self,
        epoch: int,
        loss: float,
        lr: float,
        elapsed: float,
    ) -> None:
        """保存日志条目到文件

        Args:
            epoch: 迭代次数
            loss: 损失值
            lr: 学习率
            elapsed: 已用时间
        """
        log_file = os.path.join(self.log_dir, "training_log.txt")

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"Epoch {epoch}: loss={loss:.6f}, lr={lr:.2e}, "
                f"time={elapsed:.3f}s\n"
            )

    def plot_training_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 8),
    ) -> None:
        """绘制训练曲线

        绘制损失随迭代次数变化的曲线。

        Args:
            save_path: 保存图像的路径（如果为None则不保存）
            show: 是否显示图像
            figsize: 图像大小
        """
        if not self.enable_plotting or not self.data.losses:
            logger.warning("无法绘制训练曲线：数据为空或matplotlib不可用")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('GMM Optimization Training Curves', fontsize=14)

        ax1 = axes[0, 0]
        ax1.plot(self.data.epochs, self.data.losses, 'b-', linewidth=1.5, label='Total Loss')
        ax1.axhline(y=self.data.best_loss, color='r', linestyle='--', alpha=0.7,
                   label=f'Best Loss ({self.data.best_loss:.6f})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss vs Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        if self.data.mean_losses:
            ax2.plot(self.data.epochs[:len(self.data.mean_losses)], 
                    self.data.mean_losses, 'g-', label='Mean Loss')
        if self.data.variance_losses:
            ax2.plot(self.data.epochs[:len(self.data.variance_losses)], 
                    self.data.variance_losses, 'r-', label='Variance Loss')
        if self.data.skewness_losses:
            ax2.plot(self.data.epochs[:len(self.data.skewness_losses)], 
                    self.data.skewness_losses, 'm-', label='Skewness Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Component')
        ax2.set_title('Loss Components')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if self.data.learning_rates:
            ax3.plot(self.data.epochs[:len(self.data.learning_rates)], 
                    self.data.learning_rates, 'purple', label='Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')

        ax4 = axes[1, 1]
        for name, norms in self.data.gradient_norms.items():
            if len(norms) == len(self.data.epochs):
                ax4.plot(self.data.epochs, norms, label=name, linewidth=1)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Norms')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi, bbox_inches='tight')
            logger.info(f"训练曲线已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_parameter_trajectories(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (15, 10),
    ) -> None:
        """绘制参数轨迹

        可视化各参数随优化的变化情况。

        Args:
            save_path: 保存图像的路径
            show: 是否显示图像
            figsize: 图像大小
        """
        if not self.enable_plotting or not self.data.param_history:
            logger.warning("无法绘制参数轨迹：数据为空或matplotlib不可用")
            return

        n_plots = min(len(self.data.param_history), 6)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Parameter Trajectories During Optimization', fontsize=14)
        axes = axes.flatten()

        plot_idx = 0
        for param_name, history in self.data.param_history.items():
            if plot_idx >= n_plots:
                break

            ax = axes[plot_idx]

            if not history:
                continue

            first_param = history[0]

            if first_param.dim() == 1:
                n_components = first_param.shape[0]
                for i in range(min(n_components, 5)):
                    trajectory = [p[i].item() for p in history]
                    ax.plot(self.data.epochs[:len(trajectory)], trajectory, 
                           label=f'{param_name}[{i}]')
            elif first_param.dim() == 2:
                n_comp, n_feat = first_param.shape
                for k in range(min(n_comp, 3)):
                    for j in range(min(n_feat, 3)):
                        trajectory = [p[k, j].item() for p in history]
                        ax.plot(self.data.epochs[:len(trajectory)], trajectory,
                               '--', alpha=0.7, label=f'{param_name}[{k},{j}]')

            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title(param_name)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

        for idx in range(plot_idx, n_plots):
            axes[idx].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=self.plot_dpi, bbox_inches='tight')
            logger.info(f"参数轨迹图已保存到: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def generate_report(self) -> str:
        """生成训练报告

        Returns:
            格式化的训练报告字符串
        """
        report_lines = [
            "=" * 60,
            "GMM OPTIMIZATION TRAINING REPORT",
            "=" * 60,
            "",
            f"Total Epochs: {self.data.n_records}",
            f"Final Loss: {self.data.losses[-1]:.6f}" if self.data.losses else "No data",
            f"Best Loss: {self.data.best_loss:.6f} (Epoch {self.data.best_epoch})",
            "",
        ]

        if self.data.losses:
            report_lines.extend([
                "-" * 40,
                "LOSS STATISTICS:",
                f"  Initial Loss: {self.data.losses[0]:.6f}",
                f"  Final Loss:   {self.data.losses[-1]:.6f}",
                f"  Improvement:  {(self.data.losses[0] - self.data.best_loss):.6f} "
                               f"({((self.data.losses[0] - self.data.best_loss) / max(self.data.losses[0], 1e-10)) * 100:.2f}%)",
                "",
            ])

        if self.data.gradient_norms:
            report_lines.extend([
                "-" * 40,
                "GRADIENT NORMS (final):",
            ])
            for name, norms in self.data.gradient_norms.items():
                if norms:
                    report_lines.append(f"  {name}: {norms[-1]:.6f}")
            report_lines.append("")

        report_lines.extend([
            "=" * 60,
        ])

        return "\n".join(report_lines)

    def print_report(self) -> None:
        """打印训练报告"""
        print(self.generate_report())

    def save_report(self, filepath: str) -> None:
        """保存训练报告到文件

        Args:
            filepath: 输出文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        logger.info(f"训练报告已保存到: {filepath}")

    def close(self) -> None:
        """关闭监控器，释放资源"""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        final_log_path = os.path.join(self.log_dir, "monitoring_data.json")
        self.data.save_to_json(final_log_path)

        logger.info("训练监控器已关闭")

    def __del__(self):
        """析构函数"""
        try:
            self.close()
        except Exception:
            pass


def create_monitor(
    log_dir: str = "./logs",
    enable_tensorboard: bool = True,
    **kwargs,
) -> TrainingMonitor:
    """工厂函数：创建训练监控器

    Args:
        log_dir: 日志目录
        enable_tensorboard: 是否启用TensorBoard
        **kwargs: 传递给TrainingMonitor的其他参数

    Returns:
        TrainingMonitor实例
    """
    monitor = TrainingMonitor(log_dir=log_dir, **kwargs)
    logger.info(f"创建训练监控器: log_dir={log_dir}")
    return monitor
