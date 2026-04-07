"""数据集可视化和统计分析工具模块

提供数据集探索、可视化、质量分析和验证功能：
- DatasetVisualizer: 数据集样本网格、类别分布等可视化
- DatasetAnalyzer: 数据集质量分析、异常检测
- 工具函数：下载、验证、信息生成
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
from dataclasses import dataclass, field
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DatasetStatistics:
    """数据集统计信息
    
    存储计算得到的各种统计量。
    
    Attributes:
        num_samples: 总样本数
        num_classes: 类别数
        class_distribution: 各类别样本数量 {label: count}
        image_sizes: 图像尺寸列表 [(h, w), ...]
        mean_rgb: RGB通道均值 (R_mean, G_mean, B_mean)
        std_rgb: RGB通道标准差 (R_std, G_std, B_std)
        brightness_stats: 亮度统计 {'mean': float, 'std': float, 'min': float, 'max': float}
        contrast_stats: 对比度统计 {'mean': float, 'std': float}
    """
    num_samples: int = 0
    num_classes: int = 0
    class_distribution: Dict[int, int] = field(default_factory=dict)
    image_sizes: List[Tuple[int, int]] = field(default_factory=list)
    mean_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std_rgb: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    brightness_stats: Dict[str, float] = field(default_factory=dict)
    contrast_stats: Dict[str, float] = field(default_factory=dict)


class DatasetVisualizer:
    """数据集可视化工具
    
    提供多种数据集可视化方法，包括样本网格、类别分布图、统计对比等。
    
    Attributes:
        figsize: 默认图形大小
        dpi: 图形分辨率
        
    Example:
        >>> visualizer = DatasetVisualizer(figsize=(12, 8))
        >>> visualizer.plot_sample_grid(dataset, n_rows=4, n_cols=8, save_path='samples.png')
        >>> visualizer.plot_class_distribution(dataset, save_path='distribution.png')
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        style: str = 'seaborn-v0_8-whitegrid',
    ) -> None:
        """初始化可视化器
        
        Args:
            figsize: 默认图形尺寸（宽，高）
            dpi: 图形分辨率
            style: matplotlib样式名称
        """
        self.figsize = figsize
        self.dpi = dpi
        try:
            plt.style.use(style)
        except Exception:
            plt.style.use('default')
            logger.warning(f"无法加载样式{style}，使用默认样式")
        
        logger.info(f"初始化DatasetVisualizer，图形尺寸: {figsize}, DPI: {dpi}")
    
    def plot_sample_grid(
        self,
        dataset: Dataset,
        n_rows: int = 4,
        n_cols: int = 8,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Dataset Samples",
        show_labels: bool = True,
        class_names: Optional[List[str]] = None,
    ) -> plt.Figure:
        """绘制数据集样本网格图
        
        从数据集中随机选择样本并绘制成网格。
        
        Args:
            dataset: PyTorch Dataset对象
            n_rows: 行数
            n_cols: 列数
            save_path: 保存路径，如果为None则不保存
            title: 图表标题
            show_labels: 是否显示标签
            class_names: 类别名称列表
            
        Returns:
            matplotlib Figure对象
            
        Raises:
            ValueError: 如果请求的样本数超过数据集大小
        """
        n_samples = min(n_rows * n_cols, len(dataset))
        if n_samples == 0:
            raise ValueError("数据集为空")
        
        indices = np.random.choice(len(dataset), size=n_samples, replace=False)
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(self.figsize[0], self.figsize[1] * n_rows / 4),
            dpi=self.dpi,
        )
        axes = axes.flatten() if n_samples > 1 else [axes]
        
        for idx, ax in zip(indices, axes):
            try:
                sample = dataset[idx]
                image, label, _ = sample
                
                if isinstance(image, torch.Tensor):
                    img_np = image.cpu().numpy()
                    if img_np.shape[0] in [1, 3]:
                        img_np = np.transpose(img_np, (1, 2, 0))
                        if img_np.shape[-1] == 1:
                            img_np = img_np.squeeze(-1)
                    
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    if img_np.shape[-1] == 3:
                        img_np = img_np * std + mean
                        img_np = np.clip(img_np, 0, 1)
                
                elif isinstance(image, Image.Image):
                    img_np = np.array(image)
                    if img_np.max() > 1:
                        img_np = img_np / 255.0
                else:
                    img_np = np.array(image)
                
                ax.imshow(img_np)
                ax.axis('off')
                
                if show_labels:
                    label_text = (
                        class_names[label] if class_names and label < len(class_names)
                        else str(label)
                    )
                    ax.set_title(label_text, fontsize=8)
                    
            except Exception as e:
                logger.warning(f"绘制样本{idx}失败: {e}")
                ax.text(0.5, 0.5, f'Error\n{idx}', ha='center', va='center')
                ax.axis('off')
        
        for idx in range(n_samples, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"样本网格已保存到: {save_path}")
        
        return fig
    
    def plot_class_distribution(
        self,
        dataset: Dataset,
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Class Distribution",
        class_names: Optional[List[str]] = None,
        color_map: str = 'viridis',
        show_values: bool = True,
    ) -> plt.Figure:
        """绘制类别分布直方图
        
        可视化数据集中各类别的样本数量分布。
        
        Args:
            dataset: 数据集对象
            save_path: 保存路径
            title: 图表标题
            class_names: 类别名称
            color_map: 颜色映射
            show_values: 是否在柱子上显示数值
            
        Returns:
            matplotlib Figure对象
        """
        labels = []
        for i in range(min(len(dataset), 10000)):
            try:
                _, label, _ = dataset[i]
                labels.append(label)
            except Exception as e:
                logger.warning(f"获取样本{i}标签失败: {e}")
        
        unique_labels = sorted(set(labels))
        counts = [labels.count(l) for l in unique_labels]
        
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        x_pos = range(len(unique_labels))
        bars = ax.bar(x_pos, counts, color=plt.cm.get_cmap(color_map)(np.linspace(0, 1, len(unique_labels))))
        
        ax.set_xlabel('Class Label', fontsize=11)
        ax.set_ylabel('Number of Samples', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        
        if class_names and len(class_names) >= len(unique_labels):
            ax.set_xticks(x_pos)
            ax.set_xticklabels([class_names[l] for l in unique_labels], rotation=45, ha='right')
        else:
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(l) for l in unique_labels])
        
        if show_values:
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.annotate(f'{count}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"类别分布图已保存到: {save_path}")
        
        return fig
    
    def compare_datasets(
        self,
        datasets: Dict[str, Dataset],
        save_path: Optional[Union[str, Path]] = None,
        title: str = "Dataset Comparison",
    ) -> plt.Figure:
        """对比多个数据集的分布
        
        在同一图表中对比训练/验证/测试集的类别分布。
        
        Args:
            datasets: 字典 {name: dataset}
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            matplotlib Figure对象
        """
        all_labels = {}
        for name, ds in datasets.items():
            labels = []
            for i in range(min(len(ds), 5000)):
                try:
                    _, label, _ = ds[i]
                    labels.append(label)
                except Exception:
                    pass
            all_labels[name] = labels
        
        all_unique = set()
        for labels in all_labels.values():
            all_unique.update(labels)
        all_unique = sorted(all_unique)
        
        n_datasets = len(datasets)
        fig, axes = plt.subplots(1, n_datasets, figsize=(self.figsize[0] * n_datasets / 2, self.figsize[1]), dpi=self.dpi)
        if n_datasets == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(np.linspace(0, 1, max(len(all_unique), 10)))
        
        for ax, (name, labels) in zip(axes, all_labels.items()):
            counts = [labels.count(l) for l in all_unique]
            ax.bar(range(len(all_unique)), counts, color=colors[:len(all_unique)], alpha=0.7)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(all_unique)))
            ax.set_xticklabels([str(l) for l in all_unique], rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
        
        fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"数据集对比图已保存到: {save_path}")
        
        return fig


class DatasetAnalyzer:
    """数据集分析器
    
    分析数据集质量，包括缺失值检查、异常值检测、图像属性分析等。
    
    Attributes:
        dataset: 待分析的数据集
        statistics: 计算得到的统计信息
        
    Example:
        >>> analyzer = DatasetAnalyzer(dataset)
        >>> stats = analyzer.compute_statistics()
        >>> report = analyzer.generate_report()
    """
    
    def __init__(self, dataset: Dataset) -> None:
        """初始化分析器
        
        Args:
            dataset: 要分析的PyTorch Dataset
        """
        self.dataset = dataset
        self.statistics: Optional[DatasetStatistics] = None
        self._issues: List[Dict[str, Any]] = []
        logger.info("初始化DatasetAnalyzer")
    
    def compute_statistics(self, max_samples: int = 1000) -> DatasetStatistics:
        """计算数据集统计量
        
        分析图像尺寸、颜色分布、亮度对比度等。
        
        Args:
            max_samples: 最大采样数量（用于大数据集）
            
        Returns:
            DatasetStatistics对象包含所有统计量
        """
        stats = DatasetStatistics()
        stats.num_samples = len(self.dataset)
        
        pixel_values_r = []
        pixel_values_g = []
        pixel_values_b = []
        brightness_list = []
        contrast_list = []
        
        sample_indices = list(range(min(max_samples, len(self.dataset))))
        
        for idx in sample_indices:
            try:
                sample = self.dataset[idx]
                image, label, _ = sample
                
                if hasattr(self.dataset, 'classes'):
                    stats.num_classes = len(self.dataset.classes) if isinstance(self.dataset.classes, list) else getattr(self.dataset, 'num_classes', 0)
                
                stats.class_distribution[label] = stats.class_distribution.get(label, 0) + 1
                
                if isinstance(image, torch.Tensor):
                    img_array = image.cpu().numpy()
                    if img_array.shape[0] in [1, 3]:
                        img_array = np.transpose(img_array, (1, 2, 0))
                elif isinstance(image, Image.Image):
                    img_array = np.array(image).astype(np.float32) / 255.0
                else:
                    continue
                
                h, w = img_array.shape[:2]
                stats.image_sizes.append((h, w))
                
                if img_array.ndim == 3 and img_array.shape[2] >= 3:
                    pixel_values_r.append(img_array[:, :, 0].flatten())
                    pixel_values_g.append(img_array[:, :, 1].flatten())
                    pixel_values_b.append(img_array[:, :, 2].flatten())
                
                gray = 0.299 * img_array[..., 0] + 0.587 * img_array[..., 1] + 0.114 * img_array[..., 2] if img_array.ndim == 3 else img_array
                brightness = gray.mean()
                contrast = gray.std()
                brightness_list.append(brightness)
                contrast_list.append(contrast)
                
            except Exception as e:
                issue = {
                    'index': idx,
                    'type': 'load_error',
                    'message': str(e),
                }
                self._issues.append(issue)
        
        if pixel_values_r:
            all_r = np.concatenate(pixel_values_r)
            all_g = np.concatenate(pixel_values_g)
            all_b = np.concatenate(pixel_values_b)
            
            stats.mean_rgb = (float(all_r.mean()), float(all_g.mean()), float(all_b.mean()))
            stats.std_rgb = (float(all_r.std()), float(all_g.std()), float(all_b.std()))
        
        if brightness_list:
            arr_brightness = np.array(brightness_list)
            stats.brightness_stats = {
                'mean': float(arr_brightness.mean()),
                'std': float(arr_brightness.std()),
                'min': float(arr_brightness.min()),
                'max': float(arr_brightness.max()),
            }
        
        if contrast_list:
            arr_contrast = np.array(contrast_list)
            stats.contrast_stats = {
                'mean': float(arr_contrast.mean()),
                'std': float(arr_contrast.std()),
            }
        
        self.statistics = stats
        logger.info(f"完成统计分析: {stats.num_samples}样本, 发现{len(self._issues)}个问题")
        return stats
    
    def check_integrity(self) -> Dict[str, Any]:
        """检查数据集完整性
        
        验证所有样本是否可正常加载。
        
        Returns:
            包含完整性信息的字典：
            {
                'total': int,          # 总样本数
                'valid': int,          # 有效样本数
                'corrupted': int,      # 损坏样本数
                'error_rate': float,   # 错误率
                'errors': List[dict]   # 错误详情
            }
        """
        total = len(self.dataset)
        valid_count = 0
        corrupted_count = 0
        errors = []
        
        for idx in range(total):
            try:
                sample = self.dataset[idx]
                if sample is not None:
                    valid_count += 1
            except Exception as e:
                corrupted_count += 1
                errors.append({
                    'index': idx,
                    'error_type': type(e).__name__,
                    'message': str(e)[:200],
                })
                if len(errors) <= 5:
                    logger.debug(f"样本{idx}损坏: {e}")
        
        result = {
            'total': total,
            'valid': valid_count,
            'corrupted': corrupted_count,
            'error_rate': corrupted_count / total if total > 0 else 0.0,
            'errors': errors[:20],
        }
        
        logger.info(
            f"完整性检查完成: {result['valid']}/{result['total']} 有效, "
            f"{result['corrupted']} 损坏 ({result['error_rate']:.2%})"
        )
        return result
    
    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """生成数据分析报告
        
        创建包含所有统计和分析结果的报告。
        
        Args:
            output_path: 报告保存路径（JSON格式）
            
        Returns:
            报告字典
        """
        if self.statistics is None:
            self.compute_statistics()
        
        integrity = self.check_integrity()
        
        report = {
            'dataset_info': {
                'name': self.dataset.__class__.__name__,
                'total_samples': len(self.dataset),
                'num_classes': self.statistics.num_classes,
            },
            'statistics': {
                'num_samples': self.statistics.num_samples,
                'num_classes': self.statistics.num_classes,
                'class_distribution': {
                    str(k): v for k, v in self.statistics.class_distribution.items()
                },
                'image_size_range': (
                    (min(h for h, w in self.statistics.image_sizes), min(w for h, w in self.statistics.image_sizes)) if self.statistics.image_sizes else (0, 0),
                    (max(h for h, w in self.statistics.image_sizes), max(w for h, w in self.statistics.image_sizes)) if self.statistics.image_sizes else (0, 0),
                ),
                'mean_rgb': self.statistics.mean_rgb,
                'std_rgb': self.statistics.std_rgb,
                'brightness': self.statistics.brightness_stats,
                'contrast': self.statistics.contrast_stats,
            },
            'integrity': integrity,
            'issues': self._issues[:10],
        }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"分析报告已保存到: {output_path}")
        
        return report


def download_cifar10(data_root: Union[str, Path]) -> bool:
    """自动下载CIFAR-10数据集
    
    如果数据集不存在则自动下载并验证。
    
    Args:
        data_root: 数据存储根目录
        
    Returns:
        bool: 是否成功下载/已存在
        
    Example:
        >>> success = download_cifar10('./data/cifar10')
        >>> print(f"数据集就绪: {success}")
    """
    from torchvision.datasets import CIFAR10
    
    data_path = Path(data_root)
    data_path.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"检查/下载CIFAR-10数据集到: {data_path}")
        
        train_set = CIFAR10(root=str(data_path), train=True, download=True)
        test_set = CIFAR10(root=str(data_path), train=False, download=True)
        
        logger.info(
            f"CIFAR-10准备完成: "
            f"训练集{len(train_set)}张, 测试集{len(test_set)}张"
        )
        return True
        
    except Exception as e:
        logger.error(f"下载CIFAR-10失败: {e}")
        return False


def verify_dataset_integrity(dataset: Dataset) -> Dict[str, Any]:
    """快速验证数据集完整性
    
    简化版的数据集验证函数。
    
    Args:
        dataset: 要验证的数据集
        
    Returns:
        验证结果字典
    """
    analyzer = DatasetAnalyzer(dataset)
    return analyzer.check_integrity()


def create_data_info_file(
    dataset: Dataset,
    output_path: Union[str, Path],
    include_visualization: bool = False,
) -> Dict[str, Any]:
    """生成数据集信息文件
    
    创建包含数据集详细信息的JSON文件和可选的可视化图表。
    
    Args:
        dataset: 数据集对象
        output_path: 输出文件路径
        include_visualization: 是否同时生成可视化图表
        
    Returns:
        信息字典
    """
    analyzer = DatasetAnalyzer(dataset)
    info = analyzer.generate_report(output_path=output_path)
    
    if include_visualization:
        visualizer = DatasetVisualizer()
        
        output_dir = Path(output_path).parent
        class_names = getattr(dataset, 'class_names', None) or getattr(dataset, 'classes', None)
        
        try:
            visualizer.plot_sample_grid(
                dataset,
                save_path=output_dir / 'sample_grid.png',
                class_names=class_names,
            )
        except Exception as e:
            logger.warning(f"生成样本网格失败: {e}")
        
        try:
            visualizer.plot_class_distribution(
                dataset,
                save_path=output_dir / 'class_distribution.png',
                class_names=class_names,
            )
        except Exception as e:
            logger.warning(f"生成类别分布图失败: {e}")
    
    logger.info(f"数据集信息文件已生成: {output_path}")
    return info


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("数据集可视化和分析工具演示")
    print("=" * 60)
    
    print("\n可用功能:")
    print("-" * 40)
    print("1. DatasetVisualizer:")
    print("   - plot_sample_grid(): 样本网格可视化")
    print("   - plot_class_distribution(): 类别分布图")
    print("   - compare_datasets(): 多数据集对比")
    print("\n2. DatasetAnalyzer:")
    print("   - compute_statistics(): 计算统计量")
    print("   - check_integrity(): 完整性检查")
    print("   - generate_report(): 生成分析报告")
    print("\n3. 工具函数:")
    print("   - download_cifar10(): 下载数据集")
    print("   - verify_dataset_integrity(): 快速验证")
    print("   - create_data_info_file(): 生成信息文件")
    
    print("\n✓ 模块导入成功!")
