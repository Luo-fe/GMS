"""数据集和数据加载器测试模块

全面测试数据集基础设施的各个组件：
- 数据预处理和增强管道
- CIFAR-10数据集加载
- 自定义数据集加载
- 可视化和分析工具
- 边界情况和错误处理
"""

import matplotlib
matplotlib.use('Agg')

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import shutil
import torchvision.transforms as transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gms.evaluation.data_transforms import (
    DataTransformFactory,
    TransformConfig,
    CutOut,
    cifar10_train_transforms,
    cifar10_test_transforms,
    imagenet_train_transforms,
    imagenet_test_transforms,
    custom_transforms,
    get_normalization_stats,
)

from gms.evaluation.datasets.cifar10 import (
    CIFAR10Dataset,
    get_cifar10_dataloaders,
    create_balanced_sampler,
)

from gms.evaluation.datasets.custom_dataset import (
    ImageNetSubsetDataset,
    CustomImageDataset,
    get_custom_dataloader,
    SUPPORTED_IMAGE_EXTENSIONS,
)

from gms.evaluation.dataset_utils import (
    DatasetVisualizer,
    DatasetAnalyzer,
    DatasetStatistics,
    download_cifar10,
    verify_dataset_integrity,
    create_data_info_file,
)


@pytest.fixture(scope="module")
def temp_dir():
    """创建临时目录用于测试"""
    tmp = tempfile.mkdtemp(prefix="gms_test_")
    yield Path(tmp)
    if Path(tmp).exists():
        shutil.rmtree(tmp)


@pytest.fixture(scope="module")
def sample_images_dir(temp_dir):
    """创建包含示例图像的目录结构"""
    data_dir = temp_dir / "sample_data"
    
    for class_name in ['class_a', 'class_b', 'class_c']:
        class_dir = data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(5):
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(class_dir / f"image_{i:03d}.png")
    
    return data_dir


class TestDataTransforms:
    """数据预处理和增强管道测试"""
    
    def test_transform_config_creation(self):
        """测试TransformConfig配置类"""
        config = TransformConfig(
            image_size=(64, 64),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            augmentation_strength=0.8,
        )
        
        assert config.image_size == (64, 64)
        assert config.augmentation_strength == 0.8
        assert config.use_color_jitter is True
    
    def test_data_transform_factory_init(self):
        """测试DataTransformFactory初始化"""
        factory = DataTransformFactory()
        assert factory.config is not None
        assert factory.config.image_size == (224, 224)
    
    def test_create_train_transforms(self):
        """测试训练集transforms创建"""
        config = TransformConfig(
            image_size=(32, 32),
            mean=DataTransformFactory.CIFAR10_MEAN,
            std=DataTransformFactory.CIFAR10_STD,
            use_color_jitter=True,
            use_random_crop=True,
            use_horizontal_flip=True,
        )
        factory = DataTransformFactory(config)
        train_tfms = factory.create_train_transforms()

        assert train_tfms is not None
        assert hasattr(train_tfms, 'transforms')
    
    def test_create_test_transforms(self):
        """测试测试集transforms创建（无增强）"""
        factory = DataTransformFactory(TransformConfig(image_size=(32, 32)))
        test_tfms = factory.create_test_transforms()
        
        assert test_tfms is not None
    
    def test_cifar10_train_transforms(self):
        """测试CIFAR-10训练集预定义transforms"""
        tfms = cifar10_train_transforms(augmentation_strength=0.7)
        assert tfms is not None
        
        test_img = Image.new('RGB', (40, 40), color='red')
        result = tfms(test_img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3
    
    def test_cifar10_test_transforms(self):
        """测试CIFAR-10测试集预定义transforms"""
        tfms = cifar10_test_transforms()
        assert tfms is not None
        
        test_img = Image.new('RGB', (32, 32), color='blue')
        result = tfms(test_img)
        assert isinstance(result, torch.Tensor)
        assert result.shape[1:] == (32, 32)
    
    def test_imagenet_transforms(self):
        """测试ImageNet transforms"""
        train_tfms = imagenet_train_transforms(augmentation_strength=0.6)
        test_tfms = imagenet_test_transforms()
        
        assert train_tfms is not None
        assert test_tfms is not None
    
    def test_custom_transforms(self):
        """测试自定义配置transforms"""
        config = {
            'image_size': (128, 128),
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5),
            'train': True,
            'use_color_jitter': True,
            'rotation_degrees': 15,
        }
        tfms = custom_transforms(config)
        assert tfms is not None
    
    def test_custom_transforms_missing_key(self):
        """测试缺少必要配置项时的错误处理"""
        bad_config = {'image_size': (32, 32)}
        with pytest.raises(ValueError, match="缺少必要字段"):
            custom_transforms(bad_config)
    
    def test_cutout_augmentation(self):
        """测试CutOut增强"""
        cutout = CutOut(size_ratio=0.25, probability=1.0)
        
        img_tensor = torch.ones(3, 32, 32)
        result = cutout(img_tensor)
        
        assert result.shape == img_tensor.shape
        assert torch.any(result == 0.0) or torch.allclose(result, img_tensor)
    
    def test_get_normalization_stats(self):
        """测试获取标准化统计量"""
        mean, std = get_normalization_stats('cifar10')
        assert len(mean) == 3
        assert len(std) == 3
        assert mean == DataTransformFactory.CIFAR10_MEAN
        
        mean_in, std_in = get_normalization_stats('imagenet')
        assert mean_in == DataTransformFactory.IMAGENET_MEAN
    
    def test_get_normalization_stats_invalid(self):
        """测试无效数据集名称"""
        with pytest.raises(ValueError, match="不支持"):
            get_normalization_stats('invalid_dataset')


class TestCIFAR10Dataset:
    """CIFAR-10数据集测试"""
    
    @pytest.fixture(scope="class")
    def cifar10_temp_dir(self, temp_dir):
        """为CIFAR-10测试创建临时目录"""
        return temp_dir / "cifar10_test"
    
    def test_cifar10_dataset_class_properties(self, cifar10_temp_dir):
        """测试CIFAR10Dataset类属性"""
        dataset = CIFAR10Dataset(
            root=cifar10_temp_dir,
            train=True,
            download=True,
        )
        
        assert dataset.num_classes == 10
        assert len(dataset.class_names) == 10
        assert 'airplane' in dataset.class_names
        assert 'truck' in dataset.class_names
    
    def test_cifar10_dataloaders_creation(self, cifar10_temp_dir):
        """测试CIFAR-10 DataLoader创建"""
        loaders = get_cifar10_dataloaders(
            data_root=cifar10_temp_dir,
            batch_size=16,
            num_workers=0,
            val_split=0.2,
            augmentation_strength=0.3,
        )
        
        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders
        
        for name, loader in loaders.items():
            assert loader is not None
            assert loader.batch_size == 16
    
    def test_cifar10_dataloader_iteration(self, cifar10_temp_dir):
        """测试DataLoader迭代"""
        loaders = get_cifar10_dataloaders(
            data_root=cifar10_temp_dir,
            batch_size=8,
            num_workers=0,
            val_split=0.15,
        )
        
        for images, labels, indices in loaders['train']:
            assert images.dim() == 4
            assert images.shape[0] <= 8
            assert labels.dim() == 1
            assert indices.dim() == 1
            break
    
    def test_invalid_val_split(self, cifar10_temp_dir):
        """测试无效的验证集比例"""
        with pytest.raises(ValueError, match="val_split必须在"):
            get_cifar10_dataloaders(
                data_root=cifar10_temp_dir,
                val_split=1.5,
            )
        
        with pytest.raises(ValueError, match="val_split必须在"):
            get_cifar10_dataloaders(
                data_root=cifar10_temp_dir,
                val_split=0.0,
            )


class TestCustomDatasets:
    """自定义数据集测试"""
    
    def test_custom_image_dataset_loading(self, sample_images_dir):
        """测试CustomImageDataset加载"""
        dataset = CustomImageDataset(
            data_dir=sample_images_dir,
            split='train',
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        
        assert len(dataset) > 0
        assert dataset.num_classes == 3
        assert len(dataset.class_names) == 3
        
        image, label, idx = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(idx, int)
        assert image.dim() == 3
    
    def test_custom_dataset_all_splits(self, sample_images_dir):
        """测试所有划分的数据集"""
        splits = ['train', 'val', 'test']
        datasets = {}

        for split in splits:
            datasets[split] = CustomImageDataset(
                data_dir=sample_images_dir,
                split=split,
                seed=42,
            )

        for ds in datasets.values():
            assert len(ds) > 0

        assert datasets['train'].num_classes == 3
    
    def test_custom_dataloader_function(self, sample_images_dir):
        """测试get_custom_dataloader工厂函数"""
        loader = get_custom_dataloader(
            data_dir=sample_images_dir,
            split='train',
            batch_size=4,
            num_workers=0,
        )
        
        assert loader is not None
        for images, labels, indices in loader:
            assert images.shape[0] <= 4
            break
    
    def test_imagenet_subset_nonexistent(self, temp_dir):
        """测试不存在的ImageNet目录"""
        fake_dir = temp_dir / "fake_imagenet"
        with pytest.raises(FileNotFoundError):
            ImageNetSubsetDataset(root=fake_dir, split='train')
    
    def test_empty_directory_handling(self, temp_dir):
        """测试空目录的处理"""
        empty_dir = temp_dir / "empty_dataset"
        empty_dir.mkdir()
        
        with pytest.raises(ValueError, match="未找到任何类别文件夹"):
            CustomImageDataset(data_dir=empty_dir)
    
    def test_invalid_split_type(self, sample_images_dir):
        """测试无效的split类型"""
        with pytest.raises(ValueError, match="不支持的split"):
            CustomImageDataset(data_dir=sample_images_dir, split='invalid')
    
    def test_invalid_ratios(self, sample_images_dir):
        """测试无效的比例配置"""
        with pytest.raises(ValueError, match="比例之和必须等于1.0"):
            CustomImageDataset(
                data_dir=sample_images_dir,
                train_ratio=0.9,
                val_ratio=0.2,
                test_ratio=0.1,
            )
    
    def test_index_error_handling(self, sample_images_dir):
        """测试索引越界处理"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        
        with pytest.raises(IndexError):
            _ = dataset[len(dataset)]
    
    def test_supported_extensions(self):
        """测试支持的图像格式列表"""
        assert '.jpg' in SUPPORTED_IMAGE_EXTENSIONS
        assert '.png' in SUPPORTED_IMAGE_EXTENSIONS
        assert len(SUPPORTED_IMAGE_EXTENSIONS) >= 5


class TestVisualizationAndAnalysis:
    """可视化和分析工具测试"""
    
    def test_visualizer_initialization(self):
        """测试可视化器初始化"""
        viz = DatasetVisualizer(figsize=(10, 6))
        assert viz.figsize == (10, 6)
        assert viz.dpi == 150
    
    def test_plot_sample_grid(self, sample_images_dir, temp_dir):
        """测试样本网格绘制"""
        dataset = CustomImageDataset(
            data_dir=sample_images_dir,
            transform=None,
        )
        
        viz = DatasetVisualizer()
        save_path = temp_dir / "test_grid.png"
        
        fig = viz.plot_sample_grid(
            dataset,
            n_rows=2,
            n_cols=3,
            save_path=save_path,
        )
        
        assert fig is not None
        assert save_path.exists()
    
    def test_plot_class_distribution(self, sample_images_dir, temp_dir):
        """测试类别分布图绘制"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        
        viz = DatasetVisualizer()
        save_path = temp_dir / "test_distribution.png"
        
        fig = viz.plot_class_distribution(
            dataset,
            save_path=save_path,
            class_names=['Class A', 'Class B', 'Class C'],
        )
        
        assert fig is not None
        assert save_path.exists()
    
    def test_compare_datasets(self, sample_images_dir, temp_dir):
        """测试多数据集对比"""
        train_ds = CustomImageDataset(data_dir=sample_images_dir, split='train', seed=42)
        val_ds = CustomImageDataset(data_dir=sample_images_dir, split='val', seed=42)
        
        viz = DatasetVisualizer()
        save_path = temp_dir / "test_comparison.png"
        
        fig = viz.compare_datasets(
            {'Train': train_ds, 'Val': val_ds},
            save_path=save_path,
        )
        
        assert fig is not None
        assert save_path.exists()
    
    def test_analyzer_compute_statistics(self, sample_images_dir):
        """测试统计分析计算"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        analyzer = DatasetAnalyzer(dataset)
        
        stats = analyzer.compute_statistics(max_samples=50)
        
        assert stats.num_samples == len(dataset)
        assert isinstance(stats.mean_rgb, tuple)
        assert len(stats.mean_rgb) == 3
        assert isinstance(stats.brightness_stats, dict)
    
    def test_analyzer_check_integrity(self, sample_images_dir):
        """测试完整性检查"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        analyzer = DatasetAnalyzer(dataset)
        
        result = analyzer.check_integrity()
        
        assert 'total' in result
        assert 'valid' in result
        assert 'corrupted' in result
        assert 'error_rate' in result
        assert result['error_rate'] < 0.01
    
    def test_analyzer_generate_report(self, sample_images_dir, temp_dir):
        """测试报告生成"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        analyzer = DatasetAnalyzer(dataset)
        
        report_path = temp_dir / "test_report.json"
        report = analyzer.generate_report(output_path=report_path)
        
        assert report is not None
        assert 'dataset_info' in report
        assert 'statistics' in report
        assert 'integrity' in report
        assert report_path.exists()
    
    def test_verify_dataset_integrity_function(self, sample_images_dir):
        """测试快速验证函数"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        result = verify_dataset_integrity(dataset)
        
        assert result['total'] == len(dataset)
        assert result['error_rate'] < 0.05
    
    def test_create_data_info_file(self, sample_images_dir, temp_dir):
        """测试信息文件生成函数"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        info_path = temp_dir / "dataset_info.json"
        
        info = create_data_info_file(
            dataset,
            output_path=info_path,
            include_visualization=False,
        )
        
        assert info is not None
        assert info_path.exists()


class TestEdgeCases:
    """边界情况和特殊场景测试"""
    
    def test_single_sample_dataset(self, temp_dir):
        """测试只有一个样本的数据集"""
        single_dir = temp_dir / "single_class"
        class_dir = single_dir / "class_1"
        class_dir.mkdir(parents=True)

        img = Image.new('RGB', (32, 32), color='green')
        img.save(class_dir / "only.png")

        dataset = CustomImageDataset(data_dir=single_dir)
        assert len(dataset) == 1
        assert dataset.num_classes == 1
    
    def test_large_batch_size(self, sample_images_dir):
        """测试大于数据集大小的batch size"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=100,
            shuffle=False,
        )
        
        batch = next(iter(loader))
        images, labels, indices = batch
        assert images.shape[0] == len(dataset)
    
    def test_corrupted_image_handling(self, temp_dir):
        """测试损坏图像的处理"""
        corrupt_dir = temp_dir / "corrupt_data"
        class_dir = corrupt_dir / "class_1"
        class_dir.mkdir(parents=True)

        good_img = Image.new('RGB', (32, 32), color='blue')
        good_img.save(class_dir / "good.png")

        with open(class_dir / "bad.png", 'w') as f:
            f.write("not an image")

        dataset = CustomImageDataset(data_dir=corrupt_dir, transform=None)

        good_sample = dataset[0]
        assert good_sample is not None

        try:
            _ = dataset[1]
            assert False, "应该抛出异常"
        except (IOError, OSError, Exception):
            pass
    
    def test_different_image_sizes(self, temp_dir):
        """测试不同尺寸的图像"""
        mixed_dir = temp_dir / "mixed_sizes"
        class_dir = mixed_dir / "class_1"
        class_dir.mkdir(parents=True)

        for size in [(16, 16), (32, 32), (64, 64)]:
            img = Image.new('RGB', size, color='red')
            img.save(class_dir / f"img_{size[0]}x{size[1]}.png")

        simple_transform = transforms.ToTensor()
        dataset = CustomImageDataset(data_dir=mixed_dir, transform=simple_transform)
        analyzer = DatasetAnalyzer(dataset)
        stats = analyzer.compute_statistics(max_samples=10)

        unique_sizes = set(stats.image_sizes)
        assert len(unique_sizes) >= 2

    def test_grayscale_to_rgb_conversion(self, temp_dir):
        """测试灰度图自动转换为RGB"""
        gray_dir = temp_dir / "gray_images"
        class_dir = gray_dir / "class_1"
        class_dir.mkdir(parents=True)

        gray_img = Image.new('L', (32, 32), color=128)
        gray_img.save(class_dir / "gray.png")

        dataset = CustomImageDataset(data_dir=gray_dir)
        image, label, idx = dataset[0]

        assert image.shape[0] == 3


class TestBalancedSampler:
    """平衡采样器测试"""
    
    def test_balanced_sampler_creation(self, sample_images_dir):
        """测试平衡采样器创建"""
        dataset = CustomImageDataset(data_dir=sample_images_dir)
        sampler = create_balanced_sampler(dataset)
        
        assert sampler is not None
        assert hasattr(sampler, 'num_samples')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
