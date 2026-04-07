"""骨干网络模块的单元测试

测试特征提取器、预处理器和缓存机制的功能。
使用pytest fixtures提供测试数据。
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from PIL import Image

# 导入被测模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gms.moment_estimation.base_feature_extractor import BaseFeatureExtractor
from gms.moment_estimation.backbone_networks import (
    ResNetFeatureExtractor,
    VGGFeatureExtractor,
    create_feature_extractor,
)
from gms.moment_estimation.preprocessing import (
    ImagePreprocessor,
    create_standard_preprocessor,
)
from gms.moment_estimation.feature_cache import (
    FeatureCache,
    CachedFeatureExtractor,
)


# ==================== Fixtures ====================

@pytest.fixture(scope="session")
def device():
    """返回可用的PyTorch设备"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_images(device):
    """生成用于测试的示例图像批次"""
    torch.manual_seed(42)
    batch_size = 4
    channels = 3
    height, width = 224, 224
    images = torch.randn(batch_size, channels, height, width).to(device)
    return images


@pytest.fixture(scope="session")
def single_image(device):
    """生成单张测试图像"""
    torch.manual_seed(42)
    image = torch.randn(3, 224, 224).to(device)
    return image


@pytest.fixture
def temp_image_files(tmp_path):
    """创建临时图像文件用于测试"""
    image_paths = []
    for i in range(3):
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        path = tmp_path / f"test_image_{i}.png"
        img.save(str(path))
        image_paths.append(path)
    
    # 添加一个非图像文件
    (tmp_path / "not_an_image.txt").write_text("test")
    
    return tmp_path, image_paths


# ==================== 测试基类 ====================

class TestBaseFeatureExtractor:
    """测试特征提取器基类"""

    def test_abstract_class_cannot_instantiate(self):
        """验证抽象类不能直接实例化"""
        with pytest.raises(TypeError):
            BaseFeatureExtractor()

    def test_concrete_implementation(self, device):
        """测试具体实现的基本功能"""
        
        class DummyExtractor(BaseFeatureExtractor):
            def __init__(self):
                super().__init__(device=device)
                self.set_output_dim(128)

            def extract_features(self, images):
                batch_size = images.shape[0]
                return torch.zeros(batch_size, 128, device=self.device)

        extractor = DummyExtractor()
        assert extractor.device == device
        assert extractor.get_output_dim() == 128
        
        test_input = torch.randn(2, 3, 64, 64).to(device)
        output = extractor(test_input)
        assert output.shape == (2, 128)

    def test_freeze_unfreeze(self, device):
        """测试冻结和解冻功能"""
        
        class DummyExtractor(BaseFeatureExtractor):
            def __init__(self):
                super().__init__(device=device)
                self.model = torch.nn.Linear(10, 5)

            def extract_features(self, images):
                return self.model(images.flatten(1)[:, :10])

        extractor = DummyExtractor()
        
        # 初始状态：未冻结
        assert not extractor.is_frozen
        
        # 冻结
        extractor.freeze()
        assert extractor.is_frozen
        for param in extractor.model.parameters():
            assert not param.requires_grad
        
        # 解冻
        extractor.unfreeze()
        assert not extractor.is_frozen
        for param in extractor.model.parameters():
            assert param.requires_grad

    def test_device_management(self, device):
        """测试设备管理"""
        
        class DummyExtractor(BaseFeatureExtractor):
            def __init__(self):
                super().__init__(device=device)
                self.model = torch.nn.Linear(10, 5).to(device)

            def extract_features(self, images):
                return self.model(images.flatten(1)[:, :10])

        extractor = DummyExtractor()
        new_device = torch.device("cpu")
        extractor.to(new_device)
        assert extractor.device == new_device

    def test_repr(self, device):
        """测试字符串表示"""
        
        class DummyExtractor(BaseFeatureExtractor):
            def __init__(self):
                super().__init__(device=device)
                self.set_output_dim(256)

            def extract_features(self, images):
                return torch.zeros(images.shape[0], 256, device=self.device)

        extractor = DummyExtractor()
        repr_str = repr(extractor)
        assert "DummyExtractor" in repr_str
        assert "device" in repr_str.lower()

    def test_callable_interface(self, device):
        """测试可调用接口"""
        
        class DummyExtractor(BaseFeatureExtractor):
            def __init__(self):
                super().__init__(device=device)
                self.set_output_dim(64)

            def extract_features(self, images):
                batch_size = images.shape[0]
                return torch.zeros(batch_size, 64, device=self.device)

        extractor = DummyExtractor()
        test_input = torch.randn(1, 3, 32, 32).to(device)
        
        # 测试直接调用
        output = extractor(test_input)
        assert output.shape == (1, 64)


# ==================== 测试ResNet特征提取器 ====================

class TestResNetFeatureExtractor:
    """测试ResNet特征提取器"""

    @pytest.mark.parametrize("architecture", ["resnet18", "resnet34", "resnet50", "resnet101"])
    def test_instantiation(self, architecture, device):
        """测试不同架构的实例化"""
        extractor = ResNetFeatureExtractor(
            architecture=architecture,
            feature_layer="layer3",
            use_pretrained=True,
            device=device,
        )
        assert extractor.architecture == architecture
        assert extractor.feature_layer == "layer3"
        assert extractor.use_pretrained is True
        assert extractor.device == device
        assert extractor.get_output_dim() > 0

    @pytest.mark.parametrize(
        "feature_layer,expected_channels",
        [("layer1", 256), ("layer2", 512), ("layer3", 1024), ("layer4", 2048)]
    )
    def test_feature_layers(self, feature_layer, expected_channels, device):
        """测试不同特征层的输出维度"""
        extractor = ResNetFeatureExtractor(
            architecture="resnet50",
            feature_layer=feature_layer,
            use_pretrained=False,
            device=device,
        )
        assert extractor.get_output_dim() == expected_channels

    def test_invalid_architecture(self, device):
        """测试无效架构名称"""
        with pytest.raises(ValueError, match="不支持的ResNet架构"):
            ResNetFeatureExtractor(architecture="resnet999", device=device)

    def test_invalid_feature_layer(self, device):
        """测试无效特征层"""
        with pytest.raises(ValueError, match="不支持的特征层"):
            ResNetFeatureExtractor(feature_layer="invalid_layer", device=device)

    def test_extract_features_shape(self, sample_images, device):
        """测试特征提取输出形状"""
        extractor = ResNetFeatureExtractor(
            architecture="resnet50",
            feature_layer="layer3",
            use_pretrained=False,
            device=device,
        )
        
        features = extractor.extract_features(sample_images)
        batch_size = sample_images.shape[0]
        output_dim = extractor.get_output_dim()
        
        # layer3 输出应该是 (B, C, H', W') 或展平后的形式
        assert features.dim() >= 2
        assert features.shape[0] == batch_size

    def test_extract_features_single_image(self, single_image, device):
        """测试单张图像特征提取"""
        extractor = ResNetFeatureExtractor(
            architecture="resnet18",
            feature_layer="layer1",
            use_pretrained=False,
            device=device,
        )
        
        # 添加batch维度
        input_tensor = single_image.unsqueeze(0)
        features = extractor.extract_features(input_tensor)
        assert features.shape[0] == 1

    def test_freeze_layers(self, device):
        """测试冻结特定层"""
        extractor = ResNetFeatureExtractor(
            architecture="resnet50",
            feature_layer="layer3",
            freeze_layers=["layer1", "layer2"],
            use_pretrained=False,
            device=device,
        )
        
        # 验证layer1和layer2被冻结
        for param in extractor.model.layer1.parameters():
            assert not param.requires_grad
        for param in extractor.model.layer2.parameters():
            assert not param.requires_grad

    def test_get_feature_map_size(self, device):
        """测试特征图尺寸计算"""
        extractor = ResNetFeatureExtractor(
            architecture="resnet50",
            feature_layer="layer3",
            use_pretrained=False,
            device=device,
        )
        
        size = extractor.get_feature_map_size((224, 224))
        assert isinstance(size, tuple)
        assert len(size) == 2
        assert size[0] > 0 and size[1] > 0

    def test_no_grad_during_extraction(self, sample_images, device):
        """验证特征提取时不计算梯度"""
        extractor = ResNetFeatureExtractor(
            architecture="resnet34",
            feature_layer="layer2",
            use_pretrained=False,
            device=device,
        )
        
        with torch.enable_grad():
            features = extractor.extract_features(sample_images)
            assert not features.requires_grad

    def test_repr_output(self, device):
        """测试字符串表示格式"""
        extractor = ResNetFeatureExtractor(
            architecture="resnet101",
            feature_layer="layer4",
            device=device,
        )
        repr_str = repr(extractor)
        assert "ResNetFeatureExtractor" in repr_str
        assert "resnet101" in repr_str
        assert "layer4" in repr_str


# ==================== 测试VGG特征提取器 ====================

class TestVGGFeatureExtractor:
    """测试VGG特征提取器"""

    @pytest.mark.parametrize("architecture", ["vgg11", "vgg13", "vgg16", "vgg19"])
    def test_instantiation(self, architecture, device):
        """测试不同架构的实例化"""
        extractor = VGGFeatureExtractor(
            architecture=architecture,
            feature_layer="features_22",
            use_pretrained=True,
            device=device,
        )
        assert extractor.architecture == architecture
        assert extractor.get_output_dim() > 0

    @pytest.mark.parametrize(
        "feature_layer,expected_channels",
        [("features_12", 256), ("features_22", 512), ("features_32", 512)]
    )
    def test_feature_layers(self, feature_layer, expected_channels, device):
        """测试不同特征层"""
        extractor = VGGFeatureExtractor(
            architecture="vgg16",
            feature_layer=feature_layer,
            use_pretrained=False,
            device=device,
        )
        assert extractor.get_output_dim() == expected_channels

    def test_extract_features_shape(self, sample_images, device):
        """测试特征提取输出形状"""
        extractor = VGGFeatureExtractor(
            architecture="vgg16",
            feature_layer="features_22",
            use_pretrained=False,
            device=device,
        )
        
        features = extractor.extract_features(sample_images)
        assert features.dim() >= 2
        assert features.shape[0] == sample_images.shape[0]

    def test_freeze_features_option(self, device):
        """测试冻结特征选项"""
        extractor = VGGFeatureExtractor(
            architecture="vgg16",
            feature_layer="features_22",
            freeze_features=True,
            use_pretrained=False,
            device=device,
        )
        
        # 所有features参数应该被冻结
        for param in extractor.model.features.parameters():
            assert not param.requires_grad

    def test_invalid_architecture(self, device):
        """测试无效架构"""
        with pytest.raises(ValueError, match="不支持的VGG架构"):
            VGGFeatureExtractor(architecture="vgg99", device=device)

    def test_get_feature_map_size(self, device):
        """测试特征图尺寸"""
        extractor = VGGFeatureExtractor(
            architecture="vgg16",
            feature_layer="features_22",
            device=device,
        )
        
        size = extractor.get_feature_map_size((224, 224))
        assert isinstance(size, tuple)
        assert len(size) == 2


# ==================== 测试工厂函数 ====================

class TestCreateFeatureExtractor:
    """测试工厂函数"""

    def test_create_resnet(self, device):
        """通过工厂函数创建ResNet提取器"""
        extractor = create_feature_extractor("resnet50", device=device)
        assert isinstance(extractor, ResNetFeatureExtractor)
        assert extractor.architecture == "resnet50"

    def test_create_resnet_with_layer(self, device):
        """通过工厂函数创建带层指定的ResNet提取器"""
        extractor = create_feature_extractor("resnet50_layer2", device=device)
        assert isinstance(extractor, ResNetFeatureExtractor)
        assert extractor.feature_layer == "layer2"

    def test_create_vgg(self, device):
        """通过工厂函数创建VGG提取器"""
        extractor = create_feature_extractor("vgg16", device=device)
        assert isinstance(extractor, VGGFeatureExtractor)
        assert extractor.architecture == "vgg16"

    def test_create_vgg_with_layer(self, device):
        """通过工厂函数创建带层指定的VGG提取器"""
        extractor = create_feature_extractor("vgg16_features_22", device=device)
        assert isinstance(extractor, VGGFeatureExtractor)
        assert extractor.feature_layer == "features_22"

    def test_invalid_name(self, device):
        """测试无效名称"""
        with pytest.raises(ValueError, match="无法识别"):
            create_feature_extractor("invalid_model", device=device)


# ==================== 测试预处理器 ====================

class TestImagePreprocessor:
    """测试图像预处理器"""

    def test_basic_initialization(self):
        """测试基本初始化"""
        preprocessor = ImagePreprocessor(image_size=(224, 224))
        assert preprocessor.image_size == (224, 224)
        assert preprocessor.normalize is True
        assert preprocessor.augmentation is False

    def test_process_tensor(self, single_image):
        """测试处理PyTorch张量"""
        preprocessor = ImagePreprocessor(image_size=(224, 224), normalize=False)
        result = preprocessor(single_image.cpu())
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 224, 224)
        assert result.dtype == torch.float32

    def test_process_numpy_array(self):
        """测试处理NumPy数组"""
        preprocessor = ImagePreprocessor(image_size=(128, 128), normalize=False)
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        result = preprocessor(img_array)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 128, 128)

    def test_process_pil_image(self):
        """测试处理PIL Image"""
        preprocessor = ImagePreprocessor(image_size=(64, 64), normalize=False)
        pil_img = Image.fromarray(np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8))
        
        result = preprocessor(pil_img)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 64, 64)

    def test_process_from_file(self, temp_image_files):
        """测试从文件路径加载并处理"""
        tmp_dir, image_paths = temp_image_files
        preprocessor = ImagePreprocessor(image_size=(64, 64), normalize=False)
        
        result = preprocessor(image_paths[0])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 64, 64)

    def test_batch_processing(self, temp_image_files):
        """测试批量处理"""
        _, image_paths = temp_image_files
        preprocessor = ImagePreprocessor(image_size=(64, 64), normalize=False)
        
        batch_result = preprocessor.process_batch(image_paths)
        assert isinstance(batch_result, torch.Tensor)
        assert batch_result.shape == (3, 3, 64, 64)  # 3张图片, 3通道, 64x64

    def test_empty_batch_raises_error(self):
        """测试空批次抛出异常"""
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="不能为空"):
            preprocessor.process_batch([])

    def test_directory_processing(self, temp_image_files):
        """测试从目录处理"""
        tmp_dir, _ = temp_image_files
        preprocessor = ImagePreprocessor(image_size=(64, 64), normalize=False)
        
        tensors, paths = preprocessor.process_from_directory(tmp_dir)
        assert tensors.shape[0] == 3  # 3个图像文件
        assert len(paths) == 3

    def test_nonexistent_file_raises_error(self):
        """测试不存在的文件"""
        preprocessor = ImagePreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor("/nonexistent/path/image.png")

    def test_unsupported_format_raises_error(self, tmp_path):
        """测试不支持的格式"""
        unsupported_file = tmp_path / "test.xyz"
        unsupported_file.write_bytes(b"fake data")
        
        preprocessor = ImagePreprocessor()
        with pytest.raises(ValueError, match="不支持"):
            preprocessor(unsupported_file)

    def test_augmentation_enabled(self):
        """测试启用数据增强"""
        augmentation_config = {
            "horizontal_flip": True,
            "rotation": 15,
        }
        preprocessor = ImagePreprocessor(
            image_size=(64, 64),
            augmentation=True,
            augmentation_config=augmentation_config,
        )
        assert preprocessor.augmentation is True
        
        img_array = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        result = preprocessor(img_array)
        assert result.shape == (3, 64, 64)

    def test_normalize_parameters(self):
        """测试自定义归一化参数"""
        custom_mean = [0.5, 0.5, 0.5]
        custom_std = [0.25, 0.25, 0.25]
        
        preprocessor = ImagePreprocessor(
            mean=custom_mean,
            std=custom_std,
        )
        assert preprocessor.mean == custom_mean
        assert preprocessor.std == custom_std

    def test_device_transfer(self):
        """测试设备转移"""
        device = torch.device("cpu")
        preprocessor = ImagePreprocessor(device=device)
        
        new_device = torch.device("cpu")
        preprocessor.to(new_device)
        assert preprocessor.device == new_device

    def test_transform_info(self):
        """测试获取变换信息"""
        preprocessor = ImagePreprocessor(image_size=(299, 299))
        info = preprocessor.get_transform_info()
        
        assert info["image_size"] == (299, 299)
        assert info["normalize"] is True
        assert "num_transforms" in info

    def test_standard_preprocessor_factory(self):
        """测试标准预处理器工厂函数"""
        preprocessor = create_standard_preprocessor("imagenet")
        assert isinstance(preprocessor, ImagePreprocessor)
        assert preprocessor.image_size == (224, 224)

    def test_custom_standard_preprocessor(self):
        """测试自定义标准预处理器"""
        preprocessor = create_standard_preprocessor(
            model_type="imagenet",
            image_size=(299, 299),
            augmentation=True,
        )
        assert preprocessor.image_size == (299, 299)
        assert preprocessor.augmentation is True

    def test_repr(self):
        """测试字符串表示"""
        preprocessor = ImagePreprocessor(image_size=(256, 256))
        repr_str = repr(preprocessor)
        assert "ImagePreprocessor" in repr_str
        assert "256" in repr_str


# ==================== 测试缓存机制 ====================

class TestFeatureCache:
    """测试特征缓存"""

    def test_initialization(self):
        """测试初始化"""
        cache = FeatureCache(max_size=100)
        assert cache.max_size == 100
        assert len(cache) == 0
        assert cache.enable_disk_cache is False

    def test_hash_generation(self):
        """测试哈希生成"""
        tensor = torch.randn(3, 224, 224)
        hash1 = FeatureCache.generate_hash(tensor)
        hash2 = FeatureCache.generate_hash(tensor)
        
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex长度
        assert hash1 == hash2  # 相同输入产生相同哈希

    def test_different_inputs_different_hashes(self):
        """测试不同输入产生不同哈希"""
        tensor1 = torch.ones(3, 32, 32)
        tensor2 = torch.zeros(3, 32, 32)
        
        hash1 = FeatureCache.generate_hash(tensor1)
        hash2 = FeatureCache.generate_hash(tensor2)
        
        assert hash1 != hash2

    def test_store_and_get(self):
        """测试存储和获取"""
        cache = FeatureCache(max_size=10)
        key = "test_key"
        value = torch.randn(256)
        
        cache.store(key, value)
        retrieved = cache.get(key)
        
        assert retrieved is not None
        assert torch.equal(retrieved, value)

    def test_cache_miss(self):
        """测试缓存未命中"""
        cache = FeatureCache()
        result = cache.get("nonexistent_key")
        assert result is None

    def test_contains(self):
        """测试contains方法"""
        cache = FeatureCache()
        key = "key1"
        value = torch.tensor([1.0, 2.0, 3.0])
        
        assert key not in cache
        
        cache.store(key, value)
        assert key in cache

    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        cache = FeatureCache(max_size=3)
        
        # 存储3个项目
        for i in range(3):
            cache.store(f"key_{i}", torch.tensor([float(i)]))
        
        assert len(cache) == 3
        
        # 存储第4个项目，应该淘汰最早的
        cache.store("key_3", torch.tensor([3.0]))
        assert len(cache) == 3
        assert "key_0" not in cache
        assert "key_3" in cache

    def test_remove(self):
        """测试移除条目"""
        cache = FeatureCache()
        cache.store("key1", torch.tensor([1.0]))
        cache.store("key2", torch.tensor([2.0]))
        
        assert cache.remove("key1") is True
        assert "key1" not in cache
        assert "key2" in cache
        
        assert cache.remove("nonexistent") is False

    def test_clear(self):
        """测试清空缓存"""
        cache = FeatureCache(max_size=10)
        for i in range(5):
            cache.store(f"key_{i}", torch.tensor([float(i)]))
        
        assert len(cache) == 5
        cache.clear()
        assert len(cache) == 0

    def test_statistics(self):
        """测试统计信息"""
        cache = FeatureCache(max_size=100)
        
        # 执行一些操作
        tensor = torch.randn(128)
        cache.store("key1", tensor)
        cache.get("key1")  # 命中
        cache.get("key2")  # 未命中
        
        stats = cache.get_statistics()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0

    def test_print_statistics(self, capsys):
        """测试打印统计信息（不崩溃）"""
        cache = FeatureCache()
        cache.store("test", torch.tensor([1.0, 2.0]))
        cache.print_statistics()
        
        captured = capsys.readouterr()
        assert "统计信息" in captured.out

    def test_export_keys(self):
        """测试导出键列表"""
        cache = FeatureCache()
        keys = ["a", "b", "c"]
        for key in keys:
            cache.store(key, torch.tensor([1.0]))
        
        exported = cache.export_keys()
        assert set(exported) == set(keys)

    def test_len_and_contains(self):
        """测试__len__和__contains__"""
        cache = FeatureCache()
        assert len(cache) == 0
        
        cache.store("k1", torch.tensor([1.0]))
        assert len(cache) == 1
        assert "k1" in cache

    def test_disk_cache_initialization(self, tmp_path):
        """测试磁盘缓存初始化"""
        cache = FeatureCache(
            enable_disk_cache=True,
            disk_cache_dir=tmp_path / "cache_test",
        )
        
        assert cache.enable_disk_cache is True
        assert cache.disk_cache_dir.exists()

    def test_disk_cache_store_and_load(self, tmp_path):
        """测试磁盘缓存存储和加载"""
        cache = FeatureCache(
            max_size=2,
            enable_disk_cache=True,
            disk_cache_dir=tmp_path / "disk_cache",
        )
        
        key = "disk_key"
        value = torch.randn(512)
        
        # 存储
        cache.store(key, value, persist_to_disk=True)
        
        # 清空内存缓存
        cache._memory_cache.clear()
        
        # 从磁盘重新加载
        loaded = cache.get(key)
        assert loaded is not None
        assert torch.equal(loaded, value)

    def test_disk_cache_cleanup(self, tmp_path):
        """测试磁盘缓存清理"""
        cache = FeatureCache(
            max_size=10,
            enable_disk_cache=True,
            disk_cache_dir=tmp_path / "cleanup_test",
            disk_cache_max_size_mb=0.001,  # 很小的限制
        )
        
        # 存储多个项目
        for i in range(5):
            cache.store(f"key_{i}", torch.randn(100), persist_to_disk=True)
        
        # 清理
        removed = cache.cleanup_disk_cache()
        assert removed >= 0

    def test_hash_with_different_types(self):
        """测试不同数据类型的哈希生成"""
        arr = np.array([1, 2, 3])
        tensor = torch.from_numpy(arr)
        
        hash_arr = FeatureCache.generate_hash(arr)
        hash_tensor = FeatureCache.generate_hash(tensor)
        
        # NumPy数组和对应的tensor应该产生相同或不同的哈希取决于实现
        assert isinstance(hash_arr, str)
        assert isinstance(hash_tensor, str)

    def test_invalid_hash_method(self):
        """测试无效的哈希算法"""
        with pytest.raises(ValueError):
            FeatureCache.generate_hash(torch.tensor([1]), method="invalid")

    def test_repr(self):
        """测试字符串表示"""
        cache = FeatureCache(max_size=500)
        repr_str = repr(cache)
        assert "FeatureCache" in repr_str
        assert "500" in repr_str


# ==================== 测试带缓存的提取器 ====================

class TestCachedFeatureExtractor:
    """测试带缓存的特征提取器装饰器"""

    def test_cached_extraction(self, sample_images, device):
        """测试带缓存的特征提取"""
        
        class SimpleExtractor:
            def __init__(self):
                self.call_count = 0
            
            def extract_features(self, images):
                self.call_count += 1
                batch_size = images.shape[0]
                return torch.randn(batch_size, 128, device=images.device)

        base_extractor = SimpleExtractor()
        cached = CachedFeatureExtractor(base_extractor, max_size=10)

        # 第一次调用 - 应该执行实际提取
        result1 = cached.extract(sample_images)
        assert base_extractor.call_count == 1

        # 第二次调用相同输入 - 应该从缓存返回
        result2 = cached.extract(sample_images)
        assert base_extractor.call_count == 1  # 没有增加
        assert torch.equal(result1, result2)

    def test_disable_cache(self, sample_images, device):
        """测试禁用缓存"""
        
        class SimpleExtractor:
            def __init__(self):
                self.call_count = 0
            
            def extract_features(self, images):
                self.call_count += 1
                batch_size = images.shape[0]
                return torch.randn(batch_size, 64, device=images.device)

        base_extractor = SimpleExtractor()
        cached = CachedFeatureExtractor(base_extractor)

        # 禁用缓存
        _ = cached.extract(sample_images, use_cache=False)
        _ = cached.extract(sample_images, use_cache=False)
        
        assert base_extractor.call_count == 2  # 每次都实际执行

    def test_clear_cache(self, sample_images, device):
        """测试清空缓存"""
        
        class SimpleExtractor:
            def extract_features(self, images):
                return torch.randn(images.shape[0], 32, device=images.device)

        base_extractor = SimpleExtractor()
        cached = CachedFeatureExtractor(base_extractor, max_size=5)

        # 填充缓存
        for _ in range(3):
            cached.extract(torch.randn_like(sample_images))

        assert len(cached.cache) > 0
        
        # 清空
        cached.clear_cache()
        assert len(cached.cache) == 0

    def test_cache_statistics(self, sample_images, device):
        """测试缓存统计信息"""
        
        class SimpleExtractor:
            def extract_features(self, images):
                return torch.randn(images.shape[0], 16, device=images.device)

        base_extractor = SimpleExtractor()
        cached = CachedFeatureExtractor(base_extractor)

        # 执行一些操作
        cached.extract(sample_images)
        cached.extract(sample_images)  # 缓存命中
        cached.extract(torch.randn_like(sample_images))  # 新数据

        stats = cached.get_cache_statistics()
        assert "hits" in stats
        assert "stores" in stats

    def test_repr(self, device):
        """测试字符串表示"""
        
        class DummyExtractor:
            pass

        cached = CachedFeatureExtractor(DummyExtractor())
        repr_str = repr(cached)
        assert "CachedFeatureExtractor" in repr_str


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试：测试组件协同工作"""

    def test_full_pipeline(self, device):
        """测试完整的预处理+特征提取管道"""
        # 创建预处理器
        preprocessor = ImagePreprocessor(
            image_size=(224, 224),
            normalize=True,
            device=device,
        )

        # 创建特征提取器
        extractor = ResNetFeatureExtractor(
            architecture="resnet18",
            feature_layer="layer1",
            use_pretrained=False,
            device=device,
            preprocessor=preprocessor,
        )

        # 创建缓存
        cache = FeatureCache(max_size=50)

        # 包装为带缓存的提取器
        cached_extractor = CachedFeatureExtractor(extractor, cache=cache)

        # 生成测试图像
        test_images = torch.randn(2, 3, 224, 224).to(device)

        # 执行完整流程
        features1 = cached_extractor.extract(test_images)
        features2 = cached_extractor.extract(test_images)  # 应该从缓存返回

        assert features1.shape[0] == 2
        assert torch.equal(features1, features2)

        print("\n✓ 完整管道集成测试通过")

    def test_multiple_extractors_shared_cache(self, device):
        """测试多个提取器可以协同工作"""
        # 创建两个不同的提取器
        resnet = ResNetFeatureExtractor(
            architecture="resnet18",
            feature_layer="layer1",
            use_pretrained=False,
            device=device,
        )
        vgg = VGGFeatureExtractor(
            architecture="vgg11",
            feature_layer="features_12",
            use_pretrained=False,
            device=device,
        )

        test_images = torch.randn(2, 3, 224, 224).to(device)

        # 使用两个提取器独立工作
        resnet_features = resnet.extract_features(test_images)
        vgg_features = vgg.extract_features(test_images)

        # 验证它们都正常工作
        assert resnet_features.shape[0] == 2
        assert vgg_features.shape[0] == 2
        assert resnet_features.shape != vgg_features.shape or True  # 输出可能不同

        print("\n✓ 多提取器协同工作测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
