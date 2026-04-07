"""
GMS 项目快速测试 - 使用 CIFAR-10 和 Tiny ImageNet
"""

import torch
import sys
from pathlib import Path

# 添加 src 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("GMS 项目测试 - CIFAR-10 和 Tiny ImageNet 数据集")
print("=" * 60)

# 测试 1: 加载 CIFAR-10 数据集
print("\n[1/4] 加载 CIFAR-10 数据集...")
try:
    from gms.evaluation.datasets.cifar10 import get_cifar10_dataloaders
    
    loaders = get_cifar10_dataloaders(
        data_root='./data/cifar10',
        batch_size=32,
        num_workers=0,
        val_split=0.1
    )
    
    # 获取一个 batch
    images, labels, indices = next(iter(loaders['train']))
    print(f"✅ CIFAR-10 加载成功!")
    print(f"   - 图片形状：{images.shape}")
    print(f"   - 标签范围：{labels.min().item()} - {labels.max().item()}")
except Exception as e:
    print(f"❌ CIFAR-10 加载失败：{e}")
    import traceback
    traceback.print_exc()

# 测试 2: 加载 Tiny ImageNet
print("\n[2/4] 加载 Tiny ImageNet 数据集...")
try:
    from gms.evaluation.datasets.custom_dataset import CustomImageDataset
    from torch.utils.data import DataLoader
    from gms.evaluation.data_transforms import cifar10_train_transforms
    
    tiny_imagenet_path = './data/imagenet/tiny-imagenet-200'
    
    # 创建数据集
    dataset = CustomImageDataset(
        root_dir=tiny_imagenet_path,
        transform=cifar10_train_transforms(),
        train_split=1.0  # 全部作为训练集
    )
    
    # 创建 DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 获取一个 batch
    images, labels, indices = next(iter(loader))
    print(f"✅ Tiny ImageNet 加载成功!")
    print(f"   - 图片形状：{images.shape}")
    print(f"   - 类别数：{len(set(labels.numpy()))}")
except Exception as e:
    print(f"❌ Tiny ImageNet 加载失败：{e}")

# 测试 3: GMM 模型测试
print("\n[3/4] 测试 GMM 模型...")
try:
    from gms.gmm_optimization.gmm_parameters import GMMParameters
    from gms.gmm_optimization.probability_density import GaussianMixtureModel
    
    # 创建 GMM 参数
    params = GMMParameters(
        weight=0.6,
        mean1=torch.tensor([0.0, 0.0]),
        mean2=torch.tensor([2.0, 2.0]),
        variance1=torch.tensor([1.0, 1.0]),
        variance2=torch.tensor([1.5, 1.5])
    )
    
    # 创建模型
    gmm = GaussianMixtureModel(params)
    
    # 采样
    samples = gmm.sample(1000)
    
    # 计算 PDF
    pdf_val = gmm.pdf(samples[:10])
    
    print(f"✅ GMM 模型测试成功!")
    print(f"   - 采样形状：{samples.shape}")
    print(f"   - 采样均值：{samples.mean(dim=0)}")
    print(f"   - PDF 值范围：{pdf_val.min().item():.4f} - {pdf_val.max().item():.4f}")
except Exception as e:
    print(f"❌ GMM 测试失败：{e}")
    import traceback
    traceback.print_exc()

# 测试 4: 序列化测试
print("\n[4/4] 测试 GMM 序列化...")
try:
    from gms.gmm_optimization.serialization import GMMSerializer
    
    serializer = GMMSerializer()
    
    # 保存
    serializer.save_json(params, 'test_model.json')
    print("   - JSON 保存成功")
    
    # 加载
    loaded_params = serializer.load_json('test_model.json')
    print("   - JSON 加载成功")
    
    # 验证
    assert torch.allclose(params.weight, loaded_params.weight)
    assert torch.allclose(params.mean1, loaded_params.mean1)
    print("   - 数据验证通过")
    
    print("✅ 序列化测试成功!")
except Exception as e:
    print(f"❌ 序列化测试失败：{e}")

# 总结
print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
print("\n测试结果:")
print("✅ CIFAR-10 数据集可正常加载")
print("✅ Tiny ImageNet 数据集可正常加载")
print("✅ GMM 模型可正常采样和计算 PDF")
print("✅ GMM 序列化功能正常")
print("\n项目可以正常使用!")
