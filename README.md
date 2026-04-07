# Gaussian Mixture Solver (GMS)

这是一个高斯混合模型优化框架，主要用来提升扩散模型的图像生成质量。通过统计矩匹配和自适应采样策略，让生成的图片更好看。

## 介绍

GMS 把统计矩估计和深度学习结合起来，改进了扩散模型的生成效果。不是用标准的高斯噪声，而是学习一个双分量的高斯混合模型，这个模型能更好地拟合真实数据的分布。

主要思路：先用矩匹配（均值、方差、偏度）把 GMM 拟合到你的数据上，然后用这个拟合好的 GMM 来引导扩散采样过程。

## 功能列表

- **矩估计模块**：用预训练的 ResNet/VGG 网络提取图像特征，计算一阶、二阶、三阶矩
- **GMM 参数优化**：用广义矩方法拟合双分量高斯混合模型
- **灵活的采样策略**：支持线性、余弦等多种调度，有检查点和恢复功能
- **扩散模型集成**：可以直接用到 DDPM、DDIM 这些主流扩散架构上
- **训练流程**：端到端训练，支持混合精度、梯度累积、分布式训练
- **评估指标**：FID、Inception Score、Precision & Recall 这些都有

## 安装

```bash
# 克隆项目
git clone https://github.com/Luo-fe/GMS.git
cd GMS

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac 用这个
# 或者用 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 验证一下安装成功没有
pytest tests/ -v
```

## 环境要求

- Python 3.8 或更高版本
- PyTorch 2.0 或更高版本
- CUDA 11.8 或更高版本（想用GPU训练的话）
- 内存建议 16GB 以上

## 快速开始

### 1. 矩估计

从图像特征里提取统计矩：

```python
from gms.moment_estimation import ResNetFeatureExtractor, MomentEstimator

# 加载预训练的骨干网络
backbone = ResNetFeatureExtractor(
    architecture='resnet50',
    feature_layer='layer3',
    use_pretrained=True
)

# 创建矩估计器
estimator = MomentEstimator(
    feature_dim=1024,
    output_dim=512,
    enable_mean=True,
    enable_variance=True,
    enable_skewness=True
)

# 提取矩
features = backbone.extract_features(images)  # images 是 [B, C, H, W] 格式
moments = estimator(features)

print(f"均值形状: {moments.mean.shape}")
print(f"方差形状: {moments.variance.shape}")
print(f"偏度形状: {moments.skewness.shape}")
```

### 2. GMM 拟合

拟合一个双分量高斯混合模型：

```python
from gms.gmm_optimization import (
    GaussianMixtureModel,
    AdamOptimizer,
    KMeansInitializer,
    GMMSerializer
)

# 用 K-means++ 初始化
initializer = KMeansInitializer()
init_params = initializer.initialize(feature_data)

# 优化参数
optimizer = AdamOptimizer(config)
result = optimizer.optimize(target_moments, init_params.to_optimizer_params())

# 创建 GMM 模型
gmm = GaussianMixtureModel(result.params)

# 保存模型
serializer = GMMSerializer()
serializer.save_json(result.params, 'gmm_model.json')
```

### 3. 采样

从拟合好的 GMM 生成样本：

```python
from gms.sampling import BatchGaussianMixtureSampler, CosineScheduler

# 创建采样器，用余弦调度
sampler = BatchGaussianMixtureSampler(
    scheduler=CosineScheduler(total_steps=1000),
    gmm_parameters=gmm_params
)

# 生成样本
samples = sampler.sample(n_samples=1000)
```

### 4. 扩散模型集成

用 GMS 增强你的扩散模型：

```python
from gms.diffusion_integration import (
    GMSDiffusionAdapter,
    GMSForwardProcess,
    GMSBackwardProcess,
    GMSTrainer,
    GMSInferencePipeline
)

# 创建适配器和过程
adapter = GMSDiffusionAdapter()
forward_process = GMSForwardProcess(adapter)
backward_process = GMSBackwardProcess(adapter)

# 训练
trainer = GMSTrainer(
    model=unet,
    forward_process=forward_process,
    backward_process=backward_process,
    config=config
)
history = trainer.train_full(epochs=100, dataloaders=dataloaders)

# 生成图片
pipeline = GMSInferencePipeline.from_trainer(trainer)
images = pipeline.generate(n_samples=64, method='ddim')
```

### 5. 评估

计算标准指标：

```python
from gms.evaluation import FIDCalculator, ISCalculator

# FID
fid_calc = FIDCalculator(device='cuda')
fid_score = fid_calc.calculate_fid(real_images, generated_images)

# Inception Score
is_calc = ISCalculator()
is_mean, is_std = is_calc.calculate_is(generated_images)

print(f"FID: {fid_score:.2f}")
print(f"IS: {is_mean:.2f} ± {is_std:.2f}")
```

## 项目结构

```
GMS/
├── src/gms/
│   ├── moment_estimation/    # 特征提取和矩计算
│   ├── gmm_optimization/     # GMM 拟合和优化
│   ├── sampling/             # 采样策略和验证
│   ├── diffusion_integration/ # 扩散模型适配器
│   └── evaluation/           # 评估指标和数据集
├── tests/                    # 单元测试
├── configs/                  # 配置文件
└── examples/                 # 示例脚本
```

## 数据集

这个项目支持几个常用的图像数据集：

### CIFAR-10

第一次用的时候会自动下载：

```python
from gms.evaluation import get_cifar10_dataloaders

loaders = get_cifar10_dataloaders(
    data_root='./data/cifar10',  # 数据会下载到这里
    batch_size=64,
    val_split=0.1
)

# 用的时候直接用就行
train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

第一次运行时，数据会自动下载到 `./data/cifar10` 这个文件夹里。

### 自定义数据集

自己的图片可以按文件夹分类：

```
your_dataset/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       └── img3.jpg
└── test/
    └── ...
```

然后这样加载：

```python
from gms.evaluation import get_custom_dataloader

loader = get_custom_dataloader(
    data_dir='./your_dataset',
    batch_size=32,
    train_split=0.8
)
```

### ImageNet 子集

用 ImageNet 的话：

```python
from gms.evaluation import get_imagenet_dataloaders

loaders = get_imagenet_dataloaders(
    data_root='./data/imagenet',
    batch_size=32,
    subset_classes=[0, 1, 2, 3, 4]  # 只选某些类
)
```

注意：ImageNet 要自己去官网下载，地址是 https://www.image-net.org/。

## 配置

默认配置在 `configs/` 文件夹里：

- `default_config.yaml`：一般训练参数
- `model_config.yaml`：模型架构设置

也可以在代码里直接改：

```python
from gms.diffusion_integration import TrainingConfig

config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    device='cuda',
    mixed_precision=True
)
```

## 性能

在 CIFAR-10 上，单张 RTX 3090 上的表现：

| 指标 | 数值 |
|------|------|
| FID | ~25-30 |
| Inception Score | ~8.5-9.0 |
| 训练时间 | ~12小时 (100 epochs) |
| 推理速度 | ~50张/秒 (DDIM, 50步) |

显存占用：batch_size=32 时大概 6GB。

## 常见问题

### CUDA 显存不够

调小 batch_size 或者开梯度累积：

```python
config = TrainingConfig(
    batch_size=16,  # 从32改到16
    gradient_accumulation_steps=2  # 这样有效 batch_size 还是 32
)
```

### 训练太慢

开混合精度训练：

```python
config = TrainingConfig(
    mixed_precision=True  # 能快 2-3 倍
)
```

### 生成质量不好

试试调整 GMM 初始化：

```python
# 更仔细地初始化
initializer = KMeansInitializer(
    n_init=10,  # 多跑几次 K-means
    max_iter=300
)
```

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 只运行某个文件的测试
pytest tests/test_gmm_optimizer.py -v

# 带覆盖率运行
pytest tests/ --cov=src/gms --cov-report=html
```

## 引用

如果用了这个代码，麻烦引用一下：

```bibtex
@misc{gms2024,
  title={Gaussian Mixture Solver for Diffusion Models},
  author={Luo-fe},
  year={2024},
  howpublished={GitHub Repository},
  url={https://github.com/Luo-fe/GMS}
}
```

## 许可证

MIT License - 详见 LICENSE 文件。

## 贡献

欢迎提 PR！大的改动最好先开个 issue 讨论一下。

提交前记得跑一下测试：

```bash
pytest tests/
```

## 联系方式

有问题或者建议的话，可以：
- 开 GitHub issue：https://github.com/Luo-fe/GMS/issues
- 邮件联系：2564118019@qq.com
