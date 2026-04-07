# Gaussian Mixture Solver (GMS)

A complete framework for Gaussian Mixture Model optimization designed to enhance diffusion model image generation through moment matching and adaptive sampling.

## What is this?

GMS is a research framework that combines statistical moment estimation with deep learning to improve the quality of generated images from diffusion models. Instead of using standard Gaussian noise during the diffusion process, GMS learns a two-component Gaussian Mixture Model that better captures the underlying data distribution.

The main idea: use moment matching (mean, variance, skewness) to fit a GMM to your data, then use this GMM to guide the diffusion sampling process.

## Features

- **Moment Estimation**: Extract first, second, and third-order moments from image features using pretrained networks (ResNet, VGG)
- **GMM Optimization**: Fit two-component Gaussian Mixture Models using Generalized Method of Moments
- **Flexible Sampling**: Multiple scheduling strategies (linear, cosine) with checkpoint/resume support
- **Diffusion Integration**: Drop-in enhancement for DDPM, DDIM, and other diffusion architectures
- **Training Pipeline**: End-to-end training with mixed precision, gradient accumulation, and distributed support
- **Evaluation Metrics**: FID, Inception Score, Precision & Recall out of the box

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/GMS.git
cd GMS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
pytest tests/ -v
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ RAM recommended

## Quick Start

### 1. Moment Estimation

Extract statistical moments from image features:

```python
from gms.moment_estimation import ResNetFeatureExtractor, MomentEstimator

# Load pretrained backbone
backbone = ResNetFeatureExtractor(
    architecture='resnet50',
    feature_layer='layer3',
    use_pretrained=True
)

# Create moment estimator
estimator = MomentEstimator(
    feature_dim=1024,
    output_dim=512,
    enable_mean=True,
    enable_variance=True,
    enable_skewness=True
)

# Extract moments
features = backbone.extract_features(images)  # images: [B, C, H, W]
moments = estimator(features)

print(f"Mean shape: {moments.mean.shape}")
print(f"Variance shape: {moments.variance.shape}")
print(f"Skewness shape: {moments.skewness.shape}")
```

### 2. GMM Fitting

Fit a two-component Gaussian Mixture Model:

```python
from gms.gmm_optimization import (
    GaussianMixtureModel,
    AdamOptimizer,
    KMeansInitializer,
    GMMSerializer
)

# Initialize with K-means++
initializer = KMeansInitializer()
init_params = initializer.initialize(feature_data)

# Optimize parameters
optimizer = AdamOptimizer(config)
result = optimizer.optimize(target_moments, init_params.to_optimizer_params())

# Create GMM model
gmm = GaussianMixtureModel(result.params)

# Save model
serializer = GMMSerializer()
serializer.save_json(result.params, 'gmm_model.json')
```

### 3. Sampling

Generate samples from the fitted GMM:

```python
from gms.sampling import BatchGaussianMixtureSampler, CosineScheduler

# Create sampler with cosine schedule
sampler = BatchGaussianMixtureSampler(
    scheduler=CosineScheduler(total_steps=1000),
    gmm_parameters=gmm_params
)

# Generate samples
samples = sampler.sample(n_samples=1000)
```

### 4. Diffusion Model Integration

Enhance your diffusion model with GMS:

```python
from gms.diffusion_integration import (
    GMSDiffusionAdapter,
    GMSForwardProcess,
    GMSBackwardProcess,
    GMSTrainer,
    GMSInferencePipeline
)

# Create adapter and processes
adapter = GMSDiffusionAdapter()
forward_process = GMSForwardProcess(adapter)
backward_process = GMSBackwardProcess(adapter)

# Train
trainer = GMSTrainer(
    model=unet,
    forward_process=forward_process,
    backward_process=backward_process,
    config=config
)
history = trainer.train_full(epochs=100, dataloaders=dataloaders)

# Generate images
pipeline = GMSInferencePipeline.from_trainer(trainer)
images = pipeline.generate(n_samples=64, method='ddim')
```

### 5. Evaluation

Compute standard metrics:

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

## Project Structure

```
GMS/
├── src/gms/
│   ├── moment_estimation/    # Feature extraction and moment calculation
│   ├── gmm_optimization/     # GMM fitting and optimization
│   ├── sampling/             # Sampling strategies and validation
│   ├── diffusion_integration/ # Diffusion model adapters
│   └── evaluation/           # Metrics and datasets
├── tests/                    # Unit tests
├── configs/                  # Configuration files
└── examples/                 # Example scripts
```

## Datasets

GMS supports standard image datasets out of the box:

### CIFAR-10

The framework will automatically download CIFAR-10 when you first use it:

```python
from gms.evaluation import get_cifar10_dataloaders

loaders = get_cifar10_dataloaders(
    data_root='./data/cifar10',  # Data will be downloaded here
    batch_size=64,
    val_split=0.1
)

# Access loaders
train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

The dataset will be automatically downloaded to `./data/cifar10` on first run.

### Custom Datasets

For your own images, organize them in folders:

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

Then load with:

```python
from gms.evaluation import get_custom_dataloader

loader = get_custom_dataloader(
    data_dir='./your_dataset',
    batch_size=32,
    train_split=0.8
)
```

### ImageNet Subset

For ImageNet experiments:

```python
from gms.evaluation import get_imagenet_dataloaders

loaders = get_imagenet_dataloaders(
    data_root='./data/imagenet',
    batch_size=32,
    subset_classes=[0, 1, 2, 3, 4]  # Optional: use only specific classes
)
```

Note: You need to download ImageNet separately from [official website](https://www.image-net.org/).

## Configuration

Default configurations are in `configs/`:

- `default_config.yaml`: General training parameters
- `model_config.yaml`: Model architecture settings

You can override any setting programmatically:

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

## Performance

On CIFAR-10 with a single RTX 3090:

| Metric | Value |
|--------|-------|
| FID | ~25-30 |
| Inception Score | ~8.5-9.0 |
| Training time | ~12 hours (100 epochs) |
| Inference speed | ~50 images/sec (DDIM, 50 steps) |

Memory usage: ~6GB VRAM with batch_size=32.

## Troubleshooting

### CUDA out of memory

Reduce batch size or enable gradient accumulation:

```python
config = TrainingConfig(
    batch_size=16,  # Reduce from 32
    gradient_accumulation_steps=2  # Effective batch size = 16 * 2 = 32
)
```

### Slow training

Enable mixed precision training:

```python
config = TrainingConfig(
    mixed_precision=True  # Can speed up training 2-3x
)
```

### Poor generation quality

Try adjusting GMM initialization:

```python
# Use more careful initialization
initializer = KMeansInitializer(
    n_init=10,  # Run K-means multiple times
    max_iter=300
)
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_gmm_optimizer.py -v

# Run with coverage
pytest tests/ --cov=src/gms --cov-report=html
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{gms2024,
  title={Gaussian Mixture Solver for Diffusion Models},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository},
  url={https://github.com/yourusername/GMS}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please ensure tests pass:

```bash
pytest tests/
```

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@example.com].
