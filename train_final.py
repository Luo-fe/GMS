"""
GMS 项目简化训练脚本 - 快速验证
使用 CIFAR-10 和 Tiny ImageNet 进行训练
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from datetime import datetime
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from gms.diffusion_integration.adapter import GMSDiffusionAdapter
from gms.diffusion_integration.forward_process import GMSForwardProcess, NoiseScheduler
from gms.diffusion_integration.backward_process import GMSBackwardProcess
from gms.diffusion_integration.trainer import GMSTrainer, TrainingConfig
from gms.diffusion_integration.inference import GMSInferencePipeline
from gms.evaluation.datasets.cifar10 import get_cifar10_dataloaders
from gms.evaluation.datasets.custom_dataset import CustomImageDataset


class SimpleUNet(nn.Module):
    """简化版 UNet"""
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.down1 = nn.Conv2d(base_channels, base_channels*2, 3, padding=1, stride=2)
        self.down2 = nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1, stride=2)
        self.mid = nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 4, padding=1, stride=2)
        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 4, padding=1, stride=2)
        self.out_conv = nn.Conv2d(base_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x, t=None):
        x1 = self.relu(self.in_conv(x))
        x2 = self.relu(self.down1(x1))
        x3 = self.relu(self.down2(x2))
        x4 = self.relu(self.mid(x3))
        x = self.relu(self.up2(x4))
        x = self.relu(self.up1(x + x2[:, :, :x.size(2), :x.size(3)]))
        x = self.out_conv(x + x1[:, :, :x.size(2), :x.size(3)])
        return x


def train_model(dataset_name, data_path, epochs=10, batch_size=32):
    """训练模型"""
    print(f"\n{'='*70}")
    print(f"训练：{dataset_name}")
    print(f"{'='*70}")
    
    # 1. 加载数据
    print(f"\n[1/6] 加载 {dataset_name} 数据集...")
    if dataset_name == "CIFAR-10":
        dataloaders = get_cifar10_dataloaders(
            data_root=data_path,
            batch_size=batch_size,
            num_workers=0,
            val_split=0.1
        )
    else:
        dataset = CustomImageDataset(
            data_dir=data_path,
            transform=None,
            train_split=0.9
        )
        dataloaders = {
            'train': DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            'val': DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        }
    
    train_loader = dataloaders['train']
    print(f"✅ 数据加载成功 - 训练集：{len(train_loader.dataset)} 样本")
    
    # 2. 创建模型
    print(f"\n[2/6] 创建 UNet 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleUNet(in_channels=3, out_channels=3, base_channels=64).to(device)
    print(f"✅ 模型创建成功 - 设备：{device}")
    
    # 3. 创建噪声调度器
    print(f"\n[3/6] 创建噪声调度器...")
    noise_scheduler = NoiseScheduler(
        num_timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type='linear'
    )
    print(f"✅ 噪声调度器创建成功")
    
    # 4. 创建 GMS 组件
    print(f"\n[4/6] 创建 GMS 组件...")
    adapter = GMSDiffusionAdapter()
    forward_process = GMSForwardProcess(adapter, noise_scheduler)
    backward_process = GMSBackwardProcess(adapter, noise_scheduler)
    print(f"✅ GMS 组件创建成功")
    
    # 5. 创建训练器
    print(f"\n[5/6] 创建训练器...")
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=1e-4,
        device=str(device)
    )
    
    trainer = GMSTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
        forward_process=forward_process,
        backward_process=backward_process,
        config=config
    )
    print(f"✅ 训练器创建成功")
    
    # 6. 开始训练
    print(f"\n[6/6] 开始训练 ({epochs} epochs)...")
    start_time = time.time()
    
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, _, _) in enumerate(train_loader):
            images = images.to(device)
            t = torch.randint(0, 1000, (images.size(0),), device=device)
            
            loss = trainer.train_step(images, t)
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(avg_loss)
        history['epochs'].append(epoch+1)
        
        print(f"  ✅ Epoch {epoch+1} - Loss: {avg_loss:.4f} - Time: {epoch_time:.1f}s")
        
        # 保存检查点
        if (epoch+1) % 5 == 0:
            checkpoint_dir = Path('outputs') / f'checkpoints_{dataset_name.replace(" ", "_")}'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"  💾 检查点已保存：{checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"\n✅ 训练完成！总用时：{total_time/60:.1f} 分钟")
    
    return trainer, history, model


def generate_images(model, dataset_name, num_samples=64):
    """生成图像并计算统计指标"""
    print(f"\n{'='*70}")
    print(f"生成图像 - {dataset_name}")
    print(f"{'='*70}")
    
    device = next(model.parameters()).device
    model.eval()
    
    # 创建推理 pipeline
    noise_scheduler = NoiseScheduler(num_steps=1000, beta_start=1e-4, beta_end=0.02)
    adapter = GMSDiffusionAdapter()
    forward_process = GMSForwardProcess(adapter, noise_scheduler)
    backward_process = GMSBackwardProcess(adapter, noise_scheduler)
    
    pipeline = GMSInferencePipeline(
        model=model,
        noise_scheduler=noise_scheduler,
        forward_process=forward_process,
        backward_process=backward_process,
        device=str(device)
    )
    
    # 生成图像
    print(f"\n[1/2] 生成 {num_samples} 张图像...")
    generated = pipeline.generate(n_samples=num_samples, method='ddim', num_steps=50)
    print(f"✅ 图像生成完成 - 形状：{generated.shape}")
    
    # 计算统计指标
    print(f"\n[2/2] 计算统计指标...")
    mean_val = generated.mean().item()
    std_val = generated.std().item()
    min_val = generated.min().item()
    max_val = generated.max().item()
    
    # 保存图像
    output_dir = Path('outputs') / f'generated_{dataset_name.replace(" ", "_")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存为 PNG
    try:
        from torchvision.utils import save_image
        save_image(generated[:16], output_dir / 'sample_grid.png', nrow=4, normalize=True)
        print(f"✅ 样本网格已保存：{output_dir / 'sample_grid.png'}")
    except:
        print(f"⚠️  无法保存图像（需要 torchvision）")
    
    print(f"\n生成图像统计:")
    print(f"  - 均值：{mean_val:.4f}")
    print(f"  - 标准差：{std_val:.4f}")
    print(f"  - 最小值：{min_val:.4f}")
    print(f"  - 最大值：{max_val:.4f}")
    
    return {
        'generated_images': generated,
        'statistics': {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        }
    }


def main():
    """主函数"""
    print("="*70)
    print("GMS 项目训练和评估 - 严格按照 .trae/specs 计划")
    print("="*70)
    
    results = {}
    
    # 1. CIFAR-10
    print("\n" + "="*70)
    print("阶段 1: CIFAR-10 数据集训练")
    print("="*70)
    
    cifar10_trainer, cifar10_history, cifar10_model = train_model(
        dataset_name="CIFAR-10",
        data_path='./data/cifar10',
        epochs=10,
        batch_size=32
    )
    
    cifar10_eval = generate_images(cifar10_model, "CIFAR-10", num_samples=64)
    
    results['CIFAR-10'] = {
        'history': cifar10_history,
        'evaluation': cifar10_eval
    }
    
    # 2. Tiny ImageNet
    print("\n" + "="*70)
    print("阶段 2: Tiny ImageNet 数据集训练")
    print("="*70)
    
    imagenet_trainer, imagenet_history, imagenet_model = train_model(
        dataset_name="Tiny ImageNet",
        data_path='./data/imagenet/tiny-imagenet-200',
        epochs=10,
        batch_size=32
    )
    
    imagenet_eval = generate_images(imagenet_model, "Tiny ImageNet", num_samples=64)
    
    results['Tiny ImageNet'] = {
        'history': imagenet_history,
        'evaluation': imagenet_eval
    }
    
    # 生成报告
    generate_final_report(results)


def generate_final_report(results):
    """生成最终评估报告"""
    print(f"\n{'='*70}")
    print("生成最终评估报告")
    print(f"{'='*70}")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'project': 'Gaussian Mixture Solver (GMS)',
        'version': '1.0.0',
        'datasets': {}
    }
    
    for dataset_name, result in results.items():
        history = result['history']
        evaluation = result['evaluation']
        
        report['datasets'][dataset_name] = {
            'training': {
                'epochs': len(history['epochs']),
                'final_loss': history['train_loss'][-1],
                'loss_history': history['train_loss'],
                'epochs_list': history['epochs']
            },
            'evaluation': {
                'generated_images_count': evaluation['generated_images'].shape[0],
                'statistics': evaluation['statistics']
            }
        }
    
    # 保存 JSON 报告
    report_dir = Path('outputs')
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'evaluation_report.json'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 评估报告已保存：{report_path}")
    
    # 打印摘要
    print(f"\n{'='*70}")
    print("训练结果摘要")
    print(f"{'='*70}")
    
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  训练:")
        print(f"    - 轮数：{len(result['history']['epochs'])}")
        print(f"    - 最终损失：{result['history']['train_loss'][-1]:.4f}")
        print(f"  生成图像质量:")
        print(f"    - 均值：{result['evaluation']['statistics']['mean']:.4f}")
        print(f"    - 标准差：{result['evaluation']['statistics']['std']:.4f}")
        print(f"    - 范围：[{result['evaluation']['statistics']['min']:.4f}, {result['evaluation']['statistics']['max']:.4f}]")
    
    print(f"\n💡 完整报告：{report_path}")
    print(f"💡 生成图像：outputs/generated_*/")
    print(f"\n{'='*70}")
    print("✅ 所有训练和评估完成！")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
