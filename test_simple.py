"""
GMS 项目简单测试脚本
测试核心功能是否正常工作
"""

import torch
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("GMS 项目核心功能测试")
print("=" * 60)

# 测试 1: 导入所有模块
print("\n[测试 1] 检查所有模块是否可以导入...")
try:
    # 矩估计
    from gms.moment_estimation.backbone_networks import ResNetFeatureExtractor
    from gms.moment_estimation.moment_estimators import MomentEstimator
    
    # GMM 优化
    from gms.gmm_optimization.gmm_parameters import GMMParameters
    from gms.gmm_optimization.probability_density import GaussianMixtureModel
    from gms.gmm_optimization.serialization import GMMSerializer
    
    # 采样
    from gms.sampling.batch_sampler import BatchGaussianMixtureSampler
    from gms.sampling.sampling_scheduler import CosineScheduler
    from gms.sampling.sampling_validator import SamplingValidator
    
    # 扩散模型集成
    from gms.diffusion_integration.adapter import GMSDiffusionAdapter
    from gms.diffusion_integration.trainer import TrainingConfig
    
    print("✅ 所有模块导入成功!")
except Exception as e:
    print(f"❌ 导入失败：{e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 2: 创建 GMM 模型并采样
print("\n[测试 2] 测试 GMM 模型和采样...")
try:
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
    samples = gmm.sample(100)
    print(f"   - 采样形状：{samples.shape}")
    print(f"   - 采样均值：{samples.mean(dim=0)}")
    
    # 计算 PDF
    pdf_val = gmm.pdf(samples[:5])
    print(f"   - PDF 值：{pdf_val}")
    
    print("✅ GMM 模型测试成功!")
except Exception as e:
    print(f"❌ GMM 测试失败：{e}")
    import traceback
    traceback.print_exc()

# 测试 3: 测试序列化
print("\n[测试 3] 测试 GMM 序列化...")
try:
    serializer = GMMSerializer()
    
    # 保存
    serializer.save_json(params, 'test_gmm.json')
    print("   - 保存为 JSON 成功")
    
    # 加载
    loaded_params = serializer.load_json('test_gmm.json')
    print("   - 从 JSON 加载成功")
    
    # 验证
    assert torch.allclose(params.weight, loaded_params.weight)
    assert torch.allclose(params.mean1, loaded_params.mean1)
    
    print("✅ 序列化测试成功!")
except Exception as e:
    print(f"❌ 序列化测试失败：{e}")

# 测试 4: 测试调度器
print("\n[测试 4] 测试采样调度器...")
try:
    scheduler = CosineScheduler(
        start_value=1e-4,
        end_value=0.02,
        total_steps=1000
    )
    
    # 获取调度值
    values = [scheduler.get_value(t) for t in [0, 100, 500, 999]]
    print(f"   - 调度值 (t=0,100,500,999): {values}")
    
    print("✅ 调度器测试成功!")
except Exception as e:
    print(f"❌ 调度器测试失败：{e}")

# 测试 5: 测试适配器
print("\n[测试 5] 测试扩散模型适配器...")
try:
    adapter = GMSDiffusionAdapter()
    
    # 创建 GMM 参数
    gmm_params = GMMParameters(
        weight=0.5,
        mean1=torch.zeros(10),
        mean2=torch.ones(10),
        variance1=torch.ones(10),
        variance2=torch.ones(10)
    )
    
    # 转换为噪声调度
    noise_schedule = adapter.adapt_gmm_to_diffusion(gmm_params, num_timesteps=100)
    print(f"   - 噪声调度创建成功")
    print(f"   - 时间步数：{len(noise_schedule.timesteps)}")
    
    print("✅ 适配器测试成功!")
except Exception as e:
    print(f"❌ 适配器测试失败：{e}")
    import traceback
    traceback.print_exc()

# 测试 6: 运行单元测试
print("\n[测试 6] 运行项目单元测试...")
import subprocess
result = subprocess.run(
    ['pytest', 'tests/', '-v', '--tb=short'],
    capture_output=True,
    text=True,
    timeout=300  # 5 分钟超时
)

print(f"测试运行结果：{'✅ 成功' if result.returncode == 0 else '❌ 失败'}")
print(f"通过测试数：{result.stdout.count(' PASSED')}")
print(f"失败测试数：{result.stdout.count(' FAILED')}")

# 显示部分测试结果
lines = result.stdout.split('\n')
for line in lines[-10:]:  # 显示最后 10 行
    if line.strip():
        print(f"   {line}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
