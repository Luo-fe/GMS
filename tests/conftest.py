import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture(scope="session")
def device():
    """返回可用的 PyTorch 设备（CUDA 或 CPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def sample_data(device):
    """生成用于测试的示例数据"""
    np.random.seed(42)
    data = np.random.randn(1000, 3)
    return torch.tensor(data, dtype=torch.float32, device=device)

@pytest.fixture(scope="session")
def test_config():
    """返回测试配置字典"""
    return {
        "n_components": 3,
        "max_iterations": 100,
        "tolerance": 1e-6,
        "learning_rate": 0.01,
        "batch_size": 64,
        "n_epochs": 10,
    }
