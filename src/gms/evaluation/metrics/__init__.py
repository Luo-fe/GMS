"""评估指标模块

提供生成模型评估的核心指标计算功能，包括:
- FID (Fréchet Inception Distance): 综合质量和多样性
- IS (Inception Score): 评估图像质量和清晰度
- Precision & Recall: 分别评估质量和多样性覆盖

所有指标均基于预训练的 InceptionV3 网络进行特征提取，
支持 GPU 加速、批量处理和特征缓存等优化。

快速开始:
    >>> from gms.evaluation.metrics import (
    ...     FIDCalculator,
    ...     ISCalculator,
    ...     PrecisionRecallCalculator,
    ...     calculate_fid_quick,
    ...     calculate_is_quick,
    ... )
    >>>
    >>> # 计算 FID
    >>> fid_calc = FIDCalculator(device='cuda')
    >>> fid = fid_calc.calculate_fid(real_images, gen_images)
    >>>
    >>> # 计算 IS
    >>> is_calc = ISCalculator(device='cuda')
    >>> is_mean, is_std = is_calc.calculate_is(gen_images)
    >>>
    >>> # 计算 P&R
    >>> pr_calc = PrecisionRecallCalculator(k=3)
    >>> pr_results = pr_calc.calculate_precision_recall(
    ...     real_images, gen_images, feature_extractor=fid_calc
    ... )
"""

from .fid_score import (
    FIDCalculator,
    calculate_fid_quick,
)

from .is_score import (
    ISCalculator,
    calculate_is_quick,
)

from .precision_recall import (
    PrecisionRecallCalculator,
    DistanceMetric,
    calculate_precision_recall_quick,
)

__all__ = [
    # FID
    "FIDCalculator",
    "calculate_fid_quick",
    # IS
    "ISCalculator",
    "calculate_is_quick",
    # P&R
    "PrecisionRecallCalculator",
    "DistanceMetric",
    "calculate_precision_recall_quick",
]
