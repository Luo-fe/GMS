"""Precision & Recall 计算器

基于 Kynkäänniemi et al. 2019 的 manifold 估计方法计算 Precision 和 Recall 指标。
这些指标分别衡量生成样本的质量（落在真实流形上的比例）和多样性（覆盖真实流形的程度）。

核心算法:
    1. 提取真实和生成图像的 Inception 特征
    2. 对每个生成样本找到 k 个最近的真实邻居
    3. 计算流形半径 (k-th 最近邻距离)
    4. Precision = 生成样本中距离 <= 半径的比例
    5. Recall = 真实样本中距离 <= 半径的比例

与 FID/IS 的区别:
    - FID: 综合质量和多样性，但难以解释具体问题
    - IS: 只关注生成质量，不与真实数据比较
    - Precision & Recall: 分别评估质量和多样性，更易解释

Example:
    >>> from gms.evaluation.metrics.precision_recall import PrecisionRecallCalculator
    >>> import torch
    >>>
    >>> calculator = PrecisionRecallCalculator(device='cuda', k=3)
    >>> real_images = torch.randn(1000, 3, 299, 299).clamp(0, 1)
    >>> gen_images = torch.randn(1000, 3, 299, 299).clamp(0, 1)
    >>>
    >>> results = calculator.calculate_precision_recall(real_images, gen_images)
    >>> print(f"Precision: {results['precision']:.4f}")
    >>> print(f"Recall: {results['recall']:.4f}")
"""

from typing import Optional, Dict, Any, Tuple, List
from enum import Enum
import logging
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist, cosine
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class DistanceMetric(Enum):
    """支持的度量类型"""

    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


class PrecisionRecallCalculator:
    """Precision & Recall 计算器

    基于 manifold 估计方法计算生成模型的 Precision 和 Recall。

    Attributes:
        device: 计算设备 ('cpu' 或 'cuda')
        batch_size: 批处理大小
        k: 用于估计流形半径的最近邻数量
        distance_metric: 距离度量类型

    Example:
        >>> calculator = PrecisionRecallCalculator(k=3, device='cuda')
        >>> results = calculator.calculate_precision_recall(real_imgs, gen_imgs)
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 50,
        k: int = 3,
        distance_metric: Union[str, DistanceMetric] = DistanceMetric.EUCLIDEAN,
        subset_size: int = 10000,
    ) -> None:
        """初始化 Precision & Recall 计算器

        Args:
            device: 计算设备，'cpu' 或 'cuda'
            batch_size: 特征提取时的批处理大小
            k: 用于确定流形半径的最近邻数量 (默认 3)
            distance_metric: 距离度量类型，支持 'euclidean', 'cosine', 等
            subset_size: 用于计算的最大样本数 (避免内存溢出)

        Raises:
            ValueError: 如果参数非法
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"device 必须是 'cpu' 或 'cuda', 得到 {device}")

        if k < 1:
            raise ValueError(f"k 必须为正整数，得到 {k}")

        if isinstance(distance_metric, str):
            try:
                self.distance_metric = DistanceMetric(distance_metric.lower())
            except ValueError:
                valid_metrics = [m.value for m in DistanceMetric]
                raise ValueError(
                    f"不支持的度量类型 '{distance_metric}'，"
                    f"可选值: {valid_metrics}"
                )
        else:
            self.distance_metric = distance_metric

        self.device = device
        self.batch_size = batch_size
        self.k = k
        self.subset_size = subset_size

        logger.info(
            f"初始化 PrecisionRecallCalculator: device={device}, "
            f"k={k}, metric={self.distance_metric.value}"
        )

    def _compute_distance_matrix(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
    ) -> np.ndarray:
        """计算两组特征之间的距离矩阵

        支持多种距离度量方式。

        Args:
            features_a: 第一组特征 (N_a, D)
            features_b: 第二组特征 (N_b, D)

        Returns:
            距离矩阵 (N_a, N_b)，其中 [i, j] 是 features_a[i] 和 features_b[j] 的距离
        """
        metric = self.distance_metric.value

        if metric == "cosine":
            # 余弦距离 = 1 - 余弦相似度
            distances = cdist(features_a, features_b, metric="cosine")
        elif metric == "euclidean":
            distances = cdist(features_a, features_b, metric="euclidean")
        elif metric == "manhattan":
            distances = cdist(features_a, features_b, metric="cityblock")
        elif metric == "chebyshev":
            distances = cdist(features_a, features_b, metric="chebyshev")
        else:
            distances = cdist(features_a, features_b, metric=metric)

        return distances

    def _estimate_manifold_radius(
        self,
        real_features: np.ndarray,
    ) -> float:
        """估计真实数据流形的半径

        使用 k-NN 方法：对每个真实样本找到第 k 个最近邻，
        流形半径定义为所有第 k 个最近邻距离的中位数。

        Args:
            real_features: 真实数据的特征矩阵 (N, D)

        Returns:
            流形半径 (浮点数)
        """
        logger.info(f"估计流形半径 (k={self.k})...")

        # 使用 NearestNeighbors 进行高效查询
        nbrs = NearestNeighbors(
            n_neighbors=self.k + 1,  # +1 因为包含自身
            metric=self.distance_metric.value,
        )
        nbrs.fit(real_features)

        # 获取每个点的 k+1 个邻居的距离
        distances, indices = nbrs.kneighbors(real_features)

        # 第 k 个邻居的距离 (索引 k，因为索引 0 是自身)
        kth_distances = distances[:, self.k]

        # 使用中位数作为流形半径 (比均值更鲁棒)
        radius = float(np.median(kth_distances))

        logger.info(f"流形半径估计完成: {radius:.6f}")

        return radius

    def _calculate_precision(
        self,
        gen_features: np.ndarray,
        real_features: np.ndarray,
        radius: float,
    ) -> float:
        """计算 Precision

        Precision 衡量生成的样本有多少落在真实数据流形上。
        对于每个生成样本，如果其到最近真实样本的距离 <= radius，
        则认为该生成样本在流形上。

        Args:
            gen_features: 生成样本的特征 (M, D)
            real_features: 真实样本的特征 (N, D)
            radius: 流形半径

        Returns:
            Precision 值 [0, 1]
        """
        # 为了效率，可以只检查最近的几个邻居
        nbrs = NearestNeighbors(
            n_neighbors=1,
            metric=self.distance_metric.value,
        )
        nbrs.fit(real_features)

        distances, _ = nbrs.kneighbors(gen_features)

        # 统计距离 <= radius 的比例
        precision = float(np.mean(distances <= radius))

        return precision

    def _calculate_recall(
        self,
        gen_features: np.ndarray,
        real_features: np.ndarray,
        radius: float,
    ) -> float:
        """计算 Recall

        Recall 衡量真实数据流形有多少被生成的样本覆盖。
        对于每个真实样本，如果其到最近生成样本的距离 <= radius，
        则认为该真实样本被覆盖。

        Args:
            gen_features: 生成样本的特征 (M, D)
            real_features: 真实样本的特征 (N, D)
            radius: 流形半径

        Returns:
            Recall 值 [0, 1]
        """
        nbrs = NearestNeighbors(
            n_neighbors=1,
            metric=self.distance_metric.value,
        )
        nbrs.fit(gen_features)

        distances, _ = nbrs.kneighbors(real_features)

        recall = float(np.mean(distances <= radius))

        return recall

    def _calculate_density(
        self,
        gen_features: np.ndarray,
        real_features: np.ndarray,
        radius: float,
        k_density: int = 5,
    ) -> float:
        """计算 Density (密度) 指标

        Density 衡量生成样本在流形上的密度分布。
        基于每个生成样本周围的真实样本密度。

        Args:
            gen_features: 生成样本的特征
            real_features: 真实样本的特征
            radius: 流形半径
            k_density: 密度计算的近邻数

        Returns:
            Density 值
        """
        nbrs_real = NearestNeighbors(
            n_neighbors=k_density,
            metric=self.distance_metric.value,
        )
        nbrs_real.fit(real_features)

        # 获取每个生成样本的 k 个最近真实邻居的距离
        distances, _ = nbrs_real.kneighbors(gen_features)

        # 计算每个生成样本周围的密度 (使用高斯核)
        sigma = radius / 2.0
        densities = np.exp(-(distances ** 2) / (2 * sigma ** 2))
        density = float(np.mean(np.mean(densities, axis=1)))

        return density

    def calculate_precision_recall_from_features(
        self,
        real_features: np.ndarray,
        gen_features: np.ndarray,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """从预计算的特征计算 Precision 和 Recall

        当已经提取了特征时可以直接使用此方法。

        Args:
            real_features: 真实图像特征 (N, D)
            gen_features: 生成图像特征 (M, D)
            k: 可选，覆盖默认的 k 值

        Returns:
            包含以下键的字典:
                - precision: Precision 值 [0, 1]
                - recall: Recall 值 [0, 1]
                - density: Density 值
                - manifold_radius: 使用的流形半径

        Example:
            >>> results = calculator.calculate_precision_recall_from_features(real_feats, gen_feats)
            >>> print(f"Precision: {results['precision']:.4f}")
        """
        effective_k = k or self.k

        logger.info(
            f"从预计算特征计算 P&R: real={len(real_features)}, "
            f"gen={len(gen_features)}, k={effective_k}"
        )

        # 如果样本太多，进行子采样以提高效率
        if len(real_features) > self.subset_size:
            indices = np.random.choice(
                len(real_features), size=self.subset_size, replace=False
            )
            real_features_subset = real_features[indices]
            logger.info(f"真实数据子采样: {self.subset_size}/{len(real_features)}")
        else:
            real_features_subset = real_features

        if len(gen_features) > self.subset_size:
            indices = np.random.choice(
                len(gen_features), size=self.subset_size, replace=False
            )
            gen_features_subset = gen_features[indices]
            logger.info(f"生成数据子采样: {self.subset_size}/{len(gen_features)}")
        else:
            gen_features_subset = gen_features

        # 估计流形半径
        radius = self._estimate_manifold_radius(real_features_subset)

        # 计算 Precision
        precision = self._calculate_precision(
            gen_features_subset, real_features_subset, radius
        )

        # 计算 Recall
        recall = self._calculate_recall(
            gen_features_subset, real_features_subset, radius
        )

        # 计算 Density
        density = self._calculate_density(
            gen_features_subset, real_features_subset, radius
        )

        results = {
            "precision": precision,
            "recall": recall,
            "density": density,
            "manifold_radius": radius,
        }

        logger.info(
            f"P&R 计算完成: Precision={precision:.4f}, "
            f"Recall={recall:.4f}, Density={density:.4f}"
        )

        return results

    def calculate_precision_recall(
        self,
        real_images: torch.Tensor,
        gen_images: torch.Tensor,
        feature_extractor=None,
        k: Optional[int] = None,
    ) -> Dict[str, float]:
        """计算 Precision 和 Recall (完整流程)

        从图像开始，提取特征并计算 Precision、Recall 和 Density。

        Args:
            real_images: 真实图像张量 (N, C, H, W)，值域 [0, 1]
            gen_images: 生成图像张量 (M, C, H, W)，值域 [0, 1]
            feature_extractor: 可选的特征提取器实例 (FIDCalculator 或其他)
            k: 可选，覆盖默认的 k 值

        Returns:
            包含指标结果的字典

        Raises:
            ValueError: 如果输入不是有效的图像张量

        Example:
            >>> from gms.evaluation.metrics.fid_score import FIDCalculator
            >>> fid_calc = FIDCalculator(device='cuda')
            >>> pr_calc = PrecisionRecallCalculator(k=3)
            >>> results = pr_calc.calculate_precision_recall(real_imgs, gen_imgs, fid_calc)
        """
        logger.info(
            f"开始计算 P&R: real={len(real_images)}, gen={len(gen_images)}"
        )

        # 提取特征
        if feature_extractor is not None and hasattr(feature_extractor, "extract_features"):
            logger.info("使用提供的特征提取器...")
            real_features = feature_extractor.extract_features(real_images)
            gen_features = feature_extractor.extract_features(gen_images)
        else:
            raise ValueError(
                "需要提供特征提取器 (FIDCalculator 实例)。"
                "建议先创建 FIDCalculator 并传入。"
            )

        # 从特征计算指标
        results = self.calculate_precision_recall_from_features(
            real_features, gen_features, k=k
        )

        return results

    def get_detailed_analysis(
        self,
        real_features: np.ndarray,
        gen_features: np.ndarray,
    ) -> Dict[str, Any]:
        """获取详细的分析结果

        除了基本的 Precision/Recall/Density 外，还提供：
        - 距离分布统计
        - 不同 k 值的结果对比
        - 流形覆盖率分析

        Args:
            real_features: 真实特征
            gen_features: 生成特征

        Returns:
            详细分析结果的字典
        """
        logger.info("执行详细分析...")

        analysis = {}

        # 基本指标
        basic_results = self.calculate_precision_recall_from_features(
            real_features, gen_features
        )
        analysis.update(basic_results)

        # 不同 k 值的敏感性分析
        k_values = [1, 3, 5, 10]
        k_sensitivity = {}
        for k_val in k_values:
            try:
                results_k = self.calculate_precision_recall_from_features(
                    real_features, gen_features, k=k_val
                )
                k_sensitivity[f"k={k_val}"] = {
                    "precision": results_k["precision"],
                    "recall": results_k["recall"],
                }
            except Exception as e:
                logger.warning(f"k={k_val} 分析失败: {e}")

        analysis["k_sensitivity"] = k_sensitivity

        # 距离分布统计
        distances_gen_to_real = self._compute_distance_matrix(
            gen_features[: min(len(gen_features), 1000)],
            real_features[: min(len(real_features), 1000)],
        )
        min_distances = np.min(distances_gen_to_real, axis=1)

        analysis["distance_statistics"] = {
            "mean": float(np.mean(min_distances)),
            "std": float(np.std(min_distances)),
            "median": float(np.median(min_distances)),
            "min": float(np.min(min_distances)),
            "max": float(np.max(min_distances)),
            "percentile_25": float(np.percentile(min_distances, 25)),
            "percentile_75": float(np.percentile(min_distances, 75)),
        }

        logger.info("详细分析完成")

        return analysis

    def __repr__(self) -> str:
        return (
            f"PrecisionRecallCalculator(device='{self.device}', "
            f"k={self.k}, metric={self.distance_metric.value})"
        )


def calculate_precision_recall_quick(
    real_images: torch.Tensor,
    gen_images: torch.Tensor,
    device: str = "cpu",
    k: int = 3,
) -> Dict[str, float]:
    """快速计算 Precision & Recall 的便捷函数

    一行代码计算 Precision 和 Recall，适合快速评估场景。

    Args:
        real_images: 真实图像张量
        gen_images: 生成图像张量
        device: 计算设备
        k: 近邻数

    Returns:
        包含 precision, recall, density 的字典

    Example:
        >>> results = calculate_precision_recall_quick(real_imgs, gen_imgs, device='cuda')
    """
    from .fid_score import FIDCalculator

    fid_calc = FIDCalculator(device=device)
    pr_calc = PrecisionRecallCalculator(device=device, k=k)

    return pr_calc.calculate_precision_recall(
        real_images, gen_images, feature_extractor=fid_calc
    )
