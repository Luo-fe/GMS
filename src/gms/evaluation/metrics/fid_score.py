"""FID (Fréchet Inception Distance) 计算器

使用预训练的 InceptionV3 网络计算真实图像和生成图像之间的 Fréchet 距离。
FID 是衡量生成模型质量的重要指标，值越低表示生成质量越好。

核心算法:
    1. 使用 InceptionV3 提取特征 (pool3 层, 2048 维)
    2. 计算真实和生成图像的特征统计量 (均值 μ 和协方差 Σ)
    3. 计算 FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r·Σ_g)^(1/2))

Example:
    >>> from gms.evaluation.metrics.fid_score import FIDCalculator
    >>> import torch
    >>>
    >>> # 创建计算器
    >>> calculator = FIDCalculator(device='cuda', batch_size=50)
    >>>
    >>> # 准备数据 (B, C, H, W), 值域 [0, 1]
    >>> real_images = torch.randn(1000, 3, 299, 299).clamp(0, 1)
    >>> gen_images = torch.randn(1000, 3, 299, 299).clamp(0, 1)
    >>>
    >>> # 计算 FID
    >>> fid = calculator.calculate_fid(real_images, gen_images)
    >>> print(f"FID: {fid:.2f}")
"""

from typing import Optional, Tuple, Dict, Any
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from scipy import linalg

logger = logging.getLogger(__name__)


class FIDCalculator:
    """Fréchet Inception Distance 计算器

    使用预训练的 InceptionV3 模型提取特征并计算 FID 分数。
    支持批量处理、GPU 加速、特征缓存等优化功能。

    Attributes:
        device: 计算设备 ('cpu' 或 'cuda')
        batch_size: 批处理大小
        dims: 特征维度 (默认 2048，对应 pool3 层)
        inception_model: 预训练的 InceptionV3 模型
        feature_cache: 特征缓存字典

    Example:
        >>> calculator = FIDCalculator(device='cuda', batch_size=64)
        >>> fid = calculator.calculate_fid(real_imgs, gen_imgs)
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 50,
        dims: int = 2048,
        use_cache: bool = True,
        num_workers: int = 4,
    ) -> None:
        """初始化 FID 计算器

        Args:
            device: 计算设备，'cpu' 或 'cuda'
            batch_size: 批处理大小，用于避免内存溢出
            dims: 特征维度 (InceptionV3 pool3 层输出为 2048)
            use_cache: 是否启用特征缓存
            num_workers: 数据加载的工作进程数

        Raises:
            ValueError: 如果 device 不是 'cpu' 或 'cuda'
            RuntimeError: 如果无法加载预训练模型
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"device 必须是 'cpu' 或 'cuda', 得到 {device}")

        self.device = device
        self.batch_size = batch_size
        self.dims = dims
        self.use_cache = use_cache
        self.num_workers = num_workers
        self.feature_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        logger.info(
            f"初始化 FIDCalculator: device={device}, batch_size={batch_size}, "
            f"dims={dims}, cache={'启用' if use_cache else '禁用'}"
        )

        self._load_inception_model()

    def _load_inception_model(self) -> None:
        """加载预训练的 InceptionV3 模型

        加载 torchvision 提供的 InceptionV3，修改为特征提取模式，
        输出 pool3 层的 2048 维特征向量。
        """
        logger.info("加载预训练 InceptionV3 模型...")

        try:
            inception = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )

            # 移除最后的分类层，保留到 pool3 层
            inception.fc = torch.nn.Identity()
            inception.eval()

            self.inception_model = inception.to(self.device)

            logger.info("InceptionV3 模型加载成功")
        except Exception as e:
            raise RuntimeError(f"无法加载 InceptionV3 模型: {e}")

    def _preprocess_images(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """预处理图像以适应 InceptionV3 的输入要求

        将 [0, 1] 范围的图像转换为 InceptionV3 所需的格式。
        包括 resize 到 299x299 和归一化。

        Args:
            images: 输入图像张量 (N, C, H, W)，值域 [0, 1]

        Returns:
            预处理后的图像张量
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Resize 到 299x299 如果需要
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images, size=(299, 299), mode="bilinear", align_corners=False
            )

        # ImageNet 归一化
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        images = normalize(images)

        return images

    @torch.no_grad()
    def _extract_features_batch(
        self, images: torch.Tensor
    ) -> np.ndarray:
        """从一批图像中提取 Inception 特征

        Args:
            images: 预处理后的图像张量 (N, C, 299, 299)

        Returns:
            特征矩阵 (N, 2048)
        """
        features_list = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size].to(self.device)

            # InceptionV3 需要 training=False 才能正确输出特征
            features = self.inception_model(batch)

            if isinstance(features, tuple):
                features = features[0]

            features_list.append(features.cpu().numpy())

        return np.concatenate(features_list, axis=0)

    def extract_features(
        self, images: torch.Tensor, cache_key: Optional[str] = None
    ) -> np.ndarray:
        """从图像中提取 Inception 特征

        支持批量处理和特征缓存。如果启用了缓存且提供了 cache_key，
        将检查是否已有缓存的特征。

        Args:
            images: 输入图像张量 (N, C, H, W)，值域 [0, 1]
            cache_key: 可选的缓存键名

        Returns:
            特征矩阵 (N, 2048)
        """
        if self.use_cache and cache_key is not None:
            if cache_key in self.feature_cache:
                logger.debug(f"使用缓存的特征: {cache_key}")
                cached_mean, _ = self.feature_cache[cache_key]
                return cached_mean

        logger.info(f"提取特征: {len(images)} 张图像")

        preprocessed = self._preprocess_images(images)
        features = self._extract_features_batch(preprocessed)

        logger.info(f"特征提取完成: shape={features.shape}")

        return features

    def compute_statistics(
        self, images: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算图像集的统计量 (均值和协方差矩阵)

        从图像中提取特征后计算特征的均值向量和协方差矩阵。

        Args:
            images: 输入图像张量 (N, C, H, W)

        Returns:
            Tuple 包含:
                - mean: 特征均值向量 (2048,)
                - cov: 特征协方差矩阵 (2048, 2048)
        """
        features = self.extract_features(images)
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)

        # 协方差正则化以提高数值稳定性
        cov += np.eye(cov.shape[0]) * 1e-6

        return mean, cov

    def _compute_frechet_distance(
        self,
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """计算两个高斯分布之间的 Fréchet 距离

        实现 FID 公式:
            d² = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2(Σ₁·Σ₂)^(1/2))

        Args:
            mu1: 第一个分布的均值向量
            sigma1: 第一个分布的协方差矩阵
            mu2: 第二个分布的均值向量
            sigma2: 第二个分布的协方差矩阵
            eps: 数值稳定性常数

        Returns:
            Fréchet 距离 (浮点数)

        Raises:
            ValueError: 如果输入维度不匹配
            np.linalg.LinAlgError: 如果矩阵分解失败
        """
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        if mu1.shape != mu2.shape:
            raise ValueError(
                f"均值向量维度不匹配: {mu1.shape} vs {mu2.shape}"
            )

        if (
            sigma1.shape[0] != sigma1.shape[1]
            or sigma2.shape[0] != sigma2.shape[1]
            or sigma1.shape[0] != sigma2.shape[0]
        ):
            raise ValueError(
                f"协方差矩阵维度不匹配: {sigma1.shape} vs {sigma2.shape}"
            )

        diff = mu1 - mu2

        # 计算 sqrtm(sigma1 @ sigma2) 使用更稳定的 SVD 方法
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        # 处理数值不稳定性导致的复数结果
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            logger.warning(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # 处理复数部分 (由于数值误差可能产生小的虚部)
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m} too large")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fid_value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

        return float(fid_value)

    def calculate_fid_from_features(
        self,
        real_features: np.ndarray,
        gen_features: np.ndarray,
    ) -> float:
        """从预计算的特征计算 FID

        当已经提取了特征时，可以直接使用此方法计算 FID，
        避免重复的特征提取过程。

        Args:
            real_features: 真实图像的特征矩阵 (N_real, 2048)
            gen_features: 生成图像的特征矩阵 (N_gen, 2048)

        Returns:
            FID 分数 (浮点数，越低越好)

        Example:
            >>> real_feats = calculator.extract_features(real_images)
            >>> gen_feats = calculator.extract_features(gen_images)
            >>> fid = calculator.calculate_fid_from_features(real_feats, gen_feats)
        """
        logger.info("从预计算特征计算 FID...")

        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False) + np.eye(real_features.shape[1]) * 1e-6

        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False) + np.eye(gen_features.shape[1]) * 1e-6

        fid = self._compute_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

        logger.info(f"FID 计算完成: {fid:.4f}")

        return fid

    def calculate_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        real_cache_key: Optional[str] = "real",
        gen_cache_key: Optional[str] = "generated",
    ) -> float:
        """计算两组图像之间的 FID 分数

        完整的 FID 计算流程：提取特征 → 计算统计量 → 计算 Fréchet 距离。

        Args:
            real_images: 真实图像张量 (N, C, H, W)，值域 [0, 1]
            generated_images: 生成图像张量 (M, C, H, W)，值域 [0, 1]
            real_cache_key: 真实图像特征的缓存键（可选）
            gen_cache_key: 生成图像特征的缓存键（可选）

        Returns:
            FID 分数 (浮点数，越低越好)

        Raises:
            ValueError: 如果输入不是有效的图像张量

        Example:
            >>> fid = calculator.calculate_fid(real_images, gen_images)
            >>> print(f"FID Score: {fid:.2f}")
        """
        logger.info(
            f"开始计算 FID: real={len(real_images)} 张, "
            f"generated={len(generated_images)} 张"
        )

        # 计算真实图像统计量
        if self.use_cache and real_cache_key is not None:
            if real_cache_key not in self.feature_cache:
                mean_real, cov_real = self.compute_statistics(real_images)
                self.feature_cache[real_cache_key] = (mean_real, cov_real)
            else:
                mean_real, cov_real = self.feature_cache[real_cache_key]
                logger.info(f"使用缓存的真实图像统计量: {real_cache_key}")
        else:
            mean_real, cov_real = self.compute_statistics(real_images)

        # 计算生成图像统计量
        if self.use_cache and gen_cache_key is not None:
            if gen_cache_key not in self.feature_cache:
                mean_gen, cov_gen = self.compute_statistics(generated_images)
                self.feature_cache[gen_cache_key] = (mean_gen, cov_gen)
            else:
                mean_gen, cov_gen = self.feature_cache[gen_cache_key]
                logger.info(f"使用缓存的生成图像统计量: {gen_cache_key}")
        else:
            mean_gen, cov_gen = self.compute_statistics(generated_images)

        # 计算 FID
        fid = self._compute_frechet_distance(mean_real, cov_real, mean_gen, cov_gen)

        logger.info(f"FID 计算完成: {fid:.4f}")

        return fid

    def clear_cache(self) -> None:
        """清除所有缓存的特征"""
        self.feature_cache.clear()
        logger.info("特征缓存已清除")

    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息

        Returns:
            包含缓存统计信息的字典
        """
        return {
            "num_cached_items": len(self.feature_cache),
            "cached_keys": list(self.feature_cache.keys()),
            "use_cache": self.use_cache,
        }

    def __repr__(self) -> str:
        return (
            f"FIDCalculator(device='{self.device}', batch_size={self.batch_size}, "
            f"dims={self.dims}, cache={'启用' if self.use_cache else '禁用'})"
        )


def calculate_fid_quick(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 50,
) -> float:
    """快速计算 FID 的便捷函数

        一行代码计算 FID，适合快速评估场景。

        Args:
            real_images: 真实图像张量
            generated_images: 生成图像张量
            device: 计算设备
            batch_size: 批处理大小

        Returns:
            FID 分数

        Example:
            >>> fid = calculate_fid_quick(real_imgs, gen_imgs, device='cuda')
        """
    calculator = FIDCalculator(device=device, batch_size=batch_size)
    return calculator.calculate_fid(real_images, generated_images)
