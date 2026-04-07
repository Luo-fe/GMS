"""IS (Inception Score) 计算器

使用 InceptionV3 的分类输出计算生成图像的 Inception Score。
IS 衡量生成图像的质量（清晰度）和多样性，值越高越好。

核心算法:
    1. 使用 InceptionV3 获取条件分布 p(y|x)
    2. 计算边际分布 p(y) = (1/N) Σ p(y|x_i)
    3. IS = exp(E_x[KL(p(y|x) || p(y))])
    4. 使用多个 splits 估计方差

注意事项:
    - 样本数量建议 >= 50000 以获得稳定结果
    - 图像需要 resize 到 299x299
    - IS 的理论最大值取决于类别数（ImageNet 为 1000 类）

Example:
    >>> from gms.evaluation.metrics.is_score import ISCalculator
    >>> import torch
    >>>
    >>> calculator = ISCalculator(device='cuda', batch_size=50)
    >>> gen_images = torch.randn(50000, 3, 299, 299).clamp(0, 1)
    >>>
    >>> is_mean, is_std = calculator.calculate_is(gen_images, splits=10)
    >>> print(f"IS: {is_mean:.2f} ± {is_std:.2f}")
"""

from typing import Optional, Tuple
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

logger = logging.getLogger(__name__)


class ISCalculator:
    """Inception Score 计算器

    使用预训练的 InceptionV3 模型计算生成图像的 Inception Score。
    支持批量处理、GPU 加速、多 split 方差估计等功能。

    Attributes:
        device: 计算设备 ('cpu' 或 'cuda')
        batch_size: 批处理大小
        num_classes: InceptionV3 的输出类别数 (1000 for ImageNet)
        inception_model: 预训练的 InceptionV3 模型 (完整版本)

    Example:
        >>> calculator = ISCalculator(device='cuda', batch_size=64)
        >>> is_mean, is_std = calculator.calculate_is(generated_images, splits=10)
    """

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 50,
        num_classes: int = 1000,
        resize_to: int = 299,
    ) -> None:
        """初始化 IS 计算器

        Args:
            device: 计算设备，'cpu' 或 'cuda'
            batch_size: 批处理大小，用于避免内存溢出
            num_classes: 分类器的输出类别数 (默认 1000)
            resize_to: 将图像 resize 到的目标尺寸 (默认 299)

        Raises:
            ValueError: 如果 device 不是 'cpu' 或 'cuda'
            RuntimeError: 如果无法加载预训练模型
        """
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"device 必须是 'cpu' 或 'cuda', 得到 {device}")

        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.resize_to = resize_to

        logger.info(
            f"初始化 ISCalculator: device={device}, batch_size={batch_size}, "
            f"num_classes={num_classes}"
        )

        self._load_inception_model()

    def _load_inception_model(self) -> None:
        """加载预训练的 InceptionV3 模型用于分类

        加载完整的 InceptionV3 模型（包含最后的 softmax 层），
        用于获取类别概率分布 p(y|x)。
        """
        logger.info("加载预训练 InceptionV3 模型 (分类模式)...")

        try:
            inception = models.inception_v3(
                weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )
            inception.eval()

            self.inception_model = inception.to(self.device)

            logger.info("InceptionV3 模型加载成功")
        except Exception as e:
            raise RuntimeError(f"无法加载 InceptionV3 模型: {e}")

    def _preprocess_images(
        self, images: torch.Tensor
    ) -> torch.Tensor:
        """预处理图像以适应 InceptionV3 的输入要求

        Args:
            images: 输入图像张量 (N, C, H, W)，值域 [0, 1]

        Returns:
            预处理后的图像张量
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Resize 到目标尺寸
        if images.shape[2] != self.resize_to or images.shape[3] != self.resize_to:
            images = F.interpolate(
                images,
                size=(self.resize_to, self.resize_to),
                mode="bilinear",
                align_corners=False,
            )

        # ImageNet 归一化
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        images = normalize(images)

        return images

    @torch.no_grad()
    def _get_predictions_batch(
        self, images: torch.Tensor
    ) -> np.ndarray:
        """从一批图像中获取 InceptionV3 的预测概率

        Args:
            images: 预处理后的图像张量 (N, C, 299, 299)

        Returns:
            概率分布矩阵 (N, num_classes)，每行是一个概率分布
        """
        probs_list = []

        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size].to(self.device)

            # 获取 logits
            output = self.inception_model(batch)

            # InceptionV3 在训练模式下返回 tuple (logits, aux_logits)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            # 应用 softmax 获取概率
            probs = F.softmax(logits, dim=1)

            probs_list.append(probs.cpu().numpy())

        return np.concatenate(probs_list, axis=0)

    def get_predictions(
        self, images: torch.Tensor
    ) -> np.ndarray:
        """获取图像的 InceptionV3 预测概率分布

        Args:
            images: 输入图像张量 (N, C, H, W)，值域 [0, 1]

        Returns:
            概率矩阵 (N, num_classes)
        """
        logger.info(f"计算预测概率: {len(images)} 张图像")

        preprocessed = self._preprocess_images(images)
        predictions = self._get_predictions_batch(preprocessed)

        logger.info(f"预测完成: shape={predictions.shape}")

        return predictions

    def _calculate_kl_divergence(
        self, p: np.ndarray, q: np.ndarray, eps: float = 1e-10
    ) -> float:
        """计算 KL 散度 KL(p || q)

        使用 log-sum-exp 技巧实现数值稳定的 KL 散度计算。

        Args:
            p: 第一个概率分布
            q: 第二个概率分布
            eps: 小常数防止 log(0)

        Returns:
            KL 散度值
        """
        # 数值稳定性：避免 log(0)
        p = np.clip(p, eps, 1.0)
        q = np.clip(q, eps, 1.0)

        kl_div = np.sum(p * np.log(p / q))

        return float(kl_div)

    def calculate_is_from_features(
        self,
        predictions: np.ndarray,
        splits: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """从预计算的预测概率计算 IS

        当已经获得了 InceptionV3 的预测概率时，
        可以直接使用此方法计算 IS，避免重复推理。

        Args:
            predictions: 预测概率矩阵 (N, num_classes)
            splits: 将数据分成多少组来估计方差 (默认 10)
            seed: 随机种子，用于可复现性

        Returns:
            Tuple 包含:
                - is_mean: IS 的均值
                - is_std: IS 的标准差

        Raises:
            ValueError: 如果样本数不足或 splits 参数非法

        Example:
            >>> preds = calculator.get_predictions(generated_images)
            >>> is_mean, is_std = calculator.calculate_is_from_features(preds, splits=10)
        """
        if len(predictions) < splits:
            raise ValueError(
                f"样本数量 ({len(predictions)}) 必须大于等于 splits ({splits})"
            )

        if splits <= 0:
            raise ValueError(f"splits 必须为正整数，得到 {splits}")

        logger.info(f"从预计算特征计算 IS: {len(predictions)} 个样本, {splits} 个 splits")

        # 设置随机种子以确保可复现性
        if seed is not None:
            np.random.seed(seed)

        scores = []
        n_samples = len(predictions)

        # 将数据分成多个 splits 并分别计算 IS
        split_size = n_samples // splits

        for i in range(splits):
            start_idx = i * split_size
            end_idx = (
                (i + 1) * split_size if i < splits - 1 else n_samples
            )

            # 获取当前 split 的数据
            split_preds = predictions[start_idx:end_idx]

            # 计算边际分布 p(y) = (1/m) * Σ p(y|x_i)
            marginal = np.mean(split_preds, axis=0)

            # 对每个样本计算 KL(p(y|x) || p(y))
            kl_divergences = [
                self._calculate_kl_divergence(pred, marginal)
                for pred in split_preds
            ]

            # 计算 IS = exp(E[KL])
            is_score = float(np.exp(np.mean(kl_divergences)))
            scores.append(is_score)

        is_mean = float(np.mean(scores))
        is_std = float(np.std(scores))

        logger.info(f"IS 计算完成: {is_mean:.4f} ± {is_std:.4f}")

        return is_mean, is_std

    def calculate_is(
        self,
        generated_images: torch.Tensor,
        splits: int = 10,
        seed: Optional[int] = 42,
    ) -> Tuple[float, float]:
        """计算生成图像的 Inception Score

        完整的 IS 计算流程：
        1. 使用 InceptionV3 获取预测概率 p(y|x)
        2. 计算边际分布 p(y)
        3. 计算 KL 散度和 IS
        4. 多 splits 估计方差

        Args:
            generated_images: 生成图像张量 (N, C, H, W)，值域 [0, 1]
            splits: 分割数量，用于估计方差 (默认 10)
            seed: 随机种子 (默认 42)

        Returns:
            Tuple 包含:
                - is_mean: IS 的均值 (越高越好)
                - is_std: IS 的标准差

        Raises:
            ValueError: 如果输入不是有效的图像张量或参数非法

        Example:
            >>> is_mean, is_std = calculator.calculate_is(gen_images, splits=10)
            >>> print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
        """
        logger.info(
            f"开始计算 IS: {len(generated_images)} 张图像, "
            f"{splits} 个 splits"
        )

        # 获取预测概率
        predictions = self.get_predictions(generated_images)

        # 从预测概率计算 IS
        is_mean, is_std = self.calculate_is_from_features(
            predictions, splits=splits, seed=seed
        )

        return is_mean, is_std

    def __repr__(self) -> str:
        return (
            f"ISCalculator(device='{self.device}', batch_size={self.batch_size}, "
            f"num_classes={self.num_classes})"
        )


def calculate_is_quick(
    generated_images: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 50,
    splits: int = 10,
) -> Tuple[float, float]:
    """快速计算 IS 的便捷函数

    一行代码计算 IS，适合快速评估场景。

    Args:
        generated_images: 生成图像张量
        device: 计算设备
        batch_size: 批处理大小
        splits: 分割数量

    Returns:
        Tuple (is_mean, is_std)

    Example:
        >>> is_mean, is_std = calculate_is_quick(gen_imgs, device='cuda')
    """
    calculator = ISCalculator(device=device, batch_size=batch_size)
    return calculator.calculate_is(generated_images, splits=splits)
