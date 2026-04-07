"""双分量高斯混合模型概率密度函数

实现GMM的核心概率计算功能，包括:
- 概率密度函数 (PDF)
- 对数概率密度函数 (Log-PDF)
- 后验概率计算
- 从模型采样

使用数值稳定的算法（log-sum-exp、Cholesky分解），
支持GPU加速和PyTorch自动微分。

Example:
    >>> import torch
    >>> from gms.gmm_optimization import GMMParameters, GaussianMixtureModel
    >>>
    >>> # 创建GMM参数和模型
    >>> params = GMMParameters(
    ...     weight=0.6,
    ...     mean1=torch.tensor([1.0, 2.0]),
    ...     mean2=torch.tensor([-1.0, -2.0]),
    ...     variance1=torch.tensor([0.5, 0.5]),
    ...     variance2=torch.tensor([1.0, 1.0])
    ... )
    >>> model = GaussianMixtureModel(params)
    >>>
    >>> # 计算PDF
    >>> x = torch.tensor([0.5, 0.5])
    >>> pdf_value = model.pdf(x)
    >>> print(f"PDF值: {pdf_value:.6f}")
    >>>
    >>> # 计算对数PDF（更稳定）
    >>> log_pdf = model.log_pdf(x)
    >>>
    >>> # 批量计算
    >>> X = torch.randn(100, 2)
    >>> pdfs = model.pdf(X)  # 形状: (100,)
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F
from .gmm_parameters import GMMParameters, GMMParametersConfig
import logging

logger = logging.getLogger(__name__)


class GaussianMixtureModel:
    """双分量高斯混合模型

    实现完整的GMM概率计算功能，使用数值稳定的算法。

    数学定义:
        p(x) = w · N(x|μ₁, Σ₁) + (1-w) · N(x|μ₂, Σ₂)

    其中:
        N(x|μ, Σ) = (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (x-μ)^T Σ^(-1) (x-μ))

    Attributes:
        params: GMMParameters实例，存储模型参数

    Example:
        >>> model = GaussianMixtureModel(gmm_params)
        >>> pdf_val = model.pdf(torch.randn(5))
        >>> samples = model.sample(1000)
    """

    def __init__(self, params: GMMParameters):
        """初始化GMM模型

        Args:
            params: GMMParameters参数对象

        Raises:
            ValueError: 如果参数验证失败
        """
        self.params = params
        self._log_det_cache: dict = {}
        self._inv_cache: dict = {}

        logger.debug(
            f"GaussianMixtureModel初始化完成: "
            f"d={params.dimensionality}, diag={params.is_diagonal}"
        )

    @property
    def dimensionality(self) -> int:
        """获取特征维度"""
        return self.params.dimensionality

    @property
    def n_components(self) -> int:
        """获取分量数量（固定为2）"""
        return 2

    def pdf(self, x: torch.Tensor) -> Union[float, torch.Tensor]:
        """计算概率密度函数

        计算 p(x) = w·N(x|μ₁,Σ₁) + (1-w)·N(x|μ₂,Σ₂)

        使用log-sum-exp技巧确保数值稳定性。

        Args:
            x: 输入数据点
               - 单点: 形状 (d,)
               - 批量: 形状 (n, d)

        Returns:
            PDF值:
               - 单点输入: 标量float
               - 批量输入: 形状 (n,) 的张量

        Example:
            >>> # 单点计算
            >>> pdf_val = model.pdf(torch.tensor([1.0, 2.0]))
            >>>
            >>> # 批量计算
            >>> X = torch.randn(100, 3)
            >>> pdf_vals = model.pdf(X)  # 形状: (100,)
        """
        with torch.no_grad():
            log_probs = self._compute_mixture_log_pdf(x)
            probs = torch.exp(log_probs)

            if x.dim() == 1:
                return float(probs.item())
            return probs

    def log_pdf(self, x: torch.Tensor) -> Union[float, torch.Tensor]:
        """计算对数概率密度函数

        比 pdf() 更数值稳定，推荐用于:
        - 概率比较
        - 似然计算
        - 损失函数

        Args:
            x: 输入数据点（同pdf()）

        Returns:
            对数PDF值

        Example:
            >>> log_p = model.log_pdf(torch.tensor([1.0, 2.0]))
            >>> # 等价于但更稳定
            >>> # log(model.pdf(x))
        """
        with torch.no_grad():
            log_probs = self._compute_mixture_log_pdf(x)

            if x.dim() == 1:
                return float(log_probs.item())
            return log_probs

    def component_pdf(
        self,
        x: torch.Tensor,
        component_id: int
    ) -> Union[float, torch.Tensor]:
        """计算单个分量的PDF

        计算指定高斯分量的概率密度。

        Args:
            x: 输入数据点
            component_id: 分量ID (0 或 1)

        Returns:
            该分量的PDF值

        Raises:
            ValueError: 如果component_id不是0或1

        Example:
            >>> p_component1 = model.component_pdf(x, component_id=0)
            >>> p_component2 = model.component_pdf(x, component_id=1)
        """
        if component_id not in [0, 1]:
            raise ValueError(f"component_id必须是0或1，当前值: {component_id}")

        if component_id == 0:
            mean = self.params.mean1
            cov = self.params.variance1
        else:
            mean = self.params.mean2
            cov = self.params.variance2

        with torch.no_grad():
            log_prob = self._log_gaussian_pdf(x, mean, cov)
            prob = torch.exp(log_prob)

            if x.dim() == 1:
                return float(prob.item())
            return prob

    def posterior_probability(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """计算后验概率 P(component|x)

        使用贝叶斯规则计算每个分量在给定观测x下的后验概率:

            P(k|x) = w_k · N(x|μ_k, Σ_k) / Σ_j w_j · N(x|μ_j, Σ_j)

        用于:
        - 软聚类分配
        - EM算法的E步骤
        - 责任度计算

        Args:
            x: 输入数据点
               - 单点: 形状 (d,)
               - 批量: 形状 (n, d)

        Returns:
            后验概率:
               - 单点: 形状 (2,) 的张量 [P(0|x), P(1|x)]
               - 批量: 形状 (n, 2) 的张量

        Example:
            >>> posteriors = model.posterior_probability(torch.tensor([1.0, 2.0]))
            >>> print(f"P(分量1|x)={posteriors[0]:.4f}, P(分量2|x)={posteriors[1]:.4f}")
        """
        weights = torch.tensor([self.params.weight, self.params.weight2],
                              device=x.device, dtype=x.dtype)

        log_weights = torch.log(weights.clamp(min=1e-10))

        log_comp0 = log_weights[0] + self._log_gaussian_pdf(x, self.params.mean1, self.params.variance1)
        log_comp1 = log_weights[1] + self._log_gaussian_pdf(x, self.params.mean2, self.params.variance2)

        log_probs_stack = torch.stack([log_comp0, log_comp1], dim=-1)

        log_sum_exp = self._logsumexp(log_probs_stack, dim=-1)

        posteriors = torch.exp(log_probs_stack - log_sum_exp.unsqueeze(-1))

        if x.dim() == 1:
            return posteriors.squeeze()
        return posteriors

    def sample(
        self,
        n_samples: int,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """从模型中采样

        使用以下过程生成样本:
        1. 根据混合权重选择分量
        2. 从选定的高斯分布中采样

        Args:
            n_samples: 采样数量
            seed: 随机种子（用于可复现性）

        Returns:
            采样结果，形状 (n_samples, d)

        Example:
            >>> samples = model.sample(1000, seed=42)
            >>> print(f"样本形状: {samples.shape}")  # (1000, d)
        """
        if seed is not None:
            torch.manual_seed(seed)

        device = self.params.mean1.device
        dtype = self.params.mean1.dtype

        weights_tensor = torch.tensor([self.params.weight, self.params.weight2],
                                     device=device, dtype=dtype)

        component_indices = torch.multinomial(
            weights_tensor.unsqueeze(0),
            n_samples,
            replacement=True
        ).squeeze()

        samples = torch.zeros(n_samples, self.dimensionality, device=device, dtype=dtype)

        mask0 = (component_indices == 0)
        mask1 = (component_indices == 1)

        n0 = mask0.sum().item()
        n1 = mask1.sum().item()

        if n0 > 0:
            samples[mask0] = self._sample_from_gaussian(
                self.params.mean1, self.params.variance1, int(n0)
            )
        if n1 > 0:
            samples[mask1] = self._sample_from_gaussian(
                self.params.mean2, self.params.variance2, int(n1)
            )

        logger.debug(f"已生成 {n_samples} 个样本")
        return samples

    def _compute_mixture_log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """计算混合模型的log PDF（内部方法）

        使用log-sum-exp技巧确保数值稳定性。

        Args:
            x: 输入数据

        Returns:
            log PDF值
        """
        weights = torch.tensor([self.params.weight, self.params.weight2],
                              device=x.device, dtype=x.dtype)

        log_weights = torch.log(weights.clamp(min=1e-10))

        log_comp0 = log_weights[0] + self._log_gaussian_pdf(x, self.params.mean1, self.params.variance1)
        log_comp1 = log_weights[1] + self._log_gaussian_pdf(x, self.params.mean2, self.params.variance2)

        log_probs = torch.stack([log_comp0, log_comp1], dim=-1)

        return self._logsumexp(log_probs, dim=-1)

    def _log_gaussian_pdf(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        cov: torch.Tensor
    ) -> torch.Tensor:
        """计算多元高斯分布的对数PDF（核心方法）

        使用Cholesky分解确保数值稳定性。

        数学公式:
            log N(x|μ, Σ) = -0.5 * [d*log(2π) + log|Σ| + (x-μ)^T Σ^(-1) (x-μ)]

        Args:
            x: 输入数据，形状 (d,) 或 (n, d)
            mean: 均值向量，形状 (d,)
            cov: 协方差矩阵，形状 (d,) 或 (d, d)

        Returns:
            对数PDF值（标量或形状为(n,)的张量）
        """
        d = mean.shape[-1]
        is_single_point = (x.dim() == 1)

        if is_single_point:
            diff = (x - mean).unsqueeze(0)  # (1, d)
        else:
            diff = x - mean.unsqueeze(0)    # (n, d)

        if cov.dim() == 1:
            log_det = torch.log(cov).sum()
            inv_cov = 1.0 / cov
            mahalanobis = (diff ** 2 * inv_cov).sum(dim=-1)  # (1,) or (n,)
        else:
            try:
                L = torch.linalg.cholesky(cov)
                log_det = 2.0 * torch.log(torch.diag(L)).sum()

                solve = torch.linalg.solve_triangular(
                    L, diff.transpose(-1, -2),  # (d, 1) or (d, n)
                    upper=False
                )
                mahalanobis = (solve ** 2).sum(dim=0)  # (1,) or (n,)

            except RuntimeError as e:
                logger.warning(f"Cholesky分解失败，使用特征值分解: {e}")

                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = eigenvalues.clamp(min=1e-6)
                log_det = torch.log(eigenvalues).sum()

                scaled_diff = diff @ eigenvectors  # (1, d) or (n, d)
                mahalanobis = (scaled_diff ** 2 * eigenvalues.unsqueeze(0)).sum(dim=-1)

        log_2pi = math.log(2.0 * math.pi)
        log_norm = -0.5 * (d * log_2pi + log_det + mahalanobis)

        if is_single_point:
            return log_norm.squeeze(0)  # 返回标量 (d,)
        return log_norm

    def _gaussian_pdf(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        cov: torch.Tensor
    ) -> torch.Tensor:
        """计算多元高斯分布的PDF

        通过exp(log_pdf)实现，但可能不如直接使用_log_gaussian_pdf稳定。

        Args:
            x: 输入数据
            mean: 均值向量
            cov: 协方差矩阵

        Returns:
            PDF值
        """
        return torch.exp(self._log_gaussian_pdf(x, mean, cov))

    def _sample_from_gaussian(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """从单个高斯分布中采样

        Args:
            mean: 均值向量
            cov: 协方差矩阵
            n_samples: 采样数量

        Returns:
            采样结果，形状 (n_samples, d)
        """
        d = mean.shape[-1]

        z = torch.randn(n_samples, d, device=mean.device, dtype=mean.dtype)

        if cov.dim() == 1:
            std = torch.sqrt(cov)
            samples = mean.unsqueeze(0) + z * std.unsqueeze(0)
        else:
            try:
                L = torch.linalg.cholesky(cov)
                samples = mean.unsqueeze(0) + z @ L.T
            except RuntimeError:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)
                eigenvalues = eigenvalues.clamp(min=1e-6)
                sqrt_eigenvalues = torch.sqrt(eigenvalues)
                transform = eigenvectors @ torch.diag(sqrt_eigenvalues)
                samples = mean.unsqueeze(0) + z @ transform.T

        return samples

    @staticmethod
    def _logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """数值稳定的log-sum-exp实现

        防止溢出/下溢问题。

        公式: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))

        Args:
            x: 输入张量
            dim: 求和的维度

        Returns:
            log-sum-exp结果
        """
        max_x, _ = x.max(dim=dim, keepdim=True)
        stable_x = x - max_x

        result = max_x.squeeze(dim) + torch.log(torch.exp(stable_x).sum(dim=dim))

        return result

    def compute_statistics(self) -> dict:
        """计算模型的统计特性

        计算并返回混合分布的各种统计量。

        Returns:
            包含以下键的字典:
            - 'mean': 混合均值
            - 'covariance': 混合协方差
            - 'entropy': 分布熵
            - 'mode': 众数估计
        """
        w1, w2 = self.params.weight, self.params.weight2
        mu1, mu2 = self.params.mean1, self.params.mean2
        s1, s2 = self.params.variance1, self.params.variance2

        mixed_mean = w1 * mu1 + w2 * mu2

        if self.params.is_diagonal:
            diff1 = (mu1 - mixed_mean) ** 2
            diff2 = (mu2 - mixed_mean) ** 2
            mixed_cov = w1 * (s1 + diff1) + w2 * (s2 + diff2)
        else:
            diff1 = (mu1 - mixed_mean).unsqueeze(1)
            diff2 = (mu2 - mixed_mean).unsqueeze(1)
            mixed_cov = w1 * (s1 + diff1 @ diff1.T) + w2 * (s2 + diff2 @ diff2.T)

        X_sample = self.sample(10000, seed=42)
        log_probs = self._compute_mixture_log_pdf(X_sample)
        entropy = -torch.mean(log_probs).item()

        return {
            'mean': mixed_mean.detach(),
            'covariance': mixed_cov.detach(),
            'entropy': entropy,
            'mode': None,
        }

    def __repr__(self) -> str:
        """返回模型的字符串表示"""
        return (
            f"GaussianMixtureModel("
            f"params={self.params})"
        )


def compute_kl_divergence(
    model_p: GaussianMixtureModel,
    model_q: GaussianMixtureModel,
    n_samples: int = 10000
) -> float:
    """计算两个GMM之间的KL散度

    KL(P || Q) = E_{x~P}[log P(x) - log Q(x)]

    使用蒙特卡洛近似。

    Args:
        model_p: 分布P的GMM
        model_q: 分布Q的GMM
        n_samples: 采样数量

    Returns:
        KL散度值（标量）
    """
    samples = model_p.sample(n_samples, seed=42)

    with torch.no_grad():
        log_p = model_p._compute_mixture_log_pdf(samples)
        log_q = model_q._compute_mixture_log_pdf(samples)

        kl_div = (log_p - log_q).mean().item()

    return kl_div


def compute_js_divergence(
    model_p: GaussianMixtureModel,
    model_q: GaussianMixtureModel,
    n_samples: int = 10000
) -> float:
    """计算两个GMM之间的Jensen-Shannon散度

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)

    其中 M = 0.5 * (P + Q)

    Args:
        model_p: 分布P的GMM
        model_q: 分布Q的GMM
        n_samples: 采样数量

    Returns:
        JS散度值（标量），范围[0, ln(2)]
    """
    from .gmm_parameters import GMMParameters

    w_mid = 0.5 * (model_p.params.weight + model_q.params.weight)
    mid_params = GMMParameters(
        weight=w_mid,
        mean1=0.5 * (model_p.params.mean1 + model_q.params.mean1),
        mean2=0.5 * (model_p.params.mean2 + model_q.params.mean2),
        variance1=0.5 * (model_p.params.variance1 + model_q.params.variance1),
        variance2=0.5 * (model_p.params.variance2 + model_q.params.variance2),
    )
    model_m = GaussianMixtureModel(mid_params)

    kl_pm = compute_kl_divergence(model_p, model_m, n_samples)
    kl_qm = compute_kl_divergence(model_q, model_m, n_samples)

    js_div = 0.5 * (kl_pm + kl_qm)

    return js_div


def compute_wasserstein_distance(
    model_p: GaussianMixtureModel,
    model_q: GaussianMixtureModel
) -> float:
    """计算两个GMM之间的Wasserstein距离（简化版）

    对于单变量情况精确计算，对于多变量情况使用分量间距离的加权平均。

    Args:
        model_p: 分布P的GMM
        model_q: 分布Q的GMM

    Returns:
        Wasserstein距离近似值
    """
    if model_p.dimensionality == 1:
        w_p = [model_p.params.weight, model_p.params.weight2]
        w_q = [model_q.params.weight, model_q.params.weight2]

        means_p = [model_p.params.mean1.item(), model_p.params.mean2.item()]
        means_q = [model_q.params.mean1.item(), model_q.params.mean2.item()]

        var_p = [model_p.params.variance1.item(), model_p.params.variance2.item()]
        var_q = [model_q.params.variance1.item(), model_q.params.variance2.item()]

        def _wass_1d(mu1, sigma1, mu2, sigma2):
            return abs(mu1 - mu2) + (math.sqrt(sigma1) - math.sqrt(sigma2)) ** 2

        total_dist = 0.0
        for i in range(2):
            for j in range(2):
                total_dist += w_p[i] * w_q[j] * _wass_1d(
                    means_p[i], var_p[i], means_q[j], var_q[j]
                )

        return total_dist
    else:
        warnings.warn("多变量Wasserstein距离使用简化近似")

        dist_11 = torch.norm(model_p.params.mean1 - model_q.params.mean1).item()
        dist_12 = torch.norm(model_p.params.mean1 - model_q.params.mean2).item()
        dist_21 = torch.norm(model_p.params.mean2 - model_q.params.mean1).item()
        dist_22 = torch.norm(model_p.params.mean2 - model_q.params.mean2).item()

        w_p = model_p.params.weight
        w_q = model_q.params.weight

        return (
            w_p * w_q * dist_11 +
            w_p * (1-w_q) * dist_12 +
            (1-w_p) * w_q * dist_21 +
            (1-w_p) * (1-w_q) * dist_22
        )
