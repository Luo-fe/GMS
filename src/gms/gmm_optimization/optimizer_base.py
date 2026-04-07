"""GMM优化器基础框架 - 定义优化算法的抽象接口和配置"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import time
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """优化配置数据类

    存储所有优化相关的超参数和设置。

    Attributes:
        learning_rate: 初始学习率
        max_iterations: 最大迭代次数
        convergence_threshold: 收敛阈值（损失变化小于此值时停止）
        regularization_coefficient: L2正则化系数
        early_stopping_patience: 早停耐心值（连续多少次无改善则停止）
        gradient_clip_norm: 梯度裁剪的最大范数（0表示不裁剪）
        momentum: 动量系数（用于动量优化器）
        weight_decay: 权重衰减系数
        min_lr: 最小学习率
        verbose: 是否打印详细日志
        device: 计算设备 ('cpu' 或 'cuda')
        dtype: 数据类型 (torch.float32 或 torch.float64)
        random_seed: 随机种子（用于可复现性）
        checkpoint_interval: 保存检查点的间隔（迭代次数）

    Example:
        >>> config = OptimizationConfig(
        ...     learning_rate=0.01,
        ...     max_iterations=1000,
        ...     convergence_threshold=1e-6
        ... )
    """

    learning_rate: float = 0.01
    max_iterations: int = 1000
    convergence_threshold: float = 1e-6
    regularization_coefficient: float = 1e-4
    early_stopping_patience: int = 50
    gradient_clip_norm: float = 1.0
    momentum: float = 0.9
    weight_decay: float = 0.0
    min_lr: float = 1e-7
    verbose: bool = True
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    random_seed: Optional[int] = None
    checkpoint_interval: int = 100

    def __post_init__(self):
        """验证配置参数"""
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate必须为正数，当前值: {self.learning_rate}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations必须为正整数，当前值: {self.max_iterations}")
        if self.convergence_threshold < 0:
            raise ValueError(f"convergence_threshold不能为负，当前值: {self.convergence_threshold}")
        if self.regularization_coefficient < 0:
            raise ValueError(f"regularization_coefficient不能为负，当前值: {self.regularization_coefficient}")
        if self.early_stopping_patience < 0:
            raise ValueError(f"early_stopping_patience不能为负，当前值: {self.early_stopping_patience}")
        if self.gradient_clip_norm < 0:
            raise ValueError(f"gradient_clip_norm不能为负，当前值: {self.gradient_clip_norm}")
        if not 0 <= self.momentum <= 1:
            raise ValueError(f"momentum必须在[0,1]范围内，当前值: {self.momentum}")


@dataclass
class OptimizedParams:
    """优化后的参数结果数据类

    存储GMM优化完成后的所有参数和元信息。

    Attributes:
        means: 各分量的均值向量，形状 (n_components, n_features)
        covariances: 各分量的协方差矩阵，形状 (n_components, n_features, n_features) 或 (n_components, n_features)
        weights: 混合权重，形状 (n_components,)
        converged: 是否收敛
        n_iterations: 实际迭代次数
        final_loss: 最终损失值
        loss_history: 损失历史记录列表
        optimization_time: 优化耗时（秒）
        metadata: 额外的元数据信息

    Example:
        >>> params = OptimizedParams(
        ...     means=torch.randn(2, 3),
        ...     covariances=torch.eye(3).unsqueeze(0).repeat(2, 1, 1),
        ...     weights=torch.tensor([0.5, 0.5]),
        ...     converged=True,
        ...     final_loss=0.001
        ... )
    """

    means: Optional[torch.Tensor] = None
    covariances: Optional[torch.Tensor] = None
    weights: Optional[torch.Tensor] = None
    converged: bool = False
    n_iterations: int = 0
    final_loss: float = float('inf')
    loss_history: List[float] = field(default_factory=list)
    optimization_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_components(self) -> int:
        """获取分量数量"""
        if self.weights is not None:
            return len(self.weights)
        return 0

    @property
    def n_features(self) -> int:
        """获取特征维度"""
        if self.means is not None:
            return self.means.shape[-1]
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """将结果转换为字典格式

        Returns:
            包含所有结果的字典
        """
        result_dict = {
            "means": self.means.cpu().numpy() if self.means is not None else None,
            "covariances": self.covariances.cpu().numpy() if self.covariances is not None else None,
            "weights": self.weights.cpu().numpy() if self.weights is not None else None,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "final_loss": self.final_loss,
            "loss_history": self.loss_history,
            "optimization_time": self.optimization_time,
            "metadata": self.metadata,
        }
        return result_dict

    def to_device(self, device: torch.device) -> "OptimizedParams":
        """将所有张量移动到指定设备

        Args:
            device: 目标设备

        Returns:
            新的OptimizedParams实例
        """
        new_params = OptimizedParams(
            means=self.means.to(device) if self.means is not None else None,
            covariances=self.covariances.to(device) if self.covariances is not None else None,
            weights=self.weights.to(device) if self.weights is not None else None,
            converged=self.converged,
            n_iterations=self.n_iterations,
            final_loss=self.final_loss,
            loss_history=self.loss_history.copy(),
            optimization_time=self.optimization_time,
            metadata=self.metadata.copy(),
        )
        return new_params


@dataclass
class TargetMoments:
    """目标矩数据类

    存储用于优化的目标矩（从MomentEstimator获得）。

    Attributes:
        mean: 目标均值 μ
        covariance: 目标协方差矩阵 Σ
        skewness: 目标偏度 γ
        higher_moments: 更高阶矩（可选）

    Example:
        >>> target = TargetMoments(
        ...     mean=torch.zeros(10),
        ...     covariance=torch.eye(10),
        ...     skewness=torch.zeros(10)
        ... )
    """

    mean: Optional[torch.Tensor] = None
    covariance: Optional[torch.Tensor] = None
    skewness: Optional[torch.Tensor] = None
    higher_moments: Dict[str, torch.Tensor] = field(default_factory=dict)


class EpochCallbackData:
    """回调数据容器

    用于在每次迭代结束时传递给回调函数的数据。

    Attributes:
        epoch: 当前迭代次数
        loss: 当前损失值
        params: 当前参数（字典形式）
        gradients: 当前梯度信息（可选）
        learning_rate: 当前学习率
        elapsed_time: 已用时间
        metrics: 其他指标
    """

    def __init__(
        self,
        epoch: int,
        loss: float,
        params: Dict[str, torch.Tensor],
        gradients: Optional[Dict[str, torch.Tensor]] = None,
        learning_rate: float = 0.01,
        elapsed_time: float = 0.0,
        metrics: Optional[Dict[str, float]] = None,
    ):
        self.epoch = epoch
        self.loss = loss
        self.params = params
        self.gradients = gradients
        self.learning_rate = learning_rate
        self.elapsed_time = elapsed_time
        self.metrics = metrics or {}


class BaseGMMOptimizer(ABC):
    """GMM优化器抽象基类

    定义了所有GMM优化算法必须实现的接口。
    提供通用的优化循环、回调机制和日志记录功能。

    子类需要实现:
    - _compute_loss(): 计算损失函数
    - _update_params(): 参数更新逻辑

    Attributes:
        config: 优化配置
        _callbacks: 回调函数列表
        _device: 计算设备

    Example:
        >>> class MyOptimizer(BaseGMMOptimizer):
        ...     def _compute_loss(self, params, target_moments):
        ...         # 实现损失计算
        ...         pass
        ...     def _update_params(self, params, gradients, lr):
        ...         # 实现参数更新
        ...         pass
        >>>
        >>> optimizer = MyOptimizer(OptimizationConfig())
        >>> result = optimizer.optimize(target_moments, initial_params)
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """初始化优化器

        Args:
            config: 优化配置，如果为None则使用默认配置
        """
        self.config = config or OptimizationConfig()
        self._callbacks: Dict[str, List[Callable]] = {
            "on_epoch_end": [],
            "on_convergence": [],
            "on_early_stop": [],
            "on_start": [],
        }

        self._device = torch.device(self.config.device)
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)

        logger.info(f"{self.__class__.__name__} 初始化完成")
        logger.debug(f"优化配置: {self.config}")

    @abstractmethod
    def _compute_loss(
        self,
        params: Dict[str, torch.Tensor],
        target_moments: TargetMoments,
    ) -> torch.Tensor:
        """计算损失函数（子类实现）

        Args:
            params: 当前参数字典
            target_moments: 目标矩

        Returns:
            损失值张量（标量）
        """
        pass

    @abstractmethod
    def _update_params(
        self,
        params: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor],
        learning_rate: float,
    ) -> Dict[str, torch.Tensor]:
        """更新参数（子类实现）

        Args:
            params: 当前参数字典
            gradients: 梯度字典
            learning_rate: 当前学习率

        Returns:
            更新后的参数字典
        """
        pass

    def add_callback(
        self,
        event: str,
        callback: Callable[[EpochCallbackData], None],
    ) -> None:
        """添加回调函数

        Args:
            event: 事件类型 ('on_epoch_end', 'on_convergence', 'on_early_stop', 'on_start')
            callback: 回调函数，接收EpochCallbackData作为参数

        Raises:
            ValueError: 如果事件类型无效
        """
        valid_events = ["on_epoch_end", "on_convergence", "on_early_stop", "on_start"]
        if event not in valid_events:
            raise ValueError(f"无效的事件类型: {event}。支持的事件: {valid_events}")

        self._callbacks[event].append(callback)
        logger.debug(f"已添加 '{event}' 回调函数")

    def remove_callback(
        self,
        event: str,
        callback: Callable[[EpochCallbackData], None],
    ) -> bool:
        """移除回调函数

        Args:
            event: 事件类型
            callback: 要移除的回调函数

        Returns:
            是否成功移除
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            logger.debug(f"已移除 '{event}' 回调函数")
            return True
        return False

    def _trigger_callbacks(self, event: str, data: EpochCallbackData) -> None:
        """触发指定事件的回调函数

        Args:
            event: 事件类型
            data: 回调数据
        """
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"回调函数执行失败 ({event}): {e}")

    def optimize(
        self,
        target_moments: TargetMoments,
        initial_params: Dict[str, torch.Tensor],
        loss_fn: Optional[Any] = None,
    ) -> OptimizedParams:
        """执行优化过程

        这是主要的优化入口方法。执行完整的优化循环，
        包括损失计算、梯度计算、参数更新、早停检查等。

        Args:
            target_moments: 目标矩对象
            initial_params: 初始参数字典，包含:
                           - 'means': 初始均值 (n_components, n_features)
                           - 'covariances': 初始协方差
                           - 'weights': 初始权重 (n_components,)
            loss_fn: 可选的自定义损失函数（如果提供则覆盖默认的_compute_loss）

        Returns:
            OptimizedParams 对象，包含优化结果

        Raises:
            ValueError: 如果初始参数无效
            RuntimeError: 如果优化过程中出现数值不稳定
        """
        start_time = time.time()

        self._validate_initial_params(initial_params)

        params = {
            k: v.clone().detach().to(self._device).requires_grad_(True)
            for k, v in initial_params.items()
        }

        best_loss = float('inf')
        best_params = None
        patience_counter = 0
        loss_history = []
        current_lr = self.config.learning_rate

        start_data = EpochCallbackData(
            epoch=0,
            loss=float('inf'),
            params={k: v.detach() for k, v in params.items()},
            learning_rate=current_lr,
        )
        self._trigger_callbacks("on_start", start_data)

        if self.config.verbose:
            logger.info(f"开始优化，最大迭代次数: {self.config.max_iterations}")

        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()

            try:
                total_loss = self._optimization_step(params, target_moments, loss_fn)

                with torch.no_grad():
                    loss_value = total_loss.item()

                loss_history.append(loss_value)
                elapsed = time.time() - iter_start

                gradients = {}
                if total_loss.requires_grad and total_loss.grad_fn is not None:
                    for name, param in params.items():
                        if param.grad is not None:
                            gradients[name] = param.grad.clone()

                callback_data = EpochCallbackData(
                    epoch=iteration,
                    loss=loss_value,
                    params={k: v.detach() for k, v in params.items()},
                    gradients=gradients if gradients else None,
                    learning_rate=current_lr,
                    elapsed_time=elapsed,
                )
                self._trigger_callbacks("on_epoch_end", callback_data)

                if loss_value < best_loss - self.config.convergence_threshold:
                    best_loss = loss_value
                    best_params = {k: v.detach().clone() for k, v in params.items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.config.verbose and iteration % 10 == 0:
                    logger.info(
                        f"Iteration {iteration}/{self.config.max_iterations}: "
                        f"loss={loss_value:.6f}, best={best_loss:.6f}, "
                        f"lr={current_lr:.2e}"
                    )

                if patience_counter >= self.config.early_stopping_patience:
                    if self.config.verbose:
                        logger.info(
                            f"早停触发于第 {iteration} 次迭代 "
                            f"(patience={self.config.early_stopping_patience})"
                        )

                    stop_data = EpochCallbackData(
                        epoch=iteration,
                        loss=loss_value,
                        params={k: v.detach() for k, v in params.items()},
                        learning_rate=current_lr,
                    )
                    self._trigger_callbacks("on_early_stop", stop_data)
                    break

                if len(loss_history) >= 2:
                    loss_change = abs(loss_history[-2] - loss_history[-1])
                    if loss_change < self.config.convergence_threshold:
                        if self.config.verbose:
                            logger.info(
                                f"收敛于第 {iteration} 次迭代 "
                                f"(loss_change={loss_change:.2e})"
                            )

                        conv_data = EpochCallbackData(
                            epoch=iteration,
                            loss=loss_value,
                            params={k: v.detach() for k, v in params.items()},
                            learning_rate=current_lr,
                        )
                        self._trigger_callbacks("on_convergence", conv_data)
                        break

            except RuntimeError as e:
                if "numerical" in str(e).lower() or "inf" in str(e).lower() or "nan" in str(e).lower():
                    logger.error(f"数值不稳定于第 {iteration} 次迭代: {e}")
                    if best_params is not None:
                        params = best_params
                    break
                else:
                    raise

        optimization_time = time.time() - start_time

        if best_params is not None:
            final_means = best_params.get('means')
            final_covs = best_params.get('covariances')
            final_weights = best_params.get('weights')
        else:
            final_means = params['means'].detach() if 'means' in params else None
            final_covs = params['covariances'].detach() if 'covariances' in params else None
            final_weights = params['weights'].detach() if 'weights' in params else None

        converged = (
            patience_counter < self.config.early_stopping_patience and
            len(loss_history) > 0
        )

        result = OptimizedParams(
            means=final_means,
            covariances=final_covs,
            weights=final_weights,
            converged=converged,
            n_iterations=len(loss_history),
            final_loss=best_loss if best_loss != float('inf') else (loss_history[-1] if loss_history else float('inf')),
            loss_history=loss_history,
            optimization_time=optimization_time,
            metadata={
                "optimizer_type": self.__class__.__name__,
                "config": self.config,
                "final_learning_rate": current_lr,
            },
        )

        if self.config.verbose:
            logger.info(
                f"优化完成: converged={converged}, "
                f"iterations={result.n_iterations}, "
                f"final_loss={result.final_loss:.6f}, "
                f"time={optimization_time:.2f}s"
            )

        return result

    def _optimization_step(
        self,
        params: Dict[str, torch.Tensor],
        target_moments: TargetMoments,
        loss_fn: Optional[Any],
    ) -> torch.Tensor:
        """执行单步优化

        Args:
            params: 当前参数
            target_moments: 目标矩
            loss_fn: 可选的自定义损失函数

        Returns:
            总损失值
        """
        for param in params.values():
            if param.grad is not None:
                param.grad.zero_()

        if loss_fn is not None:
            total_loss = loss_fn(params, target_moments)
        else:
            total_loss = self._compute_loss(params, target_moments)

        total_loss.backward()

        if self.config.gradient_clip_norm > 0:
            for param in params.values():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(
                        [param], self.config.gradient_clip_norm
                    )

        gradients = {name: param.grad for name, param in params.items() if param.grad is not None}

        updated_params = self._update_params(params, gradients, self.config.learning_rate)

        for name, param in params.items():
            if name in updated_params:
                param.data.copy_(updated_params[name])

        return total_loss

    def _validate_initial_params(self, params: Dict[str, torch.Tensor]) -> None:
        """验证初始参数的有效性

        Args:
            params: 初始参数字典

        Raises:
            ValueError: 如果参数无效
        """
        required_keys = ['means', 'covariances', 'weights']
        for key in required_keys:
            if key not in params:
                raise ValueError(f"缺少必需的参数: {key}")

        means = params['means']
        covariances = params['covariances']
        weights = params['weights']

        if means.dim() != 2:
            raise ValueError(f"means必须是2D张量，当前维度: {means.dim()}")

        if weights.dim() != 1:
            raise ValueError(f"weights必须是1D张量，当前维度: {weights.dim()}")

        n_components = weights.shape[0]

        if means.shape[0] != n_components:
            raise ValueError(
                f"means的第一维({means.shape[0]})必须与weights长度({n_components})匹配"
            )

        if abs(weights.sum().item() - 1.0) > 1e-6:
            raise ValueError(f"weights的和必须为1，当前和: {weights.sum().item():.6f}")

        if (weights < 0).any():
            raise ValueError("weights必须非负")

        logger.debug(f"初始参数验证通过: n_components={n_components}, n_features={means.shape[1]}")

    def get_config(self) -> OptimizationConfig:
        """获取当前优化配置

        Returns:
            OptimizationConfig实例
        """
        return self.config

    def set_device(self, device: str) -> None:
        """设置计算设备

        Args:
            device: 设备字符串 ('cpu' 或 'cuda')
        """
        self._device = torch.device(device)
        self.config.device = device
        logger.info(f"计算设备已设置为: {device}")

    def __repr__(self) -> str:
        """返回优化器的字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"lr={self.config.learning_rate}, "
            f"max_iter={self.config.max_iterations}, "
            f"device={self.config.device})"
        )


class GradientDescentOptimizer(BaseGMMOptimizer):
    """梯度下降优化器

    使用标准梯度下降算法进行GMM参数优化。
    支持动量和自适应学习率。

    Attributes:
        velocity: 动量速度缓存

    Example:
        >>> optimizer = GradientDescentOptimizer(OptimizationConfig(momentum=0.9))
        >>> result = optimizer.optimize(target_moments, initial_params)
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """初始化梯度下降优化器

        Args:
            config: 优化配置
        """
        super().__init__(config)
        self.velocity: Dict[str, torch.Tensor] = {}

    def _compute_loss(
        self,
        params: Dict[str, torch.Tensor],
        target_moments: TargetMoments,
    ) -> torch.Tensor:
        """计算矩匹配损失

        使用简单的MSE损失作为默认实现。
        子类可以重写此方法以使用更复杂的损失函数。

        Args:
            params: 当前GMM参数
            target_moments: 目标矩

        Returns:
            损失值
        """
        from .loss_functions import MomentMatchingLoss

        loss_fn = MomentMatchingLoss()
        return loss_fn(params, target_moments)

    def _update_params(
        self,
        params: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor],
        learning_rate: float,
    ) -> Dict[str, torch.Tensor]:
        """使用带动量的梯度下降更新参数

        Args:
            params: 当前参数
            gradients: 梯度
            learning_rate: 学习率

        Returns:
            更新后的参数
        """
        updated = {}

        for name, param in params.items():
            if name not in gradients:
                updated[name] = param.clone()
                continue

            grad = gradients[name]

            if self.config.momentum > 0:
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(param)

                self.velocity[name] = (
                    self.config.momentum * self.velocity[name] + grad
                )
                update = self.velocity[name]
            else:
                update = grad

            updated[name] = param - learning_rate * update

        return updated


class AdamOptimizer(BaseGMMOptimizer):
    """Adam优化器

    实现Adam自适应学习率优化算法用于GMM参数优化。

    Attributes:
        m: 一阶矩估计（动量）
        v: 二阶矩估计（未中心化的方差）
        t: 时间步计数器
        beta1: 一阶矩衰减率
        beta2: 二阶矩衰减率
        epsilon: 数值稳定性常数

    Example:
        >>> optimizer = AdamOptimizer(OptimizationConfig(learning_rate=0.001))
        >>> result = optimizer.optimize(target_moments, initial_params)
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """初始化Adam优化器

        Args:
            config: 优化配置
            beta1: 一阶矩衰减率
            beta2: 二阶矩衰减率
            epsilon: 数值稳定性常数
        """
        super().__init__(config)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}
        self.t = 0

    def _compute_loss(
        self,
        params: Dict[str, torch.Tensor],
        target_moments: TargetMoments,
    ) -> torch.Tensor:
        """计算矩匹配损失

        Args:
            params: 当前GMM参数
            target_moments: 目标矩

        Returns:
            损失值
        """
        from .loss_functions import MomentMatchingLoss

        loss_fn = MomentMatchingLoss()
        return loss_fn(params, target_moments)

    def _update_params(
        self,
        params: Dict[str, torch.Tensor],
        gradients: Dict[str, torch.Tensor],
        learning_rate: float,
    ) -> Dict[str, torch.Tensor]:
        """使用Adam规则更新参数

        Args:
            params: 当前参数
            gradients: 梯度
            learning_rate: 学习率

        Returns:
            更新后的参数
        """
        self.t += 1
        updated = {}

        for name, param in params.items():
            if name not in gradients:
                updated[name] = param.clone()
                continue

            grad = gradients[name]

            if name not in self.m:
                self.m[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)

            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad ** 2

            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            update = learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
            updated[name] = param - update

        return updated
