"""GMM优化器早停和学习率调度机制"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging
import copy
import torch

logger = logging.getLogger(__name__)


@dataclass
class EarlyStoppingConfig:
    """早停配置数据类

    Attributes:
        patience: 耐心值（连续多少次无改善则停止）
        min_delta: 最小改善量（小于此值视为无改善）
        monitor: 监控的指标 ('loss', 'metric')
        mode: 模式 ('min' 表示越小越好，'max' 表示越大越好)
        restore_best_weights: 是否在停止时恢复最佳参数
        verbose: 是否打印日志

    Example:
        >>> config = EarlyStoppingConfig(
        ...     patience=50,
        ...     min_delta=1e-6,
        ...     mode='min'
        ... )
    """

    patience: int = 50
    min_delta: float = 1e-6
    monitor: str = "loss"
    mode: str = "min"
    restore_best_weights: bool = True
    verbose: bool = True

    def __post_init__(self):
        """验证配置参数"""
        if self.patience <= 0:
            raise ValueError(f"patience必须为正整数，当前值: {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta不能为负，当前值: {self.min_delta}")
        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode必须是'min'或'max'，当前值: {self.mode}")


class EarlyStopping:
    """早停机制

    监控验证损失或指标，当指标在连续patience次迭代中无改善时停止训练。
    支持在停止时恢复最佳参数。

    Attributes:
        config: 早停配置
        best_score: 最佳分数
        best_epoch: 最佳分数对应的迭代次数
        counter: 当前连续无改善的次数
        best_params: 最佳参数（如果restore_best_weights=True）
        stopped_epoch: 停止时的迭代次数

    Example:
        >>> early_stop = EarlyStopping(patience=50, min_delta=1e-6)
        >>> optimizer.add_callback("on_epoch_end", early_stop.callback)
        >>> result = optimizer.optimize(target_moments, initial_params)
        >>> if early_stop.stopped_epoch > 0:
        ...     print(f"早停于第 {early_stop.stopped_epoch} 次迭代")
    """

    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        """初始化早停器

        Args:
            config: 早停配置，如果为None则使用默认配置
        """
        self.config = config or EarlyStoppingConfig()

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.best_params: Optional[Dict[str, torch.Tensor]] = None
        self.stopped_epoch: int = 0
        self._should_stop: bool = False

        logger.info(
            f"早停器初始化: patience={self.config.patience}, "
            f"min_delta={self.config.min_delta}, mode={self.config.mode}"
        )

    def callback(self, callback_data) -> bool:
        """回调函数：检查是否应该停止

        Args:
            callback_data: EpochCallbackData对象

        Returns:
            是否应该停止训练
        """
        current_score = self._get_score(callback_data)
        epoch = callback_data.epoch

        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.config.restore_best_weights:
                self.best_params = copy.deepcopy(callback_data.params)
            return False

        if self._is_better(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
            if self.config.restore_best_weights:
                self.best_params = copy.deepcopy(callback_data.params)

            if self.config.verbose:
                logger.info(
                    f"Epoch {epoch}: {self.config.monitor} 改善 "
                    f"({self.best_score:.6f})"
                )
        else:
            self.counter += 1

            if self.config.verbose and self.counter % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: {self.config.monitor} 无改善 "
                    f"({self.counter}/{self.config.patience})"
                )

        if self.counter >= self.config.patience:
            self.stopped_epoch = epoch
            self._should_stop = True

            if self.config.verbose:
                logger.info(
                    f"早停触发于第 {epoch} 次迭代 "
                    f"(最佳{self.config.monitor}={self.best_score:.6f} "
                    f"于第{self.best_epoch}次迭代)"
                )

            return True

        return False

    def _get_score(self, callback_data) -> float:
        """从回调数据中获取监控指标

        Args:
            callback_data: 回调数据

        Returns:
            监控指标值
        """
        if self.config.monitor == "loss":
            return callback_data.loss
        elif callback_data.metrics and self.config.monitor in callback_data.metrics:
            return callback_data.metrics[self.config.monitor]
        else:
            logger.warning(
                f"无法找到监控指标 '{self.config.monitor}'，"
                f"使用loss作为替代"
            )
            return callback_data.loss

    def _is_better(self, current: float, best: float) -> bool:
        """判断当前分数是否优于最佳分数

        Args:
            current: 当前分数
            best: 最佳分数

        Returns:
            是否更好
        """
        if self.config.mode == "min":
            return current < best - self.config.min_delta
        else:
            return current > best + self.config.min_delta

    def get_best_params(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取最佳参数

        Returns:
            最佳参数字典，如果restore_best_weights=False则返回None
        """
        return self.best_params

    def reset(self) -> None:
        """重置早停器状态"""
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.best_params = None
        self.stopped_epoch = 0
        self._should_stop = False

        logger.debug("早停器已重置")

    @property
    def should_stop(self) -> bool:
        """是否应该停止训练"""
        return self._should_stop


class LearningRateScheduler(ABC):
    """学习率调度器抽象基类

    定义学习率调度器的接口。

    子类需要实现:
    - get_lr(): 获取当前学习率
    - step(): 更新学习率

    Example:
        >>> class MyScheduler(LearningRateScheduler):
        ...     def get_lr(self, epoch):
        ...         return initial_lr / (epoch + 1)
    """

    @abstractmethod
    def get_lr(self, epoch: int) -> float:
        """获取指定迭代次数的学习率

        Args:
            epoch: 迭代次数

        Returns:
            学习率
        """
        pass

    @abstractmethod
    def step(self, epoch: int) -> float:
        """更新学习率

        Args:
            epoch: 当前迭代次数

        Returns:
            新的学习率
        """
        pass

    def __call__(self, epoch: int) -> float:
        """使实例可调用"""
        return self.get_lr(epoch)


class StepLR(LearningRateScheduler):
    """阶梯式学习率衰减

    每隔step_size次迭代，学习率乘以gamma。

    Attributes:
        initial_lr: 初始学习率
        step_size: 衰减间隔（迭代次数）
        gamma: 衰减因子

    Example:
        >>> scheduler = StepLR(initial_lr=0.1, step_size=30, gamma=0.1)
        >>> lr = scheduler.step(60)  # lr = 0.1 * 0.1 * 0.1 = 0.001
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        step_size: int = 30,
        gamma: float = 0.1,
    ):
        """初始化阶梯式学习率调度器

        Args:
            initial_lr: 初始学习率
            step_size: 衰减间隔（迭代次数）
            gamma: 衰减因子（通常为0.1或0.5）
        """
        self.initial_lr = initial_lr
        self.step_size = step_size
        self.gamma = gamma

        if initial_lr <= 0:
            raise ValueError(f"initial_lr必须为正数，当前值: {initial_lr}")
        if step_size <= 0:
            raise ValueError(f"step_size必须为正整数，当前值: {step_size}")
        if gamma <= 0 or gamma >= 1:
            raise ValueError(f"gamma必须在(0,1)范围内，当前值: {gamma}")

        logger.info(
            f"StepLR初始化: initial_lr={initial_lr}, "
            f"step_size={step_size}, gamma={gamma}"
        )

    def get_lr(self, epoch: int) -> float:
        """获取当前学习率

        Args:
            epoch: 迭代次数

        Returns:
            学习率
        """
        return self.initial_lr * (self.gamma ** (epoch // self.step_size))

    def step(self, epoch: int) -> float:
        """更新学习率

        Args:
            epoch: 当前迭代次数

        Returns:
            新的学习率
        """
        lr = self.get_lr(epoch)
        if epoch % self.step_size == 0 and epoch > 0:
            logger.debug(f"Epoch {epoch}: 学习率衰减为 {lr:.2e}")
        return lr


class ExponentialLR(LearningRateScheduler):
    """指数式学习率衰减

    学习率按指数衰减: lr = initial_lr * gamma^epoch

    Attributes:
        initial_lr: 初始学习率
        gamma: 衰减因子

    Example:
        >>> scheduler = ExponentialLR(initial_lr=0.1, gamma=0.99)
        >>> lr = scheduler.step(100)  # lr = 0.1 * 0.99^100
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        gamma: float = 0.99,
    ):
        """初始化指数式学习率调度器

        Args:
            initial_lr: 初始学习率
            gamma: 衰减因子（通常接近1，如0.99或0.995）
        """
        self.initial_lr = initial_lr
        self.gamma = gamma

        if initial_lr <= 0:
            raise ValueError(f"initial_lr必须为正数，当前值: {initial_lr}")
        if gamma <= 0 or gamma >= 1:
            raise ValueError(f"gamma必须在(0,1)范围内，当前值: {gamma}")

        logger.info(
            f"ExponentialLR初始化: initial_lr={initial_lr}, gamma={gamma}"
        )

    def get_lr(self, epoch: int) -> float:
        """获取当前学习率

        Args:
            epoch: 迭代次数

        Returns:
            学习率
        """
        return self.initial_lr * (self.gamma ** epoch)

    def step(self, epoch: int) -> float:
        """更新学习率

        Args:
            epoch: 当前迭代次数

        Returns:
            新的学习率
        """
        return self.get_lr(epoch)


class CosineAnnealingLR(LearningRateScheduler):
    """余弦退火学习率调度

    学习率按余弦曲线从initial_lr衰减到min_lr。

    Attributes:
        initial_lr: 初始学习率
        min_lr: 最小学习率
        T_max: 周期长度（迭代次数）
        eta_min: 最小学习率（同min_lr）

    Example:
        >>> scheduler = CosineAnnealingLR(initial_lr=0.1, T_max=100, min_lr=1e-6)
        >>> lr = scheduler.step(50)  # 中间位置，lr约等于0.05
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        T_max: int = 100,
        min_lr: float = 1e-6,
    ):
        """初始化余弦退火学习率调度器

        Args:
            initial_lr: 初始学习率
            T_max: 周期长度（迭代次数）
            min_lr: 最小学习率
        """
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr
        self.eta_min = min_lr

        if initial_lr <= 0:
            raise ValueError(f"initial_lr必须为正数，当前值: {initial_lr}")
        if T_max <= 0:
            raise ValueError(f"T_max必须为正整数，当前值: {T_max}")
        if min_lr < 0:
            raise ValueError(f"min_lr不能为负，当前值: {min_lr}")
        if min_lr >= initial_lr:
            raise ValueError(f"min_lr必须小于initial_lr")

        logger.info(
            f"CosineAnnealingLR初始化: initial_lr={initial_lr}, "
            f"T_max={T_max}, min_lr={min_lr}"
        )

    def get_lr(self, epoch: int) -> float:
        """获取当前学习率

        Args:
            epoch: 迭代次数

        Returns:
            学习率
        """
        return self.min_lr + (self.initial_lr - self.min_lr) * (
            1 + torch.cos(torch.tensor(epoch * 3.141592653589793 / self.T_max))
        ) / 2

    def step(self, epoch: int) -> float:
        """更新学习率

        Args:
            epoch: 当前迭代次数

        Returns:
            新的学习率
        """
        return self.get_lr(epoch)


class ReduceLROnPlateau(LearningRateScheduler):
    """基于指标的自适应学习率调整

    当指标停止改善时，降低学习率。

    Attributes:
        initial_lr: 初始学习率
        factor: 学习率衰减因子
        patience: 等待改善的迭代次数
        min_lr: 最小学习率
        mode: 'min'或'max'，指标越小越好还是越大越好
        threshold: 改善的阈值
        cooldown: 学习率降低后等待的迭代次数

    Example:
        >>> scheduler = ReduceLROnPlateau(initial_lr=0.1, factor=0.5, patience=10)
        >>> lr = scheduler.step(50, metric=0.5)  # 根据metric调整学习率
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 1e-6,
        mode: str = "min",
        threshold: float = 1e-4,
        cooldown: int = 0,
    ):
        """初始化ReduceLROnPlateau调度器

        Args:
            initial_lr: 初始学习率
            factor: 学习率衰减因子
            patience: 等待改善的迭代次数
            min_lr: 最小学习率
            mode: 'min'或'max'
            threshold: 改善阈值
            cooldown: 冷却迭代次数
        """
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        self.cooldown = cooldown

        self.current_lr = initial_lr
        self.best: Optional[float] = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

        if initial_lr <= 0:
            raise ValueError(f"initial_lr必须为正数，当前值: {initial_lr}")
        if factor <= 0 or factor >= 1:
            raise ValueError(f"factor必须在(0,1)范围内，当前值: {factor}")
        if patience <= 0:
            raise ValueError(f"patience必须为正整数，当前值: {patience}")
        if mode not in ["min", "max"]:
            raise ValueError(f"mode必须是'min'或'max'，当前值: {mode}")

        logger.info(
            f"ReduceLROnPlateau初始化: initial_lr={initial_lr}, "
            f"factor={factor}, patience={patience}, mode={mode}"
        )

    def get_lr(self, epoch: int) -> float:
        """获取当前学习率

        Args:
            epoch: 迭代次数（此调度器不使用epoch参数）

        Returns:
            当前学习率
        """
        return self.current_lr

    def step(self, epoch: int, metric: Optional[float] = None) -> float:
        """更新学习率

        Args:
            epoch: 当前迭代次数
            metric: 监控指标值（必需）

        Returns:
            新的学习率

        Raises:
            ValueError: 如果metric为None
        """
        if metric is None:
            raise ValueError("ReduceLROnPlateau需要metric参数")

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_lr

        if self.best is None:
            self.best = metric
            return self.current_lr

        if self._is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown

            logger.info(
                f"Epoch {epoch}: 学习率降低 {old_lr:.2e} -> {self.current_lr:.2e}"
            )

        return self.current_lr

    def _is_better(self, current: float, best: float) -> bool:
        """判断当前指标是否优于最佳指标

        Args:
            current: 当前指标
            best: 最佳指标

        Returns:
            是否更好
        """
        if self.mode == "min":
            return current < best - self.threshold
        else:
            return current > best + self.threshold

    def reset(self) -> None:
        """重置调度器状态"""
        self.current_lr = self.initial_lr
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0

        logger.debug("ReduceLROnPlateau已重置")


class LambdaLR(LearningRateScheduler):
    """Lambda学习率调度器

    使用自定义的lambda函数计算学习率。

    Attributes:
        initial_lr: 初始学习率
        lr_lambda: 学习率计算函数，接受epoch参数，返回乘数

    Example:
        >>> scheduler = LambdaLR(
        ...     initial_lr=0.1,
        ...     lr_lambda=lambda epoch: 0.95 ** epoch
        ... )
        >>> lr = scheduler.step(10)
    """

    def __init__(
        self,
        initial_lr: float = 0.01,
        lr_lambda: Optional[Callable[[int], float]] = None,
    ):
        """初始化Lambda学习率调度器

        Args:
            initial_lr: 初始学习率
            lr_lambda: 学习率计算函数，默认为线性衰减
        """
        self.initial_lr = initial_lr
        self.lr_lambda = lr_lambda or (lambda epoch: 1.0 / (1.0 + 0.1 * epoch))

        if initial_lr <= 0:
            raise ValueError(f"initial_lr必须为正数，当前值: {initial_lr}")

        logger.info(
            f"LambdaLR初始化: initial_lr={initial_lr}"
        )

    def get_lr(self, epoch: int) -> float:
        """获取当前学习率

        Args:
            epoch: 迭代次数

        Returns:
            学习率
        """
        return self.initial_lr * self.lr_lambda(epoch)

    def step(self, epoch: int) -> float:
        """更新学习率

        Args:
            epoch: 当前迭代次数

        Returns:
            新的学习率
        """
        return self.get_lr(epoch)


def create_scheduler(
    scheduler_type: str,
    initial_lr: float = 0.01,
    **kwargs,
) -> LearningRateScheduler:
    """工厂函数：创建学习率调度器

    Args:
        scheduler_type: 调度器类型:
                       - 'step': StepLR
                       - 'exponential': ExponentialLR
                       - 'cosine': CosineAnnealingLR
                       - 'plateau': ReduceLROnPlateau
                       - 'lambda': LambdaLR
        initial_lr: 初始学习率
        **kwargs: 传递给特定调度器的额外参数

    Returns:
        学习率调度器实例

    Raises:
        ValueError: 如果scheduler_type无效

    Example:
        >>> scheduler = create_scheduler(
        ...     'cosine',
        ...     initial_lr=0.1,
        ...     T_max=100,
        ...     min_lr=1e-6
        ... )
    """
    scheduler_creators = {
        "step": lambda: StepLR(initial_lr=initial_lr, **kwargs),
        "exponential": lambda: ExponentialLR(initial_lr=initial_lr, **kwargs),
        "cosine": lambda: CosineAnnealingLR(initial_lr=initial_lr, **kwargs),
        "plateau": lambda: ReduceLROnPlateau(initial_lr=initial_lr, **kwargs),
        "lambda": lambda: LambdaLR(initial_lr=initial_lr, **kwargs),
    }

    if scheduler_type not in scheduler_creators:
        raise ValueError(
            f"无效的scheduler_type: {scheduler_type}。"
            f"支持的类型: {list(scheduler_creators.keys())}"
        )

    scheduler = scheduler_creators[scheduler_type]()
    logger.info(f"创建学习率调度器: {type(scheduler).__name__}")

    return scheduler
