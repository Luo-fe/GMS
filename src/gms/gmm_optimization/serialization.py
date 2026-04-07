"""GMM参数序列化和反序列化

提供多种格式的参数保存和加载功能:
- JSON格式: 人类可读，适合版本控制和调试
- Pickle格式: 高效二进制，适合大模型和快速加载
- State Dict格式: PyTorch原生兼容，可与模型整合
- Gzip压缩: 减少文件大小

支持版本控制和向后兼容性检查。

Example:
    >>> from gms.gmm_optimization import GMMParameters, GMMSerializer
    >>>
    >>> # 创建序列化器
    >>> serializer = GMMSerializer()
    >>>
    >>> # 保存为JSON（人类可读）
    >>> serializer.save_json(params, "model_params.json")
    >>>
    >>> # 保存为Pickle（高效二进制）
    >>> serializer.save_pickle(params, "model_params.pkl", compress=True)
    >>>
    >>> # 加载参数
    >>> loaded_params = serializer.load_json("model_params.json")
"""

import json
import gzip
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
import torch
import numpy as np
import logging
import warnings

from .gmm_parameters import GMMParameters, GMMParametersConfig

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
CURRENT_VERSION = __version__


class SerializationConfig:
    """序列化配置类

    配置序列化过程的各种选项。

    Attributes:
        include_metadata: 是否包含元数据（时间戳、版本等）
        compression_level: gzip压缩级别 (0-9)
        pretty_print: JSON是否使用美化输出
        protocol: pickle协议版本
        check_version: 加载时是否检查版本兼容性
        backup_existing: 保存时是否备份已存在的文件
    """

    def __init__(
        self,
        include_metadata: bool = True,
        compression_level: int = 6,
        pretty_print: bool = True,
        protocol: int = pickle.HIGHEST_PROTOCOL,
        check_version: bool = True,
        backup_existing: bool = False,
    ):
        self.include_metadata = include_metadata
        self.compression_level = max(0, min(9, compression_level))
        self.pretty_print = pretty_print
        self.protocol = protocol
        self.check_version = check_version
        self.backup_existing = backup_existing


class GMMSerializer:
    """GMM参数序列化器

    提供完整的参数保存和加载功能，
    支持多种格式和压缩选项。

    支持的格式:
    - .json / .json.gz: JSON格式
    - .pkl / .pickle / .pkl.gz: Pickle格式
    - .pt / .pth / .tar: PyTorch state_dict格式

    Example:
        >>> serializer = GMMSerializer()
        >>>
        >>> # 自动检测格式保存
        >>> serializer.save(params, "model.json")  # JSON
        >>> serializer.save(params, "model.pkl")   # Pickle
        >>> serializer.save(params, "model.pt")    # State dict
        >>>
        >>> # 自动检测格式加载
        >>> params = serializer.load("model.json")
    """

    SUPPORTED_EXTENSIONS = {
        '.json': 'json',
        '.json.gz': 'json',
        '.pkl': 'pickle',
        '.pickle': 'pickle',
        '.pkl.gz': 'pickle',
        '.pt': 'state_dict',
        '.pth': 'state_dict',
        '.tar': 'state_dict',
    }

    def __init__(self, config: Optional[SerializationConfig] = None):
        """初始化序列化器

        Args:
            config: 序列化配置，如果为None则使用默认配置
        """
        self.config = config or SerializationConfig()
        logger.debug("GMMSerializer初始化完成")

    def save(
        self,
        params: GMMParameters,
        filepath: Union[str, Path],
        format: Optional[str] = None,
    ) -> None:
        """智能保存（自动检测格式）

        根据文件扩展名自动选择合适的保存格式。

        Args:
            params: GMMParameters实例
            filepath: 保存路径
            format: 可选的格式覆盖 ('json', 'pickle', 'state_dict')

        Raises:
            ValueError: 如果文件格式不支持
        """
        filepath = Path(filepath)
        ext = ''.join(filepath.suffixes)

        if format is None:
            if ext not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"不支持的文件格式: {ext}。"
                    f"支持的格式: {list(self.SUPPORTED_EXTENSIONS.keys())}"
                )
            format = self.SUPPORTED_EXTENSIONS[ext]

        if format == 'json':
            self.save_json(params, filepath)
        elif format == 'pickle':
            self.save_pickle(params, filepath)
        elif format == 'state_dict':
            self.save_state_dict(params, filepath)
        else:
            raise ValueError(f"不支持的格式: {format}")

        logger.info(f"参数已保存到: {filepath}")

    def load(
        self,
        filepath: Union[str, Path],
    ) -> GMMParameters:
        """智能加载（自动检测格式）

        Args:
            filepath: 文件路径

        Returns:
            GMMParameters实例

        Raises:
            ValueError: 如果文件格式不支持或文件损坏
        """
        filepath = Path(filepath)
        ext = ''.join(filepath.suffixes)

        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {ext}。"
                f"支持的格式: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            )

        format = self.SUPPORTED_EXTENSIONS[ext]

        if format == 'json':
            return self.load_json(filepath)
        elif format == 'pickle':
            return self.load_pickle(filepath)
        elif format == 'state_dict':
            return self.load_state_dict(filepath)

    def save_json(
        self,
        params: GMMParameters,
        filepath: Union[str, Path],
    ) -> None:
        """保存为JSON格式

        生成人类可读的JSON文件，适合:
        - 版本控制系统（Git等）
        - 手动查看和编辑
        - 跨语言/跨平台交换

        Args:
            params: GMMParameters实例
            filepath: 输出文件路径（.json 或 .json.gz）

        Example:
            >>> serializer.save_json(params, "model.json")
            >>> serializer.save_json(params, "compressed_model.json.gz")
        """
        filepath = Path(filepath)

        data = {
            'version': CURRENT_VERSION,
            'created_at': datetime.now().isoformat(),
            'parameters': self._params_to_dict(params),
        }

        if self.config.include_metadata:
            data['metadata'] = {
                'dimensionality': params.dimensionality,
                'is_diagonal': params.is_diagonal,
                'weight': params.weight,
                'weight2': params.weight2,
                'mean1_norm': float(torch.norm(params.mean1).item()),
                'mean2_norm': float(torch.norm(params.mean2).item()),
            }

        indent = 4 if self.config.pretty_print else None

        if str(filepath).endswith('.gz'):
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)

        logger.debug(f"JSON保存完成: {filepath}")

    def load_json(
        self,
        filepath: Union[str, Path],
    ) -> GMMParameters:
        """从JSON文件加载参数

        Args:
            filepath: JSON文件路径（.json 或 .json.gz）

        Returns:
            GMMParameters实例

        Raises:
            ValueError: 如果JSON格式无效或版本不兼容

        Example:
            >>> params = serializer.load_json("model.json")
        """
        filepath = Path(filepath)

        if str(filepath).endswith('.gz'):
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

        if self.config.check_version and 'version' in data:
            self._check_version_compatibility(data['version'])

        params = self._dict_to_params(data['parameters'])

        logger.info(f"从JSON加载完成: {filepath}")
        return params

    def save_pickle(
        self,
        params: GMMParameters,
        filepath: Union[str, Path],
        compress: bool = False,
    ) -> None:
        """保存为Pickle格式

        生成高效的二进制文件，适合:
        - 大模型快速保存/加载
        - 保留完整的Python对象信息
        - 内部缓存和临时数据

        Args:
            params: GMMParameters实例
            filepath: 输出文件路径
            compress: 是否使用gzip压缩

        Example:
            >>> serializer.save_pickle(params, "model.pkl")
            >>> serializer.save_pickle(params, "model.pkl.gz", compress=True)
        """
        filepath = Path(filepath)

        data = {
            '__class__': 'GMMParameters',
            'version': CURRENT_VERSION,
            'created_at': datetime.now().isoformat(),
            'data': params,
        }

        protocol = self.config.protocol

        if compress or str(filepath).endswith('.gz'):
            with gzip.open(filepath, 'wb', compresslevel=self.config.compression_level) as f:
                pickle.dump(data, f, protocol=protocol)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)

        logger.debug(f"Pickle保存完成: {filepath}")

    def load_pickle(
        self,
        filepath: Union[str, Path],
    ) -> GMMParameters:
        """从Pickle文件加载参数

        Args:
            filepath: Pickle文件路径

        Returns:
            GMMParameters实例

        Raises:
            ValueError: 如果文件损坏或对象类型错误

        Warning:
            仅加载可信来源的Pickle文件，可能存在安全风险。

        Example:
            >>> params = serializer.load_pickle("model.pkl")
        """
        filepath = Path(filepath)

        if str(filepath).endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        if not isinstance(data, dict) or '__class__' not in data:
            warnings.warn("文件格式可能不兼容，尝试直接加载...")
            if isinstance(data, GMMParameters):
                return data
            raise ValueError("无效的Pickle文件格式")

        if data.get('__class__') != 'GMMParameters':
            raise ValueError(f"对象类型错误: {data.get('__class__')}")

        if self.config.check_version and 'version' in data:
            self._check_version_compatibility(data['version'])

        params = data['data']

        logger.info(f"从Pickle加载完成: {filepath}")
        return params

    def save_state_dict(
        self,
        params: GMMParameters,
        filepath: Union[str, Path],
    ) -> None:
        """保存为PyTorch State Dict格式

        生成PyTorch原生的状态字典格式，适合:
        - 与nn.Module模型整合
        - 使用torch.load/torch.save标准接口
        - 跨设备迁移（CPU/GPU）

        Args:
            params: GMMParameters实例
            filepath: 输出文件路径 (.pt, .pth, .tar)

        Example:
            >>> serializer.save_state_dict(params, "model.pt")
            >>>
            >>> # 直接用PyTorch加载
            >>> import torch
            >>> state_dict = torch.load("model.pt")
        """
        filepath = Path(filepath)

        state_dict = {
            'version': CURRENT_VERSION,
            'created_at': datetime.now().isoformat(),
            'weight': torch.tensor([params.weight]),
            'mean1': params.mean1.detach().cpu(),
            'mean2': params.mean2.detach().cpu(),
            'variance1': params.variance1.detach().cpu(),
            'variance2': params.variance2.detach().cpu(),
            'is_diagonal': params.is_diagonal,
            'dimensionality': params.dimensionality,
        }

        if self.config.include_metadata:
            state_dict['metadata'] = {
                'config': {
                    'covariance_type': 'diagonal' if params.is_diagonal else 'full',
                    'device': str(params.mean1.device),
                    'dtype': str(params.mean1.dtype),
                }
            }

        torch.save(state_dict, filepath)

        logger.debug(f"State dict保存完成: {filepath}")

    def load_state_dict(
        self,
        filepath: Union[str, Path],
        map_location: Optional[torch.device] = None,
    ) -> GMMParameters:
        """从State Dict文件加载参数

        Args:
            filepath: 状态字典文件路径
            map_location: 可选的目标设备映射

        Returns:
            GMMParameters实例

        Raises:
            ValueError: 如果状态字典缺少必需的字段

        Example:
            >>> params = serializer.load_state_dict("model.pt")
            >>> params_gpu = serializer.load_state_dict("model.pt", map_location='cuda')
        """
        filepath = Path(filepath)

        state_dict = torch.load(filepath, map_location=map_location)

        if not isinstance(state_dict, dict):
            raise ValueError("无效的状态字典格式")

        if self.config.check_version and 'version' in state_dict:
            version = state_dict['version']
            if isinstance(version, torch.Tensor):
                version = str(version.item())
            self._check_version_compatibility(version)

        required_keys = ['weight', 'mean1', 'mean2', 'variance1', 'variance2']
        for key in required_keys:
            if key not in state_dict:
                raise ValueError(f"状态字典缺少必需字段: {key}")

        weight = float(state_dict['weight'])
        if isinstance(weight, torch.Tensor):
            weight = weight.item()

        mean1 = state_dict['mean1']
        mean2 = state_dict['mean2']
        variance1 = state_dict['variance1']
        variance2 = state_dict['variance2']

        config = GMMParametersConfig()

        params = GMMParameters(
            weight=weight,
            mean1=mean1,
            mean2=mean2,
            variance1=variance1,
            variance2=variance2,
            _config=config,
        )

        logger.info(f"从State dict加载完成: {filepath}")
        return params

    def export_to_numpy(self, params: GMMParameters) -> Dict[str, np.ndarray]:
        """导出为NumPy数组字典

        将所有张量转换为NumPy数组，用于:
        - 与非PyTorch代码交互
        - 数据分析和可视化
        - 其他框架集成

        Args:
            params: GMMParameters实例

        Returns:
            字典，键为参数名，值为numpy数组
        """
        return {
            'weight': np.array([params.weight, params.weight2]),
            'mean1': params.mean1.cpu().numpy(),
            'mean2': params.mean2.cpu().numpy(),
            'variance1': params.variance1.cpu().numpy(),
            'variance2': params.variance2.cpu().numpy(),
        }

    @staticmethod
    def _params_to_dict(params: GMMParameters) -> Dict[str, Any]:
        """将参数转换为可序列化的字典"""
        return {
            'weight': params.weight,
            'mean1': params.mean1.cpu().tolist(),
            'mean2': params.mean2.cpu().tolist(),
            'variance1': params.variance1.cpu().tolist(),
            'variance2': params.variance2.cpu().tolist(),
        }

    @staticmethod
    def _dict_to_params(data: Dict[str, Any]) -> GMMParameters:
        """从字典重建参数对象"""
        config = GMMParametersConfig()

        return GMMParameters(
            weight=float(data['weight']),
            mean1=torch.tensor(data['mean1']),
            mean2=torch.tensor(data['mean2']),
            variance1=torch.tensor(data['variance1']),
            variance2=torch.tensor(data['variance2']),
            _config=config,
        )

    @staticmethod
    def _check_version_compatibility(saved_version: str) -> None:
        """检查版本兼容性

        Args:
            saved_version: 保存时的版本号

        Raises:
            ValueError: 如果主版本号不匹配
        """
        saved_major = saved_version.split('.')[0]
        current_major = CURRENT_VERSION.split('.')[0]

        if saved_major != current_major:
            warnings.warn(
                f"版本不兼容: 文件版本={saved_version}, "
                f"当前版本={CURRENT_VERSION}。可能存在兼容性问题。"
            )
        elif saved_version != CURRENT_VERSION:
            logger.info(
                f"次版本号差异: 文件版本={saved_version}, "
                f"当前版本={CURRENT_VERSION}"
            )

    def get_file_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """获取文件的详细信息

        Args:
            filepath: 文件路径

        Returns:
            包含文件信息的字典
        """
        filepath = Path(filepath)
        stat = filepath.stat()

        info = {
            'path': str(filepath.absolute()),
            'size_bytes': stat.st_size,
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'extension': ''.join(filepath.suffixes),
        }

        ext = info['extension']
        if ext in self.SUPPORTED_EXTENSIONS:
            info['format'] = self.SUPPORTED_EXTENSIONS[ext]

        try:
            if ext.endswith('.json') or ext.endswith('.json.gz'):
                with (gzip.open if ext.endswith('.gz') else open)(filepath, 'rt') as f:
                    data = json.load(f)
                info['version'] = data.get('version', 'unknown')
                info['created_at'] = data.get('created_at', 'unknown')

            elif ext.endswith(('.pkl', '.pickle')) or ext.endswith('.pkl.gz'):
                with (gzip.open if ext.endswith('.gz') else open)(filepath, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    info['version'] = data.get('version', 'unknown')
                    info['created_at'] = data.get('created_at', 'unknown')

            elif ext in ['.pt', '.pth', '.tar']:
                state_dict = torch.load(filepath, map_location='cpu')
                if isinstance(state_dict, dict):
                    info['version'] = state_dict.get('version', 'unknown')
                    info['created_at'] = state_dict.get('created_at', 'unknown')
        except Exception as e:
            info['error'] = str(e)

        return info


def create_serializer(**kwargs) -> GMMSerializer:
    """工厂函数：创建序列化器实例

    Args:
        **kwargs: 传递给SerializationConfig的关键字参数

    Returns:
        GMMSerializer实例

    Example:
        >>> serializer = create_serializer(compression_level=9, pretty_print=False)
    """
    config = SerializationConfig(**kwargs)
    return GMMSerializer(config)
