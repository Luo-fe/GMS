"""特征缓存模块

提供高效的特征提取结果缓存机制，包括：
- 基于图像哈希的缓存键生成
- 内存缓存（LRU策略）
- 磁盘缓存支持（可选）
- 缓存命中率和统计信息
- 缓存清理和序列化方法
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import hashlib
import json
import time
from pathlib import Path
from collections import OrderedDict
from datetime import datetime
import torch
import numpy as np

logger = logging.getLogger(__name__)


class FeatureCache:
    """特征缓存管理器
    
    提供基于LRU策略的内存缓存和可选的磁盘持久化存储。
    使用图像内容的哈希值作为缓存键，确保相同图像产生相同缓存。
    
    Attributes:
        max_size: 最大缓存条目数（内存）
        memory_cache: OrderedDict实现的LRU缓存
        disk_cache_dir: 磁盘缓存目录（可选）
        enable_disk_cache: 是否启用磁盘缓存
        
    Example:
        >>> cache = FeatureCache(max_size=1000, enable_disk_cache=True)
        >>> cache.store(image_hash, features)
        >>> cached_features = cache.get(image_hash)
    """

    def __init__(
        self,
        max_size: int = 1000,
        enable_disk_cache: bool = False,
        disk_cache_dir: Optional[Union[str, Path]] = None,
        disk_cache_max_size_mb: float = 1024.0,
    ) -> None:
        """初始化特征缓存
        
        Args:
            max_size: 内存缓存最大条目数，默认1000
            enable_disk_cache: 是否启用磁盘缓存，默认False
            disk_cache_dir: 磁盘缓存目录路径，默认为 './.feature_cache'
            disk_cache_max_size_mb: 磁盘缓存最大大小(MB)，默认1024MB
        """
        self.max_size = max_size
        self.enable_disk_cache = enable_disk_cache
        self.disk_cache_max_size_mb = disk_cache_max_size_mb

        # LRU内存缓存 (OrderedDict)
        self._memory_cache: OrderedDict[str, torch.Tensor] = OrderedDict()

        # 统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "evictions": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "disk_stores": 0,
        }

        # 磁盘缓存设置
        if enable_disk_cache:
            if disk_cache_dir is None:
                disk_cache_dir = Path.cwd() / ".feature_cache"
            else:
                disk_cache_dir = Path(disk_cache_dir)
            
            self.disk_cache_dir = disk_cache_dir
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # 元数据文件
            self._metadata_file = self.disk_cache_dir / "cache_metadata.json"
            self._load_metadata()
            
            logger.info(f"磁盘缓存已启用，目录: {self.disk_cache_dir}")
        else:
            self.disk_cache_dir = None
            self._metadata_file = None
            self._disk_metadata: Dict[str, Dict] = {}

        logger.info(
            f"特征缓存初始化完成: "
            f"最大条目={max_size}, "
            f"磁盘缓存={enable_disk_cache}"
        )

    def _load_metadata(self) -> None:
        """加载磁盘缓存的元数据"""
        if self._metadata_file and self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    self._disk_metadata = json.load(f)
                logger.debug(f"已加载磁盘缓存元数据: {len(self._disk_metadata)} 条")
            except Exception as e:
                logger.warning(f"加载元数据失败: {e}，将创建新的")
                self._disk_metadata = {}
        else:
            self._disk_metadata = {}

    def _save_metadata(self) -> None:
        """保存磁盘缓存的元数据到文件"""
        if self._metadata_file:
            try:
                with open(self._metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(self._disk_metadata, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"保存元数据失败: {e}")

    @staticmethod
    def generate_hash(
        data: Union[torch.Tensor, np.ndarray, bytes, str],
        method: str = "sha256",
    ) -> str:
        """生成数据的哈希值作为缓存键
        
        Args:
            data: 输入数据，可以是张量、数组、字节或字符串
            method: 哈希算法，支持 'md5', 'sha1', 'sha256'（默认）
            
        Returns:
            十六进制哈希字符串
            
        Raises:
            ValueError: 如果哈希算法不支持或数据类型不支持
        """
        if method not in ["md5", "sha1", "sha256"]:
            raise ValueError(f"不支持的哈希算法: {method}")

        # 将各种数据类型转换为bytes
        if isinstance(data, torch.Tensor):
            raw_bytes = data.cpu().numpy().tobytes()
        elif isinstance(data, np.ndarray):
            raw_bytes = data.tobytes()
        elif isinstance(data, (str, Path)):
            if isinstance(data, Path):
                data = str(data)
            raw_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            raw_bytes = data
        else:
            raise ValueError(f"不支持的数据类型用于生成哈希: {type(data)}")

        hash_obj = hashlib.new(method)
        hash_obj.update(raw_bytes)
        return hash_obj.hexdigest()

    def get(self, key: str) -> Optional[torch.Tensor]:
        """从缓存中获取特征
        
        优先查找内存缓存，未命中则查找磁盘缓存。
        
        Args:
            key: 缓存键（通常是图像的哈希值）
            
        Returns:
            缓存的特征张量，如果未找到则返回None
        """
        # 查找内存缓存
        if key in self._memory_cache:
            # 移动到末尾（标记为最近使用）
            self._memory_cache.move_to_end(key)
            self._stats["hits"] += 1
            logger.debug(f"内存缓存命中: {key[:16]}...")
            return self._memory_cache[key]

        # 查找磁盘缓存
        if self.enable_disk_cache and key in self._disk_metadata:
            tensor = self._load_from_disk(key)
            if tensor is not None:
                # 放入内存缓存
                self._store_in_memory(key, tensor)
                self._stats["disk_hits"] += 1
                self._stats["hits"] += 1
                logger.debug(f"磁盘缓存命中: {key[:16]}...")
                return tensor
            else:
                self._stats["disk_misses"] += 1

        self._stats["misses"] += 1
        logger.debug(f"缓存未命中: {key[:16]}...")
        return None

    def store(
        self,
        key: str,
        value: torch.Tensor,
        persist_to_disk: bool = False,
    ) -> None:
        """存储特征到缓存
        
        Args:
            key: 缓存键
            value: 要缓存的特征张量
            persist_to_disk: 是否同时保存到磁盘（即使未启用自动磁盘缓存）
        """
        # 存储到内存
        self._store_in_memory(key, value)
        self._stats["stores"] += 1

        # 存储到磁盘（如果启用或显式要求）
        if (self.enable_disk_cache or persist_to_disk) and self.disk_cache_dir:
            self._save_to_disk(key, value)
            if persist_to_disk or self.enable_disk_cache:
                self._stats["disk_stores"] += 1

        logger.debug(f"特征已缓存: {key[:16]}...")

    def _store_in_memory(self, key: str, value: torch.Tensor) -> None:
        """存储到内存缓存（带LRU淘汰）"""
        # 如果已存在，先删除旧条目
        if key in self._memory_cache:
            del self._memory_cache[key]

        # 检查是否需要淘汰
        while len(self._memory_cache) >= self.max_size:
            evicted_key, _ = self._memory_cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"LRU淘汰: {evicted_key[:16]}...")

        # 存储新条目
        self._memory_cache[key] = value

    def _save_to_disk(self, key: str, value: torch.Tensor) -> None:
        """保存特征到磁盘"""
        file_path = self.disk_cache_dir / f"{key}.pt"
        
        try:
            torch.save(value, file_path)
            
            # 更新元数据
            self._disk_metadata[key] = {
                "file": str(file_path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "size_bytes": file_path.stat().st_size,
                "timestamp": datetime.now().isoformat(),
            }
            self._save_metadata()
            
            logger.debug(f"已保存到磁盘: {file_path.name}")
        except Exception as e:
            logger.error(f"保存到磁盘失败: {e}")

    def _load_from_disk(self, key: str) -> Optional[torch.Tensor]:
        """从磁盘加载特征"""
        if key not in self._disk_metadata:
            return None

        file_path = Path(self._disk_metadata[key]["file"])
        
        if not file_path.exists():
            logger.warning(f"磁盘缓存文件不存在: {file_path}")
            del self._disk_metadata[key]
            self._save_metadata()
            return None

        try:
            tensor = torch.load(file_path, map_location="cpu", weights_only=True)
            logger.debug(f"已从磁盘加载: {file_path.name}")
            return tensor
        except Exception as e:
            logger.error(f"从磁盘加载失败: {e}")
            return None

    def contains(self, key: str) -> bool:
        """检查缓存中是否存在指定键
        
        Args:
            key: 缓存键
            
        Returns:
            True如果存在，否则False
        """
        in_memory = key in self._memory_cache
        in_disk = (
            self.enable_disk_cache 
            and key in self._disk_metadata
        )
        return in_memory or in_disk

    def remove(self, key: str) -> bool:
        """从缓存中移除指定键
        
        Args:
            key: 要移除的缓存键
            
        Returns:
            True如果成功移除，False如果键不存在
        """
        removed = False

        # 从内存移除
        if key in self._memory_cache:
            del self._memory_cache[key]
            removed = True

        # 从磁盘移除
        if key in self._disk_metadata:
            file_path = Path(self._disk_metadata[key]["file"])
            if file_path.exists():
                file_path.unlink()
            del self._disk_metadata[key]
            self._save_metadata()
            removed = True

        if removed:
            logger.debug(f"已移除缓存: {key[:16]}...")

        return removed

    def clear(self, clear_disk: bool = False) -> None:
        """清空所有缓存
        
        Args:
            clear_disk: 是否同时清空磁盘缓存
        """
        # 清空内存缓存
        self._memory_cache.clear()
        logger.info("内存缓存已清空")

        # 清空磁盘缓存
        if clear_disk and self.disk_cache_dir and self.disk_cache_dir.exists():
            for file in self.disk_cache_dir.glob("*.pt"):
                file.unlink()
            self._disk_metadata.clear()
            self._save_metadata()
            logger.info("磁盘缓存已清空")

    def cleanup_disk_cache(self, max_size_mb: Optional[float] = None) -> int:
        """清理磁盘缓存，确保不超过最大大小限制
        
        Args:
            max_size_mb: 最大大小限制(MB)，使用实例默认值如果为None
            
        Returns:
            删除的文件数量
        """
        if not self.enable_disk_cache or not self.disk_cache_dir:
            return 0

        max_size_mb = max_size_mb or self.disk_cache_max_size_mb
        max_bytes = max_size_mb * 1024 * 1024

        # 计算当前总大小并按时间排序
        entries = []
        total_size = 0
        for key, meta in self._disk_metadata.items():
            file_path = Path(meta["file"])
            if file_path.exists():
                size = file_path.stat().st_size
                entries.append((key, meta["timestamp"], size))
                total_size += size

        # 如果超出限制，按时间顺序删除最旧的
        removed_count = 0
        entries.sort(key=lambda x: x[1])  # 按时间戳排序

        for key, timestamp, size in entries:
            if total_size <= max_bytes:
                break

            self.remove(key)
            total_size -= size
            removed_count += 1

        if removed_count > 0:
            logger.info(
                f"磁盘缓存清理完成: "
                f"删除了 {removed_count} 个文件, "
                f"释放空间: {(total_size - sum(e[2] for e in entries)) / 1024 / 1024:.2f} MB"
            )

        return removed_count

    def get_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            包含各项统计指标的字典
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (
            self._stats["hits"] / total_requests * 100
            if total_requests > 0
            else 0.0
        )

        # 计算内存占用
        memory_usage_mb = 0.0
        for tensor in self._memory_cache.values():
            memory_usage_mb += tensor.nelement() * tensor.element_size() / 1024 / 1024

        # 计算磁盘占用
        disk_usage_mb = 0.0
        if self.enable_disk_cache:
            for meta in self._disk_metadata.values():
                disk_usage_mb += meta.get("size_bytes", 0) / 1024 / 1024

        stats = {
            **self._stats,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_max": self.max_size,
            "memory_usage_mb": round(memory_usage_mb, 2),
            "disk_cache_enabled": self.enable_disk_cache,
            "disk_cache_size": len(self._disk_metadata),
            "disk_usage_mb": round(disk_usage_mb, 2),
        }

        return stats

    def print_statistics(self) -> None:
        """打印格式化的缓存统计信息"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("特征缓存统计信息")
        print("=" * 60)
        print(f"总请求数: {stats['total_requests']}")
        print(f"命中率: {stats['hit_rate']:.2f}%")
        print(f"  - 命中次数: {stats['hits']}")
        print(f"  - 未命中次数: {stats['misses']}")
        print("-" * 60)
        print(f"内存缓存:")
        print(f"  - 当前条目: {stats['memory_cache_size']}/{stats['memory_cache_max']}")
        print(f"  - 内存占用: {stats['memory_usage_mb']:.2f} MB")
        print(f"  - 淘汰次数: {stats['evictions']}")
        if stats['disk_cache_enabled']:
            print("-" * 60)
            print(f"磁盘缓存:")
            print(f"  - 当前条目: {stats['disk_cache_size']}")
            print(f"  - 磁盘占用: {stats['disk_usage_mb']:.2f} MB")
            print(f"  - 磁盘命中: {stats['disk_hits']}")
            print(f"  - 磁盘存储: {stats['disk_stores']}")
        print("=" * 60 + "\n")

    def export_keys(self) -> List[str]:
        """导出所有缓存键列表
        
        Returns:
            所有缓存键的列表（内存+磁盘）
        """
        keys = list(self._memory_cache.keys())
        if self.enable_disk_cache:
            keys.extend(self._disk_metadata.keys())
        return list(set(keys))

    def __len__(self) -> int:
        """返回缓存中的总条目数"""
        count = len(self._memory_cache)
        if self.enable_disk_cache:
            count += len(self._disk_metadata)
        return count

    def __contains__(self, key: str) -> bool:
        """支持 'in' 操作符"""
        return self.contains(key)

    def __repr__(self) -> str:
        """返回缓存的字符串表示"""
        return (
            f"FeatureCache("
            f"size={len(self)}, "
            f"max={self.max_size}, "
            f"disk={self.enable_disk_cache})"
        )


class CachedFeatureExtractor:
    """带缓存的特征提取器装饰器
    
    包装任何特征提取器，自动缓存提取结果。
    
    Attributes:
        extractor: 底层特征提取器
        cache: 特征缓存实例
        
    Example:
        >>> base_extractor = ResNetFeatureExtractor('resnet50')
        >>> cached_extractor = CachedFeatureExtractor(base_extractor)
        >>> features = cached_extractor.extract(images)  # 首次调用，计算并缓存
        >>> features_again = cached_extractor.extract(images)  # 从缓存返回
    """

    def __init__(
        self,
        extractor: Any,
        cache: Optional[FeatureCache] = None,
        **cache_kwargs: Any,
    ) -> None:
        """初始化带缓存的特征提取器
        
        Args:
            extractor: 被包装的特征提取器实例
            cache: 已有的缓存实例，如果为None则创建新缓存
            **cache_kwargs: 创建新缓存时的参数
        """
        self.extractor = extractor
        self.cache = cache or FeatureCache(**cache_kwargs)

        logger.info(
            f"带缓存的特征提取器初始化完成, "
            f"底层提取器: {type(extractor).__name__}"
        )

    def extract(
        self,
        images: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """提取特征（带缓存）
        
        Args:
            images: 输入图像张量
            use_cache: 是否使用缓存，默认True
            
        Returns:
            提取的特征张量
        """
        if not use_cache:
            return self.extractor.extract_features(images)

        # 生成缓存键
        cache_key = FeatureCache.generate_hash(images)

        # 尝试从缓存获取
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # 未命中，执行实际提取
        features = self.extractor.extract_features(images)

        # 存入缓存
        self.cache.store(cache_key, features)

        return features

    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.cache.get_statistics()

    def clear_cache(self, clear_disk: bool = False) -> None:
        """清空缓存"""
        self.cache.clear(clear_disk=clear_disk)

    def __getattr__(self, name: str) -> Any:
        """代理底层提取器的属性访问"""
        return getattr(self.extractor, name)

    def __repr__(self) -> str:
        """返回字符串表示"""
        return (
            f"CachedFeatureExtractor("
            f"extractor={type(self.extractor).__name__}, "
            f"cache_entries={len(self.cache)})"
        )
