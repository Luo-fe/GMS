"""批量评估脚本 - 命令行接口

提供完整的生成模型评估流程，支持多种指标的批量计算和结果报告。
可通过命令行或 Python API 调用。

功能:
    - 自动加载真实图像和生成图像
    - 计算 FID, IS, Precision, Recall 等指标
    - 保存结果到 JSON/CSV 格式
    - 生成详细的 HTML/PDF 评估报告
    - 支持多 GPU 和批处理优化

使用方式:
    命令行:
        $ python -m gms.evaluation.evaluation_script \\
            --real_data_path /path/to/real/images \\
            --generated_path /path/to/generated/images \\
            --metrics fid is precision recall \\
            --device cuda \\
            --output_dir ./results

    Python API:
        >>> from gms.evaluation.evaluation_script import run_evaluation
        >>> results = run_evaluation(
        ...     real_data_path='./real_images',
        ...     generated_path='./gen_images',
        ...     metrics=['fid', 'is', 'precision'],
        ...     device='cuda'
        ... )
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import numpy as np
import torch
from PIL import Image

from .metrics.fid_score import FIDCalculator
from .metrics.is_score import ISCalculator
from .metrics.precision_recall import PrecisionRecallCalculator

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """配置日志系统

    Args:
        log_level: 日志级别 ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_images_from_directory(
    directory_path: str,
    num_samples: Optional[int] = None,
    image_extensions: tuple = (".png", ".jpg", ".jpeg", ".bmp", ".webp"),
) -> torch.Tensor:
    """从目录加载图像

    加载指定目录中的所有图像文件，转换为张量格式。

    Args:
        directory_path: 图像目录路径
        num_samples: 要加载的图像数量 (None 表示全部)
        image_extensions: 支持的图像扩展名

    Returns:
        图像张量 (N, C, H, W)，值域 [0, 1]

    Raises:
        FileNotFoundError: 如果目录不存在或为空
        ValueError: 如果无法加载任何图像
    """
    directory = Path(directory_path)

    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory_path}")

    # 收集所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))

    # 去重（Windows 系统可能大小写不敏感）
    image_files = list(set(image_files))

    if len(image_files) == 0:
        raise ValueError(f"目录中没有找到图像文件: {directory_path}")

    logger.info(f"找到 {len(image_files)} 张图像在 {directory_path}")

    # 随机采样如果指定了数量
    if num_samples is not None and num_samples < len(image_files):
        np.random.seed(42)
        indices = np.random.choice(len(image_files), size=num_samples, replace=False)
        image_files = [image_files[i] for i in indices]
        logger.info(f"随机采样 {num_samples} 张图像")

    # 加载图像
    images = []
    for img_path in sorted(image_files):
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            # 转换为 (C, H, W) 格式
            img_tensor = img_tensor.permute(2, 0, 1)
            images.append(img_tensor)
        except Exception as e:
            logger.warning(f"无法加载图像 {img_path}: {e}")
            continue

    if len(images) == 0:
        raise ValueError("没有成功加载任何图像")

    images_tensor = torch.stack(images)
    logger.info(f"成功加载 {len(images)} 张图像, shape={images_tensor.shape}")

    return images_tensor


def load_images_from_file_list(
    file_list_path: str,
    num_samples: Optional[int] = None,
) -> torch.Tensor:
    """从文件列表加载图像

    从包含图像路径列表的文本文件中加载图像。

    Args:
        file_list_path: 文件列表路径（每行一个路径）
        num_samples: 要加载的图像数量

    Returns:
        图像张量 (N, C, H, W)
    """
    with open(file_list_path, "r") as f:
        image_paths = [line.strip() for line in f if line.strip()]

    if num_samples is not None and num_samples < len(image_paths):
        np.random.seed(42)
        indices = np.random.choice(len(image_paths), size=num_samples, replace=False)
        image_paths = [image_paths[i] for i in indices]

    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            images.append(img_tensor)
        except Exception as e:
            logger.warning(f"无法加载图像 {img_path}: {e}")
            continue

    if len(images) == 0:
        raise ValueError("没有成功加载任何图像")

    return torch.stack(images)


class EvaluationConfig:
    """评估配置类

    存储所有评估参数和配置选项。

    Attributes:
        real_data_path: 真实数据路径
        generated_path: 生成数据路径
        metrics: 要计算的指标列表
        batch_size: 批处理大小
        num_samples: 样本数量
        device: 计算设备
        output_dir: 输出目录
        save_report: 是否保存报告
    """

    def __init__(
        self,
        real_data_path: str,
        generated_path: str,
        metrics: List[str],
        batch_size: int = 50,
        num_samples: Optional[int] = None,
        device: str = "cpu",
        output_dir: str = "./evaluation_results",
        save_report: bool = True,
        k_pr: int = 3,
        is_splits: int = 10,
    ) -> None:
        self.real_data_path = real_data_path
        self.generated_path = generated_path
        self.metrics = metrics
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device
        self.output_dir = output_dir
        self.save_report = save_report
        self.k_pr = k_pr
        self.is_splits = is_splits


def run_evaluation(config: EvaluationConfig) -> Dict[str, Any]:
    """执行完整的评估流程

    根据配置执行所有请求的指标计算。

    Args:
        config: EvaluationConfig 实例

    Returns:
        包含所有结果的字典:
            - metrics: 各指标的数值结果
            - config: 使用的配置
            - timestamp: 时间戳
            - duration: 总耗时（秒）
    """
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info("=" * 60)
    logger.info("开始 GMS 评估流程")
    logger.info(f"时间戳: {timestamp}")
    logger.info("=" * 60)

    results = {
        "config": {
            "real_data_path": config.real_data_path,
            "generated_path": config.generated_path,
            "metrics": config.metrics,
            "device": config.device,
            "num_samples": config.num_samples,
        },
        "timestamp": timestamp,
        "metrics": {},
    }

    try:
        # 加载数据
        logger.info("\n" + "=" * 40)
        logger.info("步骤 1: 加载数据")
        logger.info("=" * 40)

        logger.info(f"加载真实数据: {config.real_data_path}")
        real_images = load_images_from_directory(
            config.real_data_path,
            num_samples=config.num_samples,
        )

        logger.info(f"加载生成数据: {config.generated_path}")
        gen_images = load_images_from_directory(
            config.generated_path,
            num_samples=config.num_samples,
        )

        results["config"]["num_real_images"] = len(real_images)
        results["config"]["num_gen_images"] = len(gen_images)

        # 初始化计算器
        logger.info("\n" + "=" * 40)
        logger.info("步骤 2: 初始化计算器")
        logger.info("=" * 40)

        fid_calculator = None
        if any(m in ["fid", "all", "precision", "recall"] for m in config.metrics):
            logger.info("初始化 FID 计算器...")
            fid_calculator = FIDCalculator(
                device=config.device,
                batch_size=config.batch_size,
            )

        is_calculator = None
        if any(m in ["is", "all"] for m in config.metrics):
            logger.info("初始化 IS 计算器...")
            is_calculator = ISCalculator(
                device=config.device,
                batch_size=config.batch_size,
            )

        pr_calculator = None
        if any(m in ["precision", "recall", "all"] for m in config.metrics):
            logger.info("初始化 P&R 计算器...")
            pr_calculator = PrecisionRecallCalculator(
                device=config.device,
                k=config.k_pr,
            )

        # 计算各项指标
        logger.info("\n" + "=" * 40)
        logger.info("步骤 3: 计算评估指标")
        logger.info("=" * 40)

        # FID
        if "fid" in config.metrics or "all" in config.metrics:
            logger.info("\n--- 计算 FID ---")
            try:
                fid_start = time.time()
                fid_value = fid_calculator.calculate_fid(real_images, gen_images)
                fid_time = time.time() - fid_start

                results["metrics"]["fid"] = {
                    "value": float(fid_value),
                    "computation_time": round(fid_time, 2),
                }
                logger.info(f"FID: {fid_value:.4f} (耗时 {fid_time:.2f}s)")
            except Exception as e:
                logger.error(f"FID 计算失败: {e}")
                results["metrics"]["fid"] = {"error": str(e)}

        # IS
        if "is" in config.metrics or "all" in config.metrics:
            logger.info("\n--- 计算 IS ---")
            try:
                is_start = time.time()
                is_mean, is_std = is_calculator.calculate_is(
                    gen_images, splits=config.is_splits
                )
                is_time = time.time() - is_start

                results["metrics"]["is"] = {
                    "mean": float(is_mean),
                    "std": float(is_std),
                    "splits": config.is_splits,
                    "computation_time": round(is_time, 2),
                }
                logger.info(
                    f"IS: {is_mean:.4f} ± {is_std:.4f} (耗时 {is_time:.2f}s)"
                )
            except Exception as e:
                logger.error(f"IS 计算失败: {e}")
                results["metrics"]["is"] = {"error": str(e)}

        # Precision & Recall
        if any(m in ["precision", "recall", "all"] for m in config.metrics):
            logger.info("\n--- 计算 Precision & Recall ---")
            try:
                pr_start = time.time()
                pr_results = pr_calculator.calculate_precision_recall(
                    real_images,
                    gen_images,
                    feature_extractor=fid_calculator,
                )
                pr_time = time.time() - pr_start

                results["metrics"]["precision_recall"] = {
                    **{k: float(v) for k, v in pr_results.items()},
                    "computation_time": round(pr_time, 2),
                }
                logger.info(
                    f"Precision: {pr_results['precision']:.4f}, "
                    f"Recall: {pr_results['recall']:.4f} "
                    f"(耗时 {pr_time:.2f}s)"
                )
            except Exception as e:
                logger.error(f"P&R 计算失败: {e}")
                results["metrics"]["precision_recall"] = {"error": str(e)}

        # 保存结果
        total_time = time.time() - start_time
        results["duration"] = round(total_time, 2)

        logger.info("\n" + "=" * 40)
        logger.info("步骤 4: 保存结果")
        logger.info("=" * 40)

        if config.save_report:
            _save_results(results, config.output_dir)

        logger.info("\n" + "=" * 60)
        logger.info("评估完成!")
        logger.info(f"总耗时: {total_time:.2f}s")
        logger.info("=" * 60)

        return results

    except Exception as e:
        logger.error(f"评估过程中发生错误: {e}", exc_info=True)
        results["error"] = str(e)
        return results


def _save_results(
    results: Dict[str, Any], output_dir: str
) -> None:
    """保存评估结果到文件

    将结果保存为 JSON 和 CSV 格式。

    Args:
        results: 评估结果字典
        output_dir: 输出目录路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存 JSON
    json_path = output_path / f"evaluation_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存到: {json_path}")

    # 保存 CSV 摘要
    csv_path = output_path / f"evaluation_summary_{timestamp}.csv"
    _save_csv_summary(results, csv_path)
    logger.info(f"摘要已保存到: {csv_path}")


def _save_csv_summary(
    results: Dict[str, Any], csv_path: Path
) -> None:
    """将结果摘要保存为 CSV

    Args:
        results: 结果字典
        csv_path: CSV 文件路径
    """
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value", "Details"])

        if "fid" in results.get("metrics", {}):
            fid_data = results["metrics"]["fid"]
            if "value" in fid_data:
                writer.writerow(["FID", f"{fid_data['value']:.4f}", ""])

        if "is" in results.get("metrics", {}):
            is_data = results["metrics"]["is"]
            if "mean" in is_data:
                writer.writerow([
                    "IS",
                    f"{is_data['mean']:.4f} ± {is_data['std']:.4f}",
                    f"splits={is_data.get('splits', 'N/A')}"
                ])

        if "precision_recall" in results.get("metrics", {}):
            pr_data = results["metrics"]["precision_recall"]
            if "precision" in pr_data:
                writer.writerow(["Precision", f"{pr_data['precision']:.4f}", ""])
            if "recall" in pr_data:
                writer.writerow(["Recall", f"{pr_data['recall']:.4f}", ""])

        writer.writerow([])
        writer.writerow(["Total Time", f"{results.get('duration', 'N/A')}s", ""])
        writer.writerow(["Timestamp", results.get("timestamp", "N/A"), ""])


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(
        description="GMS 批量评估工具 - 计算生成模型的评估指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 计算 FID 和 IS
  python -m gms.evaluation.evaluation_script \\
      --real_data_path ./data/real \\
      --generated_path ./data/generated \\
      --metrics fid is \\
      --device cuda

  # 计算所有指标
  python -m gms.evaluation.evaluation_script \\
      --real_data_path ./data/real \\
      --generated_path ./data/generated \\
      --metrics all \\
      --batch_size 64 \\
      --output_dir ./results
        """,
    )

    # 必需参数
    parser.add_argument(
        "--real_data_path",
        type=str,
        required=True,
        help="真实图像的目录路径",
    )
    parser.add_argument(
        "--generated_path",
        type=str,
        required=True,
        help="生成图像的目录路径",
    )

    # 可选参数
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["all"],
        choices=["fid", "is", "precision", "recall", "all"],
        help="要计算的指标 (默认: all)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="批处理大小 (默认: 50)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="用于评估的最大样本数 (默认: 全部)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="结果输出目录 (默认: ./evaluation_results)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="计算设备 (默认: cpu)",
    )
    parser.add_argument(
        "--save_report",
        action="store_true",
        default=True,
        help="是否保存详细报告 (默认: True)",
    )
    parser.add_argument(
        "--k_pr",
        type=int,
        default=3,
        help="Precision & Recall 的 k 值 (默认: 3)",
    )
    parser.add_argument(
        "--is_splits",
        type=int,
        default=10,
        help="IS 的 splits 数量 (默认: 10)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)",
    )

    return parser.parse_args()


def main() -> None:
    """主函数 - CLI 入口点"""
    args = parse_arguments()

    setup_logging(args.log_level)

    logger.info("GMS 批量评估工具")
    logger.info("=" * 60)

    config = EvaluationConfig(
        real_data_path=args.real_data_path,
        generated_path=args.generated_path,
        metrics=args.metrics,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device,
        output_dir=args.output_dir,
        save_report=args.save_report,
        k_pr=args.k_pr,
        is_splits=args.is_splits,
    )

    results = run_evaluation(config)

    # 打印摘要到控制台
    print("\n" + "=" * 60)
    print("评估结果摘要")
    print("=" * 60)

    if "metrics" in results:
        for metric_name, metric_data in results["metrics"].items():
            if "error" in metric_data:
                print(f"{metric_name.upper()}: 错误 - {metric_data['error']}")
            elif metric_name == "fid":
                print(f"FID: {metric_data['value']:.4f}")
            elif metric_name == "is":
                print(
                    f"IS: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}"
                )
            elif metric_name == "precision_recall":
                print(f"Precision: {metric_data.get('precision', 'N/A'):.4f}")
                print(f"Recall: {metric_data.get('recall', 'N/A'):.4f}")
                print(f"Density: {metric_data.get('density', 'N/A'):.6f}")

    print("-" * 60)
    print(f"总耗时: {results.get('duration', 'N/A')}s")
    print(f"结果保存在: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
