"""评估模块 - 生成模型评估工具包

提供完整的生成模型评估解决方案，包括:
- 核心指标计算 (FID, IS, Precision & Recall)
- 批量评估脚本 (命令行接口)
- HTML/PDF 报告生成
- 可视化和分析工具

模块结构:
    gms.evaluation
    ├── metrics/              # 核心指标实现
    │   ├── fid_score.py      # FID 计算
    │   ├── is_score.py       # IS 计算
    │   └── precision_recall.py  # P&R 计算
    ├── evaluation_script.py  # 批量评估 CLI
    └── report_generator.py   # 报告生成器

快速开始:
    >>> from gms.evaluation import (
    ...     run_evaluation,
    ...     EvaluationReportGenerator,
    ...     FIDCalculator,
    ... )
    >>>
    >>> # 运行批量评估
    >>> results = run_evaluation(config)
    >>>
    >>> # 生成报告
    >>> generator = EvaluationReportGenerator()
    >>> report_path = generator.generate_report(results)

命令行使用:
    $ python -m gms.evaluation.evaluation_script \\
        --real_data_path ./real_images \\
        --generated_path ./gen_images \\
        --metrics fid is precision recall \\
        --device cuda
"""

from .metrics import (
    FIDCalculator,
    ISCalculator,
    PrecisionRecallCalculator,
    DistanceMetric,
    calculate_fid_quick,
    calculate_is_quick,
    calculate_precision_recall_quick,
)

from .evaluation_script import (
    run_evaluation,
    EvaluationConfig,
    parse_arguments,
    main as cli_main,
    load_images_from_directory,
)

from .report_generator import (
    EvaluationReportGenerator,
    generate_quick_report,
)

__all__ = [
    # Metrics
    "FIDCalculator",
    "ISCalculator",
    "PrecisionRecallCalculator",
    "DistanceMetric",
    "calculate_fid_quick",
    "calculate_is_quick",
    "calculate_precision_recall_quick",
    # Script
    "run_evaluation",
    "EvaluationConfig",
    "parse_arguments",
    "cli_main",
    "load_images_from_directory",
    # Report
    "EvaluationReportGenerator",
    "generate_quick_report",
]
