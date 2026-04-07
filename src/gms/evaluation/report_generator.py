"""评估报告生成器

生成详细的 HTML/PDF 评估报告，包含:
- 项目和实验信息概览
- 所有指标的数值结果和可视化图表
- 与基线方法的对比
- 生成样本展示网格
- 分布对比图 (直方图、散点图等)
- 训练曲线 (如果有历史记录)
- 结论和建议

支持自定义模板、多语言输出和多种导出格式。

Example:
    >>> from gms.evaluation.report_generator import EvaluationReportGenerator
    >>>
    >>> generator = EvaluationReportGenerator(output_dir='./reports')
    >>> report_path = generator.generate_report(
    ...     evaluation_results=results,
    ...     experiment_name='GMM_Diffusion_v1',
    ...     generated_samples=gen_images
    ... )
    >>> print(f"报告已保存到: {report_path}")
"""

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


class EvaluationReportGenerator:
    """评估报告生成器

    生成包含指标结果、可视化和分析的 HTML/PDF 报告。

    Attributes:
        output_dir: 输出目录
        template_dir: 模板目录 (可选)
        language: 报告语言 ('zh' 或 'en')
        include_plots: 是否包含可视化图表

    Example:
        >>> generator = EvaluationReportGenerator(
        ...     output_dir='./reports',
        ...     language='zh'
        ... )
        >>> html_path = generator.generate_report(results)
    """

    def __init__(
        self,
        output_dir: str = "./reports",
        template_dir: Optional[str] = None,
        language: str = "zh",
        include_plots: bool = True,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
    ) -> None:
        """初始化报告生成器

        Args:
            output_dir: 输出目录路径
            template_dir: 自定义模板目录 (可选)
            language: 报告语言，'zh' 或 'en'
            include_plots: 是否在报告中包含图表
            figsize: 图表默认大小 (宽, 高)
            dpi: 图像分辨率

        Raises:
            ValueError: 如果语言参数不支持
        """
        if language not in ["zh", "en"]:
            raise ValueError(f"不支持的语言 '{language}'，可选值: 'zh', 'en'")

        self.output_dir = Path(output_dir)
        self.template_dir = Path(template_dir) if template_dir else None
        self.language = language
        self.include_plots = include_plots
        self.figsize = figsize
        self.dpi = dpi

        # 语言配置
        self._texts = {
            "zh": {
                "title": "GMS 评估报告",
                "overview": "实验概览",
                "metrics_results": "指标结果",
                "visualization": "可视化分析",
                "sample_gallery": "生成样本展示",
                "conclusions": "结论与建议",
                "timestamp": "生成时间",
                "experiment_name": "实验名称",
                "device": "计算设备",
                "num_samples": "样本数量",
                "computation_time": "计算耗时",
                "fid_score": "FID 分数",
                "is_score": "Inception Score",
                "precision": "精确度 (Precision)",
                "recall": "召回率 (Recall)",
                "density": "密度 (Density)",
                "lower_better": "(越低越好)",
                "higher_better": "(越高越好)",
                "baseline_comparison": "基线方法对比",
                "recommendations": "改进建议",
                "no_data": "暂无数据",
            },
            "en": {
                "title": "GMS Evaluation Report",
                "overview": "Experiment Overview",
                "metrics_results": "Metrics Results",
                "visualization": "Visualization Analysis",
                "sample_gallery": "Generated Samples Gallery",
                "conclusions": "Conclusions & Recommendations",
                "timestamp": "Generated at",
                "experiment_name": "Experiment Name",
                "device": "Device",
                "num_samples": "Number of Samples",
                "computation_time": "Computation Time",
                "fid_score": "FID Score",
                "is_score": "Inception Score",
                "precision": "Precision",
                "recall": "Recall",
                "density": "Density",
                "lower_better": "(lower is better)",
                "higher_better": "(higher is better)",
                "baseline_comparison": "Baseline Comparison",
                "recommendations": "Recommendations",
                "no_data": "No data available",
            },
        }

        logger.info(
            f"初始化 ReportGenerator: output={output_dir}, "
            f"lang={language}, plots={'启用' if include_plots else '禁用'}"
        )

    def _t(self, key: str) -> str:
        """获取翻译文本"""
        return self._texts.get(self.language, self._texts["zh"]).get(key, key)

    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        experiment_name: str = "GMS_Evaluation",
        generated_samples: Optional[torch.Tensor] = None,
        baseline_results: Optional[Dict[str, float]] = None,
        training_history: Optional[Dict[str, List[float]]] = None,
        custom_logo_path: Optional[str] = None,
    ) -> str:
        """生成完整的 HTML 评估报告

        Args:
            evaluation_results: 评估结果字典 (来自 run_evaluation)
            experiment_name: 实验名称
            generated_samples: 生成的图像样本 (用于展示)
            baseline_results: 基线方法的结果字典 (用于对比)
            training_history: 训练历史记录 (用于绘制曲线)
            custom_logo_path: 自定义 logo 图片路径

        Returns:
            生成的 HTML 文件路径

        Raises:
            ValueError: 如果 evaluation_results 为空
        """
        if not evaluation_results:
            raise ValueError("evaluation_results 不能为空")

        logger.info(f"开始生成报告: {experiment_name}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{experiment_name}_report_{timestamp}.html"
        report_path = self.output_dir / report_filename

        # 构建 HTML 内容
        html_content = self._build_html_report(
            evaluation_results=evaluation_results,
            experiment_name=experiment_name,
            generated_samples=generated_samples,
            baseline_results=baseline_results,
            training_history=training_history,
            custom_logo_path=custom_logo_path,
        )

        # 写入文件
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML 报告已生成: {report_path}")

        # 尝试转换为 PDF (可选)
        pdf_path = self._try_convert_to_pdf(report_path)

        return str(report_path)

    def _build_html_report(
        self,
        evaluation_results: Dict[str, Any],
        experiment_name: str,
        generated_samples: Optional[torch.Tensor],
        baseline_results: Optional[Dict],
        training_history: Optional[Dict],
        custom_logo_path: Optional[str],
    ) -> str:
        """构建完整的 HTML 报告内容"""

        sections = []

        # 1. 头部和样式
        sections.append(self._generate_header(experiment_name))

        # 2. 实验概览
        sections.append(self._generate_overview_section(evaluation_results))

        # 3. 指标结果表格
        sections.append(self._generate_metrics_table(evaluation_results))

        # 4. 可视化图表
        if self.include_plots:
            sections.append(self._generate_visualization_section(
                evaluation_results, training_history
            ))

        # 5. 基线对比
        if baseline_results:
            sections.append(self._generate_baseline_comparison(
                evaluation_results, baseline_results
            ))

        # 6. 样本展示
        if generated_samples is not None:
            sections.append(self._generate_sample_gallery(generated_samples))

        # 7. 结论和建议
        sections.append(self._generate_conclusions(evaluation_results))

        # 8. 页脚
        sections.append(self._generate_footer())

        return "\n".join(sections)

    def _generate_header(self, experiment_name: str) -> str:
        """生成 HTML 头部（含 CSS 样式）"""
        return f"""<!DOCTYPE html>
<html lang="{self.language}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{experiment_name} - {self._t('title')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f7fa;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .section {{
            padding: 30px 40px;
            border-bottom: 1px solid #eee;
        }}
        .section h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-value {{
            font-weight: bold;
            font-size: 1.1em;
        }}
        .good {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .bad {{ color: #dc3545; }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .grid-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .grid-gallery img {{
            width: 100%;
            height: 150px;
            object-fit: cover;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        .grid-gallery img:hover {{
            transform: scale(1.05);
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .info-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .info-card label {{
            display: block;
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .info-card value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #333;
        }}
        .footer {{
            background: #333;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }}
        .recommendation-list {{
            list-style: none;
            padding: 0;
        }}
        .recommendation-list li {{
            padding: 10px 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-left: 4px solid #28a745;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>{experiment_name}</h1>
        <p>{self._t('title')}</p>
    </div>"""

    def _generate_overview_section(self, results: Dict) -> str:
        """生成实验概览部分"""
        config = results.get("config", {})
        timestamp = results.get("timestamp", "N/A")
        duration = results.get("duration", "N/A")

        info_items = [
            (self._t("experiment_name"), config.get("real_data_path", "N/A")),
            (self._t("device"), config.get("device", "N/A")),
            (
                self._t("num_samples"),
                f"Real: {config.get('num_real_images', 'N/A')}, "
                f"Gen: {config.get('num_gen_images', 'N/A')}",
            ),
            (self._t("computation_time"), f"{duration}s"),
            (self._t("timestamp"), timestamp),
        ]

        info_cards = ""
        for label, value in info_items:
            info_cards += f"""
            <div class="info-card">
                <label>{label}</label>
                <value>{value}</value>
            </div>"""

        return f"""
    <div class="section">
        <h2>{self._t('overview')}</h2>
        <div class="info-grid">
            {info_cards}
        </div>
    </div>"""

    def _generate_metrics_table(self, results: Dict) -> str:
        """生成指标结果表格"""
        metrics = results.get("metrics", {})

        rows = ""

        # FID
        if "fid" in metrics and "value" in metrics["fid"]:
            fid_val = metrics["fid"]["value"]
            fid_class = self._get_value_class(fid_val, lower_better=True)
            rows += f"""
            <tr>
                <td><strong>{self._t('fid_score')}</strong></td>
                <td class="metric-value {fid_class}">{fid_val:.4f}</td>
                <td>{self._t('lower_better')}</td>
                <td>{metrics['fid'].get('computation_time', 'N/A')}s</td>
            </tr>"""

        # IS
        if "is" in metrics and "mean" in metrics["is"]:
            is_mean = metrics["is"]["mean"]
            is_std = metrics["is"]["std"]
            is_class = self._get_value_class(is_mean, lower_better=False)
            rows += f"""
            <tr>
                <td><strong>{self._t('is_score')}</strong></td>
                <td class="metric-value {is_class}">{is_mean:.4f} ± {is_std:.4f}</td>
                <td>{self._t('higher_better')}</td>
                <td>splits={metrics['is'].get('splits', 'N/A')}</td>
            </tr>"""

        # Precision & Recall
        if "precision_recall" in metrics:
            pr = metrics["precision_recall"]
            if "precision" in pr:
                prec_class = self._get_value_class(pr["precision"], lower_better=False)
                rows += f"""
            <tr>
                <td><strong>{self._t('precision')}</strong></td>
                <td class="metric-value {prec_class}">{pr['precision']:.4f}</td>
                <td>{self._t('higher_better')}</td>
                <td>k={pr.get('manifold_radius', 'N/A')}</td>
            </tr>"""
            if "recall" in pr:
                rec_class = self._get_value_class(pr["recall"], lower_better=False)
                rows += f"""
            <tr>
                <td><strong>{self._t('recall')}</strong></td>
                <td class="metric-value {rec_class}">{pr['recall']:.4f}</td>
                <td>{self._t('higher_better')}</td>
                <td>-</td>
            </tr>"""
            if "density" in pr:
                rows += f"""
            <tr>
                <td><strong>{self._t('density')}</strong></td>
                <td class="metric-value">{pr['density']:.6f}</td>
                <td>-</td>
                <td>-</td>
            </tr>"""

        if not rows:
            rows = f'<tr><td colspan="4">{self._t("no_data")}</td></tr>'

        return f"""
    <div class="section">
        <h2>{self._t('metrics_results')}</h2>
        <table>
            <thead>
                <tr>
                    <th>指标 (Metric)</th>
                    <th>数值 (Value)</th>
                    <th>说明 (Note)</th>
                    <th>详情 (Details)</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>"""

    def _generate_visualization_section(
        self, results: Dict, history: Optional[Dict]
    ) -> str:
        """生成可视化图表部分"""
        plots_html = ""

        try:
            # 1. 指标柱状图
            plot_base64 = self._create_metrics_bar_chart(results)
            if plot_base64:
                plots_html += f"""
            <div class="plot-container">
                <h3>指标对比图</h3>
                <img src="data:image/png;base64,{plot_base64}" alt="Metrics Bar Chart">
            </div>"""

            # 2. 训练曲线 (如果有)
            if history:
                plot_base64 = self._create_training_curves(history)
                if plot_base64:
                    plots_html += f"""
            <div class="plot-container">
                <h3>训练曲线</h3>
                <img src="data:image/png;base64,{plot_base64}" alt="Training Curves">
            </div>"""

        except Exception as e:
            logger.warning(f"生成图表时出错: {e}")
            plots_html += f"<p>图表生成失败: {e}</p>"

        if not plots_html:
            return ""

        return f"""
    <div class="section">
        <h2>{self._t('visualization')}</h2>
        {plots_html}
    </div>"""

    def _create_metrics_bar_chart(self, results: Dict) -> Optional[str]:
        """创建指标柱状图并返回 base64 编码的图片"""
        metrics = results.get("metrics", {})

        metric_names = []
        values = []
        colors = []

        if "fid" in metrics and "value" in metrics["fid"]:
            metric_names.append("FID")
            values.append(metrics["fid"]["value"])
            colors.append("#ff6b6b")

        if "is" in metrics and "mean" in metrics["is"]:
            metric_names.append("IS")
            values.append(metrics["is"]["mean"])
            colors.append("#4ecdc4")

        if "precision_recall" in metrics:
            pr = metrics["precision_recall"]
            if "precision" in pr:
                metric_names.append("Precision")
                values.append(pr["precision"])
                colors.append("#45b7d1")
            if "recall" in pr:
                metric_names.append("Recall")
                values.append(pr["recall"])
                colors.append("#96ceb4")

        if not metric_names:
            return None

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        bars = ax.bar(metric_names, values, color=colors, edgecolor="white", linewidth=1.5)

        ax.set_ylabel("Value", fontsize=12)
        ax.set_title("Evaluation Metrics Summary", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close()
        buf.seek(0)

        return base64.b64encode(buf.read()).decode("utf-8")

    def _create_training_curves(self, history: Dict) -> Optional[str]:
        """创建训练曲线图"""
        if not history:
            return None

        fig, axes = plt.subplots(
            len(history), 1, figsize=(self.figsize[0], self.figsize[1] * len(history)),
            dpi=self.dpi,
        )

        if len(history) == 1:
            axes = [axes]

        for idx, (metric_name, values) in enumerate(history.items()):
            axes[idx].plot(values, linewidth=2, color="#667eea")
            axes[idx].set_title(f"{metric_name} over Training", fontsize=12)
            axes[idx].set_xlabel("Epoch/Step")
            axes[idx].set_ylabel(metric_name)
            axes[idx].grid(alpha=0.3)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=self.dpi, bbox_inches="tight")
        plt.close()
        buf.seek(0)

        return base64.b64encode(buf.read()).decode("utf-8")

    def _generate_baseline_comparison(
        self, results: Dict, baseline: Dict
    ) -> str:
        """生成基线对比部分"""
        rows = ""
        for metric_name, baseline_val in baseline.items():
            current_val = None
            if metric_name.lower() == "fid":
                current_val = results.get("metrics", {}).get("fid", {}).get("value")
            elif metric_name.lower() == "is":
                current_val = results.get("metrics", {}).get("is", {}).get("mean")
            elif metric_name.lower() in ["precision", "recall"]:
                current_val = (
                    results.get("metrics", {})
                    .get("precision_recall", {})
                    .get(metric_name.lower())
                )

            if current_val is not None:
                diff = current_val - baseline_val
                diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                improvement = "✓" if (
                    (metric_name.lower() == "fid" and diff < 0)
                    or (metric_name.lower() != "fid" and diff > 0)
                ) else "✗"
                rows += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{baseline_val:.4f}</td>
                <td>{current_val:.4f}</td>
                <td>{diff_str} {improvement}</td>
            </tr>"""

        if not rows:
            return ""

        return f"""
    <div class="section">
        <h2>{self._t('baseline_comparison')}</h2>
        <table>
            <thead>
                <tr>
                    <th>指标</th>
                    <th>基线 (Baseline)</th>
                    <th>当前 (Current)</th>
                    <th>差异 (Difference)</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>"""

    def _generate_sample_gallery(self, samples: torch.Tensor) -> str:
        """生成样本展示网格"""
        n_samples = min(len(samples), 32)  # 最多显示 32 张
        indices = np.random.choice(len(samples), size=n_samples, replace=False)

        images_html = ""
        for idx in indices[:n_samples]:
            img = samples[idx]
            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)

            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            ax.imshow(img_np)
            ax.axis("off")

            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=80, bbox_inches="tight", pad_inches=0)
            plt.close()
            buf.seek(0)

            img_base64 = base64.b64encode(buf.read()).decode("utf-8")
            images_html += f'<img src="data:image/png;base64,{img_base64}" alt="Sample">'

        return f"""
    <div class="section">
        <h2>{self._t('sample_gallery')}</h2>
        <div class="grid-gallery">
            {images_html}
        </div>
    </div>"""

    def _generate_conclusions(self, results: Dict) -> str:
        """生成结论和建议部分"""
        recommendations = self._analyze_results(results)

        rec_list = ""
        for rec in recommendations:
            rec_list += f"<li>{rec}</li>"

        return f"""
    <div class="section">
        <h2>{self._t('conclusions')}</h2>
        <ul class="recommendation-list">
            {rec_list}
        </ul>
    </div>"""

    def _analyze_results(self, results: Dict) -> List[str]:
        """分析结果并生成建议"""
        recommendations = []
        metrics = results.get("metrics", {})

        # FID 分析
        if "fid" in metrics and "value" in metrics["fid"]:
            fid = metrics["fid"]["value"]
            if fid < 50:
                recommendations.append(
                    f"FID ({fid:.2f}) 非常低，生成质量优秀！"
                )
            elif fid < 100:
                recommendations.append(
                    f"FID ({fid:.2f}) 良好，但仍有提升空间。"
                    "建议检查模型架构或训练策略。"
                )
            else:
                recommendations.append(
                    f"FID ({fid:.2f}) 较高，需要改进。"
                    "建议：增加训练数据、调整超参数或使用更好的优化器。"
                )

        # IS 分析
        if "is" in metrics and "mean" in metrics["is"]:
            is_mean = metrics["is"]["mean"]
            if is_mean > 8:
                recommendations.append(
                    f"IS ({is_mean:.2f}) 很高，生成的图像清晰且多样。"
                )
            elif is_mean > 5:
                recommendations.append(
                    f"IS ({is_mean:.2f}) 中等。"
                    "可尝试增加模型容量或改进损失函数。"
                )
            else:
                recommendations.append(
                    f"IS ({is_mean:.2f}) 较低，图像质量需改善。"
                    "建议检查训练稳定性和收敛情况。"
                )

        # Precision & Recall 分析
        if "precision_recall" in metrics:
            pr = metrics["precision_recall"]
            if "precision" in pr and "recall" in pr:
                if pr["precision"] > 0.8 and pr["recall"] > 0.6:
                    recommendations.append(
                        "Precision 和 Recall 都很好，"
                        "模型在质量和多样性之间取得了良好平衡。"
                    )
                elif pr["precision"] > 0.8 and pr["recall"] < 0.4:
                    recommendations.append(
                        "Precision 高但 Recall 低："
                        "生成质量好但多样性不足，考虑增加采样随机性。"
                    )
                elif pr["precision"] < 0.5 and pr["recall"] > 0.6:
                    recommendations.append(
                        "Precision 低但 Recall 高："
                        "多样性好但存在模式崩塌风险，需提高生成质量。"
                    )

        if not recommendations:
            recommendations.append("请查看详细指标结果以获取更多洞察。")

        return recommendations

    def _get_value_class(
        self, value: float, lower_better: bool = True
    ) -> str:
        """根据数值返回 CSS 类名用于颜色编码"""
        if lower_better:
            if value < 50:
                return "good"
            elif value < 100:
                return "warning"
            else:
                return "bad"
        else:
            if value > 0.8:
                return "good"
            elif value > 0.5:
                return "warning"
            else:
                return "bad"

    def _generate_footer(self) -> str:
        """生成页脚"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""
    <div class="footer">
        <p>GMS Evaluation Report Generator | Generated at {timestamp}</p>
        <p>© GMS Team - Gaussian Mixture Solver</p>
    </div>
</div>
</body>
</html>"""

    def _try_convert_to_pdf(self, html_path: Path) -> Optional[str]:
        """尝试将 HTML 转换为 PDF (可选功能)"""
        try:
            from weasyprint import HTML

            pdf_path = html_path.with_suffix(".pdf")
            HTML(str(html_path)).write_pdf(str(pdf_path))
            logger.info(f"PDF 报告已生成: {pdf_path}")
            return str(pdf_path)
        except ImportError:
            logger.debug("weasyprint 未安装，跳过 PDF 生成")
            return None
        except Exception as e:
            logger.warning(f"PDF 生成失败: {e}")
            return None


def generate_quick_report(
    evaluation_results: Dict[str, Any],
    output_dir: str = "./reports",
    experiment_name: str = "GMS_Quick_Report",
) -> str:
    """快速生成报告的便捷函数

    一行代码生成评估报告。

    Args:
        evaluation_results: 评估结果
        output_dir: 输出目录
        experiment_name: 实验名称

    Returns:
        生成的报告文件路径

    Example:
        >>> path = generate_quick_report(results, output_dir='./reports')
    """
    generator = EvaluationReportGenerator(output_dir=output_dir)
    return generator.generate_report(
        evaluation_results=evaluation_results,
        experiment_name=experiment_name,
    )
