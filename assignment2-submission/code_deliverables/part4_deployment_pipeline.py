import tensorflow as tf
import json
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import os

# === 新增：matplotlib 可视化 ===
import matplotlib.pyplot as plt

@dataclass
class DeploymentTarget:
    """Configuration for different deployment targets."""
    name: str
    max_model_size_mb: float
    max_latency_ms: float
    max_memory_mb: float
    power_budget_mw: float
    compute_capability: str  # 'cloud', 'edge', 'tiny'


@dataclass
class OptimizationResult:
    """Results from model optimization."""
    model_path: str
    accuracy: float
    model_size_mb: float
    estimated_latency_ms: float
    memory_usage_mb: float
    optimization_strategy: str


class ModelOptimizer(ABC):
    @abstractmethod
    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        pass


class CloudOptimizer(ModelOptimizer):
    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        # 模拟：多 GPU 加速，吞吐量优先 -> 精度保持不变，延迟减少
        optimized_model_path = f"{target.name}_optimized.h5"
        model.save(optimized_model_path)

        return OptimizationResult(
            model_path=optimized_model_path,
            accuracy=0.92,  # 假设和原模型一致
            model_size_mb=150.0,  # 云端可以接受较大模型
            estimated_latency_ms=50.0,  # 多 GPU 加速
            memory_usage_mb=2000.0,
            optimization_strategy="Data parallelism + mixed precision"
        )


class EdgeOptimizer(ModelOptimizer):
    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        # 模拟：剪枝 + 量化 -> 模型更小，延迟降低，但精度略有下降
        optimized_model_path = f"{target.name}_optimized.tflite"

        # 转换为 TFLite 量化模型
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(optimized_model_path, "wb") as f:
            f.write(tflite_model)

        return OptimizationResult(
            model_path=optimized_model_path,
            accuracy=0.90,  # 精度略降
            model_size_mb=20.0,
            estimated_latency_ms=120.0,
            memory_usage_mb=256.0,
            optimization_strategy="Pruning + INT8 quantization"
        )


class TinyMLOptimizer(ModelOptimizer):
    def optimize(self, model: tf.keras.Model, target: DeploymentTarget) -> OptimizationResult:
        # 模拟：极限量化 + 操作符裁剪
        optimized_model_path = f"{target.name}_optimized.tflite"

        # 转换为 TFLite 且强制 INT8（此处简化为 DEFAULT 量化，便于无代表性数据也能跑通）
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(optimized_model_path, "wb") as f:
            f.write(tflite_model)

        return OptimizationResult(
            model_path=optimized_model_path,
            accuracy=0.85,  # 精度下降明显
            model_size_mb=0.5,
            estimated_latency_ms=800.0,
            memory_usage_mb=32.0,
            optimization_strategy="Full INT8 quantization + operator fusion"
        )


class MultiScaleDeploymentPipeline:
    """
    Automated pipeline for optimizing models across different deployment scales.
    """

    def __init__(self):
        self.optimizers = {
            'cloud': CloudOptimizer(),
            'edge': EdgeOptimizer(),
            'tiny': TinyMLOptimizer()
        }

        # Define deployment targets
        self.targets = {
            'cloud_server': DeploymentTarget(
                name='cloud_server',
                max_model_size_mb=1000.0,
                max_latency_ms=100.0,
                max_memory_mb=8000.0,
                power_budget_mw=50000.0,
                compute_capability='cloud'
            ),
            'edge_device': DeploymentTarget(
                name='edge_device',
                max_model_size_mb=50.0,
                max_latency_ms=200.0,
                max_memory_mb=512.0,
                power_budget_mw=2000.0,
                compute_capability='edge'
            ),
            'microcontroller': DeploymentTarget(
                name='microcontroller',
                max_model_size_mb=1.0,
                max_latency_ms=1000.0,
                max_memory_mb=64.0,
                power_budget_mw=10.0,
                compute_capability='tiny'
            )
        }

    def optimize_for_all_targets(self, baseline_model_path: str) -> Dict[str, OptimizationResult]:
        baseline_model = tf.keras.models.load_model(baseline_model_path)
        results = {}
        for target_name, target_config in self.targets.items():
            optimizer = self.optimizers[target_config.compute_capability]
            results[target_name] = optimizer.optimize(baseline_model, target_config)
        return results

    def analyze_scaling_trade_offs(self, results: Dict[str, OptimizationResult]) -> Dict[str, Any]:
        analysis = {}

        # 计算准确率 vs 延迟 Pareto 前沿（按精度降序、延迟升序）
        acc_latency = [(res.accuracy, res.estimated_latency_ms, name)
                       for name, res in results.items()]
        analysis["pareto_accuracy_latency"] = sorted(acc_latency, key=lambda x: (-x[0], x[1]))

        # 确定瓶颈
        bottlenecks = {}
        for name, res in results.items():
            tgt = self.targets[name]
            if res.model_size_mb > tgt.max_model_size_mb:
                bottlenecks[name] = "Model too large"
            elif res.memory_usage_mb > tgt.max_memory_mb:
                bottlenecks[name] = "Memory exceeded"
            elif res.estimated_latency_ms > tgt.max_latency_ms:
                bottlenecks[name] = "Latency too high"
            else:
                bottlenecks[name] = "OK"
        analysis["bottlenecks"] = bottlenecks

        # === 新增：资源占比（值/预算），用于可视化 ===
        ratios = {}
        for name, res in results.items():
            tgt = self.targets[name]
            ratios[name] = {
                "model_size_ratio": res.model_size_mb / max(tgt.max_model_size_mb, 1e-6),
                "latency_ratio": res.estimated_latency_ms / max(tgt.max_latency_ms, 1e-6),
                "memory_ratio": res.memory_usage_mb / max(tgt.max_memory_mb, 1e-6),
                "power_ratio": ( # 用功耗估计占比；这里以“更低更好”的思路，仅展示 res.power <= budget 的假设
                    (res.memory_usage_mb / max(tgt.max_memory_mb, 1e-6))  # 无真实功耗模型时，示例用内存占比近似
                )
            }
        analysis["budget_ratios"] = ratios

        return analysis

    def generate_deployment_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        recommendations = []

        # cloud_server 一般推荐数据中心部署
        if analysis["bottlenecks"].get("cloud_server") == "OK":
            recommendations.append("Deploy to cloud_server for max accuracy and throughput.")

        # edge_device 推荐低延迟应用
        if analysis["bottlenecks"].get("edge_device") == "OK":
            recommendations.append("Deploy to edge_device for real-time inference with moderate resources.")

        # microcontroller 推荐 TinyML 应用
        if analysis["bottlenecks"].get("microcontroller") == "OK":
            recommendations.append("Deploy to microcontroller for ultra-low power IoT applications.")

        return recommendations


# === 新增：可视化工具函数 ===
def _savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.close()

def plot_metric_bar(results: Dict[str, OptimizationResult], field: str, title: str, ylabel: str, outfile: str):
    names = list(results.keys())
    vals = [getattr(results[n], field) for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(6,4))
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    _savefig(outfile)

def plot_budget_ratios(analysis: Dict[str, Any], outfile: str = "ms_budget_ratios.png"):
    ratios = analysis.get("budget_ratios", {})
    if not ratios:
        return
    names = list(ratios.keys())
    keys = ["model_size_ratio", "latency_ratio", "memory_ratio", "power_ratio"]
    x = np.arange(len(names))
    width = 0.2

    plt.figure(figsize=(8,5))
    for i, k in enumerate(keys):
        vals = [ratios[n][k] for n in names]
        plt.bar(x + i*width, vals, width=width, label=k.replace("_ratio",""))

    plt.xticks(x + width*1.5, names, rotation=15)
    plt.ylabel("Ratio (value / budget)")
    plt.title("Resource Usage vs Budget (lower is better)")
    plt.legend()
    _savefig(outfile)

def plot_acc_latency_pareto(analysis: Dict[str, Any], outfile: str = "ms_acc_latency_pareto.png"):
    pts = analysis.get("pareto_accuracy_latency", [])
    if not pts:
        return
    # pts: [(acc, lat, name), ...]
    plt.figure(figsize=(6,4))
    for acc, lat, name in pts:
        plt.scatter([lat], [acc], s=60)
        plt.annotate(name, (lat, acc), textcoords="offset points", xytext=(6,6))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Latency (Pareto)")
    _savefig(outfile)

def plot_radar(results: Dict[str, OptimizationResult], outfile: str = "ms_radar.png"):
    # 对 accuracy ↑, model_size ↓, latency ↓, memory ↓, (power 用 memory 近似) ↓ 做归一化：值越大越好
    # 为演示，使用相对标尺：acc 直接用数值，其余使用 max/val 反向
    names = list(results.keys())
    if not names:
        return
    metrics = ["accuracy", "model_size_mb", "estimated_latency_ms", "memory_usage_mb"]
    # 计算反向指标的“越大越好”归一化
    accs = np.array([results[n].accuracy for n in names], dtype=float)
    sizes = np.array([results[n].model_size_mb for n in names], dtype=float)
    lats  = np.array([results[n].estimated_latency_ms for n in names], dtype=float)
    mems  = np.array([results[n].memory_usage_mb for n in names], dtype=float)

    def norm_up(x):  # 0..1
        a, b = float(np.min(x)), float(np.max(x))
        return (x - a) / max(b - a, 1e-9)

    def norm_down(x):  # 值越小越好 -> 1 - 归一
        xu = norm_up(x)
        return 1.0 - xu

    acc_n = norm_up(accs)
    size_n = norm_down(sizes)
    lat_n  = norm_down(lats)
    mem_n  = norm_down(mems)

    labels = ["Accuracy↑", "Model Size↓", "Latency↓", "Memory↓"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)

    for i, n in enumerate(names):
        vals = [acc_n[i], size_n[i], lat_n[i], mem_n[i]]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2)
        ax.fill(angles, vals, alpha=0.1, label=n)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Multi-Scale Radar (normalized, higher is better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    _savefig(outfile)


def run_multi_scale_optimization():
    pipeline = MultiScaleDeploymentPipeline()

    results = pipeline.optimize_for_all_targets('baseline_model.keras')
    analysis = pipeline.analyze_scaling_trade_offs(results)
    recommendations = pipeline.generate_deployment_recommendations(analysis)

    report = {
        'optimization_results': {k: vars(v) for k, v in results.items()},
        'scaling_analysis': analysis,
        'deployment_recommendations': recommendations
    }

    with open('multi_scale_optimization_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # === 新增：生成图表 ===
    plot_metric_bar(results, 'accuracy', 'Accuracy by Target', 'Accuracy', 'ms_acc_bar.png')
    plot_metric_bar(results, 'estimated_latency_ms', 'Latency by Target', 'Latency (ms)', 'ms_latency_bar.png')
    plot_metric_bar(results, 'model_size_mb', 'Model Size by Target', 'Model Size (MB)', 'ms_modelsize_bar.png')
    plot_metric_bar(results, 'memory_usage_mb', 'Memory Usage by Target', 'Memory (MB)', 'ms_memory_bar.png')
    # 用 memory 作为功耗近似值展示（没有真实能耗模型时的示例可视化）
    plot_metric_bar(results, 'memory_usage_mb', 'Power (approx by memory) by Target', 'Approx Power (arb.)', 'ms_power_bar.png')

    plot_budget_ratios(analysis, 'ms_budget_ratios.png')
    plot_acc_latency_pareto(analysis, 'ms_acc_latency_pareto.png')
    plot_radar(results, 'ms_radar.png')

    return report


if __name__ == "__main__":
    report = run_multi_scale_optimization()
    print("Multi-Scale Optimization Complete!")
    print("Report saved to: multi_scale_optimization_report.json")
