# part2_edge_optimization (with charts)
# 如果环境允许，可提前：!pip install tensorflow_model_optimization

import os
import time
import json
import psutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import mixed_precision
import tensorflow_model_optimization as tfmot

# ---------------------------
# 可视化工具
# ---------------------------
def _savefig_safe(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.close()

def plot_accuracy_bar(results, path="cloud_opt_accuracy.png"):
    """对比：mixed_precision.val_acc / model_parallelism.val_acc / KD(student,teacher)"""
    labels, values = [], []
    # mixed precision
    if "mixed_precision" in results and "final_accuracy" in results["mixed_precision"]:
        labels.append("Mixed Precision (val)")
        values.append(float(results["mixed_precision"]["final_accuracy"]))
    # distributed
    if "model_parallelism" in results and "final_accuracy" in results["model_parallelism"]:
        labels.append("Distributed (val)")
        values.append(float(results["model_parallelism"]["final_accuracy"]))
    # KD
    if "knowledge_distillation" in results:
        kd = results["knowledge_distillation"]
        if "student_acc" in kd:
            labels.append("KD Student (val)")
            values.append(float(kd["student_acc"]))
        if "teacher_acc" in kd:
            labels.append("KD Teacher (val)")
            values.append(float(kd["teacher_acc"]))

    if not labels:
        print("[Skip] accuracy bar: no accuracy fields found.")
        return

    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Cloud Optimizations: Validation Accuracy")
    _savefig_safe(path)

def plot_train_time_bar(results, path="cloud_opt_train_time.png"):
    """对比训练时长：mixed_precision / distributed / KD"""
    labels, values = [], []
    for k in ["mixed_precision", "model_parallelism", "knowledge_distillation"]:
        if k in results and "train_time_s" in results[k]:
            labels.append(k.replace("_", " ").title())
            values.append(float(results[k]["train_time_s"]))

    if not labels:
        print("[Skip] train time bar: no train_time_s fields found.")
        return

    plt.figure(figsize=(8, 5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Train Time (s)")
    plt.title("Cloud Optimizations: Training Time")
    _savefig_safe(path)

def plot_memory_bar(results, path="cloud_opt_memory.png"):
    """展示混合精度记录到的内存占用（单项），图中备注其它项缺失"""
    vals = []
    labels = []
    note = ""
    if "mixed_precision" in results and "memory_MB" in results["mixed_precision"]:
        labels.append("Mixed Precision (RSS)")
        vals.append(float(results["mixed_precision"]["memory_MB"]))
        note = "Only mixed precision reports memory.\nOthers N/A in this benchmark."
    else:
        print("[Skip] memory bar: no memory_MB field found.")
        return

    plt.figure(figsize=(6, 5))
    x = np.arange(len(labels))
    plt.bar(x, vals)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("Memory (MB)")
    plt.title("Process Memory Usage (RSS)")
    if note:
        plt.gcf().text(0.5, -0.05, note, ha="center", va="top", fontsize=9)
    _savefig_safe(path)

def plot_throughput_bar(results, path="cloud_opt_throughput.png"):
    """批处理优化吞吐率（样本/秒），单项柱状图"""
    if "batch_processing" not in results or "throughput_samples_per_s" not in results["batch_processing"]:
        print("[Skip] throughput bar: no throughput_samples_per_s field found.")
        return
    val = float(results["batch_processing"]["throughput_samples_per_s"])
    plt.figure(figsize=(6, 5))
    plt.bar([0], [val])
    plt.xticks([0], ["Batch Processing@config"], rotation=0)
    plt.ylabel("Throughput (samples/s)")
    plt.title("Batch Processing Throughput")
    _savefig_safe(path)

def plot_tradeoff_scatter(results, path="cloud_opt_tradeoff.png"):
    """
    “精度 vs 时间”权衡图：x=训练时长(s), y=验证准确率
    只绘制有这两个字段的点。
    """
    pts = []
    # Mixed precision
    if "mixed_precision" in results:
        r = results["mixed_precision"]
        if "train_time_s" in r and "final_accuracy" in r:
            pts.append(("Mixed Precision", float(r["train_time_s"]), float(r["final_accuracy"])))
    # Distributed
    if "model_parallelism" in results:
        r = results["model_parallelism"]
        if "train_time_s" in r and "final_accuracy" in r:
            pts.append(("Distributed", float(r["train_time_s"]), float(r["final_accuracy"])))
    # KD (用 student_acc 当作精度)
    if "knowledge_distillation" in results:
        r = results["knowledge_distillation"]
        if "train_time_s" in r and "student_acc" in r:
            pts.append(("KD Student", float(r["train_time_s"]), float(r["student_acc"])))

    if not pts:
        print("[Skip] tradeoff scatter: insufficient (time, accuracy) pairs.")
        return

    plt.figure(figsize=(7, 5))
    for name, t, acc in pts:
        plt.scatter([t], [acc], s=60)
        plt.annotate(name, (t, acc), textcoords="offset points", xytext=(6, 6))
    plt.xlabel("Train Time (s)")
    plt.ylabel("Validation Accuracy")
    plt.title("Accuracy vs. Train Time (Trade-off)")
    _savefig_safe(path)

# ---------------------------
# 优化与基准逻辑（你的原始代码）
# ---------------------------
class CloudOptimizer:
    def __init__(self, baseline_model_path):
        self.baseline_model = tf.keras.models.load_model(baseline_model_path)

    def implement_mixed_precision(self):
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        # 保证层的 dtype 兼容；这里仅保护性检查
        for layer in self.baseline_model.layers:
            if hasattr(layer, "dtype") and layer.dtype == "float16":
                layer.dtype = "float32"

        optimizer = tf.keras.optimizers.Adam()
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        self.baseline_model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        return self.baseline_model

    def implement_model_parallelism(self, strategy='mirrored'):
        if strategy == 'mirrored':
            dist_strategy = tf.distribute.MirroredStrategy()
        elif strategy == 'multi_worker_mirrored':
            dist_strategy = tf.distribute.MultiWorkerMirroredStrategy()
        elif strategy == 'parameter_server':
            cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
            dist_strategy = tf.distribute.ParameterServerStrategy(cluster_resolver)
        else:
            raise ValueError("Unsupported strategy type.")

        with dist_strategy.scope():
            distributed_model = tf.keras.models.clone_model(self.baseline_model)
            distributed_model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        return distributed_model, dist_strategy

    def optimize_batch_processing(self, target_batch_size=256):
        config = {
            "batch_size": target_batch_size,
            "gradient_accumulation_steps": max(1, target_batch_size // 64),
            "prefetch": tf.data.AUTOTUNE,
            "num_parallel_calls": tf.data.AUTOTUNE
        }
        def optimize_dataset(dataset):
            dataset = dataset.batch(config["batch_size"])
            dataset = dataset.prefetch(config["prefetch"])
            dataset = dataset.map(lambda x, y: (x, y),
                                  num_parallel_calls=config["num_parallel_calls"])
            return dataset
        return {"config": config, "optimize_dataset_fn": optimize_dataset}

    def implement_knowledge_distillation(self):
        teacher = tf.keras.models.clone_model(self.baseline_model)
        for layer in teacher.layers:
            if hasattr(layer, "units"):
                layer.units *= 2
        teacher.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        student = self.baseline_model

        def distillation_loss(y_true, y_pred, teacher_pred, temperature=5.0, alpha=0.5):
            soft_targets = tf.nn.softmax(teacher_pred / temperature)
            student_soft = tf.nn.log_softmax(y_pred / temperature)
            kl_div = tf.reduce_mean(tf.keras.losses.KLDivergence()(soft_targets, student_soft))
            hard_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
            return alpha * hard_loss + (1 - alpha) * kl_div

        def distillation_train_step(x, y, optimizer, temperature=5.0, alpha=0.5):
            with tf.GradientTape() as tape:
                teacher_pred = teacher(x, training=False)
                student_pred = student(x, training=True)
                loss = distillation_loss(y, student_pred, teacher_pred, temperature, alpha)
            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            return loss

        return teacher, student, distillation_train_step


def benchmark_cloud_optimizations():
    optimizer = CloudOptimizer('baseline_model.keras')
    results = {}

    # ========= 数据准备 =========
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train[:5000], y_train[:5000]))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test[:1000], y_test[:1000]))

    # ========= Mixed Precision =========
    start_time = time.time()
    mp_model = optimizer.implement_mixed_precision()
    history = mp_model.fit(train_ds.batch(64, drop_remainder=True),
                           epochs=1,
                           validation_data=test_ds.batch(64, drop_remainder=True),
                           verbose=0)
    elapsed = time.time() - start_time
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    results["mixed_precision"] = {
        "train_time_s": elapsed,
        "final_accuracy": float(history.history["val_accuracy"][-1]),
        "memory_MB": float(mem)
    }

    # ========= Model Parallelism =========
    start_time = time.time()
    dist_model, strategy = optimizer.implement_model_parallelism(strategy="mirrored")
    history = dist_model.fit(train_ds.batch(64, drop_remainder=True),
                             epochs=1,
                             validation_data=test_ds.batch(64, drop_remainder=True),
                             verbose=0)
    elapsed = time.time() - start_time
    results["model_parallelism"] = {
        "strategy": type(strategy).__name__,
        "train_time_s": float(elapsed),
        "final_accuracy": float(history.history["val_accuracy"][-1])
    }

    # ========= Batch Processing =========
    config = optimizer.optimize_batch_processing(target_batch_size=256)
    opt_train_ds = config["optimize_dataset_fn"](train_ds)

    start_time = time.time()
    steps = 0
    samples = 0
    for (_, __) in opt_train_ds.take(20):
        steps += 1
        samples += config["config"]["batch_size"]
    elapsed = time.time() - start_time
    throughput = samples / elapsed if elapsed > 0 else float('nan')
    results["batch_processing"] = {
        "batch_size": int(config["config"]["batch_size"]),
        "throughput_samples_per_s": float(throughput)
    }

    # ========= Knowledge Distillation =========
    teacher, student, distill_step = optimizer.implement_knowledge_distillation()
    optimizer_fn = tf.keras.optimizers.Adam()

    start_time = time.time()
    for (x_batch, y_batch) in train_ds.batch(64, drop_remainder=True).take(10):
        _ = distill_step(x_batch, y_batch, optimizer_fn)
    elapsed = time.time() - start_time

    teacher_acc = teacher.evaluate(test_ds.batch(64, drop_remainder=True), verbose=0)[1]
    student_acc = student.evaluate(test_ds.batch(64, drop_remainder=True), verbose=0)[1]
    results["knowledge_distillation"] = {
        "teacher_acc": float(teacher_acc),
        "student_acc": float(student_acc),
        "train_time_s": float(elapsed)
    }

    return results


# ---------------------------
# 主流程：跑实验 + 画图 + 导出 JSON
# ---------------------------
if __name__ == "__main__":
    results = benchmark_cloud_optimizations()

    print("Cloud Optimization Results:")
    for optimization, metrics in results.items():
        print(f"{optimization}: {metrics}")

    # 导出 JSON
    with open("multi_scale_optimization_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[Saved] multi_scale_optimization_report.json")

    # 生成图表
    plot_accuracy_bar(results, "cloud_opt_accuracy.png")
    plot_train_time_bar(results, "cloud_opt_train_time.png")
    plot_memory_bar(results, "cloud_opt_memory.png")
    plot_throughput_bar(results, "cloud_opt_throughput.png")
    plot_tradeoff_scatter(results, "cloud_opt_tradeoff.png")
