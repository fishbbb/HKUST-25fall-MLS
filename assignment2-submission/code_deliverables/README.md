# Multi-Scale Optimization and Deployment Pipeline

This repository contains a set of Python modules and scripts for **deep learning model optimization** across different deployment targets (Cloud, Edge, and TinyML). It is based on CIFAR-10 experiments with a CNN baseline model, extended with pruning, quantization, mixed precision, architecture optimization, and deployment strategy analysis.

---

## 📂 Project Structure

### 1. `part1_baseline_model.py`
- Defines and trains a moderately complex **CNN baseline model** on CIFAR-10.
- Includes:
  - Data loading and preprocessing (with augmentation).
  - Training with callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).
  - Export to `.keras` and `.tflite` formats.
  - ✅ **Visualization**:
    - Training accuracy/loss curves.
    - Confusion matrix on test set.
    - Sample predictions with ground truth vs predicted labels.

### 2. `part2_edge_optimization.py`
- Focuses on **cloud-oriented optimizations**:
  - Mixed precision training.
  - Distributed model parallelism.
  - Batch processing optimization.
  - Knowledge distillation (teacher → student).
- Includes **benchmarking functions** to compare performance.
- ✅ **Visualization**:
  - Bar charts for training time, accuracy, and memory usage.
  - Throughput comparison plots.

### 3. `edge_optimizer_module.py`
- Provides **edge deployment optimization** methods:
  - Magnitude-based pruning (with fine-tuning).
  - Post-training quantization (dynamic, float16, int8).
  - Lightweight architecture optimization (Depthwise Separable Convs).
  - Simplified NAS (random search).
  - TFLite conversion with size & latency measurement.
- ✅ **Visualization**:
  - Accuracy/size trade-off across sparsity levels.
  - Comparison of quantization strategies.
  - Search results summary.

### 4. `multi_scale_pipeline.py`
- Implements a **multi-scale deployment pipeline**:
  - Cloud (max accuracy, throughput).
  - Edge device (balanced accuracy/latency).
  - Microcontroller / TinyML (ultra-low power).
- Exports:
  - `multi_scale_optimization_report.json`
  - **Charts**:
    - Accuracy, latency, size, memory, power (bar charts).
    - Resource usage vs budget (stacked ratios).
    - Accuracy–latency Pareto front.
    - Radar chart for multi-metric comparison.

---

## 🚀 Usage

### Train and export baseline model
```bash
python part1_baseline_model.py
```

### Run cloud optimization benchmarks
```bash
python part2_edge_optimization.py
```

### Run edge optimization benchmarks
```bash
python edge_optimizer_module.py
```

### Run full multi-scale deployment pipeline
```bash
python multi_scale_pipeline.py
```

---

## 📊 Outputs

- **Reports**:
  - `edge_optimization_report.json`
  - `multi_scale_optimization_report.json`

- **Visualizations**:
  - `training_history.png`
  - `confusion_matrix.png`
  - `sample_predictions.png`
  - `ms_acc_bar.png`, `ms_latency_bar.png`, `ms_modelsize_bar.png`, `ms_memory_bar.png`, `ms_power_bar.png`
  - `ms_budget_ratios.png`
  - `ms_acc_latency_pareto.png`
  - `ms_radar.png`

---

## 🛠 Dependencies

- TensorFlow 2.x
- TensorFlow Model Optimization Toolkit (`tensorflow_model_optimization`)
- NumPy
- Matplotlib
- scikit-learn

Install via:
```bash
pip install -r requirements.txt
```

---

## 📌 Notes

- Some metrics (power usage, memory usage on microcontrollers) are **approximated** for demonstration.
- Replace representative datasets in quantization with real calibration data for production.
- NAS implementation is simplified; integrate AutoML frameworks for advanced search.

---
