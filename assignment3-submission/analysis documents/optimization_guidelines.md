# Optimization Guidelines for Hardware-Aware ML (Track B)

This guide distills the methodology and best practices used in this project into a repeatable playbook. It is platform-aware, simulator-friendly, and grounded in our artifacts:

- `part4/performance_summary.csv` (x86 anchor)
- `part3/part3_cross_platform.csv` (cross-platform deltas via analytical/simulation proxies)
- `part3/part3_platform_models.json` (roofline/energy/memory envelopes)
- Figures in `part4/` (radar, Pareto, batch scaling)

Tags (execution context): **[native-x86]**, **[QEMU-ARM]**, **[Renode-CM]**, **[WebGPU-proxy]**.

---

## 1. Design Tenets

1. **Co-design hardware & software**: treat compute peak, memory bandwidth, and power as first-class constraints.
2. **Specify constraints early**: latency (ms), accuracy target, memory (MB), energy (MJ).
3. **Iterate with multi-objective trade-offs**: use Pareto views (accuracy vs energy vs latency).
4. **Prefer platform-specific paths**: align ops to ISA/SIMD widths, kernel availability, and accelerator support.
5. **Automate checks**: enforce constraints in CI to prevent regressions.

---

## 2. Optimization Toolkit

**Architectural**
- Width multiplier (**alpha**) and input resolution (e.g., 224→160) to reduce MACs and activation size.
- Depthwise separable convolutions (MobileNet family) by default.
- Early-exit heads for duty-cycled applications.

**Quantization & Pruning**
- INT8 PTQ/QAT for deployment; FP16 for GPU‐accelerated paths.
- Pruning (30–50%) to reduce parameter memory and cache pressure; fine-tune to retain accuracy.

**Compiler & Kernel**
- Operator fusion, BN folding, XLA/JIT on desktop; CMSIS-NN kernels on MCU.
- Data layout (NHWC) to match optimized kernels.

**I/O & Batching**
- Asynchronous pre/post pipelines; keep batches small on bandwidth-limited SoCs.

---

## 3. Platform-Specific Playbooks

### 3.1 x86 Desktop / dGPU (**[native-x86]**)
- **Primary knobs**: fusion/JIT, mixed precision (FP16 on GPU), maintain 224 unless latency target is tight.
- **When tight on latency**: 160×160 + alpha ≤ 0.5; consider distillation to recover accuracy.
- **Measurement tools**: TFLite `benchmark_model`, ORT PerfTest, TF Profiler.
- **From our results (x86 anchor)**:
  - Baseline: 50 ms, 0.85 acc, 0.02 MJ
  - Latency-opt: 30 ms (−40%), 0.83 acc
  - Energy-opt: 0.015 MJ (−25%) with 0.81 acc

### 3.2 ARM Cortex-A (**[QEMU-ARM]**)
- **Primary knobs**: reduce input (160), alpha ≤ 0.5, INT8 PTQ/QAT, small batches.
- **Observations**: latency-opt yielded ≈ **1.96×** speedup and ≈ **49%** energy reduction vs baseline.

### 3.3 ARM Cortex-M (**[Renode-CM]**)
- **Primary knobs**: end-to-end INT8 (TFLite Micro + CMSIS-NN), pruning, distillation, batch=1, ≤128×128.
- **Observations**: latency-opt ≈ **1.96×** speedup, ≈ **49%** energy reduction; memory-only changes have limited latency effect without INT8.

### 3.4 Mobile GPU (**[WebGPU-proxy]**)
- **Primary knobs**: FP16 compute or INT8, alpha≈0.5, 160×160 input, kernel fusion, async data pipeline.
- **Notes**: profile in browser with ORT-Web (WebGPU); treat as relative proxy to phone GPUs.

---

## 4. Decision Framework

### 4.1 Constraint Specification (fill these before modeling)
- Latency target (ms): `L*`
- Accuracy minimum: `A*`
- Memory limit (MB): `M*`
- Energy budget (MJ): `E*`
- Platform: `x86_desktop` | `arm_cortex_a78` | `arm_cortex_m7` | `mobile_gpu`
- Batch size: `B*`

### 4.2 Strategy Selection (priority order)
- **If memory-tight (M* < 8MB)**: pruning → INT8 → width scaling → (then fusion/JIT).
- **If energy-tight (E* small)**: INT8 → resolution scaling → width scaling → pruning → distillation/QAT.
- **If latency-tight (L* < 30ms)**: resolution scaling → width scaling → fusion/JIT → mixed precision (GPU).
- **Accuracy guardrails**: distillation and/or brief QAT fine-tunes to recover accuracy.

---

## 5. Measurement & Reproducibility

- Always tag runs: [native-x86], [QEMU-ARM], [Renode-CM], [WebGPU-proxy].
- Pin tool versions; record CLI:
  - **TFLite**: `benchmark_model --graph=model.tflite --num_threads=1 --num_runs=100 …`
  - **ONNX Runtime**: `onnxruntime_perf_test -e cpu|dml|cuda model.onnx …`
- Export CSV + JSON logs and include in repo (`performance_benchmarks.xlsx` consolidates them).

---

## 6. Maintenance & Rollout

- Keep a **capability matrix** per platform (INT8? FP16? kernel coverage?).
- CI checks: fail builds if latency/memory/energy budgets are exceeded.
- For updates on constrained devices: ship **INT8 weights** + **distilled heads** with A/B guardrails.

---

## 7. Appendix: Interpreting Our Artifacts

- **Pareto frontier** (`part4/pareto_acc_vs_energy.png`): all four x86 anchor models are Pareto-optimal, offering distinct accuracy-energy trade-offs.
- **Radar charts** (`part4/radar_x86_desktop.png`): visualize multi-metric trade-offs.
- **Cross-platform deltas** (`part3/part3_cross_platform.csv`): A78/M7 show strong gains for latency-optimized; memory-only brings limited latency change without quantization.
- **Roofline envelopes** (`part3_platform_models.json`): x86 has highest peak GFLOPs and BW; M7 is bandwidth/compute constrained → prefer INT8 & smaller inputs.
