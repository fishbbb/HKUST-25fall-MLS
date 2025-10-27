# Part 4: Comprehensive Analysis & Design Recommendations
## 1) Performance Summary
- Models compared: baseline, energy_opt, latency_opt, memory_opt
- Platforms: x86_desktop
- Metrics: latency_ms, memory_mb, energy_mj, accuracy, throughput_fps
- See CSV: `performance_summary.csv`
- See radar charts: `radar_*.png`, Pareto: `pareto_acc_vs_energy.png`, Batch scaling: `batch_scaling.png`

## 2) Hardware Utilization Highlights (derived)
- See `utilization_analysis.json` for detailed per-platform utilization proxies.

## 3) Constraint Specification
```json
{
  "latency_ms_target": 35.0,
  "accuracy_min": 0.83,
  "memory_mb_limit": 12.0,
  "energy_mj_budget": 0.018,
  "platform": "arm_cortex_a78",
  "batch_size": 16,
  "throughput_fps_min": 28.571428571428573
}
```
## 4) Optimization Priority
- Proposed order (from highest priority to lowest):
  1. resolution_scaling
  2. quantization
  3. width_scaling
  4. operator_fusion
  5. bn_folding
  6. distillation
  7. qat

## 5) Top Design Space Candidates (analytic estimates)
| alpha | input | quant | latency_ms | memory_mb | energy_mj | accuracy | throughput_fps | penalty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.75 | 224x224 | fp32 | 28.12 | 10.5 | 0.0139 | 0.845 | 35.56 | 0.0 |
| 0.75 | 224x224 | fp16 | 28.12 | 10.5 | 0.0118 | 0.845 | 35.56 | 0.0 |
| 0.75 | 224x224 | int8_dr | 28.12 | 6.3 | 0.0083 | 0.835 | 35.56 | 0.0 |
| 0.75 | 224x224 | int8_ptq | 28.12 | 6.3 | 0.0083 | 0.835 | 35.56 | 0.0 |
| 0.5 | 224x224 | fp32 | 12.5 | 7.0 | 0.0095 | 0.83 | 80.0 | 0.0 |
| 0.5 | 224x224 | fp16 | 12.5 | 7.0 | 0.0081 | 0.83 | 80.0 | 0.0 |
| 0.75 | 192x192 | fp32 | 20.66 | 10.5 | 0.0118 | 0.829 | 48.4 | 0.005 |
| 0.75 | 192x192 | fp16 | 20.66 | 10.5 | 0.01 | 0.829 | 48.4 | 0.005 |
| 1.0 | 192x192 | int8_dr | 36.73 | 8.4 | 0.0098 | 0.834 | 27.22 | 0.0173 |
| 1.0 | 192x192 | int8_ptq | 36.73 | 8.4 | 0.0098 | 0.834 | 27.22 | 0.0173 |

## 6) Platform-Specific Recommendations
### x86_desktop
- Prefer operator fusion, XLA/JIT, and mixed precision (fp16 on recent GPUs) for throughput.
- Maintain input 224×224 unless latency target is tight; then 192×192 or 160×160 with distillation.
- If memory bound, apply pruning (30–50%) and INT8 dynamic range for deployment.

### arm_cortex_a78
- Reduce input to 160×160 and width(alpha≤0.5); apply INT8 PTQ with calibrated representative set.
- Fuse depthwise+pointwise blocks; avoid costly ops; prefer NHWC; keep batch small (≤16).
- Thermal guard: duty-cycle heavy inference (burst + cooldown) for sustained workloads.

### arm_cortex_m7
- Use TFLite Micro with full INT8 pipeline; model size ≤ SRAM/Flash limits.
- Consider early-exit head or class-pruning; use CMSIS-NN kernels; batch=1.
- Aggressively prune and distill; prefer 128×128 inputs.

### mobile_gpu (WebGPU proxy)
- Favor 160×160 inputs with alpha 0.5; enable FP16 compute; prioritize kernel fusion.
- Use asynchronous pre/post-processing pipeline to hide memory latency.

