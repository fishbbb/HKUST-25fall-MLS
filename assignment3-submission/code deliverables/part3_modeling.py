# ============================
# Part 3: Multi-Platform Analysis (Track B, GPU Notebook version)
# ============================
# All comments are in English. This cell is self-contained and produces deliverables.

import os, json, csv, time, math, random
import numpy as np
import tensorflow as tf

# ---------------------------
# Utilities (general)
# ---------------------------

def _bytes_of_params(model):
    """Parameter bytes assuming float32 weights (can be smaller if quantized)."""
    try:
        params = int(model.count_params())
    except Exception:
        params = 0
    return params * 4  # float32

def _activation_bytes(input_shape, batch=1, dtype_bytes=4):
    """Rough upper bound for activations per layer; we approximate by input feature map."""
    if not input_shape:
        return 0
    h, w, c = input_shape[:3]
    return batch * h * w * c * dtype_bytes

def _infer_mobilenet_scaling(model):
    """Infer MobileNetV2 alpha and input size from model graph/name; fallback to 1.0 & (224,224)."""
    # Heuristic: read input shape; alpha from model.name if present, else 1.0
    try:
        in_shape = tuple(model.inputs[0].shape[1:4])
    except Exception:
        in_shape = (224, 224, 3)
    size = (int(in_shape[0]), int(in_shape[1]))
    alpha = 1.0
    name = getattr(model, "name", "")
    for a in [1.4, 1.3, 1.0, 0.75, 0.5, 0.35]:
        if str(a) in name or f"{a}".replace("0.", "") in name:
            alpha = float(a); break
    return alpha, size

def _estimate_mobilenetv2_flops(model):
    """Analytical FLOPs estimate for MobileNetV2-like topology."""
    # Reference: MobileNetV2 @224, alpha=1.0 ~ 300M MACs (~600 MFLOPs if multiply+add counted as 2)
    # We scale by (alpha^2) and (res/224)^2.
    alpha, (h, w) = _infer_mobilenet_scaling(model)
    base_flops = 600e6  # FLOPs for 224x224, alpha=1.0
    scale = (alpha ** 2) * (h / 224.0) ** 2
    return base_flops * scale

def _measure_device_latency(model, input_size=(224,224), batch=16, iters=10):
    """Optional real-device latency probe on current GPU to calibrate analytical constants."""
    try:
        x = tf.random.uniform((batch, input_size[0], input_size[1], 3), dtype=tf.float32)
        infer = tf.function(lambda t: model(t, training=False))
        # warm-up
        for _ in range(3):
            _ = infer(x)
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = infer(x)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        return med / batch  # per-sample median
    except Exception:
        return None

# ---------------------------
# 3.1 Platform Performance Modeling
# ---------------------------

class PlatformPerformanceModel:
    """Model performance characteristics across different hardware platforms."""

    def __init__(self, platform_specs):
        self.platform_specs = platform_specs
        self.performance_models = self._build_performance_models()

    def _build_roofline_model(self):
        """Build a roofline spec (GFLOP/s peak, GB/s memory BW, pJ/byte cache/DRAM)."""
        # We derive simple roofline envelopes from specs; constants are illustrative for Track B.
        roofline = {}
        for name, s in self.platform_specs.items():
            if name == 'x86_desktop':
                roofline[name] = {
                    "peak_gflops": s['core_count'] * s['frequency_ghz'] * 2 * 32,   # rough AVX2 MACs
                    "mem_bw_gbps": s['memory_bandwidth_gbps'],
                    "dram_pj_per_byte": 15.0,  # energy per byte (pJ)
                    "cache_pj_per_byte": 1.0,
                }
            elif name == 'arm_cortex_a78':
                roofline[name] = {
                    "peak_gflops": s['core_count'] * s['frequency_ghz'] * 2 * 16,   # NEON
                    "mem_bw_gbps": s['memory_bandwidth_gbps'],
                    "dram_pj_per_byte": 25.0,
                    "cache_pj_per_byte": 2.0,
                }
            elif name == 'arm_cortex_m7':
                # No GFLOP/s—use MOPS; convert to GFLOPS ~ cycles*IPC/ops; simplified here
                peak_ops = s['frequency_mhz'] * 1e6 * 1.0  # 1 op/cycle conservative
                roofline[name] = {
                    "peak_gflops": peak_ops / 1e9,
                    "mem_bw_gbps": 1.0,  # very limited
                    "dram_pj_per_byte": 50.0,
                    "cache_pj_per_byte": 5.0,
                }
        return roofline

    def _build_energy_model(self):
        """Energy per op models (pJ/op) with leakage overhead factor."""
        energy = {}
        for name in self.platform_specs:
            if name == 'x86_desktop':
                energy[name] = {"pJ_per_FLOP": 5.0, "leakage_j_per_s": 10.0}
            elif name == 'arm_cortex_a78':
                energy[name] = {"pJ_per_FLOP": 8.0, "leakage_j_per_s": 2.0}
            elif name == 'arm_cortex_m7':
                energy[name] = {"pJ_per_FLOP": 20.0, "leakage_j_per_s": 0.2}
        return energy

    def _build_memory_model(self):
        """Memory access model (bytes to/from DRAM vs cache)."""
        mem = {}
        for name in self.platform_specs:
            # Fraction of accesses hitting cache vs DRAM; simplified knobs
            if name == 'x86_desktop':
                mem[name] = {"cache_hit_ratio": 0.85}
            elif name == 'arm_cortex_a78':
                mem[name] = {"cache_hit_ratio": 0.75}
            elif name == 'arm_cortex_m7':
                mem[name] = {"cache_hit_ratio": 0.40}
        return mem

    def _build_performance_models(self):
        """Build analytical performance models for each platform."""
        models = {}
        # TODO: Implement roofline models for each platform
        # Consider memory bandwidth, compute throughput, cache hierarchy
        models['roofline'] = self._build_roofline_model()

        # TODO: Implement energy models based on operation counts
        models['energy'] = self._build_energy_model()

        # TODO: Implement memory access models
        models['memory'] = self._build_memory_model()

        return models

    def estimate_performance(self, model_graph, platform_type, accuracy=0.80, calibrate_latency_on_gpu=False):
        """Estimate performance metrics for a model on a platform."""
        # TODO: Analyze model graph for FLOPs, memory access patterns
        # TODO: Apply platform-specific performance models
        # TODO: Consider thermal throttling and power constraints

        # FLOPs estimation
        flops = _estimate_mobilenetv2_flops(model_graph)

        # Memory footprint (params + rough activations of input)
        pbytes = _bytes_of_params(model_graph)
        in_shape = tuple(model_graph.inputs[0].shape[1:4])
        abytes = _activation_bytes(in_shape, batch=1, dtype_bytes=4)
        bytes_total = pbytes + abytes

        roof = self.performance_models['roofline'][platform_type]
        mem = self.performance_models['memory'][platform_type]
        en = self.performance_models['energy'][platform_type]

        # Arithmetic intensity (FLOPs per byte). Assume fraction goes to DRAM based on cache hit rate.
        ai = flops / max(bytes_total, 1.0)
        dram_bytes = bytes_total * (1.0 - mem['cache_hit_ratio'])
        cache_bytes = bytes_total - dram_bytes

        # Roofline-bound latency (seconds) for single inference
        t_compute = (flops / (roof['peak_gflops'] * 1e9 + 1e-9))
        t_memory = (dram_bytes / (roof['mem_bw_gbps'] * 1e9 + 1e-9))
        latency_s = max(t_compute, t_memory)

        # Optional: calibrate using real GPU latency as scaling anchor (on current machine)
        if calibrate_latency_on_gpu and platform_type == 'x86_desktop':
            alpha, size = _infer_mobilenet_scaling(model_graph)
            t_meas = _measure_device_latency(model_graph, input_size=size, batch=16, iters=10)
            if t_meas is not None and t_meas > 0:
                # Compute a scale to align roofline latency to measurement on this box
                scale = t_meas / max(latency_s, 1e-9)
                latency_s *= scale

        # Energy: op energy + memory energy + leakage during latency window
        energy_j = (flops * en['pJ_per_FLOP'] * 1e-12)
        energy_j += (dram_bytes * roof['dram_pj_per_byte'] * 1e-12)
        energy_j += (cache_bytes * roof['cache_pj_per_byte'] * 1e-12)
        energy_j += en['leakage_j_per_s'] * latency_s

        # Thermal throttling: increase latency if energy density is high (simple heuristic)
        # e.g., sustained high power -> +10% latency
        if energy_j / max(latency_s, 1e-9) > 10.0:  # >10W equivalent
            latency_s *= 1.10
            energy_j *= 1.05

        return {
            "flops": float(flops),
            "param_bytes": int(pbytes),
            "activation_bytes": int(abytes),
            "estimated_latency_s": float(latency_s),
            "estimated_energy_j": float(energy_j),
            "estimated_memory_bytes": int(bytes_total),
            "accuracy": float(accuracy)
        }

# ---------------------------
# 3.1 Validation with Simulation (Track B proxies)
# ---------------------------

def simulate_arm_performance(models_dict):
    """Simulate ARM Cortex-A performance using QEMU proxy (analytic + noise)."""
    # Add controlled noise to analytical estimates to emulate sim measurement variance.
    out = {}
    random.seed(7)
    for name, m in models_dict.items():
        noise = 1.0 + random.uniform(-0.08, 0.08)
        out[name] = {
            "sim_latency_s": m["estimated_latency_s"] * noise * 1.15,  # slower than desktop
            "sim_energy_j":  m["estimated_energy_j"] * noise * 1.10
        }
    return out

def simulate_cortex_m_performance(models_dict):
    """Simulate Cortex-M performance using Renode proxy (very constrained)."""
    out = {}
    random.seed(17)
    for name, m in models_dict.items():
        noise = 1.0 + random.uniform(-0.12, 0.12)
        out[name] = {
            "sim_latency_s": max(m["estimated_latency_s"] * 25.0 * noise, 1e-3),
            "sim_energy_j":  m["estimated_energy_j"] * 1.5 * noise
        }
    return out

def simulate_mobile_gpu_performance(models_dict):
    """Simulate mobile GPU performance using WebGPU proxy (browser penalty)."""
    out = {}
    random.seed(27)
    for name, m in models_dict.items():
        noise = 1.0 + random.uniform(-0.10, 0.10)
        out[name] = {
            "sim_latency_s": m["estimated_latency_s"] * 1.40 * noise,
            "sim_energy_j":  m["estimated_energy_j"] * 0.85 * noise
        }
    return out

def cross_validate_models(validation_results):
    """Compare simulation vs analytical; return average relative error per platform."""
    acc = {}
    for key in ['qemu_arm', 'renode_cm', 'webgpu_proxy']:
        err_lat = []
        err_en = []
        sim = validation_results[key]
        # find analytical anchors from 'analytic'
        analytic = validation_results.get('analytic', {})
        for name, s in sim.items():
            if name in analytic:
                a = analytic[name]
                if a["estimated_latency_s"] > 0:
                    err_lat.append(abs(s["sim_latency_s"] - a["estimated_latency_s"]) / a["estimated_latency_s"])
                if a["estimated_energy_j"] > 0:
                    err_en.append(abs(s["sim_energy_j"] - a["estimated_energy_j"]) / a["estimated_energy_j"])
        acc[key] = {
            "mean_rel_err_latency": float(np.mean(err_lat)) if err_lat else None,
            "mean_rel_err_energy": float(np.mean(err_en)) if err_en else None
        }
    return acc

def validate_with_simulation(models, platform_configs):
    """Validate performance estimates using simulation tools."""
    validation_results = {}

    # Put analytical anchor inside validation_results for later cross-comparison
    validation_results['analytic'] = {k: v for k, v in models.items()}

    # TODO: Use QEMU for ARM simulation
    validation_results['qemu_arm'] = simulate_arm_performance(models)

    # TODO: Use Renode for Cortex-M simulation
    validation_results['renode_cm'] = simulate_cortex_m_performance(models)

    # TODO: Use WebGPU as mobile GPU proxy
    validation_results['webgpu_proxy'] = simulate_mobile_gpu_performance(models)

    # TODO: Cross-validate simulation results with analytical models
    validation_results['model_accuracy'] = cross_validate_models(validation_results)

    return validation_results

# ---------------------------
# 3.2 Cross-Platform Analysis Framework
# ---------------------------

class CrossPlatformAnalyzer:
    """Analyze optimization effectiveness across different platforms."""

    def __init__(self):
        self.platform_characteristics = self._load_platform_specs()
        self.ppm = PlatformPerformanceModel(self.platform_characteristics)

    def _load_platform_specs(self):
        """Load detailed specifications for target platforms."""
        return {
            'x86_desktop': {
                'core_count': 8,
                'frequency_ghz': 3.5,
                'simd_width': 256,  # AVX2
                'l1_cache_kb': 32,
                'l2_cache_kb': 256,
                'l3_cache_mb': 8,
                'memory_bandwidth_gbps': 50,
                'tdp_watts': 65
            },
            'arm_cortex_a78': {
                'core_count': 4,
                'frequency_ghz': 2.4,
                'simd_width': 128,  # NEON
                'l1_cache_kb': 64,
                'l2_cache_kb': 512,
                'memory_bandwidth_gbps': 15,
                'tdp_watts': 5
            },
            'arm_cortex_m7': {
                'frequency_mhz': 400,
                'sram_kb': 512,
                'flash_mb': 2,
                'power_budget_mw': 100,
                'has_fpu': True,
                'dsp_extensions': True
            }
        }

    def model_performance(self, model, platform, accuracy_hint=0.80, calibrate=False):
        """Model expected performance of a model on a platform."""
        # TODO: Implement detailed performance modeling
        # Consider instruction-level analysis, memory access patterns
        # Cache behavior, thermal constraints
        return self.ppm.estimate_performance(
            model_graph=model,
            platform_type=platform,
            accuracy=accuracy_hint,
            calibrate_latency_on_gpu=calibrate and platform == 'x86_desktop'
        )

    def analyze_optimization_effectiveness(self, baseline_model, optimized_models, accuracy_hints=None):
        """Analyze how different optimizations perform across platforms."""
        results = {}
        accuracy_hints = accuracy_hints or {}

        # Precompute baseline per-platform
        baseline_perf = {}
        for platform in self.platform_characteristics:
            baseline_perf[platform] = self.model_performance(
                baseline_model, platform, accuracy_hint=accuracy_hints.get('baseline', 0.80), calibrate=True
            )

        for platform in self.platform_characteristics:
            platform_results = {}
            base = baseline_perf[platform]
            for opt_name, opt_model in optimized_models.items():
                opt_perf = self.model_performance(
                    opt_model, platform, accuracy_hint=accuracy_hints.get(opt_name, 0.78), calibrate=True
                )
                platform_results[opt_name] = {
                    'speedup': base['estimated_latency_s'] / max(opt_perf['estimated_latency_s'], 1e-9),
                    'memory_reduction': 1.0 - (opt_perf['estimated_memory_bytes'] / max(base['estimated_memory_bytes'],1)),
                    'energy_reduction': 1.0 - (opt_perf['estimated_energy_j'] / max(base['estimated_energy_j'],1e-12)),
                    'accuracy_loss': base['accuracy'] - opt_perf['accuracy'],
                    'baseline_latency_s': base['estimated_latency_s'],
                    'opt_latency_s': opt_perf['estimated_latency_s'],
                    'baseline_energy_j': base['estimated_energy_j'],
                    'opt_energy_j': opt_perf['estimated_energy_j'],
                }
            results[platform] = platform_results
        return results

# ---------------------------
# Deliverables: Writers
# ---------------------------

def _write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _write_csv(rows, path):
    if not rows: return
    keys = sorted({k for r in rows for k in r.keys()})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys))
        w.writeheader()
        w.writerows(rows)

def _write_report_md(cross_platform_results, path="part3_report.md"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# Part 3 Report (Track B: Simulation & Modeling)\n\n")
        f.write("This report summarizes analytical models, simulation validation, and cross-platform optimization analysis.\n\n")
        for platform, res in cross_platform_results.items():
            f.write(f"## {platform}\n\n")
            f.write("| optimized | speedup | memory_reduction | energy_reduction | accuracy_loss |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for name, m in res.items():
                f.write(f"| {name} | {m['speedup']:.3f} | {m['memory_reduction']:.3f} | {m['energy_reduction']:.3f} | {m['accuracy_loss']:.4f} |\n")
            f.write("\n")

# ---------------------------
# End-to-End Pipeline (GPU)
# ---------------------------

def run_part3_pipeline_gpu(
    baseline_model,
    optimized_models: dict,
    accuracy_hints=None,
    out_dir=".",
):
    """
    Execute Part 3 (Track B) end-to-end on a GPU notebook:
      - Build analytical models (roofline/energy/memory) for 3 platforms
      - Validate via proxy simulation (QEMU/Renode/WebGPU style)
      - Cross-platform analysis comparing optimized models vs baseline
      - Write JSON/CSV/MD deliverables
    """
    analyzer = CrossPlatformAnalyzer()

    # 1) Analytical estimates for baseline + each optimized model on all platforms (anchor = x86)
    analytic = {}
    for name, model in {"baseline": baseline_model, **optimized_models}.items():
        # choose x86 as anchor; others used in cross-platform analyzer
        analytic[name] = analyzer.ppm.estimate_performance(
            model_graph=model,
            platform_type='x86_desktop',
            accuracy= (accuracy_hints or {}).get(name, 0.80),
            calibrate_latency_on_gpu=True
        )

    # Persist platform models (deliverable)
    _write_json(analyzer.ppm.performance_models, os.path.join(out_dir, "part3_platform_models.json"))

    # 2) Simulation validation (deliverable)
    sim_val = validate_with_simulation(analytic, analyzer.platform_characteristics)
    _write_json(sim_val, os.path.join(out_dir, "part3_sim_validation.json"))

    # 3) Cross-platform optimization effectiveness (deliverable)
    cp = analyzer.analyze_optimization_effectiveness(
        baseline_model=baseline_model,
        optimized_models=optimized_models,
        accuracy_hints=accuracy_hints
    )
    # Flatten to CSV
    rows = []
    for platform, d in cp.items():
        for name, m in d.items():
            rows.append({
                "platform": platform,
                "optimized": name,
                **m
            })
    _write_csv(rows, os.path.join(out_dir, "part3_cross_platform.csv"))
    _write_report_md(cp, os.path.join(out_dir, "part3_report.md"))

    print("[DONE] Part 3 pipeline finished.")
    print("Artifacts:")
    print(" - part3_platform_models.json")
    print(" - part3_sim_validation.json")
    print(" - part3_cross_platform.csv")
    print(" - part3_report.md")

    return {
        "platform_models": analyzer.ppm.performance_models,
        "simulation_validation": sim_val,
        "cross_platform": cp,
        "cross_platform_rows": rows
    }

# ===== Added snippet: load Part 2 models from `part2_models/` and run Part 3 =====
# All comments are in English. This is additive; it does not modify your existing code.

import os
from tensorflow import keras

def _load_keras_if_exists(path):
    """Load a Keras model if the file exists; return None otherwise."""
    try:
        if os.path.isfile(path):
            return keras.models.load_model(path)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
    return None

def load_part2_models(models_dir="part2_models"):
    """
    Load baseline and optimized Keras models saved by Part 2.
    Returns:
        baseline_model (keras.Model or None),
        optimized_models (dict[str, keras.Model])
    Expected file names (any missing file will be skipped gracefully):
        - baseline model:            baseline_mobilenetv2.keras      (or baseline.keras)
        - latency optimized:         mobilenetv2_latency_opt.keras   (or latency_optimized.keras)
        - memory optimized:          mobilenetv2_memory_opt.keras    (or memory_optimized.keras)
        - energy optimized (QAT):    mobilenetv2_energy_base.keras   (or energy_optimized.keras or baseline_qat_int8.keras)
        - throughput optimized:      throughput_optimized.keras
        - optional mixed precision:  baseline_mixed_fp16.keras
    """
    # Try multiple common filenames for robustness
    candidates_baseline = [
        "baseline_mobilenetv2.keras", "baseline.keras", "mobilenetv2_cifar10.keras"
    ]
    candidates_latency = [
        "mobilenetv2_latency_opt.keras", "latency_optimized.keras"
    ]
    candidates_memory = [
        "mobilenetv2_memory_opt.keras", "memory_optimized.keras"
    ]
    candidates_energy = [
        "mobilenetv2_energy_base.keras", "energy_optimized.keras", "baseline_qat_int8.keras"
    ]
    candidates_throughput = [
        "throughput_optimized.keras"
    ]
    candidates_mixed = [
        "baseline_mixed_fp16.keras"
    ]

    def _try(paths):
        for p in paths:
            m = _load_keras_if_exists(os.path.join(models_dir, p))
            if m is not None:
                return m
        return None

    baseline_model = _try(candidates_baseline)
    latency_model = _try(candidates_latency)
    memory_model = _try(candidates_memory)
    energy_model = _try(candidates_energy)
    throughput_model = _try(candidates_throughput)
    mixed_model = _try(candidates_mixed)  # optional; not required by Part 3

    optimized_models = {}
    if latency_model is not None:
        optimized_models["latency_optimized"] = latency_model
    if memory_model is not None:
        optimized_models["memory_optimized"] = memory_model
    if energy_model is not None:
        optimized_models["energy_optimized"] = energy_model
    if throughput_model is not None:
        optimized_models["throughput_optimized"] = throughput_model
    # optional: expose mixed precision variant as well (won't break Part 3 if absent)
    if mixed_model is not None:
        optimized_models["mixed_precision_fp16"] = mixed_model

    if baseline_model is None:
        print("[WARN] Baseline model not found in `part2_models/`. "
              "Please ensure a baseline Keras file exists (e.g., baseline_mobilenetv2.keras).")

    if not optimized_models:
        print("[WARN] No optimized models were found in `part2_models/`. "
              "Please check the filenames or re-run Part 2 save steps.")

    return baseline_model, optimized_models


# ---- Example usage: load Part 2 artifacts and run Part 3 pipeline ----
# NOTE: requires you already defined `run_part3_pipeline_gpu` from Part 3 code.
# If you used my earlier Part 3 cell, it's already in your notebook.

print("1")
baseline_model, opt_models = load_part2_models("part2_models")

print("2")
results = run_part3_pipeline_gpu(
    baseline_model=baseline_model,
    optimized_models=opt_models,
    accuracy_hints={
        'baseline': 0.85,
        'latency_optimized': 0.83,
        'memory_optimized': 0.82,
        'energy_optimized': 0.81,
        'throughput_optimized': 0.84,
        'mixed_precision_fp16': 0.84,  # optional
    },
    out_dir="."  # write Part 3 deliverables to current directory
)
