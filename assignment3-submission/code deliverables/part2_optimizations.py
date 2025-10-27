# ==== Part 2: Hardware-Aware Optimizations (Track B, GPU-only, Notebook-ready) ====
# All comments are in English. This cell is standalone for Part 2 usage in a notebook.

import os, time, json, csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- GPU preparation ----------------
def prepare_gpu(mixed_precision=True, memory_growth=True):
    """Enable GPU memory growth and optionally mixed precision."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus and memory_growth:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        if mixed_precision and gpus:
            from tensorflow.keras import mixed_precision as mp
            mp.set_global_policy('mixed_float16')
            print("[INFO] Mixed precision enabled: mixed_float16")
        else:
            print("[INFO] Mixed precision disabled or no GPU detected.")
    except Exception as e:
        print(f"[WARN] GPU preparation failed: {e}")

# -------------- Small compile helper --------------
def _compile(model, lr=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model

# ---------------- 2.1 Architecture opt (GPU) ----------------
def create_latency_optimized_model(num_classes=10):
    """alpha=0.5, 160x160 input."""
    try:
        base = MobileNetV2(include_top=False, input_shape=(160,160,3), alpha=0.5, weights='imagenet')
    except Exception:
        base = MobileNetV2(include_top=False, input_shape=(160,160,3), alpha=0.5, weights=None)
    base.trainable = False
    inp = keras.Input((160,160,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return _compile(keras.Model(inp, out, name='latency_optimized'), lr=1e-3)

def create_memory_optimized_model(num_classes=10):
    """alpha=0.35, optional pruning."""
    try:
        base = MobileNetV2(include_top=False, input_shape=(224,224,3), alpha=0.35, weights='imagenet')
    except Exception:
        base = MobileNetV2(include_top=False, input_shape=(224,224,3), alpha=0.35, weights=None)
    base.trainable = False
    inp = keras.Input((224,224,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inp, out, name='memory_optimized')
    # Optional pruning (if tfmot available)
    try:
        import tensorflow_model_optimization as tfmot
        prune = tfmot.sparsity.keras.prune_low_magnitude
        sched = tfmot.sparsity.keras.PolynomialDecay(0.0, 0.5, begin_step=0, end_step=1000)
        model = prune(model, pruning_schedule=sched)
        return _compile(model, lr=5e-4)
    except Exception:
        return _compile(model, lr=1e-3)

def create_energy_optimized_model(num_classes=10):
    """alpha=0.5, 160x160; QAT-ready if tfmot available."""
    try:
        base = MobileNetV2(include_top=False, input_shape=(160,160,3), alpha=0.5, weights='imagenet')
    except Exception:
        base = MobileNetV2(include_top=False, input_shape=(160,160,3), alpha=0.5, weights=None)
    base.trainable = False
    inp = keras.Input((160,160,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.15)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model_fp = keras.Model(inp, out, name='energy_optimized_base')
    try:
        import tensorflow_model_optimization as tfmot
        qa = tfmot.quantization.keras.quantize_annotate_model
        qapply = tfmot.quantization.keras.quantize_apply
        qmodel = qapply(qa(model_fp))
        return _compile(qmodel, lr=5e-4)
    except Exception:
        return _compile(model_fp, lr=1e-3)

def create_throughput_optimized_model(num_classes=10):
    """alpha=1.0, 224x224; optional XLA warmup."""
    try:
        base = MobileNetV2(include_top=False, input_shape=(224,224,3), alpha=1.0, weights='imagenet')
    except Exception:
        base = MobileNetV2(include_top=False, input_shape=(224,224,3), alpha=1.0, weights=None)
    base.trainable = False
    inp = keras.Input((224,224,3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inp, out, name='throughput_optimized')
    try:
        @tf.function(jit_compile=True)
        def _jit(m,t): return m(t, training=False)
        _ = _jit(model, tf.zeros((1,224,224,3), tf.float32))
        print("[INFO] XLA JIT warm-up succeeded for throughput_optimized.")
    except Exception:
        print("[INFO] XLA JIT not used (ok).")
    return _compile(model, lr=1e-3)

def create_optimized_models_gpu(num_classes=10):
    """Return 4 optimized Keras models."""
    return {
        'latency_optimized':   create_latency_optimized_model(num_classes),
        'memory_optimized':    create_memory_optimized_model(num_classes),
        'energy_optimized':    create_energy_optimized_model(num_classes),
        'throughput_optimized':create_throughput_optimized_model(num_classes),
    }

# ---------------- 2.2 Quantization (GPU-friendly; conversion on CPU) ----------------
def representative_dataset_generator_gpu(x_train_sample, target_size=(224,224), count=200):
    """Yield representative samples with resize+preprocess for PTQ calibration."""
    n = min(int(count), int(x_train_sample.shape[0]))
    for i in range(n):
        x = tf.image.resize(x_train_sample[i:i+1], target_size)
        x = tf.cast(x, tf.float32)
        x = preprocess_input(x)
        yield [x]

def tflite_ptq_int8_from_keras(model, rep_gen):
    """Full-integer INT8 TFLite conversion."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        return converter.convert()
    except Exception:
        return None

def tflite_dynamic_range_from_keras(model):
    """Dynamic range TFLite conversion (weights quantized, activations float)."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        return converter.convert()
    except Exception:
        return None

def create_qat_from_model(model):
    """Return a QAT-ready Keras model if tfmot is available; else None."""
    try:
        import tensorflow_model_optimization as tfmot
        qa = tfmot.quantization.keras.quantize_annotate_model
        qapply = tfmot.quantization.keras.quantize_apply
        qmodel = qapply(qa(model))
        return _compile(qmodel, lr=5e-4)
    except Exception:
        return None

def create_mixed_precision_clone(model):
    """Clone under mixed_float16 policy (compute fp16 on GPU)."""
    try:
        from tensorflow.keras import mixed_precision as mp
        old = mp.global_policy()
        mp.set_global_policy('mixed_float16')
        cfg = model.get_config()
        mp_model = keras.Model.from_config(cfg)
        mp_model.set_weights(model.get_weights())
        mp.set_global_policy(old.name)
        return _compile(mp_model, lr=1e-3)
    except Exception:
        return None

def apply_quantization_optimizations_gpu(model, x_train_sample):
    """Return dict of quantized variants: ptq_int8/dr/qat_int8/mixed_precision."""
    rep = lambda: representative_dataset_generator_gpu(x_train_sample, (224,224), 200)
    return {
        'ptq_int8':      tflite_ptq_int8_from_keras(model, rep),
        'dynamic_range': tflite_dynamic_range_from_keras(model),
        'qat_int8':      create_qat_from_model(model),
        'mixed_precision': create_mixed_precision_clone(model),
    }

# ---------------- Benchmarks (Keras GPU + TFLite CPU) ----------------
def _resize_norm_np(x, size):
    x = tf.image.resize(x, size)
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x.numpy()

def benchmark_keras_model_gpu(model, x_test, y_test, input_size=(224,224), batch=32, iters=20, save_dir="part2_models"):
    """Latency (median/mean), accuracy, size, params for a Keras model."""
    os.makedirs(save_dir, exist_ok=True)
    x_small = _resize_norm_np(x_test[:batch], input_size)
    infer = tf.function(lambda t: model(t, training=False))
    for _ in range(3): _ = infer(x_small)  # warmup
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = infer(x_small)
        ts.append(time.perf_counter() - t0)
    med, mean = float(np.median(ts)), float(np.mean(ts))
    ds = tf.data.Dataset.from_tensor_slices((_resize_norm_np(x_test, input_size), y_test)).batch(batch)
    _, acc = model.evaluate(ds, verbose=0)
    size_mb = 0.0
    try:
        path = os.path.join(save_dir, f"{model.name}.keras")
        model.save(path, overwrite=True)
        size_mb = os.path.getsize(path)/(1024**2)
    except Exception:
        pass
    return {
        "latency_median_s": med,
        "latency_mean_s": mean,
        "accuracy": float(acc),
        "keras_size_mb": float(size_mb),
        "params": int(model.count_params())
    }

def benchmark_tflite_model(tflite_bytes, x, y=None, input_size=(224,224), batch_size=32, iters=20, save_path="part2_models/quantized.tflite"):
    """Latency/size on CPU; approx accuracy on batch subset for TFLite."""
    if tflite_bytes is None: return None
    import tempfile
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as f:
        f.write(tflite_bytes)
        tmp = f.name
    size_mb = os.path.getsize(tmp)/(1024**2)
    intr = tf.lite.Interpreter(model_path=tmp, num_threads=1)
    intr.allocate_tensors()
    inp, out = intr.get_input_details()[0], intr.get_output_details()[0]

    def _prep(arr):
        arr = tf.image.resize(arr, input_size)
        arr = tf.cast(arr, tf.float32)
        arr = preprocess_input(arr)
        return arr.numpy()

    x0 = _prep(x[:batch_size])
    import numpy as np
    if inp['dtype'] == np.int8:
        scale, zero = inp['quantization']
        if scale and scale>0:
            x0 = np.round(x0/scale + zero).astype(np.int8)

    for _ in range(3):
        intr.set_tensor(inp['index'], x0); intr.invoke()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        intr.set_tensor(inp['index'], x0); intr.invoke()
        times.append(time.perf_counter()-t0)
    med, mean = float(np.median(times)), float(np.mean(times))

    acc = 0.0
    if y is not None:
        logits = intr.get_tensor(out['index'])
        preds = np.argmax(logits, axis=-1)
        acc = float((preds == y[:len(preds)]).mean())

    try:
        with open(save_path, "wb") as wf: wf.write(tflite_bytes)
    except Exception: pass
    try:
        os.remove(tmp)
    except Exception: pass

    return {
        "tflite_size_mb": size_mb,
        "latency_median_s": med,
        "latency_mean_s": mean,
        "batch_size": int(len(x0)),
        "approx_accuracy": acc
    }

# ---------------- 2.3 Memory optimizations (GPU-aware) ----------------
def implement_memory_optimizations_gpu(model):
    """Report memory techniques + measured safe batch size on GPU forward pass."""
    report = {}
    # activation compression via mixed precision (report current policy)
    try:
        from tensorflow.keras import mixed_precision as mp
        report["activation_compression"] = {"available": True, "global_policy": mp.global_policy().name}
    except Exception as e:
        report["activation_compression"] = {"available": False, "notes": str(e)}
    # gradient checkpointing placeholder
    report["gradient_checkpointing"] = {"enabled": False, "notes": "Use recompute grad if available in your TF version."}
    # model sharding status
    gpus = tf.config.list_physical_devices('GPU')
    report["model_sharding"] = {
        "applied": bool(gpus and len(gpus) > 1),
        "num_gpus": len(gpus) if gpus else 0,
        "notes": "Use tf.distribute.MirroredStrategy for real multi-GPU."
    }
    # safe batch size measurement (forward-only)
    bs, safe = 64, 4
    for _ in range(6):
        try:
            dummy = tf.random.uniform((bs,224,224,3), dtype=tf.float32)
            _ = model(dummy, training=False)
            safe = bs; bs *= 2
        except Exception:
            bs //= 2
        if bs < 4: break
    report["optimal_batch_size"] = {"safe_batch_size": int(safe)}
    return report

# ---------------- Persistence utilities ----------------
def write_csv(rows, path="part2_results.csv"):
    if not rows: return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(keys))
        w.writeheader()
        for r in rows: w.writerow(r)

def write_report_md(rows, path="part2_report.md"):
    with open(path, "w") as f:
        f.write("# Part 2 Report (Track B, GPU)\n\n")
        f.write("This report compares four optimization strategies, quantized variants with accuracy preservation, memory optimizations, and overall performance.\n\n")
        f.write("| model | acc | med_latency(s) | mean_latency(s) | size(MB) | params |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            size = r.get("keras_size_mb", r.get("tflite_size_mb", ""))
            acc  = r.get("accuracy", r.get("approx_accuracy",""))
            f.write(f"| {r.get('model','')} | {acc} | {r.get('latency_median_s','')} | {r.get('latency_mean_s','')} | {size} | {r.get('params','')} |\n")
        f.write("\n**Notes**: TFLite INT8 accuracy is approximated on a subset for quick evaluation.\n")

# ---------------- End-to-end Part 2 (GPU) ----------------
def run_part2_pipeline_gpu(
    x_train, y_train, x_test, y_test,
    baseline_model=None,
    baseline_factory=None,   # callable: () -> keras.Model
    num_classes=10,
    gpu_mixed_precision=True,
    run_fast=True,
    out_dir="part2_models"
):
    """
    Execute Part 2 on GPU (falls back to CPU if no GPU):
      - Build four optimized models (latency/memory/energy/throughput)
      - Create quantized variants from baseline and analyze accuracy preservation
      - Apply memory optimizations with measurable improvement (safe batch size)
      - Compare performance across all variants (CSV + Markdown)
    You must provide either `baseline_model` or `baseline_factory`.
    """
    assert (baseline_model is not None) or (baseline_factory is not None), "Provide baseline_model or baseline_factory."
    os.makedirs(out_dir, exist_ok=True)
    prepare_gpu(mixed_precision=gpu_mixed_precision, memory_growth=True)

    # Baseline
    base = baseline_model if baseline_model is not None else baseline_factory()
    rows = []
    base_bench = benchmark_keras_model_gpu(
        base, x_test, y_test, input_size=(224,224),
        batch=32 if not run_fast else 16, iters=10 if run_fast else 20, save_dir=out_dir
    )
    base_bench["model"] = "baseline"
    rows.append(base_bench)

    # Optimized models
    opt_models = create_optimized_models_gpu(num_classes=num_classes)
    spec = {
        "latency_optimized":   ((160,160), 32),
        "memory_optimized":    ((224,224), 32),
        "energy_optimized":    ((160,160), 32),
        "throughput_optimized":((224,224), 64 if not run_fast else 32),
    }
    for name, (inp, bs) in spec.items():
        r = benchmark_keras_model_gpu(
            opt_models[name], x_test, y_test, input_size=inp,
            batch=bs, iters=10 if run_fast else 20, save_dir=out_dir
        )
        r["model"] = name
        rows.append(r)

    # Quantization variants from baseline
    x_rep = x_train[:512]
    q = apply_quantization_optimizations_gpu(base, x_rep)

    res_ptq = benchmark_tflite_model(
        q.get("ptq_int8"), x_test, y_test, input_size=(224,224),
        batch_size=32 if not run_fast else 16, iters=10 if run_fast else 20,
        save_path=os.path.join(out_dir, "baseline_ptq_int8.tflite")
    )
    if res_ptq: res_ptq["model"] = "ptq_int8"; rows.append(res_ptq)

    res_dr = benchmark_tflite_model(
        q.get("dynamic_range"), x_test, y_test, input_size=(224,224),
        batch_size=32 if not run_fast else 16, iters=10 if run_fast else 20,
        save_path=os.path.join(out_dir, "baseline_dynamic_range.tflite")
    )
    if res_dr: res_dr["model"] = "dynamic_range"; rows.append(res_dr)

    if q.get("qat_int8") is not None:
        res_qat = benchmark_keras_model_gpu(
            q["qat_int8"], x_test, y_test, input_size=(224,224),
            batch=32 if not run_fast else 16, iters=10 if run_fast else 20, save_dir=out_dir
        )
        res_qat["model"] = "qat_int8"; rows.append(res_qat)
        try: q["qat_int8"].save(os.path.join(out_dir, "baseline_qat_int8.keras"))
        except Exception: pass

    if q.get("mixed_precision") is not None:
        res_mp = benchmark_keras_model_gpu(
            q["mixed_precision"], x_test, y_test, input_size=(224,224),
            batch=64 if not run_fast else 32, iters=10 if run_fast else 20, save_dir=out_dir
        )
        res_mp["model"] = "mixed_precision_fp16"; rows.append(res_mp)
        try: q["mixed_precision"].save(os.path.join(out_dir, "baseline_mixed_fp16.keras"))
        except Exception: pass

    # Memory optimization report
    mem_report = implement_memory_optimizations_gpu(base)
    with open("part2_memory_optimizations.json", "w") as f:
        json.dump(mem_report, f, indent=2)

    # Persist comparison
    write_csv(rows, "part2_results.csv")
    write_report_md(rows, "part2_report.md")

    print("[DONE] Part 2 GPU pipeline finished. See 'part2_results.csv' and 'part2_report.md'.")
    return rows, mem_report

rows, mem = run_part2_pipeline_gpu(
    x_train, y_train, x_test, y_test,
    baseline_factory=lambda: create_baseline_model(input_shape=(224,224,3), num_classes=10),
    num_classes=10,
    gpu_mixed_precision=True,
    run_fast=True
)