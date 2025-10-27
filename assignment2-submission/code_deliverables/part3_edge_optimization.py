# part3

import os
import time
import json
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from keras import mixed_precision

# ====== 全局设置 ======
sparsity = tfmot.sparsity.keras
mixed_precision.set_global_policy("float32")  # Edge 侧默认 float32 训练更稳（云侧再切 mixed_float16）

# ====== 可视化工具 ======
def _savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[Saved] {path}")
    plt.close()

def plot_pruning_results(pruning_results, path="edge_pruning_results.png"):
    if not pruning_results:
        print("[Skip] pruning chart: empty results.")
        return
    keys = list(pruning_results.keys())
    ks = [k.replace("s_", "") for k in keys]
    accs = [float(pruning_results[k]["test_acc"]) for k in keys]
    sizes = [float(pruning_results[k]["tflite_kb"]) for k in keys]

    fig, ax1 = plt.subplots(figsize=(7,5))
    ax2 = ax1.twinx()
    ax1.bar(np.arange(len(keys))-0.2, accs, width=0.4)
    ax2.bar(np.arange(len(keys))+0.2, sizes, width=0.4)
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("TFLite Size (KB)")
    ax1.set_xticks(np.arange(len(keys)))
    ax1.set_xticklabels([f"sparsity={x}" for x in ks])
    ax1.set_title("Pruning: Accuracy vs TFLite Size")
    _savefig(path)

def plot_quant_sizes(quant_results, path="edge_quant_sizes.png"):
    labels, values = [], []
    if "dynamic_range" in quant_results:
        labels.append("Dynamic")
        values.append(float(quant_results["dynamic_range"]["size_kb"]))
    if "float16" in quant_results:
        labels.append("Float16")
        values.append(float(quant_results["float16"]["size_kb"]))
    if "int8" in quant_results and quant_results["int8"]:
        labels.append("INT8")
        values.append(float(quant_results["int8"]["size_kb"]))
    if not labels:
        print("[Skip] quant chart: no sizes.")
        return
    plt.figure(figsize=(6,5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel("TFLite Size (KB)")
    plt.title("Quantized TFLite Model Sizes")
    _savefig(path)

def plot_arch_vs_baseline(baseline_acc, baseline_kb, arch_acc, arch_kb, path="edge_arch_vs_baseline.png"):
    labels = ["Baseline", "Arch-Optimized"]
    accs = [baseline_acc, arch_acc]
    sizes = [baseline_kb, arch_kb]
    fig, ax1 = plt.subplots(figsize=(6,5))
    ax2 = ax1.twinx()
    x = np.arange(2)
    ax1.bar(x-0.2, accs, width=0.4)
    ax2.bar(x+0.2, sizes, width=0.4)
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Accuracy"); ax2.set_ylabel("TFLite Size (KB)")
    ax1.set_title("Baseline vs Architecture-Optimized")
    _savefig(path)

def plot_nas_scatter(search_results, path="edge_nas_search_scatter.png"):
    pts = [(r["val_acc"], r["tflite_size_kb"], r["width"], r["depth_reduce"], r["kernel_size"])
           for r in search_results if r["val_acc"] is not None and r["tflite_size_kb"] is not None]
    if not pts:
        print("[Skip] NAS scatter: insufficient points.")
        return
    plt.figure(figsize=(7,5))
    for acc, size_kb, w, d, k in pts:
        plt.scatter([size_kb], [acc], s=60)
        plt.annotate(f"w={w},d={d},k={k}", (size_kb, acc), textcoords="offset points", xytext=(6,6))
    plt.xlabel("TFLite Size (KB)")
    plt.ylabel("Validation Accuracy")
    plt.title("NAS: Accuracy vs TFLite Size")
    _savefig(path)

def plot_tflite_sizes(tflite_measures, path="edge_tflite_sizes.png"):
    labels, values = [], []
    for name in ["baseline", "arch_optimized", "best_nas"]:
        if name in tflite_measures and "dynamic" in tflite_measures[name]:
            labels.append(name)
            values.append(float(tflite_measures[name]["dynamic"]["size_kb"]))
    if not labels:
        print("[Skip] tflite sizes: no dynamic sizes found.")
        return
    plt.figure(figsize=(6,5))
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels)
    plt.ylabel("TFLite Size (KB)")
    plt.title("TFLite Sizes (Dynamic Range Quantization)")
    _savefig(path)

# ====== Edge 优化器 ======
class EdgeOptimizer:
    def __init__(self, baseline_model_path):
        try:
            self.baseline_model = tf.keras.models.load_model(baseline_model_path)
        except Exception as e:
            print("Warning: load_model failed; retry with compile=False. Error:", e)
            self.baseline_model = tf.keras.models.load_model(baseline_model_path, compile=False)

    # 1) Pruning
    def implement_pruning(self, target_sparsity=0.75, fine_tune_epochs=3, train_ds=None, val_ds=None):
        model = self.baseline_model
        pruning_params = {
            'pruning_schedule': sparsity.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=float(target_sparsity),
                begin_step=0,
                end_step=1000
            )
        }
        pruned_model = sparsity.prune_low_magnitude(model, **pruning_params)
        pruned_model.compile(optimizer=tf.keras.optimizers.Adam(),
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])
        if train_ds is not None and val_ds is not None and fine_tune_epochs > 0:
            callbacks = [sparsity.UpdatePruningStep()]
            pruned_model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                             callbacks=callbacks, verbose=0)
        stripped = sparsity.strip_pruning(pruned_model)
        return stripped

    # 2) Quantization
    def implement_quantization(self):
        quantized = {}
        model = self.baseline_model

        def convert_with_converter(converter, tag):
            tflite_model = converter.convert()
            f = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{tag}.tflite")
            f.write(tflite_model); f.flush(); f.close()
            size_kb = os.path.getsize(f.name) / 1024.0
            return {'tflite_bytes': tflite_model, 'file': f.name, 'size_kb': size_kb}

        # dynamic
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized['dynamic_range'] = convert_with_converter(converter, 'dynamic_range')

        # float16
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = {tf.float16}
        quantized['float16'] = convert_with_converter(converter, 'float16')

        # int8（此处保持随机代表性数据，仅为跑通；实际请替换为来自训练集的小样本）
        def representative_dataset():
            for _ in range(100):
                yield [np.random.rand(1, *model.input_shape[1:]).astype(np.float32)]
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            quantized['int8'] = convert_with_converter(converter, 'int8')
        except Exception as e:
            quantized['int8_error'] = str(e)

        return quantized

    # 3) Architecture optimization（扩展签名以支持 NAS 调用）
    def implement_architecture_optimization(self,
                                           input_shape=None,
                                           num_classes=None,
                                           width_multiplier=1.0,
                                           reduce_depth_by=0,
                                           kernel_size=3):
        if input_shape is None:
            input_shape = self.baseline_model.input_shape[1:]
        if num_classes is None:
            num_classes = self.baseline_model.output_shape[-1]

        def _c(v):  # 按宽度倍率调整通道数，至少为 8
            return max(8, int(round(v * width_multiplier)))

        inputs = keras.Input(shape=input_shape)
        x = inputs

        # 初始可分离卷积
        x = keras.layers.SeparableConv2D(_c(16), kernel_size, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(max_value=6)(x)

        # 轻量块，减少深度
        blocks = [32, 64, 96][:max(0, 3 - int(reduce_depth_by))]
        for f in blocks:
            x = keras.layers.SeparableConv2D(_c(f), kernel_size, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU(max_value=6)(x)
            x = keras.layers.MaxPooling2D()(x)

        x = keras.layers.GlobalMaxPooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(_c(64), activation='relu')(x)
        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)
        return model

    # 4) Simplified NAS（修复：改为通过参数传入数据与形状）
    def implement_neural_architecture_search(self, train_ds, val_ds, input_shape, num_classes,
                                             trials=5, kernel_choices=(3,5), width_choices=(0.25,0.5,0.75,1.0),
                                             depth_choices=(0,1,2)):
        search_results = []

        def evaluate_arch(width, depth_reduce, kernel_size):
            candidate = self.implement_architecture_optimization(
                input_shape=input_shape,
                num_classes=num_classes,
                width_multiplier=width,
                reduce_depth_by=depth_reduce,
                kernel_size=kernel_size
            )
            candidate.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            val_acc = None
            if train_ds is not None and val_ds is not None:
                candidate.fit(train_ds, epochs=1, validation_data=val_ds, verbose=0)
                val_acc = candidate.evaluate(val_ds, verbose=0)[1]

            size_kb = None
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(candidate)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_bytes = converter.convert()
                size_kb = len(tflite_bytes) / 1024.0
            except Exception as e:
                print(f"TFLite conversion failed: {e}")

            return {
                "width": width,
                "depth_reduce": depth_reduce,
                "kernel_size": kernel_size,
                "val_acc": val_acc,
                "tflite_size_kb": size_kb,
                "params": candidate.count_params(),
                "model": candidate
            }

        import random
        for _ in range(trials):
            w = random.choice(width_choices)
            d = random.choice(depth_choices)
            k = random.choice(kernel_choices)
            search_results.append(evaluate_arch(w, d, k))

        ranked = sorted(
            [r for r in search_results if (r["val_acc"] is not None and r["tflite_size_kb"] is not None)],
            key=lambda x: (-x["val_acc"], x["tflite_size_kb"])
        )
        best = ranked[0] if ranked else None
        best_model = best["model"] if best else None
        return best_model, search_results

    # 5) TFLite conversion + (optional) latency measure
    def create_tflite_models(self, models_dict, test_samples=None, runs=50):
        tflite_models = {}

        def measure_inference(tflite_bytes, input_data):
            interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            timings = []
            for _ in range(runs):
                idx = np.random.randint(0, input_data.shape[0])
                sample = input_data[idx:idx+1].astype(np.float32)
                if input_details[0]['dtype'] == np.int8:
                    scale, zero_point = input_details[0]['quantization']
                    sample = (sample / scale + zero_point).astype(np.int8)
                interpreter.set_tensor(input_details[0]['index'], sample)
                t0 = time.time(); interpreter.invoke(); t1 = time.time()
                timings.append((t1 - t0) * 1000.0)
            return float(np.mean(timings))

        for name, model in models_dict.items():
            if model is None:
                continue
            tflite_models[name] = {}

            # dynamic
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_dyn = converter.convert()
            size_dyn = len(tflite_dyn) / 1024.0
            tflite_models[name]['dynamic'] = {'size_kb': size_dyn}

            # float16
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = {tf.float16}
            tflite_f16 = converter.convert()
            size_f16 = len(tflite_f16) / 1024.0
            tflite_models[name]['float16'] = {'size_kb': size_f16}

            # 可选 INT8（需代表性数据集）
            # if representative_dataset: ...（此处省略，与上面 implement_quantization 相同思想）

            # 可选：测延迟
            if test_samples is not None:
                tflite_models[name]['dynamic']['latency_ms'] = measure_inference(tflite_dyn, test_samples)
                tflite_models[name]['float16']['latency_ms'] = measure_inference(tflite_f16, test_samples)

            tflite_models[name]['accuracy_check'] = "TODO"
        return tflite_models


# ====== 基准主流程 ======
def benchmark_edge_optimizations():
    optimizer = EdgeOptimizer('baseline_model.keras')
    results = {}

    # 数据
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test  = keras.utils.to_categorical(y_test, 10)

    # 小子集用于快速实验
    train_x = x_train[:2000]; train_y = y_train[:2000]
    val_x   = x_train[2000:2500]; val_y = y_train[2000:2500]
    test_x  = x_test[:500];  test_y = y_test[:500]

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).shuffle(2000).batch(64, drop_remainder=True)
    val_ds   = tf.data.Dataset.from_tensor_slices((val_x, val_y)).batch(64, drop_remainder=True)
    test_ds  = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(64, drop_remainder=True)

    # 1) Pruning
    pruning_results = {}
    for s in [0.5, 0.75]:
        pruned = optimizer.implement_pruning(target_sparsity=s, fine_tune_epochs=2, train_ds=train_ds, val_ds=val_ds)
        pruned.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        acc = float(pruned.evaluate(test_ds, verbose=0)[1])
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_pruned_s{s}.tflite")
        tmp.write(tflite_bytes); tmp.close()
        size_kb = os.path.getsize(tmp.name) / 1024.0
        pruning_results[f"s_{s}"] = {'test_acc': acc, 'tflite_kb': size_kb, 'params': int(pruned.count_params())}
    results['pruning'] = pruning_results

    # 2) Quantization（基于 baseline）
    quant_results = optimizer.implement_quantization()
    int8_info = quant_results.get('int8', None)
    if isinstance(int8_info, dict):
        int8_info = {
            'size_kb': float(int8_info.get('size_kb', float('nan'))),
            'file': str(int8_info.get('file', ''))
        }

    results['quantization'] = {
        'dynamic_size_kb': float(quant_results['dynamic_range']['size_kb']),
        'float16_size_kb': float(quant_results['float16']['size_kb']),
        'int8_info': int8_info
    }


    # 3) Architecture optimization vs baseline
    opt_model = optimizer.implement_architecture_optimization()
    opt_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    opt_model.fit(train_ds, epochs=2, validation_data=val_ds, verbose=0)
    opt_acc = float(opt_model.evaluate(test_ds, verbose=0)[1])
    # baseline quick eval
    base = optimizer.baseline_model
    base.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    base_acc = float(base.evaluate(test_ds, verbose=0)[1])

    # TFLite sizes
    def size_of_tflite(model):
        conv = tf.lite.TFLiteConverter.from_keras_model(model); conv.optimizations=[tf.lite.Optimize.DEFAULT]
        b = conv.convert(); return len(b)/1024.0

    base_kb = size_of_tflite(base)
    arch_kb = size_of_tflite(opt_model)
    results['architecture_optimization'] = {'baseline_acc': base_acc, 'baseline_kb': base_kb,
                                            'test_acc': opt_acc, 'tflite_kb': arch_kb,
                                            'params': int(opt_model.count_params())}

    # 4) NAS（修复：传参）
    best_model, search_results = optimizer.implement_neural_architecture_search(
        train_ds=train_ds, val_ds=val_ds, input_shape=base.input_shape[1:], num_classes=base.output_shape[-1],
        trials=5
    )
    if best_model is not None:
        best_params = int(best_model.count_params())
    else:
        best_params = None
    results['nas'] = {
        'best_params': best_params,
        'search_summary': [
            {
                'width': r['width'],
                'depth_reduce': r['depth_reduce'],
                'kernel_size': r['kernel_size'],
                'val_acc': r['val_acc'],
                'params': r['params'],
                'tflite_kb': r['tflite_size_kb']
            } for r in search_results
        ]
    }

    # 5) TFLite conversion + (optional) latency
    models_to_convert = {'baseline': base, 'arch_optimized': opt_model, 'best_nas': best_model}
    tflite_measures = optimizer.create_tflite_models(models_to_convert, test_samples=None)
    results['tflite_measures'] = tflite_measures

    # ========== 可视化 ==========
    plot_pruning_results(results['pruning'])
    plot_quant_sizes(quant_results)
    plot_arch_vs_baseline(base_acc, base_kb, opt_acc, arch_kb)
    plot_nas_scatter(search_results)
    plot_tflite_sizes(tflite_measures)

    
    # 导出 JSON
    with open("edge_optimization_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("[Saved] edge_optimization_report.json")

    return results


if __name__ == "__main__":
    res = benchmark_edge_optimizations()
    import pprint; pprint.pprint(res)
