class HardwareAwareDesignMethodology:
    """Framework for hardware-aware ML system design."""

    def __init__(self):
        self.design_principles = [
            'co_design_hardware_software',
            'early_constraint_specification',
            'iterative_optimization',
            'multi_objective_optimization',
            'platform_specific_tuning'
        ]

    def constraint_specification_framework(self, application_requirements):
        """Framework for specifying hardware constraints early in design."""
        # Power budget, memory limits, latency requirements, accuracy targets
        # Normalize and expand requirements into concrete numeric thresholds.
        req = dict(application_requirements or {})
        constraints = {
            "latency_ms_target": float(req.get("latency_ms_target", 50.0)),
            "accuracy_min": float(req.get("accuracy_min", 0.85)),
            "memory_mb_limit": float(req.get("memory_mb_limit", 16.0)),
            "energy_mj_budget": float(req.get("energy_mj_budget", 0.02)),
            "platform": req.get("platform", "x86_desktop"),
            "batch_size": int(req.get("batch_size", 16))
        }
        # Derived constraints
        constraints["throughput_fps_min"] = max(1.0, 1000.0 / constraints["latency_ms_target"])
        return constraints

    def optimization_priority_framework(self, constraints):
        """Determine optimization priorities based on constraints."""
        # Create decision tree for optimization strategy selection
        prio = []
        lat = constraints["latency_ms_target"]; acc = constraints["accuracy_min"]
        mem = constraints["memory_mb_limit"]; eng = constraints["energy_mj_budget"]
        plat = constraints["platform"]

        if plat in ("arm_cortex_m7", "arm_micro"):
            prio = ["quantization", "pruning", "width_scaling", "resolution_scaling", "distillation"]
        elif plat in ("arm_cortex_a78", "mobile_gpu"):
            prio = ["resolution_scaling", "quantization", "width_scaling", "operator_fusion", "bn_folding"]
        else:  # x86_desktop
            prio = ["operator_fusion", "jit_xla", "mixed_precision", "width_scaling", "quantization"]

        # Tight memory?
        if mem < 8.0:
            prio = ["pruning", "quantization", "width_scaling"] + [p for p in prio if p not in ("pruning","quantization","width_scaling")]
        # Tight energy?
        if eng < 0.015:
            prio = ["quantization", "resolution_scaling"] + [p for p in prio if p not in ("quantization","resolution_scaling")]
        # Tight latency?
        if lat < 30:
            prio = ["resolution_scaling", "width_scaling"] + [p for p in prio if p not in ("resolution_scaling","width_scaling")]

        # Accuracy guard: keep distillation/QAT near the end to recover accuracy
        if "distillation" not in prio:
            prio.append("distillation")
        if "qat" not in prio:
            prio.append("qat")
        return prio

    def design_space_exploration(self, base_model_name, constraints):
        """Systematic exploration of design alternatives (analytic and discrete search)."""
        # We do not rebuild models here; instead propose candidates and score them via constraints.
        # Candidate knobs: alpha (width), input_size, quantization level
        alphas = [1.0, 0.75, 0.5, 0.35]
        sizes = [(224,224), (192,192), (160,160), (128,128)]
        quant = ["fp32","fp16","int8_dr","int8_ptq"]

        # Scoring: lower is better for violation, higher is better for accuracy
        # We create a simple proxy score combining latency/energy/memory vs thresholds.
        proposals = []
        for a in alphas:
            for s in sizes:
                for q in quant:
                    # rough proxy models
                    scale = (a**2) * (s[0]/224.0)**2
                    latency_ms = 50.0 * scale  # relative to a=1.0, 224 baseline
                    memory_mb = 14.0 * max(a,0.35)  # width mainly determines param size
                    energy_mj = 0.02 * (0.7*scale + 0.3)  # scale + fixed
                    acc = 0.86 - (1.0-a)*0.06 - (224-s[0])*0.0005
                    if q=="fp16":
                        energy_mj *= 0.85
                    if q.startswith("int8"):
                        energy_mj *= 0.6; memory_mb *= 0.6; acc -= 0.01
                    throughput = 1000.0/latency_ms if latency_ms>0 else 0.0

                    # constraint violation penalties
                    pen = 0.0
                    pen += max(0.0, latency_ms - constraints["latency_ms_target"]) / 100.0
                    pen += max(0.0, constraints["accuracy_min"] - acc) * 5.0
                    pen += max(0.0, memory_mb - constraints["memory_mb_limit"]) / 10.0
                    pen += max(0.0, energy_mj - constraints["energy_mj_budget"]) * 20.0

                    proposals.append({
                        "base": base_model_name,
                        "alpha": a, "input": f"{s[0]}x{s[1]}", "quant": q,
                        "latency_ms": round(latency_ms,2),
                        "memory_mb": round(memory_mb,2),
                        "energy_mj": round(energy_mj,4),
                        "accuracy": round(acc,4),
                        "throughput_fps": round(throughput,2),
                        "penalty": round(pen,4)
                    })

        # Sort by lowest penalty, then highest accuracy
        proposals = sorted(proposals, key=lambda x: (x["penalty"], -x["accuracy"]))
        return proposals[:20]  # top 20 candidates
