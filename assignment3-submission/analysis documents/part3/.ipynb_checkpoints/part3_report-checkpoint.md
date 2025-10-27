# Part 3 Report (Track B: Simulation & Modeling)

This report summarizes analytical models, simulation validation, and cross-platform optimization analysis.

## x86_desktop

| optimized | speedup | memory_reduction | energy_reduction | accuracy_loss |
|---|---:|---:|---:|---:|
| latency_optimized | 0.973 | 0.671 | 0.199 | 0.0200 |
| memory_optimized | 0.894 | 0.763 | -0.063 | 0.0300 |
| throughput_optimized | 0.933 | 0.000 | -0.040 | 0.0100 |

## arm_cortex_a78

| optimized | speedup | memory_reduction | energy_reduction | accuracy_loss |
|---|---:|---:|---:|---:|
| latency_optimized | 1.960 | 0.671 | 0.491 | 0.0200 |
| memory_optimized | 1.000 | 0.763 | 0.007 | 0.0300 |
| throughput_optimized | 1.000 | 0.000 | 0.000 | 0.0100 |

## arm_cortex_m7

| optimized | speedup | memory_reduction | energy_reduction | accuracy_loss |
|---|---:|---:|---:|---:|
| latency_optimized | 1.960 | 0.671 | 0.490 | 0.0200 |
| memory_optimized | 1.000 | 0.763 | 0.001 | 0.0300 |
| throughput_optimized | 1.000 | 0.000 | 0.000 | 0.0100 |

