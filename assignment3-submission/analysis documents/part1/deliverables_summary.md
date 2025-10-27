# Part 1 Deliverables Summary

## Baseline MobileNetV2 Implementation
- Implemented with ImageNet weights, GAP + Dropout + Dense head.

## Training Logs (Transfer Learning)
- See `training_log_head.csv` and `training_log_ft.csv`.
- Strategy details: `training_strategy.txt`.

## Test Accuracy
- Accuracy: 0.8500 (>= 0.85 target: YES)

## Baseline Performance Metrics
- Single inference (median s): 0.0055675459443591535
- Batch inference (median s): 0.020177661965135485
- Memory RSS (MB): 5505.05859375
- Model size (MB): 9.058212280273438
- Params: 2270794
- FLOPs: 599076924
- Energy single (J): 0.11135091888718307
- Energy batch (J): 0.4035532393027097
