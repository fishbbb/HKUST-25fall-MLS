# Part 1 Deliverables Summary

## Baseline MobileNetV2 Implementation
- Implemented with ImageNet weights, GAP + Dropout + Dense head.

## Training Logs (Transfer Learning)
- See `training_log_head.csv` and `training_log_ft.csv`.
- Strategy details: `training_strategy.txt`.

## Test Accuracy
- Accuracy: 0.8502 (>= 0.85 target: YES)

## Baseline Performance Metrics
- Single inference (median s): 0.005273515998851508
- Batch inference (median s): 0.020048932521604
- Memory RSS (MB): 5662.15625
- Model size (MB): 9.058212280273438
- Params: 2270794
- FLOPs: 599076924
- Energy single (J): 0.10547031997703016
- Energy batch (J): 0.40097865043208003
