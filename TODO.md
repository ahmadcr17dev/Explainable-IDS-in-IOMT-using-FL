# Federated Learning Project — Fix Summary ✅

## GPU Utilization Fixes (federated_train.py)

| #   | Change                                                                                     | Status |
| --- | ------------------------------------------------------------------------------------------ | ------ |
| 1   | Removed `set_synchronous_execution(False)` — was causing nvidia-smi to show 0%             | ✅     |
| 2   | Replaced manual GradientTape loop with **`model.fit()`** — GPU-optimized C++ runtime       | ✅     |
| 3   | Replaced `train_client_step_gpu()` with `train_client_with_fit()` — fresh AdamW per client | ✅     |
| 4   | Replaced manual `evaluate_model_gpu()` with `model.evaluate()` — GPU-optimized             | ✅     |
| 5   | Replaced manual fine-tuning loop with `model.fit()` + callbacks                            | ✅     |
| 6   | Removed `prefetch_to_device("/GPU:0")` — was causing Kaggle compatibility issues           | ✅     |
| 7   | Removed optimizer creation inside `@tf.function` — prevents re-tracing                     | ✅     |
| 8   | Fresh optimizer per client via recompile — prevents optimizer state leakage                | ✅     |

## Accuracy Improvements

| #   | Change                                                              | File                 | Status |
| --- | ------------------------------------------------------------------- | -------------------- | ------ |
| 1   | Increased `TOP_FEATS` from 25 → **45** (45 features for 16 classes) | `preprocess.py`      | ✅     |
| 2   | Exported `focal_loss_fn` for reuse                                  | `model_def.py`       | ✅     |
| 3   | Client training now uses **Focal Loss + CosineDecay LR + AdamW**    | `federated_train.py` | ✅     |
| 4   | Per-client `total_steps` for proper CosineDecay scheduling          | `federated_train.py` | ✅     |
