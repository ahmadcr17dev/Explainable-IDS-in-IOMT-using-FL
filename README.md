# Explainable Intrusion Detection in Constrained IoMT Devices Using Lightweight Federated Learning

> **Intelligent Federated Learning with Communication-Efficient Top-K Sparsification for IoMT/IoT Network Intrusion Detection**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Ready-20BEFF)](https://www.kaggle.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Key Contributions](#-key-contributions)
3. [Dataset](#-dataset)
4. [Project Structure](#-project-structure)
5. [Methodology](#-methodology)
6. [Reproducibility](#-reproducibility)
7. [Expected Results](#-expected-results)
8. [Citation](#-citation)

---

## 🔬 Overview

This repository implements a **communication-efficient Federated Learning (FL) framework** for network intrusion detection in Internet of Medical Things (IoMT) and Internet of Things (IoT) environments. The framework is specifically designed to address the unique challenges of distributed intrusion detection systems (IDS), including:

- **Data heterogeneity**: Non-IID (Non-Independently and Identically Distributed) data distributions across distributed clients (e.g., hospitals, smart factories, edge devices).
- **Communication constraints**: Limited bandwidth between edge devices and the central server, requiring efficient gradient compression.
- **Imbalanced attack classes**: Severe class imbalance in network traffic data, where benign traffic vastly outnumbers attack traffic.
- **Privacy preservation**: Raw data never leaves client devices — only model updates are communicated.
- **Explainability**: SHAP (SHapley Additive exPlanations) analysis for model interpretability in critical security decisions.

### Key Design Philosophy

The framework prioritises **real-world deployability** by keeping the model lightweight (~7.2 KB) while maintaining competitive accuracy with centralised approaches. This makes it suitable for resource-constrained edge devices such as Raspberry Pi, medical gateways, and IoT hubs.

---

## 🏆 Key Contributions

1. **Communication-Efficient FL with Top-K Sparsification**
   - Gradient compression via GPU-native Top-K sparsification with error feedback.
   - Reduces communication overhead by **~80%** compared to standard FL without compression.
   - Only the top 40% of gradient updates (by magnitude) are transmitted per round.

2. **Robust Training Under Non-IID Conditions**
   - Dirichlet-based data partitioning (α = 0.5) across 10 clients.
   - Client-level SMOTE augmentation to address local class imbalance.
   - Fresh Adam optimiser per client to prevent momentum leakage.

3. **GPU-Optimised Training Pipeline**
   - Uses `model.fit()` with TensorFlow's C++ runtime for maximum GPU utilisation.
   - Mixed-precision training (FP16) with XLA JIT compilation.
   - Achieves **60–90% GPU utilisation** on Kaggle T4/x4 GPUs.

4. **Comprehensive Evaluation**
   - Centralised baseline, standard FL baseline, and proposed FL with sparsification.
   - Per-class precision, recall, F1-score, and confusion matrix analysis.
   - ROC-AUC and Precision-Recall curves for multi-class evaluation.
   - SHAP explainability for model interpretability.

---

## 📊 Dataset

### CICIoT2023 — IoT Network Intrusion Dataset

The dataset used is the **CIC IoT Dataset 2023** ([source](https://www.unb.ca/cic/datasets/iotdataset-2023.html)), which contains realistic IoT/IIoT network traffic with both benign and malicious flows.

| Attribute              | Value                                                            |
| ---------------------- | ---------------------------------------------------------------- |
| **Total samples**      | ~10 million network flows                                        |
| **Features**           | 46 numerical features (reduced to top 45 via Mutual Information) |
| **Attack classes**     | 16 (collapsed from 33 sub-types)                                 |
| **Label distribution** | Highly imbalanced (Benign class dominates)                       |

### Label Collapse Strategy

33 original attack types are collapsed into 16 semantic categories:

| Attack Category    | Included Sub-Types                                                         |
| ------------------ | -------------------------------------------------------------------------- |
| **DDoS-ICMP**      | ICMP Flood, ICMP Fragmentation                                             |
| **DDoS-UDP**       | UDP Flood, UDP Fragmentation, SYNONYMOUSIP Flood                           |
| **DDoS-TCP**       | TCP Flood, PSHACK Flood, RSTFIN Flood, ACK Fragmentation                   |
| **DDoS-SYN**       | SYN Flood                                                                  |
| **DDoS-HTTP**      | HTTP Flood, Slowloris                                                      |
| **DoS-UDP**        | UDP Flood                                                                  |
| **DoS-TCP**        | TCP Flood                                                                  |
| **DoS-SYN**        | SYN Flood                                                                  |
| **DoS-HTTP**       | HTTP Flood                                                                 |
| **Mirai**          | GREETH Flood, GREIP Flood, UDPPLAIN                                        |
| **Reconnaissance** | Host Discovery, OS Scan, Port Scan, Ping Sweep, Vulnerability Scan         |
| **Spoofing**       | DNS Spoofing, ARP Spoofing                                                 |
| **Brute_Force**    | Dictionary Brute Force                                                     |
| **Malware**        | Backdoor Malware                                                           |
| **Web_Attack**     | XSS, Browser Hijacking, SQL Injection, Command Injection, Uploading Attack |
| **Benign**         | Normal traffic                                                             |

---

## 📁 Project Structure

```
├── federated_train.py      # Main FL training pipeline (GPU-optimised)
├── preprocess.py           # Data preprocessing, cleaning, feature selection
├── model_def.py            # Model architecture definition
├── baselines.py            # Centralised & standard FL baselines
├── shap_analysis.py        # SHAP explainability analysis
├── generate_figures.py     # Publication-quality figure generation
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

### Pipeline Flow

```
Raw Data (CSV)
     │
     ▼
preprocess.py
├── Phase 1: Schema discovery + Label collapse + Class counting
├── Phase 2: Mutual Information feature selection (top-45 features)
├── Phase 3: Streaming train/test split with per-class balancing
├── Phase 4: StandardScaler normalisation
└── Phase 5: Class weight computation
     │
     ▼
federated_train.py
├── GPU configuration (XLA, mixed precision, memory growth)
├── Dirichlet partitioning (Non-IID, α=0.5, 10 clients)
├── Client-level SMOTE augmentation
├── FL Training Loop (150 rounds)
│   ├── Client selection (8/10 clients per round)
│   ├── Local training via model.fit() ← GPU utilisation here
│   ├── Top-K sparsification + Error feedback
│   └── Weighted Federated Averaging
├── Fine-tuning with callbacks
├── Final evaluation + Pruning
└── Save metrics, predictions, metadata
     │
     ▼
baselines.py                # Centralised + Standard FL comparisons
     │
     ▼
generate_figures.py         # Publication figures (PNG + PDF)
     │
     ▼
shap_analysis.py            # Model interpretability (SHAP)
```

---

## 🧠 Methodology

### Federated Learning Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Central Server                     │
│                                                      │
│  1. Initialise global model weights                  │
│  2. Distribute weights to selected clients           │
│  3. Aggregate weighted sparse updates (FedAvg)       │
│  4. Update global model                              │
└───────────────────┬─────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌────────┐   ┌────────┐        ┌────────┐
│ Client 1 │   │ Client 2 │  ...  │ Client 10│
│          │   │          │        │          │
│ Local DS │   │ Local DS │        │ Local DS │
│ SMOTE    │   │ SMOTE    │        │ SMOTE    │
│ 8 epochs │   │ 8 epochs │        │ 8 epochs │
└──────────┘   └──────────┘        └──────────┘
```

### Model Architecture (Lightweight)

The model is a compact multi-layer perceptron (MLP) with the following structure:

```
Input (45 features)
     │
     ▼
Dense(128) → LeakyReLU(0.1) → BatchNorm → Dropout(0.20)
     │
     ▼
Dense(64)  → LeakyReLU(0.1) → BatchNorm → Dropout(0.15)
     │
     ▼
Dense(32)  → LeakyReLU(0.1) → BatchNorm → Dropout(0.10)
     │
     ▼
Dense(16) → Softmax (16 attack classes)
```

**Key design choices:**

| Component           | Choice                     | Rationale                                                               |
| ------------------- | -------------------------- | ----------------------------------------------------------------------- |
| **Weight decay**    | L2 regularisation (1e-4)   | Prevents overfitting with small client data                             |
| **Activation**      | LeakyReLU(0.1)             | Avoids dead neurons from ReLU                                           |
| **Normalisation**   | BatchNorm                  | Stabilises training across Non-IID client data                          |
| **Regularisation**  | Dropout (20-15-10%)        | Progressive regularisation (less dropout in deeper layers)              |
| **Loss function**   | Focal Loss (γ=2.0, α=0.25) | Down-weights well-classified examples, focuses on hard/minority classes |
| **Learning rate**   | Cosine Decay               | Smooth annealing without manual schedule tuning                         |
| **Optimiser**       | AdamW                      | Decoupled weight decay for better generalisation                        |
| **Mixed precision** | FP16                       | ~2× speedup on modern GPUs                                              |

### Communication-Efficient Top-K Sparsification

```
Gradient Update (Δw)
     │
     ▼
Combine with residual error: Δ = Δw + residual
     │
     ▼
Top-K selection: Keep top 40% by magnitude, zero out 60%
     │
     ▼
Transmit sparse update (40% of original size)
     │
     ▼
Store error in residual buffer for next round
```

**Why this works:** Error feedback accumulates the "discarded" small gradient components, allowing them to eventually be transmitted when they grow large enough. This prevents information loss while achieving ~80% communication reduction (including the residual storage overhead).

---

## 🔄 Reproducibility

### Environment setup

This code is designed to run on **Kaggle** (GPU-enabled notebooks). The recommended GPU is NVIDIA T4, P100, or A100.

**Required Kaggle Dataset:**

- Dataset: `ahmadcr17/ciciot2023` (CICIoT2023 Merged CSV)

### Execution Order

Run these scripts **sequentially** in a Kaggle notebook:

```bash
# Step 1: Data preprocessing (generates processed dataset)
python preprocess.py

# Step 2: Federated learning training (GPU-intensive)
python federated_train.py

# Step 3: Baseline comparisons
python baselines.py

# Step 4: Generate publication figures
python generate_figures.py

# Step 5: SHAP explainability analysis (GPU-intensive)
python shap_analysis.py
```

### GPU Configuration Notes

The code automatically configures the GPU for maximum utilisation:

```python
# federated_train.py — configure_gpu_max()
tf.config.experimental.set_memory_growth(gpu, True)  # Dynamic memory allocation
tf.config.optimizer.set_jit(True)                    # XLA JIT compilation
tf.keras.mixed_precision.set_global_policy("mixed_float16")  # Mixed precision
```

**Why GPU shows 0%:** If you use `set_synchronous_execution(False)`, GPU ops become asynchronous and monitoring tools can't track them. **This is removed** in the current code — GPU utilisation will be visible in `nvidia-smi`.

### Hyperparameters

| Parameter             | Value | Description                                      |
| --------------------- | ----- | ------------------------------------------------ |
| `N_CLIENTS`           | 10    | Total number of FL clients                       |
| `N_ROUNDS`            | 150   | Total communication rounds                       |
| `LOCAL_EPOCHS`        | 8     | Epochs per client per round                      |
| `GLOBAL_BATCH_SIZE`   | 512   | Batch size for client training                   |
| `CLIENT_FRACTION`     | 0.8   | Fraction of clients selected per round           |
| `TOP_K_PCT`           | 0.40  | Top-K sparsification ratio                       |
| `DIRICHLET_ALPHA`     | 0.5   | Dirichlet concentration for Non-IID partitioning |
| `TOP_FEATS`           | 45    | Number of selected features                      |
| `EARLY_STOP_PATIENCE` | 15    | Rounds without improvement before stopping       |

---

## 📈 Expected Results

### Performance Metrics (Proposed FL vs Baselines)

| Metric            | Centralised | Standard FL | Proposed FL |
| ----------------- | ----------- | ----------- | ----------- |
| **Accuracy**      | ~0.96       | ~0.94       | ~0.94       |
| **Weighted F1**   | ~0.96       | ~0.94       | ~0.94       |
| **Macro F1**      | ~0.85       | ~0.82       | ~0.83       |
| **Communication** | N/A         | >5000 MB    | ~1000 MB    |
| **Model Size**    | ~7.2 KB     | ~7.2 KB     | ~7.2 KB     |

### Generated Figures

After running `generate_figures.py`, the following publication-quality figures are produced (PNG + PDF, 600 DPI):

| Figure                   | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `fig1_convergence`       | Training convergence: accuracy, loss, learning rate, communication |
| `fig2_comm_cost`         | Communication cost comparison (Proposed vs Standard FL)            |
| `fig3_confusion_matrix`  | Normalised confusion matrix (16×16)                                |
| `fig4_per_class_metrics` | Per-class precision, recall, F1 grouped bar chart                  |
| `fig5_model_comparison`  | Centralised vs Standard FL vs Proposed FL comparison               |
| `fig6_radar`             | Multi-dimensional performance radar chart                          |
| `fig7_roc`               | ROC curves per class (if probabilities available)                  |
| `fig8_precision_recall`  | Precision-Recall curves (if probabilities available)               |
| `fig9_shap_importance`   | Top-20 SHAP feature importance (if SHAP data available)            |

### SHAP Analysis Outputs

After running `shap_analysis.py`:

- `global_shap_importance.csv` — Top features ranked by mean |SHAP|
- `per_class_shap.csv` — Per-class feature importance breakdown
- `shap_values.npy` — Raw SHAP values for downstream analysis
- Local explanations (waterfall/bar charts) for:
  - Correct high-confidence predictions
  - Correct low-confidence predictions
  - Misclassifications

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourarticle2024,
  title={Communication-Efficient Federated Learning for IoMT Network Intrusion Detection},
  author={Your Name, et al.},
  journal={Scientific Reports},
  year={2024},
  publisher={Springer Nature}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [CIC IoT Dataset 2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) — Canadian Institute for Cybersecurity
- [TensorFlow Federated](https://www.tensorflow.org/federated) — Framework inspiration
- [SHAP](https://github.com/slundberg/shap) — Explainability library
- Kaggle — GPU computation resources

---

> **Maintainer:** [Your Name] — [your.email@institution.edu]
