# Explainable Lightweight Federated Learning-Based Intrusion Detection System for IoMT

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

Official implementation accompanying the research paper:

**"Explainable Lightweight Federated Learning-Based Intrusion Detection System for Internet of Medical Things Using SHAP"**

---

## Overview

The rapid deployment of Internet of Medical Things (IoMT) devices has significantly increased cybersecurity risks in healthcare systems. Traditional centralized intrusion detection systems require transferring sensitive medical data to a central server, creating privacy concerns and regulatory challenges.

This project proposes a **privacy-preserving Lightweight Federated Learning Intrusion Detection System (FL-IDS)** for IoMT environments.

The framework combines:

- Lightweight Neural Networks
- Federated Learning (FedAvg)
- SHAP Explainable AI
- Communication-Efficient Aggregation
- Magnitude-Based Model Pruning

allowing distributed intrusion detection without sharing raw medical data.

---

## Main Contributions

This research proposes:

- Lightweight neural network suitable for resource-constrained IoMT devices
- Federated Learning architecture for privacy-preserving collaborative training
- SHAP-based explainability for transparent intrusion detection
- Communication-efficient model aggregation
- Magnitude pruning for lightweight deployment
- Evaluation using the CICIoT2023 benchmark dataset

---

## Framework

The proposed workflow consists of the following stages:

1. Dataset preprocessing
2. Feature selection using Mutual Information
3. Data normalization
4. Federated client partitioning
5. Local client training
6. FedAvg aggregation
7. Global model optimization
8. Model pruning
9. SHAP explainability
10. Performance evaluation

---

## Dataset

Experiments are performed using the **CICIoT2023** benchmark dataset developed by the Canadian Institute for Cybersecurity (CIC).

The dataset contains realistic IoT traffic including:

- Benign Traffic
- DDoS
- DoS
- Reconnaissance
- Spoofing
- Mirai
- Web Attacks
- Brute Force
- Malware

Dataset characteristics:

- Millions of network flows
- Multiple attack categories
- Realistic IoT traffic
- Modern attack scenarios
- Suitable for Federated Learning research

---

## Model Architecture

The proposed lightweight neural network consists of:

Input Layer

↓

Dense (256)

↓

Batch Normalization

↓

LeakyReLU

↓

Dropout

↓

Dense (128)

↓

Batch Normalization

↓

LeakyReLU

↓

Dropout

↓

Dense (64)

↓

Batch Normalization

↓

LeakyReLU

↓

Dropout

↓

Softmax Output Layer

---

## Federated Learning Configuration

- Federated Averaging (FedAvg)
- Multiple distributed clients
- Local client training
- Global model aggregation
- Communication-efficient updates
- Privacy-preserving learning

No raw medical data are transmitted between clients.

---

## Explainable AI

Model decisions are interpreted using **SHAP (SHapley Additive Explanations)**.

Generated explanations include:

- Global feature importance
- Per-class feature importance
- Local prediction explanations
- Feature contribution visualization

---

## Repository Structure

```
project/
│
├── preprocess.py
├── model_def.py
├── federated_train.py
├── shap_analysis.py
│
├── processed/
│
├── results/
│
├── figures/
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/USERNAME/REPOSITORY.git

cd REPOSITORY

pip install -r requirements.txt
```

---

## Running the Project

### Step 1

Preprocess dataset

```bash
python preprocess.py
```

### Step 2

Train Federated Learning model

```bash
python federated_train.py
```

### Step 3

Generate SHAP explanations

```bash
python shap_analysis.py
```

---

## Evaluation Metrics

The proposed model is evaluated using:

- Accuracy
- Precision
- Recall
- Weighted F1-score
- Classification Report
- Communication Cost
- SHAP Feature Importance

---

## Generated Outputs

Training produces:

```
results/

global_model.h5

classification_report.json

round_metrics.csv

meta.json

feature_importance.csv

y_pred.npy

y_test.npy
```

Figures include:

```
figures/

fig3_shap_global.png

fig_shap_heatmap.png

local_explanations/
```

---

## Research Motivation

The proposed framework addresses three major challenges in IoMT security:

- Data Privacy
- Explainability
- Lightweight Deployment

while maintaining competitive intrusion detection performance.

---

## Citation

If you use this work, please cite:

```
@article{YourPaper2026,
  title={Explainable Lightweight Federated Learning-Based Intrusion Detection System for Internet of Medical Things Using SHAP},
  author={Muhammad Ahmad Mobeen and Others},
  journal={Construction Innovation (Under Review)},
  year={2026}
}
```

---

## License

MIT License

---

## Acknowledgements

- Canadian Institute for Cybersecurity (CIC)
- CICIoT2023 Dataset
- TensorFlow
- SHAP
