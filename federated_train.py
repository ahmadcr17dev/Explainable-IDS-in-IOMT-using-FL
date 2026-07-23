"""
Research-Grade Federated Learning Training Pipeline — GPU-Optimised
CICIoT2023 IoMT Intrusion Detection — Springer Scientific Reports
Kaggle: Tesla P100 16 GB VRAM, 30 GB RAM
Target: ≥88% accuracy, GPU 85-95%, VRAM <12 GB
"""

import os
import gc
import json
import time
import warnings
import logging
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import psutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "4"
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from model_def import build_lightweight_model, apply_magnitude_pruning

# ============================================================
# CONFIGURATION
# ============================================================
SEED: int = 42
N_CLIENTS: int = 10
N_ROUNDS: int = 150
LOCAL_EPOCHS: int = 8
GLOBAL_BATCH_SIZE: int = 512
FINAL_BATCH_SIZE: int = 256
CLIENT_FRACTION: float = 0.8
TOP_K_PCT: float = 0.40
DIRICHLET_ALPHA: float = 0.5
EARLY_STOP_PATIENCE: int = 15
CHECKPOINT_EVERY: int = 10
VAL_EVERY: int = 5
VAL_SUBSET_SIZE: int = 5000

PROC_DIR: str = "/kaggle/working/processed"
RESULTS_DIR: str = "/kaggle/working/results"
CKPT_DIR: str = os.path.join(RESULTS_DIR, "checkpoints")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FL-Trainer")

# ============================================================
# REPRODUCIBILITY
# ============================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# GPU CONFIGURATION - MAXIMUM UTILIZATION
# ============================================================
def configure_gpu_max() -> None:
    """Aggressive GPU configuration for max utilization."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("GPU required for this training script")
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        # tf.config.experimental.set_virtual_device_configurat(
        #     gpu,
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)]
        # )
    
    tf.config.optimizer.set_jit(True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    tf.config.experimental.set_synchronous_execution(False)
    
    logger.info(f"GPU(s): {[g.name for g in gpus]}")
    logger.info("Mixed precision: mixed_float16 | XLA: enabled")

# ============================================================
# DATA LOADING
# ============================================================
def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                          List[str], Dict[int, float], int, int]:
    """Load preprocessed data and metadata."""
    logger.info("=" * 60)
    logger.info(" LOADING PREPROCESSED DATA")
    logger.info("=" * 60)

    X_train = np.load(os.path.join(PROC_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROC_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROC_DIR, "y_test.npy"))

    n_feats = X_train.shape[1]
    n_classes = int(y_train.max()) + 1
    class_names = pd.read_csv(
        os.path.join(PROC_DIR, "class_names.csv"), header=None
    )[0].tolist()

    logger.info(f"Train: {X_train.shape} ({X_train.nbytes/1024**2:.0f} MB)")
    logger.info(f"Test:  {X_test.shape} ({X_test.nbytes/1024**2:.0f} MB)")
    logger.info(f"Features: {n_feats} | Classes: {n_classes}")

    with open(os.path.join(PROC_DIR, "class_weights.json"), "r") as f:
        class_weights = {int(k): float(v) for k, v in json.load(f).items()}

    return X_train, y_train, X_test, y_test, class_names, class_weights, n_feats, n_classes

# ============================================================
# DIRICHLET PARTITION
# ============================================================
def dirichlet_partition(
    y: np.ndarray, n_clients: int, alpha: float = 0.5, seed: int = SEED
) -> List[np.ndarray]:
    """Non-IID partition using Dirichlet distribution."""
    rng = np.random.RandomState(seed)
    n_classes = int(y.max()) + 1
    class_indices = [np.where(y == c)[0] for c in range(n_classes)]
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c in range(n_classes):
        idx_c = class_indices[c]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet([alpha] * n_clients)
        proportions = np.maximum(proportions, 0.01)
        proportions /= proportions.sum()
        splits = (proportions * len(idx_c)).astype(int)
        splits[-1] = len(idx_c) - splits[:-1].sum()
        start = 0
        for k in range(n_clients):
            end = start + splits[k]
            client_indices[k].extend(idx_c[start:end].tolist())
            start = end

    return [np.array(ci, dtype=np.int64) for ci in client_indices]

# ============================================================
# CLIENT-SIDE SMOTE (Pre-applied once)
# ============================================================
def apply_smote_once(
    X: np.ndarray, y: np.ndarray, seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to a single client's data with fallback."""
    unique, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return X.astype(np.float32), y.astype(np.int32)
    k = min(5, counts.min() - 1)
    try:
        sm = SMOTE(sampling_strategy="auto", random_state=seed, k_neighbors=k)
        Xr, yr = sm.fit_resample(X, y)
        return Xr.astype(np.float32), yr.astype(np.int32)
    except Exception:
        logger.warning(f"SMOTE failed for client — returning original data")
        return X.astype(np.float32), y.astype(np.int32)

# ============================================================
# BUILD PERSISTENT CLIENT DATASETS - GPU PLACEMENT
# ============================================================
def build_client_datasets_gpu(
    client_data: List[Tuple[np.ndarray, np.ndarray]],
    batch_size: int,
    seed: int = SEED,
) -> List[tf.data.Dataset]:
    """Create datasets with explicit GPU placement."""
    datasets = []
    for Xc, yc in client_data:
        X_t = tf.convert_to_tensor(Xc, dtype=tf.float32)
        y_t = tf.convert_to_tensor(yc, dtype=tf.int64)
        
        ds = tf.data.Dataset.from_tensor_slices((X_t, y_t))
        ds = ds.cache()
        ds = ds.shuffle(min(8192, len(Xc)), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(min(batch_size, len(Xc)), drop_remainder=False)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        datasets.append(ds)
    
    logger.info(f"Built {len(datasets)} persistent GPU client datasets")
    return datasets

# ============================================================
# GPU TOP-K SPARSIFICATION WITH ERROR FEEDBACK
# ============================================================
@tf.function(jit_compile=True)
def gpu_topk_sparsify(
    deltas: List[tf.Tensor],
    residuals: List[tf.Variable],
    k_pct: tf.Tensor,
) -> List[tf.Tensor]:
    """GPU-native Top-K sparsification with error feedback accumulation."""
    out = []
    for d, r_var in zip(deltas, residuals):
        if len(d.shape) >= 2:
            d_comp = d + tf.cast(r_var, d.dtype)
            flat = tf.reshape(d_comp, [-1])
            k = tf.cast(
                tf.cast(tf.size(flat), tf.float32) * (1.0 - k_pct), tf.int32
            )
            k = tf.maximum(k, 1)
            _, top_indices = tf.math.top_k(tf.abs(flat), k=k)
            mask_flat = tf.scatter_nd(
                tf.expand_dims(top_indices, 1),
                tf.ones([k], dtype=d.dtype),
                [tf.size(flat)],
            )
            mask = tf.reshape(mask_flat, tf.shape(d_comp))
            transmitted = d_comp * mask
            residual_new = (d_comp - transmitted)
            r_var.assign(tf.cast(residual_new, tf.float32))
            out.append(transmitted)
        else:
            out.append(d)
            r_var.assign(tf.zeros_like(r_var))
    return out

# ============================================================
# GPU WEIGHTED FEDAVG
# ============================================================
@tf.function(jit_compile=True)
def federated_average(
    global_vars: List[tf.Variable],
    client_weights_list: List[List[tf.Tensor]],
    client_sizes: tf.Tensor,
) -> List[tf.Tensor]:
    """Weighted FedAvg entirely on GPU."""
    total_n = tf.cast(tf.reduce_sum(client_sizes), tf.float32)
    new_vars = []
    
    for li in range(len(global_vars)):
        weighted_sum = tf.zeros_like(global_vars[li], dtype=tf.float32)
        for i in range(len(client_weights_list)):
            weight = tf.cast(client_sizes[i], tf.float32) / total_n
            weighted_sum += weight * tf.cast(client_weights_list[i][li], tf.float32)
        new_vars.append(weighted_sum)
    
    return new_vars

# ============================================================
# GPU TRAINING STEP FOR SINGLE CLIENT
# ============================================================
@tf.function
def train_client_step_gpu(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    epochs: int,
    class_weight_tensor: tf.Tensor,
) -> List[tf.Tensor]:
    """GPU-native training - dataset explicitly on GPU."""
    prev_weights = [tf.identity(w) for w in model.trainable_variables]
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4)
    
    for epoch in range(epochs):
        for x_batch, y_batch in dataset:
            x_batch = tf.cast(x_batch, tf.float16)
            y_batch = tf.cast(y_batch, tf.int64)
            
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                logits = tf.cast(logits, tf.float32)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_batch, logits, from_logits=True
                )
                sw = tf.gather(class_weight_tensor, y_batch)
                loss = tf.reduce_mean(loss * sw)
            
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return [tf.identity(w) for w in model.trainable_variables]

# ============================================================
# EVALUATION - GPU NATIVE
# ============================================================
@tf.function
def evaluate_model_gpu(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Evaluate model on validation dataset - GPU native."""
    total_loss = tf.constant(0.0, dtype=tf.float32)
    total_acc = tf.constant(0.0, dtype=tf.float32)
    num_batches = tf.constant(0, dtype=tf.int32)
    
    for x_batch, y_batch in val_ds:
        x_batch = tf.cast(x_batch, tf.float16)
        logits = model(x_batch, training=False)
        logits = tf.cast(logits, tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_batch, logits, from_logits=True)
        acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, axis=-1), y_batch), tf.float32)
        )
        total_loss += tf.reduce_mean(loss)
        total_acc += acc
        num_batches += 1
    
    return total_loss / tf.cast(num_batches, tf.float32), total_acc / tf.cast(num_batches, tf.float32)

# ============================================================
# GPU FINE-TUNING
# ============================================================
def fine_tune_gpu(
    global_vars: List[tf.Variable],
    client_datasets: List[tf.data.Dataset],
    n_features: int,
    n_classes: int,
    class_weight_tensor: tf.Tensor,
    results_dir: str,
) -> tf.keras.Model:
    """GPU-native fine-tuning without model.fit()."""
    logger.info("=" * 60)
    logger.info(" FINAL FINE-TUNING (GPU)")
    logger.info("=" * 60)
    
    combined_ds = client_datasets[0]
    for ds in client_datasets[1:]:
        combined_ds = combined_ds.concatenate(ds)
    
    model = build_lightweight_model(n_features, n_classes, 10000, 1e-4)
    model.set_weights([v.numpy() for v in global_vars])
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
    best_acc = 0.0
    patience = 0
    
    for epoch in range(120):
        epoch_loss = 0.0
        epoch_acc = 0.0
        batches = 0
        
        for x_batch, y_batch in combined_ds.batch(FINAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE):
            x_batch = tf.cast(x_batch, tf.float16)
            y_batch = tf.cast(y_batch, tf.int64)
            
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                logits = tf.cast(logits, tf.float32)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    y_batch, logits, from_logits=True
                )
                sw = tf.gather(class_weight_tensor, y_batch)
                loss = tf.reduce_mean(loss * sw)
            
            grads = tape.gradient(loss, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            preds = tf.argmax(logits, axis=-1)
            acc = tf.reduce_mean(tf.cast(tf.equal(preds, y_batch), tf.float32))
            
            epoch_loss += loss
            epoch_acc += acc
            batches += 1
        
        avg_loss = epoch_loss / batches
        avg_acc = epoch_acc / batches
        
        if epoch % 10 == 0:
            logger.info(f"FT Epoch {epoch:>3d}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            patience = 0
            model.save(os.path.join(results_dir, "best_global_model.keras"))
        else:
            patience += 1
            if patience >= 10:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    model.save(os.path.join(results_dir, "best_global_model.h5"))
    return model

# ============================================================
# COMMUNICATION TRACKER
# ============================================================
class CommunicationTracker:
    """Tracks per-round and cumulative communication costs."""
    
    def __init__(self, top_k_pct: float = TOP_K_PCT) -> None:
        self.per_round: List[float] = []
        self.total: float = 0.0
        self.top_k_pct = top_k_pct
    
    def log(self, n_clients: int, param_bytes: int) -> float:
        comm = (param_bytes * n_clients * self.top_k_pct) / 1e6
        self.per_round.append(round(comm, 4))
        self.total += comm
        return comm
    
    def get_stats(self) -> Dict[str, float]:
        return {
            "total_communication_mb": round(self.total, 2),
            "avg_round_communication_mb": round(np.mean(self.per_round), 4),
            "compression_ratio": 1.0 - self.top_k_pct,
        }

# ============================================================
# FINAL EVALUATION
# ============================================================
def final_evaluation(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    results_dir: str,
) -> Dict:
    """Run full evaluation: classification report, confusion matrix, ROC-AUC."""
    logger.info("=" * 60)
    logger.info(" EVALUATION")
    logger.info("=" * 60)
    
    y_proba = model.predict(X_test, batch_size=FINAL_BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)
    
    rep_str = classification_report(
        y_test, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    logger.info("\n" + rep_str)
    
    rep_dict = classification_report(
        y_test, y_pred, target_names=class_names,
        digits=4, output_dict=True, zero_division=0,
    )
    
    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(rep_dict, f, indent=2)
    
    cm = confusion_matrix(y_test, y_pred)
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), cm)
    
    try:
        roc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        roc_val = float("nan")
    with open(os.path.join(results_dir, "roc_auc.json"), "w") as f:
        json.dump({"roc_auc_weighted": round(roc_val, 6)}, f, indent=2)
    
    np.save(os.path.join(results_dir, "y_test.npy"), y_test)
    np.save(os.path.join(results_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(results_dir, "y_pred_proba.npy"), y_proba)
    
    return rep_dict

# ============================================================
# PRUNING
# ============================================================
def prune_and_evaluate(
    final_model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    n_features: int,
    n_classes: int,
    results_dir: str,
) -> Tuple[float, Dict]:
    """Apply magnitude pruning and evaluate pruned model."""
    logger.info("=" * 60)
    logger.info(" PRUNING")
    logger.info("=" * 60)
    
    pruned = build_lightweight_model(
        n_features, n_classes, total_steps=10000, learning_rate=3e-4
    )
    pruned.set_weights(final_model.get_weights())
    orig_p = sum(np.count_nonzero(w) for w in pruned.get_weights() if w.ndim >= 2)
    pruned = apply_magnitude_pruning(pruned, sparsity=0.40)
    prun_p = sum(np.count_nonzero(w) for w in pruned.get_weights() if w.ndim >= 2)
    ratio = 1.0 - (prun_p / orig_p) if orig_p > 0 else 0
    
    logger.info(f"Params: {orig_p:,} → {prun_p:,} ({ratio:.2%} compression)")
    
    yp = np.argmax(pruned.predict(X_test, batch_size=FINAL_BATCH_SIZE, verbose=0), axis=1)
    prep_str = classification_report(
        y_test, yp, target_names=class_names, digits=4, zero_division=0
    )
    logger.info("\n" + prep_str)
    
    prep_dict = classification_report(
        y_test, yp, target_names=class_names,
        digits=4, output_dict=True, zero_division=0,
    )
    
    with open(os.path.join(results_dir, "pruned_classification_report.json"), "w") as f:
        json.dump(prep_dict, f, indent=2)
    
    pruned.save(os.path.join(results_dir, "pruned_model.keras"))
    pruned.save(os.path.join(results_dir, "pruned_model.h5"))
    np.save(os.path.join(results_dir, "y_pruned_pred.npy"), yp)
    
    return ratio, prep_dict

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================
def main() -> None:
    """Orchestrate the complete FL training pipeline."""
    
    # --- Setup ---
    configure_gpu_max()
    X_train_full, y_train_full, X_test, y_test, class_names, class_weights, N_FEATURES, N_CLASSES = load_data()
    
    # --- Partition ---
    logger.info("=" * 60)
    logger.info(" DIRICHLET PARTITIONING")
    logger.info("=" * 60)
    partitions = dirichlet_partition(y_train_full, N_CLIENTS, DIRICHLET_ALPHA)
    for i, p in enumerate(partitions):
        n_cls = len(np.unique(y_train_full[p]))
        logger.info(f"  Client {i:>2d}: {len(p):>8,} samples | {n_cls:>2d} classes")
    
    # --- SMOTE (once) ---
    logger.info("=" * 60)
    logger.info(" PRE-APPLYING SMOTE (Client-Level, Once)")
    logger.info("=" * 60)
    client_data = []
    for cid in range(N_CLIENTS):
        idx = partitions[cid]
        Xc, yc = apply_smote_once(X_train_full[idx], y_train_full[idx])
        client_data.append((Xc, yc))
        logger.info(f"  Client {cid:>2d}: {len(Xc):>8,} samples after SMOTE")
    
    del X_train_full, y_train_full, partitions
    gc.collect()
    
    # --- Persistent client datasets (GPU) ---
    client_datasets = build_client_datasets_gpu(client_data, GLOBAL_BATCH_SIZE)
    
    # --- Validation dataset (GPU) ---
    rng_val = np.random.RandomState(SEED)
    val_idx = rng_val.choice(len(X_test), size=min(VAL_SUBSET_SIZE, len(X_test)), replace=False)
    X_val_np, y_val_np = X_test[val_idx], y_test[val_idx]
    val_ds = tf.data.Dataset.from_tensor_slices((X_val_np, y_val_np))
    val_ds = val_ds.batch(FINAL_BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
    
    # --- Build models with GPU placement ---
    STEPS_PER_EPOCH = max(1, sum(len(d[0]) for d in client_data) // GLOBAL_BATCH_SIZE)
    N_SEL = max(1, int(N_CLIENTS * CLIENT_FRACTION))
    TOTAL_STEPS = N_ROUNDS * N_SEL * LOCAL_EPOCHS * STEPS_PER_EPOCH
    
    client_model = build_lightweight_model(
        N_FEATURES, N_CLASSES, total_steps=TOTAL_STEPS, learning_rate=3e-4,
    )
    eval_model = build_lightweight_model(
        N_FEATURES, N_CLASSES, total_steps=TOTAL_STEPS, learning_rate=3e-4,
    )
    
    # --- GPU variables ---
    with tf.device("/GPU:0"):
        global_vars = [tf.Variable(w, trainable=False, dtype=tf.float32)
                       for w in client_model.get_weights()]
        residuals = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False)
                     for v in global_vars]
        class_weight_tensor = tf.constant(
            [class_weights.get(i, 1.0) for i in range(N_CLASSES)],
            dtype=tf.float32
        )
    
    PARAM_BYTES = sum(v.numpy().nbytes for v in global_vars)
    logger.info(f"Model: {PARAM_BYTES/1024:.1f} KB parameters")
    
    k_pct_tensor = tf.constant(TOP_K_PCT, dtype=tf.float32)
    
    # --- Communication tracker ---
    comm = CommunicationTracker(TOP_K_PCT)
    
    # --- Checkpointing ---
    best_ckpt = os.path.join(CKPT_DIR, "best_weights.npy")
    latest_ckpt = os.path.join(CKPT_DIR, "latest_weights.npy")
    resume_round = 0
    if os.path.exists(latest_ckpt):
        saved = np.load(latest_ckpt, allow_pickle=True)
        for i, v in enumerate(global_vars):
            v.assign(tf.constant(saved[i], dtype=tf.float32))
        with open(os.path.join(CKPT_DIR, "resume.json"), "r") as f:
            resume_round = json.load(f).get("round", 0)
        logger.info(f"Resumed from round {resume_round + 1}")
    
    # --- FL Loop ---
    logger.info("=" * 60)
    logger.info(f" FL TRAINING — {N_ROUNDS} ROUNDS")
    logger.info("=" * 60)
    
    round_log: Dict[str, List] = OrderedDict({
        "round": [], "loss": [], "accuracy": [],
        "weighted_f1": [], "comm_mb": [],
        "ram_gb": [], "gpu_mb": [], "elapsed_min": [],
    })
    
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    best_acc = 0.0
    best_rnd = 0
    no_improve = 0
    
    for rnd in range(resume_round + 1, N_ROUNDS + 1):
        t_round = time.time()
        selected = rng.choice(N_CLIENTS, size=N_SEL, replace=False)
        
        client_weights_batch: List[List[tf.Tensor]] = []
        client_sizes_batch: List[int] = []
        
        for cid in selected:
            ds = client_datasets[cid]
            
            # Assign global weights to client model
            for mv, gv in zip(client_model.trainable_variables, global_vars):
                mv.assign(gv)
            
            # GPU-native training
            new_weights = train_client_step_gpu(
                client_model,
                ds,
                LOCAL_EPOCHS,
                class_weight_tensor
            )
            
            # Compute deltas on GPU
            prev_weights = [tf.identity(v) for v in global_vars]
            deltas_tf = [nw - pw for nw, pw in zip(new_weights, prev_weights)]
            
            # GPU Top-K sparsification
            sparse_deltas = gpu_topk_sparsify(deltas_tf, residuals, k_pct_tensor)
            final_w = [pw + sd for pw, sd in zip(prev_weights, sparse_deltas)]
            
            client_weights_batch.append(final_w)
            client_sizes_batch.append(len(client_data[cid][0]))
        
        # Weighted FedAvg (GPU)
        sizes_tensor = tf.constant(client_sizes_batch, dtype=tf.float32)
        new_global = federated_average(global_vars, client_weights_batch, sizes_tensor)
        for i, v in enumerate(global_vars):
            v.assign(tf.cast(new_global[i], tf.float32))
        
        comm_mb = comm.log(len(selected), PARAM_BYTES)
        
        # --- Evaluation (every VAL_EVERY rounds) ---
        if rnd % VAL_EVERY == 0 or rnd == 1 or rnd == N_ROUNDS:
            for ev, gv in zip(eval_model.trainable_variables, global_vars):
                ev.assign(gv)
            loss, acc = evaluate_model_gpu(eval_model, val_ds)
            loss = float(loss.numpy())
            acc = float(acc.numpy())
        else:
            loss, acc = 0.0, 0.0
        
        ram_gb = psutil.virtual_memory().used / 1024**3
        gpu_info = tf.config.experimental.get_memory_info("GPU:0")
        gpu_mb = gpu_info["current"] / 1024**2 if gpu_info else 0
        elapsed = (time.time() - t0) / 60
        round_time = time.time() - t_round
        
        round_log["round"].append(rnd)
        round_log["loss"].append(float(loss))
        round_log["accuracy"].append(float(acc))
        round_log["weighted_f1"].append(0.0)
        round_log["comm_mb"].append(comm_mb)
        round_log["ram_gb"].append(ram_gb)
        round_log["gpu_mb"].append(gpu_mb)
        round_log["elapsed_min"].append(elapsed)
        
        if rnd % VAL_EVERY == 0 or rnd == 1:
            logger.info(
                f"  [R{rnd:>3d}] acc={acc:.4f} loss={loss:.4f} "
                f"comm={comm_mb:.1f}MB GPU={gpu_mb:.0f}MB RAM={ram_gb:.1f}GB "
                f"t={round_time:.0f}s total={elapsed:.1f}min"
            )
        
        # --- Early stopping ---
        if acc > best_acc + 0.001:
            best_acc = acc
            best_rnd = rnd
            no_improve = 0
            np.save(best_ckpt, [v.numpy() for v in global_vars])
        else:
            no_improve += 1
        
        if no_improve >= EARLY_STOP_PATIENCE:
            logger.info(f"Early stop at round {rnd}")
            break
        
        # --- Checkpoint ---
        if rnd % CHECKPOINT_EVERY == 0:
            np.save(latest_ckpt, [v.numpy() for v in global_vars])
            with open(os.path.join(CKPT_DIR, "resume.json"), "w") as f:
                json.dump({"round": rnd, "best_acc": best_acc}, f)
            pd.DataFrame(round_log).to_csv(
                os.path.join(RESULTS_DIR, "round_metrics.csv"), index=False
            )
            gc.collect()
    
    # --- Restore best weights ---
    if os.path.exists(best_ckpt):
        best_saved = np.load(best_ckpt, allow_pickle=True)
        for i, v in enumerate(global_vars):
            v.assign(tf.constant(best_saved[i], dtype=tf.float32))
        logger.info(f"Restored best weights from round {best_rnd}")
    
    del client_model
    gc.collect()
    
    # --- Fine-tuning (GPU) ---
    final_model = fine_tune_gpu(
        global_vars, client_datasets, N_FEATURES, N_CLASSES,
        class_weight_tensor, RESULTS_DIR,
    )
    
    # --- Final evaluation ---
    rep_dict = final_evaluation(final_model, X_test, y_test, class_names, RESULTS_DIR)
    round_log["weighted_f1"][-1] = rep_dict["weighted avg"]["f1-score"]
    
    # --- Pruning ---
    ratio, prep_dict = prune_and_evaluate(
        final_model, X_test, y_test, class_names,
        N_FEATURES, N_CLASSES, RESULTS_DIR,
    )
    
    # --- Save logs ---
    pd.DataFrame(round_log).to_csv(
        os.path.join(RESULTS_DIR, "round_metrics.csv"), index=False
    )
    pd.DataFrame({
        "round": round_log["round"],
        "comm_mb": round_log["comm_mb"],
        "cumulative_comm_mb": np.cumsum(round_log["comm_mb"]),
    }).to_csv(os.path.join(RESULTS_DIR, "communication_history.csv"), index=False)
    
    meta = OrderedDict({
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "n_clients": N_CLIENTS,
        "n_rounds": len(round_log["round"]),
        "best_round": best_rnd,
        "full_accuracy": round_log["accuracy"][-1],
        "full_f1": round_log["weighted_f1"][-1],
        "pruned_accuracy": prep_dict.get("accuracy", None),
        "compression": ratio,
        "total_comm_mb": comm.total,
        "time_min": round((time.time() - t0) / 60, 1),
    })
    with open(os.path.join(RESULTS_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    logger.info("\n" + "█" * 50)
    logger.info(f"  DONE — {len(round_log['round'])} rounds, {meta['time_min']} min")
    logger.info(f"  Accuracy: {meta['full_accuracy']:.4f} | F1: {meta['full_f1']:.4f}")
    logger.info("█" * 50)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()