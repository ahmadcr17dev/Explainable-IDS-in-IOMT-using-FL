%%writefile /kaggle/working/model_def.py
import os, logging
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass


def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=tf.math.log(y_pred)
        )
        pt = tf.exp(-ce)
        fl = alpha * (1 - pt)**gamma * ce
        return tf.reduce_mean(fl)
    return loss_fn


def build_lightweight_model(n_features, n_classes, 
                            learning_rate=3e-4, 
                            total_steps=10000,
                            weight_decay=1e-4,
                            clipnorm=1.0):
    inp = layers.Input(shape=(n_features,))

    x = layers.Dense(128, 
                     kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal')(inp)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Dense(64, 
                     kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Dense(32, 
                     kernel_regularizer=regularizers.l2(weight_decay),
                     kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.10)(x)
    
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out)
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        alpha=0.0
    )
    
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay,
        clipnorm=clipnorm
    )
    
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ]
    )
    return model


def apply_magnitude_pruning(model, sparsity=0.40, compile_after=True):
    weights = model.get_weights()
    pruned  = []
    total = active = 0
    for w in weights:
        if w.ndim >= 2:
            cutoff = np.percentile(np.abs(w), sparsity * 100)
            mask   = (np.abs(w) >= cutoff).astype(w.dtype)
            pruned.append(w * mask)
            total  += w.size
            active += int(mask.sum())
        else:
            pruned.append(w)
    model.set_weights(pruned)
    print(f"  Pruning done: {active}/{total} active "
          f"({100*active/total:.1f}% retained)")
    return model


def build_and_test_model():
    model = build_lightweight_model(n_features=30, n_classes=16, total_steps=10000)
    model.summary(print_fn=lambda s: print(s))
    x = np.zeros((1, 30), dtype=np.float32)
    y = model.predict(x, verbose=0)
    print(f"Model test output shape: {y.shape}")
    return model


if __name__ == '__main__':
    build_and_test_model()