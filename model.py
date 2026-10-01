"""Model 4 — 4-layer ConvLSTM for Remaining Useful Life (RUL) wear prognostics.

Extends the original scaffold (2-layer ConvLSTM) to a 4-layer stack:

    ConvLSTM2D(32) -> ConvLSTM2D(48) -> ConvLSTM2D(48) -> ConvLSTM2D(32)
        -> Flatten -> Dense(64) -> Dense(1) [RUL]

Pipeline
--------
1. Synthesize a continuous wear trajectory (accelerating degradation +
   vibration-like noise) rendered as a sequence of 2-D wear maps.
2. Cut into sliding time-windows (10 frames) with RUL targets, split
   chronologically (no leakage between train/val/test).
3. Train the 4-layer ConvLSTM with early stopping.
4. Report RMSE / MAE / R^2 on held-out test data, in both normalized
   RUL units and original cycle units; persist metrics + best weights.

Run:
    python model.py
"""

import json
import os

# Deterministic CPU training: oneDNN's custom kernels use data-dependent
# reduction order; disabling them (before TF import) makes runs reproducible.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    BatchNormalization,
    ConvLSTM2D,
    Dense,
    Flatten,
)
from tensorflow.keras.models import Model

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SEED = 42
N_CYCLES = 3200            # total life of the synthetic bearing run
WIN_LEN = 10               # temporal frames per sample
SPATIAL = 16               # 2-D wear-map resolution (16x16)
STRIDE = 2                 # sampling stride along the time axis
VAL_FRACTION = 0.15
TEST_FRACTION = 0.20
EPOCHS = 14
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
RESULTS_DIR = "results"

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ----------------------------------------------------------------------------
# Synthetic wear data
# ----------------------------------------------------------------------------
def _wear_maps(n_cycles: int, spatial: int) -> np.ndarray:
    """Render the wear process as an array of `spatial x spatial` maps.

    severity(t) accelerates with time (running-in -> steady -> fatigue);
    three fixed wear-blob centers brighten as damage accumulates, on top of
    a fixed texture field plus high-frequency noise (vibration surrogate).
    """
    rng = np.random.default_rng(SEED)
    texture = rng.standard_normal((spatial, spatial))
    texture = (texture - texture.mean()) / texture.std()

    centers = rng.random((3, 2)) * spatial
    blob = np.zeros((spatial, spatial), dtype=np.float32)
    for cy, cx in centers:
        yy, xx = np.mgrid[0:spatial, 0:spatial]
        blob += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.5**2))
    blob = (blob - blob.mean()) / (blob.std() + 1e-8)

    t = np.arange(n_cycles)
    severity = (t / n_cycles) ** 1.6          # accelerating degradation
    frames = np.zeros((n_cycles, spatial, spatial, 1), dtype=np.float32)
    for i in range(n_cycles):
        noise = rng.standard_normal((spatial, spatial, 1)) * 0.15
        frames[i] = (
            0.5 * texture[..., None]
            + severity[i] * blob[..., None]
            + noise
        )
    return frames


def generate_dataset():
    """Sliding-window RUL dataset with a chronological split."""
    frames = _wear_maps(N_CYCLES, SPATIAL)
    starts = np.arange(0, N_CYCLES - WIN_LEN + 1, STRIDE)
    X = np.stack([frames[s : s + WIN_LEN] for s in starts])
    rul = (N_CYCLES - starts - WIN_LEN) / (N_CYCLES - WIN_LEN)  # normalized RUL in (0, 1]

    n = len(starts)
    n_test = int(n * TEST_FRACTION)
    n_val = int(n * VAL_FRACTION)
    tr, va, te = slice(0, n - n_val - n_test), slice(n - n_val - n_test, n - n_test), slice(n - n_test, n)

    # Standardize input features with training-set statistics only.
    mu, sd = X[tr].mean(), X[tr].std() + 1e-8
    X = (X - mu) / sd
    return X[tr], X[va], X[te], rul[tr], rul[va], rul[te]


# ----------------------------------------------------------------------------
# Model 4 architecture
# ----------------------------------------------------------------------------
def build_model_4(input_shape=(WIN_LEN, SPATIAL, SPATIAL, 1)):
    x = Input(shape=input_shape)
    h = ConvLSTM2D(32, kernel_size=(3, 3), return_sequences=True,
                   activation="tanh", padding="same")(x)
    h = BatchNormalization()(h)
    h = ConvLSTM2D(48, kernel_size=(3, 3), return_sequences=True,
                   activation="tanh", padding="same")(h)
    h = BatchNormalization()(h)
    h = ConvLSTM2D(48, kernel_size=(3, 3), return_sequences=True,
                   activation="tanh", padding="same")(h)
    h = BatchNormalization()(h)
    h = ConvLSTM2D(32, kernel_size=(3, 3), return_sequences=False,
                   activation="tanh", padding="same")(h)
    h = BatchNormalization()(h)
    h = Flatten()(h)
    h = Dense(64, activation="relu")(h)
    out = Dense(1, activation="linear")(h)   # normalized RUL
    model = Model(inputs=x, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    model = Model(inputs=x, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="mse", metrics=["mae"])
    return model
# Platform alias: the wt-pm adapter (m02-convlstm-wear) imports the factory by this name.
build_convlstm = build_model_4

    
  

# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def _r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main():
    X_tr, X_va, X_te, y_tr, y_va, y_te = generate_dataset()
    print(f"[data] train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

    model = build_model_4()
    model.summary(print_fn=print)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    # --- Test evaluation (normalized + cycle units) -------------------------
    def report(X, y, tag):
        pred = model.predict(X, verbose=0, batch_size=BATCH_SIZE).ravel()
        y_cycles = y * (N_CYCLES - WIN_LEN)
        pred_cycles = pred * (N_CYCLES - WIN_LEN)
        rmse_n = float(np.sqrt(np.mean((y - pred) ** 2)))
        return {
            "split": tag,
            "rmse_norm": round(rmse_n, 4),
            "rmse_cycles": round(rmse_n * (N_CYCLES - WIN_LEN), 1),
            "mae_cycles": round(float(np.mean(np.abs(y_cycles - pred_cycles))), 1),
            "r2": round(_r2(y, pred), 4),
        }

    rows = [report(X_te, y_te, "test"), report(X_va, y_va, "val"), report(X_tr, y_tr, "train")]
    table = pd.DataFrame(rows)
    print("\n=== Model 4 — evaluation ===")
    print(table.to_string(index=False))

    # --- Persist results -----------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model.save_weights(f"{RESULTS_DIR}/model4_weights.weights.h5")
    metrics = {
        "model": "model-4-convlstm",
        "input_shape": [WIN_LEN, SPATIAL, SPATIAL, 1],
        "epochs_completed": len(history.history["loss"]),
        "params": int(model.count_params()),
        "metrics": rows,
    }
    with open(f"{RESULTS_DIR}/model4_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved best weights -> {RESULTS_DIR}/model4_weights.weights.h5")
    print(f"Saved metrics      -> {RESULTS_DIR}/model4_metrics.json")


if __name__ == "__main__":
    main()
