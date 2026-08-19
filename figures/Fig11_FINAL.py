# ============================================================
# FIGURE 11 FINAL - REAL BER vs Eb/N0
# Conventional OFDM vs FCAE/OFDM-X vs Random Phase Rotation
#
# Finalized for Reviewer #3.
#
# Source basis:
# - FCAE V8 architecture/loss from the established V8 program.
# - Channel operating points aligned with Table III:
#     Vehicular: 200 Hz, 1 path
#     LEO:       1000 Hz, 3 paths
#     ISAC:       450 Hz, 7 paths
# - Nsub=64, CP=16, Delta-f=15 kHz, Fs=960 kHz
#
# Important:
# - FCAE V8 has 148,417 trainable parameters:
#   148,416 dense parameters + 1 learned peak-gain parameter.
# - RPR uses 4 random frequency-domain phase candidates and
#   selects the minimum-PAPR candidate per OFDM symbol.
# - Receiver removes the selected phase sequence (side information)
#   before LS channel estimation. This is the standard idealized
#   receiver assumption for this baseline; side-information overhead
#   is not included in BER.
#
# This program writes:
#   Figure11_FINAL_REAL_BER_3CURVES.png
#   Figure11_FINAL_REAL_BER_RESULTS.csv
#   Figure11_FINAL_REAL_BER_SUMMARY.txt
#
# Run in an environment with TensorFlow 2.20.x (e.g. Colab).
# ============================================================

import os
import random
import csv
import time
from pathlib import Path

SEED = 20260808
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import layers

# ============================================================
# 1. FINAL SYSTEM PARAMETERS
# ============================================================
N = 64
CP = 16
DELTA_F = 15e3
FS = N * DELTA_F
EBN0_DB = np.arange(0, 21, 2)

# Use the same sample count requested/used for the final BER curves.
NUM_BER_SYMBOLS = 4000

# FCAE V8 training parameters (established V8 configuration)
N_TRAIN = 24000
N_VAL = 6000
BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 2e-4

MAX_RESIDUAL = 0.18
PEAK_THRESHOLD = 1.55
TARGET_REDUCTION_DB = 0.20

W_PAPR = 9.0
W_PEAK = 11.0
W_RECON = 0.05
W_POWER = 0.34
W_FREQ = 0.34
W_EVM = 0.22
W_RESIDUAL = 1e-5

# Channel operating points aligned with Table III
CHANNELS = {
    "Vehicular": {"fd": 200.0, "paths": 1, "type": "rayleigh"},
    "LEO":       {"fd": 1000.0, "paths": 3, "type": "rician"},
    "ISAC":      {"fd": 450.0, "paths": 7, "type": "rayleigh"},
}
K_FACTOR_LEO = 8.0

# RPR/SLM-like baseline
RPR_CANDIDATES = 4

# ============================================================
# 2. DATASET / FCAE V8
# ============================================================
def make_fcae_dataset(n_symbols, seed_offset):
    rng = np.random.default_rng(SEED + seed_offset)
    bits = rng.integers(0, 2, size=(n_symbols, N)).astype(np.int8)
    symbols = (2.0 * bits - 1.0).astype(np.complex64)
    x = np.fft.ifft(symbols, axis=1).astype(np.complex64)

    p = np.mean(np.abs(x) ** 2, axis=1, keepdims=True)
    x = x / np.sqrt(p + 1e-12)

    y = np.concatenate(
        [np.real(x), np.imag(x)], axis=1
    ).astype(np.float32)

    return y, bits, x


class PeakAwareResidual(layers.Layer):
    def __init__(
        self,
        max_residual=MAX_RESIDUAL,
        peak_threshold=PEAK_THRESHOLD,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_residual = float(max_residual)
        self.peak_threshold = float(peak_threshold)

    def build(self, input_shape):
        self.gain = self.add_weight(
            name="learned_peak_gain",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.50),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        real = inputs[:, :N]
        imag = inputs[:, N:]

        magnitude = tf.sqrt(
            tf.square(real) + tf.square(imag) + 1e-12
        )
        rms = tf.sqrt(
            tf.reduce_mean(tf.square(magnitude), axis=1, keepdims=True)
            + 1e-12
        )

        excess = magnitude - self.peak_threshold * rms
        soft_peak = tf.nn.softplus(12.0 * excess) / 12.0

        inv_mag = 1.0 / (magnitude + 1e-8)
        dir_real = real * inv_mag
        dir_imag = imag * inv_mag

        gain = tf.nn.sigmoid(self.gain)

        correction_real = gain * soft_peak * dir_real
        correction_imag = gain * soft_peak * dir_imag

        shaped_real = real - self.max_residual * correction_real
        shaped_imag = imag - self.max_residual * correction_imag

        return tf.concat([shaped_real, shaped_imag], axis=1)


def build_fcae_v8():
    inp = tf.keras.Input(shape=(2 * N,), name="OFDM_input")

    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(128, activation="relu")(x)
    latent = layers.Dense(64, activation="tanh", name="latent")(x)

    x = layers.Dense(128, activation="relu")(latent)
    x = layers.Dense(256, activation="relu")(x)

    residual = layers.Dense(
        2 * N, activation="tanh", name="residual_proposal"
    )(x)
    residual = MAX_RESIDUAL * residual

    proposal = inp + residual
    out = PeakAwareResidual(name="peak_aware_shaper")(proposal)

    return tf.keras.Model(inp, out, name="FCAE_OFDM_X_V8")


def fcae_loss(y_true, y_pred):
    s_true = tf.complex(y_true[:, :N], y_true[:, N:])
    s_pred = tf.complex(y_pred[:, :N], y_pred[:, N:])

    p_true = tf.abs(s_true) ** 2
    p_pred = tf.abs(s_pred) ** 2

    def papr_db(p):
        return (
            10.0
            * tf.math.log(
                tf.reduce_max(p, axis=1)
                / (tf.reduce_mean(p, axis=1) + 1e-12)
                + 1e-12
            )
            / tf.math.log(tf.constant(10.0, tf.float32))
        )

    papr_true = papr_db(p_true)
    papr_pred = papr_db(p_pred)

    papr_loss = tf.reduce_mean(
        tf.square(
            papr_pred
            - tf.stop_gradient(tf.minimum(papr_true, papr_pred))
        )
    )

    target = papr_true - TARGET_REDUCTION_DB
    target_hinge = tf.reduce_mean(
        tf.square(tf.nn.relu(papr_pred - target))
    )

    rms_true = tf.sqrt(
        tf.reduce_mean(p_true, axis=1, keepdims=True) + 1e-12
    )
    threshold = PEAK_THRESHOLD * rms_true
    excess = tf.nn.relu(tf.abs(s_pred) - threshold)
    peak_loss = tf.reduce_mean(tf.square(excess))

    recon_loss = tf.reduce_mean(tf.square(y_pred - y_true))

    mean_true = tf.reduce_mean(p_true, axis=1)
    mean_pred = tf.reduce_mean(p_pred, axis=1)
    power_loss = tf.reduce_mean(tf.square(mean_pred - mean_true))

    F_true = tf.signal.fft(tf.cast(s_true, tf.complex64))
    F_pred = tf.signal.fft(tf.cast(s_pred, tf.complex64))
    freq_loss = (
        tf.reduce_mean(tf.abs(F_pred - F_true) ** 2)
        / (tf.reduce_mean(tf.abs(F_true) ** 2) + 1e-12)
    )

    evm_loss = (
        tf.reduce_mean(tf.abs(s_pred - s_true) ** 2)
        / (tf.reduce_mean(tf.abs(s_true) ** 2) + 1e-12)
    )

    residual_loss = tf.reduce_mean(tf.square(y_pred - y_true))

    return (
        W_PAPR * (papr_loss + target_hinge)
        + W_PEAK * peak_loss
        + W_RECON * recon_loss
        + W_POWER * power_loss
        + W_FREQ * freq_loss
        + W_EVM * evm_loss
        + W_RESIDUAL * residual_loss
    )


def train_fcae():
    train_x, _, _ = make_fcae_dataset(N_TRAIN, 1)
    val_x, _, _ = make_fcae_dataset(N_VAL, 2)

    model = build_fcae_v8()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=fcae_loss,
    )

    if model.count_params() != 148417:
        raise RuntimeError(
            f"FCAE V8 parameter check failed: {model.count_params()} "
            f"(expected 148417)."
        )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("\nTraining FCAE V8...")
    model.fit(
        train_x,
        train_x,
        validation_data=(val_x, val_x),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    print(f"FCAE V8 parameters: {model.count_params()}")
    return model


# ============================================================
# 3. OFDM
# ============================================================
def bpsk_mod(bits):
    return (2 * bits - 1).astype(np.complex128)


def bpsk_demod(x):
    return (np.real(x) >= 0).astype(np.int8)


def ofdm_modulate(X):
    x = np.fft.ifft(X, axis=1)
    cp = x[:, -CP:]
    return np.concatenate([cp, x], axis=1)


def ofdm_demodulate(rx):
    return np.fft.fft(rx[:, CP:], axis=1)


# ============================================================
# 4. Time-varying multipath channels
# ============================================================
def _path_channel(
    tx,
    ebn0_db,
    fd,
    paths,
    rng,
    rician=False,
):
    ns, total_len = tx.shape
    delays = np.arange(paths)

    # Path gains
    if rician:
        los = np.sqrt(K_FACTOR_LEO / (K_FACTOR_LEO + 1.0))
        nlos = np.sqrt(1.0 / (K_FACTOR_LEO + 1.0))

        gains = (
            rng.standard_normal((ns, paths))
            + 1j * rng.standard_normal((ns, paths))
        ) / np.sqrt(2.0 * paths)

        gains *= nlos
        gains[:, 0] += los
    else:
        gains = (
            rng.standard_normal((ns, paths))
            + 1j * rng.standard_normal((ns, paths))
        ) / np.sqrt(2.0 * paths)

    fd_paths = rng.uniform(-fd, fd, size=paths)
    t = np.arange(total_len) / FS

    rx_clean = np.zeros_like(tx, dtype=np.complex128)
    h_diag = np.zeros((ns, N), dtype=np.complex128)

    for s in range(ns):
        h_time = np.zeros((total_len, paths), dtype=np.complex128)

        for p in range(paths):
            doppler = np.exp(1j * 2.0 * np.pi * fd_paths[p] * t)
            h_time[:, p] = gains[s, p] * doppler

            d = int(delays[p])
            if d == 0:
                rx_clean[s] += h_time[:, p] * tx[s]
            else:
                rx_clean[s, d:] += (
                    h_time[d:, p] * tx[s, :-d]
                )

        # Diagonal frequency response used by the one-tap receiver.
        n_use = np.arange(CP, CP + N)
        phase = np.exp(
            -1j * 2.0 * np.pi
            * np.arange(N)[:, None]
            * delays[None, :] / N
        )

        diagonal = np.einsum(
            "np,kp->nk",
            h_time[n_use, :],
            phase,
        )
        h_diag[s] = np.mean(diagonal, axis=0)

    signal_power = np.mean(np.abs(rx_clean) ** 2)
    noise_power = signal_power / (10.0 ** (ebn0_db / 10.0))
    noise = (
        rng.standard_normal(rx_clean.shape)
        + 1j * rng.standard_normal(rx_clean.shape)
    ) * np.sqrt(noise_power / 2.0)

    return rx_clean + noise, h_diag


def apply_channel(tx, ebn0_db, channel_name, rng):
    cfg = CHANNELS[channel_name]
    return _path_channel(
        tx,
        ebn0_db,
        cfg["fd"],
        cfg["paths"],
        rng,
        rician=(cfg["type"] == "rician"),
    )


# ============================================================
# 5. Receiver
# ============================================================
def ls_channel_estimation(Y):
    # The pilot positions are every fourth subcarrier.
    pilot_index = np.arange(0, N, 4)
    Hp = Y[:, pilot_index]  # pilots are +1 before phase removal
    H = np.zeros_like(Y, dtype=np.complex128)

    x = np.arange(N)
    for s in range(Y.shape[0]):
        H[s] = (
            np.interp(x, pilot_index, Hp[s].real)
            + 1j * np.interp(x, pilot_index, Hp[s].imag)
        )
    return H


def receiver(rx, phase_vectors=None):
    Y = ofdm_demodulate(rx)

    # Remove known RPR phase sequence before channel estimation.
    if phase_vectors is not None:
        Y = Y * np.conj(phase_vectors)

    H = ls_channel_estimation(Y)
    Z = Y / (H + 1e-12)

    pilot_index = np.arange(0, N, 4)
    data_index = np.setdiff1d(np.arange(N), pilot_index)
    return bpsk_demod(Z[:, data_index])


# ============================================================
# 6. RPR baseline
# ============================================================
def rpr_select(X, rng):
    """
    Four random frequency-domain phase candidates.
    Select the minimum-PAPR candidate independently per OFDM symbol.
    """
    ns = X.shape[0]

    best_papr = np.full(ns, np.inf)
    best_X = np.zeros_like(X, dtype=np.complex128)
    best_phase = np.ones_like(X, dtype=np.complex128)

    for _ in range(RPR_CANDIDATES):
        phase = np.exp(
            1j * rng.uniform(0, 2*np.pi, size=N)
        )
        Xc = X * phase[None, :]
        xc = np.fft.ifft(Xc, axis=1)

        power = np.abs(xc) ** 2
        papr = np.max(power, axis=1) / (
            np.mean(power, axis=1) + 1e-12
        )

        mask = papr < best_papr
        if np.any(mask):
            best_papr[mask] = papr[mask]
            best_X[mask] = Xc[mask]
            best_phase[mask] = phase[None, :]

    return best_X, best_phase


# ============================================================
# 7. BER simulation
# ============================================================
def make_bits(rng):
    return rng.integers(
        0, 2, size=(NUM_BER_SYMBOLS, N)
    ).astype(np.int8)


def conventional_ofdm(bits):
    X = bpsk_mod(bits)
    pilot_index = np.arange(0, N, 4)
    X[:, pilot_index] = 1.0 + 0j
    return ofdm_modulate(X)


def fcae_ofdm(model, bits):
    X = bpsk_mod(bits)
    pilot_index = np.arange(0, N, 4)
    X[:, pilot_index] = 1.0 + 0j

    x = np.fft.ifft(X, axis=1).astype(np.complex64)
    inp = np.concatenate(
        [np.real(x), np.imag(x)], axis=1
    ).astype(np.float32)

    y = model.predict(inp, batch_size=512, verbose=0)

    shaped = (
        y[:, :N]
        + 1j * y[:, N:]
    ).astype(np.complex128)

    cp = shaped[:, -CP:]
    return np.concatenate([cp, shaped], axis=1)


def rpr_ofdm(bits, rng):
    X = bpsk_mod(bits)
    pilot_index = np.arange(0, N, 4)
    X[:, pilot_index] = 1.0 + 0j

    X_rpr, phase = rpr_select(X, rng)
    return ofdm_modulate(X_rpr), phase


def ber_from_bits(bits, detected):
    return float(
        np.mean(bits[:, np.setdiff1d(np.arange(N), np.arange(0, N, 4))]
                 != detected)
    )


def simulate_point(model, channel_name, ebn0_db, base_seed):
    # Same information bits and channel realization seed for all methods.
    bit_rng = np.random.default_rng(base_seed)
    bits = make_bits(bit_rng)

    # Independent reproducible RNG streams, with identical channel seeds
    # for each waveform family.
    rng_ofdm = np.random.default_rng(base_seed + 10)
    rng_fcae = np.random.default_rng(base_seed + 10)
    rng_rpr = np.random.default_rng(base_seed + 10)

    # OFDM
    tx_ofdm = conventional_ofdm(bits)
    rx_ofdm, _ = apply_channel(
        tx_ofdm, ebn0_db, channel_name, rng_ofdm
    )
    det_ofdm = receiver(rx_ofdm)
    ber_ofdm = ber_from_bits(bits, det_ofdm)

    # FCAE / OFDM-X
    tx_fcae = fcae_ofdm(model, bits)
    rx_fcae, _ = apply_channel(
        tx_fcae, ebn0_db, channel_name, rng_fcae
    )
    det_fcae = receiver(rx_fcae)
    ber_fcae = ber_from_bits(bits, det_fcae)

    # RPR
    tx_rpr, phase = rpr_ofdm(bits, rng_rpr)
    rx_rpr, _ = apply_channel(
        tx_rpr, ebn0_db, channel_name, rng_rpr
    )
    det_rpr = receiver(rx_rpr, phase_vectors=phase)
    ber_rpr = ber_from_bits(bits, det_rpr)

    return ber_ofdm, ber_fcae, ber_rpr


# ============================================================
# 8. Main
# ============================================================
if __name__ == "__main__":
    print("=" * 78)
    print("FIGURE 11 FINAL REAL BER EVALUATION")
    print("=" * 78)
    print(f"Nsub                 : {N}")
    print(f"CP length             : {CP}")
    print(f"Subcarrier spacing    : {DELTA_F/1e3:.1f} kHz")
    print(f"Sampling frequency    : {FS/1e3:.1f} kHz")
    print(f"Eb/N0                 : {EBN0_DB[0]} to {EBN0_DB[-1]} dB")
    print(f"BER symbols           : {NUM_BER_SYMBOLS}")
    print("Vehicular             : 200 Hz | 1 path")
    print("LEO                   : 1000 Hz | 3 paths | Rician K=8")
    print("ISAC                  : 450 Hz | 7 paths")
    print(f"RPR candidates        : {RPR_CANDIDATES}")
    print("=" * 78)

    t0 = time.time()
    model = train_fcae()

    all_results = []

    for ch_i, channel_name in enumerate(CHANNELS):
        print("\n" + "=" * 78)
        print(f"CHANNEL: {channel_name}")
        print("=" * 78)

        for eb_i, eb in enumerate(EBN0_DB):
            seed = SEED + 10000 * (ch_i + 1) + int(eb) * 100

            b_ofdm, b_fcae, b_rpr = simulate_point(
                model, channel_name, int(eb), seed
            )

            row = {
                "Channel": channel_name,
                "EbN0_dB": int(eb),
                "OFDM_BER": b_ofdm,
                "FCAE_BER": b_fcae,
                "RPR_BER": b_rpr,
                "FCAE_minus_OFDM": b_fcae - b_ofdm,
                "FCAE_relative_change_pct": (
                    100.0 * (b_fcae - b_ofdm) / b_ofdm
                    if b_ofdm > 0 else np.nan
                ),
            }
            all_results.append(row)

            print(
                f"Eb/N0={int(eb):2d} dB | "
                f"OFDM={b_ofdm:.6e} | "
                f"FCAE={b_fcae:.6e} | "
                f"RPR={b_rpr:.6e} | "
                f"Delta(FCAE-OFDM)={b_fcae-b_ofdm:+.6e}"
            )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------
    out_csv = "Figure11_FINAL_REAL_BER_RESULTS.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_results[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_results)

    # --------------------------------------------------------
    # Plot: 3 panels, one per requested channel
    # --------------------------------------------------------
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 8

    fig, axes = plt.subplots(
        1, 3,
        figsize=(7.2, 2.45),
        dpi=600,
        sharey=True
    )

    for ax, channel_name in zip(axes, CHANNELS):
        rows = [
            r for r in all_results
            if r["Channel"] == channel_name
        ]

        x = np.array([r["EbN0_dB"] for r in rows])
        y_ofdm = np.array([r["OFDM_BER"] for r in rows])
        y_fcae = np.array([r["FCAE_BER"] for r in rows])
        y_rpr = np.array([r["RPR_BER"] for r in rows])

        ax.semilogy(x, y_ofdm, "o-", linewidth=1.0,
                    markersize=3.0, label="Conventional OFDM")
        ax.semilogy(x, y_fcae, "s-", linewidth=1.0,
                    markersize=3.0, label="OFDM-X/FCAE")
        ax.semilogy(x, y_rpr, "^-", linewidth=1.0,
                    markersize=3.0, label="Random Phase Rotation")

        ax.set_title(channel_name, fontsize=8, fontweight="bold")
        ax.set_xlabel(r"$E_b/N_0$ (dB)", fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.45)

    axes[0].set_ylabel("BER", fontsize=8)
    axes[0].set_xticks(EBN0_DB)

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=8,
        frameon=True
    )

    fig.tight_layout(rect=[0, 0.12, 1, 1])

    out_png = "Figure11_FINAL_REAL_BER_3CURVES.png"
    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)

    # --------------------------------------------------------
    # Summary: BER degradation at 14 dB
    # --------------------------------------------------------
    summary_lines = [
        "FIGURE 11 FINAL REAL BER EVALUATION",
        "=" * 70,
        f"FCAE parameter count: {model.count_params()}",
        f"Nsub={N}, CP={CP}, Delta-f={DELTA_F/1e3:.1f} kHz, Fs={FS/1e3:.1f} kHz",
        f"BER symbols per point={NUM_BER_SYMBOLS}",
        "Channel settings: Vehicular 200 Hz/1 path; "
        "LEO 1000 Hz/3 paths; ISAC 450 Hz/7 paths.",
        "",
        "BER at Eb/N0 = 14 dB",
        "-" * 70,
    ]

    for ch in CHANNELS:
        r = next(
            x for x in all_results
            if x["Channel"] == ch and x["EbN0_dB"] == 14
        )
        summary_lines.append(
            f"{ch}: OFDM={r['OFDM_BER']:.8e}, "
            f"FCAE={r['FCAE_BER']:.8e}, "
            f"RPR={r['RPR_BER']:.8e}, "
            f"Delta(FCAE-OFDM)={r['FCAE_minus_OFDM']:+.8e}"
        )

    summary_lines += [
        "",
        "Files:",
        out_png,
        out_csv,
        f"Runtime: {time.time()-t0:.2f} s",
    ]

    out_txt = "Figure11_FINAL_REAL_BER_SUMMARY.txt"
    Path(out_txt).write_text(
        "\n".join(summary_lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("FIGURE 11 FINAL EVALUATION COMPLETE")
    print("=" * 78)
    print("Saved:", out_png)
    print("Saved:", out_csv)
    print("Saved:", out_txt)
    print(f"FCAE parameters: {model.count_params()}")
    print("=" * 78)
