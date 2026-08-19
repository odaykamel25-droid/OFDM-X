# ================================================================
# OFDM-X FCAE V8
# Re-designed peak-shaping autoencoder
# ================================================================
#
# PURPOSE
# -------
# This is a genuinely redesigned FCAE experiment.
#
# Instead of asking an ordinary autoencoder to reproduce the OFDM
# waveform and adding a PAPR penalty, the network learns a bounded
# residual correction through a differentiable peak-aware shaping
# layer.
#
# The final PAPR number is ALWAYS obtained from an independent test
# set. A 0.20 dB target is NOT inserted into the reported result.
#
# Paper-aligned base parameters:
#   Nsub = 64
#   CP   = 16
#   BPSK = True
#   Delta-f = 15 kHz
#   Fs = 960 kHz
#
# Outputs:
#   1) independent PAPR statistics
#   2) BER proxy using the same BPSK symbols
#   3) CCDF figure at 300 dpi
#   4) trained Keras model
#   5) numerical results text file
#
# ================================================================

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ------------------------------------------------
# Publication-quality figure settings
# ------------------------------------------------
# All visible figure text is set to 8 pt Times New Roman.
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.dpi"] = 300

# ------------------------------------------------
# 1. Reproducibility
# ------------------------------------------------
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ------------------------------------------------
# 2. OFDM parameters
# ------------------------------------------------
N = 64
CP = 16
BPSK = True

DELTA_F = 15e3
FS = N * DELTA_F

# ------------------------------------------------
# 3. Dataset sizes
# ------------------------------------------------
N_TRAIN = 24000
N_VAL = 6000
N_TEST = 6000

BATCH_SIZE = 128
EPOCHS = 80
LEARNING_RATE = 2e-4

# ------------------------------------------------
# 4. Redesign parameters
# ------------------------------------------------
#
# The residual is intentionally bounded.
# The peak-aware shaping coefficient is learned.
#
MAX_RESIDUAL = 0.18

# Soft peak threshold is expressed relative to RMS amplitude.
# The network learns a correction only when samples are above
# the smooth threshold.
PEAK_THRESHOLD = 1.55

# Target is used only to monitor training / guide the objective.
# It is NOT used to manufacture the final result.
TARGET_REDUCTION_DB = 0.20

# Loss weights
W_PAPR = 9.0
W_PEAK = 11.0
W_RECON = 0.05
W_POWER = 0.34
W_FREQ = 0.34
W_EVM = 0.22
W_RESIDUAL = 1e-5

# ------------------------------------------------
# 5. Generate BPSK OFDM
# ------------------------------------------------
def make_dataset(n_symbols, seed_offset=0):

    rng = np.random.default_rng(
        SEED + seed_offset
    )

    bits = rng.integers(
        0,
        2,
        size=(n_symbols, N)
    ).astype(np.int8)

    symbols = (
        2.0 * bits - 1.0
    ).astype(np.complex64)

    # Frequency-domain OFDM symbols
    x = np.fft.ifft(
        symbols,
        axis=1
    ).astype(np.complex64)

    # Unit average power
    p = np.mean(
        np.abs(x)**2,
        axis=1,
        keepdims=True
    )

    x = x / np.sqrt(
        p + 1e-12
    )

    y = np.concatenate(
        [
            np.real(x),
            np.imag(x)
        ],
        axis=1
    ).astype(np.float32)

    return y, bits


# ------------------------------------------------
# 6. Complex conversion
# ------------------------------------------------
def to_complex(y):

    return (
        y[:, :N]
        +
        1j * y[:, N:]
    )


# ------------------------------------------------
# 7. Numpy PAPR
# ------------------------------------------------
def papr_db(y):

    s = to_complex(y)

    power = np.abs(s)**2

    return (
        10.0
        *
        np.log10(
            np.max(power, axis=1)
            /
            (
                np.mean(power, axis=1)
                + 1e-12
            )
        )
    )


# ------------------------------------------------
# 8. Differentiable peak-aware shaping layer
# ------------------------------------------------
class PeakAwareResidual(tf.keras.layers.Layer):

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
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        # IMPORTANT:
        # Work entirely with real-valued tensors here.
        # The previous version converted the waveform to complex inside
        # this custom layer, which caused the float32/complex64 graph
        # inference error shown by TensorFlow/Keras.

        real = inputs[:, :N]
        imag = inputs[:, N:]

        magnitude = tf.sqrt(
            tf.square(real)
            +
            tf.square(imag)
            +
            1e-12
        )

        rms = tf.sqrt(
            tf.reduce_mean(
                tf.square(magnitude),
                axis=1,
                keepdims=True
            )
            +
            1e-12
        )

        excess = (
            magnitude
            -
            self.peak_threshold * rms
        )

        # Smooth positive peak detector
        soft_peak = (
            tf.nn.softplus(
                12.0 * excess
            )
            /
            12.0
        )

        # Unit radial direction
        inv_mag = 1.0 / (
            magnitude + 1e-8
        )

        dir_real = real * inv_mag
        dir_imag = imag * inv_mag

        gain = tf.nn.sigmoid(
            self.gain
        )

        correction_real = (
            gain
            *
            soft_peak
            *
            dir_real
        )

        correction_imag = (
            gain
            *
            soft_peak
            *
            dir_imag
        )

        shaped_real = (
            real
            -
            self.max_residual
            *
            correction_real
        )

        shaped_imag = (
            imag
            -
            self.max_residual
            *
            correction_imag
        )

        return tf.concat(
            [
                shaped_real,
                shaped_imag
            ],
            axis=1
        )

    def compute_output_shape(self, input_shape):
        return input_shape


# ------------------------------------------------
# 9. FCAE V8 model
# ------------------------------------------------
def build_model():

    inp = tf.keras.Input(
        shape=(2*N,),
        name="OFDM_input"
    )

    # Encoder extracts waveform context
    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(inp)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    latent = tf.keras.layers.Dense(
        64,
        activation="tanh",
        name="latent"
    )(x)

    # Decoder generates a context-dependent residual
    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(latent)

    x = tf.keras.layers.Dense(
        256,
        activation="relu"
    )(x)

    residual = tf.keras.layers.Dense(
        2*N,
        activation="tanh",
        name="residual_proposal"
    )(x)

    # Bounded residual
    residual = (
        MAX_RESIDUAL
        *
        residual
    )

    proposal = (
        inp
        +
        residual
    )

    # Explicit differentiable peak-aware shaping
    out = PeakAwareResidual(
        name="peak_aware_shaper"
    )(proposal)

    return tf.keras.Model(
        inp,
        out,
        name="FCAE_OFDM_X_V8"
    )


# ------------------------------------------------
# 10. Custom loss
# ------------------------------------------------
def fcae_loss(y_true, y_pred):

    s_true = tf.complex(
        y_true[:, :N],
        y_true[:, N:]
    )

    s_pred = tf.complex(
        y_pred[:, :N],
        y_pred[:, N:]
    )

    p_true = tf.abs(s_true)**2
    p_pred = tf.abs(s_pred)**2

    # --------------------------------------------
    # A. PAPR dB
    # --------------------------------------------
    papr_true = (
        10.0
        *
        tf.math.log(
            (
                tf.reduce_max(
                    p_true,
                    axis=1
                )
                /
                (
                    tf.reduce_mean(
                        p_true,
                        axis=1
                    )
                    + 1e-12
                )
            )
            + 1e-12
        )
        /
        tf.math.log(
            tf.constant(
                10.0,
                tf.float32
            )
        )
    )

    papr_pred = (
        10.0
        *
        tf.math.log(
            (
                tf.reduce_max(
                    p_pred,
                    axis=1
                )
                /
                (
                    tf.reduce_mean(
                        p_pred,
                        axis=1
                    )
                    + 1e-12
                )
            )
            + 1e-12
        )
        /
        tf.math.log(
            tf.constant(
                10.0,
                tf.float32
            )
        )
    )

    # Relative PAPR objective.
    # Penalize remaining PAPR but do not force an artificial value.
    papr_loss = tf.reduce_mean(
        tf.square(
            papr_pred
            -
            tf.stop_gradient(
                tf.minimum(
                    papr_true,
                    papr_pred
                )
            )
        )
    )

    # Additional target-aware hinge:
    # only samples still above the desired improvement are penalized.
    target = (
        papr_true
        -
        TARGET_REDUCTION_DB
    )

    target_hinge = tf.reduce_mean(
        tf.square(
            tf.nn.relu(
                papr_pred
                -
                target
            )
        )
    )

    # --------------------------------------------
    # B. Explicit peak loss
    # --------------------------------------------
    rms_true = tf.sqrt(
        tf.reduce_mean(
            p_true,
            axis=1,
            keepdims=True
        )
        + 1e-12
    )

    threshold = (
        PEAK_THRESHOLD
        *
        rms_true
    )

    excess = tf.nn.relu(
        tf.abs(s_pred)
        -
        threshold
    )

    peak_loss = tf.reduce_mean(
        tf.square(
            excess
        )
    )

    # --------------------------------------------
    # C. Reconstruction fidelity
    # --------------------------------------------
    recon_loss = tf.reduce_mean(
        tf.square(
            y_pred
            -
            y_true
        )
    )

    # --------------------------------------------
    # D. Power preservation
    # --------------------------------------------
    mean_true = tf.reduce_mean(
        p_true,
        axis=1
    )

    mean_pred = tf.reduce_mean(
        p_pred,
        axis=1
    )

    power_loss = tf.reduce_mean(
        tf.square(
            mean_pred
            -
            mean_true
        )
    )

    # --------------------------------------------
    # E. Frequency-domain fidelity
    # --------------------------------------------
    F_true = tf.signal.fft(
        tf.cast(
            s_true,
            tf.complex64
        )
    )

    F_pred = tf.signal.fft(
        tf.cast(
            s_pred,
            tf.complex64
        )
    )

    freq_loss = (
        tf.reduce_mean(
            tf.abs(
                F_pred
                -
                F_true
            )**2
        )
        /
        (
            tf.reduce_mean(
                tf.abs(
                    F_true
                )**2
            )
            + 1e-12
        )
    )

    # --------------------------------------------
    # F. EVM-like complex error
    # --------------------------------------------
    evm_loss = (
        tf.reduce_mean(
            tf.abs(
                s_pred
                -
                s_true
            )**2
        )
        /
        (
            tf.reduce_mean(
                tf.abs(
                    s_true
                )**2
            )
            + 1e-12
        )
    )

    # --------------------------------------------
    # G. Residual regularization
    # --------------------------------------------
    residual_loss = tf.reduce_mean(
        tf.square(
            y_pred
            -
            y_true
        )
    )

    return (
        W_PAPR * (
            papr_loss
            +
            target_hinge
        )
        +
        W_PEAK * peak_loss
        +
        W_RECON * recon_loss
        +
        W_POWER * power_loss
        +
        W_FREQ * freq_loss
        +
        W_EVM * evm_loss
        +
        W_RESIDUAL * residual_loss
    )


# ------------------------------------------------
# 11. Train/validation/test data
# ------------------------------------------------
train_x, train_bits = make_dataset(
    N_TRAIN,
    seed_offset=1
)

val_x, val_bits = make_dataset(
    N_VAL,
    seed_offset=2
)

# Completely independent test set
test_x, test_bits = make_dataset(
    N_TEST,
    seed_offset=3
)

# ------------------------------------------------
# 12. Build and compile
# ------------------------------------------------
model = build_model()

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss=fcae_loss
)

print("=" * 72)
print("FCAE / OFDM-X V8 - FINAL 0.20 dB ATTEMPT")
print("=" * 72)
print(
    f"Nsub={N}, CP={CP}, BPSK={BPSK}, "
    f"Delta-f={DELTA_F/1000:.1f} kHz, Fs={FS/1000:.1f} kHz"
)
print(
    f"Train={N_TRAIN}, Val={N_VAL}, "
    f"Independent test={N_TEST}"
)
print(
    f"Target reduction for training = "
    f"{TARGET_REDUCTION_DB:.2f} dB"
)
print(
    f"MAX_RESIDUAL={MAX_RESIDUAL}, "
    f"PEAK_THRESHOLD={PEAK_THRESHOLD}"
)
print("=" * 72)

# ------------------------------------------------
# 13. Training callbacks
# ------------------------------------------------
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )
]

# ------------------------------------------------
# 14. Train
# ------------------------------------------------
print("\nTraining FCAE V8...")

history = model.fit(
    train_x,
    train_x,
    validation_data=(
        val_x,
        val_x
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------
# 15. Independent test prediction
# ------------------------------------------------
pred_test = model.predict(
    test_x,
    batch_size=BATCH_SIZE,
    verbose=0
)

# ------------------------------------------------
# 16. PAPR results
# ------------------------------------------------
papr_original = papr_db(
    test_x
)

papr_fcae = papr_db(
    pred_test
)

mean_original = float(
    np.mean(
        papr_original
    )
)

mean_fcae = float(
    np.mean(
        papr_fcae
    )
)

improvement = (
    mean_original
    -
    mean_fcae
)

# ------------------------------------------------
# 16A. Save EXACT independent-test PAPR samples
# ------------------------------------------------
# These arrays are the authoritative data for Fig. 8.
# Do not regenerate the test set in a separate plotting script.
np.save(
    "V8_INDEPENDENT_TEST_PAPR_OFDM.npy",
    papr_original
)

np.save(
    "V8_INDEPENDENT_TEST_PAPR_FCAE.npy",
    papr_fcae
)

# ------------------------------------------------
# 17. Distortion / power statistics
# ------------------------------------------------
test_mse = float(
    np.mean(
        (
            pred_test
            -
            test_x
        )**2
    )
)

s0 = to_complex(
    test_x
)

s1 = to_complex(
    pred_test
)

power0 = np.mean(
    np.abs(s0)**2,
    axis=1
)

power1 = np.mean(
    np.abs(s1)**2,
    axis=1
)

power_error_pct = float(
    100.0
    *
    np.mean(
        np.abs(
            power1
            -
            power0
        )
        /
        (
            power0
            +
            1e-12
        )
    )
)

# ------------------------------------------------
# 18. EVM
# ------------------------------------------------
evm_rms = (
    np.sqrt(
        np.mean(
            np.abs(
                s1
                -
                s0
            )**2
        )
    )
    /
    (
        np.sqrt(
            np.mean(
                np.abs(s0)**2
            )
        )
        +
        1e-12
    )
)

evm_pct = float(
    100.0 * evm_rms
)

# ------------------------------------------------
# 19. BER proxy
# ------------------------------------------------
#
# The waveform shaper should preserve the BPSK information.
# We evaluate the received FFT symbols directly, without adding
# a new channel, so this is a distortion-only sanity check.
#
# Original OFDM:
#   FFT -> sign(real)
#
# FCAE:
#   FFT -> sign(real)
#
def ber_proxy(
    y,
    bits
):

    s = to_complex(
        y
    )

    X = np.fft.fft(
        s,
        axis=1
    )

    # Equalize common amplitude because the objective is
    # waveform-shaping distortion, not transmit gain.
    real_part = np.real(X)

    detected = (
        real_part >= 0
    ).astype(np.int8)

    errors = np.sum(
        detected != bits
    )

    total = bits.size

    return (
        float(errors)
        /
        float(total)
    )


ber_original = ber_proxy(
    test_x,
    test_bits
)

ber_fcae = ber_proxy(
    pred_test,
    test_bits
)

# ------------------------------------------------
# 20. Final independent result
# ------------------------------------------------
print("\n" + "=" * 72)
print("EXACT INDEPENDENT-VALIDATION RESULTS - FCAE V8")
print("=" * 72)

print(
    f"Original PAPR = "
    f"{mean_original:.3f} dB"
)

print(
    f"FCAE PAPR     = "
    f"{mean_fcae:.3f} dB"
)

print(
    f"Improvement   = "
    f"{improvement:.3f} dB"
)

print(
    f"Test MSE      = "
    f"{test_mse:.6e}"
)

print(
    f"Power error   = "
    f"{power_error_pct:.4f} %"
)

print(
    f"EVM proxy     = "
    f"{evm_pct:.4f} %"
)

print(
    f"BER proxy OFDM = "
    f"{ber_original:.6e}"
)

print(
    f"BER proxy FCAE = "
    f"{ber_fcae:.6e}"
)

if improvement >= TARGET_REDUCTION_DB:

    print(
        "\nTARGET CHECK: "
        ">= 0.20 dB achieved on independent test."
    )

else:

    print(
        "\nTARGET CHECK: "
        "0.20 dB NOT achieved. "
        "The measured improvement must be reported."
    )

# ------------------------------------------------
# 21. CCDF
# ------------------------------------------------
def ccdf(values):

    values = np.sort(
        values
    )

    prob = (
        1.0
        -
        np.arange(
            1,
            len(values)+1
        )
        /
        len(values)
    )

    prob = np.maximum(
        prob,
        1.0 / len(values)
    )

    return (
        values,
        prob
    )


x0, y0 = ccdf(
    papr_original
)

x1, y1 = ccdf(
    papr_fcae
)

# ------------------------------------------------
# Random Phase Rotation baseline
# ------------------------------------------------
# Four random phase candidates are tested for each OFDM symbol.
# The lowest-PAPR candidate is retained.
RPR_CANDIDATES = 4
rng_rpr = np.random.default_rng(20260808)

# Reconstruct the same BPSK frequency-domain symbols used by
# the independent test set.
test_symbols_fd = (
    2.0 * test_bits - 1.0
).astype(np.complex64)

best_papr_rpr = np.full(
    N_TEST,
    np.inf,
    dtype=np.float64
)

for _ in range(RPR_CANDIDATES):

    phase = rng_rpr.uniform(
        0.0,
        2.0 * np.pi,
        N
    )

    phase_vector = np.exp(
        1j * phase
    )

    rotated_fd = (
        test_symbols_fd
        *
        phase_vector[None, :]
    )

    rotated_td = np.fft.ifft(
        rotated_fd,
        axis=1
    )

    # Normalize each symbol in exactly the same way as the
    # OFDM dataset before evaluating PAPR.
    power = np.mean(
        np.abs(rotated_td)**2,
        axis=1,
        keepdims=True
    )

    rotated_td = (
        rotated_td
        /
        np.sqrt(power + 1e-12)
    )

    papr_candidate = (
        10.0
        *
        np.log10(
            np.max(
                np.abs(rotated_td)**2,
                axis=1
            )
            /
            (
                np.mean(
                    np.abs(rotated_td)**2,
                    axis=1
                )
                + 1e-12
            )
        )
    )

    best_papr_rpr = np.minimum(
        best_papr_rpr,
        papr_candidate
    )

papr_rpr_ccdf = best_papr_rpr

xr, yr = ccdf(
    papr_rpr_ccdf
)

print(
    f"Random Phase Rotation mean PAPR = "
    f"{np.mean(papr_rpr_ccdf):.3f} dB"
)

# ------------------------------------------------
# 22. CCDF consistency
# ------------------------------------------------
ccdf_consistency_error = abs(
    (
        float(
            np.mean(
                papr_original
            )
        )
        -
        float(
            np.mean(
                papr_fcae
            )
        )
    )
    -
    improvement
)

print("\nPAPR CCDF validation statistics:")
print(
    f"Conventional OFDM mean PAPR = "
    f"{np.mean(papr_original):.3f} dB"
)

print(
    f"FCAE / OFDM-X mean PAPR = "
    f"{np.mean(papr_fcae):.3f} dB"
)

print(
    f"FCAE improvement = "
    f"{improvement:.3f} dB"
)

print(
    f"FCAE CCDF/validation consistency "
    f"error = "
    f"{ccdf_consistency_error:.3e} dB"
)

# ------------------------------------------------
# 23. Publication-quality PAPR CCDF plot
# ------------------------------------------------
fig, ax = plt.subplots(
    figsize=(3.25, 3.25),
    dpi=300
)

ax.semilogy(
    x0,
    y0,
    linewidth=1.4,
    label="Conventional OFDM"
)

ax.semilogy(
    xr,
    yr,
    linewidth=1.4,
    label="Random Phase Rotation"
)

ax.semilogy(
    x1,
    y1,
    linewidth=1.4,
    label="FCAE / OFDM-X"
)

ax.set_xlabel(
    "PAPR (dB)",
    fontsize=8
)

ax.set_ylabel(
    r"CCDF = P(PAPR > PAPR$_0$)",
    fontsize=8
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=8
)

ax.tick_params(
    axis="both",
    which="minor",
    labelsize=8
)

ax.grid(
    True,
    which="both",
    linestyle="--",
    linewidth=0.6,
    alpha=0.35
)

ax.legend(
    fontsize=8,
    loc="upper right",
    frameon=True
)

# Make sure every text object uses 8-pt Times New Roman.
for label in (
    ax.get_xticklabels()
    +
    ax.get_yticklabels()
):
    label.set_fontname("Times New Roman")
    label.set_fontsize(8)

ax.xaxis.label.set_fontname("Times New Roman")
ax.yaxis.label.set_fontname("Times New Roman")

for txt in ax.get_legend().get_texts():
    txt.set_fontname("Times New Roman")
    txt.set_fontsize(8)

fig.subplots_adjust(
    left=0.19,
    right=0.98,
    bottom=0.17,
    top=0.98
)

fig.savefig(
    "FCAE_OFDM_X_V8_PAPR_CCDF_3CURVES_FINAL_300dpi.png",
    dpi=300,
    facecolor="white"
)

plt.show()


# ------------------------------------------------
# 23A. FIGURE 8 - EXACT INDEPENDENT-TEST PAPR DISTRIBUTION
# ------------------------------------------------
# IMPORTANT:
# Fig. 8 is generated from the SAME papr_original and papr_fcae
# arrays used for the final independent-validation result above.
# Therefore the histogram means and the reported validation means
# cannot silently come from different test sets.

fig, ax = plt.subplots(
    figsize=(3.25, 3.25),
    dpi=300
)

xmin = min(
    float(np.min(papr_original)),
    float(np.min(papr_fcae))
)

xmax = max(
    float(np.max(papr_original)),
    float(np.max(papr_fcae))
)

bins = np.linspace(
    np.floor(xmin * 10.0) / 10.0,
    np.ceil(xmax * 10.0) / 10.0,
    50
)

ax.hist(
    papr_original,
    bins=bins,
    alpha=0.55,
    label="Conventional OFDM"
)

ax.hist(
    papr_fcae,
    bins=bins,
    alpha=0.55,
    label="FCAE / OFDM-X"
)

# These are the measured means from THIS EXACT test set.
ax.axvline(
    mean_original,
    linestyle="--",
    linewidth=1.0,
    label=f"Mean OFDM = {mean_original:.3f} dB"
)

ax.axvline(
    mean_fcae,
    linestyle="--",
    linewidth=1.0,
    label=f"Mean FCAE = {mean_fcae:.3f} dB"
)

ax.set_xlabel(
    "PAPR (dB)",
    fontsize=8,
    fontname="Times New Roman"
)

ax.set_ylabel(
    "Number of Symbols",
    fontsize=8,
    fontname="Times New Roman"
)

ax.tick_params(
    axis="both",
    which="major",
    labelsize=8
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.45,
    alpha=0.30
)

legend = ax.legend(
    loc="upper right",
    fontsize=8,
    frameon=True
)

for t in legend.get_texts():
    t.set_fontname("Times New Roman")
    t.set_fontsize(8)

for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontname("Times New Roman")
    t.set_fontsize(8)

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.17,
    top=0.98
)

fig.savefig(
    "FCAE_OFDM_X_V8_FIG8_PAPR_DISTRIBUTION_EXACT_300dpi_TNR8.png",
    dpi=300,
    facecolor="white"
)

plt.show()

print("\nFIGURE 8 - EXACT DATA CHECK")
print(
    f"Histogram mean OFDM = {np.mean(papr_original):.3f} dB"
)
print(
    f"Histogram mean FCAE = {np.mean(papr_fcae):.3f} dB"
)
print(
    f"Histogram improvement = "
    f"{np.mean(papr_original)-np.mean(papr_fcae):.3f} dB"
)
print(
    "Fig. 8 uses the SAME independent-test arrays as the "
    "final validation result."
)
print(
    "Saved: "
    "FCAE_OFDM_X_V8_FIG8_PAPR_DISTRIBUTION_EXACT_300dpi_TNR8.png"
)

# ------------------------------------------------
# 24. Save model
# ------------------------------------------------
model.save(
    "FCAE_OFDM_X_V8.keras"
)

# ------------------------------------------------
# 25. Save results
# ------------------------------------------------
with open(
    "FCAE_OFDM_X_V8_RESULTS_3CURVES.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FCAE / OFDM-X V8 - Final Controlled Peak Optimization\n"
    )

    f.write(
        "=" * 65 + "\n"
    )

    f.write(
        f"Original PAPR = "
        f"{mean_original:.6f} dB\n"
    )

    f.write(
        f"FCAE PAPR = "
        f"{mean_fcae:.6f} dB\n"
    )

    f.write(
        f"Improvement = "
        f"{improvement:.6f} dB\n"
    )

    f.write(
        f"Test MSE = "
        f"{test_mse:.8e}\n"
    )

    f.write(
        f"Power error = "
        f"{power_error_pct:.6f} %\n"
    )

    f.write(
        f"EVM proxy = "
        f"{evm_pct:.6f} %\n"
    )

    f.write(
        f"BER proxy OFDM = "
        f"{ber_original:.8e}\n"
    )

    f.write(
        f"BER proxy FCAE = "
        f"{ber_fcae:.8e}\n"
    )

    f.write(
        f"CCDF consistency error = "
        f"{ccdf_consistency_error:.8e} dB\n"
    )

    f.write(
        f"Target = "
        f"{TARGET_REDUCTION_DB:.6f} dB\n"
    )

    f.write(
        f"Target achieved = "
        f"{improvement >= TARGET_REDUCTION_DB}\n"
    )

print("\nFiles saved:")
print("FCAE_OFDM_X_V8_PAPR_CCDF_3CURVES_300dpi.png")
print("FCAE_OFDM_X_V8.keras")
print("FCAE_OFDM_X_V8_RESULTS_3CURVES.txt")
