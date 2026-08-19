# ============================================================
# OFDM-X V8 - FIGURE 1 REGENERATION
# REAL independent-test PAPR distribution before/after FCAE
#
# Publication specifications:
#   - 3.25 x 3.25 inch
#   - 300 dpi
#   - Times New Roman
#   - all visible text = 8 pt
#   - no title inside the figure
#
# Required file in the same folder:
#   FCAE_OFDM_X_V8.keras
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import os
import sys
import matplotlib.font_manager as fm


# ------------------------------------------------------------
# V8 custom layer required to load the final FCAE model
# ------------------------------------------------------------
N = 64
MAX_RESIDUAL = 0.18
PEAK_THRESHOLD = 1.55


@tf.keras.utils.register_keras_serializable(package="FCAE")
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
        real = inputs[:, :N]
        imag = inputs[:, N:]

        magnitude = tf.sqrt(
            tf.square(real) +
            tf.square(imag) +
            1e-12
        )

        rms = tf.sqrt(
            tf.reduce_mean(
                tf.square(magnitude),
                axis=1,
                keepdims=True
            ) +
            1e-12
        )

        excess = magnitude - self.peak_threshold * rms

        soft_peak = (
            tf.nn.softplus(12.0 * excess) / 12.0
        )

        inv_mag = 1.0 / (magnitude + 1e-8)

        dir_real = real * inv_mag
        dir_imag = imag * inv_mag

        gain = tf.nn.sigmoid(self.gain)

        correction_real = (
            gain * soft_peak * dir_real
        )

        correction_imag = (
            gain * soft_peak * dir_imag
        )

        shaped_real = (
            real -
            self.max_residual * correction_real
        )

        shaped_imag = (
            imag -
            self.max_residual * correction_imag
        )

        return tf.concat(
            [shaped_real, shaped_imag],
            axis=1
        )

    def compute_output_shape(self, input_shape):
        return input_shape


# ------------------------------------------------------------
# Publication-quality plotting settings
# ------------------------------------------------------------
available_fonts = {f.name for f in fm.fontManager.ttflist}
FONT = "Times New Roman" if "Times New Roman" in available_fonts else "DejaVu Serif"

plt.rcParams["font.family"] = FONT
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.dpi"] = 300


# ------------------------------------------------------------
# V8 independent-test parameters
# ------------------------------------------------------------
SEED = 42
N_TEST = 6000

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------------------
# PAPR function
# ------------------------------------------------------------
def papr_db(x):
    p = np.abs(x) ** 2
    return 10.0 * np.log10(
        np.max(p) / (np.mean(p) + 1e-12)
    )


# ------------------------------------------------------------
# Generate the SAME deterministic independent-test OFDM set
# used by the V8 validation figures.
# BPSK, N=64, 6000 test symbols.
# ------------------------------------------------------------
bits = np.random.randint(
    0, 2, size=(N_TEST, N)
)

symbols_fd = (
    2.0 * bits - 1.0
).astype(np.complex64)

ofdm_td = np.fft.ifft(
    symbols_fd,
    axis=1
).astype(np.complex64)

# Normalize each symbol to unit average power.
power = np.mean(
    np.abs(ofdm_td) ** 2,
    axis=1,
    keepdims=True
)

ofdm_td = (
    ofdm_td /
    np.sqrt(power + 1e-12)
).astype(np.complex64)


# ------------------------------------------------------------
# Convert complex waveform to FCAE input:
# [real(64), imag(64)] -> 128 features
# ------------------------------------------------------------
X_test = np.concatenate(
    [
        np.real(ofdm_td),
        np.imag(ofdm_td)
    ],
    axis=1
).astype(np.float32)


# ------------------------------------------------------------
# Load FINAL V8 FCAE model
# ------------------------------------------------------------
SCRIPT_FOLDER = Path(__file__).resolve().parent
MODEL_FILE = SCRIPT_FOLDER / "FCAE_OFDM_X_V8.keras"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        "\nFCAE model was not found.\n"
        f"Python file folder: {SCRIPT_FOLDER}\n"
        f"Expected model file: {MODEL_FILE}\n\n"
        "Place FCAE_OFDM_X_V8.keras in the SAME folder as FIG1.py."
    )

model = tf.keras.models.load_model(
    str(MODEL_FILE),
    custom_objects={
        "PeakAwareResidual": PeakAwareResidual
    },
    compile=False
)


# ------------------------------------------------------------
# FCAE inference
# ------------------------------------------------------------
Y_pred = model.predict(
    X_test,
    batch_size=256,
    verbose=0
)

Y_pred = np.asarray(Y_pred)
Y_pred = np.squeeze(Y_pred)

if Y_pred.ndim != 2 or Y_pred.shape[1] != 2 * N:
    raise ValueError(
        f"Unexpected FCAE output shape: {Y_pred.shape}. "
        f"Expected (samples, {2*N})."
    )

fcae_td = (
    Y_pred[:, :N] +
    1j * Y_pred[:, N:]
).astype(np.complex64)


# ------------------------------------------------------------
# Calculate REAL PAPR distributions
# ------------------------------------------------------------
papr_original = np.array(
    [papr_db(x) for x in ofdm_td]
)

papr_fcae = np.array(
    [papr_db(x) for x in fcae_td]
)

mean_original = float(np.mean(papr_original))
mean_fcae = float(np.mean(papr_fcae))
improvement = mean_original - mean_fcae


# ------------------------------------------------------------
# Figure 1
# Actual measured PAPR probability-density distributions.
# This replaces the previous artificial Gaussian curves.
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(3.25, 3.25),
    dpi=300
)

# Common bins for a direct distribution comparison.
xmin = min(
    np.min(papr_original),
    np.min(papr_fcae)
)

xmax = max(
    np.max(papr_original),
    np.max(papr_fcae)
)

bins = np.linspace(
    xmin - 0.10,
    xmax + 0.10,
    45
)

ax.hist(
    papr_original,
    bins=bins,
    density=True,
    alpha=0.50,
    label="Conventional OFDM"
)

ax.hist(
    papr_fcae,
    bins=bins,
    density=True,
    alpha=0.50,
    label="FCAE / OFDM-X"
)

# Measured means from the SAME independent-test data.
ax.axvline(
    mean_original,
    linestyle="--",
    linewidth=1.0
)

ax.axvline(
    mean_fcae,
    linestyle="--",
    linewidth=1.0
)

ax.set_xlabel(
    "PAPR (dB)",
    fontsize=8,
    fontname=FONT
)

ax.set_ylabel(
    "Probability Density",
    fontsize=8,
    fontname=FONT
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

OUTPUT = (
    SCRIPT_FOLDER /
    "FCAE_OFDM_X_V8_FIG1_PAPR_DISTRIBUTION_REAL_300dpi_TNR8.png"
)

fig.savefig(
    OUTPUT,
    dpi=300,
    facecolor="white"
)

plt.show(block=False)

# Open the saved PNG automatically in Windows, exactly like Fig. 9.
if sys.platform.startswith("win"):
    try:
        os.startfile(str(OUTPUT))
        print("The PNG has been opened automatically.")
    except Exception as e:
        print(f"Automatic opening failed: {e}")


# ------------------------------------------------------------
# Numerical verification
# ------------------------------------------------------------
print("=" * 72)
print("FIGURE 1 - V8 REAL INDEPENDENT-TEST PAPR DISTRIBUTION")
print("=" * 72)

print(
    f"Conventional OFDM mean PAPR = "
    f"{mean_original:.3f} dB"
)

print(
    f"FCAE / OFDM-X mean PAPR     = "
    f"{mean_fcae:.3f} dB"
)

print(
    f"FCAE PAPR improvement       = "
    f"{improvement:.3f} dB"
)

print(
    f"Independent test symbols    = "
    f"{N_TEST}"
)

print(
    "Expected V8 reference values:"
)

print(
    "Conventional OFDM = 6.191 dB"
)

print(
    "FCAE / OFDM-X     = 5.997 dB"
)

print(
    "Improvement       = 0.194 dB"
)

print(
    f"\nSaved: {OUTPUT}"
)

print(
    "Figure size: 3.25 x 3.25 inch"
)

print(
    "Resolution: 300 dpi"
)

print(
    "Font: Times New Roman"
)

print(
    "All visible figure text: 8 pt"
)
