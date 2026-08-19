# ============================================================
# OFDM-X V8 - FIGURE 7 REGENERATION
# Real simulated time-domain waveform before/after FCAE
#
# Publication specifications:
#   - 3.25 x 3.25 inch
#   - 300 dpi
#   - Times New Roman
#   - all visible text = 8 pt
#   - no title inside the figure
#   - deterministic BPSK / OFDM test generation
#
# Required file in the same folder:
#   FCAE_OFDM_X_V8.keras
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ------------------------------------------------------------
# V8 custom layer required to load FCAE_OFDM_X_V8.keras
# ------------------------------------------------------------
MAX_RESIDUAL = 0.18
PEAK_THRESHOLD = 1.55

@tf.keras.utils.register_keras_serializable(
    package="FCAE"
)
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
            tf.square(real)
            + tf.square(imag)
            + 1e-12
        )

        rms = tf.sqrt(
            tf.reduce_mean(
                tf.square(magnitude),
                axis=1,
                keepdims=True
            )
            + 1e-12
        )

        excess = (
            magnitude
            - self.peak_threshold * rms
        )

        soft_peak = (
            tf.nn.softplus(12.0 * excess)
            / 12.0
        )

        inv_mag = 1.0 / (
            magnitude + 1e-8
        )

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
            real
            - self.max_residual * correction_real
        )

        shaped_imag = (
            imag
            - self.max_residual * correction_imag
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
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["figure.dpi"] = 300

# ------------------------------------------------------------
# V8 simulation parameters
# ------------------------------------------------------------
SEED = 42
N = 64
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
# Generate deterministic independent-test OFDM symbols
# BPSK, N=64, matching V8 base parameters.
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
# Convert complex OFDM waveform to V8 FCAE input:
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
# Load the FINAL V8 model.
# compile=False avoids requiring the custom training loss.
# ------------------------------------------------------------
MODEL_FILE = "FCAE_OFDM_X_V8.keras"

model = tf.keras.models.load_model(
    MODEL_FILE,
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

# Handle possible extra singleton dimensions.
Y_pred = np.squeeze(Y_pred)

if Y_pred.ndim != 2 or Y_pred.shape[1] != 2 * N:
    raise ValueError(
        f"Unexpected FCAE output shape: {Y_pred.shape}. "
        f"Expected (samples, {2*N})."
    )

fcae_td = (
    Y_pred[:, :N]
    + 1j * Y_pred[:, N:]
).astype(np.complex64)

# ------------------------------------------------------------
# Select one REAL independent-test waveform.
#
# To make Fig. 7 informative, select the test waveform having
# the highest conventional-OFDM PAPR. This shows the peak
# shaping effect without inventing or manually altering data.
# ------------------------------------------------------------
papr_original = np.array(
    [papr_db(x) for x in ofdm_td]
)

idx = int(np.argmax(papr_original))

x_original = ofdm_td[idx]
x_fcae = fcae_td[idx]

# ------------------------------------------------------------
# Normalize only the displayed FCAE waveform for average-power
# matching. This is NOT used to claim additional PAPR reduction;
# it simply makes the visual comparison power-consistent.
# ------------------------------------------------------------
p0 = np.mean(np.abs(x_original) ** 2)
p1 = np.mean(np.abs(x_fcae) ** 2)

if p1 > 1e-12:
    x_fcae_plot = x_fcae * np.sqrt(p0 / p1)
else:
    x_fcae_plot = x_fcae

mag_original = np.abs(x_original)
mag_fcae = np.abs(x_fcae_plot)

# ------------------------------------------------------------
# Report the selected waveform information
# ------------------------------------------------------------
print("=" * 72)
print("FIGURE 7 - V8 INDEPENDENT-TEST WAVEFORM")
print("=" * 72)
print(f"Selected test symbol index = {idx}")
print(f"Conventional OFDM PAPR      = {papr_db(x_original):.3f} dB")
print(f"FCAE / OFDM-X PAPR          = {papr_db(x_fcae_plot):.3f} dB")
print(f"Original mean power         = {np.mean(np.abs(x_original)**2):.6f}")
print(f"FCAE displayed mean power   = {np.mean(np.abs(x_fcae_plot)**2):.6f}")

# ------------------------------------------------------------
# Publication-quality Figure 7
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=(3.25, 3.25),
    dpi=300
)

n = np.arange(N)

ax.plot(
    n,
    mag_original,
    linewidth=1.2,
    label="Conventional OFDM"
)

ax.plot(
    n,
    mag_fcae,
    linewidth=1.2,
    linestyle="--",
    label="FCAE / OFDM-X"
)

ax.set_xlabel(
    "Sample Index",
    fontsize=8,
    fontname="Times New Roman"
)

ax.set_ylabel(
    "Magnitude",
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
    linewidth=0.5,
    alpha=0.35
)

leg = ax.legend(
    fontsize=8,
    loc="upper right",
    frameon=True
)

for t in leg.get_texts():
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

# IMPORTANT: no bbox_inches='tight' so the final image remains
# exactly 3.25 x 3.25 inches at 300 dpi (~975 x 975 pixels).
OUT_FILE = (
    "FCAE_OFDM_X_V8_FIG7_WAVEFORM_"
    "300dpi_TNR8.png"
)

fig.savefig(
    OUT_FILE,
    dpi=300,
    facecolor="white"
)

plt.show()

print()
print(f"Saved: {OUT_FILE}")
print("Figure size: 3.25 x 3.25 inch")
print("Resolution: 300 dpi")
print("Expected raster size: approximately 975 x 975 pixels")
print("Font: Times New Roman")
print("All visible figure text: 8 pt")
