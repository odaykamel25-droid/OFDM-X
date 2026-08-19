# ============================================================
# FIGURE 9 - FINAL V8 RPR
# DISPLAY + SAVE VERSION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import os
import sys

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
SEED = 42
TEST_SEED = 45
N = 64
N_TEST = 6000

RPR_CANDIDATES = 4
RPR_SEED = 20260808

FIG_SIZE = (3.25, 3.25)
DPI = 300
FONT_SIZE = 8

# ------------------------------------------------------------
# Generate the same independent-test data
# ------------------------------------------------------------
rng = np.random.default_rng(TEST_SEED)

bits = rng.integers(
    0, 2, size=(N_TEST, N)
).astype(np.int8)

symbols_fd = (
    2.0 * bits - 1.0
).astype(np.complex64)

ofdm_td = np.fft.ifft(
    symbols_fd,
    axis=1
).astype(np.complex64)

power = np.mean(
    np.abs(ofdm_td) ** 2,
    axis=1,
    keepdims=True
)

ofdm_td = (
    ofdm_td /
    np.sqrt(power + 1e-12)
)


# ------------------------------------------------------------
# PAPR
# ------------------------------------------------------------
def papr_db(signals):
    p = np.abs(signals) ** 2

    return (
        10.0
        *
        np.log10(
            np.max(p, axis=1)
            /
            (
                np.mean(p, axis=1)
                +
                1e-12
            )
        )
    )


papr_ofdm = papr_db(ofdm_td)


# ------------------------------------------------------------
# Random Phase Rotation
# ------------------------------------------------------------
rng_rpr = np.random.default_rng(RPR_SEED)

best_papr = np.full(
    N_TEST,
    np.inf,
    dtype=np.float64
)

for candidate in range(RPR_CANDIDATES):

    phase = rng_rpr.uniform(
        0.0,
        2.0 * np.pi,
        N
    )

    phase_vector = np.exp(
        1j * phase
    )

    rotated_fd = (
        symbols_fd
        *
        phase_vector[None, :]
    )

    rotated_td = np.fft.ifft(
        rotated_fd,
        axis=1
    )

    rotated_power = np.mean(
        np.abs(rotated_td) ** 2,
        axis=1,
        keepdims=True
    )

    rotated_td = (
        rotated_td /
        np.sqrt(rotated_power + 1e-12)
    )

    candidate_papr = papr_db(
        rotated_td
    )

    best_papr = np.minimum(
        best_papr,
        candidate_papr
    )

papr_rpr = best_papr


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
mean_ofdm = float(np.mean(papr_ofdm))
mean_rpr = float(np.mean(papr_rpr))
reduction = mean_ofdm - mean_rpr


# ------------------------------------------------------------
# Font
# ------------------------------------------------------------
available_fonts = {
    f.name
    for f in fm.fontManager.ttflist
}

if "Times New Roman" in available_fonts:
    FONT = "Times New Roman"
else:
    FONT = "DejaVu Serif"

plt.rcParams["font.family"] = FONT
plt.rcParams["font.size"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8


# ------------------------------------------------------------
# Create figure
# ------------------------------------------------------------
fig, ax = plt.subplots(
    figsize=FIG_SIZE,
    dpi=DPI
)

xmin = min(
    np.min(papr_ofdm),
    np.min(papr_rpr)
)

xmax = max(
    np.max(papr_ofdm),
    np.max(papr_rpr)
)

bins = np.linspace(
    np.floor(xmin * 10.0) / 10.0,
    np.ceil(xmax * 10.0) / 10.0,
    45
)

ax.hist(
    papr_ofdm,
    bins=bins,
    alpha=0.55,
    label="Conventional OFDM"
)

ax.hist(
    papr_rpr,
    bins=bins,
    alpha=0.55,
    label="Random Phase Rotation"
)

ax.axvline(
    mean_ofdm,
    linestyle="--",
    linewidth=1.0,
    label=f"Mean OFDM = {mean_ofdm:.3f} dB"
)

ax.axvline(
    mean_rpr,
    linestyle="--",
    linewidth=1.0,
    label=f"Mean RPR = {mean_rpr:.3f} dB"
)

ax.set_xlabel(
    "PAPR (dB)",
    fontsize=8,
    fontname=FONT
)

ax.set_ylabel(
    "Number of Symbols",
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

for text in legend.get_texts():
    text.set_fontname(FONT)
    text.set_fontsize(8)

for text in (
    ax.get_xticklabels()
    +
    ax.get_yticklabels()
):
    text.set_fontname(FONT)
    text.set_fontsize(8)

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.17,
    top=0.98
)


# ------------------------------------------------------------
# Save to the SAME folder as this Python file
# ------------------------------------------------------------
script_folder = Path(__file__).resolve().parent

output_png = (
    script_folder /
    "FCAE_OFDM_X_FIG9_RPR_V8_300dpi_TimesNewRoman_8pt.png"
)

fig.savefig(
    output_png,
    dpi=300,
    facecolor="white"
)


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------
print("=" * 72)
print("FIGURE 9 - FINAL V8 UNIFIED RANDOM PHASE ROTATION")
print("=" * 72)
print(f"Conventional OFDM mean PAPR = {mean_ofdm:.3f} dB")
print(f"Random Phase Rotation mean PAPR = {mean_rpr:.3f} dB")
print(f"RPR PAPR reduction = {reduction:.3f} dB")
print(f"Independent test symbols = {N_TEST}")
print(f"Test seed = {TEST_SEED}")
print(f"RPR candidates = {RPR_CANDIDATES}")
print(f"RPR seed = {RPR_SEED}")
print()
print(f"SAVED HERE:")
print(output_png)
print()
print("Figure size: 3.25 x 3.25 inch")
print("Resolution: 300 dpi")
print(f"Font: {FONT}")
print("All visible figure text: 8 pt")
print("=" * 72)


# ------------------------------------------------------------
# Open the saved PNG automatically in Windows
# ------------------------------------------------------------
if sys.platform.startswith("win"):
    try:
        os.startfile(str(output_png))
        print("The PNG has been opened automatically.")
    except Exception as e:
        print(f"Automatic opening failed: {e}")


# ------------------------------------------------------------
# Also display using Matplotlib
# ------------------------------------------------------------
try:
    plt.show(block=True)
except Exception as e:
    print(f"Matplotlib display was unavailable: {e}")
finally:
    plt.close(fig)
