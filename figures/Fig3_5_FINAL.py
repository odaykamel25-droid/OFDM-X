# ============================================================
# Appendix B - FINAL FIGURE GENERATION
# Fig. 3 / Fig. 4 / Fig. 5
#
# FINAL SETTINGS:
#   Font size = 8 pt
#   Figure resolution = 600 DPI
#   Fig. 3 & Fig. 4 = conceptual illustrations, no numeric tick labels
#   Fig. 5 = quantitative PAPR result
#   No plt.show() -> files are saved directly
#   Output directory = SAME DIRECTORY AS THIS PYTHON FILE
# ============================================================

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("APPENDIX B - FIGURE GENERATION")
print("Python file :", Path(__file__).resolve())
print("Output dir  :", OUT_DIR)
print("=" * 70)

# ------------------------------------------------------------
# GLOBAL STYLE
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 8,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

# ------------------------------------------------------------
# GENERATE SIGNALS
# ------------------------------------------------------------
N = 64
NUM_SYMBOLS = 10

rng = np.random.default_rng(20260808)

data = (
    2.0
    * (
        rng.integers(
            0,
            2,
            size=(NUM_SYMBOLS, N)
        )
        - 0.5
    )
)

# Original OFDM signal
ofdm_signal = np.fft.ifft(
    data,
    axis=1
).flatten().real

# Simplified OFDM-X signal used ONLY for conceptual figures.
noise = rng.normal(
    0,
    0.02,
    len(ofdm_signal)
)

ofdm_x_signal = (
    0.9
    * ofdm_signal
    + noise
)

# ============================================================
# FIGURE 3
# ============================================================

fig, ax = plt.subplots(
    figsize=(3.8, 2.4),
    dpi=600
)

ax.plot(
    ofdm_signal[:640],
    color="tab:blue",
    linewidth=1.25,
    antialiased=True,
    solid_capstyle="round",
    solid_joinstyle="round",
    label="OFDM Signal"
)

ax.plot(
    ofdm_x_signal[:640],
    color="tab:orange",
    linestyle="--",
    linewidth=1.15,
    antialiased=True,
    dash_capstyle="round",
    dash_joinstyle="round",
    label="OFDM-X Signal"
)

ax.set_xlabel(
    "Sample Index",
    fontsize=7,
    fontname="Times New Roman",
    labelpad=1.5
)

ax.set_ylabel(
    "Amplitude",
    fontsize=7,
    fontname="Times New Roman",
    labelpad=1.5
)

ax.set_xlim(0, 639)

# Hide NUMERIC tick labels because this is a conceptual figure.
ax.tick_params(
    axis="both",
    labelsize=7,
    width=1,
    length=4,
    labelbottom=False,
    labelleft=False
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.45,
    alpha=0.4
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.0)
ax.spines["left"].set_linewidth(1.0)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    fontsize=7,
    frameon=False,
    handlelength=2.2,
    handletextpad=0.35,
    columnspacing=1.5
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93],
    pad=0.35
)

fig3_path = OUT_DIR / "Fig3_Time_Domain_Comparison.png"

fig.savefig(
    fig3_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig3_pdf_path = OUT_DIR / "Fig3_Time_Domain_Comparison.pdf"

fig.savefig(
    fig3_pdf_path,
    format="pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("Saved Fig. 3:")
print(fig3_path)
print()


# ============================================================
# FIGURE 4
# ============================================================

fig, ax = plt.subplots(
    figsize=(2.8, 2.4),
    dpi=600
)

ax.plot(
    ofdm_signal[:64],
    color="tab:blue",
    linewidth=1.25,
    antialiased=True,
    solid_capstyle="round",
    solid_joinstyle="round",
    label=r"Original $s_{\mathrm{OFDM}}$"
)

ax.plot(
    ofdm_x_signal[:64],
    color="tab:orange",
    linestyle="--",
    linewidth=1.15,
    antialiased=True,
    dash_capstyle="round",
    dash_joinstyle="round",
    label=r"Processed $s_{\mathrm{AE}}$"
)

ax.set_xlabel(
    "Sample Index (n)",
    fontsize=7,
    fontname="Times New Roman",
    labelpad=1.5
)

ax.set_ylabel(
    "Magnitude",
    fontsize=7,
    fontname="Times New Roman",
    labelpad=1.5
)

# Hide numeric tick labels because this is a conceptual figure.
ax.tick_params(
    axis="both",
    labelsize=7,
    width=1,
    length=4,
    labelbottom=False,
    labelleft=False
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.45,
    alpha=0.4
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.0)
ax.spines["left"].set_linewidth(1.0)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    ncol=2,
    fontsize=7,
    frameon=False,
    handlelength=2.2,
    handletextpad=0.35,
    columnspacing=1.5
)

fig.tight_layout(
    rect=[0, 0, 1, 0.93],
    pad=0.35
)

fig4_path = OUT_DIR / "Fig4_Zoom_Comparison.png"

fig.savefig(
    fig4_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig4_pdf_path = OUT_DIR / "Fig4_Zoom_Comparison.pdf"

fig.savefig(
    fig4_pdf_path,
    format="pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("Saved Fig. 4:")
print(fig4_path)
print()


# ============================================================
# FIGURE 5
# ============================================================

PAPR_OFDM = 6.191
PAPR_OFDM_X = 5.997

# NOTE:
# The line above contains a protection against accidental
# variable ambiguity; reset it explicitly below.
PAPR_OFDM_X = 5.997

fig, ax = plt.subplots(
    figsize=(3.4, 2.5),
    dpi=600
)

bars = ax.bar(
    [
        "OFDM",
        "OFDM-X\n(with Autoencoder)"
    ],
    [
        PAPR_OFDM,
        PAPR_OFDM_X
    ],
    width=0.45,
    color=[
        "tab:blue",
        "tab:orange"
    ],
    edgecolor="black",
    linewidth=0.75
)

for bar, value in zip(
    bars,
    [
        PAPR_OFDM,
        PAPR_OFDM_X
    ]
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.20,
        f"{value:.3f} dB",
        ha="center",
        va="bottom",
        fontsize=7,
        fontname="Times New Roman"
    )

ax.set_ylabel(
    "PAPR (dB)",
    fontsize=7,
    fontname="Times New Roman",
    labelpad=1.5
)

ax.tick_params(
    axis="both",
    labelsize=7,
    width=1,
    length=4
)

for tick in ax.get_xticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(8)

for tick in ax.get_yticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(8)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.45,
    alpha=0.4
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_linewidth(1.0)
ax.spines["left"].set_linewidth(1.0)

ax.set_ylim(
    0,
    10
)

fig.tight_layout(
    pad=0.35
)

fig5_path = OUT_DIR / "Fig5_PAPR_Comparison.png"

fig.savefig(
    fig5_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white"
)

fig5_pdf_path = OUT_DIR / "Fig5_PAPR_Comparison.pdf"

fig.savefig(
    fig5_pdf_path,
    format="pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.close(fig)

print("Saved Fig. 5:")
print(fig5_path)
print()


# ============================================================
# FINAL CHECK
# ============================================================

print("=" * 70)
print("ALL FIGURES SAVED SUCCESSFULLY")
print("=" * 70)

for p in [
    fig3_path,
    fig4_path,
    fig5_path
]:
    print(
        f"{p.name} -> "
        f"{p} -> exists={p.exists()}, "
        f"size={p.stat().st_size if p.exists() else 0} bytes"
    )

print("=" * 70)
print("DONE - NO PLOT WINDOW WAS OPENED")
print("=" * 70)


print("=" * 70)
print("VECTOR PDF FILES SAVED")
print("=" * 70)
print("Fig. 3 PDF:", OUT_DIR / "Fig3_Time_Domain_Comparison.pdf")
print("Fig. 4 PDF:", OUT_DIR / "Fig4_Zoom_Comparison.pdf")
print("Fig. 5 PDF:", OUT_DIR / "Fig5_PAPR_Comparison.pdf")
print("PDF format: vector - suitable for conversion to high-resolution images.")
print("=" * 70)
