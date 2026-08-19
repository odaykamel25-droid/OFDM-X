# ============================================================
# FIGURE 8 - EXACT FCAE / OFDM-X V8 PAPR DISTRIBUTION
# ============================================================
# This script DOES NOT regenerate test data.
# It reads the exact independent-test PAPR arrays saved by V8:
#
#   V8_INDEPENDENT_TEST_PAPR_OFDM.npy
#   V8_INDEPENDENT_TEST_PAPR_FCAE.npy
#
# Therefore the histogram, mean lines, and reported PAPR values
# all refer to exactly the same 6000 independent-test samples.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["axes.titlesize"] = 9
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.dpi"] = 600

PAPR_OFDM_FILE = "V8_INDEPENDENT_TEST_PAPR_OFDM.npy"
PAPR_FCAE_FILE = "V8_INDEPENDENT_TEST_PAPR_FCAE.npy"

papr_ofdm = np.load(PAPR_OFDM_FILE)
papr_fcae = np.load(PAPR_FCAE_FILE)

mean_ofdm = float(np.mean(papr_ofdm))
mean_fcae = float(np.mean(papr_fcae))
improvement = mean_ofdm - mean_fcae

print("=" * 72)
print("FIGURE 8 - EXACT V8 INDEPENDENT-TEST DATA")
print("=" * 72)
print(f"Conventional OFDM mean PAPR = {mean_ofdm:.3f} dB")
print(f"FCAE / OFDM-X mean PAPR     = {mean_fcae:.3f} dB")
print(f"FCAE improvement             = {improvement:.3f} dB")
print(f"Number of OFDM samples      = {len(papr_ofdm)}")
print(f"Number of FCAE samples      = {len(papr_fcae)}")

xmin = min(np.min(papr_ofdm), np.min(papr_fcae))
xmax = max(np.max(papr_ofdm), np.max(papr_fcae))

bins = np.linspace(
    np.floor(xmin * 10.0) / 10.0,
    np.ceil(xmax * 10.0) / 10.0,
    50
)

fig, ax = plt.subplots(
    figsize=(3.25, 3.25),
    dpi=300
)

ax.hist(
    papr_ofdm,
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

ax.axvline(
    mean_ofdm,
    linestyle="--",
    linewidth=1.0,
    label=f"Mean OFDM = {mean_ofdm:.3f} dB"
)

ax.axvline(
    mean_fcae,
    linestyle="--",
    linewidth=1.0,
    label=f"Mean FCAE = {mean_fcae:.3f} dB"
)

ax.set_xlabel(
    "PAPR (dB)",
    fontsize=9,
    fontname="Times New Roman"
)

ax.set_ylabel(
    "Number of Symbols",
    fontsize=9,
    fontname="Times New Roman"
)

ax.tick_params(axis="both", which="major", labelsize=9)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.45,
    alpha=0.30
)

legend = ax.legend(
    loc="upper right",
    fontsize=9,
    frameon=True
)

for t in legend.get_texts():
    t.set_fontname("Times New Roman")
    t.set_fontsize(9)

for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontname("Times New Roman")
    t.set_fontsize(9)

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.17,
    top=0.98
)

out = "FCAE_OFDM_X_V8_FIG8_PAPR_DISTRIBUTION_EXACT_300dpi_TNR8.png"

fig.savefig(
    out,
    dpi=600,
    facecolor="white"
)

plt.show()

print(f"\nSaved: {out}")
print("Figure size: 3.25 x 3.25 inch")
print("Resolution: 600 dpi")
print("Font: Times New Roman")
print("All visible figure text: 9 pt")
