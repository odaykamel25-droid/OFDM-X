# ============================================================
# FIGURE 11 - SAVE-ONLY VERSION
# Times New Roman - 600 dpi
#
# This script does NOT rerun the BER simulation.
# It reads the already generated CSV and ALWAYS saves the
# figure to the same folder as the CSV.
#
# Required file:
#     Figure11_FINAL_REAL_BER_RESULTS.csv
#
# Output:
#     Figure11_FINAL_3CURVES_TIMES_NEW_ROMAN_600dpi.png
# ============================================================

import matplotlib

# Important for Visual Studio Code / Windows:
# use a non-interactive backend so the file is saved even if
# no graphical window is available.
matplotlib.use("Agg")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 1. Locate the CSV
# ============================================================

CSV_NAME = "Figure11_FINAL_REAL_BER_RESULTS.csv"
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / CSV_NAME

if not CSV_PATH.exists():
    raise FileNotFoundError(
        "\nCSV file not found.\n"
        f"Expected location:\n{CSV_PATH}\n\n"
        "Put this Python file in the same folder as the CSV."
    )


# ============================================================
# 2. Force Times New Roman
# ============================================================

FONT_NAME = "Times New Roman"

# Windows standard location.
windows_fonts = [
    Path(r"C:\Windows\Fonts\times.ttf"),
    Path(r"C:\Windows\Fonts\timesbd.ttf"),
    Path(r"C:\Windows\Fonts\timesi.ttf"),
    Path(r"C:\Windows\Fonts\timesbi.ttf"),
]

# Add the Windows font explicitly when running on Windows.
# This makes Matplotlib use the real Times New Roman font.
for font_path in windows_fonts:
    if font_path.exists():
        try:
            font_manager.fontManager.addfont(str(font_path))
        except Exception:
            pass

font_manager._load_fontmanager(try_read_cache=False)

# Check availability after explicit Windows registration.
available_fonts = {
    f.name for f in font_manager.fontManager.ttflist
}

if FONT_NAME not in available_fonts:
    raise RuntimeError(
        "\nTimes New Roman was not found.\n"
        "On Windows, verify that C:\\Windows\\Fonts\\times.ttf exists.\n"
    )

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.serif": [FONT_NAME],
    "font.size": 8,
})


# ============================================================
# 3. Read and validate CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

required = [
    "Channel",
    "EbN0_dB",
    "OFDM_BER",
    "FCAE_BER",
    "RPR_BER",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing CSV columns: {missing}"
    )

channels = ["Vehicular", "LEO", "ISAC"]
expected_ebn0 = np.arange(0, 21, 2)

for channel in channels:

    sub = df[
        df["Channel"].astype(str).str.strip().str.lower()
        == channel.lower()
    ].sort_values("EbN0_dB")

    if len(sub) != 11:
        raise ValueError(
            f"{channel}: expected 11 Eb/N0 points, found {len(sub)}"
        )

    if not np.array_equal(
        sub["EbN0_dB"].to_numpy(),
        expected_ebn0
    ):
        raise ValueError(
            f"{channel}: Eb/N0 values must be 0,2,...,20 dB."
        )


# ============================================================
# 4. Create Figure 11
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.2, 2.75),
    dpi=600,
    sharey=True,
)

# Distinct styles make the three curves visible even when
# numerical BER values are very close.
styles = [
    (
        "OFDM_BER",
        "Conventional OFDM",
        "o",
        "-",
        1.15,
        4.2,
    ),
    (
        "FCAE_BER",
        "OFDM-X/FCAE",
        "s",
        "--",
        1.15,
        4.1,
    ),
    (
        "RPR_BER",
        "Random Phase Rotation",
        "^",
        "-.",
        1.15,
        4.3,
    ),
]

legend_handles = []
legend_labels = []

for panel_index, (ax, channel) in enumerate(
    zip(axes, channels)
):

    sub = df[
        df["Channel"].astype(str).str.strip().str.lower()
        == channel.lower()
    ].sort_values("EbN0_dB")

    x = sub["EbN0_dB"].to_numpy()

    for (
        column,
        label,
        marker,
        linestyle,
        linewidth,
        markersize,
    ) in styles:

        line, = ax.semilogy(
            x,
            sub[column].to_numpy(),
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            markersize=markersize,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=label,
        )

        if panel_index == 0:
            legend_handles.append(line)
            legend_labels.append(label)

    ax.set_title(
        channel,
        fontname=FONT_NAME,
        fontsize=9,
        fontweight="bold",
        pad=4,
    )

    ax.set_xlabel(
        r"$E_b/N_0$ (dB)",
        fontname=FONT_NAME,
        fontsize=8,
    )

    ax.set_xticks(expected_ebn0)
    ax.set_xlim(-0.5, 20.5)

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.45,
        alpha=0.65,
    )

    ax.tick_params(
        axis="both",
        labelsize=7.5,
    )

    for tick in ax.get_xticklabels():
        tick.set_fontname(FONT_NAME)

    for tick in ax.get_yticklabels():
        tick.set_fontname(FONT_NAME)

axes[0].set_ylabel(
    "BER",
    fontname=FONT_NAME,
    fontsize=8,
)

# ============================================================
# 5. One common legend
# ============================================================

fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    frameon=True,
    prop={
        "family": FONT_NAME,
        "size": 8,
    },
    handlelength=2.0,
    handletextpad=0.6,
    columnspacing=1.4,
    borderpad=0.4,
)


# ============================================================
# 6. Layout
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.995,
    top=0.87,
    bottom=0.28,
    wspace=0.08,
)


# ============================================================
# 7. SAVE - NO plt.show()
# ============================================================

OUTPUT_NAME = (
    "Figure11_FINAL_3CURVES_TIMES_NEW_ROMAN_600dpi.png"
)

OUTPUT_PATH = SCRIPT_DIR / OUTPUT_NAME

fig.savefig(
    OUTPUT_PATH,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 8. Confirmation
# ============================================================

print("=" * 75)
print("FIGURE 11 SAVED SUCCESSFULLY")
print("=" * 75)
print(f"CSV input : {CSV_PATH}")
print(f"PNG saved : {OUTPUT_PATH}")
print("Font      : Times New Roman")
print("DPI       : 600")
print("Curves    : 3")
print("Panels    : Vehicular | LEO | ISAC")
print()
print("The figure was saved without opening a plot window.")
print("=" * 75)
