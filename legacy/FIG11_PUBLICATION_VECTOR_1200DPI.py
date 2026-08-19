# ============================================================
# FIGURE 11 - PUBLICATION QUALITY / VECTOR VERSION
# ============================================================
# Source:
#   Figure11_FINAL_REAL_BER_RESULTS.csv
#
# Curves:
#   1) Conventional OFDM
#   2) OFDM-X/FCAE
#   3) Random Phase Rotation
#
# OUTPUTS:
#   Figure11_FINAL_PUBLICATION.svg       <-- vector, unlimited zoom
#   Figure11_FINAL_PUBLICATION.pdf       <-- vector, unlimited zoom
#   Figure11_FINAL_PUBLICATION_1200dpi.png
#
# The BER data are NEVER changed.
# No jitter, offsets, smoothing, interpolation, or curve fitting.
#
# IMPORTANT:
# The script explicitly loads Times New Roman from Windows when
# available. On your Windows/VS Code machine this should produce
# real Times New Roman text.
# ============================================================

import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ============================================================
# 1. Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_NAME = "Figure11_FINAL_REAL_BER_RESULTS.csv"
CSV_PATH = SCRIPT_DIR / CSV_NAME

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"\nCSV file not found:\n{CSV_PATH}\n\n"
        "Place this script in the same folder as "
        f"{CSV_NAME}"
    )


# ============================================================
# 2. Force Times New Roman
# ============================================================

FONT_NAME = "Times New Roman"

# Windows Times New Roman font files.
# These are the standard Windows font locations.
windows_font_files = [
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
    r"C:\Windows\Fonts\timesbi.ttf",
]

for fp in windows_font_files:
    if Path(fp).exists():
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass

font_manager._load_fontmanager(try_read_cache=False)

font_names = {f.name for f in font_manager.fontManager.ttflist}

if FONT_NAME not in font_names:
    raise RuntimeError(
        "\nTimes New Roman was not found.\n"
        "Please verify that C:\\Windows\\Fonts\\times.ttf exists, "
        "then restart VS Code and run again.\n"
    )

plt.rcParams.update({
    "font.family": FONT_NAME,
    "font.serif": [FONT_NAME],
    "text.usetex": False,
    "svg.fonttype": "none",      # keep SVG text as text
    "pdf.fonttype": 42,         # TrueType font embedding
    "ps.fonttype": 42,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
})


# ============================================================
# 3. Read CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

required = [
    "Channel",
    "EbN0_dB",
    "OFDM_BER",
    "FCAE_BER",
    "RPR_BER",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
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
# 4. Publication figure size
# ============================================================
# Large physical size is important in addition to high DPI.
# The vector outputs (SVG/PDF) are resolution-independent.

FIG_W = 11.8
FIG_H = 4.55

fig, axes = plt.subplots(
    1,
    3,
    figsize=(FIG_W, FIG_H),
    sharey=True,
)


# ============================================================
# 5. Style definitions
# ============================================================
# The lines are deliberately different so overlapping curves
# remain distinguishable without altering the data.

styles = [
    {
        "column": "OFDM_BER",
        "label": "Conventional OFDM",
        "color": "tab:blue",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 1.8,
        "markersize": 6.5,
        "markevery": 1,
        "zorder": 5,
    },
    {
        "column": "FCAE_BER",
        "label": "OFDM-X/FCAE",
        "color": "tab:orange",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 1.8,
        "markersize": 6.2,
        "markevery": 1,
        "zorder": 6,
    },
    {
        "column": "RPR_BER",
        "label": "Random Phase Rotation",
        "color": "tab:green",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 1.8,
        "markersize": 6.8,
        "markevery": 1,
        "zorder": 7,
    },
]


legend_handles = []
legend_labels = []


# ============================================================
# 6. Draw panels
# ============================================================

for panel_index, (ax, channel) in enumerate(
    zip(axes, channels)
):

    sub = df[
        df["Channel"].astype(str).str.strip().str.lower()
        == channel.lower()
    ].sort_values("EbN0_dB")

    x = sub["EbN0_dB"].to_numpy()

    for style in styles:

        line, = ax.semilogy(
            x,
            sub[style["column"]].to_numpy(),

            color=style["color"],

            marker=style["marker"],
            linestyle=style["linestyle"],

            linewidth=style["linewidth"],
            markersize=style["markersize"],

            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.2,

            markevery=style["markevery"],

            solid_capstyle="round",
            solid_joinstyle="round",

            label=style["label"],
            zorder=style["zorder"],
        )

        if panel_index == 0:
            legend_handles.append(line)
            legend_labels.append(style["label"])


    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------

    ax.set_title(
        channel,
        fontname=FONT_NAME,
        fontsize=13,
        fontweight="bold",
        pad=9,
    )


    # --------------------------------------------------------
    # X axis
    # --------------------------------------------------------

    ax.set_xlabel(
        r"$E_b/N_0$ (dB)",
        fontname=FONT_NAME,
        fontsize=10.5,
        labelpad=6,
    )

    ax.set_xlim(-0.6, 20.6)
    ax.set_xticks(expected_ebn0)


    # --------------------------------------------------------
    # Y axis
    # --------------------------------------------------------

    if panel_index == 0:
        ax.set_ylabel(
            "BER",
            fontname=FONT_NAME,
            fontsize=10.5,
            labelpad=7,
        )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        True,
        which="major",
        linestyle=":",
        linewidth=0.65,
        alpha=0.65,
    )

    ax.grid(
        True,
        which="minor",
        linestyle=":",
        linewidth=0.35,
        alpha=0.40,
    )


    # --------------------------------------------------------
    # Ticks
    # --------------------------------------------------------

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=9.5,
        length=5,
        width=0.9,
    )

    for tick in ax.get_xticklabels():
        tick.set_fontname(FONT_NAME)
        tick.set_fontsize(9.5)

    for tick in ax.get_yticklabels():
        tick.set_fontname(FONT_NAME)
        tick.set_fontsize(9.5)


    # --------------------------------------------------------
    # Spines
    # --------------------------------------------------------

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


# ============================================================
# 7. Global y-limits
# ============================================================
# Use the actual data range with a small margin so the curves
# occupy more of the figure.

all_ber_values = np.concatenate([
    df["OFDM_BER"].to_numpy(),
    df["FCAE_BER"].to_numpy(),
    df["RPR_BER"].to_numpy(),
])

positive = all_ber_values[all_ber_values > 0]

y_min = positive.min()
y_max = positive.max()

# Log margins.
lower = 10 ** (
    np.floor(np.log10(y_min)) - 0.10
)
upper = 10 ** (
    np.ceil(np.log10(y_max)) + 0.05
)

axes[0].set_ylim(lower, upper)


# ============================================================
# 8. ONE shared legend
# ============================================================

fig.legend(
    legend_handles,
    legend_labels,

    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),

    ncol=3,

    frameon=True,

    prop={
        "family": FONT_NAME,
        "size": 10,
    },

    handlelength=3.0,
    handletextpad=0.7,
    columnspacing=2.0,

    borderpad=0.6,
)


# ============================================================
# 9. Layout
# ============================================================

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    top=0.87,
    bottom=0.235,
    wspace=0.08,
)


# ============================================================
# 10. Save VECTOR outputs
# ============================================================

svg_path = SCRIPT_DIR / "Figure11_FINAL_PUBLICATION.svg"
pdf_path = SCRIPT_DIR / "Figure11_FINAL_PUBLICATION.pdf"

fig.savefig(
    svg_path,
    format="svg",
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    pdf_path,
    format="pdf",
    bbox_inches="tight",
    facecolor="white",
)


# ============================================================
# 11. Save ULTRA-HIGH-RESOLUTION RASTER
# ============================================================

png_path = SCRIPT_DIR / "Figure11_FINAL_PUBLICATION_1200dpi.png"

fig.savefig(
    png_path,
    format="png",
    dpi=1200,
    bbox_inches="tight",
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 12. Print final file information
# ============================================================

print("=" * 80)
print("FIGURE 11 PUBLICATION-QUALITY EXPORT COMPLETE")
print("=" * 80)
print(f"Input CSV : {CSV_PATH}")
print()
print("Saved files:")
print(f"SVG       : {svg_path}")
print(f"PDF       : {pdf_path}")
print(f"PNG       : {png_path}")
print()
print(f"Font      : {FONT_NAME}")
print(f"Figure    : {FIG_W} x {FIG_H} inches")
print("Raster    : 1200 dpi")
print("Panels    : Vehicular | LEO | ISAC")
print("Curves    : OFDM | OFDM-X/FCAE | Random Phase Rotation")
print()
print("IMPORTANT:")
print("- The CSV data were not modified.")
print("- No jitter was introduced.")
print("- No smoothing/interpolation was used.")
print("- SVG and PDF are vector outputs and remain sharp at any zoom.")
print("=" * 80)
