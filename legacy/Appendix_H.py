"""
Appendix H

PAPR comparison between conventional OFDM and OFDM with random phase rotation

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# -------------------------------
preferred_fonts = ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"]
available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
chosen_font = None
for font in preferred_fonts:
    if any(font in f for f in available_fonts):
        chosen_font = font
        break
if chosen_font is None:
    chosen_font = "serif"   # fallback 
plt.rcParams["font.family"] = chosen_font
plt.rcParams["font.weight"] = "bold"
np.random.seed(42)
# -------------------------------
# Parameters
# -------------------------------
N = 64                 
num_symbols = 5000   
M = 2                  # BPSK
# -------------------------------
# Function PAPR
# -------------------------------
def compute_papr(ofdm_symbol):
    power = np.abs(ofdm_symbol) ** 2
    return 10 * np.log10(np.max(power) / np.mean(power))
# -------------------------------
def clipping(ofdm_symbol, clipping_ratio=1.2):
    avg_power = np.mean(np.abs(ofdm_symbol) ** 2)
    threshold = np.sqrt(clipping_ratio * avg_power)
    clipped = np.where(np.abs(ofdm_symbol) > threshold,
                       threshold * ofdm_symbol / np.abs(ofdm_symbol),
                       ofdm_symbol)
    return clipped

# -------------------------------
papr_original = []
papr_reduced = []
for _ in range(num_symbols):
    bits = np.random.randint(0, M, N)
    symbols = 2*bits - 1   # BPSK
    ofdm_symbol = np.fft.ifft(symbols)
    papr_original.append(compute_papr(ofdm_symbol))
    # ---------------------------
    ofdm_clipped = clipping(ofdm_symbol, clipping_ratio=1.2)
    papr_reduced.append(compute_papr(ofdm_clipped))
# -------------------------------
plt.figure(figsize=(3.2, 3.2), dpi=300)
plt.hist(papr_original, bins=40, alpha=0.7,
         label="Original OFDM", color="blue", edgecolor="black")
plt.hist(papr_reduced, bins=40, alpha=0.7,
         label="Clipped OFDM (PAPR Reduced)", color="orange", edgecolor="black")
plt.xlabel("PAPR (dB)", fontsize=8)  
plt.ylabel("Number of Symbols", fontsize=8)  
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(fontsize=8, fontweight='normal')
plt.yticks(fontsize=8, fontweight='normal')
plt.legend(fontsize=4, loc="upper right", frameon=True,
           fancybox=True, facecolor="white")
plt.tight_layout()
plt.show()
