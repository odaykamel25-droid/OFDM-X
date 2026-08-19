"""
Appendix A

PAPR Distribution Before and After AI Processing

This source code corresponds to Appendix A of the manuscript:

"Implementing a Hybrid AI-Driven Adaptive Waveform for Integrated Sensing and Communication in 6G Using OFDM-X"

Author:
Oday Kamil Hamid
"""
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi
x = np.linspace(6, 14, 500)
sigma = 1.0 / sqrt(2.0 * pi)   # ≈ 0.39894
mu1 = 8.6                      # (Original OFDM)
mu2 = 10.0                    # (Autoencoder Output)
def gaussian_pdf(x, mu, sigma):
 return (1.0/(sigma*sqrt(2.0*pi))) * np.exp(-0.5*((x-mu)/sigma)**2)
y1 = gaussian_pdf(x, mu1, sigma)
y2 = gaussian_pdf(x, mu2, sigma)
plt.figure(figsize=(3.2, 3.2), dpi=300)
plt.plot(x, y1, label='Original OFDM')      
plt.plot(x, y2, label='Autoencoder Output')
plt.xlabel("PAPR (dB)", fontsize=8)
plt.ylabel("Density", fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylim(0, 1.2)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(
loc='upper right',
fontsize=7,
frameon=True,
 borderpad=0.2,
 handlelength=1.0,
 handletextpad=0.3
)
plt.tight_layout()
plt.savefig("figure1.png", dpi=300)
plt.show()
