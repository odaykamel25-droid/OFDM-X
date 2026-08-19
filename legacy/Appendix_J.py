"""
Appendix J

Time domain comparison (OFDM, AFDM, and OFDM-X)
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
N = 128   # number of samples
# ----- 1) Conventional OFDM -----
ofdm = np.random.randn(N) * 0.1
ofdm[::15] += 0.2  # add strong peaks to simulate PAPR
# ----- 2) AFDM (chirp-based modulation) -----
n = np.arange(N)
chirp = np.cos(2*np.pi*0.01*n**2) * 0.1  # quadratic chirp
afdm = chirp + np.random.randn(N)*0.01
# ----- 3) OFDM-X (hybrid) -----
ofdm_x = 0.5*ofdm + 0.5*afdm   # simple hybrid to illustrate
# ----- Plot -----
plt.figure(figsize=(9,4))
plt.plot(np.abs(ofdm), label="OFDM", alpha=0.8)
plt.plot(np.abs(afdm), label="AFDM", alpha=0.8)
plt.plot(np.abs(ofdm_x), label="OFDM-X", alpha=0.8)
plt.xlabel("Sample index (n)", fontsize=12)
plt.ylabel("Magnitude", fontsize=12)
#plt.title("Time-domain Signals", fontsize=10)
# legend بحجم 8 (بدون Bold)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("time_domain_ofdm_afdm_ofdmx.png", dpi=600)
plt.show()

