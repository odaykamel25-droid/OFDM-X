"""
Appendix F

SIMULATING PAPR and Visualizing Before/After

"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
# -------------------------------
# Simulate OFDM signal (64 subcarriers, BPSK)
# -------------------------------
N = 64
X = 2 * (np.random.randint(0, 2, N) - 0.5)
s = np.fft.ifft(X)
# -------------------------------
# Autoencoder model
# -------------------------------
input_signal = Input(shape=(N,))
encoded = Dense(32, activation='relu')(input_signal)
decoded = Dense(N, activation='linear')(encoded)
autoencoder = Model(input_signal, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
# -------------------------------
# Train autoencoder on simulated data
# -------------------------------
data = np.stack([np.fft.ifft(2 * (np.random.randint(0, 2, N) - 0.5)).real for _ in range(1000)])
autoencoder.fit(data, data, epochs=20, batch_size=32, verbose=0)
# -------------------------------  
# Reduce PAPR (process signal)   
# -------------------------------
compressed_signal = autoencoder.predict(np.expand_dims(s.real, axis=0))[0]
# -------------------------------
# Plot Original vs Processed signal
# -------------------------------
plt.figure(figsize=(3.2, 3.2), dpi=300)
plt.plot(s.real[:128], label='Original $s_{OFDM}$', linewidth=1)
plt.plot(compressed_signal[:128], '--', label='Processed $s_{AE}$', linewidth=1)
plt.xlabel("Sample index (n)", fontsize=8)
plt.ylabel("Magnitude", fontsize=8)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.legend(fontsize=6, loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
