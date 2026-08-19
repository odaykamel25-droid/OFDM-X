"""
Appendix C

Python Implementation: Autoencoder for PAPR Reduction

"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
# Simulate OFDM signal (64 subcarriers, BPSK)
N = 64
X = 2 * (np.random.randint(0, 2, N) - 0.5)
s = np.fft.ifft(X)
# Autoencoder model
input_signal = Input(shape=(N,))
encoded = Dense(32, activation='relu')(input_signal)
decoded = Dense(N, activation='linear')(encoded)
autoencoder = Model(input_signal, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
# Train on simulated data
data = np.stack([np.fft.ifft(2 * (np.random.randint(0, 2, N) -0.5)).real for _ in range(1000)])
autoencoder.fit(data, data, epochs=20, batch_size=32)
# Reduce PAPR
compressed_signal=autoencoder.predict(np.expand_dims(s.real,axis=0))[0]
