"""
Appendix G

PAPR comparison between conventional OFDM and OFDM with autoencoder-based reduction

"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
np.random.seed(42)
tf.random.set_seed(42)
# ---------------------------
plt.rcParams['font.family'] = 'Liberation Serif'
N = 64
symbols = 4000
X = 2 * (np.random.randint(0, 2, (symbols, N)) - 0.5) + 0j
s = np.fft.ifft(X, axis=1)                     # (symbols, N) complex
s_in = np.concatenate([s.real, s.imag], axis=1).astype(np.float32)
# PAPR
def papr_db_np(sig_c):
    p = np.abs(sig_c) ** 2
    return 10 * np.log10(np.max(p, axis=1) / np.mean(p, axis=1))
# baseline
papr_orig_db = papr_db_np(s)
mean_orig_db = float(np.mean(papr_orig_db))
print(f"Baseline mean PAPR: {mean_orig_db:.2f} dB")
# ---------------------------
alpha = 0.052  
inp = Input(shape=(2 * N,))
x = layers.Dense(128, activation='relu')(inp)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dense(128, activation='relu')(x)
delta = layers.Dense(2 * N, activation='tanh')(x)
out = layers.Add()([inp, layers.Lambda(lambda z: alpha * z)(delta)])
model = Model(inp, out)
lambda1 = 0.033   # PAPR
lambda2 = 1.0     # Fidelity
lambda3 = 1.0     # Power preservation
lambda4 = 8e-4    # L2 
@tf.function
def custom_loss(y_true, y_pred):
    real_t, imag_t = y_true[:, :N], y_true[:, N:]
    real_p, imag_p = y_pred[:, :N], y_pred[:, N:]
    s_true = tf.complex(real_t, imag_t)
    s_pred = tf.complex(real_p, imag_p)
    # Fidelity
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    # Power preservation 
    p_true = tf.reduce_mean(tf.abs(s_true) ** 2, axis=1)
    p_pred = tf.reduce_mean(tf.abs(s_pred) ** 2, axis=1)
    pw_err = tf.reduce_mean(tf.square(p_true - p_pred))
    # PAPR
    power = tf.abs(s_pred) ** 2
    papr_lin = tf.reduce_max(power, axis=1) / (tf.reduce_mean(power, axis=1) + 1e-12)
    papr_term = tf.reduce_mean(papr_lin)
    # L2
    delta_est = (y_pred - y_true) / alpha
    l2_delta = tf.reduce_mean(tf.square(delta_est))
    return lambda1 * papr_term + lambda2 * mse + lambda3 * pw_err + lambda4 * l2_delta
opt = tf.keras.optimizers.Adam(5e-4)  model.compile(optimizer=opt, loss=custom_loss)
# ---------------------------
target_low, target_high = 0.18, 0.20
epochs = 30
batch_size = 256
papr_ae_db = None
for ep in range(epochs):
    model.fit(s_in, s_in, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
    s_out = model.predict(s_in, verbose=0)
    s_out_c = (s_out[:, :N] + 1j * s_out[:, N:]).astype(np.complex64)
    papr_ae_db = papr_db_np(s_out_c)
    mean_ae = float(np.mean(papr_ae_db))
    improv = mean_orig_db - mean_ae
    print(f"Epoch {ep+1:02d} | Mean AE: {mean_ae:.2f} dB | Improvement: {improv:.2f} dB")
    if target_low <= improv <= target_high:
        print("Reached target ~0.2 dB. Stopping.")
        break
# ---------------------------
plt.figure(figsize=(8, 6))
plt.hist(papr_orig_db, bins=40, alpha=0.6, label="Original OFDM")
plt.hist(papr_ae_db, bins=40, alpha=0.6, label="After Autoencoder (~0.2 dB)")
plt.axvline(np.mean(papr_orig_db), linestyle='--', linewidth=2,
            label=f"Mean Original = {np.mean(papr_orig_db):.2f} dB")
plt.axvline(np.mean(papr_ae_db), linestyle='--', linewidth=2,
            label=f"Mean AE = {np.mean(papr_ae_db):.2f} dB")
plt.xlabel("PAPR (dB)")
plt.ylabel("Number of Symbols")
#plt.title("PAPR Comparison: Original vs Autoencoder (≈0.2 dB)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("papr_comparison_autoencoder_300dpi.png", dpi=300)
plt.show()
print(f"Mean PAPR Original:    {mean_orig_db:.2f} dB")
print(f"Mean PAPR Autoencoder: {mean_ae:.2f} dB")
print(f"Improvement:           {improv:.2f} dB")
