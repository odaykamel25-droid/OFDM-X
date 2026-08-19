import os, time, platform
import numpy as np
import tensorflow as tf

MODEL_FILE = "FCAE_OFDM_X_V8.keras"
INPUT_DIM = 128
WARMUP_RUNS = 100
TIMING_RUNS = 1000
BATCH_SIZE = 128

@tf.keras.utils.register_keras_serializable(package="FCAE")
class PeakAwareResidual(tf.keras.layers.Layer):
    def __init__(self, max_residual=0.18, peak_threshold=1.55, **kwargs):
        super().__init__(**kwargs)
        self.max_residual = float(max_residual)
        self.peak_threshold = float(peak_threshold)

    def build(self, input_shape):
        self.gain = self.add_weight(
            name="learned_peak_gain", shape=(1,),
            initializer=tf.keras.initializers.Constant(0.50),
            trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        real, imag = inputs[:, :64], inputs[:, 64:]
        magnitude = tf.sqrt(tf.square(real)+tf.square(imag)+1e-12)
        rms = tf.sqrt(tf.reduce_mean(tf.square(magnitude), axis=1, keepdims=True)+1e-12)
        excess = magnitude - self.peak_threshold*rms
        soft_peak = tf.nn.softplus(12.0*excess)/12.0
        inv_mag = 1.0/(magnitude+1e-8)
        gain = tf.nn.sigmoid(self.gain)
        cr = gain*soft_peak*real*inv_mag
        ci = gain*soft_peak*imag*inv_mag
        return tf.concat([
            real-self.max_residual*cr,
            imag-self.max_residual*ci], axis=1)

    def get_config(self):
        c = super().get_config()
        c.update({"max_residual":self.max_residual,
                  "peak_threshold":self.peak_threshold})
        return c

if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(
        f"{MODEL_FILE} was not found. Put this program beside the final V8 model.")

print("="*72)
print("FCAE V8 - STANDALONE INFERENCE LATENCY")
print("="*72)
print("Loading final FCAE V8 model...")

model = tf.keras.models.load_model(
    MODEL_FILE,
    custom_objects={"PeakAwareResidual": PeakAwareResidual},
    compile=False,
    safe_mode=False)

params = model.count_params()
print("Model loaded successfully.")
print(f"Model file       : {MODEL_FILE}")
print(f"Parameters       : {params:,}")
print(f"Input dimension  : {INPUT_DIM}")
print(f"TensorFlow       : {tf.__version__}")
print(f"Python           : {platform.python_version()}")
print(f"Platform         : {platform.platform()}")
print(f"Devices          : {tf.config.list_physical_devices()}")

if params != 148417:
    raise RuntimeError(
        f"This is NOT the Table-I FCAE V8 model: {params:,} parameters found; "
        "expected 148,417.")

np.random.seed(42)
x1 = tf.convert_to_tensor(np.random.randn(1,INPUT_DIM).astype(np.float32))
x128 = tf.convert_to_tensor(np.random.randn(BATCH_SIZE,INPUT_DIM).astype(np.float32))

@tf.function(reduce_retracing=True)
def forward(x):
    return model(x, training=False)

print("\nWarming up TensorFlow...")
for _ in range(WARMUP_RUNS):
    _ = forward(x1).numpy()
for _ in range(WARMUP_RUNS):
    _ = forward(x128).numpy()
print("Warm-up completed.")

def measure(x):
    t = np.empty(TIMING_RUNS, dtype=np.float64)
    for i in range(TIMING_RUNS):
        start = time.perf_counter()
        _ = forward(x).numpy()
        t[i] = (time.perf_counter()-start)*1000.0
    return t

print("\nMeasuring batch size = 1...")
t1 = measure(x1)
print("Measuring batch size = 128...")
t128 = measure(x128)

def stats(t):
    return {k: float(v) for k,v in {
        "mean":np.mean(t), "median":np.median(t), "std":np.std(t),
        "min":np.min(t), "max":np.max(t),
        "p95":np.percentile(t,95), "p99":np.percentile(t,99)
    }.items()}

s1, s128 = stats(t1), stats(t128)
per_symbol = s128["median"]/BATCH_SIZE
throughput = BATCH_SIZE/(s128["median"]/1000.0)

print("\n"+"="*72)
print("FINAL FCAE V8 LATENCY RESULTS")
print("="*72)
print("\nBATCH SIZE = 1")
for k,v in s1.items(): print(f"{k.capitalize():<9}: {v:.6f} ms")
print("\nBATCH SIZE = 128")
for k,v in s128.items(): print(f"{k.capitalize():<9}: {v:.6f} ms")
print(f"\nMedian latency per OFDM waveform from batch=128: {per_symbol:.6f} ms")
print(f"Approximate throughput: {throughput:.2f} OFDM waveforms/s")

out = "FCAE_V8_STANDALONE_LATENCY_RESULTS.txt"
with open(out,"w",encoding="utf-8") as f:
    f.write("FCAE V8 - STANDALONE INFERENCE LATENCY\n")
    f.write("="*72+"\n")
    f.write(f"Model: {MODEL_FILE}\nParameters: {params:,}\n")
    f.write(f"TensorFlow: {tf.__version__}\nPython: {platform.python_version()}\n")
    f.write(f"Platform: {platform.platform()}\nDevices: {tf.config.list_physical_devices()}\n\n")
    f.write("Batch size = 1\n")
    for k,v in s1.items(): f.write(f"{k}: {v:.6f} ms\n")
    f.write("\nBatch size = 128\n")
    for k,v in s128.items(): f.write(f"{k}: {v:.6f} ms\n")
    f.write(f"\nMedian latency per OFDM waveform from batch=128: {per_symbol:.6f} ms\n")
    f.write(f"Approximate throughput: {throughput:.2f} OFDM waveforms/s\n")
print(f"\nResults saved to: {out}")
