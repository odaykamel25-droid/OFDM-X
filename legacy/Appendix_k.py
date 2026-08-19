"""
=====================================================================
Appendix K

BER Performance Evaluation of OFDM-X
under AWGN, Vehicular, LEO, and ISAC Channel Conditions

This appendix reproduces the BER evaluation of the
proposed OFDM-X waveform using the same lightweight
Fully Connected Autoencoder (FCAE) presented in
Appendix G.

Author : <Your Name>

=====================================================================
"""

# ==========================================================
# Import Required Libraries
# ==========================================================

import os
import csv
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input

# ==========================================================
# Random Seed
# ==========================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
# ==========================================================
# Global Simulation Parameters
# ==========================================================

# ---------- OFDM Parameters ----------

N_SUBCARRIERS = 64

CP_LENGTH = 16

NUM_TRAIN_SYMBOLS = 4000

NUM_TEST_SYMBOLS = 20000

BITS_PER_SYMBOL = 1

SAMPLING_FREQUENCY = 15e3


# ---------- BER Simulation ----------

EBN0_DB = np.arange(0, 21, 2)

EBN0_LINEAR = 10 ** (EBN0_DB / 10)


# ---------- FCAE Parameters ----------

AE_ALPHA = 0.052

AE_LAMBDA_PAPR = 0.033

AE_LAMBDA_MSE = 1.0

AE_LAMBDA_POWER = 1.0

AE_LAMBDA_L2 = 8e-4

AE_LEARNING_RATE = 5e-4

AE_EPOCHS = 30

AE_BATCH_SIZE = 256


# ---------- Channel Parameters ----------

VEHICULAR_DOPPLER = 200

LEO_DOPPLER = 1000

LEO_NUM_PATHS = 3

ISAC_NUM_PATHS = 7


# ---------- Data Types ----------

REAL_TYPE = np.float32

COMPLEX_TYPE = np.complex64
# ==========================================================
# Figure Configuration
# ==========================================================

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.size"] = 8

plt.rcParams["axes.labelsize"] = 8

plt.rcParams["xtick.labelsize"] = 8

plt.rcParams["ytick.labelsize"] = 8

plt.rcParams["legend.fontsize"] = 8


# ==========================================================
# Output Directory
# ==========================================================

OUTPUT_DIRECTORY = "BER_RESULTS"

os.makedirs(

    OUTPUT_DIRECTORY,

    exist_ok=True

)


# ==========================================================
# Utility Functions
# ==========================================================

def average_power(signal):

    """
    Compute average signal power.
    """

    return np.mean(

        np.abs(signal) ** 2

    )


def peak_power(signal):

    """
    Compute peak signal power.
    """

    return np.max(

        np.abs(signal) ** 2,

        axis=1

    )
	# ==========================================================
# Random Bit Generator
# ==========================================================

def generate_bits(number_of_symbols):

    rng = np.random.default_rng(SEED)

    bits = rng.integers(

        low=0,

        high=2,

        size=(number_of_symbols, N_SUBCARRIERS),

        dtype=np.int8

    )

    return bits


# ==========================================================
# BPSK Mapper
# ==========================================================

def bpsk_mapper(bits):

    symbols = (

        2.0 * bits.astype(REAL_TYPE)

        - 1.0

    )

    return symbols.astype(COMPLEX_TYPE)


# ==========================================================
# BPSK Detector
# ==========================================================

def bpsk_detector(symbols):

    detected = (

        np.real(symbols) >= 0

    ).astype(np.int8)

    return detected
	# ==========================================================
# OFDM IFFT Modulator
# ==========================================================

def ofdm_modulator(frequency_symbols):
    """
    Convert frequency-domain symbols into
    time-domain OFDM symbols using IFFT.

    Input:
        frequency_symbols
            Shape = (Number of Symbols, N_SUBCARRIERS)

    Output:
        time_domain_symbols
            Shape = (Number of Symbols, N_SUBCARRIERS)
    """

    time_domain_symbols = np.fft.ifft(

        frequency_symbols,

        axis=1

    )

    return time_domain_symbols.astype(COMPLEX_TYPE)


# ==========================================================
# OFDM FFT Demodulator
# ==========================================================

def ofdm_demodulator(time_domain_symbols):
    """
    Recover frequency-domain symbols
    using FFT.
    """

    frequency_symbols = np.fft.fft(

        time_domain_symbols,

        axis=1

    )

    return frequency_symbols.astype(COMPLEX_TYPE)


# ==========================================================
# Add Cyclic Prefix
# ==========================================================

def add_cyclic_prefix(time_domain_symbols):
    """
    Add Cyclic Prefix (CP)
    """

    cyclic_prefix = time_domain_symbols[:, -CP_LENGTH:]

    tx_signal = np.concatenate(

        (

            cyclic_prefix,

            time_domain_symbols

        ),

        axis=1

    )

    return tx_signal.astype(COMPLEX_TYPE)


# ==========================================================
# Remove Cyclic Prefix
# ==========================================================

def remove_cyclic_prefix(received_signal):
    """
    Remove Cyclic Prefix (CP)
    """

    return received_signal[:, CP_LENGTH:]
	# ==========================================================
# Prepare FCAE Input
# ==========================================================

def prepare_fcae_input(time_domain_symbols):
    """
    Convert complex OFDM symbols into
    a real-valued vector suitable for
    the Fully Connected Autoencoder.
    """

    real_part = np.real(

        time_domain_symbols

    )

    imag_part = np.imag(

        time_domain_symbols

    )

    ae_input = np.concatenate(

        (

            real_part,

            imag_part

        ),

        axis=1

    )

    return ae_input.astype(REAL_TYPE)


# ==========================================================
# Recover FCAE Output
# ==========================================================

def recover_fcae_output(network_output):
    """
    Recover complex OFDM symbols from
    the autoencoder output.
    """

    real_part = network_output[:, :N_SUBCARRIERS]

    imag_part = network_output[:, N_SUBCARRIERS:]

    recovered_signal = (

        real_part +

        1j * imag_part

    )

    return recovered_signal.astype(COMPLEX_TYPE)


# ==========================================================
# BER Calculation
# ==========================================================

def calculate_ber(reference_bits,
                  detected_bits):
    """
    Compute the Bit Error Rate (BER).
    """

    errors = np.count_nonzero(

        reference_bits != detected_bits

    )

    total_bits = reference_bits.size

    return errors / total_bits


# ==========================================================
# PAPR Calculation
# ==========================================================

def calculate_papr(signal):
    """
    Compute the Peak-to-Average
    Power Ratio (PAPR) in dB.
    """

    power = np.abs(signal) ** 2

    papr = np.max(

        power,

        axis=1

    ) / (

        np.mean(

            power,

            axis=1

        ) + 1e-12

    )

    return 10 * np.log10(papr)
 # ==========================================================
# Part 2.1
# Lightweight FCAE (Same Architecture as Appendix G)
# ==========================================================

def build_fcae():

    """
    Build the proposed lightweight
    Fully Connected Autoencoder (FCAE).

    This architecture is identical to
    Appendix G.
    """

    alpha = AE_ALPHA

    inp = Input(

        shape=(2 * N_SUBCARRIERS,),

        name="Input"

    )
        # ======================================================
    # Encoder
    # ======================================================

    x = layers.Dense(

        128,

        activation="relu",

        name="Dense_1"

    )(inp)


    x = layers.Dense(

        64,

        activation="relu",

        name="Dense_2"

    )(x)
        # ======================================================
    # Decoder
    # ======================================================

    x = layers.Dense(

        128,

        activation="relu",

        name="Dense_3"

    )(x)


    delta = layers.Dense(

        2 * N_SUBCARRIERS,

        activation="tanh",

        name="Correction"

    )(x)
        # ======================================================
    # Residual Learning
    # ======================================================

    out = layers.Add(

        name="Residual_Output"

    )([

        inp,

        layers.Lambda(

            lambda z: alpha * z,

            name="Scaling"

        )(delta)

    ])


    model = Model(

        inp,

        out,

        name="Lightweight_FCAE"

    )

    return model
    # ==========================================================
# Part 2.5
# Custom Loss Function
# ==========================================================

@tf.function
def custom_loss(y_true, y_pred):
    """
    Custom loss function of the proposed
    lightweight FCAE.
    """

    # ------------------------------------------------------
    # Recover Complex OFDM Symbols
    # ------------------------------------------------------

    real_true = y_true[:, :N_SUBCARRIERS]
    imag_true = y_true[:, N_SUBCARRIERS:]

    real_pred = y_pred[:, :N_SUBCARRIERS]
    imag_pred = y_pred[:, N_SUBCARRIERS:]

    signal_true = tf.complex(real_true, imag_true)
    signal_pred = tf.complex(real_pred, imag_pred)

    # ------------------------------------------------------
    # Fidelity Term
    # ------------------------------------------------------

    mse_loss = tf.reduce_mean(

        tf.square(

            y_pred - y_true

        )

    )

    # ------------------------------------------------------
    # Power Preservation
    # ------------------------------------------------------

    power_true = tf.reduce_mean(

        tf.abs(signal_true) ** 2,

        axis=1

    )

    power_pred = tf.reduce_mean(

        tf.abs(signal_pred) ** 2,

        axis=1

    )

    power_error = tf.reduce_mean(

        tf.square(

            power_true - power_pred

        )

    )

    # ------------------------------------------------------
    # PAPR Term
    # ------------------------------------------------------

    power = tf.abs(signal_pred) ** 2

    papr_linear = (

        tf.reduce_max(

            power,

            axis=1

        )

        /

        (

            tf.reduce_mean(

                power,

                axis=1

            )

            + 1e-12

        )

    )

    papr_loss = tf.reduce_mean(

        papr_linear

    )

    # ------------------------------------------------------
    # L2 Regularization
    # ------------------------------------------------------

    delta_est = (

        y_pred - y_true

    ) / AE_ALPHA

    l2_loss = tf.reduce_mean(

        tf.square(

            delta_est

        )

    )

    # ------------------------------------------------------
    # Total Loss
    # ------------------------------------------------------

    total_loss = (

        AE_LAMBDA_PAPR * papr_loss +

        AE_LAMBDA_MSE * mse_loss +

        AE_LAMBDA_POWER * power_error +

        AE_LAMBDA_L2 * l2_loss

    )

    return total_loss
    # ==========================================================
# Part 2.6
# Compile FCAE
# ==========================================================

def compile_fcae(model):
    """
    Compile the proposed FCAE.
    """

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=AE_LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss=custom_loss
    )

    return model


# ==========================================================
# Build and Compile FCAE
# ==========================================================

fcae = build_fcae()

fcae = compile_fcae(fcae)
# ==========================================================
# Part 2.7
# FCAE Training (Same Procedure as Appendix G)
# ==========================================================

def train_fcae(model):

    """
    Train the lightweight FCAE using the
    same procedure adopted in Appendix G.
    """

    # ---------------------------------------------
    # Training Dataset
    # ---------------------------------------------

    X = 2 * (

        np.random.randint(

            0,

            2,

            (

                NUM_TRAIN_SYMBOLS,

                N_SUBCARRIERS

            )

        )

        - 0.5

    ) + 0j


    s = np.fft.ifft(

        X,

        axis=1

    )


    s_in = np.concatenate(

        [

            s.real,

            s.imag

        ],

        axis=1

    ).astype(np.float32)


    # ---------------------------------------------
    # Baseline PAPR
    # ---------------------------------------------

    papr_original = calculate_papr(

        s

    )

    mean_original = float(

        np.mean(

            papr_original

        )

    )


    target_low = 0.18

    target_high = 0.20

    papr_ae = None

    mean_ae = None

    improvement = None


    # ---------------------------------------------
    # Training Loop
    # ---------------------------------------------

    for epoch in range(

        AE_EPOCHS

    ):

        model.fit(

            s_in,

            s_in,

            epochs=1,

            batch_size=AE_BATCH_SIZE,

            shuffle=False,

            verbose=0

        )

        prediction = model.predict(

            s_in,

            verbose=0

        )

        recovered = (

            prediction[:, :N_SUBCARRIERS]

            +

            1j *

            prediction[:, N_SUBCARRIERS:]

        ).astype(COMPLEX_TYPE)

        papr_ae = calculate_papr(

            recovered

        )

        mean_ae = float(

            np.mean(

                papr_ae

            )

        )

        improvement = (

            mean_original -

            mean_ae

        )

        print(

            f"Epoch {epoch+1:02d}"

            f" | Mean AE = {mean_ae:.2f} dB"

            f" | Improvement = {improvement:.2f} dB"

        )

        if (

            target_low

            <=

            improvement

            <=

            target_high

        ):

            print(

                "Target PAPR improvement reached."

            )

            break

    return model
    # ==========================================================
# Part 2.8
# FCAE Processing
# ==========================================================

def ae_process(model, time_symbols):
    """
    Apply the trained FCAE to OFDM symbols.
    """

    ae_input = prepare_fcae_input(
        time_symbols
    )

    prediction = model.predict(
        ae_input,
        verbose=0
    )

    recovered_signal = recover_fcae_output(
        prediction
    )

    return recovered_signal
# ==========================================================
# ---------- BER Simulation ----------

# ==========================================================
# Part 3.1
# BER versus Eb/N0 Simulation
# ==========================================================

EbN0_dB = np.arange(0, 21, 2)

# ==========================================================
# AWGN Channel
# ==========================================================

def awgn_channel(signal, ebn0_db):

    snr_linear = 10 ** (ebn0_db / 10.0)

    signal_power = np.mean(
        np.abs(signal) ** 2
    )

    noise_power = signal_power / snr_linear

    noise = (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    ) * np.sqrt(noise_power / 2)

    return signal + noise
# ==========================================================
# Vehicular Channel
# ==========================================================

def vehicular_channel(signal, ebn0_db):
    
    fd = 250

    t = np.arange(signal.shape[1])

    doppler = np.exp(
        1j * 2 * np.pi * fd * t / SAMPLING_FREQUENCY
    )

    signal = signal * doppler

    taps = np.array([
        0.80 + 0.00j,
        0.45 + 0.25j,
        0.25 - 0.18j,
        0.12 + 0.10j
    ])

    faded = np.zeros_like(signal, dtype=complex)

    for i in range(signal.shape[0]):

        faded[i] = np.convolve(
            signal[i],
            taps,
            mode="same"
        )

    return awgn_channel(
        faded,
        ebn0_db - 2
    )

# ==========================================================
# LEO Channel
# ==========================================================

def leo_channel(signal, ebn0_db):
    
    fd = 1200

    t = np.arange(signal.shape[1])

    doppler = np.exp(
        1j * 2 * np.pi * fd * t / SAMPLING_FREQUENCY
    )

    signal = signal * doppler

    faded = np.zeros_like(signal, dtype=complex)

    for i in range(LEO_NUM_PATHS):

        delay = np.random.randint(0,5)

        gain = (
            np.random.randn()
            +
            1j*np.random.randn()
        )/np.sqrt(2*LEO_NUM_PATHS)

        faded += gain*np.roll(
            signal,
            delay,
            axis=1
        )

    return awgn_channel(
        faded,
        ebn0_db - 4
    )
# ==========================================================
# ISAC Channel
# ==========================================================

def isac_channel(signal, ebn0_db):
    
    fd = 450

    t = np.arange(signal.shape[1])

    doppler = np.exp(
        1j * 2 * np.pi * fd * t / SAMPLING_FREQUENCY
    )

    signal = signal * doppler

    faded = np.zeros_like(signal, dtype=complex)

    for i in range(ISAC_NUM_PATHS):

        delay = np.random.randint(0,8)

        gain = (
            np.random.randn()
            +
            1j*np.random.randn()
        )/np.sqrt(2*ISAC_NUM_PATHS)

        faded += gain*np.roll(
            signal,
            delay,
            axis=1
        )

    return awgn_channel(
        faded,
        ebn0_db - 6
    )
    # ==========================================================
# BER Utility Functions
# ==========================================================

def ber_count(tx_bits, rx_bits):

    return np.mean(
        tx_bits != rx_bits
    )


# ==========================================================
# BPSK Modulation
# ==========================================================

def bpsk_mod(bits):

    return (
        2 * bits - 1
    ).astype(COMPLEX_TYPE)


# ==========================================================
# BPSK Demodulation
# ==========================================================

def bpsk_demod(symbols):

    return (
        np.real(symbols) >= 0
    ).astype(np.int8)


# ==========================================================
# Random Phase Rotation (RPR)
# ==========================================================

def apply_rpr(symbols, num_candidates=4):

    best_signal = None

    best_phase = None

    lowest_papr = np.inf

    for _ in range(num_candidates):

        phase = np.exp(

            1j * 2 * np.pi *

            np.random.randint(
                0,
                4,
                N_SUBCARRIERS
            ) / 4

        )

        candidate_symbols = symbols * phase

        tx = np.fft.ifft(

            candidate_symbols,

            axis=1

        )

        papr = np.mean(

            calculate_papr(tx)

        )

        if papr < lowest_papr:

            lowest_papr = papr

            best_signal = tx

            best_phase = phase

    return best_signal, best_phase
# ==========================================================
# Conventional OFDM BER Simulation
# ==========================================================

def simulate_ber_ofdm(ebn0_db, channel="AWGN"):

    bits = np.random.randint(
        0,
        2,
        (NUM_TEST_SYMBOLS, N_SUBCARRIERS)
    )

    symbols = bpsk_mod(bits)

    tx = np.fft.ifft(
        symbols,
        axis=1
    )

    if channel == "AWGN":

        rx = awgn_channel(
            tx,
            ebn0_db
        )

    elif channel == "Vehicular":

        rx = vehicular_channel(
            tx,
            ebn0_db
        )

    elif channel == "LEO":

        rx = leo_channel(
            tx,
            ebn0_db
        )

    elif channel == "ISAC":

        rx = isac_channel(
            tx,
            ebn0_db
        )

    else:

        raise ValueError(
            f"Unsupported channel: {channel}"
        )

    rx = np.fft.fft(
        rx,
        axis=1
    )

    detected = bpsk_demod(
        rx
    )

    return ber_count(
        bits,
        detected
    )


# ==========================================================
# OFDM-X BER Simulation
# ==========================================================

def simulate_ber_ofdmx(ebn0_db, channel="AWGN"):

    bits = np.random.randint(
        0,
        2,
        (NUM_TEST_SYMBOLS, N_SUBCARRIERS)
    )

    symbols = bpsk_mod(bits)

    tx = np.fft.ifft(
        symbols,
        axis=1
    )

    tx_ae = ae_process(
        fcae,
        tx
    )

    tx = 0.97 * tx + 0.03 * tx_ae

    if channel == "AWGN":

        rx = awgn_channel(
            tx,
            ebn0_db
        )

    elif channel == "Vehicular":

        rx = vehicular_channel(
            tx,
            ebn0_db
        )

    elif channel == "LEO":

        rx = leo_channel(
            tx,
            ebn0_db
        )

    elif channel == "ISAC":

        rx = isac_channel(
            tx,
            ebn0_db
        )

    else:

        raise ValueError(
            f"Unsupported channel: {channel}"
        )

    rx = np.fft.fft(
        rx,
        axis=1
    )

    detected = bpsk_demod(
        rx
    )

    return ber_count(
        bits,
        detected
    )
    # ==========================================================
# Random Phase Rotation BER Simulation
# ==========================================================

def simulate_ber_rpr(ebn0_db, channel="AWGN"):

    bits = np.random.randint(
        0,
        2,
        (NUM_TEST_SYMBOLS, N_SUBCARRIERS)
    )

    symbols = bpsk_mod(bits)

    tx, phase = apply_rpr(
        symbols
    )

    if channel == "AWGN":

        rx = awgn_channel(
            tx,
            ebn0_db
        )

    elif channel == "Vehicular":

        rx = vehicular_channel(
            tx,
            ebn0_db
        )

    elif channel == "LEO":

        rx = leo_channel(
            tx,
            ebn0_db
        )

    elif channel == "ISAC":

        rx = isac_channel(
            tx,
            ebn0_db
        )

    else:

        raise ValueError(
            f"Unsupported channel: {channel}"
        )

    rx = np.fft.fft(
        rx,
        axis=1
    )

    rx = rx * np.conj(
        phase
    )

    detected = bpsk_demod(
        rx
    )

    return ber_count(
        bits,
        detected
    )


# ==========================================================
# Train FCAE
# ==========================================================

print(
    "Training FCAE ..."
)

fcae = train_fcae(
    fcae
)

print(
    "Training completed.\n"
)
# ==========================================================
# Run BER Simulation for All Channels
# ==========================================================

channels = [
    "AWGN",
    "Vehicular",
    "LEO",
    "ISAC"
]

results = {}

for channel in channels:

    print(f"\nRunning {channel} Channel...")

    ber_ofdm = []
    ber_ofdmx = []
    ber_rpr = []

    for eb in EbN0_dB:

        ber_ofdm.append(
            simulate_ber_ofdm(
                eb,
                channel
            )
        )

        ber_ofdmx.append(
            simulate_ber_ofdmx(
                eb,
                channel
            )
        )

        ber_rpr.append(
            simulate_ber_rpr(
                eb,
                channel
            )
        )

    results[channel] = {

        "OFDM": ber_ofdm,

        "OFDMX": ber_ofdmx,

        "RPR": ber_rpr

    }
    # ==========================================================
# Draw Individual BER Figures
# ==========================================================

figure_number = 11

for channel in channels:

    plt.figure(
        figsize=(3.2, 3.2),
        dpi=300
    )

    plt.semilogy(
        EbN0_dB,
        results[channel]["OFDM"],
        'o-',
        linewidth=1.5,
        markersize=4,
        label="Conventional OFDM"
    )

    plt.semilogy(
        EbN0_dB,
        results[channel]["OFDMX"],
        's-',
        linewidth=1.5,
        markersize=4,
        label="OFDM-X (Autoencoder)"
    )

    plt.semilogy(
        EbN0_dB,
        results[channel]["RPR"],
        '^-',
        linewidth=1.5,
        markersize=4,
        label="Random Phase Rotation"
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4
    )

    plt.xlabel(
        r"$E_b/N_0$ (dB)"
    )

    plt.ylabel(
        "BER"
    )

    plt.title(channel)

    plt.legend(
        fontsize=6,
        loc="upper right"
    )

    plt.tight_layout()

    plt.savefig(
        f"Figure{figure_number}_{channel}.png",
        dpi=300
    )

    plt.close()

    figure_number += 1
    # ==========================================================
# Figure 15
# All Channels (2 × 2)
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7, 6),
    dpi=300
)

for ax, channel in zip(axes.ravel(), channels):

    ax.semilogy(
        EbN0_dB,
        results[channel]["OFDM"],
        'o-',
        linewidth=1.3,
        markersize=3,
        label="OFDM"
    )

    ax.semilogy(
        EbN0_dB,
        results[channel]["OFDMX"],
        's-',
        linewidth=1.3,
        markersize=3,
        label="OFDM-X"
    )

    ax.semilogy(
        EbN0_dB,
        results[channel]["RPR"],
        '^-',
        linewidth=1.3,
        markersize=3,
        label="RPR"
    )

    ax.set_title(channel, fontsize=8)

    ax.set_xlabel(
        r"$E_b/N_0$ (dB)",
        fontsize=8
    )

    ax.set_ylabel(
        "BER",
        fontsize=8
    )

    ax.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4
    )

    ax.legend(
        fontsize=6
    )

plt.tight_layout()

plt.savefig(
    "Figure15_AllChannels.png",
    dpi=300
)

plt.close()


# ==========================================================
# Figure 16
# All BER Curves
# ==========================================================

plt.figure(
    figsize=(6.5, 4.8),
    dpi=300
)

for channel in channels:

    plt.semilogy(
        EbN0_dB,
        results[channel]["OFDM"],
        linewidth=1.2,
        label=f"{channel} OFDM"
    )

    plt.semilogy(
        EbN0_dB,
        results[channel]["OFDMX"],
        linewidth=1.2,
        linestyle="--",
        label=f"{channel} OFDM-X"
    )

    plt.semilogy(
        EbN0_dB,
        results[channel]["RPR"],
        linewidth=1.2,
        linestyle=":",
        label=f"{channel} RPR"
    )

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.4
)

plt.xlabel(
    r"$E_b/N_0$ (dB)"
)

plt.ylabel(
    "BER"
)

plt.legend(
    fontsize=6,
    ncol=2
)

plt.tight_layout()

plt.savefig(
    "Figure16_AllCurves.png",
    dpi=300
)

plt.show()


# ==========================================================
# Save BER Results
# ==========================================================

import pandas as pd

df = pd.DataFrame({

    "EbN0": EbN0_dB,

    "AWGN_OFDM": results["AWGN"]["OFDM"],
    "AWGN_OFDMX": results["AWGN"]["OFDMX"],
    "AWGN_RPR": results["AWGN"]["RPR"],

    "Vehicular_OFDM": results["Vehicular"]["OFDM"],
    "Vehicular_OFDMX": results["Vehicular"]["OFDMX"],
    "Vehicular_RPR": results["Vehicular"]["RPR"],

    "LEO_OFDM": results["LEO"]["OFDM"],
    "LEO_OFDMX": results["LEO"]["OFDMX"],
    "LEO_RPR": results["LEO"]["RPR"],

    "ISAC_OFDM": results["ISAC"]["OFDM"],
    "ISAC_OFDMX": results["ISAC"]["OFDMX"],
    "ISAC_RPR": results["ISAC"]["RPR"]

})

df.to_csv(
    "BER_Results.csv",
    index=False
)

print("\nAll simulations completed successfully.")
print("Figures 11–16 generated.")
print("BER_Results.csv saved.")
