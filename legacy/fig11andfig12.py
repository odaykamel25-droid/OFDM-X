# ==========================================================
# BER Simulation for OFDM / OFDM-X
# Physical Channel Model
# ==========================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# RANDOM PHASE ROTATION (RPR) FOR BER COMPARISON
# ==========================================================
# This is used ONLY for the reviewer-requested BER comparison.
# It does not alter FCAE training, PAPR validation, or CCDF.
# The same final channel/equalization path is used as OFDM
# and OFDM-X.
# ==========================================================

RPR_NUM_CANDIDATES = 8

def apply_rpr_for_ber(symbols, rng):
    """Apply one reproducible random phase rotation per OFDM symbol."""
    n_sym, n_sub = symbols.shape
    phases = rng.choice(
        np.array([1.0, -1.0, 1.0j, -1.0j], dtype=np.complex128),
        size=(n_sym, 1)
    )
    return symbols * phases, phases

import csv

# ==========================================================
# FINAL REPRODUCIBILITY CONTROLS
# ==========================================================
# These controls do NOT change the FCAE architecture, loss,
# channel model, Doppler settings, or simulation parameters.
# They only make model initialization/training reproducible
# and preserve the exact trained model used for the final
# PAPR/CCDF/BER results.
# ==========================================================

REPRO_SEED = 20260808

os.environ["PYTHONHASHSEED"] = str(REPRO_SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

np.random.seed(REPRO_SEED)
random.seed(REPRO_SEED)
tf.random.set_seed(REPRO_SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

from tensorflow.keras import layers, Model, Input

np.random.seed(42)
tf.random.set_seed(42)

# ==========================================================
# System Parameters
# ==========================================================

N = 64                  # Number of subcarriers

CP = 16                 # Cyclic Prefix

NUM_SYMBOLS = 5000      # Monte Carlo symbols

EbN0_dB = np.arange(0,22,2)

# Subcarrier spacing = 15 kHz; sampling rate for N=64 is N*Delta_f
SUBCARRIER_SPACING = 15000
Fs = N * SUBCARRIER_SPACING

# ==========================================================
# Channel Parameters
# ==========================================================

VEHICULAR_FD = 250

LEO_FD = 1200

ISAC_FD = 450

K_FACTOR = 8

VEHICULAR_PATHS = 4

LEO_PATHS = 3

ISAC_PATHS = 7

# ==========================================================
# Pilot Parameters
# ==========================================================

PILOT_SPACING = 4

pilot_index = np.arange(
    0,
    N,
    PILOT_SPACING
)

data_index = np.setdiff1d(
    np.arange(N),
    pilot_index
)
# ==========================================================
# BPSK Mapper / Demapper
# ==========================================================

def bpsk_mod(bits):

    return (2 * bits - 1).astype(np.complex128)


def bpsk_demod(symbols):

    return (np.real(symbols) >= 0).astype(np.int8)


# ==========================================================
# BER
# ==========================================================

def ber_count(tx_bits, rx_bits):

    return np.mean(tx_bits != rx_bits)


# ==========================================================
# OFDM Modulator
# ==========================================================

def ofdm_modulate(symbols):

    time_signal = np.fft.ifft(
        symbols,
        axis=1
    )

    cp = time_signal[:, -CP:]

    tx = np.concatenate(
        (cp, time_signal),
        axis=1
    )

    return tx


# ==========================================================
# OFDM Demodulator
# ==========================================================

def ofdm_demodulate(rx):

    rx = rx[:, CP:]

    return np.fft.fft(
        rx,
        axis=1
    )


# ==========================================================
# AWGN
# ==========================================================

def awgn(signal, ebn0_db):

    snr = 10 ** (ebn0_db / 10)

    signal_power = np.mean(
        np.abs(signal) ** 2
    )

    noise_power = signal_power / snr

    noise = (

        np.random.randn(*signal.shape)

        +

        1j * np.random.randn(*signal.shape)

    ) * np.sqrt(noise_power / 2)

    return signal + noise
# ==========================================================
# Pilot Insertion
# ==========================================================

def create_ofdm_symbol(bits):

    symbol = np.zeros(
        (bits.shape[0], N),
        dtype=np.complex128
    )

    symbol[:, pilot_index] = 1 + 0j

    symbol[:, data_index] = bpsk_mod(bits)

    return symbol


# ==========================================================
# Pilot Extraction
# ==========================================================

def extract_data(received):

    return received[:, data_index]


# ==========================================================
# LS Channel Estimation
# ==========================================================

def ls_channel_estimation(received):

    pilots = received[:, pilot_index]

    H_pilot = pilots / (1 + 0j)

    H = np.zeros(
        (received.shape[0], N),
        dtype=np.complex128
    )

    for i in range(received.shape[0]):

        H[i] = np.interp(

            np.arange(N),

            pilot_index,

            H_pilot[i].real

        ) + 1j * np.interp(

            np.arange(N),

            pilot_index,

            H_pilot[i].imag

        )

    return H


# ==========================================================
# Zero Forcing Equalizer
# ==========================================================

def zf_equalizer(received, H):

    return received / (H + 1e-12)
# ==========================================================
# AWGN Channel
# ==========================================================

def awgn_channel(tx, ebn0_db):

    return awgn(tx, ebn0_db)


# ==========================================================
# Rayleigh Channel
# ==========================================================

# ==========================================================
# Vehicular Rayleigh Channel + Block Doppler
# ==========================================================

def rayleigh_channel(tx, ebn0_db, num_paths, fd):

    num_symbols = tx.shape[0]
    total_len = tx.shape[1]

    # ------------------------------------------------------
    # Initial Rayleigh path gains
    # ------------------------------------------------------

    taps0 = (
        np.random.randn(
            num_symbols,
            num_paths
        )
        +
        1j * np.random.randn(
            num_symbols,
            num_paths
        )
    ) / np.sqrt(2 * num_paths)

    # ------------------------------------------------------
    # Independent Doppler frequencies for each path
    # ------------------------------------------------------

    doppler_freq = np.random.uniform(
        -fd,
        fd,
        num_paths
    )

    # OFDM symbol duration including CP

    symbol_time = (
        N + CP
    ) / Fs

    symbol_index = np.arange(
        num_symbols
    )

    # ------------------------------------------------------
    # Doppler evolution
    # ------------------------------------------------------

    doppler_phase = np.exp(
        1j
        * 2
        * np.pi
        * symbol_index[:, None]
        * symbol_time
        * doppler_freq[None, :]
    )

    taps = taps0 * doppler_phase

    # ------------------------------------------------------
    # Normalize channel power
    # ------------------------------------------------------

    power = np.sum(
        np.abs(taps) ** 2,
        axis=1,
        keepdims=True
    )

    taps = taps / np.sqrt(
        power + 1e-12
    )

    # ------------------------------------------------------
    # Multipath delays within CP
    # ------------------------------------------------------

    delays = np.arange(
        num_paths
    )

    # ------------------------------------------------------
    # Apply block-fading multipath channel
    # ------------------------------------------------------

    faded = np.zeros_like(
        tx,
        dtype=np.complex128
    )

    for i in range(num_symbols):

        for p in range(num_paths):

            d = delays[p]

            if d == 0:

                faded[i] += (
                    taps[i, p]
                    *
                    tx[i]
                )

            else:

                shifted = np.zeros(
                    total_len,
                    dtype=np.complex128
                )

                shifted[d:] = tx[
                    i,
                    :-d
                ]

                faded[i] += (
                    taps[i, p]
                    *
                    shifted
                )

    # ------------------------------------------------------
    # Frequency response for each OFDM symbol
    # ------------------------------------------------------

    H_freq = np.fft.fft(
        taps,
        n=N,
        axis=1
    )

    # ------------------------------------------------------
    # Receiver-compatible channel array
    # ------------------------------------------------------

    h = np.zeros(
        (
            num_symbols,
            total_len
        ),
        dtype=np.complex128
    )

    h[:, CP:] = H_freq

    # ------------------------------------------------------
    # Add AWGN
    # ------------------------------------------------------

    rx = awgn(
        faded,
        ebn0_db
    )

    return rx, h
# ==========================================================
# Vehicular Rayleigh Channel with True Intra-Symbol Doppler
# Diagnostic ICI model
# ==========================================================

def rayleigh_channel_ici(
    tx,
    ebn0_db,
    num_paths,
    fd
):

    num_symbols = tx.shape[0]
    total_len = tx.shape[1]

    # ------------------------------------------------------
    # Independent Rayleigh path gains
    # ------------------------------------------------------

    taps = (
        np.random.randn(
            num_symbols,
            num_paths
        )
        +
        1j * np.random.randn(
            num_symbols,
            num_paths
        )
    ) / np.sqrt(2 * num_paths)

    delays = np.arange(num_paths)

    # ------------------------------------------------------
    # True intra-symbol Doppler
    # ------------------------------------------------------

    t = np.arange(total_len) / Fs

    fd_paths = np.random.uniform(
        -fd,
        fd,
        num_paths
    )

    faded = np.zeros_like(
        tx,
        dtype=np.complex128
    )

    # Time-varying channel taps:
    # h_i[n,p] = path gain at sample n for path p.
    h_taps = np.zeros(
        (
            num_symbols,
            total_len,
            num_paths
        ),
        dtype=np.complex128
    )

    for i in range(num_symbols):

        for p in range(num_paths):

            d = delays[p]

            doppler = np.exp(
                1j
                * 2
                * np.pi
                * fd_paths[p]
                * t
            )

            path_gain = taps[i, p] * doppler

            h_taps[i, :, p] = path_gain

            if d == 0:

                faded[i] += (
                    path_gain
                    * tx[i]
                )

            else:

                faded[i, d:] += (
                    path_gain[d:]
                    * tx[i, :-d]
                )

    # ------------------------------------------------------
    # Add AWGN
    # ------------------------------------------------------

    rx = awgn(
        faded,
        ebn0_db
    )

    # ------------------------------------------------------
    # Diagonal frequency response for one-tap EQ
    #
    # This is ONLY the diagonal part of the time-varying
    # channel. The off-diagonal terms are the residual ICI.
    # Therefore this test intentionally does not remove all ICI.
    # ------------------------------------------------------

    n_use = np.arange(
        CP,
        CP + N
    )

    phase = np.exp(
        -1j
        * 2
        * np.pi
        * np.arange(N)[:, None]
        * delays[None, :]
        / N
    )

    instantaneous_h = h_taps[
        :,
        n_use,
        :
    ]

    diagonal_gain = np.einsum(
        "snp,kp->snk",
        instantaneous_h,
        phase
    )

    H_diag = np.mean(
        diagonal_gain,
        axis=1
    )

    # H_diag shape is exactly (NUM_SYMBOLS, N).
    return rx, H_diag


# ==========================================================
# LEO Rician 3-Path Channel
# Multipath only - Doppler disabled
# ==========================================================

def rician_channel(tx, ebn0_db):

    num_symbols = tx.shape[0]
    total_len = tx.shape[1]

    # ------------------------------------------------------
    # Rician K-factor
    # ------------------------------------------------------

    los = np.sqrt(
        K_FACTOR / (K_FACTOR + 1)
    )

    nlos = np.sqrt(
        1 / (K_FACTOR + 1)
    )

    # ------------------------------------------------------
    # Generate LEO multipath taps
    # ------------------------------------------------------

    taps = (
        np.random.randn(
            num_symbols,
            LEO_PATHS
        )
        +
        1j * np.random.randn(
            num_symbols,
            LEO_PATHS
        )
    ) / np.sqrt(2)

    taps *= (
        nlos /
        np.sqrt(LEO_PATHS)
    )

    # ------------------------------------------------------
    # LOS component
    # ------------------------------------------------------

    taps[:, 0] += los

    # ------------------------------------------------------
    # Normalize total channel power
    # ------------------------------------------------------

    power = np.sum(
        np.abs(taps) ** 2,
        axis=1,
        keepdims=True
    )

    taps = taps / np.sqrt(
        power + 1e-12
    )

    # ------------------------------------------------------
    # Apply multipath channel
    # ------------------------------------------------------

    faded = np.zeros_like(
        tx,
        dtype=np.complex128
    )

    for i in range(num_symbols):

        faded[i] = np.convolve(
            tx[i],
            taps[i],
            mode="full"
        )[:total_len]

    # ------------------------------------------------------
    # Frequency response
    # ------------------------------------------------------

    H_freq = np.fft.fft(
        taps,
        n=N,
        axis=1
    )

    # ------------------------------------------------------
    # Receiver-compatible channel array
    # ------------------------------------------------------

    h = np.zeros(
        (
            num_symbols,
            total_len
        ),
        dtype=np.complex128
    )

    h[:, CP:] = H_freq

    # ------------------------------------------------------
    # AWGN
    # ------------------------------------------------------

    rx = awgn(
        faded,
        ebn0_db
    )

    return rx, h
# ISAC Channel
# ==========================================================

def isac_channel(tx, ebn0_db):

    return rayleigh_channel(

        tx,

        ebn0_db,

        ISAC_PATHS,

        ISAC_FD

    )


# ==========================================================
# Channel Selector
# ==========================================================

def apply_channel(tx, ebn0_db, channel):

    # Main BER curves use the block-fading channel models below.
    # The true intra-symbol ICI model is evaluated separately in
    # rayleigh_channel_ici() and the diagnostic section at the end.

    if channel=="AWGN":

        rx = awgn_channel(
            tx,
            ebn0_db
        )

        h = np.ones_like(tx)

    elif channel=="Vehicular":

        # Use the true time-varying Doppler channel here.
        # The receiver equalizes only the diagonal response,
        # so the off-diagonal Doppler terms remain as ICI.
        rx,h = rayleigh_channel_ici(

            tx,

            ebn0_db,

            VEHICULAR_PATHS,

            VEHICULAR_FD

        )

    elif channel=="LEO":

        rx,h = rician_channel(

            tx,

            ebn0_db

        )

    elif channel=="ISAC":

        rx,h = isac_channel(

            tx,

            ebn0_db

        )

    else:

        raise ValueError(
            "Unknown channel"
        )

    return rx,h
# ==========================================================
# FCAE PARAMETERS
# ==========================================================

AE_ALPHA = 0.052

AE_LAMBDA_PAPR = 0.033
AE_LAMBDA_MSE = 1.0
AE_LAMBDA_POWER = 1.0
AE_LAMBDA_L2 = 8e-4

AE_LEARNING_RATE = 5e-4
AE_EPOCHS = 30
AE_BATCH_SIZE = 256
AE_TRAIN_SEED = 20260808
AE_VALIDATION_SEED = 20260809


# ==========================================================
# FCAE INPUT / OUTPUT
# ==========================================================

def prepare_fcae_input(time_symbols):

    real_part = np.real(
        time_symbols
    )

    imag_part = np.imag(
        time_symbols
    )

    return np.concatenate(
        (
            real_part,
            imag_part
        ),
        axis=1
    ).astype(
        np.float32
    )


def recover_fcae_output(network_output):

    real_part = network_output[
        :,
        :N
    ]

    imag_part = network_output[
        :,
        N:
    ]

    return (
        real_part
        +
        1j * imag_part
    ).astype(
        np.complex128
    )


# ==========================================================
# FCAE MODEL
# ==========================================================

def build_fcae():

    inp = Input(
        shape=(2 * N,)
    )

    x = layers.Dense(
        128,
        activation="relu"
    )(inp)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    delta = layers.Dense(
        2 * N,
        activation="tanh"
    )(x)

    scaled_delta = layers.Lambda(
        lambda z: AE_ALPHA * z
    )(delta)

    out = layers.Add()(
        [
            inp,
            scaled_delta
        ]
    )

    return Model(
        inp,
        out
    )

# ==========================================================
# FCAE LOSS
# ==========================================================

def fcae_loss(y_true, y_pred):
    
    # ======================================================
    # Recover complex time-domain signals
    # ======================================================

    real_t = y_true[:, :N]
    imag_t = y_true[:, N:]

    real_p = y_pred[:, :N]
    imag_p = y_pred[:, N:]

    s_true = tf.complex(
        real_t,
        imag_t
    )

    s_pred = tf.complex(
        real_p,
        imag_p
    )

    # ======================================================
    # 1) Time-domain fidelity
    # ======================================================

    mse = tf.reduce_mean(
        tf.square(
            y_pred - y_true
        )
    )

    # ======================================================
    # 2) Power preservation
    # ======================================================

    p_true = tf.reduce_mean(
        tf.abs(s_true) ** 2,
        axis=1
    )

    p_pred = tf.reduce_mean(
        tf.abs(s_pred) ** 2,
        axis=1
    )

    power_error = tf.reduce_mean(
        tf.square(
            p_true - p_pred
        )
    )

    # ======================================================
    # 3) PAPR reduction
    # ======================================================

    power = tf.abs(s_pred) ** 2

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

    # ======================================================
    # 4) Frequency-domain fidelity
    #    Preserve OFDM subcarrier information
    # ======================================================

    S_true = tf.signal.fft(
        tf.cast(
            s_true,
            tf.complex64
        )
    )

    S_pred = tf.signal.fft(
        tf.cast(
            s_pred,
            tf.complex64
        )
    )

    fd_error = tf.reduce_mean(
        tf.abs(
            S_pred - S_true
        ) ** 2
    )

    fd_reference = (
        tf.reduce_mean(
            tf.abs(S_true) ** 2
        )
        + 1e-12
    )

    fd_mse = (
        fd_error
        /
        fd_reference
    )

    # ======================================================
    # 5) L2 residual
    # ======================================================

    delta_est = (
        y_pred - y_true
    ) / AE_ALPHA

    l2_loss = tf.reduce_mean(
        tf.square(
            delta_est
        )
    )

    # ======================================================
    # TOTAL FCAE LOSS
    # ======================================================

    return (
        AE_LAMBDA_PAPR * papr_loss
        +
        AE_LAMBDA_MSE * mse
        +
        AE_LAMBDA_POWER * power_error
        +
        AE_LAMBDA_L2 * l2_loss
        +
        5.0 * fd_mse
    )
    # ==========================================================
# BUILD + COMPILE
# ==========================================================

# FINAL_SEED_BEFORE_MODEL
np.random.seed(REPRO_SEED)
random.seed(REPRO_SEED)
tf.random.set_seed(REPRO_SEED)

fcae = build_fcae()

fcae.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=AE_LEARNING_RATE
    ),
    loss=fcae_loss
)
def train_fcae(model):

    # ------------------------------------------------------
    # Fixed FCAE training set for reproducible PAPR results
    # ------------------------------------------------------

    rng = np.random.default_rng(AE_TRAIN_SEED)

    X = (
        2
        * (
            rng.integers(
                0,
                2,
                (NUM_SYMBOLS, N)
            )
            - 0.5
        )
    ).astype(np.complex64)

    s = np.fft.ifft(
        X,
        axis=1
    )

    s_in = prepare_fcae_input(
        s
    )

    # ------------------------------------------------------
    # Training loop
    # Shuffle is disabled so the reported PAPR trajectory is
    # reproducible and independent of batch reordering.
    # ------------------------------------------------------

    for epoch in range(AE_EPOCHS):

        model.fit(
            s_in,
            s_in,
            epochs=1,
            batch_size=AE_BATCH_SIZE,
            shuffle=False,
            verbose=0
        )

        if (
            (epoch + 1) % 5 == 0
            or epoch == 0
        ):

            prediction = model.predict(
                s_in,
                verbose=0
            )

            recovered = recover_fcae_output(
                prediction
            )

            papr_original = np.mean(
                10
                * np.log10(
                    np.max(
                        np.abs(s) ** 2,
                        axis=1
                    )
                    /
                    (
                        np.mean(
                            np.abs(s) ** 2,
                            axis=1
                        )
                        + 1e-12
                    )
                )
            )

            papr_ae = np.mean(
                10
                * np.log10(
                    np.max(
                        np.abs(recovered) ** 2,
                        axis=1
                    )
                    /
                    (
                        np.mean(
                            np.abs(recovered) ** 2,
                            axis=1
                        )
                        + 1e-12
                    )
                )
            )

            print(
                f"FCAE Epoch {epoch+1:02d}"
                f" | Original PAPR = {papr_original:.3f} dB"
                f" | AE PAPR = {papr_ae:.3f} dB"
                f" | Improvement = "
                f"{papr_original - papr_ae:.3f} dB"
            )

    # ------------------------------------------------------
    # Independent validation set
    # This is NOT used for training or model selection.
    # It checks that the PAPR reduction is not only a
    # training-set effect.
    # ------------------------------------------------------

    val_rng = np.random.default_rng(
        AE_VALIDATION_SEED
    )

    X_val = (
        2
        * (
            val_rng.integers(
                0,
                2,
                (NUM_SYMBOLS, N)
            )
            - 0.5
        )
    ).astype(np.complex64)

    s_val = np.fft.ifft(
        X_val,
        axis=1
    )

    val_input = prepare_fcae_input(
        s_val
    )

    val_prediction = model.predict(
        val_input,
        verbose=0
    )

    val_recovered = recover_fcae_output(
        val_prediction
    )

    val_papr_original = np.mean(
        10
        * np.log10(
            np.max(
                np.abs(s_val) ** 2,
                axis=1
            )
            /
            (
                np.mean(
                    np.abs(s_val) ** 2,
                    axis=1
                )
                + 1e-12
            )
        )
    )

    val_papr_ae = np.mean(
        10
        * np.log10(
            np.max(
                np.abs(val_recovered) ** 2,
                axis=1
            )
            /
            (
                np.mean(
                    np.abs(val_recovered) ** 2,
                    axis=1
                )
                + 1e-12
            )
        )
    )

    print(
        "FCAE Validation"
        f" | Original PAPR = {val_papr_original:.3f} dB"
        f" | AE PAPR = {val_papr_ae:.3f} dB"
        f" | Improvement = "
        f"{val_papr_original - val_papr_ae:.3f} dB"
    )

    # Return the exact independent-validation data and FCAE output
    # used for the reported validation PAPR. This prevents the
    # subsequent CCDF calculation from regenerating a second
    # validation realization.
    return model, X_val, s_val, val_recovered

# ==========================================================
# TRAIN FCAE ONCE
# ========================================================

fcae, X_val_final, s_val_final, val_recovered_final = train_fcae(fcae)

# ==========================================================
# FINAL VALIDATION PAPR REPORT
# ==========================================================
def papr_mean_db_final(signals):
    p = np.abs(signals) ** 2
    return float(
        np.mean(
            10.0
            * np.log10(
                np.max(p, axis=1)
                /
                (np.mean(p, axis=1) + 1e-12)
            )
        )
    )

final_original_papr = papr_mean_db_final(s_val_final)
final_fcae_papr = papr_mean_db_final(val_recovered_final)
final_papr_improvement = final_original_papr - final_fcae_papr

print("\nFINAL REPRODUCIBLE VALIDATION RESULT:")
print(f"Original PAPR = {final_original_papr:.6f} dB")
print(f"FCAE PAPR     = {final_fcae_papr:.6f} dB")
print(f"Improvement   = {final_papr_improvement:.6f} dB")

# ==========================================================
# SAVE FINAL FCAE MODEL + EXACT VALIDATION DATA
# ==========================================================
fcae.save("FCAE_FINAL_MODEL.keras")

np.savez(
    "FCAE_FINAL_VALIDATION_DATA.npz",
    X_val=X_val_final,
    s_val=s_val_final,
    val_recovered=val_recovered_final
)

print("\nFinal FCAE model saved: FCAE_FINAL_MODEL.keras")
print("Exact validation data saved: FCAE_FINAL_VALIDATION_DATA.npz")

# ==========================================================
# FCAE PROCESSING
# ==========================================================

def ae_process(
    model,
    time_symbols
):

    ae_input = prepare_fcae_input(
        time_symbols
    )

    prediction = model.predict(
        ae_input,
        verbose=0
    )

    return recover_fcae_output(
        prediction
    )


def simulate_ber_rpr_final(ebn0_db, channel, rng):
    """
    BER for Random Phase Rotation using the SAME final OFDM
    channel/demodulation/equalization path as the accepted BER
    simulation.

    The common phase rotation is applied to the complete OFDM
    frequency-domain symbol (including pilots). The LS channel
    estimator therefore observes the rotation through the pilots,
    so no separate phase-removal step is required after ZF.
    """
    bits = rng.integers(
        0,
        2,
        (NUM_SYMBOLS, len(data_index))
    )

    tx_symbol = create_ofdm_symbol(bits)

    # One reproducible random common phase per OFDM symbol.
    # The phase is applied to the whole OFDM symbol, including pilots.
    phase_set = np.array(
        [1.0, -1.0, 1.0j, -1.0j],
        dtype=np.complex128
    )
    phases = rng.choice(
        phase_set,
        size=(NUM_SYMBOLS, 1)
    )

    tx_symbol_rpr = tx_symbol * phases

    # Exactly the same OFDM modulation path as conventional OFDM.
    tx = ofdm_modulate(tx_symbol_rpr)

    # EXACT final channel interface:
    # apply_channel(tx, Eb/N0, channel)
    rx, h = apply_channel(
        tx,
        ebn0_db,
        channel
    )

    # EXACT final receiver path.
    rx = ofdm_demodulate(rx)

    H_est = ls_channel_estimation(rx)

    rx = zf_equalizer(
        rx,
        H_est
    )

    data = extract_data(rx)

    detected = bpsk_demod(data)

    return ber_count(
        bits,
        detected
    )



def simulate_ber_ofdm(
    ebn0_db,
    channel
):

    bits = np.random.randint(
        0,
        2,
        (
            NUM_SYMBOLS,
            len(data_index)
        )
    )

    tx_symbol = create_ofdm_symbol(
        bits
    )

    # Conventional OFDM: IFFT + CP, with NO FCAE.
    tx = ofdm_modulate(
        tx_symbol
    )

    rx, h = apply_channel(
        tx,
        ebn0_db,
        channel
    )

    rx = ofdm_demodulate(
        rx
    )

    H_est = ls_channel_estimation(
        rx
    )

    rx = zf_equalizer(
        rx,
        H_est
    )

    data = extract_data(
        rx
    )

    detected = bpsk_demod(
        data
    )

    return ber_count(
        bits,
        detected
    )


# ==========================================================
# OFDM-X BER
# ==========================================================

def simulate_ber_ofdmx(
    ebn0_db,
    channel
):

    bits = np.random.randint(
        0,
        2,
        (
            NUM_SYMBOLS,
            len(data_index)
        )
    )

    tx_symbol = create_ofdm_symbol(
        bits
    )

    # FCAE expects exactly N useful time-domain samples.
    tx = np.fft.ifft(
        tx_symbol,
        axis=1
    )

    # Apply the trained FCAE before adding the CP.
    tx = ae_process(
        fcae,
        tx
    )

    # Add CP after FCAE processing.
    cp = tx[:, -CP:]

    tx = np.concatenate(
        (
            cp,
            tx
        ),
        axis=1
    )

    rx, h = apply_channel(
        tx,
        ebn0_db,
        channel
    )

    rx = ofdm_demodulate(
        rx
    )

    H_est = ls_channel_estimation(
        rx
    )

    rx = zf_equalizer(
        rx,
        H_est
    )

    data = extract_data(
        rx
    )

    detected = bpsk_demod(
        data
    )

    return ber_count(
        bits,
        detected
    )


# ==========================================================
# Run BER Simulation
# ==========================================================

channels = [
    "AWGN",
    "Vehicular",
    "LEO",
    "ISAC"
]

results = {}

for channel in channels:

    print(
        f"\nRunning {channel} Channel ..."
    )

    ber_ofdm = []
    ber_ofdmx = []

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

    results[channel] = {
        "OFDM": ber_ofdm,
        "OFDMX": ber_ofdmx
    }

print(
    "\nBER Simulation Completed."
)


# ==========================================================
# BASIC OFDM + AWGN CHECK
# ==========================================================

print(
    "\nBasic OFDM-AWGN check:"
)

for eb in EbN0_dB:

    ber = simulate_ber_ofdm(
        eb,
        "AWGN"
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )


# ==========================================================
# LEO OFDM CHECK
# ==========================================================

print(
    "\nLEO OFDM check:"
)

for eb in EbN0_dB:

    ber = simulate_ber_ofdm(
        eb,
        "LEO"
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )


# ==========================================================
# VEHICULAR OFDM CHECK
# ==========================================================

print(
    "\nVehicular OFDM check:"
)

for eb in EbN0_dB:

    ber = simulate_ber_ofdm(
        eb,
        "Vehicular"
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )


# ==========================================================
# ISAC OFDM CHECK
# ==========================================================

print(
    "\nISAC OFDM check:"
)

for eb in EbN0_dB:

    ber = simulate_ber_ofdm(
        eb,
        "ISAC"
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )


# ==========================================================
# CURRENT OFDM-X CHECK
# ==========================================================

print(
    "\nVehicular OFDM-X check:"
)

for eb in EbN0_dB:

    ber = simulate_ber_ofdmx(
        eb,
        "Vehicular"
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )


# ==========================================================
# FAIR VEHICULAR DOPPLER TEST
# ==========================================================

print(
    "\nFair Vehicular Doppler test:"
)

for eb in EbN0_dB:

    bits = np.random.randint(
        0,
        2,
        (
            NUM_SYMBOLS,
            len(data_index)
        )
    )

    tx_symbol = create_ofdm_symbol(
        bits
    )

    tx = ofdm_modulate(
        tx_symbol
    )

    np.random.seed(12345)

    rx0, h0 = rayleigh_channel_ici(
        tx,
        eb,
        VEHICULAR_PATHS,
        0
    )

    rx0 = ofdm_demodulate(
        rx0
    )

    H0 = ls_channel_estimation(
        rx0
    )

    rx0 = zf_equalizer(
        rx0,
        H0
    )

    data0 = extract_data(
        rx0
    )

    detected0 = bpsk_demod(
        data0
    )

    ber0 = ber_count(
        bits,
        detected0
    )

    np.random.seed(12345)

    rx_fd, h_fd = rayleigh_channel_ici(
        tx,
        eb,
        VEHICULAR_PATHS,
        VEHICULAR_FD
    )

    rx_fd = ofdm_demodulate(
        rx_fd
    )

    H_fd = ls_channel_estimation(
        rx_fd
    )

    rx_fd = zf_equalizer(
        rx_fd,
        H_fd
    )

    data_fd = extract_data(
        rx_fd
    )

    detected_fd = bpsk_demod(
        data_fd
    )

    ber_fd = ber_count(
        bits,
        detected_fd
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  fd=0 = {ber0:.6e}"
        f"  fd={VEHICULAR_FD} = {ber_fd:.6e}"
    )


# ==========================================================
# VEHICULAR PERFECT CSI CHECK
# ==========================================================

print(
    "\nVehicular Perfect CSI check:"
)

for eb in EbN0_dB:

    bits = np.random.randint(
        0,
        2,
        (
            NUM_SYMBOLS,
            len(data_index)
        )
    )

    tx_symbol = create_ofdm_symbol(
        bits
    )

    tx = ofdm_modulate(
        tx_symbol
    )

    rx, h = apply_channel(
        tx,
        eb,
        "Vehicular"
    )

    rx = ofdm_demodulate(
        rx
    )

    H_perfect = h if h.shape[1] == N else h[:, CP:]

    rx = zf_equalizer(
        rx,
        H_perfect
    )

    data = extract_data(
        rx
    )

    detected = bpsk_demod(
        data
    )

    ber = ber_count(
        bits,
        detected
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )


# ==========================================================
# FAIR VEHICULAR CSI COMPARISON
# ==========================================================

print(
    "\nFair Vehicular CSI comparison:"
)

for eb in EbN0_dB:

    bits = np.random.randint(
        0,
        2,
        (
            NUM_SYMBOLS,
            len(data_index)
        )
    )

    tx_symbol = create_ofdm_symbol(
        bits
    )

    tx = ofdm_modulate(
        tx_symbol
    )

    rx, h = apply_channel(
        tx,
        eb,
        "Vehicular"
    )

    rx = ofdm_demodulate(
        rx
    )

    # Perfect CSI
    H_perfect = h if h.shape[1] == N else h[:, CP:]

    rx_perfect = zf_equalizer(
        rx.copy(),
        H_perfect
    )

    data_perfect = extract_data(
        rx_perfect
    )

    detected_perfect = bpsk_demod(
        data_perfect
    )

    ber_perfect = ber_count(
        bits,
        detected_perfect
    )

    # LS CSI
    H_ls = ls_channel_estimation(
        rx
    )

    rx_ls = zf_equalizer(
        rx.copy(),
        H_ls
    )

    data_ls = extract_data(
        rx_ls
    )

    detected_ls = bpsk_demod(
        data_ls
    )

    ber_ls = ber_count(
        bits,
        detected_ls
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  LS = {ber_ls:.6e}"
        f"  Perfect CSI = {ber_perfect:.6e}"
    )


# ==========================================================
# TRUE VEHICULAR DOPPLER / ICI CHECK (diagnostic physical ICI model)
# ==========================================================

print(
    "\nTrue Vehicular Doppler ICI check:"
)

# ----------------------------------------------------------
# FAIR TEST:
# Keep the transmitted bits and the channel realization
# identical for every Eb/N0 value. Only the AWGN level
# changes. This prevents random channel changes from
# producing artificial BER fluctuations.
# ----------------------------------------------------------

bits_ici = np.random.randint(
    0,
    2,
    (
        NUM_SYMBOLS,
        len(data_index)
    )
)

tx_symbol_ici = create_ofdm_symbol(
    bits_ici
)

tx_ici = ofdm_modulate(
    tx_symbol_ici
)

ICI_TEST_SEED = 20260808

for eb in [0, 8, 12, 16, 20]:

    # Reset the random generator before each Eb/N0 point.
    # The channel realization and random sequence are therefore
    # reproducible across Eb/N0; only the AWGN level changes.
    np.random.seed(
        ICI_TEST_SEED
    )

    rx, H_diag = rayleigh_channel_ici(
        tx_ici,
        eb,
        VEHICULAR_PATHS,
        VEHICULAR_FD
    )

    rx = ofdm_demodulate(
        rx
    )

    # H_diag is already the diagonal frequency response
    # with exactly N subcarriers.
    rx_eq = zf_equalizer(
        rx,
        H_diag
    )

    data = extract_data(
        rx_eq
    )

    detected = bpsk_demod(
        data
    )

    ber = ber_count(
        bits_ici,
        detected
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )


# ==========================================================
# FILE INTEGRITY NOTE
# ==========================================================
# This version preserves the numerical simulation structure of v3.
# FCAE training/validation, main BER curves, and the true-ICl diagnostic
# remain separate so that the Figure 11 results are not silently changed.
# ==========================================================
# FIGURE 12 - PAPR CCDF COMPARISON
# Conventional OFDM vs Random Phase Rotation vs FCAE
# ==========================================================

print("\nGenerating PAPR CCDF comparison ...")


# ----------------------------------------------------------
# 1) Generate the SAME independent validation set
# ----------------------------------------------------------
# This uses the same validation seed and number of symbols
# already used in train_fcae().
# The validation set is NOT used for FCAE training.
# ----------------------------------------------------------

# ----------------------------------------------------------
# 1) Use the EXACT independent validation realization already
# used inside train_fcae().
#
# This guarantees that Figure 12 uses the same validation
# samples and the same trained FCAE output that produced the
# reported validation PAPR. No second validation realization
# is generated and the network is not called again.
# ----------------------------------------------------------

X_ccdf = X_val_final
s_ccdf = s_val_final
s_fcae_ccdf = val_recovered_final


# ----------------------------------------------------------
# 4) PAPR per OFDM symbol
# ----------------------------------------------------------

def papr_per_symbol_db(signals):

    power = np.abs(signals) ** 2

    return (
        10
        * np.log10(
            np.max(
                power,
                axis=1
            )
            /
            (
                np.mean(
                    power,
                    axis=1
                )
                + 1e-12
            )
        )
    )


papr_ofdm_ccdf = papr_per_symbol_db(
    s_ccdf
)

papr_fcae_ccdf = papr_per_symbol_db(
    s_fcae_ccdf
)


# ----------------------------------------------------------
# 5) Random Phase Rotation
# ----------------------------------------------------------
# Four random phase candidates are used, consistent with
# the RPR implementation used previously in the project.
#
# The phase rotation is applied independently to the
# subcarriers in the frequency domain.
# The candidate producing the lowest PAPR is selected for
# each OFDM symbol.
# ----------------------------------------------------------

RPR_CANDIDATES = 4

rng_rpr = np.random.default_rng(
    20260808
)

best_papr_rpr = np.full(
    NUM_SYMBOLS,
    np.inf,
    dtype=np.float64
)


for candidate in range(
    RPR_CANDIDATES
):

    # Random phase rotation for each subcarrier
    phase = rng_rpr.uniform(
        0,
        2 * np.pi,
        N
    )

    phase_vector = np.exp(
        1j * phase
    )

    # Apply phase rotation in frequency domain
    X_rotated = (
        X_ccdf
        *
        phase_vector[None, :]
    )

    # Convert to time domain
    s_rotated = np.fft.ifft(
        X_rotated,
        axis=1
    )

    # Calculate PAPR for every OFDM symbol
    papr_candidate = papr_per_symbol_db(
        s_rotated
    )

    # Keep the best candidate independently
    # for each OFDM symbol
    best_papr_rpr = np.minimum(
        best_papr_rpr,
        papr_candidate
    )


papr_rpr_ccdf = best_papr_rpr


# ----------------------------------------------------------
# 6) CCDF calculation
# ----------------------------------------------------------

def calculate_ccdf(papr_values):

    papr_sorted = np.sort(
        papr_values
    )

    ccdf = (
        1.0
        -
        np.arange(
            1,
            len(papr_sorted) + 1
        )
        /
        len(papr_sorted)
    )

    return (
        papr_sorted,
        ccdf
    )


papr_x_ofdm, ccdf_ofdm = calculate_ccdf(
    papr_ofdm_ccdf
)

papr_x_rpr, ccdf_rpr = calculate_ccdf(
    papr_rpr_ccdf
)

papr_x_fcae, ccdf_fcae = calculate_ccdf(
    papr_fcae_ccdf
)


# ----------------------------------------------------------
# 7) Plot Figure 12
# ----------------------------------------------------------

plt.figure(
    figsize=(6, 4),
    dpi=300
)

plt.semilogy(
    papr_x_ofdm,
    ccdf_ofdm,
    linewidth=1.8,
    label="Conventional OFDM"
)

plt.semilogy(
    papr_x_rpr,
    ccdf_rpr,
    linewidth=1.8,
    label="Random Phase Rotation"
)

plt.semilogy(
    papr_x_fcae,
    ccdf_fcae,
    linewidth=1.8,
    label="FCAE / OFDM-X"
)

plt.xlabel(
    "PAPR (dB)"
)

plt.ylabel(
    "CCDF = P(PAPR > PAPR₀)"
)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.4
)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    "Figure12_PAPR_CCDF.png",
    dpi=300,
    bbox_inches="tight"
)

# Save the exact CCDF samples used in Figure 12 for reproducibility.
ccdf_table = np.column_stack(
    (
        papr_x_ofdm,
        ccdf_ofdm,
        papr_x_rpr,
        ccdf_rpr,
        papr_x_fcae,
        ccdf_fcae
    )
)

np.savetxt(
    "Figure12_PAPR_CCDF_Data.csv",
    ccdf_table,
    delimiter=",",
    header=(
        "OFDM_PAPR_dB,OFDM_CCDF,"
        "RPR_PAPR_dB,RPR_CCDF,"
        "FCAE_PAPR_dB,FCAE_CCDF"
    ),
    comments=""
)

plt.show()


# ----------------------------------------------------------
# 8) Print mean PAPR for consistency checking
# ----------------------------------------------------------

print(
    "\nPAPR CCDF validation statistics:"
)

print(
    f"Conventional OFDM mean PAPR = "
    f"{np.mean(papr_ofdm_ccdf):.3f} dB"
)

print(
    f"FCAE / OFDM-X mean PAPR = "
    f"{np.mean(papr_fcae_ccdf):.3f} dB"
)

print(
    f"FCAE improvement = "
    f"{np.mean(papr_ofdm_ccdf) - np.mean(papr_fcae_ccdf):.3f} dB"
)

print(
    f"Random Phase Rotation mean PAPR = "
    f"{np.mean(papr_rpr_ccdf):.3f} dB"
)

# Explicit consistency check. Because the CCDF uses the exact
# validation output retained from train_fcae(), this should be
# zero up to floating-point precision.
validation_papr_from_ccdf = np.mean(
    papr_fcae_ccdf
)

validation_papr_direct = np.mean(
    10
    * np.log10(
        np.max(
            np.abs(val_recovered_final) ** 2,
            axis=1
        )
        /
        (
            np.mean(
                np.abs(val_recovered_final) ** 2,
                axis=1
            )
            + 1e-12
        )
    )
)

validation_consistency_error = abs(
    validation_papr_from_ccdf
    -
    validation_papr_direct
)

print(
    f"FCAE CCDF/validation consistency error = "
    f"{validation_consistency_error:.3e} dB"
)

print(
    "\nFigure 12 generated successfully:"
    " Figure12_PAPR_CCDF.png"
)

print(
    "CCDF data saved to:"
    " Figure12_PAPR_CCDF_Data.csv"
)



# ==========================================================
# Produces BER curves for:
#   1) Conventional OFDM
#   2) OFDM-X / FCAE
#   3) Random Phase Rotation
# under AWGN, Vehicular, LEO, and ISAC.
#
# This section is intentionally separate from the finalized
# FCAE PAPR/CCDF result and does not modify it.
# ==========================================================

EbN0_RPR = np.arange(0, 21, 2)
rpr_results = {}

for channel_name in ["Vehicular", "LEO", "ISAC"]:
    print(f"\nRunning RPR BER for {channel_name} ...")
    channel_seed = {"Vehicular": 1002, "LEO": 1003, "ISAC": 1004}[channel_name]
    rng_rpr = np.random.default_rng(20260808 + channel_seed)

    rpr_curve = []
    for eb in EbN0_RPR:
        rpr_curve.append(
            simulate_ber_rpr_final(
                eb, channel_name, rng_rpr
            )
        )

    rpr_results[channel_name] = np.array(rpr_curve)

# Save reviewer comparison data.
np.savez(
    "Figure11_THREE_WAY_BER_RESULTS.npz",
    EbN0=EbN0_RPR,
    RPR_Vehicular=rpr_results["Vehicular"],
    RPR_LEO=rpr_results["LEO"],
    RPR_ISAC=rpr_results["ISAC"],
)

print("\nFigure 11 RPR BER data saved:")
print("Figure11_THREE_WAY_BER_RESULTS.npz")




# ==========================================================
# ============================================================
# BER completeness check
# ============================================================
_expected_ebn0 = np.arange(0, 21, 2)
if not np.array_equal(np.asarray(EbN0_RPR), _expected_ebn0):
    raise RuntimeError(
        "Figure 11 BER sweep is incomplete: expected Eb/N0 = 0,2,...,20 dB."
    )

# ============================================================
# Numerical BER table for reviewer response
# ============================================================
if "results" in globals():
    print("\n" + "=" * 108)
    print("Figure 11 BER NUMERICAL RESULTS")
    print("Comparison: Conventional OFDM vs OFDM-X/FCAE vs Random Phase Rotation")
    print("=" * 108)
    print(
        f"{'Channel':<12} {'Eb/N0(dB)':>9} "
        f"{'OFDM BER':>14} {'OFDM-X/FCAE BER':>18} "
        f"{'RPR BER':>14} {'BER Change (%)':>16}"
    )
    print("-" * 108)

    channel_order_table = ["Vehicular", "LEO", "ISAC"]

    for ch in channel_order_table:
        ofdm_ber = np.asarray(results[ch]["OFDM"], dtype=float)
        fcae_ber = np.asarray(results[ch]["OFDMX"], dtype=float)
        rpr_ber = np.asarray(rpr_results[ch], dtype=float)

        if len(ofdm_ber) != len(_expected_ebn0) or len(fcae_ber) != len(_expected_ebn0) or len(rpr_ber) != len(_expected_ebn0):
            raise RuntimeError(
                f"Figure 11 BER data incomplete for {ch}: "
                f"expected {len(_expected_ebn0)} points at 2-dB spacing."
            )

        for i, eb in enumerate(EbN0_RPR):
            # Relative BER change of OFDM-X/FCAE with respect to conventional OFDM.
            if ofdm_ber[i] > 0:
                ber_change = 100.0 * (fcae_ber[i] - ofdm_ber[i]) / ofdm_ber[i]
            else:
                ber_change = np.nan

            print(
                f"{ch:<12} {eb:>9.1f} "
                f"{ofdm_ber[i]:>14.6e} {fcae_ber[i]:>18.6e} "
                f"{rpr_ber[i]:>14.6e} {ber_change:>16.3f}"
            )

    print("=" * 108)
    print("BER Change (%) = 100 × (OFDM-X/FCAE BER − Conventional OFDM BER) / Conventional OFDM BER")
    print("Positive values indicate BER degradation; negative values indicate BER improvement.")
    print("=" * 108)
    print("Completeness check: ISAC Eb/N0 = 12 dB is included.")
    print("=" * 108 + "\n")



# ============================================================
# Figure 11 - FINAL BER COMPARISON
# Reviewer #3 requirement:
# Conventional OFDM vs OFDM-X/FCAE vs Random Phase Rotation
# under Vehicular, LEO, and ISAC channels.
#
# IMPORTANT:
# - Uses the already calculated BER arrays.
# - Does NOT modify simulation data.
# - Three curves are plotted for every panel.
# - Screen preview: 100 dpi
# - Publication file: 600 dpi
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 8

CHANNELS_FIG11 = ["Vehicular", "LEO", "ISAC"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(11.5, 4.6),
    dpi=300,
    squeeze=False
)
axes = axes.ravel()

# Make sure the three actual data arrays are available.
required = ["results", "rpr_results", "EbN0_dB", "EbN0_RPR"]
missing = [x for x in required if x not in globals()]
if missing:
    raise RuntimeError(
        "Figure 11 cannot be generated because these calculated "
        f"results are missing: {missing}"
    )

legend_handles = []
legend_labels = [
    "Conventional OFDM",
    "OFDM-X / FCAE",
    "Random Phase Rotation"
]

for i, channel in enumerate(CHANNELS_FIG11):

    ax = axes[i]

    ofdm_ber = np.asarray(
        results[channel]["OFDM"],
        dtype=float
    )

    fcae_ber = np.asarray(
        results[channel]["OFDMX"],
        dtype=float
    )

    rpr_ber = np.asarray(
        rpr_results[channel],
        dtype=float
    )

    # --------------------------------------------------------
    # Sanity checks: all three curves must exist and have
    # valid positive BER values for semilog plotting.
    # --------------------------------------------------------
    if len(ofdm_ber) != len(EbN0_dB):
        raise RuntimeError(
            f"{channel}: OFDM BER length does not match Eb/N0."
        )

    if len(fcae_ber) != len(EbN0_dB):
        raise RuntimeError(
            f"{channel}: FCAE BER length does not match Eb/N0."
        )

    if len(rpr_ber) != len(EbN0_RPR):
        raise RuntimeError(
            f"{channel}: RPR BER length does not match Eb/N0."
        )

    # Replace only exact zeros for visualization on a log axis.
    # The underlying BER arrays are NOT modified.
    ofdm_plot = np.maximum(ofdm_ber, 1e-7)
    fcae_plot = np.maximum(fcae_ber, 1e-7)
    rpr_plot = np.maximum(rpr_ber, 1e-7)

    # --------------------------------------------------------
    # Three curves
    # --------------------------------------------------------
    h1, = ax.semilogy(
        EbN0_dB,
        ofdm_plot,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        markevery=1,
        label=legend_labels[0]
    )

    h2, = ax.semilogy(
        EbN0_dB,
        fcae_plot,
        marker="s",
        linestyle="--",
        linewidth=1.5,
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        markevery=1,
        label=legend_labels[1]
    )

    h3, = ax.semilogy(
        EbN0_RPR,
        rpr_plot,
        marker="^",
        linestyle="-.",
        linewidth=1.5,
        markersize=4.5,
        markerfacecolor="white",
        markeredgewidth=0.9,
        markevery=1,
        label=legend_labels[2]
    )

    if i == 0:
        legend_handles = [h1, h2, h3]

    # --------------------------------------------------------
    # Panel title and labels
    # --------------------------------------------------------
    ax.set_title(
        channel,
        fontsize=8,
        fontname="Times New Roman",
        fontweight="bold",
        pad=5
    )

    ax.set_xlabel(
        r"$E_b/N_0$ (dB)",
        fontsize=8,
        fontname="Times New Roman"
    )

    ax.set_ylabel(
        "BER",
        fontsize=8,
        fontname="Times New Roman"
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=8,
        width=0.7,
        length=3
    )

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
        tick.set_fontsize(8)

    ax.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.45,
        alpha=0.45
    )

    ax.text(
    0.97,
    0.96,
    f"({chr(97 + i)})",
    transform=ax.transAxes,
    fontsize=8,
    fontname="Times New Roman",
    fontweight="bold",
    va="top",
    ha="right"
)

# ------------------------------------------------------------
# Shared legend
# ------------------------------------------------------------
fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.005),
    ncol=3,
    frameon=True,
    prop={
        "family": "Times New Roman",
        "size": 8
    },
    handlelength=2.0,
    handletextpad=0.6,
    columnspacing=1.5,
    borderpad=0.5
)

fig.subplots_adjust(
    left=0.065,
    right=0.995,
    top=0.91,
    bottom=0.21,
    wspace=0.25
)

FIG11_OUTPUT = (
    "Figure11_BER_FINAL_"
    "Vehicular_LEO_ISAC_"
    "3Curves_600dpi_TimesNewRoman_8pt.png"
)

# PNG: high-resolution raster for Microsoft Word
fig.savefig(
    FIG11_OUTPUT,
    dpi=600,
    facecolor="white",
    bbox_inches="tight"
)

# SVG: vector version for Microsoft Word / Office workflows
FIG11_SVG = (
    "Figure11_BER_FINAL_Vehicular_LEO_ISAC_MSWORD.svg"
)
fig.savefig(
    FIG11_SVG,
    format="svg",
    facecolor="white",
    bbox_inches="tight"
)

plt.show()

print()
print("=" * 80)
print("FIGURE 11 FINAL")
print("=" * 80)
print("Panels : Vehicular | LEO | ISAC")
print("Curves : OFDM | OFDM-X/FCAE | Random Phase Rotation")
print("Font   : Times New Roman, 8 pt")
print("Output : 600 DPI")
print(f"PNG    : {FIG11_OUTPUT}")
print(f"SVG    : {FIG11_SVG}")
print("=" * 80)

# Print the actual BER values used by Figure 11 so that the
# manuscript text can be based on the same simulation output.
print("\nBER VALUES USED IN FIGURE 11")
for channel in CHANNELS_FIG11:
    print(f"\n{channel}")
    print("Eb/N0(dB) | OFDM | FCAE | RPR")
    for j, snr in enumerate(EbN0_dB):
        rpr_val = rpr_results[channel][j]
        print(
            f"{snr:8.1f} | "
            f"{results[channel]['OFDM'][j]:.8e} | "
            f"{results[channel]['OFDMX'][j]:.8e} | "
            f"{rpr_val:.8e}"
        )
