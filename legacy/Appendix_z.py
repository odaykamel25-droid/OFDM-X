# ==========================================================
# BER Simulation for OFDM / OFDM-X
# Physical Channel Model
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

np.random.seed(42)

# ==========================================================
# System Parameters
# ==========================================================

N = 64                  # Number of subcarriers

CP = 16                 # Cyclic Prefix

NUM_SYMBOLS = 5000      # Monte Carlo symbols

EbN0_dB = np.arange(0,22,2)

Fs = 15000

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
    # Rayleigh path gains
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
    ) / np.sqrt(
        2 * num_paths
    )

    # ------------------------------------------------------
    # Path delays
    # ------------------------------------------------------

    delays = np.arange(
        num_paths
    )

    # ------------------------------------------------------
    # Time axis
    # ------------------------------------------------------

    t = np.arange(
        total_len
    ) / Fs

    # ------------------------------------------------------
    # Doppler frequency for each path
    # ------------------------------------------------------

    fd_paths = np.random.uniform(
        -fd,
        fd,
        num_paths
    )

    faded = np.zeros_like(
        tx,
        dtype=np.complex128
    )

    # Store exact time-domain channel
    # for diagnostic equalization

    h_time = np.zeros(
        (
            num_symbols,
            total_len
        ),
        dtype=np.complex128
    )

    # ------------------------------------------------------
    # Apply time-varying multipath
    # ------------------------------------------------------

    for i in range(num_symbols):

        for p in range(num_paths):

            d = delays[p]

            doppler = np.exp(
                1j *
                2 *
                np.pi *
                fd_paths[p] *
                t
            )

            path_gain = (
                taps[i, p] *
                doppler
            )

            h_time[i] += path_gain

            if d == 0:

                faded[i] += (
                    path_gain *
                    tx[i]
                )

            else:

                shifted = np.zeros(
                    total_len,
                    dtype=np.complex128
                )

                shifted[d:] = (
                    tx[i, :-d] *
                    doppler[d:]
                )

                faded[i] += (
                    taps[i, p] *
                    shifted
                )

    # ------------------------------------------------------
    # AWGN
    # ------------------------------------------------------

    rx = awgn(
        faded,
        ebn0_db
    )

    return rx, h_time
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

    if channel=="AWGN":

        rx = awgn_channel(
            tx,
            ebn0_db
        )

        h = np.ones_like(tx)

    elif channel=="Vehicular":

        rx,h = rayleigh_channel(

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
# Conventional OFDM BER
# ==========================================================
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

fcae = build_fcae()

fcae.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=AE_LEARNING_RATE
    ),
    loss=fcae_loss
)
def train_fcae(model):
    
        X = (
            2
            *
            (
                np.random.randint(
                    0,
                    2,
                    (
                        NUM_SYMBOLS,
                        N
                    )
                )
                - 0.5
            )
        ).astype(
            np.complex64
        )

        s = np.fft.ifft(
            X,
            axis=1
        )

        s_in = prepare_fcae_input(
            s
        )

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

            if (
                (epoch + 1) % 5 == 0
                or
                epoch == 0
            ):

                prediction = model.predict(
                    s_in,
                    verbose=0
                )

                recovered = recover_fcae_output(
                    prediction
                )

                papr_original = np.mean(
                    10 * np.log10(
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
                    10 * np.log10(
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

        return model
# ==========================================================
# TRAIN FCAE ONCE
# ========================================================

fcae = train_fcae(fcae)
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

    # ----------------------------------------------------------
# OFDM modulation WITHOUT CP
# ----------------------------------------------------------

tx = np.fft.ifft(
    tx_symbol,
    axis=1
)

# ----------------------------------------------------------
# FCAE processing
# FCAE expects exactly N = 64 samples
# ----------------------------------------------------------

tx = ae_process(
    fcae,
    tx
)

# ----------------------------------------------------------
# Add cyclic prefix AFTER FCAE
# ----------------------------------------------------------

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

    print(f"\nRunning {channel} Channel ...")

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

print("\nBER Simulation Completed.")
# ==========================================================
# Figure 11
# ==========================================================

plt.figure(figsize=(6,4), dpi=300)

for channel in channels:

    plt.semilogy(

        EbN0_dB,

        results[channel]["OFDM"],

        'o-',

        linewidth=1.5,

        markersize=4,

        label=f"{channel} OFDM"

    )

    plt.semilogy(

        EbN0_dB,

        results[channel]["OFDMX"],

        's--',

        linewidth=1.5,

        markersize=4,

        label=f"{channel} OFDM-X"

    )

plt.grid(True, which="both", linestyle="--", alpha=0.4)

plt.xlabel(r"$E_b/N_0$ (dB)")

plt.ylabel("BER")

plt.legend(fontsize=7, ncol=2)

plt.tight_layout()

plt.savefig(

    "Figure11_BER.png",

    dpi=300

)

plt.show()
# ==========================================================
# BASIC OFDM + AWGN CHECK
# ==========================================================

print("\nBasic OFDM-AWGN check:")

test_ber = []

for eb in EbN0_dB:

    ber = simulate_ber_ofdm(
        eb,
        "AWGN"
    )

    test_ber.append(ber)

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )
    print("\nLEO OFDM check:")

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

print("\nVehicular OFDM check:")

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

print("\nISAC OFDM check:")

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

print("\nCurrent OFDM-X check:")

for eb in EbN0_dB:

    ber = simulate_ber_ofdmx(
        eb,
        "ISAC"
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )
    # ==========================================================
# FAIR VEHICULAR DOPPLER TEST
# fd = 0 Hz versus fd = 200 Hz
# Same bits, channel realization, and noise
# ==========================================================

print("\nFair Vehicular Doppler test:")

for eb in EbN0_dB:

    # ------------------------------------------------------
    # Generate the same transmitted data
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # fd = 0 Hz
    # ------------------------------------------------------

    np.random.seed(12345)

    rx0, h0 = rayleigh_channel(
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

    # ------------------------------------------------------
    # fd = 200 Hz
    # Same random seed
    # ------------------------------------------------------

    np.random.seed(12345)

    rx200, h200 = rayleigh_channel(
        tx,
        eb,
        VEHICULAR_PATHS,
        VEHICULAR_FD
    )

    rx200 = ofdm_demodulate(
        rx200
    )

    H200 = ls_channel_estimation(
        rx200
    )

    rx200 = zf_equalizer(
        rx200,
        H200
    )

    data200 = extract_data(
        rx200
    )

    detected200 = bpsk_demod(
        data200
    )

    ber200 = ber_count(
        bits,
        detected200
    )

    # ------------------------------------------------------
    # Print comparison
    # ------------------------------------------------------

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  fd=0 = {ber0:.6e}"
        f"  fd={VEHICULAR_FD} = {ber200:.6e}"
    )
    # ==========================================================
# VEHICULAR PERFECT CSI CHECK
# ==========================================================

print("\nVehicular Perfect CSI check:")

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

    # Apply Vehicular channel
    rx, h = apply_channel(
        tx,
        eb,
        "Vehicular"
    )

    # OFDM demodulation
    rx = ofdm_demodulate(
        rx
    )

    # ------------------------------------------------------
    # Perfect CSI
    # Use the actual channel response
    # ------------------------------------------------------

    H_perfect = h[:, CP:]

    # Zero-Forcing equalization
    rx = zf_equalizer(
        rx,
        H_perfect
    )

    # Extract data
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
# FAIR VEHICULAR: LS vs PERFECT CSI
# Same bits + same channel realization
# ==========================================================

def fair_vehicular_csi_test(ebn0_db):

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

    # ------------------------------------------------------
    # Same channel realization
    # ------------------------------------------------------

    rx, h = apply_channel(
        tx,
        ebn0_db,
        "Vehicular"
    )

    rx = ofdm_demodulate(
        rx
    )

    # ------------------------------------------------------
    # Perfect CSI
    # ------------------------------------------------------

    H_perfect = h[:, CP:]

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

    # ------------------------------------------------------
    # LS CSI
    # ------------------------------------------------------

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

    return ber_ls, ber_perfect
print("\nFair Vehicular CSI comparison:")

for eb in EbN0_dB:

    ber_ls, ber_perfect = fair_vehicular_csi_test(
        eb
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  LS = {ber_ls:.6e}"
        f"  Perfect CSI = {ber_perfect:.6e}"
    )
    # ==========================================================
# TRUE VEHICULAR DOPPLER / ICI CHECK
# ==========================================================

print("\nTrue Vehicular Doppler ICI check:")

for eb in [0, 8, 12, 16, 20]:

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

    rx, h_time = rayleigh_channel_ici(
        tx,
        eb,
        VEHICULAR_PATHS,
        VEHICULAR_FD
    )

    rx = ofdm_demodulate(
        rx
    )

    # ------------------------------------------------------
    # Diagnostic frequency-domain channel
    # ------------------------------------------------------

    H_diag = np.fft.fft(
        h_time[:, CP:],
        axis=1
    )

    H_diag = H_diag[
        :,
        :N
    ]

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
        bits,
        detected
    )

    print(
        f"Eb/N0 = {eb:2d} dB"
        f"  BER = {ber:.6e}"
    )