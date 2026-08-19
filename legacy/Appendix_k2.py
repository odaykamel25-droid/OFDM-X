# ==========================================================
# BER Simulation for OFDM / OFDM-X
# Physical Channel Model
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import csv

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

def rayleigh_channel(tx, ebn0_db, num_paths, fd):
    
    h = (

        np.random.randn(tx.shape[0], 1)

        +

        1j * np.random.randn(tx.shape[0], 1)

    ) / np.sqrt(2)

    h = np.repeat(
        h,
        tx.shape[1],
        axis=1
    )

    rx = tx * h

    rx = awgn(
        rx,
        ebn0_db
    )

    return rx, h


# ==========================================================
# Rician Channel
# ==========================================================

def rician_channel(tx, ebn0_db):
    
    los = np.sqrt(
        K_FACTOR / (K_FACTOR + 1)
    )

    nlos = np.sqrt(
        1 / (K_FACTOR + 1)
    )

    ray = (

        np.random.randn(tx.shape[0], 1)

        +

        1j * np.random.randn(tx.shape[0], 1)

    ) / np.sqrt(2)

    h = los + nlos * ray

    h = np.repeat(
        h,
        tx.shape[1],
        axis=1
    )

    rx = tx * h

    rx = awgn(
        rx,
        ebn0_db
    )

    return rx, h

# ==========================================================
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

    H_est = h[:, CP:]

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

    tx = ofdm_modulate(
        tx_symbol
    )

    # سيتم استبدال هذا لاحقاً بـ FCAE الحقيقي

    tx = 0.97 * tx + 0.03 * np.tanh(tx)

    rx, h = apply_channel(
        tx,
        ebn0_db,
        channel
    )

    rx = ofdm_demodulate(
        rx
    )

    H_est = h[:, CP:]

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