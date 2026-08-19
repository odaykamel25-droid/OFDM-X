# ==========================================================
# BER Performance Evaluation for OFDM / OFDM-X / RPR
# AWGN, Vehicular, LEO and ISAC Channels
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.signal import fftconvolve

# ==========================================================
# Random Seed
# ==========================================================

np.random.seed(42)

# ==========================================================
# System Parameters
# ==========================================================

N_SUBCARRIERS = 64

CP_LENGTH = 16

NUM_TEST_SYMBOLS = 10000

EBN0_DB = np.arange(
    0,
    21,
    2
)

SAMPLING_FREQUENCY = 15000

MODULATION_ORDER = 2

COMPLEX_TYPE = np.complex128

# ==========================================================
# Channel Parameters
# ==========================================================

VEHICULAR_DOPPLER = 250

LEO_DOPPLER = 1200

ISAC_DOPPLER = 450

LEO_K_FACTOR = 8

VEHICULAR_NUM_PATHS = 4

LEO_NUM_PATHS = 3

ISAC_NUM_PATHS = 7

# ==========================================================
# Plot Style
# ==========================================================

plt.rcParams["font.family"] = "Times New Roman"

plt.rcParams["font.size"] = 8

plt.rcParams["axes.labelweight"] = "bold"

plt.rcParams["axes.titleweight"] = "bold"
# ==========================================================
# BER Function
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
# Add / Remove Cyclic Prefix
# ==========================================================

def add_cp(signal):

    cp = signal[:, -CP_LENGTH:]

    return np.concatenate(
        (cp, signal),
        axis=1
    )


def remove_cp(signal):

    return signal[
        :,
        CP_LENGTH:
    ]


# ==========================================================
# OFDM Modulator
# ==========================================================

def ofdm_modulate(symbols):

    time_signal = np.fft.ifft(
        symbols,
        axis=1
    )

    return add_cp(
        time_signal
    )


# ==========================================================
# OFDM Demodulator
# ==========================================================

def ofdm_demodulate(signal):

    signal = remove_cp(
        signal
    )

    return np.fft.fft(
        signal,
        axis=1
    )


# ==========================================================
# AWGN
# ==========================================================

def awgn(signal, ebn0_db):

    snr = 10 ** (
        ebn0_db / 10.0
    )

    power = np.mean(
        np.abs(signal) ** 2
    )

    noise_power = power / snr

    noise = (

        np.random.randn(
            *signal.shape
        )

        +

        1j * np.random.randn(
            *signal.shape
        )

    ) * np.sqrt(
        noise_power / 2
    )

    return signal + noise
	# ==========================================================
# Rayleigh Fading Channel
# ==========================================================

def rayleigh_channel(signal, ebn0_db, num_paths, doppler):

    n_sym, n_samp = signal.shape

    t = np.arange(n_samp) / SAMPLING_FREQUENCY

    faded = np.zeros_like(
        signal,
        dtype=COMPLEX_TYPE
    )

    for _ in range(num_paths):

        gain = (
            np.random.randn()
            +
            1j * np.random.randn()
        ) / np.sqrt(2 * num_paths)

        delay = np.random.randint(
            0,
            CP_LENGTH
        )

        fd = doppler * (
            0.8 + 0.4 * np.random.rand()
        )

        phase = np.exp(
            1j * 2 * np.pi * fd * t
        )

        delayed = np.roll(
            signal,
            delay,
            axis=1
        )

        faded += gain * delayed * phase

    return awgn(
        faded,
        ebn0_db
    )


# ==========================================================
# Rician Fading Channel
# ==========================================================

def rician_channel(
    signal,
    ebn0_db,
    num_paths,
    doppler,
    k_factor
):

    n_sym, n_samp = signal.shape

    t = np.arange(n_samp) / SAMPLING_FREQUENCY

    los = np.sqrt(
        k_factor /
        (k_factor + 1)
    )

    nlos = np.sqrt(
        1 /
        (k_factor + 1)
    )

    faded = np.zeros_like(
        signal,
        dtype=COMPLEX_TYPE
    )

    phase = np.exp(
        1j * 2 * np.pi * doppler * t
    )

    faded += los * signal * phase

    for _ in range(num_paths):

        gain = (
            np.random.randn()
            +
            1j * np.random.randn()
        ) / np.sqrt(
            2 * num_paths
        )

        delay = np.random.randint(
            0,
            CP_LENGTH
        )

        fd = doppler * (
            0.7 + 0.6 * np.random.rand()
        )

        phase = np.exp(
            1j * 2 * np.pi * fd * t
        )

        delayed = np.roll(
            signal,
            delay,
            axis=1
        )

        faded += (
            nlos *
            gain *
            delayed *
            phase
        )

    return awgn(
        faded,
        ebn0_db
    )


# ==========================================================
# Channel Selector
# ==========================================================

def apply_channel(
    signal,
    ebn0_db,
    channel
):

    if channel == "AWGN":

        return awgn(
            signal,
            ebn0_db
        )

    elif channel == "Vehicular":

        return rayleigh_channel(
            signal,
            ebn0_db,
            VEHICULAR_NUM_PATHS,
            VEHICULAR_DOPPLER
        )

    elif channel == "LEO":

        return rician_channel(
            signal,
            ebn0_db,
            LEO_NUM_PATHS,
            LEO_DOPPLER,
            LEO_K_FACTOR
        )

    elif channel == "ISAC":

        return rayleigh_channel(
            signal,
            ebn0_db,
            ISAC_NUM_PATHS,
            ISAC_DOPPLER
        )

    else:

        raise ValueError(
            "Unknown channel."
        )


# ==========================================================
# One-Tap Frequency Equalizer
# ==========================================================

def equalize(rx_symbols):

    power = np.abs(
        rx_symbols
    ) ** 2

    return rx_symbols / (
        np.sqrt(
            power + 1e-12
        )
    )
	# ==========================================================
# PAPR Calculation
# ==========================================================

def calculate_papr(signal):

    power = np.abs(signal) ** 2

    peak = np.max(
        power,
        axis=1
    )

    average = np.mean(
        power,
        axis=1
    )

    return 10 * np.log10(
        peak / average
    )


# ==========================================================
# Random Phase Rotation (RPR)
# ==========================================================

def apply_rpr(
    symbols,
    num_candidates=8
):

    best_signal = None

    best_phase = None

    lowest_papr = np.inf

    for _ in range(num_candidates):

        phase = np.exp(

            1j *

            2 *

            np.pi *

            np.random.randint(
                0,
                4,
                N_SUBCARRIERS
            ) / 4

        )

        rotated = symbols * phase

        tx = ofdm_modulate(
            rotated
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
# Placeholder OFDM-X Processing
# Replace later with FCAE
# ==========================================================

def ofdmx_process(tx):

    clipped = np.tanh(
        1.2 * np.real(tx)
    ) + 1j * np.tanh(
        1.2 * np.imag(tx)
    )

    return 0.97 * tx + 0.03 * clipped
	# ==========================================================
# Conventional OFDM BER Simulation
# ==========================================================

def simulate_ber_ofdm(
    ebn0_db,
    channel
):

    bits = np.random.randint(
        0,
        2,
        (
            NUM_TEST_SYMBOLS,
            N_SUBCARRIERS
        )
    )

    symbols = bpsk_mod(
        bits
    )

    tx = ofdm_modulate(
        symbols
    )

    rx = apply_channel(
        tx,
        ebn0_db,
        channel
    )

    rx = ofdm_demodulate(
        rx
    )

    rx = equalize(
        rx
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

def simulate_ber_ofdmx(
    ebn0_db,
    channel
):

    bits = np.random.randint(
        0,
        2,
        (
            NUM_TEST_SYMBOLS,
            N_SUBCARRIERS
        )
    )

    symbols = bpsk_mod(
        bits
    )

    tx = ofdm_modulate(
        symbols
    )

    tx = ofdmx_process(
        tx
    )

    rx = apply_channel(
        tx,
        ebn0_db,
        channel
    )

    rx = ofdm_demodulate(
        rx
    )

    rx = equalize(
        rx
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

def simulate_ber_rpr(
    ebn0_db,
    channel
):

    bits = np.random.randint(
        0,
        2,
        (
            NUM_TEST_SYMBOLS,
            N_SUBCARRIERS
        )
    )

    symbols = bpsk_mod(
        bits
    )

    tx, phase = apply_rpr(
        symbols
    )

    rx = apply_channel(
        tx,
        ebn0_db,
        channel
    )

    rx = ofdm_demodulate(
        rx
    )

    rx = rx * np.conj(
        phase
    )

    rx = equalize(
        rx
    )

    detected = bpsk_demod(
        rx
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

    ber_rpr = []

    for eb in EBN0_DB:

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

print("\nBER Simulation Completed.")
# ==========================================================
# Draw Individual BER Figures (Figures 11-14)
# ==========================================================

figure_number = 11

for channel in channels:

    plt.figure(
        figsize=(3.2,3.2),
        dpi=300
    )

    plt.semilogy(
        EBN0_DB,
        results[channel]["OFDM"],
        'o-',
        linewidth=1.5,
        markersize=4,
        label="Conventional OFDM"
    )

    plt.semilogy(
        EBN0_DB,
        results[channel]["OFDMX"],
        's-',
        linewidth=1.5,
        markersize=4,
        label="OFDM-X"
    )

    plt.semilogy(
        EBN0_DB,
        results[channel]["RPR"],
        '^-',
        linewidth=1.5,
        markersize=4,
        label="RPR"
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
        fontsize=6
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
# ==========================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(7,6),
    dpi=300
)

for ax, channel in zip(
    axes.ravel(),
    channels
):

    ax.semilogy(
        EBN0_DB,
        results[channel]["OFDM"],
        'o-',
        linewidth=1.2,
        markersize=3,
        label="OFDM"
    )

    ax.semilogy(
        EBN0_DB,
        results[channel]["OFDMX"],
        's-',
        linewidth=1.2,
        markersize=3,
        label="OFDM-X"
    )

    ax.semilogy(
        EBN0_DB,
        results[channel]["RPR"],
        '^-',
        linewidth=1.2,
        markersize=3,
        label="RPR"
    )

    ax.set_title(channel)

    ax.set_xlabel(
        r"$E_b/N_0$ (dB)"
    )

    ax.set_ylabel(
        "BER"
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
# ==========================================================

plt.figure(
    figsize=(6.5,4.8),
    dpi=300
)

for channel in channels:

    plt.semilogy(
        EBN0_DB,
        results[channel]["OFDM"],
        linewidth=1.3,
        label=f"{channel} OFDM"
    )

    plt.semilogy(
        EBN0_DB,
        results[channel]["OFDMX"],
        '--',
        linewidth=1.3,
        label=f"{channel} OFDM-X"
    )

    plt.semilogy(
        EBN0_DB,
        results[channel]["RPR"],
        ':',
        linewidth=1.3,
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
# Save Results
# ==========================================================

df = pd.DataFrame({

    "EbN0": EBN0_DB,

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

print("\nSimulation Finished Successfully.")
print("Figures 11-16 Generated.")
print("BER_Results.csv Saved.")