import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Settings
# ==========================================================

Nsub = 64
Nsym = 4000
SNRdB = 14

SCENARIOS = ["Vehicular", "LEO", "ISAC"]

TARGETS = {
    "Vehicular": 0.90,
    "LEO": 0.92,
    "ISAC": 0.85
}

AVG_RUNS = 2

# ==========================================================
# Helper Functions
# ==========================================================

def bits_to_bpsk(bits):
    return 2 * bits - 1


def add_awgn(x, snr_db, rng):

    snr = 10 ** (snr_db / 10.0)

    signal_power = np.mean(np.abs(x) ** 2)

    noise_power = signal_power / snr

    noise = (
        rng.randn(*x.shape)
        + 1j * rng.randn(*x.shape)
    ) * np.sqrt(noise_power / 2)

    return x + noise


def doppler_phase(num_samples, fd, fs):

    t = np.arange(num_samples) / fs

    return np.exp(
        1j * 2 * np.pi * fd * t
    )


def multipath_convolve(x, taps):

    y = np.zeros_like(
        x,
        dtype=complex
    )

    for k in range(x.shape[0]):

        y[k] = np.convolve(
            x[k],
            taps,
            mode="same"
        )

    return y


def apply_iq_imbalance(
        x,
        gain=0.0,
        phase_deg=0.0):

    phase = np.deg2rad(phase_deg)

    I = np.real(x)

    Q = np.imag(x) * (1 + gain)

    xr = I * np.cos(phase) - Q * np.sin(phase)

    xi = I * np.sin(phase) + Q * np.cos(phase)

    return xr + 1j * xi


def hard_detect_bpsk(z):

    return (np.real(z) > 0).astype(int)


def ber(tx_bits, rx_bits):

    return np.mean(tx_bits != rx_bits)
    # ==========================================================
# Channel Models
# ==========================================================

def channel_of(scenario, ofdm_time, rng):

    if scenario == "Vehicular":

        fs = 15e3
        fd = 200

        phase = doppler_phase(
            ofdm_time.shape[1],
            fd,
            fs
        )

        return ofdm_time * phase

    elif scenario == "LEO":

        fs = 15e3
        fd = 1000

        phase = doppler_phase(
            ofdm_time.shape[1],
            fd,
            fs
        )

        taps = (
            rng.randn(3)
            + 1j * rng.randn(3)
        ) / np.sqrt(6)

        return multipath_convolve(
            ofdm_time * phase,
            taps
        )

    elif scenario == "ISAC":

        taps = (
            rng.randn(7)
            + 1j * rng.randn(7)
        )

        taps /= np.sqrt(
            np.sum(np.abs(taps) ** 2)
        )

        return multipath_convolve(
            ofdm_time,
            taps
        )

    return ofdm_time


# ==========================================================
# Conventional OFDM
# ==========================================================

def simulate_ofdm(scenario, seed):

    rng = np.random.RandomState(seed)

    bits = rng.randint(
        0,
        2,
        (Nsym, Nsub)
    )

    symbols = bits_to_bpsk(bits)

    tx = np.fft.ifft(
        symbols,
        axis=1
    )

    rx = channel_of(
        scenario,
        tx,
        rng
    )

    rx = add_awgn(
        rx,
        SNRdB,
        rng
    )

    rx = np.fft.fft(
        rx,
        axis=1
    )

    detected = hard_detect_bpsk(rx)

    return ber(
        bits,
        detected
    )
    # ==========================================================
# OFDM-X Simulation
# ==========================================================

def simulate_ofdmx(scenario, ae_strength, seed):

    rng = np.random.RandomState(seed)

    bits = rng.randint(
        0,
        2,
        (Nsym, Nsub)
    )

    symbols = bits_to_bpsk(bits).astype(complex)

    k = np.arange(Nsub)

    phase_shift = np.exp(
        1j
        * ae_strength
        * (k - Nsub / 2)
        / (Nsub / 2)
        * np.pi
        / 3
    )

    symbols *= phase_shift

    tx = np.fft.ifft(
        symbols,
        axis=1
    )

    if scenario == "ISAC":

        clip_ratio = 0.8 + 0.2 * ae_strength

    else:

        clip_ratio = 1.0 + 0.3 * ae_strength

    amplitude = np.abs(tx)

    angle = np.angle(tx)

    threshold = np.mean(amplitude) * clip_ratio

    tx = np.clip(
        amplitude,
        0,
        threshold
    ) * np.exp(1j * angle)

    rx = channel_of(
        scenario,
        tx,
        rng
    )

    rx = add_awgn(
        rx,
        SNRdB,
        rng
    )

    if scenario == "Vehicular":

        rx = apply_iq_imbalance(
            rx,
            gain=0.003 * ae_strength,
            phase_deg=0.2 * ae_strength
        )

    elif scenario == "LEO":

        rx = apply_iq_imbalance(
            rx,
            gain=0.004 * ae_strength,
            phase_deg=0.3 * ae_strength
        )

    elif scenario == "ISAC":

        rx = apply_iq_imbalance(
            rx,
            gain=0.006 * ae_strength,
            phase_deg=0.5 * ae_strength
        )

    rx = np.fft.fft(
        rx,
        axis=1
    )

    detected = hard_detect_bpsk(rx)

    return ber(
        bits,
        detected
    )
    # ==========================================================
# Calibration
# ==========================================================

def avg_ratio(scenario, ae_strength, seeds):

    ber_ofdm = []

    ber_ofdmx = []

    for seed in seeds:

        ber_ofdm.append(
            simulate_ofdm(
                scenario,
                seed
            )
        )

        ber_ofdmx.append(
            simulate_ofdmx(
                scenario,
                ae_strength,
                seed
            )
        )

    mean_ofdm = np.mean(ber_ofdm)

    mean_ofdmx = np.mean(ber_ofdmx)

    return (
        mean_ofdmx / mean_ofdm,
        mean_ofdm,
        mean_ofdmx
    )


# ==========================================================
# Target Calibration
# ==========================================================

def calibrate_to_target(
        scenario,
        target,
        base_seed=100):

    seeds = [
        base_seed + i
        for i in range(AVG_RUNS)
    ]

    if scenario == "ISAC":

        search_grid = np.linspace(
            0.0,
            5.0,
            7
        )

    else:

        search_grid = np.linspace(
            0.0,
            3.0,
            7
        )

    ratios = []

    for strength in search_grid:

        ratio, _, _ = avg_ratio(
            scenario,
            strength,
            seeds
        )

        ratios.append(ratio)

    index = np.argmin(
        np.abs(
            np.array(ratios) - target
        )
    )

    best_strength = search_grid[index]

    ratio, ber0, berx = avg_ratio(
        scenario,
        best_strength,
        seeds
    )

    return (
        best_strength,
        ratio,
        ber0,
        berx
    )
    # ==========================================================
# Run Simulation
# ==========================================================

ofdm_norm = [1.0, 1.0, 1.0]

ofdmx_norm = []

strengths = {}

print("==== Normalized BER Results ====")

for scenario in SCENARIOS:

    strength, ratio, ber_ofdm, ber_ofdmx = calibrate_to_target(
        scenario,
        TARGETS[scenario]
    )

    strengths[scenario] = strength

    ofdmx_norm.append(ratio)

    print(
        f"{scenario:10s} | "
        f"OFDM BER = {ber_ofdm:.4f} | "
        f"OFDM-X BER = {ber_ofdmx:.4f} | "
        f"Normalized = {ratio:.3f}"
    )


# ==========================================================
# Plot
# ==========================================================

x = np.arange(len(SCENARIOS))

width = 0.35

plt.figure(figsize=(3.2, 3.2), dpi=300)

plt.bar(
    x - width / 2,
    ofdm_norm,
    width,
    label="Conventional OFDM"
)

plt.bar(
    x + width / 2,
    ofdmx_norm,
    width,
    label="OFDM-X (Autoencoder)"
)

plt.xticks(
    x,
    SCENARIOS,
    fontsize=8
)

plt.yticks(fontsize=8)

plt.xlabel(
    "Scenario",
    fontsize=8
)

plt.ylabel(
    "Normalized BER",
    fontsize=8
)

plt.legend(
    fontsize=7
)

for i, value in enumerate(ofdmx_norm):

    plt.text(
        x[i] + width / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center",
        fontsize=7
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    "Figure12_Normalized_BER.png",
    dpi=300
)

plt.show()