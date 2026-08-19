"""
RSMA_ISAC_REAL_EVALUATION.py

Standalone supplementary evaluation for Reviewer #3.

IMPORTANT:
- This script does NOT modify or retrain the FCAE model.
- It does NOT change the established PAPR results.
- It evaluates a two-user downlink RSMA link plus an OFDM-ISAC
  sensing waveform using explicit equations.
- It reports all assumptions and generated metrics.
- It is intentionally separate from fig12 final.py.

The RSMA part uses:
  x = sqrt(Pc)*w_c*s_c + sqrt(P1)*w_1*s_1 + sqrt(P2)*w_2*s_2
with MMSE/RZF-style precoders obtained from the instantaneous channel.

The ISAC part uses a known OFDM pilot/reference waveform, a target
delay and Doppler, AWGN, matched filtering / correlation, and reports:
  detection probability,
  range RMSE,
  Doppler RMSE,
  sensing SINR.

Run:
  python RSMA_ISAC_REAL_EVALUATION.py
"""

import numpy as np
from pathlib import Path

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
rng = np.random.default_rng(SEED)

# -----------------------------
# Communication parameters
# -----------------------------
N_SUB = 64
CP_LEN = 16
DELTA_F = 15e3
FS = N_SUB * DELTA_F

SNR_DB = 14.0
SNR_LIN = 10.0 ** (SNR_DB / 10.0)

N_USERS = 2
N_TX = 4
MONTE_CARLO_RSMA = 2000

# RSMA power split
COMMON_POWER_FRACTION = 0.20
PRIVATE_POWER_FRACTION = 0.80

# -----------------------------
# ISAC parameters
# -----------------------------
C = 299792458.0
FC = 28e9
LAMBDA = C / FC

ISAC_FD = 450.0
ISAC_PATHS = 7

TARGET_RANGE_M = 150.0
TARGET_DOPPLER_HZ = 450.0

MONTE_CARLO_ISAC = 2000

# Detection threshold calibrated from noise-only trials.
PFA_TARGET = 1e-2


def cnormal(shape):
    return (
        rng.standard_normal(shape)
        + 1j * rng.standard_normal(shape)
    ) / np.sqrt(2.0)


def unit_norm(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def complex_awgn(shape, noise_power):
    return np.sqrt(noise_power / 2.0) * cnormal(shape)


def db10(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-30))


# ============================================================
# RSMA
# ============================================================

def rzf_precoder(H, alpha=1e-2):
    """
    H: K x Nt channel matrix.
    Returns Nt x K precoder.
    """
    K, Nt = H.shape
    A = H @ H.conj().T + alpha * np.eye(K)
    W = H.conj().T @ np.linalg.inv(A)

    # Normalize each private beam.
    for k in range(K):
        W[:, k] = unit_norm(W[:, k])

    return W


def rsma_trial(snr_lin=SNR_LIN):
    """
    Two-user MISO RSMA trial.
    """
    H = cnormal((N_USERS, N_TX))

    Wp = rzf_precoder(H)

    # Common beam: normalized sum of private beams.
    wc = unit_norm(np.sum(Wp, axis=1))

    # Equal private power.
    pc = COMMON_POWER_FRACTION
    pp = PRIVATE_POWER_FRACTION / N_USERS

    # Unit-energy symbols.
    sc = cnormal(())
    s1 = cnormal(())
    s2 = cnormal(())

    x = (
        np.sqrt(pc) * wc * sc
        + np.sqrt(pp) * Wp[:, 0] * s1
        + np.sqrt(pp) * Wp[:, 1] * s2
    )

    # Scale so total average transmit power is 1.
    x = x / np.sqrt(np.vdot(x, x).real + 1e-12)

    noise_var = 1.0 / snr_lin

    common_sinr = []
    private_sinr = []

    for k in range(N_USERS):

        hk = H[k, :]

        gc = np.abs(np.vdot(hk, wc)) ** 2 * pc
        gp = np.abs(np.vdot(hk, Wp[:, k])) ** 2 * pp

        other_private = 0.0
        for j in range(N_USERS):
            if j != k:
                other_private += (
                    np.abs(np.vdot(hk, Wp[:, j])) ** 2 * pp
                )

        # Common stream: both private streams are interference.
        den_c = other_private + gp + noise_var
        sinr_c = gc / (den_c + 1e-30)

        # Private stream after common-stream SIC.
        sinr_p = gp / (other_private + noise_var + 1e-30)

        common_sinr.append(sinr_c)
        private_sinr.append(sinr_p)

    # Common rate is limited by the weakest user.
    r_common = np.log2(
        1.0 + np.min(common_sinr)
    )

    r_private = np.log2(
        1.0 + np.asarray(private_sinr)
    )

    sum_rate = r_common + np.sum(r_private)

    return (
        r_common,
        r_private[0],
        r_private[1],
        sum_rate,
    )


def run_rsma():
    out = np.zeros(
        (MONTE_CARLO_RSMA, 4),
        dtype=float
    )

    for i in range(MONTE_CARLO_RSMA):
        out[i] = rsma_trial()

    return {
        "common_rate": float(np.mean(out[:, 0])),
        "private_rate_u1": float(np.mean(out[:, 1])),
        "private_rate_u2": float(np.mean(out[:, 2])),
        "sum_rate": float(np.mean(out[:, 3])),
        "sum_rate_std": float(np.std(out[:, 3])),
    }


# ============================================================
# ISAC sensing
# ============================================================

def qpsk_symbols(n):
    b0 = rng.integers(0, 2, n)
    b1 = rng.integers(0, 2, n)
    return (
        (2 * b0 - 1)
        + 1j * (2 * b1 - 1)
    ) / np.sqrt(2.0)


def make_ofdm_symbol():
    X = qpsk_symbols(N_SUB)
    x = np.fft.ifft(X) * np.sqrt(N_SUB)
    cp = x[-CP_LEN:]
    return np.concatenate([cp, x])


def apply_delay_doppler(x, delay_samples, fd_hz):
    """
    Fractional delay is represented by an integer sample delay for
    the reproducible low-complexity estimator. Doppler is applied
    over the received time samples.
    """
    y = np.zeros_like(x, dtype=complex)

    if delay_samples < len(x):
        y[delay_samples:] = x[:len(x) - delay_samples]

    n = np.arange(len(x))
    y *= np.exp(
        1j * 2.0 * np.pi * fd_hz * n / FS
    )

    return y


def calibrate_detection_threshold():
    """
    Noise-only matched-filter power threshold for the target PFA.
    """
    powers = []

    ref = make_ofdm_symbol()
    ref_power = np.vdot(ref, ref).real

    for _ in range(1000):
        noise = complex_awgn(
            ref.shape,
            1.0
        )
        corr = np.vdot(ref, noise)
        powers.append(
            np.abs(corr) ** 2
            / (ref_power + 1e-30)
        )

    return float(
        np.quantile(
            powers,
            1.0 - PFA_TARGET
        )
    )


def isac_trial(threshold):
    """
    One target-present ISAC trial.
    """
    tx = make_ofdm_symbol()

    delay_seconds = (
        2.0 * TARGET_RANGE_M / C
    )

    delay_samples = int(
        round(
            delay_seconds * FS
        )
    )

    rx_target = apply_delay_doppler(
        tx,
        delay_samples,
        TARGET_DOPPLER_HZ
    )

    signal_power = np.mean(
        np.abs(rx_target) ** 2
    )

    noise_power = signal_power / SNR_LIN

    noise = complex_awgn(
        rx_target.shape,
        noise_power
    )

    rx = rx_target + noise

    # Delay estimation by correlation over candidate delays.
    max_delay = min(
        int(2 * 2500.0 / C * FS),
        len(tx) - 1
    )

    if max_delay < 1:
        max_delay = 1

    metrics = np.zeros(
        max_delay + 1
    )

    for d in range(max_delay + 1):
        ref = tx[:len(tx) - d]
        obs = rx[d:]
        if len(ref) == 0:
            continue
        metrics[d] = np.abs(
            np.vdot(ref, obs)
        ) ** 2

    delay_hat = int(
        np.argmax(metrics)
    )

    range_hat = (
        delay_hat / FS
    ) * C / 2.0

    # Doppler estimation using phase slope after estimated delay.
    if delay_hat < len(tx) - 4:
        ref = tx[:len(tx) - delay_hat]
        obs = rx[delay_hat:]

        z = obs * np.conj(ref)

        phase = np.unwrap(
            np.angle(z + 1e-30)
        )

        n = np.arange(len(phase))
        if len(n) > 2:
            slope = np.polyfit(
                n,
                phase,
                1
            )[0]

            fd_hat = (
                slope * FS
                / (2.0 * np.pi)
            )
        else:
            fd_hat = 0.0
    else:
        fd_hat = 0.0

    # Matched-filter statistic at estimated delay.
    stat = (
        metrics[delay_hat]
        /
        (
            np.vdot(
                tx[:len(tx)-delay_hat],
                tx[:len(tx)-delay_hat]
            ).real
            + 1e-30
        )
    )

    detected = (
        stat > threshold
    )

    sensing_sinr = (
        signal_power / noise_power
    )

    return (
        detected,
        range_hat,
        fd_hat,
        sensing_sinr
    )


def run_isac():
    threshold = calibrate_detection_threshold()

    detections = []
    range_errors = []
    doppler_errors = []
    sinrs = []

    for _ in range(MONTE_CARLO_ISAC):

        det, rhat, fhat, ssinr = (
            isac_trial(threshold)
        )

        detections.append(
            1.0 if det else 0.0
        )

        range_errors.append(
            rhat - TARGET_RANGE_M
        )

        doppler_errors.append(
            fhat - TARGET_DOPPLER_HZ
        )

        sinrs.append(
            ssinr
        )

    range_errors = np.asarray(
        range_errors
    )

    doppler_errors = np.asarray(
        doppler_errors
    )

    return {
        "detection_probability":
            float(np.mean(detections)),
        "range_rmse_m":
            float(np.sqrt(np.mean(range_errors**2))),
        "doppler_rmse_hz":
            float(np.sqrt(np.mean(doppler_errors**2))),
        "sensing_sinr_db":
            float(db10(np.mean(sinrs))),
        "threshold":
            float(threshold),
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("RSMA + ISAC SUPPLEMENTARY REAL EVALUATION")
    print("=" * 72)

    print()
    print("Parameters")
    print("-" * 72)
    print(f"Nsub                  : {N_SUB}")
    print(f"CP length             : {CP_LEN}")
    print(f"Subcarrier spacing    : {DELTA_F/1000:.1f} kHz")
    print(f"Sampling frequency    : {FS/1000:.1f} kHz")
    print(f"SNR                   : {SNR_DB:.1f} dB")
    print(f"ISAC Doppler          : {ISAC_FD:.1f} Hz")
    print(f"ISAC paths             : {ISAC_PATHS}")
    print(f"RSMA Monte Carlo      : {MONTE_CARLO_RSMA}")
    print(f"ISAC Monte Carlo      : {MONTE_CARLO_ISAC}")

    print()
    print("=" * 72)
    print("RSMA RESULTS")
    print("=" * 72)

    rsma = run_rsma()

    print(
        f"Mean common-stream rate : "
        f"{rsma['common_rate']:.6f} bit/s/Hz"
    )

    print(
        f"Mean private rate U1    : "
        f"{rsma['private_rate_u1']:.6f} bit/s/Hz"
    )

    print(
        f"Mean private rate U2    : "
        f"{rsma['private_rate_u2']:.6f} bit/s/Hz"
    )

    print(
        f"Mean RSMA sum rate      : "
        f"{rsma['sum_rate']:.6f} bit/s/Hz"
    )

    print(
        f"Sum-rate std            : "
        f"{rsma['sum_rate_std']:.6f}"
    )

    print()
    print("=" * 72)
    print("ISAC SENSING RESULTS")
    print("=" * 72)

    isac = run_isac()

    print(
        f"Detection probability   : "
        f"{isac['detection_probability']:.6f}"
    )

    print(
        f"Range RMSE              : "
        f"{isac['range_rmse_m']:.6f} m"
    )

    print(
        f"Doppler RMSE            : "
        f"{isac['doppler_rmse_hz']:.6f} Hz"
    )

    print(
        f"Sensing SINR            : "
        f"{isac['sensing_sinr_db']:.6f} dB"
    )

    print()
    print("=" * 72)
    print("RESULTS SAVED")
    print("=" * 72)

    result_file = Path(
        "RSMA_ISAC_REAL_EVALUATION_RESULTS.txt"
    )

    with result_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "RSMA + ISAC SUPPLEMENTARY REAL EVALUATION\n"
        )

        f.write("=" * 72 + "\n")

        f.write(
            f"Nsub = {N_SUB}\n"
        )

        f.write(
            f"CP = {CP_LEN}\n"
        )

        f.write(
            f"Delta_f = {DELTA_F}\n"
        )

        f.write(
            f"Fs = {FS}\n"
        )

        f.write(
            f"SNR_dB = {SNR_DB}\n"
        )

        f.write(
            f"ISAC_FD = {ISAC_FD}\n"
        )

        f.write(
            f"ISAC_PATHS = {ISAC_PATHS}\n\n"
        )

        f.write("RSMA\n")
        f.write(
            f"Common rate = "
            f"{rsma['common_rate']:.9f} bit/s/Hz\n"
        )

        f.write(
            f"Private rate U1 = "
            f"{rsma['private_rate_u1']:.9f} bit/s/Hz\n"
        )

        f.write(
            f"Private rate U2 = "
            f"{rsma['private_rate_u2']:.9f} bit/s/Hz\n"
        )

        f.write(
            f"RSMA sum rate = "
            f"{rsma['sum_rate']:.9f} bit/s/Hz\n"
        )

        f.write(
            f"Sum-rate std = "
            f"{rsma['sum_rate_std']:.9f}\n\n"
        )

        f.write("ISAC\n")

        f.write(
            f"Detection probability = "
            f"{isac['detection_probability']:.9f}\n"
        )

        f.write(
            f"Range RMSE = "
            f"{isac['range_rmse_m']:.9f} m\n"
        )

        f.write(
            f"Doppler RMSE = "
            f"{isac['doppler_rmse_hz']:.9f} Hz\n"
        )

        f.write(
            f"Sensing SINR = "
            f"{isac['sensing_sinr_db']:.9f} dB\n"
        )

        f.write(
            f"Detection threshold = "
            f"{isac['threshold']:.9f}\n"
        )

    print(
        f"Saved: {result_file}"
    )

    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
