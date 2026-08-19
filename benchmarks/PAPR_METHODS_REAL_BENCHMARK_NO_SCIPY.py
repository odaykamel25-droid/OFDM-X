# ============================================================
# PAPR_METHODS_REAL_BENCHMARK.py
# ============================================================
# REAL PAPR / COMPLEXITY BENCHMARK FOR REVIEWER #3
#
# Common conditions:
#   N = 64 subcarriers
#   BPSK
#   4000 OFDM waveforms
#   CP = 16 (not included in PAPR calculation)
#   Oversampling factor L = 1 (no oversampling)
#
# Methods:
#   1) Conventional OFDM
#   2) SLM: 4 QPSK phase candidates
#   3) PTS: 4 contiguous subblocks, phase set {1,-1,j,-j},
#      exhaustive search over 4^3 = 64 relative combinations
#   4) DCT spreading: orthonormal DCT-II implemented with NumPy
#      (NO SciPy required)
#   5) WHT spreading: normalized Walsh-Hadamard transform
#   6) Companding: mu-law, mu=255
#   7) FCAE / OFDM-X: established project reference values
#
# IMPORTANT:
#   DCT/WHT are generic transform-spreading OFDM baselines.
#   They are NOT claimed to reproduce the AFDM/OCDM implementation
#   of Ref. [5].
#
# Established FCAE reference values:
#   Mean PAPR              = 5.997 dB
#   PAPR improvement       = 0.194 dB
#   Parameters             = 148,417
#   Approximate FLOPs      = 294,912
#   Median latency         = 2.817 ms / waveform
#
# Outputs:
#   PAPR_METHODS_REAL_BENCHMARK.csv
#   PAPR_METHODS_REAL_BENCHMARK.txt
# ============================================================

import csv
import time
from pathlib import Path

import numpy as np


# ============================================================
# 1. COMMON PARAMETERS
# ============================================================

SEED = 20260808

N = 64
CP = 16
NUM_SYMBOLS = 4000
OVERSAMPLING = 1

SLM_CANDIDATES = 4

PTS_SUBBLOCKS = 4
PTS_PHASES = np.array(
    [1.0, -1.0, 1.0j, -1.0j],
    dtype=np.complex128
)

COMPANDING_MU = 255.0

# Established FCAE project values
FCAE_PAPR_DB = 5.997
FCAE_IMPROVEMENT_DB = 0.194
FCAE_PARAMETERS = 148_417
FCAE_FLOPS = 294_912
FCAE_LATENCY_MS = 2.817


# ============================================================
# 2. RANDOM DATA
# ============================================================

rng = np.random.default_rng(SEED)

bits = rng.integers(
    0,
    2,
    size=(NUM_SYMBOLS, N),
    dtype=np.int8
)

# BPSK: 0 -> -1, 1 -> +1
X = (
    2.0 * bits - 1.0
).astype(np.complex128)


# ============================================================
# 3. PAPR
# ============================================================

def papr_per_symbol_db(signals):
    """Return PAPR in dB for every waveform."""
    power = np.abs(signals) ** 2

    return (
        10.0
        * np.log10(
            np.max(power, axis=1)
            /
            (
                np.mean(power, axis=1)
                + 1e-12
            )
        )
    )


def mean_papr_db(signals):
    return float(
        np.mean(
            papr_per_symbol_db(signals)
        )
    )


# Approximate radix-2 complex FFT operation count.
def fft_complex_flops(n):
    return 5.0 * n * np.log2(n)


# ============================================================
# 4. ORTHONORMAL DCT-II USING NUMPY ONLY
# ============================================================
# This replaces scipy.fft.dct.
#
# DCT-II:
#   X[k] = alpha(k) * sum_n x[n] cos(pi/N*(n+1/2)*k)
#
# with orthonormal scaling.
# ============================================================

def dct_ortho(x):
    """
    Orthonormal DCT-II implemented with NumPy only.

    Input:
        x : shape (num_waveforms, N)

    Output:
        transformed array with same shape.

    No SciPy dependency.
    """
    n = x.shape[1]

    sample_index = np.arange(n)[:, None]
    freq_index = np.arange(n)[None, :]

    basis = np.cos(
        np.pi
        / n
        * (sample_index + 0.5)
        * freq_index
    )

    basis[:, 0] /= np.sqrt(n)

    if n > 1:
        basis[:, 1:] *= np.sqrt(
            2.0 / n
        )

    return x @ basis


# ============================================================
# 5. FAST WALSH-HADAMARD MATRIX
# ============================================================

def hadamard_matrix(n):
    """Return normalized Hadamard matrix for power-of-two n."""
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(
            "WHT requires N to be a power of two."
        )

    H = np.array(
        [[1.0]],
        dtype=np.float64
    )

    while H.shape[0] < n:
        H = np.block(
            [
                [H, H],
                [H, -H],
            ]
        )

    return H / np.sqrt(n)


WHT = hadamard_matrix(N)


# ============================================================
# 6. CONVENTIONAL OFDM
# ============================================================

t0 = time.perf_counter()

x_ofdm = np.fft.ifft(
    X,
    axis=1
)

elapsed_ofdm = (
    time.perf_counter()
    - t0
)

baseline_papr = mean_papr_db(
    x_ofdm
)

results = []


results.append(
    {
        "Method": "Conventional OFDM",
        "Mean_PAPR_dB": baseline_papr,
        "PAPR_Improvement_dB": 0.0,
        "Measured_ms_per_waveform":
            1000.0
            * elapsed_ofdm
            / NUM_SYMBOLS,
        "Estimated_FLOPs":
            fft_complex_flops(N),
        "Notes":
            "One 64-point IFFT; L=1"
    }
)


# ============================================================
# 7. SLM
# ============================================================
# Four reproducible QPSK phase sequences.
# The same candidate sequences are applied across the batch.
# The lowest-PAPR candidate is selected per waveform.
# ============================================================

def run_slm(X_data, seed, candidates=4):

    local_rng = np.random.default_rng(
        seed
    )

    n_waveforms = X_data.shape[0]

    best_papr = np.full(
        n_waveforms,
        np.inf,
        dtype=np.float64
    )

    best_signal = np.empty_like(
        X_data,
        dtype=np.complex128
    )

    phase_set = np.array(
        [1.0, -1.0, 1.0j, -1.0j],
        dtype=np.complex128
    )

    # Candidate phase sequences.
    phase_sequences = local_rng.choice(
        phase_set,
        size=(candidates, N)
    )

    # Fix one reference subcarrier to unity.
    phase_sequences[:, 0] = 1.0

    for u in range(candidates):

        phases = (
            phase_sequences[u]
            [None, :]
        )

        X_rotated = (
            X_data
            * phases
        )

        x_rotated = np.fft.ifft(
            X_rotated,
            axis=1
        )

        p = np.abs(
            x_rotated
        ) ** 2

        candidate_papr = (
            np.max(
                p,
                axis=1
            )
            /
            (
                np.mean(
                    p,
                    axis=1
                )
                + 1e-12
            )
        )

        mask = (
            candidate_papr
            < best_papr
        )

        best_papr[mask] = (
            candidate_papr[mask]
        )

        best_signal[mask] = (
            x_rotated[mask]
        )

    return best_signal


t0 = time.perf_counter()

x_slm = run_slm(
    X,
    SEED + 1,
    SLM_CANDIDATES
)

elapsed_slm = (
    time.perf_counter()
    - t0
)

slm_papr = mean_papr_db(
    x_slm
)

results.append(
    {
        "Method": "SLM",
        "Mean_PAPR_dB": slm_papr,
        "PAPR_Improvement_dB":
            baseline_papr - slm_papr,
        "Measured_ms_per_waveform":
            1000.0
            * elapsed_slm
            / NUM_SYMBOLS,
        "Estimated_FLOPs":
            SLM_CANDIDATES
            * fft_complex_flops(N),
        "Notes":
            f"{SLM_CANDIDATES} QPSK phase candidates"
    }
)


# ============================================================
# 8. PTS
# ============================================================
# Four contiguous subblocks.
# First phase is fixed to 1 to remove common phase ambiguity.
# Remaining 3 phases have 4 choices each => 4^3 = 64 combos.
# ============================================================

def run_pts(X_data, subblocks=4):

    if N % subblocks != 0:
        raise ValueError(
            "N must be divisible by PTS_SUBBLOCKS."
        )

    n_waveforms = X_data.shape[0]
    block_len = N // subblocks

    partial = np.empty(
        (
            n_waveforms,
            subblocks,
            N
        ),
        dtype=np.complex128
    )

    # Partial IFFTs.
    for v in range(subblocks):

        masked = np.zeros_like(
            X_data
        )

        start = v * block_len
        stop = (
            v + 1
        ) * block_len

        masked[
            :,
            start:stop
        ] = X_data[
            :,
            start:stop
        ]

        partial[
            :,
            v,
            :
        ] = np.fft.ifft(
            masked,
            axis=1
        )

    combinations = np.array(
        np.meshgrid(
            PTS_PHASES,
            PTS_PHASES,
            PTS_PHASES,
            indexing="ij"
        )
    ).reshape(
        3,
        -1
    ).T

    best_papr = np.full(
        n_waveforms,
        np.inf,
        dtype=np.float64
    )

    best_signal = np.empty(
        (
            n_waveforms,
            N
        ),
        dtype=np.complex128
    )

    # Process combinations in small chunks.
    for start in range(
        0,
        len(combinations),
        16
    ):

        c = combinations[
            start:start + 16
        ]

        candidate = (
            partial[
                :,
                0,
                None,
                :
            ]
            +
            c[
                None,
                :,
                0,
                None
            ]
            * partial[
                :,
                1,
                None,
                :
            ]
            +
            c[
                None,
                :,
                1,
                None
            ]
            * partial[
                :,
                2,
                None,
                :
            ]
            +
            c[
                None,
                :,
                2,
                None
            ]
            * partial[
                :,
                3,
                None,
                :
            ]
        )

        power = np.abs(
            candidate
        ) ** 2

        candidate_papr = (
            np.max(
                power,
                axis=2
            )
            /
            (
                np.mean(
                    power,
                    axis=2
                )
                + 1e-12
            )
        )

        local_index = np.argmin(
            candidate_papr,
            axis=1
        )

        local_best = candidate_papr[
            np.arange(
                n_waveforms
            ),
            local_index
        ]

        mask = (
            local_best
            < best_papr
        )

        if np.any(mask):

            best_papr[mask] = (
                local_best[mask]
            )

            best_signal[mask] = (
                candidate[
                    np.arange(
                        n_waveforms
                    )[mask],
                    local_index[mask],
                    :
                ]
            )

    return best_signal


t0 = time.perf_counter()

x_pts = run_pts(
    X,
    PTS_SUBBLOCKS
)

elapsed_pts = (
    time.perf_counter()
    - t0
)

pts_papr = mean_papr_db(
    x_pts
)

pts_combinations = (
    len(PTS_PHASES)
    ** (
        PTS_SUBBLOCKS - 1
    )
)

results.append(
    {
        "Method": "PTS",
        "Mean_PAPR_dB": pts_papr,
        "PAPR_Improvement_dB":
            baseline_papr - pts_papr,
        "Measured_ms_per_waveform":
            1000.0
            * elapsed_pts
            / NUM_SYMBOLS,
        "Estimated_FLOPs":
            (
                PTS_SUBBLOCKS
                * fft_complex_flops(N)
                +
                pts_combinations
                * N
            ),
        "Notes":
            f"{PTS_SUBBLOCKS} contiguous subblocks; "
            f"{pts_combinations} phase combinations"
    }
)


# ============================================================
# 9. DCT SPREADING
# ============================================================

t0 = time.perf_counter()

X_dct = dct_ortho(
    X.real
)

x_dct = np.fft.ifft(
    X_dct.astype(
        np.complex128
    ),
    axis=1
)

elapsed_dct = (
    time.perf_counter()
    - t0
)

dct_papr = mean_papr_db(
    x_dct
)

results.append(
    {
        "Method": "DCT spreading",
        "Mean_PAPR_dB": dct_papr,
        "PAPR_Improvement_dB":
            baseline_papr - dct_papr,
        "Measured_ms_per_waveform":
            1000.0
            * elapsed_dct
            / NUM_SYMBOLS,
        "Estimated_FLOPs":
            (
                2.0 * N * N
                +
                fft_complex_flops(N)
            ),
        "Notes":
            "Orthonormal DCT-II + one 64-point IFFT; NumPy-only"
    }
)


# ============================================================
# 10. WHT SPREADING
# ============================================================

t0 = time.perf_counter()

X_wht = (
    X.real
    @ WHT
)

x_wht = np.fft.ifft(
    X_wht.astype(
        np.complex128
    ),
    axis=1
)

elapsed_wht = (
    time.perf_counter()
    - t0
)

wht_papr = mean_papr_db(
    x_wht
)

results.append(
    {
        "Method": "WHT spreading",
        "Mean_PAPR_dB": wht_papr,
        "PAPR_Improvement_dB":
            baseline_papr - wht_papr,
        "Measured_ms_per_waveform":
            1000.0
            * elapsed_wht
            / NUM_SYMBOLS,
        "Estimated_FLOPs":
            (
                N * np.log2(N)
                +
                fft_complex_flops(N)
            ),
        "Notes":
            "Normalized WHT + one 64-point IFFT"
    }
)


# ============================================================
# 11. COMPANDING
# ============================================================
# Mu-law companding with mu=255.
# Average power is restored after companding.
# Since PAPR is scale invariant, restoration does not change PAPR.
# ============================================================

def run_companding(
    x,
    mu=255.0
):

    magnitude = np.abs(
        x
    )

    phase = np.angle(
        x
    )

    peak = np.max(
        magnitude,
        axis=1,
        keepdims=True
    )

    normalized = (
        magnitude
        /
        (
            peak
            + 1e-12
        )
    )

    compressed = (
        np.log1p(
            mu
            * normalized
        )
        /
        np.log1p(
            mu
        )
    )

    out = (
        compressed
        *
        np.exp(
            1j * phase
        )
    )

    input_power = np.mean(
        np.abs(x) ** 2,
        axis=1,
        keepdims=True
    )

    output_power = np.mean(
        np.abs(out) ** 2,
        axis=1,
        keepdims=True
    )

    out *= np.sqrt(
        input_power
        /
        (
            output_power
            + 1e-12
        )
    )

    return out


t0 = time.perf_counter()

x_comp = run_companding(
    x_ofdm,
    COMPANDING_MU
)

elapsed_comp = (
    time.perf_counter()
    - t0
)

comp_papr = mean_papr_db(
    x_comp
)

results.append(
    {
        "Method": "Companding",
        "Mean_PAPR_dB": comp_papr,
        "PAPR_Improvement_dB":
            baseline_papr - comp_papr,
        "Measured_ms_per_waveform":
            1000.0
            * elapsed_comp
            / NUM_SYMBOLS,
        "Estimated_FLOPs":
            (
                fft_complex_flops(N)
                +
                8.0 * N
            ),
        "Notes":
            f"Mu-law companding; mu={COMPANDING_MU:.0f}"
    }
)


# ============================================================
# 12. FCAE ESTABLISHED PROJECT REFERENCE
# ============================================================

results.append(
    {
        "Method": "FCAE / OFDM-X",
        "Mean_PAPR_dB":
            FCAE_PAPR_DB,
        "PAPR_Improvement_dB":
            FCAE_IMPROVEMENT_DB,
        "Measured_ms_per_waveform":
            FCAE_LATENCY_MS,
        "Estimated_FLOPs":
            FCAE_FLOPS,
        "Notes":
            (
                "Established project result; "
                f"{FCAE_PARAMETERS} trainable parameters; "
                "2.817 ms median batch=1 CPU latency"
            )
    }
)


# ============================================================
# 13. SAVE CSV
# ============================================================

out_dir = Path(
    __file__
).resolve().parent

csv_path = (
    out_dir
    /
    "PAPR_METHODS_REAL_BENCHMARK.csv"
)

fieldnames = [
    "Method",
    "Mean_PAPR_dB",
    "PAPR_Improvement_dB",
    "Measured_ms_per_waveform",
    "Estimated_FLOPs",
    "Notes",
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 14. SAVE TEXT SUMMARY
# ============================================================

txt_path = (
    out_dir
    /
    "PAPR_METHODS_REAL_BENCHMARK.txt"
)

with open(
    txt_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "REAL PAPR / COMPLEXITY BENCHMARK\n"
    )

    f.write(
        "=" * 90
        + "\n"
    )

    f.write(
        f"Nsub                  : {N}\n"
        f"CP length             : {CP}\n"
        f"BPSK                  : Yes\n"
        f"Number of waveforms   : {NUM_SYMBOLS}\n"
        f"Oversampling factor   : L={OVERSAMPLING} (no oversampling)\n"
        f"SLM candidates        : {SLM_CANDIDATES}\n"
        f"PTS subblocks         : {PTS_SUBBLOCKS}\n"
        f"PTS phase combinations: {pts_combinations}\n"
        f"Companding mu         : {COMPANDING_MU:.0f}\n\n"
    )

    f.write(
        "RESULTS\n"
        + "-" * 90
        + "\n"
    )

    for row in results:

        f.write(
            f"{row['Method']:<24}"
            f"PAPR={row['Mean_PAPR_dB']:.4f} dB | "
            f"Improvement={row['PAPR_Improvement_dB']:.4f} dB | "
            f"Time={row['Measured_ms_per_waveform']:.6f} ms/waveform | "
            f"FLOPs~{row['Estimated_FLOPs']:.0f}\n"
        )

    f.write(
        "\nIMPORTANT NOTES\n"
        + "-" * 90
        + "\n"
    )

    f.write(
        "1. All non-FCAE methods were evaluated with N=64, "
        "BPSK, 4000 waveforms, and L=1.\n"
    )

    f.write(
        "2. DCT and WHT are generic transform-spreading "
        "OFDM baselines; they are not a reproduction of "
        "the AFDM/OCDM implementation of Ref. [5].\n"
    )

    f.write(
        "3. FCAE values are the established project reference "
        "measurements and were not retrained inside this benchmark.\n"
    )

    f.write(
        "4. The measured wall-clock times are Python/NumPy "
        "implementation timings on the benchmark machine and "
        "should not be interpreted as hardware-independent latency.\n"
    )


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("=" * 90)
print(
    "REAL PAPR / COMPLEXITY BENCHMARK COMPLETE"
)
print("=" * 90)

for row in results:

    print(
        f"{row['Method']:<24}"
        f"PAPR={row['Mean_PAPR_dB']:.4f} dB | "
        f"Improvement={row['PAPR_Improvement_dB']:.4f} dB | "
        f"Time={row['Measured_ms_per_waveform']:.6f} ms | "
        f"FLOPs~{row['Estimated_FLOPs']:.0f}"
    )

print()
print("Saved:")
print(csv_path)
print(txt_path)
print("=" * 90)
