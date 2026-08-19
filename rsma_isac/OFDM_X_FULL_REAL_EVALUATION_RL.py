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

The ISAC part uses a coherent burst of 32 known OFDM reference symbols,
a target delay and Doppler, AWGN, matched filtering / correlation, and
slow-time phase progression for Doppler estimation. It reports:
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
# OFDM-X adaptive waveform engine -- REAL Actor-Critic
# ============================================================

# The Actor-Critic implementation below is adapted directly from the
# verified Figure 13 adaptive OFDM/AFDM evaluation code in the uploaded
# project archive. It measures paired BER for both waveform actions on
# the same channel/noise realization and learns the action utility.

# 3. BPSK / transforms
# ============================================================
def generate_bpsk():
    bits = rng.integers(0, 2, size=N_SUB, dtype=np.int8)
    x = (2 * bits - 1).astype(np.complex128)
    return bits, x


# Real Actor-Critic training configuration inherited from the verified Figure 13 implementation.
RL_SCENARIOS = {
    "Vehicular": {"fd": 200.0, "paths": 1},
    "LEO": {"fd": 1000.0, "paths": 3},
    "ISAC": {"fd": 450.0, "paths": 7},
}
RL_DOPPLER_SWEEP = np.arange(0.0, 2001.0, 100.0)
RL_TRAIN_SYMBOLS_PER_EPISODE = 512
RL_TRAIN_BLOCKS = int(np.ceil(RL_TRAIN_SYMBOLS_PER_EPISODE / N_SUB))
RL_EVAL_BLOCKS = int(np.ceil(4000 / N_SUB))
RL_TRAIN_EPISODES = 1200
RL_AFDM_COMPLEXITY_PENALTY = 0.0050
RL_UTILITY_LEARNING_SCALE = 100.0

def dft_matrix():
    n = np.arange(N_SUB)
    k = np.arange(N_SUB)
    return np.exp(-2j * np.pi * np.outer(k, n) / N_SUB) / np.sqrt(N_SUB)


RL_F = dft_matrix()


def afdm_matrix(fd_ref):
    # DAFT parameterization based on the maximum Doppler used by
    # the corresponding channel condition.
    alpha = max(1, int(np.ceil(float(fd_ref) / DELTA_F)))
    c1 = (2.0 * alpha + 1.0) / (2.0 * N_SUB)
    c2 = 1.0 / (2.0 * N_SUB)

    n = np.arange(N_SUB)
    D1 = np.diag(np.exp(-1j * 2 * np.pi * c1 * n**2))
    D2 = np.diag(np.exp(-1j * 2 * np.pi * c2 * n**2))
    return D2 @ RL_F @ D1


def ofdm_modulate(x):
    useful = RL_F.conj().T @ x
    return np.concatenate([useful[-CP_LEN:], useful])


def afdm_modulate(x, fd_ref):
    A = afdm_matrix(fd_ref)
    useful = A.conj().T @ x
    return np.concatenate([useful[-CP_LEN:], useful])

# ============================================================
# 4. Time-varying multipath channel
# ============================================================
def make_channel(fd_max, paths):
    paths = int(paths)

    delays = rng.integers(0, CP_LEN, size=paths)

    # Exponential power-delay profile.
    p = np.exp(-0.6 * np.arange(paths))
    p /= np.sum(p)

    gains = (
        rng.normal(size=paths)
        + 1j * rng.normal(size=paths)
    ) / np.sqrt(2.0)
    gains *= np.sqrt(p)
    gains /= np.sqrt(np.sum(np.abs(gains) ** 2) + 1e-12)

    dopplers = rng.uniform(-fd_max, fd_max, size=paths)

    # H_time: actual time-varying channel.
    # H_static: zero-Doppler reference channel used by the
    # conventional OFDM one-tap receiver.
    H_time = np.zeros((N_SUB, N_SUB), dtype=np.complex128)
    H_static = np.zeros((N_SUB, N_SUB), dtype=np.complex128)
    n = np.arange(N_SUB)

    for delay, gain, fd in zip(delays, gains, dopplers):
        phase = np.exp(1j * 2 * np.pi * fd * n / FS)
        for nn in range(N_SUB):
            kk = (nn - int(delay)) % N_SUB
            H_time[nn, kk] += gain * phase[nn]
            H_static[nn, kk] += gain

    return H_time, H_static

def add_common_awgn(y_o_clean, y_a_clean):
    # The same normalized complex Gaussian sample is used for the
    # paired comparison. Each waveform is normalized by its own
    # received signal power, so the comparison is not biased by
    # different transform-domain power distributions.
    seed = int(rng.integers(0, 2**31 - 1))
    local = np.random.default_rng(seed)

    w = (
        local.normal(size=N_SUB)
        + 1j * local.normal(size=N_SUB)
    ) / np.sqrt(2.0)

    def add(y):
        p = np.mean(np.abs(y) ** 2)
        noise_power = p / (10 ** (SNR_DB / 10.0))
        return y + np.sqrt(noise_power) * w

    return add(y_o_clean), add(y_a_clean)

# ============================================================
# 5. Waveform-specific practical receivers
# ============================================================
# Conventional OFDM uses a standard one-tap frequency-domain
# equalizer based on the zero-Doppler/reference channel. This is
# deliberately a practical receiver: time variation produces ICI
# that is not explicitly inverted.
#
# AFDM uses a DAFT-domain full linear-MMSE equalizer with the
# measured time-varying channel matrix. The adaptive framework
# therefore selects both the waveform and its corresponding
# receiver processing chain.

def ofdm_one_tap_detect(y, H_static):
    z = RL_F @ y
    Hf = RL_F @ H_static @ RL_F.conj().T
    h = np.diag(Hf)

    snr = 10 ** (SNR_DB / 10.0)
    return (np.conj(h) / (np.abs(h) ** 2 + 1.0 / snr)) * z


def afdm_mmse_detect(y, H_time, A):
    z = A @ y
    Heff = A @ H_time @ A.conj().T

    snr = 10 ** (SNR_DB / 10.0)
    reg = 1.0 / snr

    M = Heff.conj().T @ Heff + reg * np.eye(N_SUB)
    b = Heff.conj().T @ z
    return np.linalg.solve(M, b)
# 6. Real paired BER measurement
# ============================================================
def paired_ber_block(fd_max, paths):
    bits, x = generate_bpsk()

    H_time, H_static = make_channel(fd_max, paths)
    A = afdm_matrix(max(float(fd_max), DELTA_F))

    tx_o = ofdm_modulate(x)
    tx_a = np.concatenate([
        (A.conj().T @ x)[-CP_LEN:],
        A.conj().T @ x
    ])

    # Same physical channel realization.
    y_o_clean = H_time @ tx_o[CP_LEN:]
    y_a_clean = H_time @ tx_a[CP_LEN:]

    y_o, y_a = add_common_awgn(
        y_o_clean,
        y_a_clean
    )

    x_o = ofdm_one_tap_detect(y_o, H_static)
    x_a = afdm_mmse_detect(y_a, H_time, A)

    bh_o = (np.real(x_o) >= 0).astype(np.int8)
    bh_a = (np.real(x_a) >= 0).astype(np.int8)

    return (
        int(np.sum(bh_o != bits)),
        int(np.sum(bh_a != bits)),
        N_SUB
    )


def paired_ber(fd_max, paths, blocks):
    err_o = 0
    err_a = 0
    total = 0

    for _ in range(int(blocks)):
        eo, ea, n = paired_ber_block(fd_max, paths)
        err_o += eo
        err_a += ea
        total += n

    return err_o / total, err_a / total

# ============================================================
# 7. Actor-Critic policy
# ============================================================
class ActorCritic:
    # State = normalized Doppler, normalized paths, bias.
    # Action 0 = OFDM, action 1 = AFDM.
    def __init__(self):
        self.Wa = rng.normal(0, 0.02, (3, 2))
        self.ba = np.array([0.5, -0.5], dtype=float)
        self.Wv = rng.normal(0, 0.02, 3)
        self.bv = 0.0
        self.actor_lr = 0.08
        self.critic_lr = 0.10

    def state(self, fd, paths):
        return np.array([
            float(fd) / RL_DOPPLER_SWEEP[-1],
            float(paths) / 7.0,
            1.0
        ])

    def probs(self, s):
        z = s @ self.Wa + self.ba
        z -= np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def value(self, s):
        return float(s @ self.Wv + self.bv)

    def sample(self, s, epsilon):
        p = self.probs(s)
        if rng.random() < epsilon:
            a = int(rng.integers(0, 2))
        else:
            a = int(rng.choice(2, p=p))
        return a, p

    def update_counterfactual(self, s, u_ofdm, u_afdm):
        # Counterfactual contextual actor-critic update:
        # both actions are evaluated on the same paired BER trial,
        # so the actor receives information about both choices.
        p = self.probs(s)
        utilities = np.array([u_ofdm, u_afdm], dtype=float)
        expected = float(np.dot(p, utilities))

        # Critic learns the expected utility under the current policy.
        advantage = np.clip(expected - self.value(s), -1.0, 1.0)
        self.Wv += self.critic_lr * advantage * s
        self.bv += self.critic_lr * advantage

        # Exact gradient of E_pi[utility] w.r.t. softmax logits.
        grad_logits = p * (utilities - expected)
        self.Wa += self.actor_lr * np.outer(s, grad_logits)
        self.ba += self.actor_lr * grad_logits

# ============================================================
# 8. Adaptive utility
# ============================================================
def utility(ber, action):
    # Higher utility is better.
    complexity = (
        0.0
        if action == 0
        else RL_AFDM_COMPLEXITY_PENALTY
    )
    return -float(ber) - complexity


def reward_from_pair(ber_o, ber_a, action):
    u_o = utility(ber_o, 0)
    u_a = utility(ber_a, 1)

    selected = u_o if action == 0 else u_a
    alternative = u_a if action == 0 else u_o

    # Positive reward = selected waveform has higher measured utility.
    scale = max(
        abs(u_o),
        abs(u_a),
        1.0 / (N_SUB * RL_TRAIN_BLOCKS),
        1e-6
    )

    return float(np.clip(
        (selected - alternative) / scale,
        -1.0,
        1.0
    ))

# ============================================================
# 9. Training
# ============================================================
def train_agent():
    agent = ActorCritic()

    print("\nTraining Actor-Critic from measured BER + complexity utility...")
    print(
        f"AFDM complexity penalty = "
        f"{RL_AFDM_COMPLEXITY_PENALTY:.4f}"
    )

    for ep in range(1, RL_TRAIN_EPISODES + 1):
        fd = float(rng.uniform(0, RL_DOPPLER_SWEEP[-1]))
        paths = int(rng.choice([1, 3, 7]))
        s = agent.state(fd, paths)

        epsilon = max(
            0.08,
            0.45 * (1.0 - ep / RL_TRAIN_EPISODES)
        )

        # Both waveform actions are measured on the same paired
        # channel/data/noise realization. The actor therefore learns
        # from counterfactual evidence rather than from a single
        # sampled action.
        bo, ba = paired_ber(
            fd,
            paths,
            RL_TRAIN_BLOCKS
        )

        uo = utility(bo, 0)
        ua = utility(ba, 1)
        p_before = agent.probs(s)

        agent.update_counterfactual(
            s,
            uo * RL_UTILITY_LEARNING_SCALE,
            ua * RL_UTILITY_LEARNING_SCALE
        )

        action = int(np.argmax(agent.probs(s)))
        label = "AFDM" if action else "OFDM"
        reward = (ua - uo)

        if ep % 100 == 0:
            print(
                f"Episode {ep:3d}/{RL_TRAIN_EPISODES} | "
                f"fd={fd:7.1f} Hz | paths={paths} | "
                f"OFDM BER={bo:.6f} | AFDM BER={ba:.6f} | "
                f"action={label:4s} | "
                f"P(AFDM)={p_before[1]:.3f} -> {agent.probs(s)[1]:.3f} | "
                f"U_AFDM-U_OFDM={reward:+.5f}"
            )

    return agent


def run_real_actor_critic_selection():
    """
    Train and evaluate the actual Actor-Critic policy from the verified
    Figure 13 implementation, using the same measured paired BER utility.
    """
    agent = train_agent()

    rows = []
    for name, cfg in RL_SCENARIOS.items():
        fd = cfg["fd"]
        paths = cfg["paths"]

        s = agent.state(fd, paths)
        p = agent.probs(s)
        action = int(np.argmax(p))

        bo, ba = paired_ber(
            fd,
            paths,
            RL_EVAL_BLOCKS
        )

        bx = bo if action == 0 else ba

        rows.append({
            "scenario": name,
            "doppler_hz": float(fd),
            "paths": int(paths),
            "ofdm_ber": float(bo),
            "afdm_ber": float(ba),
            "ofdm_x_ber": float(bx),
            "p_ofdm": float(p[0]),
            "p_afdm": float(p[1]),
            "action": int(action),
            "selected": "AFDM" if action else "CP-OFDM",
        })

    return {
        "agent": agent,
        "rows": rows,
    }


# Established FCAE V8 result. This is intentionally kept fixed and
# is not altered by the supplementary RSMA/ISAC evaluation.
FCAE_V8_PAPR_ORIGINAL_DB = 6.191
FCAE_V8_PAPR_DB = 5.997
FCAE_V8_PAPR_IMPROVEMENT_DB = 0.194

FCAE_V8_TEST_MSE = 2.614626e-04
FCAE_V8_POWER_ERROR_PERCENT = 1.4139
FCAE_V8_EVM_PROXY_PERCENT = 2.2868
FCAE_V8_BER_PROXY_OFDM = 0.0
FCAE_V8_BER_PROXY_FCAE = 0.0


def run_ofdm_x_full_evaluation():
    """
    Run the complete supplementary OFDM-X framework evaluation with
    the actual Actor-Critic selection engine.
    """
    rl_results = run_real_actor_critic_selection()

    # Use a fresh, fixed RNG stream for the established RSMA+ISAC
    # evaluation so RL training does not alter their reproducible
    # Monte Carlo results.
    global rng
    rng = np.random.default_rng(SEED)

    rsma_results = run_rsma()
    isac_results = run_isac()

    waveform_results = []
    for row in rl_results["rows"]:
        waveform_results.append({
            "scenario": row["scenario"],
            "state": np.array([
                row["doppler_hz"] / RL_DOPPLER_SWEEP[-1],
                row["paths"] / 7.0,
                1.0
            ]),
            "action_index": row["action"],
            "waveform": row["selected"],
            "doppler_hz": row["doppler_hz"],
            "paths": row["paths"],
            "p_ofdm": row["p_ofdm"],
            "p_afdm": row["p_afdm"],
            "ofdm_ber": row["ofdm_ber"],
            "afdm_ber": row["afdm_ber"],
            "ofdm_x_ber": row["ofdm_x_ber"],
        })

    return {
        "waveform_selection": waveform_results,
        "rsma": rsma_results,
        "isac": isac_results,
        "fcae": {
            "original_papr_db": FCAE_V8_PAPR_ORIGINAL_DB,
            "fcae_papr_db": FCAE_V8_PAPR_DB,
            "papr_improvement_db": FCAE_V8_PAPR_IMPROVEMENT_DB,
            "test_mse": FCAE_V8_TEST_MSE,
            "power_error_percent": FCAE_V8_POWER_ERROR_PERCENT,
            "evm_proxy_percent": FCAE_V8_EVM_PROXY_PERCENT,
            "ber_proxy_ofdm": FCAE_V8_BER_PROXY_OFDM,
            "ber_proxy_fcae": FCAE_V8_BER_PROXY_FCAE,
        },
    }

# ============================================================
# ISAC sensing -- V2 coherent OFDM burst
# ============================================================

# The V2 estimator uses a coherent burst of OFDM symbols and
# estimates Doppler from inter-symbol phase progression.
# This replaces the old single-symbol sample-slope estimator.

N_ISAC_SYMBOLS = 32
TARGET_RANGE_M = 150.0
TARGET_DOPPLER_HZ = 450.0
MONTE_CARLO_ISAC = 2000

# One complete CP-OFDM symbol duration.
T_SYM = (N_SUB + CP_LEN) / FS


def qpsk_symbols(n):
    b0 = rng.integers(0, 2, n)
    b1 = rng.integers(0, 2, n)
    return (
        (2 * b0 - 1)
        + 1j * (2 * b1 - 1)
    ) / np.sqrt(2.0)


def make_ofdm_burst(n_symbols=N_ISAC_SYMBOLS):
    """
    Generate a coherent burst of OFDM reference symbols.
    Each symbol has an independently generated known QPSK
    reference, while the burst timing is preserved.
    """
    burst = []

    for _ in range(n_symbols):
        X = qpsk_symbols(N_SUB)
        x = np.fft.ifft(X) * np.sqrt(N_SUB)
        cp = x[-CP_LEN:]
        burst.append(np.concatenate([cp, x]))

    return np.asarray(burst, dtype=complex)


def apply_delay_doppler_burst(
    burst,
    delay_samples,
    fd_hz
):
    """
    Apply an integer-sample target delay and coherent Doppler
    across the complete OFDM burst.

    Doppler is referenced to the absolute sample time so that
    phase progression is preserved from one OFDM symbol to the
    next.
    """
    n_symbols, n_samples = burst.shape

    y = np.zeros_like(burst, dtype=complex)

    if delay_samples < n_samples:
        y[:, delay_samples:] = (
            burst[:, :n_samples - delay_samples]
        )

    absolute_n = np.arange(n_symbols * n_samples)
    doppler_phase = np.exp(
        1j * 2.0 * np.pi * fd_hz
        * absolute_n / FS
    ).reshape(n_symbols, n_samples)

    y *= doppler_phase

    return y


def estimate_range_from_burst(
    tx_burst,
    rx_burst,
    max_delay
):
    """
    Estimate target delay by coherent averaging of the
    normalized correlation metric across the OFDM burst.
    """
    metrics = np.zeros(max_delay + 1, dtype=float)

    for d in range(max_delay + 1):
        total = 0.0

        for m in range(tx_burst.shape[0]):
            ref = tx_burst[m, :tx_burst.shape[1] - d]
            obs = rx_burst[m, d:]

            if len(ref) == 0:
                continue

            corr = np.vdot(ref, obs)
            total += np.abs(corr) ** 2

        metrics[d] = total

    delay_hat = int(np.argmax(metrics))

    range_hat = (
        delay_hat / FS
    ) * C / 2.0

    return delay_hat, range_hat, metrics


def estimate_doppler_from_burst(
    tx_burst,
    rx_burst,
    delay_hat
):
    """
    Estimate Doppler from phase progression across coherent
    OFDM symbols.

    For each symbol, the complex matched-filter correlation
    provides one slow-time complex sample. The unwrapped phase
    of these slow-time samples is then fitted versus symbol
    index:

        phi[m] ~= phi0 + 2*pi*fd*m*T_SYM

    so that:

        fd_hat = slope / (2*pi*T_SYM)
    """
    slow_time = []

    for m in range(tx_burst.shape[0]):
        ref = tx_burst[m, :tx_burst.shape[1] - delay_hat]
        obs = rx_burst[m, delay_hat:]

        if len(ref) == 0:
            continue

        corr = np.vdot(ref, obs)

        # Normalize only for numerical conditioning; phase is
        # unchanged by positive real normalization.
        denom = (
            np.sqrt(
                np.vdot(ref, ref).real
                * np.vdot(obs, obs).real
            )
            + 1e-30
        )

        slow_time.append(corr / denom)

    slow_time = np.asarray(slow_time, dtype=complex)

    if len(slow_time) < 3:
        return 0.0

    phase = np.unwrap(
        np.angle(slow_time)
    )

    m = np.arange(len(phase), dtype=float)

    slope = np.polyfit(
        m,
        phase,
        1
    )[0]

    fd_hat = (
        slope
        / (2.0 * np.pi * T_SYM)
    )

    return float(fd_hat)


def calibrate_detection_threshold():
    """
    Noise-only matched-filter power threshold for the target PFA.
    The same coherent-burst statistic used by the target-present
    trials is used for threshold calibration.
    """
    powers = []

    for _ in range(1000):
        tx_burst = make_ofdm_burst()

        noise = complex_awgn(
            tx_burst.shape,
            1.0
        )

        # Average normalized matched-filter power across symbols.
        vals = []

        for m in range(N_ISAC_SYMBOLS):
            corr = np.vdot(
                tx_burst[m],
                noise[m]
            )

            ref_power = (
                np.vdot(
                    tx_burst[m],
                    tx_burst[m]
                ).real
                + 1e-30
            )

            vals.append(
                np.abs(corr) ** 2
                / ref_power
            )

        powers.append(
            np.mean(vals)
        )

    return float(
        np.quantile(
            powers,
            1.0 - PFA_TARGET
        )
    )


def isac_trial(threshold):
    """
    One target-present ISAC trial using a coherent 32-symbol
    OFDM burst.
    """
    tx_burst = make_ofdm_burst()

    delay_seconds = (
        2.0 * TARGET_RANGE_M / C
    )

    delay_samples = int(
        round(
            delay_seconds * FS
        )
    )

    rx_target = apply_delay_doppler_burst(
        tx_burst,
        delay_samples,
        TARGET_DOPPLER_HZ
    )

    signal_power = np.mean(
        np.abs(rx_target) ** 2
    )

    noise_power = (
        signal_power / SNR_LIN
    )

    noise = complex_awgn(
        rx_target.shape,
        noise_power
    )

    rx = rx_target + noise

    # Search a physically meaningful range window.
    max_delay = min(
        int(
            2.0 * 2500.0 / C * FS
        ),
        N_SUB + CP_LEN - 1
    )

    max_delay = max(
        1,
        max_delay
    )

    delay_hat, range_hat, metrics = (
        estimate_range_from_burst(
            tx_burst,
            rx,
            max_delay
        )
    )

    # Detection statistic at estimated delay.
    stat_values = []

    for m in range(N_ISAC_SYMBOLS):
        ref = tx_burst[
            m,
            :tx_burst.shape[1] - delay_hat
        ]

        obs = rx[
            m,
            delay_hat:
        ]

        corr = np.vdot(
            ref,
            obs
        )

        ref_power = (
            np.vdot(
                ref,
                ref
            ).real
            + 1e-30
        )

        stat_values.append(
            np.abs(corr) ** 2
            / ref_power
        )

    stat = float(
        np.mean(stat_values)
    )

    detected = (
        stat > threshold
    )

    fd_hat = estimate_doppler_from_burst(
        tx_burst,
        rx,
        delay_hat
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
            float(np.sqrt(
                np.mean(range_errors ** 2)
            )),
        "doppler_rmse_hz":
            float(np.sqrt(
                np.mean(doppler_errors ** 2)
            )),
        "sensing_sinr_db":
            float(
                db10(np.mean(sinrs))
            ),
        "threshold":
            float(threshold),
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 72)
    print("OFDM-X FULL REAL EVALUATION")
    print("=" * 72)

    print()
    print("Common Parameters")
    print("-" * 72)
    print(f"Nsub                  : {N_SUB}")
    print(f"CP length             : {CP_LEN}")
    print(f"Subcarrier spacing    : {DELTA_F/1000:.1f} kHz")
    print(f"Sampling frequency    : {FS/1000:.1f} kHz")
    print(f"SNR                   : {SNR_DB:.1f} dB")
    print(f"ISAC Doppler          : {ISAC_FD:.1f} Hz")
    print(f"ISAC paths             : {ISAC_PATHS}")
    print(f"ISAC coherent symbols  : {N_ISAC_SYMBOLS}")
    print(f"RSMA Monte Carlo      : {MONTE_CARLO_RSMA}")
    print(f"ISAC Monte Carlo      : {MONTE_CARLO_ISAC}")

    results = run_ofdm_x_full_evaluation()

    print()
    print("=" * 72)
    print("OFDM-X ACTOR-CRITIC ADAPTIVE WAVEFORM SELECTION")
    print("=" * 72)

    for item in results["waveform_selection"]:
        print(
            f"{item['scenario']:10s} : "
            f"{item['waveform']:8s} | "
            f"Doppler={item['doppler_hz']:.1f} Hz | "
            f"Paths={item['paths']} | "
            f"P(OFDM)={item['p_ofdm']:.4f} | "
            f"P(AFDM)={item['p_afdm']:.4f} | "
            f"OFDM BER={item['ofdm_ber']:.6f} | "
            f"AFDM BER={item['afdm_ber']:.6f} | "
            f"OFDM-X BER={item['ofdm_x_ber']:.6f}"
        )

    print()
    print("=" * 72)
    print("RSMA RESULTS")
    print("=" * 72)

    rsma = results["rsma"]

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

    isac = results["isac"]

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
    print("ESTABLISHED FCAE V8 RESULTS")
    print("=" * 72)

    fcae = results["fcae"]

    print(
        f"Original PAPR           : "
        f"{fcae['original_papr_db']:.3f} dB"
    )

    print(
        f"FCAE PAPR               : "
        f"{fcae['fcae_papr_db']:.3f} dB"
    )

    print(
        f"PAPR improvement       : "
        f"{fcae['papr_improvement_db']:.3f} dB"
    )

    print(
        f"Test MSE                : "
        f"{fcae['test_mse']:.6e}"
    )

    print(
        f"Power error             : "
        f"{fcae['power_error_percent']:.4f} %"
    )

    print(
        f"EVM proxy               : "
        f"{fcae['evm_proxy_percent']:.4f} %"
    )

    print(
        f"BER proxy OFDM          : "
        f"{fcae['ber_proxy_ofdm']:.6f}"
    )

    print(
        f"BER proxy FCAE          : "
        f"{fcae['ber_proxy_fcae']:.6f}"
    )

    print()
    print("=" * 72)
    print("RESULTS SAVED")
    print("=" * 72)

    result_file = Path(
        "OFDM_X_FULL_REAL_EVALUATION_RESULTS.txt"
    )

    with result_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "OFDM-X FULL REAL EVALUATION\n"
        )
        f.write("=" * 72 + "\n\n")

        f.write("COMMON PARAMETERS\n")
        f.write(
            f"Nsub = {N_SUB}\n"
            f"CP = {CP_LEN}\n"
            f"Delta_f = {DELTA_F}\n"
            f"Fs = {FS}\n"
            f"SNR_dB = {SNR_DB}\n"
            f"ISAC_FD = {ISAC_FD}\n"
            f"ISAC_PATHS = {ISAC_PATHS}\n"
            f"N_ISAC_SYMBOLS = {N_ISAC_SYMBOLS}\n"
            f"RSMA Monte Carlo = {MONTE_CARLO_RSMA}\n"
            f"ISAC Monte Carlo = {MONTE_CARLO_ISAC}\n\n"
        )

        f.write("ACTOR-CRITIC ADAPTIVE WAVEFORM SELECTION\n")
        for item in results["waveform_selection"]:
            f.write(
                f"{item['scenario']}: "
                f"{item['waveform']}, "
                f"Doppler={item['doppler_hz']:.3f} Hz, "
                f"Paths={item['paths']}, "
                f"P(OFDM)={item['p_ofdm']:.9f}, "
                f"P(AFDM)={item['p_afdm']:.9f}, "
                f"OFDM BER={item['ofdm_ber']:.9f}, "
                f"AFDM BER={item['afdm_ber']:.9f}, "
                f"OFDM-X BER={item['ofdm_x_ber']:.9f}\n"
            )

        f.write("\nRSMA\n")
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
            f"{rsma['sum_rate_std']:.9f}\n"
        )

        f.write("\nISAC\n")
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

        f.write("\nFCAE V8 ESTABLISHED RESULTS\n")
        f.write(
            f"Original PAPR = "
            f"{fcae['original_papr_db']:.9f} dB\n"
        )
        f.write(
            f"FCAE PAPR = "
            f"{fcae['fcae_papr_db']:.9f} dB\n"
        )
        f.write(
            f"PAPR improvement = "
            f"{fcae['papr_improvement_db']:.9f} dB\n"
        )
        f.write(
            f"Test MSE = "
            f"{fcae['test_mse']:.9e}\n"
        )
        f.write(
            f"Power error = "
            f"{fcae['power_error_percent']:.6f} %\n"
        )
        f.write(
            f"EVM proxy = "
            f"{fcae['evm_proxy_percent']:.6f} %\n"
        )
        f.write(
            f"BER proxy OFDM = "
            f"{fcae['ber_proxy_ofdm']:.9f}\n"
        )
        f.write(
            f"BER proxy FCAE = "
            f"{fcae['ber_proxy_fcae']:.9f}\n"
        )

    print(
        f"Saved: {result_file}"
    )

    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
