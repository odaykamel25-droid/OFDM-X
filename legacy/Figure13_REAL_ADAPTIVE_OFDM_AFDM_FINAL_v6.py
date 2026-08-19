# ============================================================
# FIGURE 13 - REAL ADAPTIVE OFDM/AFDM SWITCHING (FINAL v2)
# ============================================================
# Reviewer #3 - Major Comment 3
#
# This version demonstrates adaptive waveform selection using
# measured BER plus a small, explicit computational-cost term.
# The cost term is included because OFDM-X is claimed to adapt
# the waveform according to channel conditions and implementation
# constraints; it prevents AFDM from being selected when both
# waveforms provide essentially identical BER.
#
# IMPORTANT:
#   No switching labels are hard-coded.
#   The Actor-Critic learns from measured paired BER/utility.
#
# Paper parameters retained:
#   Nsub = 64
#   CP = 16
#   Nsym = 4000 transmitted BPSK symbols (evaluation)
#   Eb/N0 = 14 dB
#   Delta-f = 15 kHz
#   Fs = 960 kHz
#   Vehicular = 200 Hz / 1 path
#   LEO       = 1000 Hz / 3 paths
#   ISAC      = 450 Hz / 7 paths
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Receiver chains:
#   OFDM: conventional one-tap frequency-domain MMSE equalizer
#         based on the reference (zero-Doppler) channel.
#   AFDM: DAFT-domain full linear-MMSE equalizer using the
#         measured time-varying channel.
# This is intentional: it models the practical receiver processing
# associated with each waveform rather than giving both waveforms
# the same idealized detector.

# ============================================================
# 1. Reproducibility
# ============================================================
SEED = 20260809
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
rng = np.random.default_rng(SEED)

# ============================================================
# 2. Paper parameters
# ============================================================
N = 64
CP = 16
NSYM = 4000
SNR_DB = 14.0
DELTA_F = 15_000.0
FS = N * DELTA_F

SCENARIOS = {
    "Vehicular": {"fd": 200.0, "paths": 1},
    "LEO":       {"fd": 1000.0, "paths": 3},
    "ISAC":      {"fd": 450.0, "paths": 7},
}

# Additional validation experiment for the adaptive decision.
DOPPLER_SWEEP = np.arange(0.0, 2001.0, 100.0)

# Training data volume.  The paper Nsym=4000 is used for the
# final evaluation, while shorter episodes are used for learning.
TRAIN_SYMBOLS_PER_EPISODE = 512
TRAIN_BLOCKS = int(np.ceil(TRAIN_SYMBOLS_PER_EPISODE / N))
EVAL_BLOCKS = int(np.ceil(NSYM / N))
TRAIN_EPISODES = 1200

# A small complexity penalty is part of the adaptive utility.
# OFDM is the lower-complexity reference. AFDM has a higher
# transform/equalization burden. This is NOT a BER manipulation.
AFDM_COMPLEXITY_PENALTY = 0.0050
UTILITY_LEARNING_SCALE = 100.0

# ============================================================
# 3. BPSK / transforms
# ============================================================
def generate_bpsk():
    bits = rng.integers(0, 2, size=N, dtype=np.int8)
    x = (2 * bits - 1).astype(np.complex128)
    return bits, x


def dft_matrix():
    n = np.arange(N)
    k = np.arange(N)
    return np.exp(-2j * np.pi * np.outer(k, n) / N) / np.sqrt(N)


F = dft_matrix()


def afdm_matrix(fd_ref):
    # DAFT parameterization based on the maximum Doppler used by
    # the corresponding channel condition.
    alpha = max(1, int(np.ceil(float(fd_ref) / DELTA_F)))
    c1 = (2.0 * alpha + 1.0) / (2.0 * N)
    c2 = 1.0 / (2.0 * N)

    n = np.arange(N)
    D1 = np.diag(np.exp(-1j * 2 * np.pi * c1 * n**2))
    D2 = np.diag(np.exp(-1j * 2 * np.pi * c2 * n**2))
    return D2 @ F @ D1


def ofdm_modulate(x):
    useful = F.conj().T @ x
    return np.concatenate([useful[-CP:], useful])


def afdm_modulate(x, fd_ref):
    A = afdm_matrix(fd_ref)
    useful = A.conj().T @ x
    return np.concatenate([useful[-CP:], useful])

# ============================================================
# 4. Time-varying multipath channel
# ============================================================
def make_channel(fd_max, paths):
    paths = int(paths)

    delays = rng.integers(0, CP, size=paths)

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
    H_time = np.zeros((N, N), dtype=np.complex128)
    H_static = np.zeros((N, N), dtype=np.complex128)
    n = np.arange(N)

    for delay, gain, fd in zip(delays, gains, dopplers):
        phase = np.exp(1j * 2 * np.pi * fd * n / FS)
        for nn in range(N):
            kk = (nn - int(delay)) % N
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
        local.normal(size=N)
        + 1j * local.normal(size=N)
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
    z = F @ y
    Hf = F @ H_static @ F.conj().T
    h = np.diag(Hf)

    snr = 10 ** (SNR_DB / 10.0)
    return (np.conj(h) / (np.abs(h) ** 2 + 1.0 / snr)) * z


def afdm_mmse_detect(y, H_time, A):
    z = A @ y
    Heff = A @ H_time @ A.conj().T

    snr = 10 ** (SNR_DB / 10.0)
    reg = 1.0 / snr

    M = Heff.conj().T @ Heff + reg * np.eye(N)
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
        (A.conj().T @ x)[-CP:],
        A.conj().T @ x
    ])

    # Same physical channel realization.
    y_o_clean = H_time @ tx_o[CP:]
    y_a_clean = H_time @ tx_a[CP:]

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
        N
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
            float(fd) / DOPPLER_SWEEP[-1],
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
        else AFDM_COMPLEXITY_PENALTY
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
        1.0 / (N * TRAIN_BLOCKS),
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
        f"{AFDM_COMPLEXITY_PENALTY:.4f}"
    )

    for ep in range(1, TRAIN_EPISODES + 1):
        fd = float(rng.uniform(0, DOPPLER_SWEEP[-1]))
        paths = int(rng.choice([1, 3, 7]))
        s = agent.state(fd, paths)

        epsilon = max(
            0.08,
            0.45 * (1.0 - ep / TRAIN_EPISODES)
        )

        # Both waveform actions are measured on the same paired
        # channel/data/noise realization. The actor therefore learns
        # from counterfactual evidence rather than from a single
        # sampled action.
        bo, ba = paired_ber(
            fd,
            paths,
            TRAIN_BLOCKS
        )

        uo = utility(bo, 0)
        ua = utility(ba, 1)
        p_before = agent.probs(s)

        agent.update_counterfactual(
            s,
            uo * UTILITY_LEARNING_SCALE,
            ua * UTILITY_LEARNING_SCALE
        )

        action = int(np.argmax(agent.probs(s)))
        label = "AFDM" if action else "OFDM"
        reward = (ua - uo)

        if ep % 100 == 0:
            print(
                f"Episode {ep:3d}/{TRAIN_EPISODES} | "
                f"fd={fd:7.1f} Hz | paths={paths} | "
                f"OFDM BER={bo:.6f} | AFDM BER={ba:.6f} | "
                f"action={label:4s} | "
                f"P(AFDM)={p_before[1]:.3f} -> {agent.probs(s)[1]:.3f} | "
                f"U_AFDM-U_OFDM={reward:+.5f}"
            )

    return agent

# ============================================================
# 10. Evaluation
# ============================================================
def evaluate_paper_scenarios(agent):
    print("\n" + "=" * 108)
    print("PAPER-ALIGNED SCENARIO RESULTS")
    print("=" * 108)
    print(
        f"{'Channel':12s}"
        f"{'Doppler':>10s}"
        f"{'Paths':>8s}"
        f"{'OFDM BER':>14s}"
        f"{'AFDM BER':>14s}"
        f"{'OFDM-X BER':>14s}"
        f"{'Selected':>12s}"
    )
    print("-" * 108)

    for name, cfg in SCENARIOS.items():
        fd = cfg["fd"]
        paths = cfg["paths"]

        s = agent.state(fd, paths)
        p = agent.probs(s)
        action = int(np.argmax(p))

        bo, ba = paired_ber(
            fd,
            paths,
            EVAL_BLOCKS
        )

        bx = bo if action == 0 else ba

        print(
            f"{name:12s}"
            f"{fd:10.0f}"
            f"{paths:8d}"
            f"{bo:14.6f}"
            f"{ba:14.6f}"
            f"{bx:14.6f}"
            f"{('AFDM' if action else 'OFDM'):>12s}"
        )

    print("=" * 108)


def evaluate_sweep(agent):
    bo_list = []
    ba_list = []
    bx_list = []
    p_list = []
    selected = []

    # Alternate between the paper-relevant channel path classes.
    # This is an additional state sweep, not a replacement for the
    # Vehicular/LEO/ISAC scenario results.
    path_schedule = [1, 1, 1, 1, 3, 3, 3, 3, 7, 7, 7, 7, 3, 3, 1, 1, 7, 7, 3, 3, 1]

    for idx, fd in enumerate(DOPPLER_SWEEP):
        paths = path_schedule[idx]
        s = agent.state(fd, paths)
        p = agent.probs(s)
        action = int(np.argmax(p))

        bo, ba = paired_ber(
            fd,
            paths,
            EVAL_BLOCKS
        )

        bx = bo if action == 0 else ba

        bo_list.append(bo)
        ba_list.append(ba)
        bx_list.append(bx)
        p_list.append(p[1])
        selected.append(
            "AFDM" if action == 1 else "OFDM"
        )

    return (
        np.array(bo_list),
        np.array(ba_list),
        np.array(bx_list),
        np.array(p_list),
        selected
    )

# ============================================================
# 11. Figure 13
# ============================================================
def make_figure_13(agent, bo, ba, bx, p_afdm, selected):
    fd_example = 1000.0
    _, x = generate_bpsk()

    o_time = ofdm_modulate(x)[CP:]
    a_time = afdm_modulate(x, fd_example)[CP:]

    s = agent.state(fd_example, 3)
    p0 = agent.probs(s)
    action0 = int(np.argmax(p0))
    selected_time = a_time if action0 == 1 else o_time

    fig, ax = plt.subplots(2, 2, figsize=(8.2, 6.1))

    ax[0, 0].plot(np.abs(o_time), label="Conventional OFDM")
    ax[0, 0].plot(np.abs(a_time), label="AFDM")
    ax[0, 0].plot(
        np.abs(selected_time),
        "--",
        linewidth=1.8,
        label="RL-selected OFDM-X"
    )
    ax[0, 0].set_title("(a) Time-domain waveforms at 1000 Hz Doppler")
    ax[0, 0].set_xlabel("Sample index")
    ax[0, 0].set_ylabel("Magnitude")
    ax[0, 0].grid(alpha=0.25)
    ax[0, 0].legend(fontsize=7)

    ax[0, 1].plot(
        DOPPLER_SWEEP,
        p_afdm,
        marker="o",
        markersize=3
    )
    ax[0, 1].axhline(0.5, linestyle="--", linewidth=1)
    ax[0, 1].set_title("(b) Learned AFDM selection probability")
    ax[0, 1].set_xlabel("Maximum Doppler (Hz)")
    ax[0, 1].set_ylabel("P(AFDM)")
    ax[0, 1].set_ylim(0, 1.05)
    ax[0, 1].grid(alpha=0.25)

    ax[1, 0].semilogy(
        DOPPLER_SWEEP,
        np.maximum(bo, 1.0 / NSYM),
        label="Conventional OFDM"
    )
    ax[1, 0].semilogy(
        DOPPLER_SWEEP,
        np.maximum(ba, 1.0 / NSYM),
        label="AFDM"
    )
    ax[1, 0].semilogy(
        DOPPLER_SWEEP,
        np.maximum(bx, 1.0 / NSYM),
        "--",
        linewidth=1.8,
        label="RL-based OFDM-X"
    )
    ax[1, 0].set_title("(c) Measured BER under increasing Doppler")
    ax[1, 0].set_xlabel("Maximum Doppler (Hz)")
    ax[1, 0].set_ylabel("BER")
    ax[1, 0].grid(alpha=0.25)
    ax[1, 0].legend(fontsize=7)

    numeric = np.array([
        1 if s == "AFDM" else 0
        for s in selected
    ])

    ax[1, 1].step(
        DOPPLER_SWEEP,
        numeric,
        where="mid"
    )
    ax[1, 1].set_title("(d) RL-selected waveform across channel states")
    ax[1, 1].set_xlabel("Maximum Doppler (Hz)")
    ax[1, 1].set_ylabel("Selected waveform")
    ax[1, 1].set_yticks([0, 1], ["OFDM", "AFDM"])
    ax[1, 1].grid(alpha=0.25)

    fig.suptitle(
        "Adaptive OFDM/AFDM Waveform Selection Using Actor-Critic RL",
        fontsize=11
    )
    plt.tight_layout()

    out = "Figure13_REAL_ADAPTIVE_OFDM_AFDM_FINAL_v6.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved as: {out}")
    plt.show()

# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":
    print("=" * 108)
    print("REAL OFDM-X ADAPTIVE WAVEFORM SIMULATION - FINAL v6")
    print("=" * 108)
    print(
        f"Nsub={N}, CP={CP}, BPSK=True, Nsym={NSYM}, "
        f"Eb/N0={SNR_DB:.1f} dB"
    )
    print(
        f"Delta-f={DELTA_F/1000:.1f} kHz, Fs={FS/1000:.1f} kHz"
    )
    print(
        "Paper scenarios: Vehicular=200 Hz/1 path; "
        "LEO=1000 Hz/3 paths; ISAC=450 Hz/7 paths"
    )
    print(
        f"Evaluation symbols={EVAL_BLOCKS * N} "
        f"(at least {NSYM} BPSK symbols)"
    )

    agent = train_agent()
    evaluate_paper_scenarios(agent)

    print("\nRunning additional Doppler sweep for Figure 13...")
    bo, ba, bx, p, selected = evaluate_sweep(agent)

    print("\n")
    print(
        f"{'Doppler':>10s}"
        f"{'Paths':>8s}"
        f"{'OFDM BER':>14s}"
        f"{'AFDM BER':>14s}"
        f"{'OFDM-X BER':>14s}"
        f"{'Selected':>12s}"
    )
    print("-" * 64)

    path_schedule = [1, 1, 1, 1, 3, 3, 3, 3, 7, 7, 7, 7, 3, 3, 1, 1, 7, 7, 3, 3, 1]
    for i, (fd, x0, x1, xx, sel) in enumerate(zip(
        DOPPLER_SWEEP, bo, ba, bx, selected
    )):
        print(
            f"{fd:10.0f}"
            f"{path_schedule[i]:8d}"
            f"{x0:14.6f}"
            f"{x1:14.6f}"
            f"{xx:14.6f}"
            f"{sel:>12s}"
        )

    make_figure_13(
        agent,
        bo,
        ba,
        bx,
        p,
        selected
    )
