# ============================================================
# FIGURE 13 - REAL ADAPTIVE OFDM/AFDM SWITCHING
# v7: fixed-path Doppler sweep + statistically stronger BER
# ============================================================
#
# This version keeps the paper parameters and separates:
#   A) paper-aligned scenarios:
#      Vehicular = 200 Hz / 1 path
#      LEO       = 1000 Hz / 3 paths
#      ISAC      = 450 Hz / 7 paths
#
#   B) adaptive Doppler-sweep experiment:
#      Doppler = 0...2000 Hz
#      PATHS   = 3 FIXED
#
# The adaptive sweep therefore isolates Doppler instead of
# simultaneously changing Doppler and the number of paths.
#
# BER evaluation uses many independent blocks to reduce the
# probability of reporting zero BER merely because too few
# bits were tested.
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 20260809

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)

rng = np.random.default_rng(SEED)

# ============================================================
# 2. PAPER PARAMETERS
# ============================================================

N = 64
CP = 16
NSYM = 4000

BPSK = True
SNR_DB = 14.0

DELTA_F = 15_000.0
FS = N * DELTA_F

SCENARIOS = {
    "Vehicular": {"fd": 200.0, "paths": 1},
    "LEO":       {"fd": 1000.0, "paths": 3},
    "ISAC":      {"fd": 450.0, "paths": 7},
}

# ============================================================
# 3. ADAPTIVE SWEEP
# ============================================================
# IMPORTANT:
# Number of paths is fixed at 3 for every Doppler point.
# This isolates the effect of Doppler.

DOPPLER_SWEEP = np.arange(
    0.0,
    2001.0,
    100.0
)

ADAPTIVE_PATHS = 3

# More blocks for statistically meaningful BER.
# Each block carries N BPSK bits.
TRAIN_BLOCKS = 8
EVAL_BLOCKS = 200

TRAIN_EPISODES = 1000

# ============================================================
# 4. BPSK
# ============================================================

def generate_bpsk():
    bits = rng.integers(
        0,
        2,
        size=N,
        dtype=np.int8
    )

    x = (
        2 * bits - 1
    ).astype(np.complex128)

    return bits, x


# ============================================================
# 5. OFDM
# ============================================================

def ofdm_modulate(x):

    useful = (
        np.fft.ifft(x)
        * np.sqrt(N)
    )

    return np.concatenate(
        (useful[-CP:], useful)
    )


def ofdm_demodulate(y):

    return (
        np.fft.fft(y)
        / np.sqrt(N)
    )


# ============================================================
# 6. AFDM
# ============================================================

def afdm_parameters(fd_max):

    alpha_max = max(
        1,
        int(
            np.ceil(
                float(fd_max)
                / DELTA_F
            )
        )
    )

    c1 = (
        2.0 * alpha_max + 1.0
    ) / (2.0 * N)

    c2 = 1.0 / (2.0 * N)

    return c1, c2


def daft_matrix(fd_max):

    c1, c2 = afdm_parameters(
        fd_max
    )

    n = np.arange(N)
    m = np.arange(N)

    phase = -2j * np.pi * (
        n[:, None] * m[None, :] / N
        + c1 * n[:, None] ** 2
        + c2 * m[None, :] ** 2
    )

    return (
        np.exp(phase)
        / np.sqrt(N)
    )


def afdm_modulate(
    x,
    fd_max
):

    A = daft_matrix(fd_max)

    useful = A.conj().T @ x

    return np.concatenate(
        (useful[-CP:], useful)
    )


def afdm_demodulate(
    y,
    fd_max
):

    A = daft_matrix(fd_max)

    return A @ y


# ============================================================
# 7. CHANNEL
# ============================================================

def make_channel(
    fd_max,
    paths
):

    paths = int(paths)

    delays = rng.integers(
        0,
        CP,
        size=paths
    )

    gains = (
        rng.normal(size=paths)
        + 1j * rng.normal(size=paths)
    ) / np.sqrt(2.0)

    # Exponential power delay profile.
    pdp = np.exp(
        -0.7 * np.arange(paths)
    )

    gains *= np.sqrt(pdp)

    gains /= np.sqrt(
        np.sum(
            np.abs(gains) ** 2
        )
        + 1e-12
    )

    # For reproducibility and a meaningful Doppler sweep,
    # Doppler frequencies are uniformly distributed in
    # [-fd_max, fd_max].
    dopplers = rng.uniform(
        -float(fd_max),
        float(fd_max),
        size=paths
    )

    # Baseband channel matrix for the useful N samples.
    H = np.zeros(
        (N, N),
        dtype=np.complex128
    )

    n = np.arange(N)

    # Each path contributes a delay and a time-varying phase.
    for delay, gain, fd in zip(
        delays,
        gains,
        dopplers
    ):

        phase = np.exp(
            1j
            * 2.0
            * np.pi
            * fd
            * n
            / FS
        )

        # Diagonal Doppler operator + circular delay.
        for k in range(N):

            H[
                k,
                (k - int(delay)) % N
            ] += (
                gain
                * phase[k]
            )

    return H


# ============================================================
# 8. AWGN
# ============================================================

def add_awgn(
    y,
    local_rng
):

    power = np.mean(
        np.abs(y) ** 2
    )

    snr_linear = (
        10.0 ** (
            SNR_DB / 10.0
        )
    )

    noise_power = (
        power
        / snr_linear
    )

    noise = np.sqrt(
        noise_power / 2.0
    ) * (
        local_rng.normal(
            size=y.shape
        )
        + 1j
        * local_rng.normal(
            size=y.shape
        )
    )

    return y + noise


# ============================================================
# 9. FAIR RECEIVERS
# ============================================================

def ofdm_equalizer(
    y,
    H
):

    Y = ofdm_demodulate(y)

    F = (
        np.fft.fft(
            np.eye(N)
        )
        / np.sqrt(N)
    )

    Heff = (
        F
        @ H
        @ F.conj().T
    )

    # Linear MMSE equalizer.
    noise_var = 10.0 ** (
        -SNR_DB / 10.0
    )

    G = (
        Heff.conj().T
        @ np.linalg.inv(
            Heff
            @ Heff.conj().T
            + noise_var * np.eye(N)
        )
    )

    return G @ Y


def afdm_equalizer(
    y,
    H,
    fd_max
):

    Y = afdm_demodulate(
        y,
        fd_max
    )

    A = daft_matrix(
        fd_max
    )

    Heff = (
        A
        @ H
        @ A.conj().T
    )

    noise_var = 10.0 ** (
        -SNR_DB / 10.0
    )

    G = (
        Heff.conj().T
        @ np.linalg.inv(
            Heff
            @ Heff.conj().T
            + noise_var * np.eye(N)
        )
    )

    return G @ Y


# ============================================================
# 10. ONE PAIRED BER BLOCK
# ============================================================

def paired_block(
    fd_max,
    paths
):

    bits, x = generate_bpsk()

    H = make_channel(
        fd_max,
        paths
    )

    tx_ofdm = ofdm_modulate(x)
    tx_afdm = afdm_modulate(
        x,
        fd_max
    )

    # The channel operates on the useful N samples.
    r0 = H @ tx_ofdm[CP:]
    r1 = H @ tx_afdm[CP:]

    # Same normalized noise realization for both waveforms.
    noise_rng = np.random.default_rng(
        int(
            rng.integers(
                0,
                2**31 - 1
            )
        )
    )

    y0 = add_awgn(
        r0,
        noise_rng
    )

    # Re-create the identical complex Gaussian sample.
    # This makes the paired comparison deterministic and fair.
    noise_rng = np.random.default_rng(
        int(
            rng.integers(
                0,
                2**31 - 1
            )
        )
    )

    # Independent AWGN is physically valid. The channel/data
    # realization remains paired. We deliberately avoid forcing
    # identical noise samples because the two transmitted
    # waveforms have different received powers.
    y1 = add_awgn(
        r1,
        noise_rng
    )

    x0 = ofdm_equalizer(
        y0,
        H
    )

    x1 = afdm_equalizer(
        y1,
        H,
        fd_max
    )

    b0 = (
        np.real(x0) >= 0
    ).astype(np.int8)

    b1 = (
        np.real(x1) >= 0
    ).astype(np.int8)

    e0 = np.sum(
        b0 != bits
    )

    e1 = np.sum(
        b1 != bits
    )

    return int(e0), int(e1)


# ============================================================
# 11. BER WITH MANY BLOCKS
# ============================================================

def paired_ber(
    fd_max,
    paths,
    blocks
):

    errors0 = 0
    errors1 = 0

    total_bits = (
        int(blocks)
        * N
    )

    for _ in range(
        int(blocks)
    ):

        e0, e1 = paired_block(
            fd_max,
            paths
        )

        errors0 += e0
        errors1 += e1

    return (
        errors0 / total_bits,
        errors1 / total_bits
    )


# ============================================================
# 12. ACTOR-CRITIC
# ============================================================

class ActorCritic:

    def __init__(self):

        # State:
        #   normalized Doppler
        #   normalized path count
        #   bias
        self.W = rng.normal(
            0.0,
            0.03,
            size=(3, 2)
        )

        self.b = np.zeros(2)

        self.VW = rng.normal(
            0.0,
            0.03,
            size=3
        )

        self.Vb = 0.0

        self.actor_lr = 0.015
        self.critic_lr = 0.03

    def state(
        self,
        fd,
        paths
    ):

        return np.array(
            [
                fd / 2000.0,
                paths / 7.0,
                1.0
            ],
            dtype=float
        )

    def probs(
        self,
        s
    ):

        z = (
            s @ self.W
            + self.b
        )

        z -= np.max(z)

        e = np.exp(z)

        return e / np.sum(e)

    def value(
        self,
        s
    ):

        return float(
            s @ self.VW
            + self.Vb
        )

    def choose(
        self,
        s,
        epsilon
    ):

        p = self.probs(s)

        if rng.random() < epsilon:

            action = int(
                rng.integers(
                    0,
                    2
                )
            )

        else:

            action = int(
                np.argmax(p)
            )

        return action, p

    def update(
        self,
        s,
        action,
        reward
    ):

        p = self.probs(s)

        V = self.value(s)

        adv = float(
            np.clip(
                reward - V,
                -1.0,
                1.0
            )
        )

        self.VW += (
            self.critic_lr
            * adv
            * s
        )

        self.Vb += (
            self.critic_lr
            * adv
        )

        target = np.zeros(2)
        target[action] = 1.0

        grad = (
            target - p
        )

        self.W += (
            self.actor_lr
            * adv
            * np.outer(
                s,
                grad
            )
        )

        self.b += (
            self.actor_lr
            * adv
            * grad
        )


# ============================================================
# 13. REWARD
# ============================================================

def reward_from_ber(
    ber_ofdm,
    ber_afdm,
    action
):

    selected = (
        ber_ofdm
        if action == 0
        else ber_afdm
    )

    other = (
        ber_afdm
        if action == 0
        else ber_ofdm
    )

    # Small complexity preference for conventional OFDM when
    # communication performance is effectively equal.
    complexity_penalty = 0.002

    reward = (
        other - selected
    )

    if (
        abs(
            ber_ofdm
            - ber_afdm
        )
        < 1e-5
    ):

        if action == 1:
            reward -= complexity_penalty

    # Scale only the learning reward. BER itself remains unchanged.
    return float(
        np.clip(
            reward * 100.0,
            -1.0,
            1.0
        )
    )


# ============================================================
# 14. TRAINING
# ============================================================

def train_agent():

    agent = ActorCritic()

    print(
        "\nTraining Actor-Critic "
        "from statistically measured paired BER..."
    )

    for ep in range(
        1,
        TRAIN_EPISODES + 1
    ):

        fd = float(
            rng.uniform(
                0,
                2000
            )
        )

        # Training contains all three path classes,
        # but the evaluation sweep below uses fixed paths=3.
        paths = int(
            rng.choice(
                [1, 3, 7]
            )
        )

        s = agent.state(
            fd,
            paths
        )

        epsilon = (
            0.35
            * (
                1.0
                - ep / TRAIN_EPISODES
            )
            + 0.05
        )

        action, p = agent.choose(
            s,
            epsilon
        )

        b0, b1 = paired_ber(
            fd,
            paths,
            TRAIN_BLOCKS
        )

        reward = reward_from_ber(
            b0,
            b1,
            action
        )

        agent.update(
            s,
            action,
            reward
        )

        if ep % 100 == 0:

            name = (
                "OFDM"
                if action == 0
                else "AFDM"
            )

            print(
                f"Episode "
                f"{ep:4d}/"
                f"{TRAIN_EPISODES} | "
                f"fd={fd:7.1f} Hz | "
                f"paths={paths} | "
                f"OFDM BER={b0:.6f} | "
                f"AFDM BER={b1:.6f} | "
                f"action={name:4s} | "
                f"P(AFDM)={p[1]:.3f}"
            )

    return agent


# ============================================================
# 15. PAPER SCENARIOS
# ============================================================

def paper_scenarios(
    agent
):

    print("\n")
    print("=" * 100)

    print(
        "PAPER-ALIGNED SCENARIOS"
    )

    print("=" * 100)

    print(
        f"{'Channel':12s}"
        f"{'Doppler':>10s}"
        f"{'Paths':>8s}"
        f"{'OFDM BER':>14s}"
        f"{'AFDM BER':>14s}"
        f"{'OFDM-X BER':>14s}"
        f"{'Selected':>12s}"
    )

    print("-" * 100)

    for name, cfg in SCENARIOS.items():

        fd = cfg["fd"]
        paths = cfg["paths"]

        s = agent.state(
            fd,
            paths
        )

        p = agent.probs(s)

        action = int(
            np.argmax(p)
        )

        b0, b1 = paired_ber(
            fd,
            paths,
            EVAL_BLOCKS
        )

        bx = (
            b0
            if action == 0
            else b1
        )

        print(
            f"{name:12s}"
            f"{fd:10.0f}"
            f"{paths:8d}"
            f"{b0:14.6f}"
            f"{b1:14.6f}"
            f"{bx:14.6f}"
            f"{('AFDM' if action else 'OFDM'):>12s}"
        )

    print("=" * 100)


# ============================================================
# 16. FIXED-PATH DOPPLER SWEEP
# ============================================================

def fixed_path_sweep(
    agent
):

    ofdm = []
    afdm = []
    adaptive = []
    p_afdm = []
    selected = []

    print("\n")
    print(
        "DOPPLER SWEEP WITH FIXED PATHS=3"
    )

    print(
        f"{'Doppler':>10s}"
        f"{'Paths':>8s}"
        f"{'OFDM BER':>14s}"
        f"{'AFDM BER':>14s}"
        f"{'OFDM-X BER':>14s}"
        f"{'Selected':>12s}"
    )

    print("-" * 80)

    for fd in DOPPLER_SWEEP:

        paths = ADAPTIVE_PATHS

        s = agent.state(
            fd,
            paths
        )

        p = agent.probs(s)

        b0, b1 = paired_ber(
            fd,
            paths,
            EVAL_BLOCKS
        )

        action = int(
            np.argmax(p)
        )

        bx = (
            b0
            if action == 0
            else b1
        )

        name = (
            "AFDM"
            if action == 1
            else "OFDM"
        )

        ofdm.append(b0)
        afdm.append(b1)
        adaptive.append(bx)
        p_afdm.append(p[1])
        selected.append(name)

        print(
            f"{fd:10.0f}"
            f"{paths:8d}"
            f"{b0:14.6f}"
            f"{b1:14.6f}"
            f"{bx:14.6f}"
            f"{name:>12s}"
        )

    return (
        np.array(ofdm),
        np.array(afdm),
        np.array(adaptive),
        np.array(p_afdm),
        selected
    )


# ============================================================
# 17. FIGURE 13
# ============================================================

def make_figure(
    agent,
    ofdm,
    afdm,
    adaptive,
    p_afdm,
    selected
):

    fd_example = 1000.0
    paths_example = 3

    bits, x = generate_bpsk()

    ofdm_time = (
        np.fft.ifft(x)
        * np.sqrt(N)
    )

    afdm_time = (
        afdm_modulate(
            x,
            fd_example
        )[CP:]
    )

    s = agent.state(
        fd_example,
        paths_example
    )

    p0 = agent.probs(s)

    action = int(
        np.argmax(p0)
    )

    selected_time = (
        ofdm_time
        if action == 0
        else afdm_time
    )

    fig, ax = plt.subplots(
        2,
        2,
        figsize=(8.5, 6.2)
    )

    # --------------------------------------------------------
    # (a)
    # --------------------------------------------------------

    ax[0, 0].plot(
        np.abs(ofdm_time),
        label="Conventional OFDM"
    )

    ax[0, 0].plot(
        np.abs(afdm_time),
        label="AFDM"
    )

    ax[0, 0].plot(
        np.abs(selected_time),
        "--",
        linewidth=2,
        label="RL-selected OFDM-X"
    )

    ax[0, 0].set_title(
        "(a) Time-domain waveforms at "
        "1000 Hz Doppler"
    )

    ax[0, 0].set_xlabel(
        "Sample index"
    )

    ax[0, 0].set_ylabel(
        "Magnitude"
    )

    ax[0, 0].grid(
        True,
        alpha=0.25
    )

    ax[0, 0].legend(
        fontsize=7
    )

    # --------------------------------------------------------
    # (b)
    # --------------------------------------------------------

    ax[0, 1].plot(
        DOPPLER_SWEEP,
        p_afdm,
        marker="o",
        markersize=3
    )

    ax[0, 1].axhline(
        0.5,
        linestyle="--",
        linewidth=1
    )

    ax[0, 1].set_title(
        "(b) Learned AFDM selection probability"
    )

    ax[0, 1].set_xlabel(
        "Maximum Doppler (Hz)"
    )

    ax[0, 1].set_ylabel(
        "P(AFDM)"
    )

    ax[0, 1].set_ylim(
        0,
        1.05
    )

    ax[0, 1].grid(
        True,
        alpha=0.25
    )

    # --------------------------------------------------------
    # (c)
    # --------------------------------------------------------

    floor = 1.0 / (
        EVAL_BLOCKS * N
    )

    ax[1, 0].semilogy(
        DOPPLER_SWEEP,
        np.maximum(
            ofdm,
            floor
        ),
        label="Conventional OFDM"
    )

    ax[1, 0].semilogy(
        DOPPLER_SWEEP,
        np.maximum(
            afdm,
            floor
        ),
        label="AFDM"
    )

    ax[1, 0].semilogy(
        DOPPLER_SWEEP,
        np.maximum(
            adaptive,
            floor
        ),
        "--",
        linewidth=2,
        label="RL-based OFDM-X"
    )

    ax[1, 0].set_title(
        "(c) Measured BER under increasing "
        "Doppler (3 paths)"
    )

    ax[1, 0].set_xlabel(
        "Maximum Doppler (Hz)"
    )

    ax[1, 0].set_ylabel(
        "BER"
    )

    ax[1, 0].grid(
        True,
        alpha=0.25
    )

    ax[1, 0].legend(
        fontsize=7
    )

    # --------------------------------------------------------
    # (d)
    # --------------------------------------------------------

    selected_num = np.array(
        [
            1 if s == "AFDM"
            else 0
            for s in selected
        ]
    )

    ax[1, 1].step(
        DOPPLER_SWEEP,
        selected_num,
        where="mid"
    )

    ax[1, 1].set_title(
        "(d) RL-selected waveform "
        "(3 paths)"
    )

    ax[1, 1].set_xlabel(
        "Maximum Doppler (Hz)"
    )

    ax[1, 1].set_ylabel(
        "Selected waveform"
    )

    ax[1, 1].set_yticks(
        [0, 1],
        ["OFDM", "AFDM"]
    )

    ax[1, 1].grid(
        True,
        alpha=0.25
    )

    fig.suptitle(
        "Adaptive OFDM/AFDM Waveform "
        "Selection Using Actor-Critic RL",
        fontsize=11
    )

    plt.tight_layout()

    filename = (
        "Figure13_REAL_ADAPTIVE_"
        "OFDM_AFDM_FIXEDPATHS_v7.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\nFigure saved as: {filename}"
    )

    plt.show()


# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 100)

    print(
        "REAL OFDM-X ADAPTIVE "
        "WAVEFORM SIMULATION - v7"
    )

    print("=" * 100)

    print(
        f"Nsub={N}, "
        f"CP={CP}, "
        f"BPSK={BPSK}, "
        f"Nsym={NSYM}, "
        f"Eb/N0={SNR_DB:.1f} dB"
    )

    print(
        f"Delta-f={DELTA_F/1000:.1f} kHz, "
        f"Fs={FS/1000:.1f} kHz"
    )

    print(
        "Paper scenarios: "
        "Vehicular=200 Hz/1 path; "
        "LEO=1000 Hz/3 paths; "
        "ISAC=450 Hz/7 paths"
    )

    print(
        "Adaptive Doppler sweep: "
        "0-2000 Hz with FIXED 3 paths"
    )

    agent = train_agent()

    paper_scenarios(
        agent
    )

    (
        ofdm,
        afdm,
        adaptive,
        p_afdm,
        selected
    ) = fixed_path_sweep(
        agent
    )

    make_figure(
        agent,
        ofdm,
        afdm,
        adaptive,
        p_afdm,
        selected
    )
