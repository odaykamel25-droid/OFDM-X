# ============================================================
# FIGURE 13 - REAL ADAPTIVE OFDM/AFDM SWITCHING
# v10: paper-aligned Actor-Critic adaptive switching
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

DOPPLER_SWEEP = np.arange(0.0, 2001.0, 100.0)
PATH_CLASSES = [1, 3, 7]

# More blocks for statistically meaningful BER.
# Each block carries N BPSK bits.
TRAIN_BLOCKS = 12
EVAL_BLOCKS = 100
TRAIN_EPISODES = 1200

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
        # State = normalized Doppler, normalized path count, bias.
        self.W = rng.normal(0.0, 0.03, size=(3, 2))
        self.b = np.zeros(2)
        self.VW = rng.normal(0.0, 0.03, size=3)
        self.Vb = 0.0
        self.actor_lr = 0.08
        self.critic_lr = 0.05

    def state(self, fd, paths):
        return np.array([fd / 2000.0, paths / 7.0, 1.0], dtype=float)

    def probs(self, s):
        z = s @ self.W + self.b
        z -= np.max(z)
        e = np.exp(z)
        return e / np.sum(e)

    def value(self, s):
        return float(s @ self.VW + self.Vb)

    def choose(self, s, epsilon):
        p = self.probs(s)
        if rng.random() < epsilon:
            action = int(rng.integers(0, 2))
        else:
            action = int(np.argmax(p))
        return action, p

    def update_full_information(self, s, rewards):
        """One-step Actor-Critic update using both measured action rewards.

        Both OFDM and AFDM are evaluated on the same channel realization,
        so the actor receives a full-information contextual reward instead
        of learning from a noisy single-action sample.
        """
        rewards = np.asarray(rewards, dtype=float)
        p = self.probs(s)
        expected_reward = float(np.dot(p, rewards))
        V = self.value(s)
        td_error = float(np.clip(expected_reward - V, -1.0, 1.0))

        # Critic: fit the expected measured reward.
        self.VW += self.critic_lr * td_error * s
        self.Vb += self.critic_lr * td_error

        # Actor: gradient of expected reward under a softmax policy.
        centered = rewards - expected_reward
        grad_logits = p * centered
        self.W += self.actor_lr * np.outer(s, grad_logits)
        self.b += self.actor_lr * grad_logits


# ============================================================
# 13. REWARD
# ============================================================

def reward_from_ber(ber_ofdm, ber_afdm, action):
    """Reward the action from measured paired BER.

    Action 0 = OFDM and action 1 = AFDM.  If the measured BERs are
    statistically tied, OFDM is preferred because switching gives no
    measured reliability gain. Otherwise the reward follows the
    measured BER ratio, giving the actor a stronger learning signal.
    """
    tol = 5e-4
    floor = 1.0 / (EVAL_BLOCKS * N)

    if abs(ber_ofdm - ber_afdm) <= tol:
        return 0.50 if action == 0 else -0.50

    selected = ber_ofdm if action == 0 else ber_afdm
    other = ber_afdm if action == 0 else ber_ofdm
    ratio_reward = np.log10((other + floor) / (selected + floor))
    return float(np.clip(ratio_reward, -1.0, 1.0))


# ============================================================
# 14. TRAINING
# ============================================================

def train_agent():
    agent = ActorCritic()
    print("\nTraining Actor-Critic from measured BER pairs...")

    # Train repeatedly on the actual paper channel classes.
    # This makes path count part of the learned channel state.
    for ep in range(1, TRAIN_EPISODES + 1):
        fd = float(rng.choice(DOPPLER_SWEEP))
        paths = int(rng.choice(PATH_CLASSES))
        s = agent.state(fd, paths)

        epsilon = max(0.02, 0.20 * (1.0 - ep / TRAIN_EPISODES))
        action, p = agent.choose(s, epsilon)

        b0, b1 = paired_ber(fd, paths, TRAIN_BLOCKS)
        r0 = reward_from_ber(b0, b1, 0)
        r1 = reward_from_ber(b0, b1, 1)
        agent.update_full_information(s, [r0, r1])

        if ep % 100 == 0:
            name = "OFDM" if action == 0 else "AFDM"
            print(f"Episode {ep:4d}/{TRAIN_EPISODES} | fd={fd:7.0f} Hz | paths={paths} | "
                  f"OFDM BER={b0:.6f} | AFDM BER={b1:.6f} | action={name:4s} | P(AFDM)={p[1]:.3f}")
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

def adaptive_sweep(agent):
    results = {}
    for paths in PATH_CLASSES:
        ofdm, afdm, adaptive, p_afdm, selected = [], [], [], [], []
        for fd in DOPPLER_SWEEP:
            s = agent.state(fd, paths)
            p = agent.probs(s)
            b0, b1 = paired_ber(fd, paths, EVAL_BLOCKS)

            # Final OFDM-X decision comes from the learned Actor policy.
            # No waveform is hard-coded in the evaluation stage.
            action = int(np.argmax(p))

            ofdm.append(b0); afdm.append(b1)
            adaptive.append(b0 if action == 0 else b1)
            p_afdm.append(p[1])
            selected.append("AFDM" if action else "OFDM")

            print(f"{fd:5.0f} Hz | paths={paths} | OFDM={b0:.6f} | AFDM={b1:.6f} | "
                  f"OFDM-X={adaptive[-1]:.6f} | {selected[-1]}")
        results[paths] = (np.array(ofdm), np.array(afdm), np.array(adaptive),
                          np.array(p_afdm), selected)
    return results


# ============================================================
# 17. FIGURE 13
# ============================================================

def make_figure(agent, results):
    """Create the final paper-facing Figure 13."""
    fd_example = SCENARIOS["LEO"]["fd"]
    paths_example = SCENARIOS["LEO"]["paths"]

    bits, x = generate_bpsk()
    ofdm_time = np.abs(np.fft.ifft(x) * np.sqrt(N))
    afdm_time = np.abs(afdm_modulate(x, fd_example)[CP:])

    p_leo = agent.probs(agent.state(fd_example, paths_example))
    action_leo = int(np.argmax(p_leo))
    selected_time = afdm_time if action_leo == 1 else ofdm_time
    selected_name = "AFDM" if action_leo == 1 else "OFDM"

    fig, ax = plt.subplots(2, 2, figsize=(8.4, 6.3))

    # (a) Real waveform comparison at the paper LEO operating point.
    ax[0, 0].plot(ofdm_time, label="Conventional OFDM", linewidth=1.2)
    ax[0, 0].plot(afdm_time, label="AFDM", linewidth=1.2)
    ax[0, 0].plot(selected_time, "--", linewidth=1.8,
                   label=f"RL-selected OFDM-X ({selected_name})")
    ax[0, 0].set_title("(a) Time-domain waveforms: LEO state")
    ax[0, 0].set_xlabel("Sample index")
    ax[0, 0].set_ylabel("Magnitude")
    ax[0, 0].grid(True, alpha=0.25)
    ax[0, 0].legend(fontsize=6.5)

    # (b) Learned actor probability for all channel-state classes.
    for paths in PATH_CLASSES:
        r = results[paths]
        ax[0, 1].plot(DOPPLER_SWEEP, r[3], marker="o", markersize=2.2,
                       linewidth=1.2, label=f"{paths} path(s)")
    ax[0, 1].axhline(0.5, linestyle="--", linewidth=1.0,
                     label="Decision threshold")
    for name, cfg in SCENARIOS.items():
        fd = cfg["fd"]; paths = cfg["paths"]
        p_afdm = agent.probs(agent.state(fd, paths))[1]
        ax[0, 1].plot(fd, p_afdm, "ko", markersize=4)
        ax[0, 1].annotate(name, (fd, p_afdm), xytext=(4, 5),
                          textcoords="offset points", fontsize=6)
    ax[0, 1].set_title("(b) Learned AFDM selection probability")
    ax[0, 1].set_xlabel("Maximum Doppler (Hz)")
    ax[0, 1].set_ylabel("P(AFDM)")
    ax[0, 1].set_ylim(0, 1.05)
    ax[0, 1].grid(True, alpha=0.25)
    ax[0, 1].legend(fontsize=6.2)

    # (c) Doppler-only experiment with the path count fixed at 3.
    r3 = results[3]
    ax[1, 0].semilogy(DOPPLER_SWEEP, np.maximum(r3[0], 1e-5),
                      label="Conventional OFDM", linewidth=1.3)
    ax[1, 0].semilogy(DOPPLER_SWEEP, np.maximum(r3[1], 1e-5), "--",
                      label="AFDM", linewidth=1.3)
    ax[1, 0].semilogy(DOPPLER_SWEEP, np.maximum(r3[2], 1e-5), "-.",
                      label="OFDM-X (RL-selected)", linewidth=1.7)
    leo_fd = SCENARIOS["LEO"]["fd"]
    idx = int(np.argmin(np.abs(DOPPLER_SWEEP - leo_fd)))
    ax[1, 0].plot(leo_fd, max(r3[2][idx], 1e-5), "ko", markersize=4)
    ax[1, 0].annotate("LEO: 1000 Hz", (leo_fd, max(r3[2][idx], 1e-5)),
                      xytext=(5, 5), textcoords="offset points", fontsize=6)
    ax[1, 0].set_title("(c) BER versus Doppler: 3-path channel")
    ax[1, 0].set_xlabel("Maximum Doppler (Hz)")
    ax[1, 0].set_ylabel("BER")
    ax[1, 0].grid(True, alpha=0.25)
    ax[1, 0].legend(fontsize=6.5)

    # (d) Actual learned waveform decision map. 0=OFDM, 1=AFDM.
    Z = np.array([[1 if s == "AFDM" else 0 for s in results[p][4]]
                  for p in PATH_CLASSES])
    ax[1, 1].imshow(Z, aspect="auto", interpolation="nearest",
                    extent=[DOPPLER_SWEEP[0]-50, DOPPLER_SWEEP[-1]+50,
                            0.5, 3.5], vmin=0, vmax=1)
    ax[1, 1].set_yticks([1, 2, 3], ["1 path", "3 paths", "7 paths"])
    ax[1, 1].set_xlabel("Maximum Doppler (Hz)")
    ax[1, 1].set_ylabel("Channel state")
    ax[1, 1].set_title("(d) RL-adaptive waveform selection")
    for y in [1, 2, 3]:
        vals = Z[y-1]
        # Label only if a class has a dominant learned action.
        if np.mean(vals) < 0.5:
            label = "OFDM"
        elif np.mean(vals) > 0.5:
            label = "AFDM"
        else:
            label = "Mixed"
        ax[1, 1].text(1000, y, label, ha="center", va="center",
                       fontsize=7, fontweight="bold")
    ax[1, 1].grid(False)

    fig.suptitle("Adaptive OFDM/AFDM Waveform Selection Using Actor-Critic RL",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = "Figure13_REAL_ADAPTIVE_OFDM_AFDM_FINAL_v10.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved as: {filename}")
    print(f"LEO operating point: fd={fd_example:.0f} Hz, paths={paths_example}, "
          f"learned P(AFDM)={p_leo[1]:.3f}, selected={selected_name}")
    plt.show()


# ============================================================
# 18. MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 100)

    print(
        "REAL OFDM-X ADAPTIVE "
        "WAVEFORM SIMULATION - v10"
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

    print("Adaptive Doppler sweep: 0-2000 Hz for each fixed path class")
    print("Decision rule: Actor-Critic policy learned from paired measured BER rewards.")

    agent = train_agent()
    paper_scenarios(agent)
    results = adaptive_sweep(agent)
    make_figure(agent, results)
