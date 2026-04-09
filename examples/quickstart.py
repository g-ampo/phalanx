"""Phalanx quickstart: Gaussian channel + Lyapunov scheduler + Stackelberg adversary.

Run 1000 slots, print cost with 95% CI, generate one figure.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phalanx.channels import GaussianVARChannel
from phalanx.config import ChannelConfig
from phalanx.schedulers import LyapunovDPP
from phalanx.adversaries import StackelbergAdversary
from phalanx.metrics import CostMetric, compute_ci
from phalanx.core import Simulation
from phalanx.plotting import PHALANX_RCPARAMS

plt.rcParams.update(PHALANX_RCPARAMS)

# --- Setup ---
cfg = ChannelConfig(M=8, d_obs=4, var_coefficient=0.9)
channel = GaussianVARChannel(cfg, rng=np.random.default_rng(42))
scheduler = LyapunovDPP(V=5.0, F=4, J_total=0.5)
adversary = StackelbergAdversary(J_total=0.5)
metric = CostMetric()
sim = Simulation()

# --- Multi-seed run ---
n_seeds, T = 10, 1000
costs = []
trajectories = []
for i in range(n_seeds):
    seed = 42 + i * 1000
    channel.reset(rng=np.random.default_rng(seed))
    result = sim.run(channel, scheduler, adversary, metric, T=T, seed=seed)
    costs.append(result["final_avg"])
    trajectories.append(result["trajectory"]["time_avg_cost"])

mean, lo, hi = compute_ci(costs)
print(f"Avg cost: {mean:.4f} +/- [{lo:.4f}, {hi:.4f}] (95% CI, {n_seeds} seeds)")

# --- Figure ---
fig, ax = plt.subplots(figsize=(8, 4))
for traj in trajectories:
    ax.plot(traj, color="steelblue", alpha=0.2, linewidth=0.5)
avg_traj = np.mean(trajectories, axis=0)
ax.plot(avg_traj, color="darkblue", linewidth=1.5, label="Mean")
ax.set_xlabel("Time slot")
ax.set_ylabel("Time-averaged cost")
ax.legend()
fig.savefig("quickstart.pdf", bbox_inches="tight", pad_inches=0.05)
print("Saved: quickstart.pdf")
