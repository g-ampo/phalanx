"""Phalanx DPP convergence and adversarial robustness validation.

Validates two well-known properties of Lyapunov drift-plus-penalty
scheduling under adversarial perturbation:

1. Cost convergence: time-averaged cost approaches the optimum at rate
   O(1/V) as the tradeoff parameter V increases (Neely 2010).

2. Adversarial robustness: cost degradation under increasing adversary
   budget J, comparing Stackelberg-aware vs. unaware scheduling.

Compares four schedulers: LyapunovDPP (Stackelberg-aware), LyapunovOnly
(no adversary model), Oracle, and RoundRobin.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phalanx.channels import GaussianVARChannel
from phalanx.config import ChannelConfig
from phalanx.schedulers import LyapunovDPP, RoundRobin, Oracle
from phalanx.schedulers.lyapunov import LyapunovOnly
from phalanx.adversaries import StackelbergAdversary, NoAdversary
from phalanx.metrics import CostMetric, compute_ci
from phalanx.core import Simulation
from phalanx.plotting import PHALANX_RCPARAMS

plt.rcParams.update(PHALANX_RCPARAMS)

N_SEEDS = 10
T = 2000
M = 8


def run_sweep(scheduler_factory, adversary_factory, param_name, param_values,
              channel_cfg=None):
    """Run a parameter sweep across multiple seeds."""
    if channel_cfg is None:
        channel_cfg = ChannelConfig(M=M, d_obs=4, var_coefficient=0.9)

    sim = Simulation()
    results = {"param": param_values, "mean": [], "lo": [], "hi": []}

    for val in param_values:
        costs = []
        for s in range(N_SEEDS):
            seed = 42 + s * 1000
            rng = np.random.default_rng(seed)
            channel = GaussianVARChannel(channel_cfg, rng)
            sched = scheduler_factory(val)
            adv = adversary_factory(val)
            metric = CostMetric()
            result = sim.run(channel, sched, adv, metric, T=T, seed=seed)
            costs.append(result["final_avg"])
        m, lo, hi = compute_ci(costs)
        results["mean"].append(m)
        results["lo"].append(lo)
        results["hi"].append(hi)
        print(f"  {param_name}={val:8.3f}  cost={m:.4f} [{lo:.4f}, {hi:.4f}]")

    return results


# ====================================================================
# Experiment 1: DPP cost convergence vs V
# ====================================================================
print("=" * 60)
print("DPP Cost Convergence vs V")
print("=" * 60)

V_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
schedulers = {
    "LyapunovDPP": lambda V: LyapunovDPP(V=V, F=4, J_total=0.5),
    "LyapunovOnly": lambda V: LyapunovOnly(V=V, F=4),
    "Oracle": lambda V: Oracle(J_total=0.5),
    "RoundRobin": lambda V: RoundRobin(),
}

fig1, ax1 = plt.subplots(figsize=(8, 4))
for sched_name, sched_factory in schedulers.items():
    print(f"\n  Scheduler: {sched_name}")
    res = run_sweep(
        scheduler_factory=sched_factory,
        adversary_factory=lambda V: StackelbergAdversary(J_total=0.5),
        param_name="V", param_values=V_values,
    )
    ax1.plot(V_values, res["mean"], marker="o", label=sched_name, markersize=7)
    ax1.fill_between(V_values, res["lo"], res["hi"], alpha=0.15)

ax1.set_xlabel("V (DPP tradeoff parameter)")
ax1.set_ylabel("Time-averaged cost")
ax1.legend()
ax1.set_xscale("log")
fig1.savefig("dpp_cost_vs_V.pdf", bbox_inches="tight", pad_inches=0.05)
print("\nSaved: dpp_cost_vs_V.pdf")

# ====================================================================
# Experiment 2: Adversarial robustness vs J
# ====================================================================
print("\n" + "=" * 60)
print("Adversarial Robustness vs J")
print("=" * 60)

J_values = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
V_fixed = 10.0

fig2, ax2 = plt.subplots(figsize=(8, 4))
for sched_name in ["LyapunovDPP", "LyapunovOnly", "Oracle", "RoundRobin"]:
    print(f"\n  Scheduler: {sched_name}")

    if sched_name == "LyapunovDPP":
        sf = lambda J: LyapunovDPP(V=V_fixed, F=4, J_total=J)
    elif sched_name == "LyapunovOnly":
        sf = lambda J: LyapunovOnly(V=V_fixed, F=4)
    elif sched_name == "Oracle":
        sf = lambda J: Oracle(J_total=J)
    else:
        sf = lambda J: RoundRobin()

    res = run_sweep(
        scheduler_factory=sf,
        adversary_factory=lambda J: StackelbergAdversary(J_total=J) if J > 0 else NoAdversary(),
        param_name="J", param_values=J_values,
    )
    ax2.plot(J_values, res["mean"], marker="s", label=sched_name, markersize=7)
    ax2.fill_between(J_values, res["lo"], res["hi"], alpha=0.15)

ax2.set_xlabel("Adversary budget J")
ax2.set_ylabel("Time-averaged cost")
ax2.legend()
fig2.savefig("dpp_cost_vs_J.pdf", bbox_inches="tight", pad_inches=0.05)
print("\nSaved: dpp_cost_vs_J.pdf")
