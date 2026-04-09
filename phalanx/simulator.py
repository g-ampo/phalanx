"""Multi-seed parallel simulation runner with results aggregation.

Uses ``multiprocessing.get_context("spawn")`` for safe parallelism
across platforms (especially important with PyTorch / CUDA).
"""
from __future__ import annotations

import copy
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from phalanx.core import Adversary, Channel, Metric, Scheduler, Simulation
from phalanx.metrics.ci import compute_ci, compute_ci_array


def _run_single_seed(args: Tuple) -> Dict[str, Any]:
    """Worker function for a single-seed run (picklable top-level function).

    Args:
        args: Tuple of (channel, scheduler, adversary, metric, T, seed,
              F, lambda_mean, predictor, obs_window_size).

    Returns:
        Dict with the seed's results.
    """
    channel, scheduler, adversary, metric, T, seed, F, lambda_mean, \
        predictor, obs_window_size = args
    sim = Simulation()
    result = sim.run(
        channel=channel,
        scheduler=scheduler,
        adversary=adversary,
        metric=metric,
        T=T,
        seed=seed,
        F=F,
        lambda_mean=lambda_mean,
        predictor=predictor,
        obs_window_size=obs_window_size,
    )
    return {
        "seed": seed,
        "final_avg": result["final_avg"],
        "summary": result["summary"],
        "trajectory": result["trajectory"],
    }


class MultiSeedSimulator:
    """Run N independent seeds in parallel and aggregate results.

    Uses the ``spawn`` multiprocessing context to avoid fork-safety
    issues with CUDA and complex C extensions.

    Args:
        n_seeds: Number of independent random seeds.
        n_workers: Number of parallel worker processes.  If 0, runs
            sequentially in the main process (useful for debugging).
    """

    def __init__(self, n_seeds: int = 30, n_workers: int = 4):
        self.n_seeds = n_seeds
        self.n_workers = n_workers

    def run(
        self,
        channel: Channel,
        scheduler: Scheduler,
        adversary: Adversary,
        metric: Metric,
        T: int = 10_000,
        base_seed: int = 42,
        F: int = 4,
        lambda_mean: float = 0.35,
        burn_in: int = 1000,
        predictor: Optional[Any] = None,
        obs_window_size: int = 20,
    ) -> Dict[str, Any]:
        """Execute multi-seed simulation.

        Args:
            channel: Channel model (will be deep-copied per seed).
            scheduler: Scheduling policy (will be deep-copied per seed).
            adversary: Adversary model (will be deep-copied per seed).
            metric: Metric collector (will be deep-copied per seed).
            T: Number of time slots per seed.
            base_seed: Starting seed; seed_i = base_seed + i.
            F: Number of demand streams.
            lambda_mean: Mean arrival rate per stream.
            burn_in: Burn-in slots to discard before aggregation.
            predictor: Optional observation predictor (deep-copied per seed).
            obs_window_size: Number of past observations for the predictor.

        Returns:
            Dict with keys:
            - ``"ci"``: (mean, lower, upper) for the primary metric.
            - ``"mean"``, ``"ci_low"``, ``"ci_high"``: unpacked CI.
            - ``"per_seed"``: list of per-seed summary dicts.
            - ``"trajectory_ci"``: (mean, lower, upper) arrays for
              the primary trajectory if available.
            - ``"n_seeds"``: number of seeds completed.
        """
        # Build per-seed argument tuples
        seeds = [base_seed + i for i in range(self.n_seeds)]
        task_args = []
        for seed in seeds:
            rng_ch = np.random.default_rng(seed)
            ch = copy.deepcopy(channel)
            ch.reset(rng_ch)
            sch = copy.deepcopy(scheduler)
            adv = copy.deepcopy(adversary)
            met = copy.deepcopy(metric)
            pred = copy.deepcopy(predictor)
            task_args.append(
                (ch, sch, adv, met, T, seed, F, lambda_mean,
                 pred, obs_window_size)
            )

        # Execute
        if self.n_workers <= 0:
            # Sequential (debug mode)
            results = [_run_single_seed(a) for a in task_args]
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=min(self.n_workers, self.n_seeds)) as pool:
                results = pool.map(_run_single_seed, task_args)

        # Aggregate
        return self._aggregate(results, burn_in)

    def _aggregate(
        self, results: List[Dict[str, Any]], burn_in: int
    ) -> Dict[str, Any]:
        """Aggregate per-seed results into CIs and summaries."""
        per_seed_summaries = [r["summary"] for r in results]

        # Primary metric: "mean" key from summary
        values = [r.get("final_avg", s.get("mean", 0.0))
                  for r, s in zip(results, per_seed_summaries)]
        ci = compute_ci(values)
        mean, lo, hi = ci

        # Trajectory CI (if available)
        trajectory_ci = None
        trajectories = [r.get("trajectory", {}) for r in results]
        common_keys: set = set()
        for t in trajectories:
            common_keys.update(t.keys())
        for key in ["time_avg_cost", "mean_age", "total_backlog",
                     "queue_backlog", "per_slot_mean_delay",
                     "prediction_nmse"]:
            if key in common_keys:
                arrays = [
                    t[key] for t in trajectories
                    if key in t and len(t[key]) > 0
                ]
                if len(arrays) >= 2:
                    trajectory_ci = {
                        "key": key,
                        "mean_lower_upper": compute_ci_array(arrays),
                    }
                break

        return {
            "ci": ci,
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "per_seed": per_seed_summaries,
            "trajectory_ci": trajectory_ci,
            "n_seeds": len(results),
        }
