"""Publication-ready plotting utilities.

Publication-ready figure defaults:
- Large canvas, large fonts, data-dominant after LaTeX column-width scaling
- PDF output, bbox_inches='tight', pad_inches=0.05
- No in-plot titles (LaTeX caption handles it)
- Distinct markers AND line styles per scheme (grayscale-readable)
- Confidence intervals as shaded fill_between bands
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Standard publication figure style
# ---------------------------------------------------------------------------

PHALANX_RCPARAMS = {
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.markersize": 7,
}

# Line styles and markers for distinguishable schemes
_STYLES = [
    {"marker": "o", "linestyle": "-"},
    {"marker": "s", "linestyle": "--"},
    {"marker": "^", "linestyle": "-."},
    {"marker": "D", "linestyle": ":"},
    {"marker": "v", "linestyle": "-"},
    {"marker": "P", "linestyle": "--"},
    {"marker": "X", "linestyle": "-."},
    {"marker": "*", "linestyle": ":"},
    {"marker": "h", "linestyle": "-"},
    {"marker": "p", "linestyle": "--"},
]


def apply_style() -> None:
    """Apply Phalanx publication style to matplotlib."""
    plt.rcParams.update(PHALANX_RCPARAMS)


def _get_style(idx: int) -> Dict[str, str]:
    """Return a marker/linestyle dict for scheme index *idx*."""
    return _STYLES[idx % len(_STYLES)]


# ---------------------------------------------------------------------------
# plot_comparison: bar chart comparing schedulers
# ---------------------------------------------------------------------------

def plot_comparison(
    names: Sequence[str],
    means: Sequence[float],
    ci_lows: Optional[Sequence[float]] = None,
    ci_highs: Optional[Sequence[float]] = None,
    ylabel: str = "Cost",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 4),
    colors: Optional[Sequence[str]] = None,
) -> matplotlib.figure.Figure:
    """Bar chart comparing schedulers on a single metric.

    Args:
        names: Scheduler display names.
        means: Mean metric value per scheduler.
        ci_lows: Lower CI bounds (optional).
        ci_highs: Upper CI bounds (optional).
        ylabel: Y-axis label.
        save_path: If provided, saves figure as PDF.
        figsize: Figure size in inches.
        colors: Optional per-bar colors.

    Returns:
        The matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(names))
    bar_colors = colors if colors is not None else [f"C{i}" for i in range(len(names))]

    if ci_lows is not None and ci_highs is not None:
        yerr_low = np.array(means) - np.array(ci_lows)
        yerr_high = np.array(ci_highs) - np.array(means)
        yerr = np.array([yerr_low, yerr_high])
        ax.bar(x, means, yerr=yerr, capsize=5, color=bar_colors,
               edgecolor="black", linewidth=0.5)
    else:
        ax.bar(x, means, color=bar_colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", pad_inches=0.05)
    return fig


# ---------------------------------------------------------------------------
# plot_convergence: time-averaged metric over time
# ---------------------------------------------------------------------------

def plot_convergence(
    traces: Dict[str, np.ndarray],
    ci_bands: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ylabel: str = "Time-Averaged Cost",
    xlabel: str = "Time Slot",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 4),
    subsample: int = 1,
) -> matplotlib.figure.Figure:
    """Plot convergence of time-averaged metrics for multiple schemes.

    Args:
        traces: Dict mapping scheme name to 1-D array of time-averaged values.
        ci_bands: Optional dict mapping scheme name to (lower, upper) arrays.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        save_path: If provided, saves figure as PDF.
        figsize: Figure size in inches.
        subsample: Plot every *subsample*-th point (for readability).

    Returns:
        The matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, trace) in enumerate(traces.items()):
        style = _get_style(i)
        t = np.arange(len(trace))
        idx = t[::subsample]
        ax.plot(
            idx,
            trace[idx],
            label=name,
            marker=style["marker"],
            linestyle=style["linestyle"],
            markevery=max(1, len(idx) // 10),
        )
        if ci_bands is not None and name in ci_bands:
            lo, hi = ci_bands[name]
            ax.fill_between(idx, lo[idx], hi[idx], alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", pad_inches=0.05)
    return fig


# ---------------------------------------------------------------------------
# plot_sweep: metric vs parameter sweep
# ---------------------------------------------------------------------------

def plot_sweep(
    param_values: Sequence[float],
    results: Dict[str, List[float]],
    ci_results: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    xlabel: str = "Parameter",
    ylabel: str = "Cost",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (8, 4),
) -> matplotlib.figure.Figure:
    """Plot metric vs parameter sweep for multiple schemes.

    Args:
        param_values: Swept parameter values (x-axis).
        results: Dict mapping scheme name to list of metric values
            (one per parameter value).
        ci_results: Optional dict mapping scheme name to list of
            (ci_lower, ci_upper) tuples.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        save_path: If provided, saves figure as PDF.
        figsize: Figure size in inches.

    Returns:
        The matplotlib Figure.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    x = np.array(param_values)
    for i, (name, vals) in enumerate(results.items()):
        style = _get_style(i)
        y = np.array(vals)
        ax.plot(
            x,
            y,
            label=name,
            marker=style["marker"],
            linestyle=style["linestyle"],
        )
        if ci_results is not None and name in ci_results:
            ci_list = ci_results[name]
            lo = np.array([c[0] for c in ci_list])
            hi = np.array([c[1] for c in ci_list])
            ax.fill_between(x, lo, hi, alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()

    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", pad_inches=0.05)
    return fig
