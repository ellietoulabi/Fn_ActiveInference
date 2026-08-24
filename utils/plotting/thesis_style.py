"""
Shared visual style for every thesis plotting script (Stage 1 SA Red-Blue-Button,
Stage 2 MA Red-Blue-Button, Stage 3 Overcooked SAL).

This module is the single source of truth for colors, figure export settings, and
confidence-interval math, so that a color, a DPI, or a CI formula means the same
thing in every chapter's figures rather than being redefined (and silently
drifting) per script. Extracted from what `overcooked_clean_plots.py` and
`plot_sal_semantic_action_level.py` already independently defined -- no color
values changed, so refactoring existing scripts to import from here is a pure
relocation, not a redesign. See ai/02-debug.md for the extraction record.

Two confidence-interval helpers are provided, deliberately kept distinct rather
than unified into one, because they serve genuinely different jobs:
  - `ci95`: parametric (t-distribution) CI for a small set of independent SEED
    summaries (one scalar per seed) -- used for per-seed dot plots / bar charts.
  - `bootstrap_ci`: percentile bootstrap CI for a per-STEP curve aggregated
    across seeds -- used for learning curves over a long trajectory, where a
    parametric assumption per-timestep is less appropriate and resampling
    across seeds is the natural approach.
Forcing these into a single function would be a statistical change dressed up
as a style refactor; they are kept separate on purpose.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Figure export
# =============================================================================

# Standardized on 300 (Stage 1's prior standard) rather than Stage 3's prior 150
# -- a pure resolution increase for Stage 3's figures, same content/layout, no
# visual regression, just crisper output for print/thesis inclusion.
DPI = 300


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path: Path | str, fig=None) -> None:
    """
    Save the current (or given) figure at the standard thesis DPI, tight bbox,
    white background, and close it. Matches the union of what
    overcooked_clean_plots.py and plot_sal_semantic_action_level.py already did
    (the latter additionally set facecolor="white" explicitly; folded in here
    since it's a strict improvement -- guards against a transparent background
    surviving into a printed/exported PDF).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        plt.tight_layout()
        plt.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close()
    else:
        fig.tight_layout()
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)


# =============================================================================
# Confidence intervals
# =============================================================================

_TCRIT_TABLE = {
    2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
    9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145,
    16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093,
}


def ci95(x) -> float:
    """
    95% CI half-width across independent seeds via Student's t (exact critical
    value for n<=20, falls back to the normal-approximation 1.96 beyond that).
    Use after reducing to one value per seed (e.g. per-seed mean success rate),
    not on raw per-step/per-episode observations.
    """
    x = pd.Series(x).dropna()
    n = len(x)
    if n <= 1:
        return np.nan
    tcrit = _TCRIT_TABLE.get(n, 1.96)
    return tcrit * x.std(ddof=1) / np.sqrt(n)


def bootstrap_ci(samples: np.ndarray, n_boot: int = 2000, ci: float = 95.0, seed: int = 0):
    """
    Percentile bootstrap mean + CI band for a (n_seeds, n_steps) array -- one
    row per seed, one column per step/timestep. Resamples seeds with
    replacement `n_boot` times and returns the empirical percentile band at
    every step, which is more appropriate than a per-timestep parametric CI
    when a full curve (not just one scalar) is being summarized across seeds.

    Returns
    -------
    mean : ndarray, shape (n_steps,)
    ci_low, ci_high : ndarray, shape (n_steps,)
    """
    samples = np.asarray(samples, dtype=float)
    n_seeds = samples.shape[0]
    rng = np.random.default_rng(seed)
    mean = np.nanmean(samples, axis=0)
    if n_seeds <= 1:
        return mean, mean.copy(), mean.copy()
    boot = np.empty((n_boot, samples.shape[1]), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot[b] = np.nanmean(samples[idx], axis=0)
    low = (100.0 - ci) / 2.0
    high = 100.0 - low
    ci_low = np.nanpercentile(boot, low, axis=0)
    ci_high = np.nanpercentile(boot, high, axis=0)
    return mean, ci_low, ci_high


# =============================================================================
# Paradigm / agent colors
# =============================================================================

# Short keys ("ind"/"ic"/"fc") match Stage 3's existing plot_sal_triple_comparison.py
# palette exactly (same hex values, not redesigned). Long keys are aliases onto
# the identical hex value, so Stage 2's MA Red-Blue-Button paradigms (same three
# names, same underlying concepts -- Independent/IndividuallyCollective/
# FullyCollective -- as Stage 3's Overcooked paradigms) render in the SAME color
# in both stages: a reader who learns "orange = Independent" in one chapter
# doesn't have to relearn it in the next.
_PARADIGM_BASE = {
    "ind": "#E67E22",
    "ic": "#2E86AB",
    "fc": "#27AE60",
}

# Stage 1's nine-agent palette (unchanged values, relocated here), extended with
# the long paradigm names so Stage 2/3 code that looks up colors via
# AGENT_COLORS (e.g. plot_sa_redbluebuttons_nine.py's plotting functions,
# reused as-is for Stage 2) resolves "Independent"/"IndividuallyCollective"/
# "FullyCollective" to the same colors PARADIGM_COLORS below uses. OPSRL's
# color is reused as-is in Stage 2's MA Red-Blue-Button plots, so "OPSRL" reads
# the same way in every stage it appears.
AGENT_COLORS = {
    "AIF": "#1A1A1A",
    "QLearning": "#000000",
    "Vanilla": "#2E86AB",
    "Recency0.99": "#06A77D",
    "Recency0.95": "#A23B72",
    "Recency0.9": "#F18F01",
    "Recency0.85": "#E63946",
    "TrajSampling": "#7209B7",
    "OPSRL": "#FF6B35",
    "Independent": _PARADIGM_BASE["ind"],
    "IndividuallyCollective": _PARADIGM_BASE["ic"],
    "FullyCollective": _PARADIGM_BASE["fc"],
    # Diagnostic-only variant (agents/ActiveInferenceSAExact +
    # generative_models/SA_ActiveInference/RedBlueButtonExact, ai/02-debug.md
    # 2026-08-22 "FOLLOW-UP" entry) -- exact info-gain + zero movement noise,
    # isolated from and never reported alongside "AIF" as an official result.
    "AIF_Exact": "#C0392B",
}

PARADIGM_COLORS = {
    **_PARADIGM_BASE,
    **AGENT_COLORS,
}

# Default single-series accent, used by plots that show exactly one paradigm's
# data at a time (e.g. Stage 3's per-paradigm learning-curve scripts) and have
# no fixed paradigm identity of their own to color by.
AGG_COLOR = _PARADIGM_BASE["ic"]
