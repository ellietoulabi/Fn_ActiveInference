"""
Single consolidated script: every plot and table for Stage 1 (SA Red-Blue-
Button), covering both the standard headline learning-curve/ECDF plots and
the relocation-aligned H1 adaptation analyses.

Style convention: no descriptive figure titles or side-panel explanations --
just the plots themselves. Structural labels that distinguish panels from
each other (e.g. "Cold Start" / "Pretrained", or an agent's name on its own
facet) are kept, since those are needed to read a multi-panel figure at all.

Export: every figure is saved twice -- a 600 dpi PNG (well above the common
300 dpi print minimum) and a vector PDF sidecar of the same name, so scaling
in LaTeX never pixelates. This only affects this script's own THESIS_DPI
constant, not the shared thesis_style.DPI used by other stages' scripts.

Data: compare_nine_agents.py / compare_nine_agents_pretrained.py's own CSV +
sibling `nine_agents_configs*.json` output. No re-simulation; every derived
quantity (shortest path length, failure reason, recovery episode) is computed
directly from logged actions/rewards/steps against the environment's own
documented rules (environments/RedBlueButton/SingleAgentRedBlueButton.py).

Output layout
-------------
<output-dir>/main/       -- direct H1 evidence, thesis-main-text-ready:
  r1_relocation_aligned_success.png
  r2_adaptation_summary_{cold_start,pretrained}.png   (the seed-level scalar
      summary: first-post-relocation success, post-relocation adaptation
      area, and paired AIF-vs-baseline differences, all with 95% CI across
      seed-level values -- also available as
      tables/main_results_table_{protocol}_display.csv)
  r4b_conditional_episode_length_relocation_aligned.png

<output-dir>/appendix/   -- diagnostic / supporting material:
  h1_success_rate_curve[.png / _{cold_start,pretrained}_{means_only,facet,
      sparse_ci,grouped}.png]
  unconditional_episode_length_curve[...same 5 layouts...]  (was
      "h3_episode_length_curve" -- renamed to avoid implying this is thesis
      hypothesis H3, an unrelated Overcooked-scale claim)
  h4_first_success_ecdf.png, h5_stable_success_ecdf.png
  r3_successful_episode_efficiency.png, r3b_appendix_raw_successful_length.png
  r4_conditional_episode_length_calendar.png
  r5_recovery_time_ecdf.png, r6_config_difficulty.png, r7_failure_reasons.png

  (the old "h2_mean_return_curve" is not generated at all: with a +1/-1
  per-episode reward, mean return is an exact linear rescaling of success
  rate and carries no separate information)

<output-dir>/tables/     -- all CSV output (no visual style needed):
  summary_table_{protocol}.csv
  main_results_table_{protocol}[_display].csv
  paired_differences_{protocol}.csv
  pretrain_convergence_table.csv
  completeness_table.csv

Usage
-----
python utils/plotting/plot_sa_stage1_all.py \\
    --cold-start-dir thesis_logs/01_sa_redbluebuttons/sa_redbluebutton_ep400_cfg50_step50_30seed \\
    --pretrained-dir thesis_logs/01_sa_redbluebuttons/sa_redbluebutton_pretrained_ep400_cfg50_step50_30seed \\
    --episodes-per-config 50 \\
    -o thesis_plots/01_sa_redbluebuttons/stage1_all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from thesis_style import AGENT_COLORS, ci95, bootstrap_ci, ensure_dir  # noqa: E402

# Larger-than-default text throughout: axis tick labels, axis labels, and
# legends are all hard to read at matplotlib's defaults once a figure is
# shrunk to fit a thesis page. Tick DENSITY is untouched (no change to how
# many ticks matplotlib places) -- only the font size of each tick's label.
plt.rcParams.update({
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
})

_SEED_RE = re.compile(r"_seed(\d+)_")

PREFERRED_ORDER = [
    "AIF", "QLearning", "Vanilla", "Recency0.99", "Recency0.95",
    "Recency0.9", "Recency0.85", "TrajSampling", "OPSRL",
]

# Display-only relabeling: the raw strings above ("QLearning", "Vanilla", ...)
# are the literal `agent` values in the CSVs and MUST stay unchanged for data
# loading/filtering/color lookup (AGENT_COLORS is keyed on them). This dict is
# used only where a name is rendered (legend, title, tick label) -- never for
# matching against the data.
DISPLAY_NAMES = {
    "QLearning": "Q-learning",
    "Vanilla": "Dyna-Q",
    "Recency0.99": "Dyna-Q, recency 0.99",
    "Recency0.95": "Dyna-Q, recency 0.95",
    "Recency0.9": "Dyna-Q, recency 0.90",
    "Recency0.85": "Dyna-Q, recency 0.85",
    "TrajSampling": "Trajectory-sampling Dyna-Q",
}


def disp(agent: str) -> str:
    return DISPLAY_NAMES.get(agent, agent)

# Fixed colors encode PROTOCOL (not agent) in per-agent-faceted plots, since
# each subplot there is already dedicated to a single agent.
_COLD_START_COLOR = "#4C72B0"
_PRETRAINED_COLOR = "#DD8452"


# =============================================================================
# Figure export -- thesis-grade resolution: a high-DPI PNG (safely above the
# common 300 dpi minimum print requirement) plus a vector PDF sidecar (so
# scaling in LaTeX never pixelates), independent of the shared thesis_style.DPI
# constant used by every OTHER stage's plotting script (not touched, so this
# doesn't change any already-approved Stage 2/3 figure).
# =============================================================================

THESIS_DPI = 600


def save(fig, out_path: Path, bottom_legend: bool = False, top_legend: bool = False):
    """`bottom_legend`/`top_legend` reserve extra margin so a fig.legend()
    placed outside the axes doesn't overlap the subplots' own x-axis labels
    or title-row subplots -- tight_layout() alone doesn't know about legends
    added outside the axes after layout."""
    if bottom_legend:
        fig.tight_layout(rect=[0, 0.09, 1, 1])
    elif top_legend:
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=THESIS_DPI, bbox_inches="tight", facecolor="white")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out_path} (+ {pdf_path.name})")


# =============================================================================
# Loading
# =============================================================================

def _seed_from_name(path: Path) -> Optional[int]:
    m = _SEED_RE.search(path.name)
    return int(m.group(1)) if m else None


def _manhattan(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _shortest_path_length(config: dict, start: Tuple[int, int] = (0, 0)) -> int:
    """dist(start, red) + press + dist(red, blue) + press."""
    red = tuple(config["red_pos"])
    blue = tuple(config["blue_pos"])
    return _manhattan(start, red) + 1 + _manhattan(red, blue) + 1


def load_protocol(log_dir: Path, episodes_per_config: int, max_steps: int = 50) -> pd.DataFrame:
    """
    Returns one row per (seed, agent, episode) with columns:
      seed, agent, episode, config_idx, k, L_observed, L_shortest, excess,
      success, has_press, failure_reason
    `failure_reason` (meaningful only where success is False): "blue_before_red"
    (terminated before max_steps -- the environment's only early-termination
    path for a loss), "timeout_no_press" (ran the full max_steps, never once
    pressed ANY button), or "timeout_partial_progress" (ran the full
    max_steps, pressed at least once, still didn't complete the win).
    """
    log_dir = Path(log_dir)
    csv_files = sorted(log_dir.glob("nine_agents_comparison*.csv"))
    config_files = {
        s: f for f in log_dir.glob("nine_agents_configs*.json")
        if (s := _seed_from_name(f)) is not None
    }
    if not csv_files:
        raise FileNotFoundError(f"No nine_agents_comparison*.csv found in {log_dir}")

    frames: List[pd.DataFrame] = []
    for f in csv_files:
        seed = _seed_from_name(f)
        if seed is None or seed not in config_files:
            print(f"  Skipping {f.name}: no matching configs JSON found", file=sys.stderr)
            continue
        configs = json.loads(config_files[seed].read_text())

        df = pd.read_csv(f, usecols=["seed", "agent", "episode", "step", "reward", "action_name"])
        df = df.sort_values(["agent", "episode", "step"])
        ep = df.groupby(["agent", "episode"], as_index=False).agg(
            L_observed=("step", "max"),
            final_reward=("reward", "last"),
            has_press=("action_name", lambda s: bool((s == "PRESS").any())),
        )
        ep["seed"] = seed
        ep["config_idx"] = (ep["episode"] - 1) // episodes_per_config
        ep["k"] = (ep["episode"] - 1) % episodes_per_config + 1
        ep["success"] = ep["final_reward"] > 0

        shortest_by_config = {i: _shortest_path_length(c) for i, c in enumerate(configs)}
        ep["L_shortest"] = ep["config_idx"].map(shortest_by_config)
        ep["excess"] = ep["L_observed"] - ep["L_shortest"]

        def _reason(row):
            if row["success"]:
                return "win"
            if row["L_observed"] < max_steps:
                return "blue_before_red"
            return "timeout_partial_progress" if row["has_press"] else "timeout_no_press"

        ep["failure_reason"] = ep.apply(_reason, axis=1)
        frames.append(ep)

    if not frames:
        raise RuntimeError(f"No usable (csv, configs) pairs found in {log_dir}")
    out = pd.concat(frames, ignore_index=True)
    n_seeds = out["seed"].nunique()
    n_configs = out["config_idx"].nunique()
    print(f"  Loaded {log_dir.name}: {n_seeds} seeds, {n_configs} config blocks, "
          f"{out['agent'].nunique()} agents, {len(out)} episodes")
    return out


def agent_order(present: List[str]) -> List[str]:
    ordered = [a for a in PREFERRED_ORDER if a in present]
    ordered += [a for a in present if a not in ordered]
    return ordered


# =============================================================================
# Aggregation helpers
# =============================================================================

def seed_by_k_matrix(
    ep: pd.DataFrame, agent: str, value_col: str, episodes_per_config: int,
    only_success: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Two-stage aggregation: average `value_col` across relocation blocks
    within each seed first (per k), giving a (n_seeds, episodes_per_config)
    matrix ready for bootstrap_ci."""
    sub = ep[ep["agent"] == agent]
    if only_success:
        sub = sub[sub["success"]]
    seeds = np.sort(sub["seed"].unique())
    mat = np.full((len(seeds), episodes_per_config), np.nan)
    grouped = sub.groupby(["seed", "k"])[value_col].mean()
    for si, seed in enumerate(seeds):
        for k in range(1, episodes_per_config + 1):
            if (seed, k) in grouped.index:
                mat[si, k - 1] = grouped.loc[(seed, k)]
    return seeds, mat


def per_seed_scalar(ep: pd.DataFrame, agent: str, mode: str, episodes_per_config: int) -> pd.Series:
    """mode="first_episode": success at k=1 (block-averaged per seed).
    mode="block_mean": success averaged over the whole block (per seed)."""
    seeds, mat = seed_by_k_matrix(ep, agent, "success", episodes_per_config)
    if mode == "first_episode":
        vals = mat[:, 0]
    elif mode == "block_mean":
        vals = np.nanmean(mat, axis=1)
    else:
        raise ValueError(mode)
    return pd.Series(vals, index=seeds)


def rolling_curve_per_seed(
    ep: pd.DataFrame, agent: str, episodes: np.ndarray, value_col: str, window: int,
    only_success: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-seed rolling mean of `value_col` over the raw/calendar episode
    axis. If only_success, non-successful episodes contribute NaN instead of
    their raw value (so a window's mean reflects only the successes inside
    it, e.g. for conditional episode length)."""
    sub = ep[ep["agent"] == agent].copy()
    if only_success:
        sub[value_col] = np.where(sub["success"], sub[value_col], np.nan)
    pivot = sub.pivot(index="seed", columns="episode", values=value_col).reindex(columns=episodes)
    smoothed = pivot.T.rolling(window=window, min_periods=1).mean().T
    return smoothed.index.values, smoothed.values


# =============================================================================
# HEADLINE PLOTS (whole-run, calendar time, all agents overlaid)
# =============================================================================

# Extra per-protocol layout variants for each headline curve (success rate,
# mean return, episode length), ported from plot_sa_redbluebuttons_nine.py's
# original 5-layout system (overlaid / means_only / facet / sparse_ci /
# grouped) -- restyled to match this script's current conventions (no
# titles beyond structural panel/facet labels, larger fonts, save()'s
# high-dpi PNG + PDF export). Each variant is generated once per protocol
# (cold_start, pretrained) rather than combined side-by-side the way the
# main h1/h2/h3 overlaid plots are, matching the original script's design.
AGENT_GROUPS: List[Tuple[str, List[str]]] = [
    ("Recency variants", ["Recency0.99", "Recency0.95", "Recency0.9", "Recency0.85"]),
    ("Baselines & other", ["AIF", "QLearning", "Vanilla", "TrajSampling", "OPSRL"]),
]


def prepare_agent_curves(
    ep: pd.DataFrame, agents: List[str], episodes: np.ndarray, value_col: str,
    window: int, only_success: bool = False,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per agent: (mean, ci_low, ci_high, valid-mask) over the raw episode axis."""
    prepared = {}
    for agent in agents:
        _, mat = rolling_curve_per_seed(ep, agent, episodes, value_col, window, only_success)
        mean, lo, hi = bootstrap_ci(mat)
        prepared[agent] = (mean, lo, hi, ~np.isnan(mean))
    return prepared


def layout_means_only(prepared, agents, episodes, ylabel, episodes_per_config, out_path):
    fig, ax = plt.subplots(figsize=(10, 6.2))
    max_ep = float(episodes.max())
    for agent in agents:
        mean, _, _, valid = prepared[agent]
        if not np.any(valid):
            continue
        is_aif = agent == "AIF"
        color = AGENT_COLORS.get(agent, "#888888")
        ax.plot(episodes[valid], mean[valid], color=color, label=disp(agent),
                linewidth=3.0 if is_aif else 1.6, zorder=10 if is_aif else 3)
    for boundary in range(episodes_per_config, int(max_ep), episodes_per_config):
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(0, max_ep + 1)
    save(fig, out_path)


def layout_facet(prepared, agents, episodes, ylabel, episodes_per_config, out_path):
    n = len(agents)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.5 * nrows), sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()
    max_ep = float(episodes.max())

    for i, focus in enumerate(agents):
        ax = axes_flat[i]
        for agent in agents:
            mean, lo, hi, valid = prepared[agent]
            if not np.any(valid):
                continue
            if agent == focus:
                color = AGENT_COLORS.get(agent, "#888888")
                ax.plot(episodes[valid], mean[valid], color=color, linewidth=2.4, zorder=3)
                if np.any((hi[valid] - lo[valid]) > 1e-9):
                    ax.fill_between(episodes[valid], lo[valid], hi[valid], color=color, alpha=0.28, zorder=2)
            else:
                ax.plot(episodes[valid], mean[valid], color="#bbbbbb", linewidth=0.9, alpha=0.85, zorder=1)
        for boundary in range(episodes_per_config, int(max_ep), episodes_per_config):
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
        ax.set_title(disp(focus), fontsize=14, fontweight="bold" if focus == "AIF" else "normal")
        ax.grid(True, axis="y", alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.supxlabel("Episode")
    fig.supylabel(ylabel)
    axes_flat[0].set_xlim(0, max_ep + 1)
    save(fig, out_path)


def layout_sparse_ci(prepared, agents, episodes, ylabel, episodes_per_config, out_path, every=25):
    fig, ax = plt.subplots(figsize=(10, 6.2))
    max_ep = float(episodes.max())
    for agent in agents:
        mean, lo, hi, valid = prepared[agent]
        if not np.any(valid):
            continue
        is_aif = agent == "AIF"
        color = AGENT_COLORS.get(agent, "#888888")
        ax.plot(episodes[valid], mean[valid], color=color, label=disp(agent),
                linewidth=3.0 if is_aif else 1.6, zorder=3)
        idx = np.where(valid)[0]
        if len(idx) == 0:
            continue
        ep_vals = episodes[idx]
        tick_eps = np.unique(np.arange(ep_vals.min(), ep_vals.max() + 1, every, dtype=int))
        sel = idx[np.isin(ep_vals.astype(int), tick_eps)]
        if len(sel) == 0:
            sel = idx[:: max(1, len(idx) // 8)]
        yerr = np.vstack([mean[sel] - lo[sel], hi[sel] - mean[sel]])
        if np.any((hi[sel] - lo[sel]) > 1e-9):
            ax.errorbar(episodes[sel], mean[sel], yerr=yerr, color=color, fmt="none",
                        capsize=3, elinewidth=1.3, alpha=0.85, zorder=2)
    for boundary in range(episodes_per_config, int(max_ep), episodes_per_config):
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", ncol=3)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xlim(0, max_ep + 1)
    save(fig, out_path)


def layout_grouped(prepared, agents, episodes, ylabel, episodes_per_config, out_path):
    max_ep = float(episodes.max())
    panels: List[Tuple[str, List[str]]] = []
    placed = set()
    for title, members in AGENT_GROUPS:
        algs = [a for a in members if a in agents]
        if algs:
            panels.append((title, algs))
            placed.update(algs)
    extras = [a for a in agents if a not in placed]
    if extras:
        panels.append(("Other", extras))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.4), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, algs) in zip(axes, panels):
        for agent in algs:
            mean, lo, hi, valid = prepared[agent]
            if not np.any(valid):
                continue
            is_aif = agent == "AIF"
            color = AGENT_COLORS.get(agent, "#888888")
            ax.plot(episodes[valid], mean[valid], color=color, label=disp(agent),
                    linewidth=3.0 if is_aif else 1.6, zorder=3)
            if np.any((hi[valid] - lo[valid]) > 1e-9):
                ax.fill_between(episodes[valid], lo[valid], hi[valid], color=color, alpha=0.14, zorder=2)
        for boundary in range(episodes_per_config, int(max_ep), episodes_per_config):
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
        ax.set_title(title, fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(ylabel)
    fig.supxlabel("Episode")
    axes[-1].set_xlim(0, max_ep + 1)
    save(fig, out_path)


def generate_all_layouts(
    protocol_data: Dict[str, pd.DataFrame], value_col: str, ylabel: str, out_dir: Path,
    base_name: str, episodes_per_config: int, window: int = 50, only_success: bool = False,
):
    for protocol, ep in protocol_data.items():
        episodes = np.sort(ep["episode"].unique())
        agents = agent_order(sorted(ep["agent"].unique()))
        prepared = prepare_agent_curves(ep, agents, episodes, value_col, window, only_success)
        layout_means_only(prepared, agents, episodes, ylabel, episodes_per_config,
                           out_dir / f"{base_name}_{protocol}_means_only.png")
        layout_facet(prepared, agents, episodes, ylabel, episodes_per_config,
                     out_dir / f"{base_name}_{protocol}_facet.png")
        layout_sparse_ci(prepared, agents, episodes, ylabel, episodes_per_config,
                         out_dir / f"{base_name}_{protocol}_sparse_ci.png")
        layout_grouped(prepared, agents, episodes, ylabel, episodes_per_config,
                       out_dir / f"{base_name}_{protocol}_grouped.png")


def _overlaid_curve_plot(
    protocol_data: Dict[str, pd.DataFrame], value_col: str, ylabel: str,
    out_path: Path, window: int = 50, only_success: bool = False,
    episodes_per_config: Optional[int] = None,
):
    protocols = list(protocol_data.keys())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.6 * len(protocols), 5.0), squeeze=False)
    agents = agent_order(sorted(set().union(*[set(d["agent"].unique()) for d in protocol_data.values()])))

    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        ep = protocol_data[protocol]
        episodes = np.sort(ep["episode"].unique())
        for agent in agents:
            _, mat = rolling_curve_per_seed(ep, agent, episodes, value_col, window, only_success)
            mean, lo, hi = bootstrap_ci(mat)
            is_aif = agent == "AIF"
            color = AGENT_COLORS.get(agent, "#888888")
            ax.plot(episodes, mean, color=color, label=disp(agent),
                    linewidth=3.0 if is_aif else 1.4, alpha=1.0 if is_aif else 0.85,
                    zorder=10 if is_aif else 3)
            ax.fill_between(episodes, lo, hi, color=color, alpha=0.18 if is_aif else 0.08, zorder=2)
        if episodes_per_config:
            max_ep = int(episodes.max())
            for boundary in range(episodes_per_config, max_ep, episodes_per_config):
                ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.35, zorder=1)
        ax.set_xlabel("Episode")
        ax.set_title(protocol.replace("_", " ").title(), fontsize=14)
        ax.set_xlim(1, int(episodes.max()))
        ax.grid(alpha=0.2)

    axes[0, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(agents), 5), frameon=False, fontsize=12)
    save(fig, out_path, bottom_legend=True)


def plot_success_rate_curve(protocol_data, out_dir, episodes_per_config):
    _overlaid_curve_plot(
        protocol_data, "success", "Empirical success rate (rolling)",
        out_dir / "h1_success_rate_curve.png", episodes_per_config=episodes_per_config,
    )
    generate_all_layouts(protocol_data, "success", "Empirical success rate (rolling)", out_dir,
                         "h1_success_rate_curve", episodes_per_config)


# Mean-return curve (was h2_mean_return_curve) intentionally removed: with a
# +1/-1 per-episode reward, mean return is an exact linear rescaling of
# success rate (return = 2*p(success) - 1) and carries no information the
# success-rate curve doesn't already show.


def plot_episode_length_curve(protocol_data, out_dir, episodes_per_config):
    """Unconditional episode length (includes timeout-truncated failures).
    Renamed from the earlier "h3_..." basename -- this is diagnostic
    supporting material, not thesis hypothesis H3 (a different, unrelated
    claim about Overcooked), and the old name invited that confusion."""
    _overlaid_curve_plot(
        protocol_data, "L_observed", "Episode length, steps (rolling)",
        out_dir / "unconditional_episode_length_curve.png", episodes_per_config=episodes_per_config,
    )
    generate_all_layouts(protocol_data, "L_observed", "Episode length, steps (rolling)", out_dir,
                         "unconditional_episode_length_curve", episodes_per_config)


def compute_first_success_episode(ep: pd.DataFrame) -> pd.DataFrame:
    """One row per (seed, agent): raw episode number of the first success in
    the whole run, or NaN if the agent never won at all."""
    def first_ep(g):
        succ = g.loc[g["success"], "episode"]
        return succ.min() if len(succ) else np.nan
    return ep.groupby(["seed", "agent"]).apply(first_ep, include_groups=False).reset_index(name="first_success_episode")


def compute_stable_success_episode(ep: pd.DataFrame, window: int = 50, theta: float = 0.8) -> pd.DataFrame:
    """One row per (seed, agent): first raw episode at which a full
    (min_periods=window) rolling window of success rate reaches >= theta, or
    NaN if never reached."""
    rows = []
    for (seed, agent), g in ep.groupby(["seed", "agent"]):
        g = g.sort_values("episode")
        roll = g["success"].rolling(window=window, min_periods=window).mean()
        hit = g.loc[roll >= theta, "episode"]
        rows.append({"seed": seed, "agent": agent, "stable_success_episode": hit.min() if len(hit) else np.nan})
    return pd.DataFrame(rows)


def _ecdf_plot(
    protocol_data: Dict[str, pd.DataFrame], compute_fn, value_col: str, max_episode_lookup: Dict[str, int],
    xlabel: str, out_path: Path,
):
    protocols = list(protocol_data.keys())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.6 * len(protocols), 5.0), squeeze=False)
    agents = agent_order(sorted(set().union(*[set(d["agent"].unique()) for d in protocol_data.values()])))

    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        ep = protocol_data[protocol]
        events = compute_fn(ep)
        n_seeds_total = events.groupby("agent")["seed"].nunique()
        max_ep = max_episode_lookup[protocol]
        x_axis = np.arange(1, max_ep + 1)
        for agent in agents:
            sub = events[events["agent"] == agent]
            n_total = n_seeds_total.get(agent, len(sub))
            if n_total == 0:
                continue
            recovered_by = [(sub[value_col] <= x).sum() / n_total for x in x_axis]
            is_aif = agent == "AIF"
            color = AGENT_COLORS.get(agent, "#888888")
            ax.step(x_axis, recovered_by, where="post", color=color, label=disp(agent),
                    linewidth=3.0 if is_aif else 1.4, alpha=1.0 if is_aif else 0.85,
                    zorder=10 if is_aif else 3)
        ax.set_xlabel(xlabel)
        ax.set_title(protocol.replace("_", " ").title(), fontsize=14)
        ax.set_xlim(1, max_ep)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)

    axes[0, 0].set_ylabel("Fraction of seeds")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(agents), 5), frameon=False, fontsize=12)
    save(fig, out_path, bottom_legend=True)


def plot_first_success_ecdf(protocol_data, out_dir):
    max_ep = {p: int(d["episode"].max()) for p, d in protocol_data.items()}
    _ecdf_plot(protocol_data, compute_first_success_episode, "first_success_episode", max_ep,
               "Episode", out_dir / "h4_first_success_ecdf.png")


def plot_stable_success_ecdf(protocol_data, out_dir, window=50, theta=0.8):
    max_ep = {p: int(d["episode"].max()) for p, d in protocol_data.items()}
    fn = lambda ep: compute_stable_success_episode(ep, window, theta)  # noqa: E731
    _ecdf_plot(protocol_data, fn, "stable_success_episode", max_ep,
               "Episode", out_dir / "h5_stable_success_ecdf.png")


# =============================================================================
# RELOCATION-ALIGNED H1 ANALYSES
# =============================================================================

def plot_relocation_aligned_success(protocol_data, episodes_per_config, out_path):
    protocols = list(protocol_data.keys())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.6 * len(protocols), 5.0), squeeze=False)
    agents = agent_order(sorted(set().union(*[set(d["agent"].unique()) for d in protocol_data.values()])))
    k_axis = np.arange(1, episodes_per_config + 1)

    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        ep = protocol_data[protocol]
        for agent in agents:
            _, mat = seed_by_k_matrix(ep, agent, "success", episodes_per_config)
            mean, lo, hi = bootstrap_ci(mat)
            is_aif = agent == "AIF"
            color = AGENT_COLORS.get(agent, "#888888")
            ax.plot(k_axis, mean, color=color, label=disp(agent),
                    linewidth=3.0 if is_aif else 1.4, alpha=1.0 if is_aif else 0.85,
                    zorder=10 if is_aif else 3)
            ax.fill_between(k_axis, lo, hi, color=color, alpha=0.18 if is_aif else 0.08, zorder=2)
        ax.axvline(1, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, zorder=1)
        ax.set_xlabel("Episode since relocation")
        ax.set_title(protocol.replace("_", " ").title(), fontsize=14)
        ax.set_ylim(-0.03, 1.1)
        ax.set_xlim(1, episodes_per_config)
        ax.grid(alpha=0.2)

    axes[0, 0].set_ylabel("Empirical success rate")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(agents), 5), frameon=False, fontsize=12)
    save(fig, out_path, bottom_legend=True)


def plot_adaptation_summary(ep, protocol, episodes_per_config, out_path):
    agents = agent_order(sorted(ep["agent"].unique()))
    baselines = [a for a in agents if a != "AIF"]
    metrics = [("first_episode", "Success: episode 1 after relocation"),
               ("block_mean", "Success: mean over full post-relocation block")]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), squeeze=False)
    rng = np.random.default_rng(0)

    per_metric_scalars = {mode: {a: per_seed_scalar(ep, a, mode, episodes_per_config) for a in agents}
                          for mode, _ in metrics}

    for col, (mode, title) in enumerate(metrics):
        ax = axes[0, col]
        scalars = per_metric_scalars[mode]
        for i, agent in enumerate(agents):
            vals = scalars[agent].dropna().values
            color = AGENT_COLORS.get(agent, "#888888")
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=14, color=color, alpha=0.35, linewidths=0, zorder=2)
            m, ci = np.mean(vals), ci95(vals)
            ax.errorbar([i], [m], yerr=[ci] if not np.isnan(ci) else [0], fmt="D", color=color,
                        markersize=7, capsize=4, elinewidth=2, zorder=5, markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xticks(range(len(agents)))
        ax.set_xticklabels([disp(a) for a in agents], rotation=40, ha="right", fontsize=12)
        ax.set_ylabel("Empirical success rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=13)
        ax.grid(axis="y", alpha=0.2)

    for col, (mode, title) in enumerate(metrics):
        ax = axes[1, col]
        scalars = per_metric_scalars[mode]
        aif_vals = scalars["AIF"]
        ys = np.arange(len(baselines))
        for y, baseline in zip(ys, baselines):
            common = aif_vals.index.intersection(scalars[baseline].index)
            diffs = (aif_vals.loc[common] - scalars[baseline].loc[common]).dropna().values
            if len(diffs) == 0:
                continue
            m, ci = np.mean(diffs), ci95(diffs)
            color = AGENT_COLORS.get(baseline, "#888888")
            ax.errorbar([m], [y], xerr=[ci] if not np.isnan(ci) else [0], fmt="o", color=color,
                        markersize=7, capsize=4, elinewidth=2, markeredgecolor="black", markeredgewidth=0.6)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([disp(a) for a in baselines], fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("Δ empirical success rate\n(AIF − baseline), paired by seed")
        ax.set_title(f"Paired difference", fontsize=13)
        ax.grid(axis="x", alpha=0.2)

    save(fig, out_path)


def plot_successful_episode_efficiency(
    protocol_data, episodes_per_config, out_path, value_col="excess",
    ylabel="Excess path length among successes\n(observed − shortest, steps)",
):
    protocols = list(protocol_data.keys())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.6 * len(protocols), 5.0), squeeze=False)
    agents = agent_order(sorted(set().union(*[set(d["agent"].unique()) for d in protocol_data.values()])))
    k_axis = np.arange(1, episodes_per_config + 1)

    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        ep = protocol_data[protocol]
        for agent in agents:
            _, mat = seed_by_k_matrix(ep, agent, value_col, episodes_per_config, only_success=True)
            mean, lo, hi = bootstrap_ci(mat)
            is_aif = agent == "AIF"
            color = AGENT_COLORS.get(agent, "#888888")
            ax.plot(k_axis, mean, color=color, label=disp(agent),
                    linewidth=3.0 if is_aif else 1.4, alpha=1.0 if is_aif else 0.85,
                    zorder=10 if is_aif else 3)
            ax.fill_between(k_axis, lo, hi, color=color, alpha=0.18 if is_aif else 0.08, zorder=2)
        ax.axvline(1, color="gray", linestyle=":", linewidth=1.2, alpha=0.7, zorder=1)
        ax.set_xlabel("Episode since relocation")
        ax.set_title(protocol.replace("_", " ").title(), fontsize=14)
        ax.set_xlim(1, episodes_per_config)
        ax.grid(alpha=0.2)

    axes[0, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(agents), 5), frameon=False, fontsize=12)
    save(fig, out_path, bottom_legend=True)


def plot_conditional_episode_length_relocation_aligned(protocol_data, out_path, episodes_per_config):
    agents = agent_order(sorted(set().union(*[set(d["agent"].unique()) for d in protocol_data.values()])))
    n = len(agents)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    k_axis = np.arange(1, episodes_per_config + 1)
    # linestyle differs (solid vs dashed), not just color -- when the two
    # curves coincide almost exactly (e.g. AIF, whose behavior doesn't
    # depend on whether the OTHER agents were pretrained), a color-only
    # encoding lets the later-drawn line fully hide the earlier one,
    # visually erasing a real, present spike in the "cold start" curve.
    protocol_style = [
        ("cold_start", "Cold start", _COLD_START_COLOR, "-"),
        ("pretrained", "Pretrained", _PRETRAINED_COLOR, "--"),
    ]

    for ax, agent in zip(axes_flat, agents):
        for protocol, label, color, ls in protocol_style:
            if protocol not in protocol_data:
                continue
            ep = protocol_data[protocol]
            _, mat = seed_by_k_matrix(ep, agent, "L_observed", episodes_per_config, only_success=True)
            mean, lo, hi = bootstrap_ci(mat)
            ax.plot(k_axis, mean, color=color, linewidth=1.8, linestyle=ls, label=label, zorder=3)
            ax.fill_between(k_axis, lo, hi, color=color, alpha=0.18, zorder=2)
        ax.axvline(1, color="gray", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
        ax.set_title(disp(agent), fontsize=14, fontweight="bold" if agent == "AIF" else "normal")
        ax.set_xlim(1, episodes_per_config)
        ax.grid(alpha=0.2)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.supxlabel("Episode since relocation")
    fig.supylabel("Episode length among\nsuccessful episodes (steps)")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.42, 1.06), ncol=2, frameon=False, fontsize=12)
    save(fig, out_path, top_legend=True)


def plot_conditional_episode_length_calendar(protocol_data, out_path, episodes_per_config, window=50):
    agents = agent_order(sorted(set().union(*[set(d["agent"].unique()) for d in protocol_data.values()])))
    n = len(agents)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    protocol_style = [
        ("cold_start", "Cold start", _COLD_START_COLOR, "-"),
        ("pretrained", "Pretrained", _PRETRAINED_COLOR, "--"),
    ]
    max_ep = max(int(d["episode"].max()) for d in protocol_data.values())

    for ax, agent in zip(axes_flat, agents):
        for protocol, label, color, ls in protocol_style:
            if protocol not in protocol_data:
                continue
            ep = protocol_data[protocol]
            episodes = np.sort(ep["episode"].unique())
            _, mat = rolling_curve_per_seed(ep, agent, episodes, "L_observed", window, only_success=True)
            mean, lo, hi = bootstrap_ci(mat)
            ax.plot(episodes, mean, color=color, linewidth=1.8, linestyle=ls, label=label, zorder=3)
            ax.fill_between(episodes, lo, hi, color=color, alpha=0.18, zorder=2)
        for boundary in range(episodes_per_config, max_ep, episodes_per_config):
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
        ax.set_title(disp(agent), fontsize=14, fontweight="bold" if agent == "AIF" else "normal")
        ax.set_xlim(1, max_ep)
        ax.grid(alpha=0.2)
    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.supxlabel("Episode")
    fig.supylabel(f"Episode length among successes\n(steps, rolling window={window})")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.42, 1.06), ncol=2, frameon=False, fontsize=12)
    save(fig, out_path, top_legend=True)


def compute_recovery_events(ep: pd.DataFrame) -> pd.DataFrame:
    def first_success_k(g):
        succ = g.loc[g["success"], "k"]
        return succ.min() if len(succ) else np.nan
    return ep.groupby(["seed", "agent", "config_idx"]).apply(first_success_k, include_groups=False).reset_index(name="recovery_k")


def plot_recovery_ecdf(protocol_data, episodes_per_config, out_path):
    protocols = list(protocol_data.keys())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.6 * len(protocols), 5.0), squeeze=False)
    agents = agent_order(sorted(set().union(*[set(d["agent"].unique()) for d in protocol_data.values()])))
    k_axis = np.arange(1, episodes_per_config + 1)

    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        ep = protocol_data[protocol]
        rec = compute_recovery_events(ep)
        n_events_total = rec.groupby("agent").size()
        for agent in agents:
            sub = rec[rec["agent"] == agent]
            n_total = n_events_total.get(agent, len(sub))
            recovered_by = [(sub["recovery_k"] <= k).sum() / n_total for k in k_axis]
            is_aif = agent == "AIF"
            color = AGENT_COLORS.get(agent, "#888888")
            ax.step(k_axis, recovered_by, where="post", color=color, label=disp(agent),
                    linewidth=3.0 if is_aif else 1.4, alpha=1.0 if is_aif else 0.85,
                    zorder=10 if is_aif else 3)
        ax.axvline(1, color="gray", linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
        ax.set_xlabel("Episode since relocation (k)")
        ax.set_title(protocol.replace("_", " ").title(), fontsize=14)
        ax.set_xlim(1, episodes_per_config)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)

    axes[0, 0].set_ylabel("Fraction of relocation events\nwith ≥1 success by episode k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(agents), 5), frameon=False, fontsize=12)
    save(fig, out_path, bottom_legend=True)


def plot_config_difficulty(protocol_data, out_path):
    protocols = list(protocol_data.keys())
    fig, axes = plt.subplots(1, len(protocols), figsize=(5.8 * len(protocols), 4.2), squeeze=False)
    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        ep = protocol_data[protocol]
        configs = ep.drop_duplicates(["seed", "config_idx"])[["seed", "config_idx", "L_shortest"]]
        vals = configs["L_shortest"].values
        bins = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1)
        ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="white")
        ax.set_title(f"{protocol.replace('_', ' ').title()} (n={len(vals)}, mean={vals.mean():.2f}, sd={vals.std():.2f})",
                     fontsize=13)
        ax.set_xlabel("Shortest possible path length (steps)")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Number of relocation configs")
    save(fig, out_path)


_FAILURE_COLORS = {
    "win": "#2E7D32", "blue_before_red": "#C0392B",
    "timeout_partial_progress": "#E67E22", "timeout_no_press": "#7F8C8D",
}
_FAILURE_ORDER = ["win", "blue_before_red", "timeout_partial_progress", "timeout_no_press"]
_FAILURE_LABELS = {
    "win": "Win", "blue_before_red": "Lose: blue before red",
    "timeout_partial_progress": "Timeout (attempted a press)", "timeout_no_press": "Timeout (never pressed)",
}


def compute_failure_reason_table(ep: pd.DataFrame) -> pd.DataFrame:
    counts = ep.groupby(["agent", "failure_reason"]).size().unstack(fill_value=0)
    for cat in _FAILURE_ORDER:
        if cat not in counts.columns:
            counts[cat] = 0
    counts = counts[_FAILURE_ORDER]
    frac = counts.div(counts.sum(axis=1), axis=0)
    return frac.loc[agent_order(list(frac.index))]


def plot_failure_reasons(protocol_data, out_path):
    protocols = list(protocol_data.keys())
    fig, axes = plt.subplots(1, len(protocols), figsize=(6.8 * len(protocols), 5.0), squeeze=False)
    for col, protocol in enumerate(protocols):
        ax = axes[0, col]
        frac = compute_failure_reason_table(protocol_data[protocol])
        bottom = np.zeros(len(frac))
        x = np.arange(len(frac))
        for cat in _FAILURE_ORDER:
            ax.bar(x, frac[cat].values, bottom=bottom, color=_FAILURE_COLORS[cat], label=_FAILURE_LABELS[cat], width=0.7)
            bottom += frac[cat].values
        ax.set_xticks(x)
        ax.set_xticklabels([disp(a) for a in frac.index], rotation=40, ha="right", fontsize=12)
        ax.set_title(protocol.replace("_", " ").title(), fontsize=14)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("Fraction of episodes")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=12)
    save(fig, out_path, bottom_legend=True)


# =============================================================================
# TABLES (no visual style needed)
# =============================================================================

def compute_summary_table(ep: pd.DataFrame, episodes_per_config: int) -> pd.DataFrame:
    overall = ep.groupby("agent").agg(n_seeds=("seed", "nunique"), success_rate=("success", "mean"),
                                       mean_steps=("L_observed", "mean"))
    succ = ep[ep["success"]]
    cond = succ.groupby("agent").agg(mean_steps_success=("L_observed", "mean"),
                                      mean_excess_success=("excess", "mean"),
                                      mean_shortest=("L_shortest", "mean"))
    cond["efficiency_pct"] = 100.0 * cond["mean_shortest"] / cond["mean_steps_success"]
    tail_k = range(max(1, episodes_per_config - 9), episodes_per_config + 1)
    shock = ep[ep["k"] == 1].groupby("agent")["success"].mean().rename("shock_success")
    steady = ep[ep["k"].isin(tail_k)].groupby("agent")["success"].mean().rename("steady_success")
    rec = compute_recovery_events(ep)
    rec_stats = rec.groupby("agent").agg(median_recovery_k=("recovery_k", "median"),
                                          mean_recovery_k=("recovery_k", "mean"),
                                          frac_never_recovered=("recovery_k", lambda x: x.isna().mean()))
    table = overall.join([cond[["mean_steps_success", "mean_excess_success", "efficiency_pct"]], shock, steady, rec_stats])
    return table.loc[agent_order(list(table.index))]


def print_and_save_summary_table(ep, episodes_per_config, protocol, out_dir):
    table = compute_summary_table(ep, episodes_per_config)
    csv_path = out_dir / f"summary_table_{protocol}.csv"
    table.round(4).to_csv(csv_path)
    print(f"\n=== Summary table: {protocol} ===")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(table.round(3))
    print(f"  Saved {csv_path}")


def per_seed_mean_excess_success(ep: pd.DataFrame, agent: str) -> pd.Series:
    sub = ep[(ep["agent"] == agent) & (ep["success"])]
    return sub.groupby("seed")["excess"].mean()


_MAIN_METRICS = [
    ("first_post_relocation_success", "First-post-relocation success"),
    ("post_relocation_adaptation_area", "Post-relocation adaptation area"),
    ("successful_excess_steps", "Successful excess steps"),
]


def _metric_per_seed(ep, agent, metric_key, episodes_per_config):
    if metric_key == "first_post_relocation_success":
        return per_seed_scalar(ep, agent, "first_episode", episodes_per_config)
    if metric_key == "post_relocation_adaptation_area":
        return per_seed_scalar(ep, agent, "block_mean", episodes_per_config)
    if metric_key == "successful_excess_steps":
        return per_seed_mean_excess_success(ep, agent)
    raise ValueError(metric_key)


def build_main_results_table(ep, episodes_per_config) -> pd.DataFrame:
    agents = agent_order(sorted(ep["agent"].unique()))
    rows = []
    for agent in agents:
        row = {"agent": agent}
        for key, _ in _MAIN_METRICS:
            vals = _metric_per_seed(ep, agent, key, episodes_per_config).dropna().values
            row[f"{key}_mean"] = np.mean(vals) if len(vals) else np.nan
            row[f"{key}_ci95"] = ci95(vals) if len(vals) else np.nan
            row[f"{key}_n"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows).set_index("agent")


def _fmt_est_ci(mean, ci, decimals=3):
    if np.isnan(mean):
        return "—"
    if np.isnan(ci):
        return f"{mean:.{decimals}f} [n=1]"
    return f"{mean:.{decimals}f} [{mean - ci:.{decimals}f}, {mean + ci:.{decimals}f}]"


def print_main_results_table(table, protocol, out_dir):
    display_rows = [{"Method": agent, **{label: _fmt_est_ci(row[f"{key}_mean"], row[f"{key}_ci95"])
                                          for key, label in _MAIN_METRICS}}
                     for agent, row in table.iterrows()]
    display_df = pd.DataFrame(display_rows).set_index("Method")
    print(f"\n=== Main results table (mean [95% CI]): {protocol} ===")
    with pd.option_context("display.width", 160, "display.max_colwidth", 40):
        print(display_df)
    table.round(4).to_csv(out_dir / f"main_results_table_{protocol}.csv")
    display_df.to_csv(out_dir / f"main_results_table_{protocol}_display.csv")
    print(f"  Saved main_results_table_{protocol}[_display].csv")


def build_paired_difference_table(ep, episodes_per_config) -> pd.DataFrame:
    agents = agent_order(sorted(ep["agent"].unique()))
    baselines = [a for a in agents if a != "AIF"]
    rows = []
    for key, label in _MAIN_METRICS:
        aif_vals = _metric_per_seed(ep, "AIF", key, episodes_per_config)
        for baseline in baselines:
            base_vals = _metric_per_seed(ep, baseline, key, episodes_per_config)
            common = aif_vals.index.intersection(base_vals.index)
            diffs = (aif_vals.loc[common] - base_vals.loc[common]).dropna().values
            rows.append({"metric": label, "baseline": baseline, "n_seeds": len(diffs),
                         "mean_diff_AIF_minus_baseline": np.mean(diffs) if len(diffs) else np.nan,
                         "ci95": ci95(diffs) if len(diffs) else np.nan})
    return pd.DataFrame(rows)


def print_paired_difference_table(ep, episodes_per_config, protocol, out_dir):
    table = build_paired_difference_table(ep, episodes_per_config)
    print(f"\n=== Paired differences (AIF - baseline, mean [95% CI]): {protocol} ===")
    display = table.copy()
    display["AIF - baseline [95% CI]"] = [_fmt_est_ci(m, c) for m, c in
                                           zip(table["mean_diff_AIF_minus_baseline"], table["ci95"])]
    with pd.option_context("display.width", 160):
        print(display[["metric", "baseline", "n_seeds", "AIF - baseline [95% CI]"]].to_string(index=False))
    table.round(4).to_csv(out_dir / f"paired_differences_{protocol}.csv", index=False)
    print(f"  Saved paired_differences_{protocol}.csv")


def build_pretrain_convergence_table(pretrained_dir: Path) -> Optional[pd.DataFrame]:
    pretrained_dir = Path(pretrained_dir)
    files = sorted(pretrained_dir.glob("pretrain_stats_seed*.json"))
    if not files:
        print(f"  No pretrain_stats_seed*.json found in {pretrained_dir}", file=sys.stderr)
        return None
    rows = []
    for f in files:
        seed = _seed_from_name(f)
        stats = json.loads(f.read_text())
        for agent, s in stats.items():
            rows.append({"seed": seed, "agent": agent, "n_episodes": s.get("n_episodes"),
                         "converged": s.get("converged"), "final_windowed_win_rate": s.get("final_windowed_win_rate")})
    df = pd.DataFrame(rows)
    summary = df.groupby("agent").agg(n_seeds=("seed", "nunique"), frac_converged=("converged", "mean"),
                                       mean_pretrain_episodes=("n_episodes", "mean"),
                                       median_pretrain_episodes=("n_episodes", "median"),
                                       max_pretrain_episodes=("n_episodes", "max"),
                                       mean_final_win_rate=("final_windowed_win_rate", "mean"))
    return summary.loc[agent_order(list(summary.index))]


def print_pretrain_convergence_table(pretrained_dir, out_dir):
    table = build_pretrain_convergence_table(pretrained_dir)
    if table is None:
        return
    print("\n=== Pretraining convergence / budget table ===")
    with pd.option_context("display.width", 160):
        print(table.round(3))
    table.round(4).to_csv(out_dir / "pretrain_convergence_table.csv")
    print("  Saved pretrain_convergence_table.csv")


def build_completeness_table(protocol_data, expected_seeds=30) -> pd.DataFrame:
    rows = []
    for protocol, ep in protocol_data.items():
        agents = agent_order(sorted(ep["agent"].unique()))
        all_seeds = set(range(expected_seeds))
        for agent in agents:
            sub = ep[ep["agent"] == agent]
            present_seeds = set(sub["seed"].unique())
            missing = sorted(all_seeds - present_seeds)
            ep_counts = sub.groupby("seed")["episode"].nunique()
            rows.append({"protocol": protocol, "agent": agent, "n_seeds_present": len(present_seeds),
                         "n_seeds_expected": expected_seeds,
                         "missing_seeds": ",".join(map(str, missing)) if missing else "",
                         "min_episodes_per_seed": ep_counts.min() if len(ep_counts) else 0,
                         "max_episodes_per_seed": ep_counts.max() if len(ep_counts) else 0,
                         "uniform_episode_count": bool(ep_counts.nunique() <= 1) if len(ep_counts) else True})
    return pd.DataFrame(rows)


def print_completeness_table(protocol_data, out_dir, expected_seeds=30):
    table = build_completeness_table(protocol_data, expected_seeds)
    print("\n=== Appendix: run completeness (seeds / episodes / missing runs) ===")
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(table.to_string(index=False))
    table.to_csv(out_dir / "completeness_table.csv", index=False)
    print("  Saved completeness_table.csv")
    flagged = table[(table["n_seeds_present"] < table["n_seeds_expected"]) | (~table["uniform_episode_count"])]
    if len(flagged):
        print("  ⚠ Incomplete runs detected:")
        print(flagged.to_string(index=False))
    else:
        print("  ✓ All agents/protocols have full seed coverage and uniform episode counts.")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cold-start-dir", type=Path,
                        default=Path("thesis_logs/01_sa_redbluebuttons/sa_redbluebutton_ep400_cfg50_step50_30seed"))
    parser.add_argument("--pretrained-dir", type=Path,
                        default=Path("thesis_logs/01_sa_redbluebuttons/sa_redbluebutton_pretrained_ep400_cfg50_step50_30seed"))
    parser.add_argument("--episodes-per-config", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("thesis_plots/01_sa_redbluebuttons/stage1_all"))
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    main_dir = ensure_dir(out_dir / "main")
    appendix_dir = ensure_dir(out_dir / "appendix")
    tables_dir = ensure_dir(out_dir / "tables")
    epc = args.episodes_per_config

    print("Loading cold-start protocol...")
    cold = load_protocol(args.cold_start_dir, epc, args.max_steps)
    print("Loading pretrained protocol...")
    warm = load_protocol(args.pretrained_dir, epc, args.max_steps)
    protocol_data = {"cold_start": cold, "pretrained": warm}

    # --- Main text: the direct, validated H1 evidence -----------------------
    print("\n--- Main-text plots ---")
    plot_relocation_aligned_success(protocol_data, epc, main_dir / "r1_relocation_aligned_success.png")
    plot_adaptation_summary(cold, "cold_start", epc, main_dir / "r2_adaptation_summary_cold_start.png")
    plot_adaptation_summary(warm, "pretrained", epc, main_dir / "r2_adaptation_summary_pretrained.png")
    plot_conditional_episode_length_relocation_aligned(
        protocol_data, main_dir / "r4b_conditional_episode_length_relocation_aligned.png", epc,
    )

    # --- Appendix: diagnostic / supporting material --------------------------
    print("\n--- Appendix plots ---")
    plot_success_rate_curve(protocol_data, appendix_dir, epc)  # h1 + its 4 layout variants
    plot_episode_length_curve(protocol_data, appendix_dir, epc)  # unconditional length + layouts
    plot_first_success_ecdf(protocol_data, appendix_dir)
    plot_stable_success_ecdf(protocol_data, appendix_dir)
    plot_successful_episode_efficiency(
        protocol_data, epc, appendix_dir / "r3_successful_episode_efficiency.png",
    )
    plot_successful_episode_efficiency(
        protocol_data, epc, appendix_dir / "r3b_appendix_raw_successful_length.png",
        value_col="L_observed", ylabel="Episode length among successes (steps)",
    )
    plot_conditional_episode_length_calendar(
        protocol_data, appendix_dir / "r4_conditional_episode_length_calendar.png", epc,
    )
    plot_recovery_ecdf(protocol_data, epc, appendix_dir / "r5_recovery_time_ecdf.png")
    plot_config_difficulty(protocol_data, appendix_dir / "r6_config_difficulty.png")
    plot_failure_reasons(protocol_data, appendix_dir / "r7_failure_reasons.png")

    print("\n--- Tables ---")
    print_and_save_summary_table(cold, epc, "cold_start", tables_dir)
    print_and_save_summary_table(warm, epc, "pretrained", tables_dir)
    print_main_results_table(build_main_results_table(cold, epc), "cold_start", tables_dir)
    print_main_results_table(build_main_results_table(warm, epc), "pretrained", tables_dir)
    print_paired_difference_table(cold, epc, "cold_start", tables_dir)
    print_paired_difference_table(warm, epc, "pretrained", tables_dir)
    print_pretrain_convergence_table(args.pretrained_dir, tables_dir)
    print_completeness_table(protocol_data, tables_dir)

    print(f"\nDone. Main-text figures: {main_dir}")
    print(f"      Appendix figures:  {appendix_dir}")
    print(f"      Tables:            {tables_dir}")


if __name__ == "__main__":
    main()
