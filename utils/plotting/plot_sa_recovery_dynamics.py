"""
SA Red-Blue-Button: does performance recover after a button relocation, and
how fast?

For each agent: how does episode length / success rate behave in the
episodes immediately following a button relocation, versus once the agent
has resettled into that config? One figure (2 panels: episode length,
success rate, both vs. position-in-config-block) plus one summary table
(CSV + printed) per dataset -- no extra chart variants, no annotation
clutter.

Usage:
    python3 utils/plotting/plot_sa_recovery_dynamics.py \\
        thesis_logs/01_sa_redbluebuttons/<dataset> \\
        --episodes-per-config 25 \\
        -o thesis_plots/01_sa_redbluebuttons/<dataset>
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from thesis_style import AGENT_COLORS, ensure_dir, savefig  # noqa: E402

AGENT_ORDER = [
    "AIF", "QLearning", "Vanilla", "Recency0.99", "Recency0.95",
    "Recency0.9", "Recency0.85", "TrajSampling", "OPSRL",
]

STEADY_FRAC = 0.5    # steady-state = last 50% of a config block
RECOVER_TOL = 0.10   # "recovered" = within 10% of that block's own steady-state length


def latest_seed_files(folder: str, pattern: str) -> dict:
    files = glob.glob(os.path.join(folder, pattern))
    by_seed = {}
    for f in files:
        m = re.search(r"seed(\d+)_(\d{8}_\d{6})", os.path.basename(f))
        if not m:
            continue
        seed_idx, ts = m.group(1), m.group(2)
        if seed_idx not in by_seed or ts > by_seed[seed_idx][1]:
            by_seed[seed_idx] = (f, ts)
    return {k: v[0] for k, v in by_seed.items()}


def load_episodes(folder: str) -> pd.DataFrame:
    files = latest_seed_files(folder, "nine_agents_comparison_*.csv")
    if not files:
        raise FileNotFoundError(f"No nine_agents_comparison_*.csv under {folder}")
    frames = [pd.read_csv(f) for f in files.values()]
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(["seed", "agent", "episode"])
    ep = g.agg(length=("step", "max"), reward=("reward", "last")).reset_index()
    ep["success"] = (ep["reward"] == 1.0).astype(int)
    return ep


def add_position(ep: pd.DataFrame, episodes_per_config: int) -> pd.DataFrame:
    ep = ep.copy()
    ep["block"] = (ep["episode"] - 1) // episodes_per_config
    ep["position"] = (ep["episode"] - 1) % episodes_per_config
    return ep


def per_event_recovery(ep: pd.DataFrame, episodes_per_config: int) -> pd.DataFrame:
    """Per (agent, seed, block>0): episodes until episode length returns to
    within RECOVER_TOL of that block's own steady-state length. Mirrors the
    already-published Stage 1 methodology (ai/05-defense.md sec 7.1.1)."""
    steady_start = int(episodes_per_config * (1 - STEADY_FRAC))
    rows = []
    for (agent, seed, block), g in ep[ep["block"] > 0].groupby(["agent", "seed", "block"]):
        g = g.sort_values("position")
        steady_vals = g.loc[g["position"] >= steady_start, "length"]
        if len(steady_vals) == 0:
            continue
        steady_mean = steady_vals.mean()
        thresh = steady_mean * (1 + RECOVER_TOL)
        recovered = g.loc[g["length"] <= thresh, "position"]
        if len(recovered) == 0:
            continue
        rows.append({
            "agent": agent, "seed": seed, "block": block,
            "episodes_to_recover": int(recovered.iloc[0]),
            "steady_length": steady_mean,
            "shock_length": g.iloc[0]["length"],
        })
    return pd.DataFrame(rows)


def build_summary_table(ep: pd.DataFrame, episodes_per_config: int, recovery: pd.DataFrame) -> pd.DataFrame:
    steady_start = int(episodes_per_config * (1 - STEADY_FRAC))
    post_reloc = ep[ep["block"] > 0]
    rows = []
    for agent in AGENT_ORDER:
        a = post_reloc[post_reloc["agent"] == agent]
        if len(a) == 0:
            continue
        shock = a[a["position"] == 0]
        steady = a[a["position"] >= steady_start]
        rec = recovery[recovery["agent"] == agent]["episodes_to_recover"]
        rows.append({
            "agent": agent,
            "shock_success_rate": shock["success"].mean(),
            "steady_success_rate": steady["success"].mean(),
            "shock_mean_length": shock["length"].mean(),
            "steady_mean_length": steady["length"].mean(),
            "median_episodes_to_recover": rec.median() if len(rec) else np.nan,
            "mean_episodes_to_recover": rec.mean() if len(rec) else np.nan,
            "n_relocation_events": len(rec),
        })
    return pd.DataFrame(rows)


def plot_recovery(ep: pd.DataFrame, episodes_per_config: int, out_dir: Path, title_suffix: str) -> Path:
    ensure_dir(out_dir)
    post_reloc = ep[ep["block"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for agent in AGENT_ORDER:
        a = post_reloc[post_reloc["agent"] == agent]
        if len(a) == 0:
            continue
        color = AGENT_COLORS.get(agent, "#888")
        by_pos = a.groupby("position").agg(length=("length", "mean"), success=("success", "mean"))
        by_pos = by_pos.reindex(range(episodes_per_config))
        axes[0].plot(by_pos.index, by_pos["length"], color=color, lw=2, label=agent)
        axes[1].plot(by_pos.index, by_pos["success"], color=color, lw=2, label=agent)

    axes[0].set_xlabel("Episodes since relocation")
    axes[0].set_ylabel("Mean episode length")
    axes[0].set_title("Episode length recovery")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_xlabel("Episodes since relocation")
    axes[1].set_ylabel("Mean success rate")
    axes[1].set_title("Success rate recovery")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="center right", fontsize=8, ncol=1, bbox_to_anchor=(1.32, 0.5))

    fig.suptitle(f"Recovery after button relocation{title_suffix}", y=1.02)
    out_path = out_dir / "recovery_dynamics.png"
    savefig(out_path, fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SA Red-Blue-Button recovery-after-relocation plot + table")
    parser.add_argument("logs_dir")
    parser.add_argument("--episodes-per-config", type=int, required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    ep = load_episodes(args.logs_dir)
    ep = add_position(ep, args.episodes_per_config)
    recovery = per_event_recovery(ep, args.episodes_per_config)
    summary = build_summary_table(ep, args.episodes_per_config, recovery)

    csv_path = out_dir / "recovery_summary.csv"
    summary.to_csv(csv_path, index=False)

    print(f"\n=== {args.logs_dir} (episodes_per_config={args.episodes_per_config}) ===")
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))

    plot_path = plot_recovery(ep, args.episodes_per_config, out_dir, f" ({Path(args.logs_dir).name})")
    print(f"\nSaved plot: {plot_path}")
    print(f"Saved table: {csv_path}")


if __name__ == "__main__":
    main()
