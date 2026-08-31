"""
MA Red-Blue-Button: does performance recover after a button relocation, and
how fast -- for each of Independent/FullyCollective/IndividuallyCollective/
OPSRL(-pretrained), overlaid on one comparison plot?

Motivation: plot_ma_redbluebuttons.py's standard learning-curve plots use a
50-episode rolling smoothing window (W_CURVE, borrowed from Stage 1), but MA
Red-Blue-Button relocates the buttons every 20 episodes by default -- so
that smoothing window blends 2-3 full relocation cycles into one point and
visibly erases the sawtooth recovery pattern entirely (confirmed directly:
the standard success-rate plot is within a few points of dead flat across
every relocation boundary). This script sidesteps smoothing altogether by
aligning every episode to its "position since the most recent relocation"
instead of raw episode index, then aggregating across every relocation
event and every seed at each position -- the exact same technique already
used for Stage 1 (utils/plotting/plot_sa_recovery_dynamics.py /
thesis_plots/01_sa_redbluebuttons/relocation_adaptation/).

Usage:
    python3 utils/plotting/plot_ma_redbluebutton_recovery.py \\
        --independent thesis_logs/02_ma_redbluebuttons/ma_ind_redbluebutton_step50_30seed \\
        --fc thesis_logs/02_ma_redbluebuttons/ma_fc_redbluebutton_step50_30seed \\
        --ic thesis_logs/02_ma_redbluebuttons/ma_ic_redbluebutton_step50_30seed \\
        --opsrl thesis_logs/02_ma_redbluebuttons/ma_opsrl_redbluebutton_step50_30seed \\
        --opsrl-pretrained thesis_logs/02_ma_redbluebuttons/ma_opsrl_pretrained_redbluebutton_step50_30seed \\
        --match-seeds \\
        --episodes-per-config 20 \\
        -o thesis_plots/02_ma_redbluebuttons/recovery_matched_seeds_step50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from thesis_style import ensure_dir, savefig  # noqa: E402
from plot_ma_redbluebuttons import (  # noqa: E402
    AGENT_COLORS,
    PARADIGM_LABELS,
    PARADIGM_ORDER,
    MA_EPISODES_PER_CONFIG_DEFAULT,
    load_ma_redbluebutton_dir,
)

STEADY_FRAC = 0.5   # steady-state = last 50% of a configuration block
RECOVER_TOL = 0.10  # "recovered" (episode length) = within 10% of that block's own steady-state


def load_all_paradigms(dirs: Dict[str, Path], match_seeds: bool) -> pd.DataFrame:
    """Same episode-level loader/seed-matching as plot_ma_redbluebuttons.py,
    duplicated in miniature here rather than imported since that module's
    version returns a concatenated frame without per-paradigm dict access,
    which this script needs for the match_seeds intersection step."""
    loaded = {}
    for key, path in dirs.items():
        if path is None:
            continue
        label = PARADIGM_LABELS[key]
        print(f"Loading {label} from {path} ...")
        ep = load_ma_redbluebutton_dir(path, label)
        print(f"  {ep['seed'].nunique()} seeds, {ep['episode'].nunique()} episodes")
        loaded[label] = ep

    if match_seeds:
        common = None
        for ep in loaded.values():
            seeds = set(ep["seed"].unique())
            common = seeds if common is None else (common & seeds)
        common = sorted(common)
        print(f"\nmatch_seeds=True: restricting every paradigm to the "
              f"{len(common)} common seeds: {common}")
        for label, ep in loaded.items():
            before = ep["seed"].nunique()
            loaded[label] = ep[ep["seed"].isin(common)]
            after = loaded[label]["seed"].nunique()
            if after < before:
                print(f"  {label}: dropped {before - after} seed(s) ({before} -> {after})")

    return loaded


def add_position(ep: pd.DataFrame, episodes_per_config: int) -> pd.DataFrame:
    ep = ep.copy()
    ep["block"] = (ep["episode"] - 1) // episodes_per_config
    ep["position"] = (ep["episode"] - 1) % episodes_per_config
    return ep


def per_event_recovery(ep: pd.DataFrame, episodes_per_config: int) -> pd.DataFrame:
    """Per (seed, block>0): episodes until episode_length returns to within
    RECOVER_TOL of that block's own steady-state length."""
    steady_start = int(episodes_per_config * (1 - STEADY_FRAC))
    rows = []
    for seed, g in ep[ep["block"] > 0].groupby("seed"):
        for block, gb in g.groupby("block"):
            gb = gb.sort_values("position")
            steady_vals = gb.loc[gb["position"] >= steady_start, "episode_length"]
            if len(steady_vals) == 0:
                continue
            steady_mean = steady_vals.mean()
            thresh = steady_mean * (1 + RECOVER_TOL)
            recovered = gb.loc[gb["episode_length"] <= thresh, "position"]
            if len(recovered) == 0:
                continue
            rows.append({"seed": seed, "block": block, "episodes_to_recover": int(recovered.iloc[0])})
    return pd.DataFrame(rows)


def build_summary(label: str, ep: pd.DataFrame, episodes_per_config: int) -> dict:
    steady_start = int(episodes_per_config * (1 - STEADY_FRAC))
    post = ep[ep["block"] > 0]
    shock = post[post["position"] == 0]
    steady = post[post["position"] >= steady_start]
    recovery = per_event_recovery(ep, episodes_per_config)
    return {
        "paradigm": label,
        "n_seeds": ep["seed"].nunique(),
        "shock_success_rate": shock["success"].mean(),
        "steady_success_rate": steady["success"].mean(),
        "shock_mean_length_all": shock["episode_length"].mean(),
        "steady_mean_length_all": steady["episode_length"].mean(),
        # Win-conditional lengths -- the natural recovery signal, not
        # contaminated by the rising timeout rate the way the "_all" columns
        # above are. Compare these two, not the "_all" pair, when arguing
        # about recovery speed.
        "shock_mean_length_wins_only": shock.loc[shock["success"] == 1, "episode_length"].mean(),
        "steady_mean_length_wins_only": steady.loc[steady["success"] == 1, "episode_length"].mean(),
        "median_episodes_to_recover": recovery["episodes_to_recover"].median() if len(recovery) else np.nan,
        "mean_episodes_to_recover": recovery["episodes_to_recover"].mean() if len(recovery) else np.nan,
        "n_relocation_events": len(recovery),
    }


def build_event_aligned(ep: pd.DataFrame, episodes_per_config: int, pre_window: int) -> pd.DataFrame:
    """
    Re-index episodes onto a single continuous "time relative to relocation"
    axis that includes the tail of the OLD configuration as negative
    positions, so a plot can show stable-before -> the relocation -> recovery
    -after as one line instead of only ever starting the clock at the event.

    For each (seed, block>0) relocation event: takes that block's own
    episodes as event_position = 0..episodes_per_config-1 (the "after"), and
    the PRECEDING block's last `pre_window` episodes as
    event_position = -pre_window..-1 (the "before", i.e. that prior
    configuration's own steady state, since it's the tail end of a block that
    already had many episodes to settle).
    """
    frames = []
    for seed, g in ep.groupby("seed"):
        blocks = sorted(g["block"].unique())
        for block in blocks:
            if block == 0:
                continue  # no preceding configuration to compare against
            prev_tail = g[(g["block"] == block - 1) & (g["position"] >= episodes_per_config - pre_window)].copy()
            prev_tail["event_position"] = prev_tail["position"] - episodes_per_config
            curr = g[g["block"] == block].copy()
            curr["event_position"] = curr["position"]
            combo = pd.concat([prev_tail, curr], ignore_index=True)
            combo["event_id"] = f"{seed}_{block}"
            frames.append(combo)
    return pd.concat(frames, ignore_index=True) if frames else ep.iloc[0:0]


def plot_event_aligned(
    loaded: Dict[str, pd.DataFrame], episodes_per_config: int, pre_window: int, out_dir: Path,
) -> Path:
    ensure_dir(out_dir)
    import matplotlib.pyplot as plt

    MIN_WINS_PER_BUCKET = 10
    x_range = list(range(-pre_window, episodes_per_config))
    order = [PARADIGM_LABELS[k] for k in PARADIGM_ORDER if PARADIGM_LABELS[k] in loaded]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for label in order:
        ep = loaded[label]
        aligned = build_event_aligned(ep, episodes_per_config, pre_window)
        wins_only = aligned[aligned["success"] == 1]
        color = AGENT_COLORS.get(label, "#888888")

        by_pos = aligned.groupby("event_position")["success"].mean().reindex(x_range)
        axes[0].plot(x_range, by_pos, color=color, lw=2.2, label=label)

        by_pos_wins = wins_only.groupby("event_position")["episode_length"].mean().reindex(x_range)
        wins_per_bucket = wins_only.groupby("event_position").size().reindex(x_range, fill_value=0)
        if wins_per_bucket.mean() >= MIN_WINS_PER_BUCKET:
            axes[1].plot(x_range, by_pos_wins, color=color, lw=2.2, label=label)

    axes[0].set_ylabel("Mean success rate")
    axes[0].set_title("Before vs. after a relocation: success rate")
    axes[0].set_ylim(-0.03, 1.03)

    axes[1].set_ylabel("Mean episode length (wins only)")
    axes[1].set_title("Before vs. after a relocation: episode length (wins only)")

    for ax in axes:
        ax.axvline(-0.5, color="black", linestyle=":", linewidth=1.3, alpha=0.7)
        ax.text(-0.5, 0.97, "relocation", transform=ax.get_xaxis_transform(),
                fontsize=8, color="#333333", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ax.set_xlabel("Episode, relative to relocation (negative = before, in the OLD configuration)")
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="center left", fontsize=9, bbox_to_anchor=(1.0, 0.5))

    fig.suptitle("MA Red-Blue-Button: performance across a relocation event", y=1.03)
    out_path = out_dir / "event_aligned_recovery.png"
    savefig(out_path, fig)
    return out_path


def plot_recovery(loaded: Dict[str, pd.DataFrame], episodes_per_config: int, out_dir: Path) -> Path:
    """
    Three panels, deliberately kept separate rather than collapsed into one
    "mean episode length" line:
      1. Success rate vs. time-since-relocation -- the honest failure-mode
         story (timeout/deadlock rate rising after the first episode or two).
      2. Mean length over ALL episodes -- kept for transparency; this is the
         panel that looks "backwards" (gets LONGER after relocation) purely
         because a growing timeout RATE mixes in more 50-step failures, not
         because winning episodes themselves get slower.
      3. Mean length among WINS ONLY -- the natural recovery signal: a
         successful episode right after relocation takes longer (the agent
         still has to locate the relocated buttons) and gets faster once
         belief has reconverged, exactly the classic recovery shape. This
         was always in the data; it was just averaged together with panel 2's
         timeout inflation before.

    Panel 3 excludes any paradigm whose average wins-per-position-bucket is
    below MIN_WINS_PER_BUCKET: a "mean" over a handful of wins (or fewer) is
    not a real curve, just sampling noise, and no amount of experiment
    redesign fixes that short of an impractically large seed count for a
    paradigm that rarely wins in the first place (confirmed directly: cold-
    start OPSRL averages ~2 wins per bucket here, vs 60-85 for the AIF
    paradigms) -- its near-zero flat line in panel 1 is already the correct,
    meaningful characterization of its behavior.
    """
    ensure_dir(out_dir)
    import matplotlib.pyplot as plt

    MIN_WINS_PER_BUCKET = 10

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    order = [PARADIGM_LABELS[k] for k in PARADIGM_ORDER if PARADIGM_LABELS[k] in loaded]

    for label in order:
        ep = loaded[label]
        post = ep[ep["block"] > 0]
        wins_only = post[post["success"] == 1]
        color = AGENT_COLORS.get(label, "#888888")

        by_pos_all = post.groupby("position").agg(
            length=("episode_length", "mean"), success=("success", "mean")
        ).reindex(range(episodes_per_config))
        by_pos_wins = wins_only.groupby("position")["episode_length"].mean().reindex(range(episodes_per_config))
        wins_per_bucket = wins_only.groupby("position").size().reindex(range(episodes_per_config), fill_value=0)

        axes[0].plot(by_pos_all.index, by_pos_all["success"], color=color, lw=2.2, label=label)
        axes[1].plot(by_pos_all.index, by_pos_all["length"], color=color, lw=2.2, label=label)
        if wins_per_bucket.mean() >= MIN_WINS_PER_BUCKET:
            axes[2].plot(by_pos_wins.index, by_pos_wins, color=color, lw=2.2, label=label)
        else:
            print(f"  {label}: excluded from win-only panel (avg {wins_per_bucket.mean():.1f} "
                  f"wins/bucket < {MIN_WINS_PER_BUCKET} -- not enough wins for a meaningful mean)")

    excluded = [
        label for label in order
        if not any(line.get_label() == label for line in axes[2].get_lines())
    ]

    axes[0].set_xlabel("Episodes since relocation")
    axes[0].set_ylabel("Mean success rate")
    axes[0].set_title("Success rate recovery")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].set_xlabel("Episodes since relocation")
    axes[1].set_ylabel("Mean episode length (all episodes)")
    axes[1].set_title("All-episode length\n(inflated by rising timeout rate, not a recovery signal)")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[2].set_xlabel("Episodes since relocation")
    axes[2].set_ylabel("Mean episode length (wins only)")
    axes[2].set_title("Length among WINS only\n(the real recovery signal)")
    axes[2].grid(True, axis="y", alpha=0.25)
    if excluded:
        axes[2].text(
            0.5, -0.24, f"excluded (too few wins to average): {', '.join(excluded)}",
            transform=axes[2].transAxes, ha="center", fontsize=8.5, color="#666666",
        )

    # Full-paradigm legend (from panel 0, which always has every paradigm)
    # placed once for the whole figure rather than per-panel.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", fontsize=9, bbox_to_anchor=(1.0, 0.5))

    fig.suptitle(f"MA Red-Blue-Button: recovery after button relocation "
                 f"(episodes_per_config={episodes_per_config})", y=1.02)
    out_path = out_dir / "recovery_dynamics.png"
    savefig(out_path, fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="MA Red-Blue-Button relocation-recovery plot + table")
    parser.add_argument("--independent", type=Path, default=None)
    parser.add_argument("--fc", type=Path, default=None)
    parser.add_argument("--ic", type=Path, default=None)
    parser.add_argument("--opsrl", type=Path, default=None)
    parser.add_argument("--opsrl-pretrained", type=Path, default=None)
    parser.add_argument("--episodes-per-config", type=int, default=MA_EPISODES_PER_CONFIG_DEFAULT)
    parser.add_argument("--match-seeds", action="store_true")
    parser.add_argument("--pre-window", type=int, default=10,
                         help="How many episodes of the OLD configuration's tail to show before "
                              "the relocation in the before/after plot (default 10)")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args()

    dirs = {
        "independent": args.independent, "fc": args.fc, "ic": args.ic,
        "opsrl": args.opsrl, "opsrl_pretrained": args.opsrl_pretrained,
    }
    dirs = {k: v for k, v in dirs.items() if v is not None}
    if not dirs:
        parser.error("Provide at least one of --independent/--fc/--ic/--opsrl/--opsrl-pretrained")

    out_dir = ensure_dir(args.out)
    loaded = load_all_paradigms(dirs, args.match_seeds)
    loaded = {label: add_position(ep, args.episodes_per_config) for label, ep in loaded.items()}

    summaries = [build_summary(label, ep, args.episodes_per_config) for label, ep in loaded.items()]
    table = pd.DataFrame(summaries)
    table.to_csv(out_dir / "recovery_summary.csv", index=False)
    print("\n=== Recovery summary ===")
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))

    plot_path = plot_recovery(loaded, args.episodes_per_config, out_dir)
    print(f"\nSaved plot: {plot_path}")

    event_plot_path = plot_event_aligned(loaded, args.episodes_per_config, args.pre_window, out_dir)
    print(f"Saved plot: {event_plot_path}")
    print(f"Saved table: {out_dir / 'recovery_summary.csv'}")


if __name__ == "__main__":
    main()
