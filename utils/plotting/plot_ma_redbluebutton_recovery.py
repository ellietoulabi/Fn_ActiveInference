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
    """
    ensure_dir(out_dir)
    import matplotlib.pyplot as plt

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

        axes[0].plot(by_pos_all.index, by_pos_all["success"], color=color, lw=2.2, label=label)
        axes[1].plot(by_pos_all.index, by_pos_all["length"], color=color, lw=2.2, label=label)
        axes[2].plot(by_pos_wins.index, by_pos_wins, color=color, lw=2.2, label=label)

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
    axes[2].legend(loc="center left", fontsize=9, bbox_to_anchor=(1.02, 0.5))

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
    print(f"Saved table: {out_dir / 'recovery_summary.csv'}")


if __name__ == "__main__":
    main()
