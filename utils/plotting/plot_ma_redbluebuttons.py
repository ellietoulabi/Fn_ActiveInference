"""
MA Red-Blue-Button (Stage 2): clean comparison plots across the three AIF
paradigms (Independent, FullyCollective, IndividuallyCollective) and the
cold-start OPSRL baseline, from the per-step CSV logs produced by
run_two_aif_agents_{independent,fully_collective,individually_collective}.py
and run_two_opsrl_agents.py.

Deliberately reuses Stage 1's plotting machinery (plot_sa_redbluebuttons_nine.py)
rather than re-implementing it: Stage 2's data has the exact same shape Stage 1's
does (multiple seeds, episodes with a per-episode win/lose/timeout outcome,
periodic button-relocation config boundaries) -- only Stage 1's data comes as
one CSV with an `agent` column selecting among 9 algorithms, while Stage 2's
comes as one CSV per (paradigm, seed) with no `agent` column. This script's
only real job is to load Stage 2's CSVs into the exact same
[seed, algorithm, episode, episode_return, success, episode_length] shape
Stage 1's `build_curves_per_seed`/`save_all_learning_curve_variants`/
`plot_c_ecdf_first_success`/`plot_d_ecdf_stable_success` already expect, then
call those functions directly -- guaranteeing the same style, not just a
matching one, since it's literally the same plotting code. Colors come from
thesis_style.py, which is also what Stage 1's script (as of this refactor) and
Stage 3's SAL scripts pull from, so "orange = Independent, blue =
IndividuallyCollective, green = FullyCollective" reads the same way in every
stage's figures.

Data scope (2026-08-22, per explicit decision): cold-start OPSRL only. The
pretrained-OPSRL variants (run_two_opsrl_agents_pretrained{,_sweep}.py) don't
yet have enough seeds run to plot with a real CI -- see ai/02-debug.md.

Usage:
    python utils/plotting/plot_ma_redbluebuttons.py \\
        --independent aug21logs/ma_ind_redbluebutton_step50_30seed \\
        --fc aug21logs/ma_fc_redbluebutton_step50_30seed \\
        --ic aug21logs/ma_ic_redbluebutton_step50_30seed \\
        --opsrl aug21logs/ma_opsrl_redbluebutton_step50_30seed \\
        -o thesis_plots/02_ma_redbluebuttons/comparison_step50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from thesis_style import PARADIGM_COLORS, ensure_dir  # noqa: E402
from plot_sa_redbluebuttons_nine import (  # noqa: E402
    AGENT_COLORS,
    EPISODES_PER_CONFIG_DEFAULT,
    W_STABLE,
    THETA,
    build_curves_per_seed,
    save_all_learning_curve_variants,
    plot_c_ecdf_first_success,
    plot_d_ecdf_stable_success,
)

# Keep the reused Stage-1 module's AGENT_COLORS in sync with thesis_style's
# paradigm colors (same values already, this just guards against the two
# drifting apart if either is edited independently in the future).
AGENT_COLORS.update({k: v for k, v in PARADIGM_COLORS.items() if k in (
    "Independent", "IndividuallyCollective", "FullyCollective", "OPSRL",
)})

# MA Red-Blue-Button's default protocol (run_two_aif_agents_*.sh /
# two_opsrl.sh): 100 episodes, config changes every 20 episodes.
MA_EPISODES_PER_CONFIG_DEFAULT = 20

PARADIGM_LABELS = {
    "independent": "Independent",
    "fc": "FullyCollective",
    "ic": "IndividuallyCollective",
    "opsrl": "OPSRL",
}

# Plot order: floor -> ceiling -> contribution -> RL baseline, matching the
# thesis's own H2 framing (Independent floor, FullyCollective ceiling,
# IndividuallyCollective the contribution) with OPSRL last as the RL baseline.
PARADIGM_ORDER = ["independent", "fc", "ic", "opsrl"]


def load_ma_redbluebutton_dir(logs_dir: Path, label: str) -> pd.DataFrame:
    """
    Load every per-seed CSV in a directory (one paradigm) into per-episode
    rows: [seed, algorithm, episode, episode_return, success, episode_length].

    Matches Stage 1's load_episode_data()'s aggregation exactly (sum of
    per-step reward = episode_return, success = episode_return >= 1.0, episode
    length = max step) -- MA Red-Blue-Button's CSVs are one row per (episode,
    step) with a `reward` and `step` column, same as Stage 1's, just without
    an `agent` column since each file is already one (paradigm, seed).
    """
    files = sorted(logs_dir.glob("*.csv"))
    files = [f for f in files if not f.name.endswith("_stats.json")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {logs_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=lambda c: c in ("seed", "episode", "step", "reward"))
        if not {"seed", "episode", "step", "reward"} <= set(df.columns):
            raise ValueError(f"{f} missing one of seed/episode/step/reward columns")
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)

    ep = raw.groupby(["seed", "episode"], as_index=False).agg(
        episode_return=("reward", "sum"),
        episode_length=("step", "max"),
    )
    ep["success"] = (ep["episode_return"] >= 1.0).astype(int)
    ep["algorithm"] = label
    return ep


def load_all_paradigms(dirs: Dict[str, Path]) -> pd.DataFrame:
    parts = []
    for key, path in dirs.items():
        if path is None:
            continue
        label = PARADIGM_LABELS[key]
        print(f"Loading {label} from {path} ...")
        ep = load_ma_redbluebutton_dir(path, label)
        n_seeds = ep["seed"].nunique()
        n_eps = ep["episode"].nunique()
        print(f"  {n_seeds} seeds, {n_eps} distinct episode indices, "
              f"{len(ep)} (seed,episode) rows")
        parts.append(ep)
    return pd.concat(parts, ignore_index=True)


def plot_mean_success_bar(episode_df: pd.DataFrame, agent_names: List[str], output_path: Path) -> None:
    """
    Headline bar chart: mean success rate per paradigm +/- 95% CI across seeds,
    matching Stage 3's plot_sal_triple_comparison.py::plot_mean_soups bar style
    (mean +/- CI bars, same color convention) extended to include OPSRL.

    Each x-tick label is annotated with its actual seed count (n=...) rather
    than leaving seed count implicit in a shared title -- when paradigms have
    unequal coverage (as IndividuallyCollective currently does, ~30 min/episode
    per its own well-documented per-step cost -- see ai/02-debug.md), a bare
    "mean success rate" bar for the under-sampled paradigm can badly overstate
    performance via survivorship bias: seeds that finished within whatever
    time budget produced the log are disproportionately the FAST-solving (and
    so also easy/successful) ones, while seeds that got stuck in a hard,
    long-running episode are more likely to be exactly the ones missing from
    the log entirely. Confirmed directly for this dataset, not assumed: IC's
    10 available seeds average 4.5 steps/episode vs. Independent's 15.9 across
    all 30 -- consistent with, though not conclusive proof of, this bias.
    """
    import matplotlib.pyplot as plt
    from thesis_style import ci95, savefig

    seed_means = (
        episode_df.groupby(["algorithm", "seed"])["success"].mean().reset_index()
    )
    seed_counts = seed_means.groupby("algorithm")["seed"].nunique()
    max_n = seed_counts.max()

    means, errs, colors, labels = [], [], [], []
    for alg in agent_names:
        vals = seed_means.loc[seed_means["algorithm"] == alg, "success"]
        means.append(vals.mean())
        errs.append(ci95(vals))
        colors.append(AGENT_COLORS.get(alg, "#888888"))
        n = seed_counts.get(alg, 0)
        flag = " *" if n < max_n else ""
        labels.append(f"{alg}\n(n={n}){flag}")
        if n < max_n:
            print(f"  WARNING: {alg} has only {n}/{max_n} seeds -- its bar may be "
                  f"survivorship-biased (see plot_mean_success_bar's docstring). "
                  f"Do not cite this number without checking whether the missing "
                  f"seeds are missing for a reason correlated with the outcome.")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=6, color=colors, alpha=0.9, edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Mean success rate")
    title = "Mean success rate by paradigm (95% CI)"
    if (seed_counts < max_n).any():
        title += "  [* = incomplete seed coverage, see caveat]"
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(output_path)
    print(f"  Saved {output_path}")


def plot_seed_dotplot(episode_df: pd.DataFrame, agent_names: List[str], metric: str,
                       ylabel: str, output_path: Path) -> None:
    """Per-seed dot plot for one metric, one column of dots per paradigm, same
    visual idiom as Stage 1's seed-dot plots (mean line + individual seed dots)."""
    import matplotlib.pyplot as plt
    from thesis_style import ci95, savefig

    seed_vals = episode_df.groupby(["algorithm", "seed"])[metric].mean().reset_index()
    seed_counts = seed_vals.groupby("algorithm")["seed"].nunique()
    max_n = seed_counts.max()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rng = np.random.default_rng(0)
    tick_labels = []
    for i, alg in enumerate(agent_names):
        vals = seed_vals.loc[seed_vals["algorithm"] == alg, metric].to_numpy()
        color = AGENT_COLORS.get(alg, "#888888")
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=color, alpha=0.55, s=28, zorder=2)
        m = vals.mean()
        err = ci95(vals)
        ax.errorbar([i], [m], yerr=[err] if not np.isnan(err) else None, fmt="o",
                    color=color, markersize=10, markeredgecolor="black", capsize=6, zorder=3)
        n = seed_counts.get(alg, 0)
        tick_labels.append(f"{alg}\n(n={n})" + (" *" if n < max_n else ""))
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} per seed (dots), mean +/- 95% CI")
    ax.grid(True, axis="y", alpha=0.3)
    savefig(output_path)
    print(f"  Saved {output_path}")


def run_ma_redbluebutton_plots(
    dirs: Dict[str, Path],
    output_dir: Path,
    episodes_per_config: int = MA_EPISODES_PER_CONFIG_DEFAULT,
) -> None:
    output_dir = ensure_dir(output_dir)
    episode_df = load_all_paradigms(dirs)

    agent_names = np.array([PARADIGM_LABELS[k] for k in PARADIGM_ORDER if k in dirs and dirs[k] is not None])
    episodes = np.sort(episode_df["episode"].unique())
    print(f"\nEpisodes 1-{episodes.max()}, paradigms: {list(agent_names)}")

    returns_per_agent, success_per_agent, length_per_agent = build_curves_per_seed(
        episode_df, agent_names, episodes
    )

    print("\nCreating learning curves...")
    save_all_learning_curve_variants(
        success_per_agent, agent_names, episodes, output_dir,
        stem="success_rate", ylabel="Success rate", title_base="Mean success rate vs episode",
        episodes_per_config=episodes_per_config, ylim=(-0.05, 1.05),
    )
    save_all_learning_curve_variants(
        returns_per_agent, agent_names, episodes, output_dir,
        stem="episode_return", ylabel="Episode return", title_base="Mean episode return vs episode",
        episodes_per_config=episodes_per_config,
    )
    save_all_learning_curve_variants(
        length_per_agent, agent_names, episodes, output_dir,
        stem="episode_length", ylabel="Episode length (steps)", title_base="Mean episode length vs episode",
        episodes_per_config=episodes_per_config,
    )

    print("\nCreating ECDF plots...")
    plot_c_ecdf_first_success(success_per_agent, agent_names, episodes, output_dir / "ecdf_first_success.png")
    print(f"  Saved {output_dir / 'ecdf_first_success.png'}")
    plot_d_ecdf_stable_success(success_per_agent, agent_names, episodes, output_dir / "ecdf_stable_success.png")
    print(f"  Saved {output_dir / 'ecdf_stable_success.png'}")

    print("\nCreating headline comparison plots...")
    plot_mean_success_bar(episode_df, list(agent_names), output_dir / "mean_success_rate_bar.png")
    plot_seed_dotplot(episode_df, list(agent_names), "success", "Success rate", output_dir / "success_rate_seed_dots.png")
    plot_seed_dotplot(episode_df, list(agent_names), "episode_return", "Mean episode return", output_dir / "episode_return_seed_dots.png")
    plot_seed_dotplot(episode_df, list(agent_names), "episode_length", "Mean episode length", output_dir / "episode_length_seed_dots.png")

    print(f"\nDone. Saved MA Red-Blue-Button comparison plots to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MA Red-Blue-Button clean comparison plots (Independent/FullyCollective/"
                    "IndividuallyCollective AIF + cold-start OPSRL)."
    )
    parser.add_argument("--independent", type=Path, default=None, help="Independent AIF logs directory")
    parser.add_argument("--fc", type=Path, default=None, help="FullyCollective AIF logs directory")
    parser.add_argument("--ic", type=Path, default=None, help="IndividuallyCollective AIF logs directory")
    parser.add_argument("--opsrl", type=Path, default=None, help="Cold-start OPSRL logs directory")
    parser.add_argument("--out", "-o", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--episodes-per-config", type=int, default=MA_EPISODES_PER_CONFIG_DEFAULT,
        help="CI/vertical-line spacing for config-relocation boundaries (default 20)",
    )
    args = parser.parse_args()

    dirs = {
        "independent": args.independent,
        "fc": args.fc,
        "ic": args.ic,
        "opsrl": args.opsrl,
    }
    dirs = {k: v for k, v in dirs.items() if v is not None}
    if not dirs:
        parser.error("Provide at least one of --independent/--fc/--ic/--opsrl")

    run_ma_redbluebutton_plots(dirs, args.out, args.episodes_per_config)


if __name__ == "__main__":
    main()
