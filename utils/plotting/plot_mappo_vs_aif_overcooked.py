"""
Overcooked (cramped_room): MAPPO sample-efficiency curve vs. the three AIF
paradigms' zero-shot mean deliveries -- with the training budget at which
MAPPO's own learning curve first reaches each AIF paradigm's mean deliveries
explicitly marked on the plot.

Data sources (no re-simulation):
- MAPPO: thesis_logs/03_ma_overcooked/mappo_checkpoint_curve_step1500_budgets..._30seed/
  mappo_curve_summary_seed{76..105}.json -- one file per training seed, each
  giving that seed's own live-policy deliveries at every checkpoint budget
  (n_train_seeds=1 per file; this script aggregates across the 30 files itself).
- AIF: thesis_logs/03_ma_overcooked/sal_{ind_30seed,ic_30seed_collisionfix,
  fc_30seed_collisionfix}/*.csv -- final cumulative_reward_a0 / 20 = deliveries
  for that seed (team-shared reward, not doubled -- see ai/02-debug.md's
  "combined return was ~2x team return" fix, already applied to these logs).

Usage
-----
python utils/plotting/plot_mappo_vs_aif_overcooked.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from thesis_style import PARADIGM_COLORS, DPI, ci95, ensure_dir  # noqa: E402

_SEED_RE = re.compile(r"seed(\d+)\.json$")

AIF_PARADIGMS = [
    ("Independent", "sal_ind_30seed"),
    ("IndividuallyCollective", "sal_ic_30seed_collisionfix"),
    ("FullyCollective", "sal_fc_30seed_collisionfix"),
]
_MAPPO_COLOR = "#555555"


def load_mappo_curve(log_dir: Path) -> pd.DataFrame:
    """Returns long-form (seed, budget, deliveries)."""
    log_dir = Path(log_dir)
    files = sorted(log_dir.glob("mappo_curve_summary_seed*.json"))
    if not files:
        raise FileNotFoundError(f"No mappo_curve_summary_seed*.json in {log_dir}")
    rows = []
    for f in files:
        m = _SEED_RE.search(f.name)
        seed = int(m.group(1))
        data = json.loads(f.read_text())
        for budget_str, stats in data["budgets"].items():
            rows.append({
                "seed": seed,
                "budget": int(budget_str),
                "deliveries": stats["mean_deliveries_across_train_seeds"],
            })
    df = pd.DataFrame(rows)
    print(f"  Loaded MAPPO curve: {df['seed'].nunique()} seeds, "
          f"{df['budget'].nunique()} budgets from {log_dir.name}")
    return df


def load_aif_deliveries(log_dir: Path) -> pd.Series:
    """Returns per-seed final deliveries (episode_seed -> deliveries)."""
    log_dir = Path(log_dir)
    files = sorted(log_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs in {log_dir}")
    vals = {}
    for f in files:
        df = pd.read_csv(f, usecols=["episode_seed", "cumulative_reward_a0"])
        seed = int(df["episode_seed"].iloc[0])
        vals[seed] = df["cumulative_reward_a0"].iloc[-1] / 20.0
    return pd.Series(vals, name="deliveries")


def aggregate_mappo_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Per-budget mean + 95% CI across seeds (percentile bootstrap, matching
    the rest of this project's curve-aggregation convention)."""
    budgets = np.sort(df["budget"].unique())
    seeds = np.sort(df["seed"].unique())
    mat = np.full((len(seeds), len(budgets)), np.nan)
    pivot = df.pivot(index="seed", columns="budget", values="deliveries").reindex(
        index=seeds, columns=budgets
    )
    mat = pivot.values
    rng = np.random.default_rng(0)
    n_boot = 2000
    boot = np.empty((n_boot, len(budgets)))
    for b in range(n_boot):
        idx = rng.integers(0, len(seeds), size=len(seeds))
        boot[b] = np.nanmean(mat[idx], axis=0)
    mean = np.nanmean(mat, axis=0)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return pd.DataFrame({"budget": budgets, "mean": mean, "lo": lo, "hi": hi})


def find_crossing_budget(curve: pd.DataFrame, target: float) -> Tuple[float, bool]:
    """
    First ACTUALLY-TESTED training budget at which MAPPO's mean deliveries
    curve reaches or exceeds `target` -- no interpolation between tested
    budgets, since the curve is only ever known at the 11 points that were
    really evaluated and a fractional "217,188 steps" implies a precision
    the noisy, widely-spaced data doesn't support. Returns (budget, reached)
    where `reached` is False if the curve never reaches the target within
    the tested range (in which case the returned budget is the largest
    tested one, marking where the search gave up rather than a real crossing).
    """
    b = curve["budget"].values
    m = curve["mean"].values
    for i in range(len(b)):
        if m[i] >= target:
            return float(b[i]), True
    return float(b[-1]), False


def plot_mappo_vs_aif(
    mappo_curve: pd.DataFrame,
    aif_means: Dict[str, Tuple[float, float]],
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(mappo_curve["budget"], mappo_curve["mean"], color=_MAPPO_COLOR,
             linewidth=2.6, marker="o", markersize=5, label="MAPPO (mean over 30 seeds)", zorder=5)
    ax.fill_between(mappo_curve["budget"], mappo_curve["lo"], mappo_curve["hi"],
                     color=_MAPPO_COLOR, alpha=0.15, zorder=2)

    y_top = max(mappo_curve["hi"].max(), max(m for m, _ in aif_means.values())) * 1.15
    ax.set_ylim(0, y_top)
    ax.set_xlim(mappo_curve["budget"].min() - 5000, mappo_curve["budget"].max() * 1.02)

    # Sort paradigms by mean purely for consistent legend/table ordering.
    ordered = sorted(aif_means.items(), key=lambda kv: kv[1][0])
    for paradigm, (mean_val, ci_val) in ordered:
        color = PARADIGM_COLORS.get(paradigm, "#888888")
        ax.axhline(mean_val, color=color, linestyle="--", linewidth=1.8, alpha=0.8,
                    zorder=3, label=f"{paradigm} (zero-shot, {mean_val:.2f})")
        ax.axhspan(mean_val - ci_val, mean_val + ci_val, color=color, alpha=0.08, zorder=1)

    # No crossing-point markers/annotations on the plot itself -- see
    # crossing_points.csv / crossing_points.md for the (non-interpolated)
    # table instead, which is where that detail belongs.

    ax.set_xlabel("MAPPO training budget (environment steps)", fontsize=13)
    ax.set_ylabel("Mean soups delivered per 1500-step run", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    ax.grid(alpha=0.2)

    plt.tight_layout()

    # Standard-DPI PNG (matches every other thesis figure), a high-DPI PNG
    # for print-quality use, and a vector PDF (infinite effective resolution,
    # smallest file size of the three, preferred for LaTeX inclusion).
    HIGH_DPI = 600
    png_path = out_path.with_suffix(".png")
    hi_png_path = out_path.with_name(out_path.stem + "_hires.png")
    pdf_path = out_path.with_suffix(".pdf")
    plt.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(hi_png_path, dpi=HIGH_DPI, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {png_path}")
    print(f"  Saved {hi_png_path} ({HIGH_DPI} DPI)")
    print(f"  Saved {pdf_path} (vector)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mappo-dir", type=Path,
        default=Path(
            "thesis_logs/03_ma_overcooked/"
            "mappo_checkpoint_curve_step1500_budgets0_25000_75000_100000_150000_200000_250000_300000_350000_400000_500000_30seed"
        ),
    )
    parser.add_argument("--overcooked-dir", type=Path, default=Path("thesis_logs/03_ma_overcooked"))
    parser.add_argument(
        "-o", "--output-dir", type=Path,
        default=Path("thesis_plots/03_ma_overcooked/mappo_vs_aif"),
    )
    args = parser.parse_args()
    out_dir = ensure_dir(args.output_dir)

    print("Loading MAPPO checkpoint curve...")
    mappo_raw = load_mappo_curve(args.mappo_dir)
    mappo_curve = aggregate_mappo_curve(mappo_raw)
    mappo_curve.round(4).to_csv(out_dir / "mappo_curve_aggregated.csv", index=False)

    print("Loading AIF paradigm deliveries...")
    aif_means = {}
    aif_seedlevel = {}
    for paradigm, subdir in AIF_PARADIGMS:
        vals = load_aif_deliveries(args.overcooked_dir / subdir)
        aif_seedlevel[paradigm] = vals
        aif_means[paradigm] = (float(vals.mean()), float(ci95(vals.values)))
        print(f"  {paradigm}: mean={vals.mean():.3f} [95% CI ±{ci95(vals.values):.3f}], n={len(vals)} seeds")

    print("\nCrossing points (first actually-tested MAPPO budget, no interpolation):")
    rows = []
    for paradigm, (mean_val, ci_val) in sorted(aif_means.items(), key=lambda kv: kv[1][0]):
        cross_x, reached = find_crossing_budget(mappo_curve, mean_val)
        mappo_mean_at_cross = float(mappo_curve.loc[mappo_curve["budget"] == cross_x, "mean"].iloc[0])
        status = f"{cross_x:,.0f} steps" if reached else f"NOT REACHED by {cross_x:,.0f} steps (largest tested budget)"
        print(f"  MAPPO matches {paradigm} (mean={mean_val:.2f}): {status}")
        rows.append({
            "paradigm": paradigm, "aif_mean_deliveries": round(mean_val, 3), "aif_ci95": round(ci_val, 3),
            "mappo_budget_first_reaching_it": cross_x, "mappo_mean_at_that_budget": round(mappo_mean_at_cross, 3),
            "reached_within_tested_range": reached,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "crossing_points.csv", index=False)

    md_lines = [
        "| Paradigm | Zero-shot mean deliveries | MAPPO budget first reaching it | MAPPO's mean at that budget |",
        "|---|---|---|---|",
    ]
    for r in rows:
        budget_str = f"{r['mappo_budget_first_reaching_it']:,.0f}" + ("" if r["reached_within_tested_range"] else " (not reached)")
        md_lines.append(
            f"| {r['paradigm']} | {r['aif_mean_deliveries']:.2f} ± {r['aif_ci95']:.2f} | "
            f"{budget_str} | {r['mappo_mean_at_that_budget']:.2f} |"
        )
    (out_dir / "crossing_points.md").write_text("\n".join(md_lines) + "\n")
    print(f"  Saved {out_dir / 'crossing_points.csv'}")
    print(f"  Saved {out_dir / 'crossing_points.md'}")

    print("\nPlotting...")
    plot_mappo_vs_aif(mappo_curve, aif_means, out_dir / "mappo_vs_aif_deliveries.png")

    print(f"\nDone. Saved to {out_dir}")


if __name__ == "__main__":
    main()
