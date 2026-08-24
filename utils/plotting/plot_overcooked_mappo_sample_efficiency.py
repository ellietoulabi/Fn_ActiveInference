"""
Final MAPPO-vs-AIF Overcooked comparison: a sample-efficiency curve.

Standalone, new module -- reads the three AIF paradigms' already-computed
30-seed logs (thesis_logs/03_ma_overcooked/sal_{ind,ic,fc}_30seed*) read-only
via the existing, unmodified load_seed_csvs() loader, and reads the new
MAPPO checkpoint-curve JSON output (run_mappo_checkpoint_curve.py). No AIF
file and no existing MAPPO file is imported for anything other than reading
already-published numbers or reusing an already-verified CSV loader.

Produces one figure: x = cumulative MAPPO training steps (log scale), y =
mean soups delivered per run. The three AIF paradigms are flat reference
lines (zero training) with a shaded 95% CI band from their 30-seed spread;
MAPPO is a growing curve (mean +/- across training seeds) from the
checkpoint sweep. This directly answers "how much training does MAPPO need
to match paradigm X" as three separate crossover points, rather than
forcing a single win/lose verdict at one arbitrary budget.

Usage:
    python3 utils/plotting/plot_overcooked_mappo_sample_efficiency.py \\
        --mappo-curve-dir thesis_logs/03_ma_overcooked/mappo_checkpoint_curve \\
        -o thesis_plots/03_ma_overcooked/mappo_sample_efficiency
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from plot_sal_semantic_action_level import load_seed_csvs  # noqa: E402  (read-only reuse)
from thesis_style import DPI, PARADIGM_COLORS, ci95, ensure_dir, savefig  # noqa: E402

DELIVERY_REWARD = 20.0
AIF_LOG_DIRS = {
    "ind": project_root / "thesis_logs/03_ma_overcooked/sal_ind_30seed",
    "ic": project_root / "thesis_logs/03_ma_overcooked/sal_ic_30seed_collisionfix",
    "fc": project_root / "thesis_logs/03_ma_overcooked/sal_fc_30seed_collisionfix",
}
AIF_LABELS = {"ind": "Independent", "ic": "Individually Collective", "fc": "Fully Collective"}
MAPPO_COLOR = "#7D3C98"


def deliveries_per_run(df) -> float:
    team_reward = df[["reward_a0", "reward_a1"]].max(axis=1)
    return float(team_reward.sum() / DELIVERY_REWARD)


def load_aif_reference(paradigm: str) -> dict:
    log_dir = AIF_LOG_DIRS[paradigm]
    dfs, _detected, seed_labels = load_seed_csvs(log_dir, paradigm)
    deliveries = np.array([deliveries_per_run(df) for df in dfs])
    return {
        "paradigm": paradigm,
        "label": AIF_LABELS[paradigm],
        "n_seeds": len(deliveries),
        "mean": float(deliveries.mean()),
        "std": float(deliveries.std()),
        "ci95": ci95(deliveries),
        "per_seed": deliveries.tolist(),
    }


def load_mappo_curve(curve_dir: Path) -> dict:
    summary_path = curve_dir / "mappo_curve_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        budgets = sorted(int(b) for b in summary["budgets"].keys())
        rows = []
        for b in budgets:
            entry = summary["budgets"][str(b)]
            rows.append(
                {
                    "budget": b,
                    "mean": entry["mean_deliveries_across_train_seeds"],
                    "std": entry["std_deliveries_across_train_seeds"],
                    "n_train_seeds": entry["n_train_seeds"],
                }
            )
        return {"rows": rows, "source": "summary"}

    # Fall back to per-train-seed partial files if the run hasn't finished /
    # written a final summary yet -- aggregate whatever budgets are common
    # across whatever training seeds have reported in so far.
    per_seed_files = sorted(curve_dir.glob("mappo_curve_trainseed*.json"))
    if not per_seed_files:
        raise FileNotFoundError(f"No MAPPO curve data found under {curve_dir}")
    by_budget: dict[int, list[float]] = {}
    for f in per_seed_files:
        payload = json.loads(f.read_text())
        for b_str, entry in payload["results_by_budget"].items():
            by_budget.setdefault(int(b_str), []).append(entry["mean_deliveries"])
    rows = [
        {
            "budget": b,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n_train_seeds": len(vals),
        }
        for b, vals in sorted(by_budget.items())
    ]
    return {"rows": rows, "source": "partial"}


def plot_sample_efficiency(aif_refs: dict, mappo_curve: dict, out_dir: Path) -> Path:
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(9, 6))

    for key in ("ind", "ic", "fc"):
        ref = aif_refs[key]
        color = PARADIGM_COLORS.get(key, "#333333")
        ax.axhline(ref["mean"], color=color, linewidth=2, linestyle="-", label=f"{ref['label']} (zero-shot)")
        ax.axhspan(ref["mean"] - ref["ci95"], ref["mean"] + ref["ci95"], color=color, alpha=0.12)

    rows = mappo_curve["rows"]
    budgets = np.array([r["budget"] for r in rows], dtype=float)
    means = np.array([r["mean"] for r in rows])
    stds = np.array([r["std"] for r in rows])
    # log scale can't show 0 steps; nudge the untrained point to a small
    # positive value purely for x-axis placement, annotate it explicitly.
    plot_budgets = np.where(budgets <= 0, 1.0, budgets)

    ax.plot(plot_budgets, means, color=MAPPO_COLOR, linewidth=2.5, marker="o", markersize=6, label="MAPPO (this sweep)", zorder=5)
    ax.fill_between(plot_budgets, means - stds, means + stds, color=MAPPO_COLOR, alpha=0.18, zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Cumulative MAPPO training steps (log scale)")
    ax.set_ylabel("Mean soups delivered per run")
    ax.set_title("Overcooked cramped_room: MAPPO sample efficiency vs. zero-training AIF paradigms")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)

    note = (
        f"MAPPO: {rows[0]['n_train_seeds'] if rows else 0} training seed(s) x "
        f"{len(AIF_LOG_DIRS) and 'N'} eval episode(s)/checkpoint  |  "
        f"AIF: 30 seeds each (IND/IC/FC), shaded band = 95% CI"
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="#555555")

    out_path = out_dir / "mappo_sample_efficiency_curve.png"
    savefig(out_path, fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAPPO sample-efficiency curve vs. AIF paradigms")
    parser.add_argument("--mappo-curve-dir", type=str, default="thesis_logs/03_ma_overcooked/mappo_checkpoint_curve")
    parser.add_argument("-o", "--out-dir", type=str, default="thesis_plots/03_ma_overcooked/mappo_sample_efficiency")
    args = parser.parse_args()

    curve_dir = Path(args.mappo_curve_dir)
    if not curve_dir.is_absolute():
        curve_dir = project_root / curve_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    print("Loading AIF 30-seed reference numbers (read-only)...")
    aif_refs = {k: load_aif_reference(k) for k in ("ind", "ic", "fc")}
    for k, ref in aif_refs.items():
        print(f"  {ref['label']}: mean={ref['mean']:.2f} +/- 95%CI={ref['ci95']:.2f} (n={ref['n_seeds']})")

    print(f"Loading MAPPO checkpoint curve from {curve_dir}...")
    mappo_curve = load_mappo_curve(curve_dir)
    print(f"  {len(mappo_curve['rows'])} budget point(s), source={mappo_curve['source']}")
    for r in mappo_curve["rows"]:
        print(f"    budget={r['budget']:>8}  mean_deliveries={r['mean']:.2f} +/- {r['std']:.2f}  (n_train_seeds={r['n_train_seeds']})")

    out_path = plot_sample_efficiency(aif_refs, mappo_curve, out_dir)
    print(f"\nWrote figure: {out_path}")

    summary = {
        "aif_reference": aif_refs,
        "mappo_curve": mappo_curve,
    }
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
