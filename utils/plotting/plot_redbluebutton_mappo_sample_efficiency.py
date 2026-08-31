"""
Final MAPPO-vs-AIF MA Red-Blue-Button comparison: a sample-efficiency curve.

Standalone, new module -- reads already-computed AIF/OPSRL stats JSON files
read-only, and reads the new MAPPO checkpoint-curve JSON output
(run_scripts_red_blue_doors/multi_agent/run_mappo_checkpoint_curve.py). No
AIF file and no existing MAPPO file is imported for anything other than
reading already-published numbers.

Unlike the Overcooked version, MA Red-Blue-Button's MAPPO environment has NO
teleportation -- every action is a real primitive step, identical to what
the AIF paradigms and OPSRL play against -- so there's no environment-
dynamics asymmetry to disclose here.

IMPORTANT data-provenance note (2026-08-24): the existing 30-seed datasets
under thesis_logs/02_ma_redbluebuttons/ (ma_ind_redbluebutton_step50_30seed,
ma_opsrl_redbluebutton_step50_30seed) are STALE -- their timestamps
(2026-08-14) predate the no-privileged-reset config-boundary fix
(2026-08-20) and OPSRL's thompson_samples=1->10 correction (this file's own
OPSRL data has thompson_samples=1, the pre-fix, invalid value). This script
therefore defaults to the small-but-CURRENT 5-seed reference data from
thesis_logs/02_ma_redbluebuttons/test_5seed_ep100_cfg20_step50/ instead
(confirmed thompson_samples=10, generated on the current, fully-fixed code)
-- override with --ind-stats-glob/--opsrl-stats-glob once a real 30-seed
re-run exists on the current code.

Produces one figure: x = cumulative MAPPO training steps (log scale), y =
success rate (%). AIF/OPSRL are flat reference lines (zero training) with a
shaded 95% CI band from their seed spread (n explicitly annotated -- small
for now, see note above); MAPPO is a growing curve (mean +/- across
training seeds) from the checkpoint sweep.

Usage:
    python3 utils/plotting/plot_redbluebutton_mappo_sample_efficiency.py \\
        --mappo-curve-dir thesis_logs/02_ma_redbluebuttons/mappo_checkpoint_curve \\
        -o thesis_plots/02_ma_redbluebuttons/mappo_sample_efficiency
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np

from thesis_style import PARADIGM_COLORS, ci95, ensure_dir, savefig  # noqa: E402

MAPPO_COLOR = "#7D3C98"

# Default reference sources: the fresh, current-code 5-seed sanity-check run
# (see module docstring for why the older 30-seed directories are excluded).
DEFAULT_REFERENCES = {
    "independent": {
        "label": "Independent (zero-shot)",
        "color_key": "ind",
        "stats_path": project_root / "thesis_logs/02_ma_redbluebuttons/test_5seed_ep100_cfg20_step50/independent_stats.json",
    },
    "opsrl": {
        "label": "OPSRL cold-start (zero-shot RL baseline)",
        "color_key": "OPSRL",
        "stats_path": project_root / "thesis_logs/02_ma_redbluebuttons/test_5seed_ep100_cfg20_step50/opsrl_coldstart_stats.json",
    },
}


def load_reference_from_stats_json(path: Path) -> dict:
    """A *_stats.json file with a top-level success_rate/n_seeds and a
    seed_summaries list (per-seed success_rate) -- the same format both
    run_two_aif_agents_independent.py and run_two_opsrl_agents.py write."""
    d = json.loads(Path(path).read_text())
    per_seed = np.array([s["success_rate"] for s in d["seed_summaries"]])
    return {
        "n_seeds": len(per_seed),
        "mean": float(per_seed.mean()),
        "std": float(per_seed.std()),
        "ci95": ci95(per_seed) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed.tolist(),
        "source": str(path),
    }


def load_reference_from_glob(pattern: str) -> dict:
    """Aggregate across many per-seed *_stats.json files (the 30-seed-style
    layout, one file per seed) -- each file's own top-level success_rate is
    one seed's sample."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    per_seed = []
    for f in files:
        d = json.loads(Path(f).read_text())
        per_seed.append(d["success_rate"])
    per_seed = np.array(per_seed)
    return {
        "n_seeds": len(per_seed),
        "mean": float(per_seed.mean()),
        "std": float(per_seed.std()),
        "ci95": ci95(per_seed) if len(per_seed) > 1 else 0.0,
        "per_seed": per_seed.tolist(),
        "source": pattern,
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
                    "mean": entry["mean_success_rate_across_train_seeds"],
                    "std": entry["std_success_rate_across_train_seeds"],
                    "n_train_seeds": entry["n_train_seeds"],
                }
            )
        return {"rows": rows, "source": "summary"}

    per_seed_files = sorted(curve_dir.glob("mappo_curve_trainseed*.json"))
    if not per_seed_files:
        raise FileNotFoundError(f"No MAPPO curve data found under {curve_dir}")
    by_budget: dict[int, list[float]] = {}
    for f in per_seed_files:
        payload = json.loads(f.read_text())
        for b_str, entry in payload["results_by_budget"].items():
            by_budget.setdefault(int(b_str), []).append(entry["success_rate"])
    rows = [
        {"budget": b, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_train_seeds": len(vals)}
        for b, vals in sorted(by_budget.items())
    ]
    return {"rows": rows, "source": "partial"}


def plot_sample_efficiency(references: dict, mappo_curve: dict, out_dir: Path, using_override: bool) -> Path:
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(9, 6))

    for key, ref in references.items():
        color = PARADIGM_COLORS.get(references_meta[key]["color_key"], "#333333")
        label = f"{references_meta[key]['label']} (n={ref['n_seeds']})"
        ax.axhline(ref["mean"], color=color, linewidth=2, linestyle="-", label=label)
        if ref["ci95"] > 0:
            ax.axhspan(max(0, ref["mean"] - ref["ci95"]), ref["mean"] + ref["ci95"], color=color, alpha=0.12)

    rows = mappo_curve["rows"]
    budgets = np.array([r["budget"] for r in rows], dtype=float)
    means = np.array([r["mean"] for r in rows])
    stds = np.array([r["std"] for r in rows])
    plot_budgets = np.where(budgets <= 0, 1.0, budgets)

    n_ts = rows[0]["n_train_seeds"] if rows else 0
    ax.plot(plot_budgets, means, color=MAPPO_COLOR, linewidth=2.5, marker="o", markersize=6,
            label=f"MAPPO (n={n_ts} training seeds)", zorder=5)
    ax.fill_between(plot_budgets, np.clip(means - stds, 0, 100), np.clip(means + stds, 0, 100),
                     color=MAPPO_COLOR, alpha=0.18, zorder=4)

    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Cumulative MAPPO training steps (log scale)")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("MA Red-Blue-Button: MAPPO sample efficiency vs. zero-training baselines\n(no teleportation -- same primitive-action environment for all methods)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", alpha=0.2)

    if using_override:
        note = (f"Reference sources: --ind-stats-glob/--opsrl-stats-glob override "
                 f"(n={references['independent']['n_seeds']}/{references['opsrl']['n_seeds']} seeds each, current-code data).")
    else:
        note = "Reference n is currently small (5-seed sanity check, not yet a full 30-seed re-run on current code) -- see script docstring for why."
    fig.text(0.5, 0.01, note, ha="center", fontsize=8, color="#555555")

    out_path = out_dir / "mappo_sample_efficiency_curve.png"
    savefig(out_path, fig)
    return out_path


def main() -> None:
    global references_meta
    parser = argparse.ArgumentParser(description="Plot MAPPO sample-efficiency curve vs. AIF/OPSRL for MA Red-Blue-Button")
    parser.add_argument("--mappo-curve-dir", type=str, default="thesis_logs/02_ma_redbluebuttons/mappo_checkpoint_curve")
    parser.add_argument("--ind-stats-glob", type=str, default=None,
                         help="Glob of per-seed IND *_stats.json (30-seed layout). Overrides the default single-file 5-seed reference.")
    parser.add_argument("--opsrl-stats-glob", type=str, default=None,
                         help="Glob of per-seed OPSRL *_stats.json (30-seed layout). Overrides the default single-file 5-seed reference.")
    parser.add_argument("-o", "--out-dir", type=str, default="thesis_plots/02_ma_redbluebuttons/mappo_sample_efficiency")
    args = parser.parse_args()

    curve_dir = Path(args.mappo_curve_dir)
    if not curve_dir.is_absolute():
        curve_dir = project_root / curve_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    references_meta = DEFAULT_REFERENCES
    references = {}
    print("Loading reference numbers (read-only)...")
    if args.ind_stats_glob:
        references["independent"] = load_reference_from_glob(args.ind_stats_glob)
    else:
        references["independent"] = load_reference_from_stats_json(DEFAULT_REFERENCES["independent"]["stats_path"])
    if args.opsrl_stats_glob:
        references["opsrl"] = load_reference_from_glob(args.opsrl_stats_glob)
    else:
        references["opsrl"] = load_reference_from_stats_json(DEFAULT_REFERENCES["opsrl"]["stats_path"])
    for k, ref in references.items():
        print(f"  {DEFAULT_REFERENCES[k]['label']}: mean={ref['mean']:.1f}% +/- 95%CI={ref['ci95']:.1f} (n={ref['n_seeds']})")

    print(f"Loading MAPPO checkpoint curve from {curve_dir}...")
    mappo_curve = load_mappo_curve(curve_dir)
    print(f"  {len(mappo_curve['rows'])} budget point(s), source={mappo_curve['source']}")
    for r in mappo_curve["rows"]:
        print(f"    budget={r['budget']:>8}  success_rate={r['mean']:.1f}% +/- {r['std']:.1f}  (n_train_seeds={r['n_train_seeds']})")

    using_override = bool(args.ind_stats_glob or args.opsrl_stats_glob)
    out_path = plot_sample_efficiency(references, mappo_curve, out_dir, using_override)
    print(f"\nWrote figure: {out_path}")

    summary = {"references": references, "mappo_curve": mappo_curve}
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
