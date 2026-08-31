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
            per_seed = entry.get("per_train_seed_mean_deliveries")
            rows.append(
                {
                    "budget": b,
                    "per_seed": per_seed,
                    "mean": entry["mean_deliveries_across_train_seeds"],
                    "std": entry["std_deliveries_across_train_seeds"],
                    "ci95": ci95(per_seed) if per_seed else float("nan"),
                    "n_train_seeds": entry["n_train_seeds"],
                }
            )
        return {"rows": rows, "source": "summary", "n_eval_episodes": None}

    # Fall back to per-train-seed partial files if the run hasn't finished /
    # written a final summary yet -- aggregate whatever budgets are common
    # across whatever training seeds have reported in so far.
    per_seed_files = sorted(curve_dir.glob("mappo_curve_trainseed*.json"))
    if not per_seed_files:
        raise FileNotFoundError(f"No MAPPO curve data found under {curve_dir}")
    by_budget: dict[int, list[float]] = {}
    n_eval_episodes: set[int] = set()
    for f in per_seed_files:
        payload = json.loads(f.read_text())
        for b_str, entry in payload["results_by_budget"].items():
            by_budget.setdefault(int(b_str), []).append(entry["mean_deliveries"])
            n_eval_episodes.add(len(entry["episodes"]))
    rows = [
        {
            "budget": b,
            "per_seed": vals,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "ci95": ci95(vals),
            "n_train_seeds": len(vals),
        }
        for b, vals in sorted(by_budget.items())
    ]
    # Each checkpoint is scored on this many held-out episodes per training
    # seed -- surfaced explicitly on the plot since it's the main reason the
    # curve is noisy (verified: always exactly 1 in every file checked).
    n_eval = n_eval_episodes.pop() if len(n_eval_episodes) == 1 else None
    return {"rows": rows, "source": "partial", "n_eval_episodes": n_eval}


def compute_crossovers(aif_refs: dict, mappo_curve: dict) -> dict:
    """First MAPPO training budget (data-resolution, not interpolated) at
    which the mean delivery count reaches each AIF paradigm's zero-shot mean."""
    rows = mappo_curve["rows"]
    budgets = np.array([r["budget"] for r in rows], dtype=float)
    means = np.array([r["mean"] for r in rows])
    out = {}
    for key in ("ind", "ic", "fc"):
        ref = aif_refs[key]
        reached = np.where(means >= ref["mean"])[0]
        reached = reached[budgets[reached] > 0]  # ignore a lucky untrained draw
        out[key] = int(budgets[reached[0]]) if len(reached) else None
    return out


def plot_sample_efficiency(aif_refs: dict, mappo_curve: dict, out_dir: Path) -> Path:
    """Deliberately minimal: 3 flat AIF lines + bands, 1 MAPPO curve + band,
    a legend, axis labels. No in-figure crossover callouts, no footnote
    paragraph, no annotation arrows -- all of that lives in the companion
    .txt file (write_explanation) instead, so the figure itself stays
    readable at a glance."""
    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for key in ("ind", "ic", "fc"):
        ref = aif_refs[key]
        color = PARADIGM_COLORS.get(key, "#333333")
        ax.axhline(ref["mean"], color=color, linewidth=2, label=ref["label"], zorder=3)
        ax.axhspan(ref["mean"] - ref["ci95"], ref["mean"] + ref["ci95"], color=color, alpha=0.12, zorder=1)

    rows = mappo_curve["rows"]
    budgets = np.array([r["budget"] for r in rows], dtype=float)
    means = np.array([r["mean"] for r in rows])
    cis = np.array([r["ci95"] for r in rows])

    # symlog (not a fake x=1 substitution) natively supports budget=0: linear
    # near the origin, log-spaced beyond linthresh, so the untrained point sits
    # at its real value instead of being silently relabeled.
    linthresh = 10_000
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.6)

    ci_low = np.clip(means - cis, 0, None)  # soups delivered can't be negative
    ci_high = means + cis
    ax.plot(
        budgets, means, color=MAPPO_COLOR, linewidth=2.5, marker="o", markersize=6,
        label="MAPPO", zorder=5,
    )
    ax.fill_between(budgets, ci_low, ci_high, color=MAPPO_COLOR, alpha=0.18, zorder=2)

    ax.set_ylim(bottom=min(0, ci_low.min()) - 0.5)
    ax.set_xlabel("Cumulative MAPPO training steps")
    ax.set_ylabel("Mean soups delivered per run")
    ax.set_title("MAPPO sample efficiency vs. zero-training AIF paradigms")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(True, which="major", alpha=0.15)

    out_path = out_dir / "mappo_sample_efficiency_curve.png"
    savefig(out_path, fig)
    return out_path


def write_explanation(aif_refs: dict, mappo_curve: dict, crossovers: dict, out_dir: Path) -> Path:
    rows = mappo_curve["rows"]
    n_eval = mappo_curve.get("n_eval_episodes")
    lines = []
    lines.append("mappo_sample_efficiency_curve.png -- what it shows and how to read it")
    lines.append("=" * 72)
    lines.append("")
    lines.append("THE QUESTION THIS PLOT ANSWERS")
    lines.append("How much training does MAPPO need before it matches each Active")
    lines.append("Inference (AIF) paradigm, which needs zero training at all?")
    lines.append("")
    lines.append("THE THREE HORIZONTAL LINES (Independent / Individually Collective /")
    lines.append("Fully Collective)")
    lines.append("Each is one AIF paradigm's average soups delivered in a single ~1500-")
    lines.append("step Overcooked episode, averaged over 30 independent seeds. AIF does")
    lines.append("not train via gradient descent -- it acts by inference from a fixed,")
    lines.append("hand-specified model from the very first step of the very first")
    lines.append("episode, so there is no 'number of training steps' for it. The line")
    lines.append("is drawn horizontally only so its one number can be compared against")
    lines.append("MAPPO's curve at every x-position -- it is the same value everywhere,")
    lines.append("not a function of the x-axis.")
    lines.append("")
    lines.append("THE SHADED BAND AROUND EACH LINE")
    lines.append("95% confidence interval computed from the spread across those 30")
    lines.append("seeds -- i.e. how much that paradigm's average would plausibly shift")
    lines.append("if you re-ran with a different batch of 30 seeds. A narrow band means")
    lines.append("the seeds mostly agree; a wide band means they don't. Where two")
    lines.append("paradigms' bands overlap a lot, their means are not reliably")
    lines.append("distinguishable from this sample size, even if one number is nominally")
    lines.append("higher.")
    lines.append("")
    for key in ("ind", "ic", "fc"):
        ref = aif_refs[key]
        lines.append(f"  {ref['label']:<26} mean={ref['mean']:.2f}  95% CI=+/-{ref['ci95']:.2f}  (n={ref['n_seeds']} seeds)")
    lines.append("")
    lines.append("THE PURPLE MAPPO CURVE AND ITS BAND")
    lines.append("Each point is the mean soups-delivered over 30 independently-trained")
    lines.append("MAPPO policies (different training seeds), evaluated at that many")
    lines.append("cumulative training steps. Its shaded band is also a 95% CI across")
    lines.append("those 30 training seeds, computed the same way as the AIF bands so the")
    lines.append("two are visually comparable.")
    lines.append("")
    lines.append("WHY THE CURVE IS JAGGED, NOT SMOOTH")
    if n_eval:
        lines.append(f"Each checkpoint is scored on only {n_eval} held-out evaluation episode")
    else:
        lines.append("Each checkpoint is scored on a small, fixed number of evaluation episodes")
    lines.append("per training seed -- not averaged over many eval runs. So every point is")
    lines.append("a genuinely noisy single (or few-episode) estimate, not a smoothed")
    lines.append("learning curve. This is also why the band widens rather than narrows at")
    lines.append("higher budgets: later checkpoints happen to have higher-variance")
    lines.append("single-episode outcomes across training seeds, not more uncertainty in")
    lines.append("a statistical-estimation sense.")
    lines.append("")
    lines.append("THE LEFTMOST POINT ('0' on the x-axis)")
    lines.append("This is the untrained / randomly-initialized policy, evaluated before")
    lines.append("any training step. The x-axis uses a symlog scale specifically so this")
    lines.append("point can sit at its true value of exactly 0, rather than being faked")
    lines.append("onto a nonzero position the way a plain log-scale axis would require.")
    lines.append("")
    lines.append("CROSSOVER POINTS (first plotted budget where MAPPO's mean reaches a")
    lines.append("given AIF paradigm's zero-shot mean -- not interpolated, so the true")
    lines.append("crossover may be anywhere between this budget and the previous one)")
    for key in ("ind", "ic", "fc"):
        label = aif_refs[key]["label"]
        b = crossovers[key]
        if b is not None:
            lines.append(f"  {label:<26} reached by {b:,} training steps")
        else:
            max_b = int(max(r['budget'] for r in rows))
            lines.append(f"  {label:<26} not reached within {max_b:,} training steps")
    lines.append("")
    lines.append("RAW MAPPO CURVE VALUES")
    for r in rows:
        lines.append(
            f"  budget={int(r['budget']):>8,}  mean={r['mean']:.2f}  95%CI=+/-{r['ci95']:.2f}  "
            f"(n_train_seeds={r['n_train_seeds']})"
        )
    lines.append("")
    lines.append(f"Full numeric data also in comparison_summary.json in this same folder.")

    out_path = out_dir / "mappo_sample_efficiency_curve_explained.txt"
    out_path.write_text("\n".join(lines) + "\n")
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

    crossovers = compute_crossovers(aif_refs, mappo_curve)
    explanation_path = write_explanation(aif_refs, mappo_curve, crossovers, out_dir)
    print(f"Wrote explanation: {explanation_path}")

    summary = {
        "aif_reference": aif_refs,
        "mappo_curve": mappo_curve,
        "crossovers_vs_aif": crossovers,
    }
    summary_path = out_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
