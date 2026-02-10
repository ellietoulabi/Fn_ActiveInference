"""
Compare the three two-agent AIF pairings on RedBlueButton plus PPO under the same settings.

Runs Fully Collective, Individually Collective, Independent, and PPO paradigms with
identical seeds, episodes, episodes-per-config, and max-steps, then prints a
comparison table and optionally saves CSV/plots.

Fair comparison: all AIF scripts use policy_len=2, gamma=2.0, alpha=1.0, num_iter=16,
same env (TwoAgentRedBlueButton 3x3), and same config generation. PPO sees the same
state as the Fully Collective AIF agent (env_obs_to_model_obs flattened).
"""

import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime

# Project root for running scripts and resolving paths
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

MULTI_AGENT_DIR = project_root / "run_scripts_red_blue_doors" / "multi_agent"
SCRIPTS = {
    "Fully Collective": MULTI_AGENT_DIR / "run_two_aif_agents_fully_collective.py",
    "Individually Collective": MULTI_AGENT_DIR / "run_two_aif_agents_individually_collective.py",
    "Independent": MULTI_AGENT_DIR / "run_two_aif_agents_independent.py",
    "PPO": MULTI_AGENT_DIR / "run_two_ppo_agents.py",
}


def run_paradigm(
    script_path: Path,
    seeds: int,
    episodes: int,
    episodes_per_config: int,
    max_steps: int,
    stats_output: Path,
    quiet: bool = True,
    episode_progress: bool = False,
) -> dict:
    """Run one paradigm script and return parsed stats JSON."""
    cmd = [
        sys.executable,
        str(script_path),
        "--seeds",
        str(seeds),
        "--episodes",
        str(episodes),
        "--episodes-per-config",
        str(episodes_per_config),
        "--max-steps",
        str(max_steps),
        "--stats-output",
        str(stats_output),
    ]
    if episode_progress:
        cmd.append("--episode-progress")
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=quiet,
        text=True,
    )
    if result.returncode != 0 and quiet:
        print(result.stdout or "")
        print(result.stderr or "", file=sys.stderr)
        raise RuntimeError(f"Script failed: {script_path.name} (exit {result.returncode})")
    with open(stats_output) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare Fully Collective, Individually Collective, Independent, and PPO "
            "two-agent pairings on RedBlueButton."
        )
    )
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per seed (default: 200)")
    parser.add_argument(
        "--episodes-per-config",
        type=int,
        default=40,
        help="Episodes per map config (default: 40)",
    )
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per episode (default: 50)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for stats JSONs and comparison CSV "
            "(default: logs/compare_three_pairings_plus_ppo_<timestamp>)"
        ),
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only load existing stats from --output-dir and print comparison (skip running scripts)",
    )
    parser.add_argument("--verbose", action="store_true", help="Show script stdout when running")
    parser.add_argument(
        "--episode-progress",
        action="store_true",
        help="Pass --episode-progress to AIF paradigms and PPO (if supported)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "logs" / f"compare_three_pairings_plus_ppo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_files = {
        name: output_dir / f"stats_{name.lower().replace(' ', '_')}.json"
        for name in SCRIPTS
    }

    if not args.no_run:
        print("=" * 80)
        print("COMPARING THREE TWO-AGENT AIF PAIRINGS + PPO (same seeds, episodes, env settings)")
        print("=" * 80)
        print(
            f"  Seeds: {args.seeds}  Episodes per seed: {args.episodes}  "
            f"Episodes per config: {args.episodes_per_config}  Max steps: {args.max_steps}"
        )
        print()

        for name, script_path in SCRIPTS.items():
            if not script_path.exists():
                print(f"  ERROR: Script not found: {script_path}")
                sys.exit(1)
            print(f"  Running {name}...")
            run_paradigm(
                script_path,
                seeds=args.seeds,
                episodes=args.episodes,
                episodes_per_config=args.episodes_per_config,
                max_steps=args.max_steps,
                stats_output=stats_files[name],
                quiet=not args.verbose,
                episode_progress=args.episode_progress,
            )
        print("  Done.\n")

    # Load stats (from run or from existing files)
    all_stats = {}
    for name, path in stats_files.items():
        if not path.exists():
            print(f"  ERROR: Stats file not found: {path} (run without --no-run first)")
            sys.exit(1)
        with open(path) as f:
            all_stats[name] = json.load(f)

    # Comparison table
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    print(
        f"{'Paradigm':<25} {'Success %':>10} {'Mean reward':>12} "
        f"{'Mean steps':>12} {'Total episodes':>15}"
    )
    print("-" * 80)
    for name in SCRIPTS:
        s = all_stats[name]
        sr = s["success_rate"]
        mr = s["mean_reward"]
        ms = s["mean_steps"]
        te = s["total_episodes"]
        print(f"{name:<25} {sr:>9.1f}% {mr:>+11.2f} {ms:>11.1f} {te:>15}")
    print()

    # Per-seed table (if multiple seeds)
    n_seeds = all_stats["Fully Collective"].get("n_seeds", 1)
    if n_seeds > 1 and all(all_stats[n].get("seed_summaries") for n in SCRIPTS):
        print("Per-seed success rate (%):")
        print(f"  {'Seed':<8} " + "  ".join(f"{n[:12]:>12}" for n in SCRIPTS))
        seeds_list = [ss["seed"] for ss in all_stats["Fully Collective"]["seed_summaries"]]
        for seed in seeds_list:
            row = [f"  {seed:<6}"]
            for name in SCRIPTS:
                ss = next((s for s in all_stats[name]["seed_summaries"] if s["seed"] == seed), None)
                row.append(f"{ss['success_rate']:>11.1f}%" if ss else "   N/A")
            print(" ".join(row))
        print()

    # Save comparison CSV
    comparison_path = output_dir / "comparison.csv"
    with open(comparison_path, "w") as f:
        f.write(
            "paradigm,success_rate,mean_reward,std_reward,mean_steps,std_steps,"
            "total_episodes,total_successes\n"
        )
        for name in SCRIPTS:
            s = all_stats[name]
            f.write(
                f"{name},{s['success_rate']},{s['mean_reward']},{s['std_reward']},"
                f"{s['mean_steps']},{s['std_steps']},{s['total_episodes']},"
                f"{s['total_successes']}\n"
            )
    print(f"Comparison CSV: {comparison_path}")

    # Summary JSON
    summary_path = output_dir / "comparison_summary.json"
    summary = {
        "settings": {
            "seeds": args.seeds,
            "episodes_per_seed": args.episodes,
            "episodes_per_config": args.episodes_per_config,
            "max_steps": args.max_steps,
        },
        "paradigms": {name: all_stats[name] for name in SCRIPTS},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

