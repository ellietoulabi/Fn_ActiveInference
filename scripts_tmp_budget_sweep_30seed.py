"""
MAPPO 30-seed sweep at exactly 200,000 and 250,000 training steps, matching
the real AIF (IND/IC/FC) seed protocol: episode_seed = 76+i for i in 0..29
(cc_scripts/overcooked/*.sh's own convention).

Goal (per explicit request): show the "sample hungry" cliff -- 200k
(pre-competence) vs 250k (post-competence, per the seed-76 spot check) --
across all 30 seeds, not just one, so it can be plotted against the real
IND/IC/FC 30-seed results in thesis_logs/03_ma_overcooked/.

mappo_simple.py exposes a single --seed flag (unlike AIF's separate
episode_seed/a0_seed/a1_seed) -- it drives both training RNG and the
evaluation env's reset seed. Using --seed = episode_seed (76..105) is the
closest analogous mapping given that CLI design; documented here rather
than left implicit.

Each (seed, budget) pair trains a FRESH policy from scratch, then evaluates
deterministically for 1500 steps (matching AIF's real per-seed budget).

Saves to thesis_logs/03_ma_overcooked/mappo_30seed_200k_250k/:
  results.csv                                  -- one row per (seed, budget)
  training_curve_seed<S>_budget<N>.csv         -- full training curve per run
"""
import csv
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = "/Users/ellie/dev/source/Fn_ActiveInference"
os.chdir(PROJECT_ROOT)

SEEDS = [76 + i for i in range(30)]
BUDGETS = [200000, 250000]
OUT_DIR = "thesis_logs/03_ma_overcooked/mappo_30seed_200k_250k"
OUT_CSV = f"{OUT_DIR}/results.csv"

ITER_RE = re.compile(
    r"\[iter (\d+)\] steps=(\d+) episodes_done=([\d.]+) return_mean=(\S+) "
    r"min=(\S+) max=(\S+) ep_len_mean=(\S+)"
)

os.makedirs(OUT_DIR, exist_ok=True)
if not os.path.exists(OUT_CSV):
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "train_steps_budget", "final_team_return", "final_deliveries", "elapsed_sec"])

# Resume-safe: skip (seed, budget) pairs already recorded, in case this needs restarting.
done = set()
if os.path.exists(OUT_CSV):
    with open(OUT_CSV) as f:
        for row in csv.DictReader(f):
            done.add((int(row["seed"]), int(row["train_steps_budget"])))

for seed in SEEDS:
    for budget in BUDGETS:
        if (seed, budget) in done:
            print(f"=== seed={seed} budget={budget}: already done, skipping ===", flush=True)
            continue
        ckpt_dir = f"checkpoints/mappo_30seed_sweep_seed{seed}_budget{budget}"
        subprocess.run(["rm", "-rf", ckpt_dir])
        cmd = [
            sys.executable, "-u", "agents/PPO/MA_PPO/mappo_simple.py",
            "--layout", "cramped_room", "--horizon", "400", "--seed", str(seed),
            "--max-train-steps", str(budget), "--log-every", "1",
            "--num-workers", "4", "--envs-per-worker", "4",
            "--checkpoint-dir", ckpt_dir,
            "--run-episode", "--eval-steps", "1500", "--deterministic",
        ]
        print(f"=== seed={seed} budget={budget}: starting ===", flush=True)
        t0 = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - t0
        text = out.stdout + out.stderr

        curve_rows = [m.groups() for m in ITER_RE.finditer(text)]
        curve_path = f"{OUT_DIR}/training_curve_seed{seed}_budget{budget}.csv"
        with open(curve_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "steps", "episodes_done", "return_mean", "return_min", "return_max", "ep_len_mean"])
            w.writerows(curve_rows)

        m_final = re.search(r"Episode totals: a0=([+-]?[\d.]+)", text)
        team_return = float(m_final.group(1)) if m_final else None
        deliveries = team_return / 20.0 if team_return is not None else None

        print(
            f"seed={seed} budget={budget} elapsed={elapsed:.0f}s "
            f"team_return={team_return} deliveries={deliveries}",
            flush=True,
        )
        if team_return is None:
            print("  (no 'Episode totals' line found -- dumping last 30 lines)")
            print("\n".join(text.splitlines()[-30:]))

        with open(OUT_CSV, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([seed, budget, team_return, deliveries, f"{elapsed:.0f}"])

print("=== 30-seed sweep complete ===")
