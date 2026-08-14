"""
MAPPO training-budget sweep, single seed (episode_seed=76, matching AIF's
real seed76 logs), against the teleporting environment.

Goal (per explicit request): start at 1500 steps (AIF's own real
per-episode budget) and increase, evaluating deterministically at 1500
steps each time (matching AIF's real protocol exactly), to find where
MAPPO's performance crosses IND / IC / FC's real logged seed-76 results:
  IND: team_return=100.0  (5 deliveries)
  IC:  team_return=120.0  (6 deliveries)
  FC:  team_return=240.0  (12 deliveries)

Each budget trains a FRESH policy from scratch (not continued from the
previous budget's checkpoint) so "trained for N steps" means exactly N,
not N plus whatever came before.

Per explicit follow-up request: also records the FULL training-time curve
for each budget (every "[iter ...] steps=... episodes_done=... return_mean=..."
line printed during that budget's training run), not just the final
deterministic-eval number, saved to its own per-budget CSV.

Saves results to thesis_logs/03_ma_overcooked/mappo_budget_sweep_seed76/:
  results.csv                          -- one row per budget, final eval number
  training_curve_budget_<N>.csv        -- one row per training iteration, for budget N
"""
import csv
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = "/Users/ellie/dev/source/Fn_ActiveInference"
os.chdir(PROJECT_ROOT)

STEP_SCHEDULE = [1500, 5000, 15000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000]
TARGETS = {"IND": 100.0, "IC": 120.0, "FC": 240.0}
OUT_DIR = "thesis_logs/03_ma_overcooked/mappo_budget_sweep_seed76"
OUT_CSV = f"{OUT_DIR}/results.csv"

ITER_RE = re.compile(
    r"\[iter (\d+)\] steps=(\d+) episodes_done=([\d.]+) return_mean=(\S+) "
    r"min=(\S+) max=(\S+) ep_len_mean=(\S+)"
)

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["train_steps_budget", "final_team_return", "final_deliveries",
                "matches_IND", "matches_IC", "matches_FC", "elapsed_sec"])

for budget in STEP_SCHEDULE:
    ckpt_dir = f"checkpoints/mappo_budget_sweep_seed76_{budget}"
    subprocess.run(["rm", "-rf", ckpt_dir])
    cmd = [
        sys.executable, "-u", "agents/PPO/MA_PPO/mappo_simple.py",
        "--layout", "cramped_room", "--horizon", "400", "--seed", "76",
        "--max-train-steps", str(budget), "--log-every", "1",
        "--num-workers", "4", "--envs-per-worker", "4",
        "--checkpoint-dir", ckpt_dir,
        "--run-episode", "--eval-steps", "1500", "--deterministic",
    ]
    print(f"=== budget={budget} steps: starting ===", flush=True)
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0
    text = out.stdout + out.stderr

    # Parse every training iteration line for this budget's full curve.
    curve_rows = []
    for m in ITER_RE.finditer(text):
        it, steps, episodes_done, return_mean, rmin, rmax, ep_len_mean = m.groups()
        curve_rows.append([it, steps, episodes_done, return_mean, rmin, rmax, ep_len_mean])
    curve_path = f"{OUT_DIR}/training_curve_budget_{budget}.csv"
    with open(curve_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "steps", "episodes_done", "return_mean", "return_min", "return_max", "ep_len_mean"])
        w.writerows(curve_rows)

    m_final = re.search(r"Episode totals: a0=([+-]?[\d.]+)", text)
    team_return = float(m_final.group(1)) if m_final else None
    deliveries = team_return / 20.0 if team_return is not None else None

    matches = {}
    for name, target in TARGETS.items():
        matches[name] = bool(team_return is not None and team_return >= target)

    print(
        f"budget={budget} elapsed={elapsed:.0f}s team_return={team_return} "
        f"deliveries={deliveries} matches={matches} "
        f"train_iters_logged={len(curve_rows)}",
        flush=True,
    )
    if curve_rows:
        last = curve_rows[-1]
        print(f"  last training iter: steps={last[1]} episodes_done={last[2]} return_mean={last[3]}", flush=True)
    if team_return is None:
        print("  (no 'Episode totals' line found -- dumping last 40 lines of output)")
        print("\n".join(text.splitlines()[-40:]))

    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([budget, team_return, deliveries, matches["IND"], matches["IC"], matches["FC"], f"{elapsed:.0f}"])

print("=== sweep complete ===")
