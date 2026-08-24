"""
MAPPO 30-seed sweep at the literal "same environment-interaction budget as
AIF" point (--max-train-steps 1500, which rounds up to 2000 actual steps due
to train_batch_size=1000 granularity) -- completes the same-budget comparison
across the full 30-seed pool, following up on the single-seed (76) spot
check that showed zero completed episodes.
"""
import csv, os, re, subprocess, sys, time

PROJECT_ROOT = "/Users/ellie/dev/source/Fn_ActiveInference"
os.chdir(PROJECT_ROOT)

SEEDS = [76 + i for i in range(30)]
BUDGET = 1500
OUT_DIR = "thesis_logs/03_ma_overcooked/mappo_30seed_200k_250k"
OUT_CSV = f"{OUT_DIR}/results_samebudget_1500.csv"

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "train_steps_budget", "final_team_return", "final_deliveries", "episodes_completed", "elapsed_sec"])

for seed in SEEDS:
    ckpt_dir = f"checkpoints/mappo_samebudget_seed{seed}"
    subprocess.run(["rm", "-rf", ckpt_dir])
    cmd = [
        sys.executable, "-u", "agents/PPO/MA_PPO/mappo_simple.py",
        "--layout", "cramped_room", "--horizon", "400", "--seed", str(seed),
        "--max-train-steps", str(BUDGET), "--log-every", "1",
        "--num-workers", "4", "--envs-per-worker", "4",
        "--checkpoint-dir", ckpt_dir,
        "--run-episode", "--eval-steps", "1500", "--deterministic",
    ]
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    elapsed = time.time() - t0
    text = out.stdout + out.stderr
    m_final = re.search(r"Episode totals: a0=([+-]?[\d.]+)", text)
    team_return = float(m_final.group(1)) if m_final else None
    deliveries = team_return / 20.0 if team_return is not None else None
    m_ep = re.findall(r"episodes_done=([\d.]+)", text)
    episodes_completed = m_ep[-1] if m_ep else "0"
    print(f"seed={seed} elapsed={elapsed:.0f}s team_return={team_return} deliveries={deliveries} episodes_completed={episodes_completed}", flush=True)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([seed, BUDGET, team_return, deliveries, episodes_completed, f"{elapsed:.0f}"])

print("=== 2000-step (literal same-budget) 30-seed sweep complete ===")
