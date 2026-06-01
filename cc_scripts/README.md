# Compute Canada jobs — semantic action-level (ind / ic / fc)

SLURM scripts to run **Independent**, **IndividuallyCollective**, and **FullyCollective** Overcooked experiments on Alliance (Compute Canada) with per-step **CSV** (Excel) and **JSONL** (full beliefs / maps / q_pi). Verbose stdout (`--log-steps`) is off on the cluster for speed.

## Scripts (use these)

| Script | Paradigm | Python runner |
|--------|----------|----------------|
| `ind_semantic_action_level.sh` | Independent — two agents, each plans alone | `run_scripts_overcooked/run_independent_semantic_action_level_sweep.py` |
| `ic_semantic_action_level.sh` | Individually collective — two agents, each plans over joint pairs | `run_scripts_overcooked/run_individually_collective_policy_semantic_action_level_seed_sweep.py` |
| `fc_semantic_action_level.sh` | Fully collective — one brain, puppet partner | `run_scripts_overcooked/run_fully_collective_semantic_action_level.py` |

Older jobs in this folder (`two_aif_*.sh`, `eight.sh`, etc.) target legacy `run_scripts/multi_agent/` runners, not the semantic-action-level stack above.

## Default run (what you get out of the box)

Each script is a **SLURM array** with **10 tasks** (`--array=0-9`). Each task runs:

- **1 episode** (`--n-runs 1`)
- **1500 primitive steps** per episode (`MAX_STEPS=1500`, `--log-csv --log-jsonl`)
- Layout: `cramped_room`
- `gamma=4.0`, `alpha=8.0`, stochastic policy selection

### Seeds per array index `i` (0 … 9)

| Index `i` | Episode seed | Agent 0 / brain seed | Agent 1 seed (ind & ic only) |
|-----------|--------------|----------------------|------------------------------|
| 0 | 76 | 1000 | 2000 |
| 1 | 77 | 1001 | 2001 |
| … | … | … | … |
| 9 | 85 | 1009 | 2009 |

Formula: `episode = 76 + i`, `agent0/brain = 1000 + i`, `agent1 = 2000 + i`.

### Walltime (adjust if your partition has a lower cap)

| Script | `#SBATCH --time` | Notes |
|--------|------------------|--------|
| ind | `0-6:00` | 1500 steps + JSONL; usually sufficient |
| fc | `3-00:00` | 1500 steps + JSONL |
| ic | `3-00:00` | Slowest; 400 joint policies × 2 agents per step |

If jobs are killed for time, increase `--time` in the script or reduce `MAX_STEPS` (see below).

## Before you submit

1. **Push to GitHub.** Jobs clone `https://github.com/ellietoulabi/Fn_ActiveInference.git` on the compute node. They install **`cc_scripts/requirements-cc-sal.txt`** (not the repo-root `requirements.txt`). Push this file and the updated `*_semantic_action_level.sh` scripts before submitting.

2. **Account.** Scripts use `#SBATCH --account=def-jrwright`. Change this line if your allocation differs.

3. **Log destination.** A short per-seed `.log` (sweep summary) and step CSVs are copied to:

   - Independent: `/home/toulabin/projects/def-jrwright/toulabin/logs/sal_ind/`
   - IC: `.../logs/sal_ic/`
   - FC: `.../logs/sal_fc/`

   Edit `DEST_BASE` in each script if your home path differs.

4. **Submit from the repo** (paths are relative to the cloned project root after install):

   ```bash
   cd ~/projects/def-jrwright/toulabin/Fn_ActiveInference   # or your clone on CC
   sbatch cc_scripts/ind_semantic_action_level.sh
   sbatch cc_scripts/ic_semantic_action_level.sh
   sbatch cc_scripts/fc_semantic_action_level.sh
   ```

## What each job does

1. Load `python/3.11.4` and `scipy-stack`
2. Clone the repo into `$SLURM_TMPDIR`, create a venv, `pip install -r cc_scripts/requirements-cc-sal.txt`
3. Set `PYTHONPATH` to the repo root and `environments/overcooked_ai/src`
4. Run one array task with `--log-csv` and `--log-jsonl` (one row / one JSON object per primitive step)
5. Save a small stdout `.log` (banner, totals, errors) and step CSV + JSONL under `DEST_BASE`
6. SLURM also writes `*_sal_%A_%a.out` in the directory you submitted from

### Stdout log files

- `ind_sal_ep76_a0_1000_a1_2000.log`
- `ic_sal_ep76_a0_1000_a1_2000.log`
- `fc_sal_ep76_brain1000.log`

### Step CSV files (open in Excel)

Written by `run_scripts_overcooked/sal_step_csv_log.py` via `--log-csv`. One row per env primitive step; columns include policy indices, semantic destination/mode, executed primitives, rewards, and (where applicable) policy-posterior entropy.

- Independent: `sal_ind_ep76_a0_1000_a1_2000_<timestamp>.csv`
- IC: `sal_ic_ep76_a0_1000_a1_2000_<timestamp>.csv`
- FC: `sal_fc_ep76_brain1000_<timestamp>.csv`

On the cluster, CSVs and JSONL are built under `$SLURM_TMPDIR/logs_sal/` then copied next to the `.log` files in `sal_ind` / `sal_ic` / `sal_fc`.

### Step JSONL files (full beliefs + maps)

Written by `run_scripts_overcooked/sal_step_detail_log.py` via `--log-jsonl`. One JSON object per step: `map_before` / `map_after`, `state_beliefs`, full `q_pi`, selected policy, rewards.

- `sal_ic_ep76_a0_1000_a1_2000_<timestamp>.jsonl` (and analogous `sal_ind_*`, `sal_fc_*`)

### Optional verbose stdout (usually off on CC)

Add `--log-steps` to the `python` line for human-readable maps and belief bars in the `.log` (much slower on IC).

Locally:

```bash
export PYTHONPATH=.:environments/overcooked_ai/src
python -u run_scripts_overcooked/run_independent_semantic_action_level_sweep.py \
  --n-runs 1 --episode-seeds 76 --agent0-seeds 1000 --agent1-seeds 2000 \
  --max-steps 10 --log-csv
# default CSV dir: <repo>/logs/
```

## Overrides at submit time

Environment variables are read before the Python command (defaults in parentheses):

| Variable | Default | Effect |
|----------|---------|--------|
| `MAX_STEPS` | `2000` | Episode length (primitive env steps) |
| `GAMMA` | `4.0` | Policy precision in EFE |
| `ALPHA` | `8.0` | Stochastic policy sampling sharpness |

Examples:

```bash
# Shorter smoke test (one seed index still runs one episode)
MAX_STEPS=50 sbatch cc_scripts/ind_semantic_action_level.sh

# More seeds (edit --array in the script, or override)
sbatch --array=0-9 cc_scripts/ind_semantic_action_level.sh   # 10 seeds → episodes 76–85

# Single seed for debugging
sbatch --array=2 cc_scripts/fc_semantic_action_level.sh      # only index 2 → ep 78, brain 1002
```

To change `gamma` / `alpha` for a whole batch:

```bash
GAMMA=8.0 ALPHA=16.0 sbatch cc_scripts/ic_semantic_action_level.sh
```

Disabling epistemic value (faster, less exploration in policy eval) requires adding `--noig` to the `python` line in the script; it is not exposed as an env var today.

## Monitoring

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
tail -f ind_sal_<JOBID>_<TASKID>.out    # in submit directory
```

After completion, inspect the copied `.log`, `.csv`, and `.jsonl` files under `logs/sal_ind`, `sal_ic`, or `sal_fc`.

## Local dry run (same command, no SLURM)

From the **repository root**:

```bash
export PYTHONPATH=.:environments/overcooked_ai/src

python -u run_scripts_overcooked/run_independent_semantic_action_level_sweep.py \
  --n-runs 1 --episode-seeds 76 --agent0-seeds 1000 --agent1-seeds 2000 \
  --max-steps 5 --log-csv
```

Swap the runner path for ic/fc as in the table above.

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| `ModuleNotFoundError: No module named 'numpy'` or `'scipy'` | After `PYTHONPATH` is set, scipy-stack packages may not be visible in the venv. Pull latest scripts: they run `pip install --ignore-installed numpy scipy` into the venv and verify `numpy`, `scipy.sparse`, and `gymnasium` **with** `PYTHONPATH` before the sweep starts. |
| `opencv-python` / dummy wheel error during `pip install` | Scripts must use `requirements-cc-sal.txt`, not `requirements.txt`. OpenCV is not needed for these runs (terminal logs only). |
| `Dependencies installed` then `exit=1` with empty log | Old scripts continued after a failed `pip install`; pull latest scripts (they `exit 1` if pip fails). |
| `ModuleNotFoundError: utils` | `PYTHONPATH` must include repo root (scripts set this on CC) |
| `unrecognized arguments: --max-steps` | Push/pull latest repo; ind/ic sweeps need the `--max-steps` flag |
| Job `TIMEOUT` | Raise `#SBATCH --time` or lower `MAX_STEPS`; IC is the usual culprit |
| Empty `DEST_BASE` | Run failed before copy; read `*_sal_%A_%a.out` on the login node |
| `exit=1`, no CSV, little in SLURM `.out` | Python traceback is in the copied `.log` under `sal_ic/` (stdout was redirected). Re-submit after pulling latest scripts: they print the last 100 log lines on failure and run a preflight import check. If you add `--log-steps` manually, `PYTHONIOENCODING=utf-8` in `_sal_common.sh` avoids Unicode map-bar crashes on ASCII locales. |
| Stale code on cluster | Jobs always clone `main`; merge and push before `sbatch` |
