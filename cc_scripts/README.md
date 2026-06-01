# Compute Canada jobs — semantic action-level (ind / ic / fc)

SLURM scripts to run **Independent**, **IndividuallyCollective**, and **FullyCollective** Overcooked experiments on Alliance (Compute Canada) with full per-step logs.

## Scripts (use these)

| Script | Paradigm | Python runner |
|--------|----------|----------------|
| `ind_semantic_action_level.sh` | Independent — two agents, each plans alone | `run_scripts_overcooked/run_independent_semantic_action_level_sweep.py` |
| `ic_semantic_action_level.sh` | Individually collective — two agents, each plans over joint pairs | `run_scripts_overcooked/run_individually_collective_policy_semantic_action_level_seed_sweep.py` |
| `fc_semantic_action_level.sh` | Fully collective — one brain, puppet partner | `run_scripts_overcooked/run_fully_collective_semantic_action_level.py` |

Older jobs in this folder (`two_aif_*.sh`, `eight.sh`, etc.) target legacy `run_scripts/multi_agent/` runners, not the semantic-action-level stack above.

## Default run (what you get out of the box)

Each script is a **SLURM array** with **5 tasks** (`--array=0-4`). Each task runs:

- **1 episode** (`--n-runs 1`)
- **2000 primitive steps** per episode (`MAX_STEPS=2000`, `--log-steps`)
- Layout: `cramped_room`
- `gamma=4.0`, `alpha=8.0`, stochastic policy selection

### Seeds per array index `i` (0 … 4)

| Index `i` | Episode seed | Agent 0 / brain seed | Agent 1 seed (ind & ic only) |
|-----------|--------------|----------------------|------------------------------|
| 0 | 76 | 1000 | 2000 |
| 1 | 77 | 1001 | 2001 |
| 2 | 78 | 1002 | 2002 |
| 3 | 79 | 1003 | 2003 |
| 4 | 80 | 1004 | 2004 |

Formula: `episode = 76 + i`, `agent0/brain = 1000 + i`, `agent1 = 2000 + i`.

### Walltime (adjust if your partition has a lower cap)

| Script | `#SBATCH --time` | Notes |
|--------|------------------|--------|
| ind | `0-3:00` | Usually finishes well under this for 2000 steps |
| fc | `2-00:00` | ~hours per 2000-step episode |
| ic | `2-00:00` | Slowest; 2000 steps can take many hours (400 joint policies × 2 agents per step) |

If jobs are killed for time, increase `--time` in the script or reduce `MAX_STEPS` (see below).

## Before you submit

1. **Push to GitHub.** Jobs clone `https://github.com/ellietoulabi/Fn_ActiveInference.git` on the compute node. The runners must include `--max-steps` on the ind/ic sweep scripts (already in repo if you merged those changes).

2. **Account.** Scripts use `#SBATCH --account=def-jrwright`. Change this line if your allocation differs.

3. **Log destination.** Full per-step logs are copied to:

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
2. Clone the repo into `$SLURM_TMPDIR`, create a venv, `pip install -r requirements.txt`
3. Set `PYTHONPATH` to the repo root and `environments/overcooked_ai/src`
4. Run one array task with `--log-steps` (verbose: map, observations, beliefs, policy posteriors, executed primitives, rewards)
5. Save stdout/stderr to a per-seed `.log` under `DEST_BASE`
6. SLURM also writes `*_sal_%A_%a.out` in the directory you submitted from

Log file names look like:

- `ind_sal_ep76_a0_1000_a1_2000.log`
- `ic_sal_ep76_a0_1000_a1_2000.log`
- `fc_sal_ep76_brain1000.log`

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

After completion, inspect the copied logs under `logs/sal_ind`, `sal_ic`, or `sal_fc`.

## Local dry run (same command, no SLURM)

From the **repository root**:

```bash
export PYTHONPATH=.:environments/overcooked_ai/src

python -u run_scripts_overcooked/run_independent_semantic_action_level_sweep.py \
  --n-runs 1 --episode-seeds 76 --agent0-seeds 1000 --agent1-seeds 2000 \
  --max-steps 5 --log-steps
```

Swap the runner path for ic/fc as in the table above.

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| `ModuleNotFoundError: utils` | `PYTHONPATH` must include repo root (scripts set this on CC) |
| `unrecognized arguments: --max-steps` | Push/pull latest repo; ind/ic sweeps need the `--max-steps` flag |
| Job `TIMEOUT` | Raise `#SBATCH --time` or lower `MAX_STEPS`; IC is the usual culprit |
| Empty `DEST_BASE` | Run failed before copy; read `*_sal_%A_%a.out` on the login node |
| Stale code on cluster | Jobs always clone `main`; merge and push before `sbatch` |
