# Compute Canada jobs — RedBlueButton

SLURM scripts to run two-agent **RedBlueButton** experiments on Alliance (Compute Canada). All jobs clone the repo fresh into `$SLURM_TMPDIR`, build a venv, install dependencies (excluding `opencv-python`, which isn't needed here and fails to install on Alliance), then run one seed/episode-index per array task.

> These scripts live in `cc_scripts/redbluebutton/`. For Overcooked jobs, see `cc_scripts/overcooked/README.md`.

## Scripts

| Script | What it runs | Python runner |
|--------|---------------|----------------|
| `two_aif_independent.sh` | Two AIF agents, Independent paradigm (each plans alone) | `run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_independent.py` |
| `two_aif_individually_collective.sh` | Two AIF agents, IndividuallyCollective paradigm (joint model per agent) | `run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_individually_collective.py` |
| `two_aif_fully_collective.sh` | Two AIF agents, FullyCollective paradigm (centralized joint planner + follower) | `run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_fully_collective.py` |
| `three_plus_ppo.sh` | Compares three AIF pairings + a PPO baseline | `run_scripts_red_blue_doors/compare_agents/compare_three_pairings_plus_ppo.py` |
| `nine.sh` | Nine-agent comparison (Q-learning variants, etc.) | `run_scripts_red_blue_doors/compare_agents/compare_nine_agents.py` |
| `eight.sh` | Eight-agent comparison | `run_scripts_red_blue_doors/compare_agents/compare_eight_agents.py` |

## Before you submit

1. **Push to GitHub.** Jobs clone `https://github.com/ellietoulabi/Fn_ActiveInference.git` on the compute node, so any local changes must be pushed to the branch that gets cloned (default branch) before submitting.
2. **Account.** Scripts use `#SBATCH --account=def-jrwright`. Change this line if your allocation differs.
3. **Log destination.** Output CSV/JSON files are copied to `${HOME}/projects/def-jrwright/toulabin/logs/` after each task finishes. Edit `DEST_BASE` in the script if your home path differs.
4. **Submit from the repo:**

   ```bash
   cd ~/projects/def-jrwright/toulabin/Fn_ActiveInference   # or your clone on CC
   sbatch cc_scripts/redbluebutton/two_aif_independent.sh
   sbatch cc_scripts/redbluebutton/two_aif_individually_collective.sh
   sbatch cc_scripts/redbluebutton/two_aif_fully_collective.sh
   sbatch cc_scripts/redbluebutton/three_plus_ppo.sh
   sbatch cc_scripts/redbluebutton/nine.sh
   sbatch cc_scripts/redbluebutton/eight.sh
   ```

## What the `two_aif_*.sh` jobs do

1. Load `python/3.11.4` and `scipy-stack`
2. Clone the repo into `$SLURM_TMPDIR`, create a venv, install `requirements.txt` minus `opencv-python`
3. Run one array task (`SEED_IDX = $SLURM_ARRAY_TASK_ID`) with `--verbose --episode-progress --show-beliefs --show-policies`
4. Copy the resulting step CSV and `_stats.json` summary to `DEST_BASE`

Each is a SLURM array job (`--array=0-4` by default → 5 seeds). Override with `sbatch --array=0 ...` for a single seed, or edit `--episodes` / `--episodes-per-config` / `--max-steps` in the script directly (these aren't exposed as env-var overrides).

### Output files

- CSV: `two_aif_agents_<paradigm>_seeds{N}_ep{E}_<timestamp>.csv` — one row per primitive step (positions, button states, actions, rewards, beliefs, policies).
- Stats JSON: `two_aif_agents_<paradigm>_seeds{N}_ep{E}_<timestamp>_stats.json` — aggregate success rate, mean reward/steps, per-seed summary.
- SLURM's own stdout/stderr (containing the `--verbose` belief/policy printout) stays as `slurm-<jobid>_<taskid>.out` in the directory you ran `sbatch` from; it is **not** copied to `DEST_BASE`.

## Walltime notes

`two_aif_individually_collective.sh` is the slowest of the three paradigms (36×36 joint-action space, computed per agent per step) and defaults to 2000 episodes; raise `#SBATCH --time` if it doesn't fit the default `0-2:00` budget, or test with `sbatch --array=0` first and check elapsed time via `sacct`.

## Monitoring

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,JobName,State,Elapsed,MaxRSS
tail -f slurm-<jobid>_<taskid>.out    # in submit directory
```

## Local dry run (same command, no SLURM)

```bash
export PYTHONPATH=.
python -u run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_independent.py \
  --seed 0 --episodes 2 --episodes-per-config 2 --max-steps 5
```

Swap the runner path for individually-collective / fully-collective as in the table above.
