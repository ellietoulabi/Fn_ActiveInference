# Compute Canada (Alliance) SLURM scripts

SLURM job scripts for running experiments on the Compute Canada / Alliance cluster, organized by environment:

| Folder | Environment | Details |
|--------|-------------|---------|
| [`redbluebutton/`](redbluebutton/README.md) | Two-agent RedBlueButton (AIF paradigms, PPO/Q-learning comparisons) | `redbluebutton/README.md` |
| [`overcooked/`](overcooked/README.md) | Overcooked semantic-action-level (Independent / IndividuallyCollective / FullyCollective) | `overcooked/README.md` |

All jobs follow the same basic pattern: clone `https://github.com/ellietoulabi/Fn_ActiveInference.git` fresh into `$SLURM_TMPDIR`, build a venv, install dependencies, run one seed/config per SLURM array task, and copy the resulting logs/CSVs/JSON to a durable path under `$HOME`.

Push any local changes to the cloned branch before submitting — jobs always pull from GitHub, never from your local working copy.
