# Notes

## MA Red-Blue-Button — plotting commands

Each of the three paradigm run scripts has a built-in `--plots` flag (2x2 figure:
success-rate learning curve + other panels, saved as PNG to `results/`). Add it
to any run to generate figures alongside the console summary. Same
`--seed/--episodes/--episodes-per-config/--max-steps` convention across all three
(currently all on the `ActiveInferenceRedBlueButtonExact` package, deterministic
action selection).

```bash
# Independent
python3 run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_independent.py \
  --seed 0 --episodes 20 --episodes-per-config 5 --max-steps 15 --plots

# Fully Collective
python3 run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_fully_collective.py \
  --seed 0 --episodes 20 --episodes-per-config 5 --max-steps 15 --plots

# Individually Collective
python3 run_scripts_red_blue_doors/multi_agent/run_two_aif_agents_individually_collective.py \
  --seed 0 --episodes 20 --episodes-per-config 5 --max-steps 15 --plots
```

Output: `results/two_aif_agents_{independent,fully_collective,individually_collective}_seeds1_ep20.png`

Add `--print-steps` to see a live per-episode win/loss line in stdout (useful for
watching a run in progress rather than waiting for the final summary), and
`--log-csv` to get a per-step CSV under `logs/` (needed if you want to tally
wins/episodes from a run that's still in progress, since `--plots` only writes
at the very end).

---

## Overcooked (Stage 3) — plotting commands

```bash
# Single-paradigm summary + plots
python3 utils/plotting/plot_sal_semantic_action_level.py <log_dir> -o <output_dir>

# Paired comparison between two paradigms on their common episode_seeds
python3 utils/plotting/plot_sal_pair_comparison.py \
  --a <log_dir_A> --b <log_dir_B> --label-a <A> --label-b <B> -o <output_dir>
```

Example (used this session):
```bash
python3 utils/plotting/plot_sal_semantic_action_level.py \
  thesis_logs/03_ma_overcooked/sal_fc_30seed_collisionfix \
  -o thesis_plots/03_ma_overcooked/sal_fc_30seed_collisionfix

python3 utils/plotting/plot_sal_pair_comparison.py \
  --a thesis_logs/03_ma_overcooked/sal_fc_30seed_collisionfix \
  --b thesis_logs/03_ma_overcooked/sal_ind_30seed \
  --label-a FC --label-b IND \
  -o thesis_plots/03_ma_overcooked/compare_fc_ind
```



python3 utils/plotting/plot_sal_triple_comparison.py \
  --base-dir thesis_logs/03_ma_overcooked \
  -o thesis_plots/03_ma_overcooked/compare_ind_ic_fc 2>&1 | tail -20





sbatch cc_scripts/redbluebutton/nine.sh
sbatch cc_scripts/redbluebutton/two_aif_fully_collective.sh
sbatch cc_scripts/redbluebutton/two_aif_independent.sh
sbatch cc_scripts/redbluebutton/two_aif_individually_collective.sh


# default (50 steps) — just submit normally
sbatch cc_scripts/redbluebutton/nine.sh
sbatch cc_scripts/redbluebutton/two_aif_fully_collective.sh
sbatch cc_scripts/redbluebutton/two_aif_independent.sh
sbatch cc_scripts/redbluebutton/two_aif_individually_collective.sh
sbatch cc_scripts/redbluebutton/mappo.sh

# 30-step variant — set MAX_STEPS before sbatch
MAX_STEPS=15 sbatch cc_scripts/redbluebutton/nine.sh
MAX_STEPS=15 sbatch cc_scripts/redbluebutton/two_aif_fully_collective.sh
MAX_STEPS=15 sbatch cc_scripts/redbluebutton/two_aif_independent.sh
MAX_STEPS=15 sbatch cc_scripts/redbluebutton/two_aif_individually_collective.sh
MAX_STEPS=15 sbatch cc_scripts/redbluebutton/mappo.sh

# any other value works the same way, e.g. 100 steps:
MAX_STEPS=100 sbatch cc_scripts/redbluebutton/nine.sh





# max-steps = 50
MAX_STEPS=50 sbatch cc_scripts/redbluebutton/two_aif_fully_collective.sh
MAX_STEPS=50 sbatch cc_scripts/redbluebutton/two_aif_independent.sh
MAX_STEPS=50 sbatch cc_scripts/redbluebutton/two_aif_individually_collective.sh

# max-steps = 30
MAX_STEPS=30 sbatch cc_scripts/redbluebutton/two_aif_fully_collective.sh
MAX_STEPS=30 sbatch cc_scripts/redbluebutton/two_aif_independent.sh
MAX_STEPS=30 sbatch cc_scripts/redbluebutton/two_aif_individually_collective.sh






Budget=1500: 0 deliveries (consistent with before). Training curve shows only 2 iterations, 0 episodes completed during training itself — confirms this budget is too small to even finish one training episode, let alone learn. Continuing to watch.

Budget=5000: 0 deliveries, 5 training iterations, still 0 episodes completed during training (400-step horizon means 5000 steps across 16 parallel envs still isn't quite enough for any to finish). Continuing.

Budget=15000: still 0 deliveries at eval, but training curve shows 32 episodes completed by the end (first budget where episodes actually finish). Continuing.

Budget=50000: 0 deliveries at eval, 112 episodes completed during training. Continuing.

Budget=100000: still 0 deliveries, 240 episodes completed during training. Continuing.

Budget=150000: still 0 deliveries at eval, 368 episodes completed. This matches the earlier coarse sweep's result exactly. Now moving into the finer-grained range (200k-400k) where the crossovers should actually be.



MAX_TRAIN_STEPS=200000 sbatch cc_scripts/overcooked/mappo_semantic_action_level.sh
MAX_TRAIN_STEPS=250000 sbatch cc_scripts/overcooked/mappo_semantic_action_level.sh




# --- Setup 1: 400 episodes / relocate every 50 / max_steps 30 ---
EPISODES=400 EPISODES_PER_CONFIG=50 MAX_STEPS=30 sbatch cc_scripts/redbluebutton/nine.sh
EPISODES=400 EPISODES_PER_CONFIG=50 MAX_STEPS=30 sbatch cc_scripts/redbluebutton/nine_pretrained.sh

# --- Setup 2: 400 episodes / relocate every 50 / max_steps 50 ---
EPISODES=400 EPISODES_PER_CONFIG=50 MAX_STEPS=50 sbatch cc_scripts/redbluebutton/nine.sh
EPISODES=400 EPISODES_PER_CONFIG=50 MAX_STEPS=50 sbatch cc_scripts/redbluebutton/nine_pretrained.sh

# --- Setup 3: 200 episodes / relocate every 25 / max_steps 30 ---
EPISODES=200 EPISODES_PER_CONFIG=25 MAX_STEPS=30 sbatch cc_scripts/redbluebutton/nine.sh
EPISODES=200 EPISODES_PER_CONFIG=25 MAX_STEPS=30 sbatch cc_scripts/redbluebutton/nine_pretrained.sh

# --- Setup 4: 200 episodes / relocate every 25 / max_steps 50 ---
EPISODES=200 EPISODES_PER_CONFIG=25 MAX_STEPS=50 sbatch cc_scripts/redbluebutton/nine.sh
EPISODES=200 EPISODES_PER_CONFIG=25 MAX_STEPS=50 sbatch cc_scripts/redbluebutton/nine_pretrained.sh
