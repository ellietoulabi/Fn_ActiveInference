"""
Sample-efficiency curve for MAPPO on Overcooked cramped_room.

Standalone, new module -- does NOT edit agents/PPO/MA_PPO/mappo_simple.py or
run_scripts_overcooked/run_mappo_semantic_action_level_sweep.py, and does not
touch any Active Inference agent/generative-model file. It only *imports*
(read-only) the already-verified CTDE architecture and helpers from
mappo_simple.py (build_config, AIFObsOvercookedMAEnv, _select_action,
AGENT_IDS) -- the model being trained here is exactly the same
CentralizedCriticPPOTorchRLModule already used and verified by that file.

Why this exists (see ai/04-writeup.md's Overcooked MAPPO "Open fairness
questions" section, item 1, and the 2026-08-24 discussion that led here):
a single MAPPO training budget is not a fair single number to compare against
zero-training Active Inference -- too little budget and MAPPO looks broken,
too much and it can look artificially strong, and the documented Overcooked
MAPPO "sample-hungry cliff" (near-zero performance below ~200-250k steps, a
real jump above it) means the "right" budget is not obvious. Instead of
picking one budget, this script trains MAPPO once per training seed and
evaluates the *same live, in-memory policy* at a whole ladder of step budgets
along the way (no checkpoint save/reload needed -- evaluation happens
directly against the algo object mid-training), producing a full
performance-vs-training-steps curve that can be plotted against the three
AIF paradigms' already-computed, zero-training reference numbers.

Usage (from repo root, with the venv active):
    python -u run_scripts_overcooked/run_mappo_checkpoint_curve.py \\
        --train-seeds 0 1 \\
        --budgets 0 25000 50000 100000 150000 200000 225000 250000 275000 300000 350000 400000 \\
        --eval-episode-seeds 76 77 78 79 80 \\
        --eval-max-steps 1500 \\
        --out-dir thesis_logs/03_ma_overcooked/mappo_checkpoint_curve
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

# Read-only reuse of the already-verified CTDE architecture and env wrapper --
# nothing in this import list is redefined or monkey-patched anywhere below.
from agents.PPO.MA_PPO.mappo_simple import (  # noqa: E402
    AGENT_IDS,
    AIFObsOvercookedMAEnv,
    RAY_AVAILABLE,
    _get_total_env_steps,
    _select_action,
    _summarize_iter,
    build_config,
)

try:
    import ray
    import torch
except ImportError:
    ray = None
    torch = None

DELIVERY_REWARD = 20.0


def evaluate_live_algo(
    algo,
    *,
    episode_seeds: list[int],
    max_steps: int,
    layout: str,
    horizon: int,
    shared_policy: bool,
    stochastic: bool,
    agent0_seed: int | None = None,
    agent1_seed: int | None = None,
) -> list[dict]:
    """Run one episode per eval seed against the live (in-memory) policy.

    Deterministic given (episode_seed, per-agent generator seed) -- mirrors
    the reproducibility fix already established for MAPPO evaluation
    elsewhere in this project (ai/02-debug.md, MAPPO baseline section):
    a torch.Generator per agent, seeded from the episode seed, rather than
    drawing from PyTorch's unseeded global RNG.

    When agent0_seed/agent1_seed are given (the cluster/one-seed-per-task
    usage, matching the AIF paradigms' and the existing MAPPO comparator's
    exact seed convention -- see _default_seed_lists in
    run_mappo_semantic_action_level_sweep.py), those exact values seed the
    two agents' generators instead of the ep_seed-derived fallback used by
    the local multi-seed pilot mode. Only meaningful with a single-element
    episode_seeds list (one array task = one exact AIF-matched seed triple).
    """
    results = []
    for ep_seed in episode_seeds:
        env = AIFObsOvercookedMAEnv({"layout": layout, "horizon": horizon})
        obs, _infos = env.reset(seed=int(ep_seed))
        if agent0_seed is not None and agent1_seed is not None:
            explicit_seeds = {AGENT_IDS[0]: int(agent0_seed), AGENT_IDS[1]: int(agent1_seed)}
            agent_generators = {
                aid: torch.Generator().manual_seed(explicit_seeds[aid]) for aid in AGENT_IDS
            }
        else:
            agent_generators = {
                aid: torch.Generator().manual_seed(int(ep_seed) * 1000 + i)
                for i, aid in enumerate(AGENT_IDS)
            }
        total_reward = 0.0
        for step in range(1, max_steps + 1):
            actions = {}
            for aid in AGENT_IDS:
                pid = "shared" if shared_policy else aid
                module = algo.get_module(pid)
                act_int, _logits = _select_action(
                    module, obs[aid], stochastic=stochastic, generator=agent_generators[aid]
                )
                actions[aid] = act_int
            obs, rewards, terminated, truncated, _next_infos = env.step(actions)
            # Shared team reward: both agents receive the identical value each
            # step (same convention already used throughout this project --
            # see ai/02-debug.md, "combined return was ~2x team return").
            r = float(rewards[AGENT_IDS[0]])
            total_reward += r
            if terminated.get("__all__") or truncated.get("__all__"):
                break
        results.append(
            {
                "episode_seed": int(ep_seed),
                "return": total_reward,
                "deliveries": total_reward / DELIVERY_REWARD,
            }
        )
    return results


def _make_train_args(args, *, train_seed: int) -> argparse.Namespace:
    ns = argparse.Namespace()
    ns.layout = args.layout
    ns.horizon = args.horizon
    ns.seed = int(train_seed)
    ns.episode_seed = None  # domain-randomized training, matching --mode pretrained's philosophy
    ns.lr = args.lr
    ns.gamma = args.gamma
    ns.gae_lambda = args.gae_lambda
    ns.clip_eps = args.clip_eps
    ns.vf_coef = args.vf_coef
    ns.ent_coef = args.ent_coef
    ns.train_batch_size = args.train_batch_size
    ns.minibatch_size = args.minibatch_size
    ns.epochs = args.epochs
    ns.shared_policy = not args.separate_policies
    ns.num_workers = args.num_workers
    ns.envs_per_worker = args.envs_per_worker
    ns.gpus = args.gpus
    return ns


def _save_partial(out_dir: Path, train_seed: int, results_by_budget: dict) -> None:
    out_path = out_dir / f"mappo_curve_trainseed{train_seed}.json"
    payload = {
        "train_seed": train_seed,
        "budgets_completed": sorted(results_by_budget.keys()),
        "results_by_budget": {str(k): v for k, v in sorted(results_by_budget.items())},
    }
    out_path.write_text(json.dumps(payload, indent=2))


def run_training_seed(
    train_seed: int,
    budgets: list[int],
    args,
    eval_episode_seeds: list[int],
    out_dir: Path,
) -> dict:
    train_args = _make_train_args(args, train_seed=train_seed)
    cfg = build_config(train_args)
    algo = cfg.build_algo()
    shared_policy = not args.separate_policies

    sorted_budgets = sorted(set(int(b) for b in budgets))
    results_by_budget: dict[int, dict] = {}

    def _eval_and_record(budget: int, actual_steps: int) -> None:
        t0 = time.time()
        episodes = evaluate_live_algo(
            algo,
            episode_seeds=eval_episode_seeds,
            max_steps=args.eval_max_steps,
            layout=args.layout,
            horizon=args.horizon,
            shared_policy=shared_policy,
            stochastic=not args.deterministic_eval,
            agent0_seed=args.agent0_seed,
            agent1_seed=args.agent1_seed,
        )
        dt = time.time() - t0
        deliveries = [e["deliveries"] for e in episodes]
        returns = [e["return"] for e in episodes]
        results_by_budget[budget] = {
            "actual_steps": actual_steps,
            "episodes": episodes,
            "mean_deliveries": float(np.mean(deliveries)),
            "std_deliveries": float(np.std(deliveries)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "eval_wall_seconds": dt,
        }
        print(
            f"[train_seed={train_seed}] budget={budget} (actual_steps={actual_steps}) "
            f"mean_deliveries={np.mean(deliveries):.2f} +/- {np.std(deliveries):.2f} "
            f"(eval took {dt:.1f}s)"
        )
        _save_partial(out_dir, train_seed, results_by_budget)

    try:
        next_idx = 0
        if sorted_budgets and sorted_budgets[0] == 0:
            print(f"[train_seed={train_seed}] Evaluating budget=0 (untrained network)...")
            _eval_and_record(0, 0)
            next_idx = 1

        if next_idx >= len(sorted_budgets):
            return results_by_budget

        max_budget = sorted_budgets[-1]
        total_steps = 0
        iteration = 0
        while total_steps < max_budget:
            iteration += 1
            result = algo.train()
            total_steps = _get_total_env_steps(result)
            if iteration % max(1, args.log_every) == 0:
                summary = _summarize_iter(result)
                print(
                    f"[train_seed={train_seed}] iter={iteration} steps={total_steps} "
                    f"return_mean={summary['return_mean']} ep_len_mean={summary['ep_len_mean']}"
                )
            while next_idx < len(sorted_budgets) and total_steps >= sorted_budgets[next_idx]:
                b = sorted_budgets[next_idx]
                print(f"[train_seed={train_seed}] Reached budget {b} at actual_steps={total_steps}; evaluating...")
                _eval_and_record(b, total_steps)
                next_idx += 1
            if next_idx >= len(sorted_budgets):
                break
    finally:
        algo.stop()

    return results_by_budget


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MAPPO sample-efficiency curve on Overcooked cramped_room (new, standalone module)"
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[0, 25000, 50000, 100000, 150000, 200000, 225000, 250000, 275000, 300000, 350000, 400000],
        help="Cumulative training-step budgets to evaluate at (0 = untrained network).",
    )
    parser.add_argument("--eval-episode-seeds", type=int, nargs="+", default=[76, 77, 78, 79, 80])
    parser.add_argument("--eval-max-steps", type=int, default=1500)
    parser.add_argument(
        "--agent0-seed", type=int, default=None,
        help="Exact agent_0 RNG seed (matches the AIF paradigms' agent0_seed convention, "
             "e.g. 1000+SEED_IDX). Only meaningful with a single --eval-episode-seeds value "
             "(one array task = one exact AIF-matched seed triple). Omit for the local "
             "multi-seed pilot mode, where each agent's generator is derived from its own "
             "episode seed instead.",
    )
    parser.add_argument(
        "--agent1-seed", type=int, default=None,
        help="Exact agent_1 RNG seed (matches the AIF paradigms' agent1_seed convention, "
             "e.g. 2000+SEED_IDX). See --agent0-seed.",
    )
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--horizon", type=int, default=1510)
    parser.add_argument("--out-dir", type=str, default="thesis_logs/03_ma_overcooked/mappo_checkpoint_curve")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--ent-coef", type=float, default=0.05)
    parser.add_argument("--train-batch-size", type=int, default=4000)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--separate-policies", action="store_true")
    parser.add_argument("--deterministic-eval", action="store_true", help="Greedy eval instead of seeded-stochastic")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)

    args = parser.parse_args()

    if not RAY_AVAILABLE:
        raise RuntimeError("Ray RLlib required for MAPPO (pip install ray[rllib] torch)")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MAPPO CHECKPOINT / SAMPLE-EFFICIENCY CURVE (Overcooked cramped_room)")
    print(f"  train seeds: {args.train_seeds}")
    print(f"  budgets: {args.budgets}")
    print(f"  eval episode seeds: {args.eval_episode_seeds} x {args.eval_max_steps} steps each")
    print(f"  out dir: {out_dir}")
    print("=" * 72)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=max(2, args.num_workers + 1))

    try:
        all_results = {}
        for ts in args.train_seeds:
            print(f"\n{'=' * 72}\nTraining seed {ts}\n{'=' * 72}")
            res = run_training_seed(ts, args.budgets, args, args.eval_episode_seeds, out_dir)
            all_results[ts] = res

        # Aggregate across training seeds per budget.
        summary = {"budgets": {}}
        sorted_budgets = sorted(set(int(b) for b in args.budgets))
        for b in sorted_budgets:
            per_seed_means = [all_results[ts][b]["mean_deliveries"] for ts in args.train_seeds if b in all_results[ts]]
            per_seed_returns = [all_results[ts][b]["mean_return"] for ts in args.train_seeds if b in all_results[ts]]
            if not per_seed_means:
                continue
            summary["budgets"][str(b)] = {
                "n_train_seeds": len(per_seed_means),
                "mean_deliveries_across_train_seeds": float(np.mean(per_seed_means)),
                "std_deliveries_across_train_seeds": float(np.std(per_seed_means)),
                "mean_return_across_train_seeds": float(np.mean(per_seed_returns)),
                "std_return_across_train_seeds": float(np.std(per_seed_returns)),
                "per_train_seed_mean_deliveries": per_seed_means,
            }
        summary_path = out_dir / "mappo_curve_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote summary: {summary_path}")
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
