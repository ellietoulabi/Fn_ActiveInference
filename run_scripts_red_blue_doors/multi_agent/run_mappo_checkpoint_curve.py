"""
Sample-efficiency curve for MAPPO on MA Red-Blue-Button.

Standalone, new module -- does NOT edit
run_scripts_red_blue_doors/multi_agent/run_two_ppo_agents.py or any Active
Inference agent/generative-model file. Only *imports* (read-only) the
already-verified CTDE architecture and helpers from run_two_ppo_agents.py
(CentralizedCriticPPOTorchRLModule, RedBlueButtonPPOWrapper, OWN_OBS_DIM,
run_seed_experiment, build_eval_configs, _train_rng_seed) -- the model being
trained here is exactly the same architecture already used and verified by
that file, and evaluation reuses that file's own run_seed_experiment
unmodified, so a checkpoint's score here is computed by the exact same code
path as an official MAPPO run's final number.

Unlike the Overcooked version of this idea (run_scripts_overcooked/
run_mappo_checkpoint_curve.py), this environment has NO teleportation --
RedBlueButtonPPOWrapper wraps the real TwoAgentRedBlueButtonEnv and every
action is a real primitive step (up/down/left/right/press/noop), identical
to what the AIF paradigms and OPSRL play against. So there is no
environment-dynamics asymmetry to disclose here the way there is for
Overcooked -- this comparison is on genuinely equal footing.

Same motivation as the Overcooked version (see that module's docstring and
ai/04-writeup.md's Overcooked MAPPO "Open fairness questions"): rather than
picking one MAPPO training budget and reporting a single number, train once
per seed and evaluate the SAME live, in-training policy at a ladder of
step budgets, using the exact same 100-episode/20-per-config/max-steps
scored protocol the AIF paradigms and OPSRL are evaluated on -- producing a
full performance-vs-training-steps curve to plot against their zero-training
reference numbers.

Like run_two_ppo_agents.py itself, this module supports BOTH of that file's
established training protocols via --mode, applied at every budget in the
ladder rather than just at one final budget:
  - "pretrained" (default): training maps are domain-randomized every
    episode (config_mode="domain_random") -- the policy generalizes across
    many layouts rather than specializing to the one it'll be scored on.
  - "online": training maps follow the exact same schedule evaluation uses
    for this seed (config_mode="matched_schedule", via build_eval_configs) --
    a budget-matched approximation of learning from only the AIF-comparable
    experience, not a much larger, hidden, out-of-distribution budget.
Run this script twice (once per --mode) to get both curves; see the
sibling cluster launcher's MODE env var for the matching cluster pattern.

Usage (from repo root, with the venv active):
    python -u run_scripts_red_blue_doors/multi_agent/run_mappo_checkpoint_curve.py \\
        --mode pretrained \\
        --train-seeds 0 1 2 \\
        --budgets 0 2000 4000 6000 10000 15000 20000 30000 40000 50000 75000 100000 \\
        --eval-episodes 100 --eval-episodes-per-config 20 --max-steps 50 \\
        --out-dir thesis_logs/02_ma_redbluebuttons/mappo_checkpoint_curve_pretrained

    python -u run_scripts_red_blue_doors/multi_agent/run_mappo_checkpoint_curve.py \\
        --mode online \\
        --train-seeds 0 1 2 \\
        --budgets 0 2000 4000 6000 10000 15000 20000 30000 40000 50000 75000 100000 \\
        --eval-episodes 100 --eval-episodes-per-config 20 --max-steps 50 \\
        --out-dir thesis_logs/02_ma_redbluebuttons/mappo_checkpoint_curve_online
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Read-only reuse of the already-verified CTDE architecture, env wrapper, and
# evaluation loop -- nothing in this import list is redefined or
# monkey-patched anywhere below.
from run_scripts_red_blue_doors.multi_agent.run_two_ppo_agents import (  # noqa: E402
    CentralizedCriticPPOTorchRLModule,
    OWN_OBS_DIM,
    RAY_AVAILABLE,
    RedBlueButtonPPOWrapper,
    _train_rng_seed,
    build_eval_configs,
    run_seed_experiment,
)

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec
except ImportError:
    ray = None
    PPOConfig = None
    RLModuleSpec = None


def _get_total_env_steps(result) -> int:
    """Same extraction logic as agents/PPO/MA_PPO/mappo_simple.py's own
    helper of the same name -- duplicated here (not imported) so this module
    has zero dependency on the Overcooked MAPPO file, keeping the two
    checkpoint-curve modules fully independent."""
    env_runners = result.get("env_runners", {}) or {}
    for key in (
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
        "num_env_steps_taken_lifetime",
    ):
        if key in env_runners:
            try:
                return int(env_runners[key])
            except Exception:
                pass
    for key in ("num_env_steps_sampled_lifetime", "num_env_steps_sampled", "timesteps_total"):
        if key in result:
            try:
                return int(result[key])
            except Exception:
                pass
    return 0


def build_training_config(*, seed: int, max_steps: int, mode: str, eval_episodes: int,
                           eval_episodes_per_config: int, lr: float, gamma: float,
                           gae_lambda: float, clip_eps: float, ent_coef: float,
                           vf_coef: float, hidden_dim: int, num_workers: int,
                           envs_per_worker: int):
    """Mirrors run_two_ppo_agents.py::train_ppo's PPOConfig construction
    exactly (same hyperparameters, same two --mode training-map strategies)
    -- duplicated rather than called directly because train_ppo trains to a
    fixed budget and stops/saves in one shot, with no hook for live
    intermediate evaluation.

    mode="pretrained": domain-randomized training maps (config_mode=
    "domain_random"), matching train_ppo's own default -- the policy
    generalizes across layouts.
    mode="online": training maps follow the exact schedule evaluation uses
    for this seed (config_mode="matched_schedule"), matching train_ppo's
    "online" mode -- a budget-matched approximation of learning only from
    AIF-comparable experience, not a hidden larger/out-of-distribution one.
    """
    env_config = {
        "width": 3,
        "height": 3,
        "agent1_start_pos": (0, 0),
        "agent2_start_pos": (2, 2),
        "max_steps": max_steps,
        "episodes_per_config": eval_episodes_per_config,
    }
    if mode == "pretrained":
        env_config["config_mode"] = "domain_random"
        env_config["config_rng_seed"] = _train_rng_seed(seed)
    elif mode == "online":
        env_config["config_mode"] = "matched_schedule"
        env_config["schedule_configs"] = build_eval_configs(seed, eval_episodes, eval_episodes_per_config)
    else:
        raise ValueError(f"Unknown mode: {mode!r} (expected 'pretrained' or 'online')")
    env_instance = RedBlueButtonPPOWrapper(env_config)
    train_batch_size = min(2000, 200 * max_steps)

    rl_module_spec = RLModuleSpec(
        module_class=CentralizedCriticPPOTorchRLModule,
        model_config={"own_obs_dim": OWN_OBS_DIM, "hidden_dim": hidden_dim},
    )

    config = (
        PPOConfig()
        .environment(env=RedBlueButtonPPOWrapper, env_config=env_config)
        .training(
            lr=lr,
            gamma=gamma,
            lambda_=gae_lambda,
            clip_param=clip_eps,
            entropy_coeff=ent_coef,
            vf_loss_coeff=vf_coef,
            train_batch_size=train_batch_size,
            minibatch_size=128,
            num_epochs=10,
            grad_clip=10.0,
            grad_clip_by="global_norm",
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=num_workers, num_envs_per_env_runner=envs_per_worker, num_cpus_per_env_runner=1)
        .multi_agent(
            policies={
                "agent_0": (None, env_instance.observation_space["agent_0"], env_instance.action_space["agent_0"], {}),
                "agent_1": (None, env_instance.observation_space["agent_1"], env_instance.action_space["agent_1"], {}),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
        )
        .rl_module(rl_module_spec=rl_module_spec)
        .debugging(seed=seed)
    )
    return config, train_batch_size


def _save_partial(out_dir: Path, train_seed: int, mode: str, results_by_budget: dict) -> None:
    out_path = out_dir / f"mappo_curve_trainseed{train_seed}.json"
    payload = {
        "train_seed": train_seed,
        "mode": mode,
        "budgets_completed": sorted(results_by_budget.keys()),
        "results_by_budget": {str(k): v for k, v in sorted(results_by_budget.items())},
    }
    out_path.write_text(json.dumps(payload, indent=2))


def run_training_seed(train_seed: int, budgets: list[int], args, out_dir: Path) -> dict:
    config, train_batch_size = build_training_config(
        seed=train_seed, max_steps=args.max_steps, mode=args.mode,
        eval_episodes=args.eval_episodes, eval_episodes_per_config=args.eval_episodes_per_config,
        lr=args.lr, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip_eps=args.clip_eps, ent_coef=args.ent_coef,
        vf_coef=args.vf_coef, hidden_dim=args.hidden_dim,
        num_workers=args.num_workers, envs_per_worker=args.envs_per_worker,
    )
    algo = config.build_algo()

    sorted_budgets = sorted(set(int(b) for b in budgets))
    results_by_budget: dict[int, dict] = {}

    def _eval_and_record(budget: int, actual_steps: int) -> None:
        t0 = time.time()
        episodes = run_seed_experiment(
            algo, seed=train_seed, num_episodes=args.eval_episodes,
            episodes_per_config=args.eval_episodes_per_config, max_steps=args.max_steps,
            stochastic=not args.deterministic_eval,
        )
        dt = time.time() - t0
        successes = [1.0 if e["success"] else 0.0 for e in episodes]
        rewards = [e["reward"] for e in episodes]
        steps = [e["steps"] for e in episodes]
        results_by_budget[budget] = {
            "actual_steps": actual_steps,
            "n_episodes": len(episodes),
            "success_rate": float(np.mean(successes)) * 100.0,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_steps": float(np.mean(steps)),
            "std_steps": float(np.std(steps)),
            "eval_wall_seconds": dt,
        }
        print(
            f"[train_seed={train_seed}] budget={budget} (actual_steps={actual_steps}) "
            f"success_rate={np.mean(successes) * 100.0:.1f}% mean_reward={np.mean(rewards):.3f} "
            f"mean_steps={np.mean(steps):.2f} (eval took {dt:.1f}s)"
        )
        _save_partial(out_dir, train_seed, args.mode, results_by_budget)

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
                print(f"[train_seed={train_seed}] iter={iteration} steps={total_steps} (batch={train_batch_size})")
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
        description="MAPPO sample-efficiency curve on MA Red-Blue-Button (new, standalone module)"
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--mode", type=str, choices=["pretrained", "online"], default="pretrained",
        help="Training map-config strategy, matching run_two_ppo_agents.py's own --mode exactly: "
             "'pretrained' trains on domain-randomized maps (generalizes); 'online' trains on the "
             "exact same map schedule evaluation uses for this seed (config_mode=matched_schedule, "
             "a budget-matched approximation of learning from only AIF-comparable experience). "
             "Run this script twice, once per mode, to get both curves.",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+",
        default=[0, 2000, 4000, 6000, 10000, 15000, 20000, 30000, 40000, 50000, 75000, 100000],
        help="Cumulative training-step budgets to evaluate at (0 = untrained network).",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes-per-config", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--out-dir", type=str, default="thesis_logs/02_ma_redbluebuttons/mappo_checkpoint_curve")

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--deterministic-eval", action="store_true", help="Greedy eval instead of seeded-stochastic")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)

    args = parser.parse_args()

    if not RAY_AVAILABLE:
        raise RuntimeError("Ray RLlib required for MAPPO (pip install ray[rllib] torch)")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MAPPO CHECKPOINT / SAMPLE-EFFICIENCY CURVE (MA Red-Blue-Button)")
    print(f"  mode: {args.mode}")
    print(f"  train seeds: {args.train_seeds}")
    print(f"  budgets: {args.budgets}")
    print(f"  eval protocol: {args.eval_episodes} episodes / {args.eval_episodes_per_config} per config / max_steps={args.max_steps}")
    print(f"  out dir: {out_dir}")
    print("=" * 72)

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=max(2, args.num_workers + 1))

    try:
        all_results = {}
        for ts in args.train_seeds:
            print(f"\n{'=' * 72}\nTraining seed {ts}\n{'=' * 72}")
            res = run_training_seed(ts, args.budgets, args, out_dir)
            all_results[ts] = res

        summary = {"mode": args.mode, "budgets": {}}
        sorted_budgets = sorted(set(int(b) for b in args.budgets))
        for b in sorted_budgets:
            per_seed_success = [all_results[ts][b]["success_rate"] for ts in args.train_seeds if b in all_results[ts]]
            per_seed_reward = [all_results[ts][b]["mean_reward"] for ts in args.train_seeds if b in all_results[ts]]
            per_seed_steps = [all_results[ts][b]["mean_steps"] for ts in args.train_seeds if b in all_results[ts]]
            if not per_seed_success:
                continue
            summary["budgets"][str(b)] = {
                "n_train_seeds": len(per_seed_success),
                "mean_success_rate_across_train_seeds": float(np.mean(per_seed_success)),
                "std_success_rate_across_train_seeds": float(np.std(per_seed_success)),
                "mean_reward_across_train_seeds": float(np.mean(per_seed_reward)),
                "std_reward_across_train_seeds": float(np.std(per_seed_reward)),
                "mean_steps_across_train_seeds": float(np.mean(per_seed_steps)),
                "per_train_seed_success_rate": per_seed_success,
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
