"""
Train (optional) and evaluate MAPPO on Overcooked cramped_room with SAL-style CSV logs.

Matches AIF seed sweep: episode_seed 76–85, agent0 1000–1009, agent1 2000–2009.
Trains MAPPO for 1M env steps per seed (separate checkpoint), then evaluates
1500 primitive steps per seed. Logs under logs/sal_mappo/.

Usage (from repo root, with .venv active):
    export PYTHONPATH=.:environments/overcooked_ai/src
    .venv/bin/python3 run_scripts_overcooked/run_mappo_semantic_action_level_sweep.py \\
        --max-steps 1500 --n-runs 10 --log-csv --plot

    # Eval only (checkpoint already trained for that episode_seed):
    .venv/bin/python3 run_scripts_overcooked/run_mappo_semantic_action_level_sweep.py \\
        --no-train --max-steps 1500 --log-csv --plot
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
overcooked_src = PROJECT_ROOT / "environments" / "overcooked_ai" / "src"
if overcooked_src.exists():
    sys.path.insert(0, str(overcooked_src))

import sal_step_csv_log as sal_csv
import run_independent_semantic_action_level as ind
from generative_models.MA_ActiveInference_Monotonic.Overcooked.cramped_room.IndependentWithSemanticPoliciesActionLevel import (
    model_init as ind_model_init,
)

from agents.PPO.MA_PPO.mappo_simple import (
    AGENT_IDS,
    AIFObsOvercookedMAEnv,
    RAY_AVAILABLE,
    _select_action,
    build_config,
    train,
)

try:
    import ray
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm
except ImportError:
    ray = None
    torch = None
    Algorithm = None


def _ppo_policy_stats(logits) -> dict:
    if torch is None or logits is None:
        return {
            "policy_idx": "",
            "q_pi_entropy": "",
            "top_policy_prob": "",
            "top_policy_plan": "",
        }
    probs = torch.softmax(logits.reshape(-1), dim=0)
    p = probs.detach().cpu().numpy()
    idx = int(np.argmax(p))
    H = float(-np.sum(p * np.log(p + 1e-16)))
    dest, mode = sal_csv.semantic_dest_mode(ind_model_init.ACTION_NAMES, idx)
    plan = f"{dest}:{mode}" if dest else str(idx)
    return {
        "policy_idx": idx,
        "q_pi_entropy": round(H, 6),
        "top_policy_prob": round(float(p[idx]), 6),
        "top_policy_plan": plan,
    }


def _default_seed_lists(n_runs: int, episode_start: int) -> tuple[list[int], list[int], list[int]]:
    return (
        [episode_start + i for i in range(n_runs)],
        [1000 + i for i in range(n_runs)],
        [2000 + i for i in range(n_runs)],
    )


def run_one_episode_logged(
    algo,
    *,
    episode_seed: int,
    agent0_seed: int,
    agent1_seed: int,
    max_steps: int,
    log_dir: Path,
    shared_policy: bool,
    stochastic: bool,
    layout: str,
    horizon: int,
) -> tuple[float, float, Path]:
    env = AIFObsOvercookedMAEnv({"layout": layout, "horizon": horizon})
    obs, _infos = env.reset(seed=episode_seed)
    step_csv = sal_csv.open_mappo_log(log_dir, episode_seed, agent0_seed, agent1_seed)

    total_reward_0 = 0.0
    total_reward_1 = 0.0

    for step in range(1, max_steps + 1):
        actions = {}
        stats = {}
        for i_agent, aid in enumerate(AGENT_IDS):
            pid = "shared" if shared_policy else aid
            module = algo.get_module(pid)
            act_int, logits = _select_action(module, obs[aid], stochastic=stochastic)
            actions[aid] = act_int
            stats[aid] = _ppo_policy_stats(logits)

        _obs, rewards, terminated, truncated, next_infos = env.step(actions)

        r0 = float(rewards["agent_0"])
        r1 = float(rewards["agent_1"])
        total_reward_0 += r0
        total_reward_1 += r1

        sem0 = next_infos["agent_0"].get("semantic", {})
        sem1 = next_infos["agent_1"].get("semantic", {})
        prim0 = int(sem0.get("primitive", ind_model_init.STAY))
        prim1 = int(sem1.get("primitive", ind_model_init.STAY))
        ps0 = stats["agent_0"]
        ps1 = stats["agent_1"]

        step_csv.write(
            {
                "paradigm": "mappo",
                "episode_seed": episode_seed,
                "agent0_seed": agent0_seed,
                "agent1_seed": agent1_seed,
                "step": step,
                "a0_policy_idx": ps0["policy_idx"],
                **sal_csv._meta_cols(
                    {"destination": sem0.get("destination", ""), "mode": sem0.get("mode", "")},
                    "a0",
                ),
                "a0_primitive": prim0,
                "a0_primitive_name": ind.PRIMITIVE_ACTION_NAMES.get(prim0, str(prim0)),
                "a0_q_pi_entropy": ps0["q_pi_entropy"],
                "a0_top_policy_prob": ps0["top_policy_prob"],
                "a0_top_policy_plan": ps0["top_policy_plan"],
                "a1_policy_idx": ps1["policy_idx"],
                **sal_csv._meta_cols(
                    {"destination": sem1.get("destination", ""), "mode": sem1.get("mode", "")},
                    "a1",
                ),
                "a1_primitive": prim1,
                "a1_primitive_name": ind.PRIMITIVE_ACTION_NAMES.get(prim1, str(prim1)),
                "a1_q_pi_entropy": ps1["q_pi_entropy"],
                "a1_top_policy_prob": ps1["top_policy_prob"],
                "a1_top_policy_plan": ps1["top_policy_plan"],
                "reward_a0": r0,
                "reward_a1": r1,
                "cumulative_reward_a0": total_reward_0,
                "cumulative_reward_a1": total_reward_1,
                "terminated": bool(terminated.get("__all__")),
                "truncated": bool(truncated.get("__all__")),
            }
        )

        obs = _obs
        if terminated.get("__all__") or truncated.get("__all__"):
            break

    step_csv.close()
    return total_reward_0, total_reward_1, step_csv.path


def _find_rllib_checkpoint_root(path: Path) -> Path | None:
    """Return directory containing rllib_checkpoint.json under path (or path itself)."""
    if not path.exists():
        return None
    if (path / "rllib_checkpoint.json").is_file():
        return path.resolve()
    nested = sorted(path.glob("checkpoint_*"), key=lambda x: x.stat().st_mtime)
    for sub in reversed(nested):
        if (sub / "rllib_checkpoint.json").is_file():
            return sub.resolve()
    for ckpt_json in sorted(path.rglob("rllib_checkpoint.json"), key=lambda p: p.stat().st_mtime):
        return ckpt_json.parent.resolve()
    return None


def _checkpoint_is_valid(path: Path) -> bool:
    return _find_rllib_checkpoint_root(path) is not None


def _resolve_checkpoint(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    root = _find_rllib_checkpoint_root(p)
    if root is None:
        raise ValueError(
            f"No RLlib checkpoint under {p} (expected rllib_checkpoint.json). "
            "Training saves only when a seed finishes — wait for 1M steps or use --checkpoint."
        )
    return str(root)


def _seed_checkpoint_dir(base: Path, episode_seed: int) -> Path:
    return base / f"ep{int(episode_seed)}"


def _make_train_args(args, *, episode_seed: int, checkpoint_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        layout=args.layout,
        horizon=args.horizon,
        seed=int(episode_seed),
        episode_seed=int(episode_seed),
        iterations=0,
        max_train_steps=args.max_train_steps,
        log_every=args.log_every,
        num_workers=args.num_workers,
        envs_per_worker=args.envs_per_worker,
        gpus=args.gpus,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        epochs=args.epochs,
        shared_policy=not args.separate_policies,
        separate_policies=args.separate_policies,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint=None,
        no_train=False,
        run_episode=False,
        eval_steps=args.max_steps,
        stochastic=True,
        deterministic=False,
        step_log=False,
        step_log_every=1,
        run_label=f"train ep={episode_seed}",
        checkpoint_every=getattr(args, "checkpoint_every", 0),
    )


def run_sweep(args) -> list[Path]:
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray RLlib required for MAPPO (pip install ray[rllib] torch)")

    episode_seeds, agent0_seeds, agent1_seeds = _default_seed_lists(args.n_runs, args.episode_start)
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    ckpt_base = Path(args.checkpoint_dir)
    if not ckpt_base.is_absolute():
        ckpt_base = PROJECT_ROOT / ckpt_base

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=max(2, args.num_workers + 1))

    shared = not args.separate_policies
    csv_paths: list[Path] = []

    try:
        shared_ckpt: str | None = None
        if not args.no_train and not args.per_seed_train:
            shared_dir = ckpt_base / "shared"
            print(f"Training shared policy {args.max_train_steps:,} env steps -> {shared_dir}")
            shared_ckpt = train(
                _make_train_args(args, episode_seed=args.train_seed, checkpoint_dir=shared_dir)
            )

        for i in range(args.n_runs):
            ep = episode_seeds[i]
            a0 = agent0_seeds[i]
            a1 = agent1_seeds[i]
            seed_ckpt_dir = _seed_checkpoint_dir(ckpt_base, ep)

            print("=" * 72)
            print(f"Seed {i + 1}/{args.n_runs}: episode_seed={ep}  a0={a0}  a1={a1}")
            print("=" * 72)

            if not args.no_train and args.per_seed_train:
                if args.skip_existing_train and _checkpoint_is_valid(seed_ckpt_dir):
                    print(f"  Skip train (valid checkpoint): {seed_ckpt_dir}")
                else:
                    print(
                        f"  Training {args.max_train_steps:,} env steps "
                        f"-> {seed_ckpt_dir}"
                    )
                    train(_make_train_args(args, episode_seed=ep, checkpoint_dir=seed_ckpt_dir))

            if args.checkpoint:
                ckpt_src = Path(args.checkpoint)
            elif shared_ckpt is not None:
                ckpt_src = Path(shared_ckpt)
            elif args.per_seed_train:
                ckpt_src = seed_ckpt_dir
            else:
                ckpt_src = ckpt_base / "shared"

            if not _checkpoint_is_valid(ckpt_src):
                msg = f"  No checkpoint for episode_seed={ep} under {ckpt_src}"
                if args.no_train:
                    print(msg + " — skip (train first or wait for run to finish)")
                    continue
                raise ValueError(msg)

            checkpoint_path = _resolve_checkpoint(str(ckpt_src))
            print(f"  Evaluating {args.max_steps} steps from: {checkpoint_path}")
            algo = Algorithm.from_checkpoint(checkpoint_path)
            try:
                r0, r1, path = run_one_episode_logged(
                    algo,
                    episode_seed=ep,
                    agent0_seed=a0,
                    agent1_seed=a1,
                    max_steps=args.max_steps,
                    log_dir=log_dir,
                    shared_policy=shared,
                    stochastic=not args.deterministic,
                    layout=args.layout,
                    horizon=args.horizon,
                )
                print(
                    f"  total_reward A0={r0:.1f} A1={r1:.1f} sum={r0 + r1:.1f}  -> {path.name}"
                )
                csv_paths.append(path)
            finally:
                algo.stop()

        if args.no_train and not csv_paths:
            ready = [
                ep
                for ep in episode_seeds
                if _checkpoint_is_valid(_seed_checkpoint_dir(ckpt_base, ep))
            ]
            raise RuntimeError(
                f"No eval CSVs written: 0/{args.n_runs} seeds have checkpoints under {ckpt_base}. "
                f"Ready: {ready or 'none'}. "
                "Run without --no-train, or wait until training logs 'Saved checkpoint' per seed."
            )
        return csv_paths
    finally:
        try:
            ray.shutdown()
        except Exception:
            pass


def run_plots(log_dir: Path) -> None:
    plot_single = PROJECT_ROOT / "utils/plotting/plot_sal_semantic_action_level.py"
    plot_pair = PROJECT_ROOT / "utils/plotting/plot_sal_pair_comparison.py"
    plot_triple = PROJECT_ROOT / "utils/plotting/plot_sal_triple_comparison.py"
    out_mappo = PROJECT_ROOT / "results/Overcooked/MAPPO"
    pairs = [
        (PROJECT_ROOT / "logs/sal_fc", "FC", out_mappo.parent / "compare_fc_mappo"),
        (PROJECT_ROOT / "logs/sal_ind", "IND", out_mappo.parent / "compare_ind_mappo"),
    ]
    subprocess.run(
        [sys.executable, str(plot_single), str(log_dir), "-o", str(out_mappo), "--no-show"],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    for other_dir, label, out_dir in pairs:
        if not other_dir.is_dir() or not list(other_dir.glob("sal_*.csv")):
            print(f"Skip pair vs {label}: no logs in {other_dir}")
            continue
        subprocess.run(
            [
                sys.executable,
                str(plot_pair),
                "--a",
                str(other_dir),
                "--b",
                str(log_dir),
                "--label-a",
                label,
                "--label-b",
                "MAPPO",
                "-o",
                str(out_dir),
            ],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
    fc_dir = PROJECT_ROOT / "logs/sal_fc"
    ind_dir = PROJECT_ROOT / "logs/sal_ind"
    ic_dir = PROJECT_ROOT / "logs/sal_ic"
    triple_args = [
        sys.executable,
        str(plot_triple),
        "--fc",
        str(fc_dir),
        "--ind",
        str(ind_dir),
        "--mappo",
        str(log_dir),
        "-o",
        str(out_mappo.parent / "compare_fc_ind_mappo"),
    ]
    if (
        fc_dir.is_dir()
        and ind_dir.is_dir()
        and list(fc_dir.glob("sal_*.csv"))
        and list(ind_dir.glob("sal_*.csv"))
    ):
        subprocess.run(triple_args, check=True, cwd=str(PROJECT_ROOT))
    if ic_dir.is_dir() and list(ic_dir.glob("sal_*.csv")):
        subprocess.run(
            [
                sys.executable,
                str(plot_pair),
                "--a",
                str(ic_dir),
                "--b",
                str(log_dir),
                "--label-a",
                "IC",
                "--label-b",
                "MAPPO",
                "-o",
                str(out_mappo.parent / "compare_ic_mappo"),
            ],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        if fc_dir.is_dir() and list(fc_dir.glob("sal_*.csv")):
            subprocess.run(
                [
                    sys.executable,
                    str(plot_pair),
                    "--a",
                    str(fc_dir),
                    "--b",
                    str(ic_dir),
                    "--label-a",
                    "FC",
                    "--label-b",
                    "IC",
                    "-o",
                    str(out_mappo.parent / "compare_fc_ic"),
                ],
                check=True,
                cwd=str(PROJECT_ROOT),
            )
        if ind_dir.is_dir() and list(ind_dir.glob("sal_*.csv")):
            subprocess.run(
                [
                    sys.executable,
                    str(plot_pair),
                    "--a",
                    str(ind_dir),
                    "--b",
                    str(ic_dir),
                    "--label-a",
                    "IND",
                    "--label-b",
                    "IC",
                    "-o",
                    str(out_mappo.parent / "compare_ind_ic"),
                ],
                check=True,
                cwd=str(PROJECT_ROOT),
            )
        if (
            fc_dir.is_dir()
            and ind_dir.is_dir()
            and list(fc_dir.glob("sal_*.csv"))
            and list(ind_dir.glob("sal_*.csv"))
        ):
            subprocess.run(
                [
                    sys.executable,
                    str(plot_triple),
                    "--fc",
                    str(fc_dir),
                    "--ind",
                    str(ind_dir),
                    "--ic",
                    str(ic_dir),
                    "--mappo",
                    str(log_dir),
                    "-o",
                    str(out_mappo.parent / "compare_fc_ind_ic_mappo"),
                ],
                check=True,
                cwd=str(PROJECT_ROOT),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPPO SAL seed sweep with CSV logging")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--episode-start", type=int, default=76)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--horizon", type=int, default=1510, help="Env horizon (>= max-steps)")
    parser.add_argument("--log-dir", type=str, default="logs/sal_mappo")
    parser.add_argument("--log-csv", action="store_true", default=True)

    parser.add_argument("--no-train", action="store_true", help="Eval only; skip seeds with no checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Shared RLlib checkpoint for all seeds (overrides per-seed dirs).",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/mappo_sal_cramped_room")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=1_000_000,
        help="Env steps to train per episode_seed (default: 1M).",
    )
    parser.add_argument(
        "--single-train",
        action="store_true",
        help="Train once (shared checkpoint) then eval all seeds. Default: train per seed.",
    )
    parser.add_argument(
        "--skip-existing-train",
        action="store_true",
        help="Skip training when a valid RLlib checkpoint exists under ep{N}/.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=200_000,
        help="Save intermediate RLlib checkpoint every N env steps (0=final only).",
    )
    parser.add_argument("--train-seed", type=int, default=0, help="RLlib seed if --single-train")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--envs-per-worker", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
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
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--plot", action="store_true", help="Run single + pair plots after eval")
    args = parser.parse_args()
    args.per_seed_train = not args.single_train

    if not args.log_csv:
        print("Note: --log-csv is on by default for this runner")

    print("=" * 72)
    print("MAPPO SEMANTIC ACTION LEVEL — SEED SWEEP")
    print(f"  train: {args.max_train_steps:,} env steps per seed" if args.per_seed_train and not args.no_train else "  train: skipped")
    print(f"  eval:  {args.max_steps} steps x {args.n_runs} seeds")
    if args.no_train and args.per_seed_train and not args.checkpoint:
        ckpt_base = Path(args.checkpoint_dir)
        if not ckpt_base.is_absolute():
            ckpt_base = PROJECT_ROOT / ckpt_base
        ep_seeds, _, _ = _default_seed_lists(args.n_runs, args.episode_start)
        ready = [ep for ep in ep_seeds if _checkpoint_is_valid(_seed_checkpoint_dir(ckpt_base, ep))]
        print(f"  checkpoints ready: {len(ready)}/{args.n_runs} under {ckpt_base}")
    print("=" * 72)

    paths = run_sweep(args)
    print(f"\nWrote {len(paths)} CSV file(s) to {args.log_dir}")

    if args.plot:
        run_plots(Path(args.log_dir))


if __name__ == "__main__":
    main()
