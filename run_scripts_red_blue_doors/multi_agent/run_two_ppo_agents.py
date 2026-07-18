"""
Run two PPO agents on TwoAgentRedBlueButton with the same state as AIF agents.

Uses Ray RLlib PPO (PPOConfig). Observation is the same as the AIF model observation:
env_obs_to_model_obs(env_obs, width) flattened to a 10-D vector, so PPO sees the same
state as the Fully Collective AIF agent. Both PPO agents receive this joint observation.
Runs the same evaluation protocol as the AIF scripts (seeds, episodes, episodes_per_config,
configs) and supports --stats-output for comparison.

Two training protocols are supported via --mode:
  - "pretrained" (default): generous offline training budget, on domain-randomized maps
    (not a single fixed map), then the policy is frozen for evaluation.
  - "online": training env-steps are capped to roughly the AIF paradigms' total experience
    budget (episodes * max_steps by default), and training uses the *same* map schedule as
    evaluation (episodes_per_config), so PPO isn't given a much larger, hidden, out-of-
    distribution training budget the way "pretrained" is. This is a budget-matched
    approximation of online learning (it still uses RLlib's own rollout + a final frozen-
    policy evaluation pass), not literal step-by-step interleaved learning.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import json
import csv
import argparse
from datetime import datetime
from tqdm import tqdm
from gymnasium import spaces

# Same env as AIF scripts
from environments.RedBlueButton.TwoAgentRedBlueButton import TwoAgentRedBlueButtonEnv
# Same observation -> state conversion as AIF (FullyCollective model)
from generative_models.MA_ActiveInference.RedBlueButton.FullyCollective import env_utils as aif_env_utils

try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.algorithm import Algorithm
    import ray
    import torch
    RAY_AVAILABLE = True
except ImportError as e:
    RAY_AVAILABLE = False
    IMPORT_ERR = str(e)


# -----------------------------------------------------------------------------
# Observation: same state as AIF agent (model_obs flattened)
# -----------------------------------------------------------------------------

def env_obs_to_ppo_vector(env_obs, width=3):
    """
    Convert env observation to the same state the AIF agent sees, as a flat vector.
    Uses the same env_obs_to_model_obs as FullyCollective, then flattens to fixed order.
    """
    model_obs = aif_env_utils.env_obs_to_model_obs(env_obs, width=width)
    # Fixed order matching model_obs keys (same as AIF state)
    return np.array([
        float(model_obs["agent1_pos"]),
        float(model_obs["agent2_pos"]),
        float(model_obs["agent1_on_red_button"]),
        float(model_obs["agent1_on_blue_button"]),
        float(model_obs["agent2_on_red_button"]),
        float(model_obs["agent2_on_blue_button"]),
        float(model_obs["red_button_state"]),
        float(model_obs["blue_button_state"]),
        float(model_obs["game_result"]),
        float(model_obs["button_just_pressed"]),
    ], dtype=np.float32)


OBS_VECTOR_SIZE = 10


# -----------------------------------------------------------------------------
# RLlib multi-agent env wrapper (same base env as AIF)
# -----------------------------------------------------------------------------

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    MultiAgentEnv = object


class RedBlueButtonPPOWrapper(MultiAgentEnv):
    """
    Wraps TwoAgentRedBlueButton (same as AIF) for RLlib.
    Both agents receive the same observation: AIF model_obs flattened (same state as AIF).

    Training map-config strategy is controlled via env_config["config_mode"]:
      - "fixed": button positions fixed at construction for the whole run (legacy behavior).
      - "domain_random": a fresh random map is drawn (from env_config["config_rng_seed"],
        offset per parallel env instance) every time the env resets, so training sees the
        distribution of maps rather than a single one.
      - "matched_schedule": cycles deterministically through env_config["schedule_configs"]
        (advancing one entry every env_config["episodes_per_config"] resets) -- i.e. the
        exact same map sequence evaluation uses for this seed.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None and hasattr(config, "env_config"):
            kwargs.update(getattr(config, "env_config", {}))
        elif isinstance(config, dict):
            kwargs.update(config)

        self._config_mode = kwargs.pop("config_mode", "fixed")
        self._episodes_per_config = kwargs.pop("episodes_per_config", None) or 1
        self._schedule_configs = kwargs.pop("schedule_configs", None)
        cfg_rng_seed = kwargs.pop("config_rng_seed", None)

        # Offset the RNG per parallel env-runner / vector-env instance so multiple training
        # env copies don't all draw the identical "random" map sequence.
        worker_index = int(getattr(config, "worker_index", 0) or 0) if config is not None else 0
        vector_index = int(getattr(config, "vector_index", 0) or 0) if config is not None else 0
        self._cfg_rng = (
            np.random.default_rng(cfg_rng_seed + worker_index * 1009 + vector_index * 97)
            if cfg_rng_seed is not None else None
        )
        self._reset_count = 0

        self.base_env = TwoAgentRedBlueButtonEnv(**kwargs)
        self.width = self.base_env.width
        self.agents = ["agent_0", "agent_1"]
        self.possible_agents = self.agents.copy()

        # Observation space: same state as AIF, flattened to vector
        self.observation_space = spaces.Dict({
            "agent_0": spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_VECTOR_SIZE,), dtype=np.float32
            ),
            "agent_1": spaces.Box(
                low=-np.inf, high=np.inf, shape=(OBS_VECTOR_SIZE,), dtype=np.float32
            ),
        })
        self.action_space = spaces.Dict({
            "agent_0": spaces.Discrete(6),
            "agent_1": spaces.Discrete(6),
        })

    def _obs_to_vec(self, env_obs):
        return env_obs_to_ppo_vector(env_obs, width=self.width)

    def _next_training_config(self):
        if self._config_mode == "matched_schedule" and self._schedule_configs:
            idx = min(
                self._reset_count // max(1, self._episodes_per_config),
                len(self._schedule_configs) - 1,
            )
            return self._schedule_configs[idx]
        if self._config_mode == "domain_random" and self._cfg_rng is not None:
            return generate_random_config(self._cfg_rng)
        return {"red_pos": self.base_env.red_button, "blue_pos": self.base_env.blue_button}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._config_mode in ("domain_random", "matched_schedule"):
            cfg = self._next_training_config()
            self.base_env = TwoAgentRedBlueButtonEnv(
                width=self.base_env.width,
                height=self.base_env.height,
                red_button_pos=cfg["red_pos"],
                blue_button_pos=cfg["blue_pos"],
                agent1_start_pos=self.base_env.agent1_start_pos,
                agent2_start_pos=self.base_env.agent2_start_pos,
                max_steps=self.base_env.max_steps,
            )
        self._reset_count += 1
        env_obs, _ = self.base_env.reset(seed=seed, options=options)
        vec = self._obs_to_vec(env_obs)
        observations = {"agent_0": vec.copy(), "agent_1": vec.copy()}
        infos = {"agent_0": {}, "agent_1": {}}
        return observations, infos

    def step(self, actions):
        action1 = int(actions["agent_0"])
        action2 = int(actions["agent_1"])
        env_obs, reward, terminated, truncated, info = self.base_env.step((action1, action2))
        reward = float(reward)
        vec = self._obs_to_vec(env_obs)
        observations = {"agent_0": vec.copy(), "agent_1": vec.copy()}
        rewards = {"agent_0": reward, "agent_1": reward}
        terminated_dict = {"agent_0": terminated, "agent_1": terminated, "__all__": terminated}
        truncated_dict = {"agent_0": truncated, "agent_1": truncated, "__all__": truncated}
        infos = {"agent_0": info.copy(), "agent_1": info.copy()}
        return observations, rewards, terminated_dict, truncated_dict, infos


# -----------------------------------------------------------------------------
# Config generation (same as AIF scripts)
# -----------------------------------------------------------------------------

def generate_random_config(rng, grid_width=3, grid_height=3):
    available = []
    for x in range(grid_width):
        for y in range(grid_height):
            if (x, y) not in [(0, 0), (grid_width - 1, grid_height - 1)]:
                available.append((x, y))
    rng.shuffle(available)
    return {"red_pos": available[0], "blue_pos": available[1]}


def _train_rng_seed(seed):
    """
    Derive a training-config RNG seed decoupled from the evaluation RNG (which uses
    `seed` directly). Without this, np.random.default_rng(seed) drawn once for training
    and np.random.default_rng(seed) drawn again for evaluation produce the *same* first
    config -- i.e. PPO's training map would silently be identical to eval config #0.
    """
    return (int(seed) * 7919 + 104729) & 0xFFFFFFFF


def build_eval_configs(seed, num_episodes, episodes_per_config):
    """Same map-config schedule used for evaluation (and, in --mode online, for training)."""
    rng = np.random.default_rng(seed)
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    return [generate_random_config(rng) for _ in range(num_configs)]


# -----------------------------------------------------------------------------
# Train PPO
# -----------------------------------------------------------------------------

def train_ppo(
    episodes_per_seed,
    episodes_per_config,
    max_steps,
    seed=0,
    checkpoint_dir=None,
    verbose=True,
    mode="pretrained",
    config_mode=None,
    train_steps_budget=None,
):
    """
    Train the PPO policy.

    mode="pretrained": generous offline training budget (legacy formula, unless
        train_steps_budget is given explicitly). Defaults to domain-randomized training
        maps (config_mode="domain_random") so the frozen policy isn't evaluated purely
        out-of-distribution on 7/8 unseen maps.

    mode="online": training env-steps capped to train_steps_budget, which defaults to
        episodes_per_seed * max_steps -- i.e. the same nominal total experience budget
        the AIF paradigms get across their scored episodes. Defaults to training on the
        *same* map schedule evaluation uses (config_mode="matched_schedule"). This is a
        budget-matched approximation of "online" learning: it still trains via RLlib's
        own env-runner rollout and then freezes the policy for a final evaluation pass,
        rather than literally interleaving learning step-by-step with the AIF-style
        evaluation loop.
    """
    if not RAY_AVAILABLE:
        raise RuntimeError(f"Ray RLlib not available: {IMPORT_ERR}")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    if config_mode is None:
        config_mode = "matched_schedule" if mode == "online" else "domain_random"

    env_config = {
        "width": 3,
        "height": 3,
        "agent1_start_pos": (0, 0),
        "agent2_start_pos": (2, 2),
        "max_steps": max_steps,
        "config_mode": config_mode,
        "episodes_per_config": episodes_per_config,
    }

    if config_mode == "fixed":
        train_rng = np.random.default_rng(_train_rng_seed(seed))
        fixed_config = generate_random_config(train_rng)
        env_config["red_button_pos"] = fixed_config["red_pos"]
        env_config["blue_button_pos"] = fixed_config["blue_pos"]
    elif config_mode == "domain_random":
        env_config["config_rng_seed"] = _train_rng_seed(seed)
    elif config_mode == "matched_schedule":
        env_config["schedule_configs"] = build_eval_configs(seed, episodes_per_seed, episodes_per_config)
    else:
        raise ValueError(f"Unknown config_mode: {config_mode}")

    env_instance = RedBlueButtonPPOWrapper(env_config)

    train_batch_size = min(2000, 200 * max_steps)
    config = (
        PPOConfig()
        .environment(env=RedBlueButtonPPOWrapper, env_config=env_config)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            train_batch_size=train_batch_size,
            minibatch_size=128,
            num_epochs=10,
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=4, num_cpus_per_env_runner=1)
        .multi_agent(
            policies={
                "agent_0": (None, env_instance.observation_space["agent_0"], env_instance.action_space["agent_0"], {}),
                "agent_1": (None, env_instance.observation_space["agent_1"], env_instance.action_space["agent_1"], {}),
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id,
        )
        .debugging(seed=seed)
    )

    if checkpoint_dir is None:
        checkpoint_dir = (
            project_root / "logs"
            / f"ppo_redblue_two_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    algo = config.build_algo()

    if train_steps_budget is None and mode == "online":
        train_steps_budget = episodes_per_seed * max_steps
    if train_steps_budget is not None:
        num_iterations = max(1, train_steps_budget // train_batch_size)
    else:
        num_iterations = max(50, (episodes_per_seed * 2) // 4)

    if verbose:
        total_steps = num_iterations * train_batch_size
        print(
            f"Training PPO (mode={mode}, config_mode={config_mode}) for {num_iterations} "
            f"iterations x {train_batch_size} batch size = {total_steps} total training "
            f"env-steps..."
        )
    for i in range(num_iterations):
        algo.train()
    path = algo.save(str(checkpoint_dir))
    try:
        if hasattr(path, "checkpoint") and hasattr(path.checkpoint, "path"):
            checkpoint_path = path.checkpoint.path
        else:
            checkpoint_path = str(checkpoint_dir)
    except Exception:
        checkpoint_path = str(checkpoint_dir)
    algo.stop()

    training_meta = {
        "mode": mode,
        "config_mode": config_mode,
        "num_iterations": num_iterations,
        "train_batch_size": train_batch_size,
        "total_training_env_steps": num_iterations * train_batch_size,
    }
    return checkpoint_path, training_meta


# -----------------------------------------------------------------------------
# Evaluate with same protocol as AIF (seeds, episodes, configs)
# -----------------------------------------------------------------------------

def run_seed_experiment(
    algo, seed, num_episodes, episodes_per_config, max_steps,
    progress_callback=None, csv_writer=None,
):
    np.random.seed(seed)
    results = []
    configs = build_eval_configs(seed, num_episodes, episodes_per_config)
    action_names = TwoAgentRedBlueButtonEnv.ACTION_MEANING
    env = None
    for episode in range(1, num_episodes + 1):
        config_idx = (episode - 1) // episodes_per_config
        config = configs[config_idx]
        if (episode - 1) % episodes_per_config == 0 or env is None:
            env = TwoAgentRedBlueButtonEnv(
                width=3, height=3,
                red_button_pos=config["red_pos"],
                blue_button_pos=config["blue_pos"],
                agent1_start_pos=(0, 0),
                agent2_start_pos=(2, 2),
                max_steps=max_steps,
            )
        obs, _ = env.reset(seed=seed + episode)
        vec = env_obs_to_ppo_vector(obs, width=env.width)
        episode_reward = 0.0
        step = 0
        for step in range(1, max_steps + 1):
            actions = {}
            action_probs = {}
            for agent_id in ["agent_0", "agent_1"]:
                try:
                    module = algo.get_module(agent_id)
                    obs_tensor = torch.FloatTensor(vec).unsqueeze(0)
                    with torch.no_grad():
                        fwd = module.forward_inference({"obs": obs_tensor})
                    if "action_dist_inputs" in fwd:
                        try:
                            dist_cls = module.get_inference_action_dist_cls()
                            dist = dist_cls.from_logits(fwd["action_dist_inputs"])
                            action = dist.deterministic_sample() if hasattr(dist, "deterministic_sample") else dist.sample()
                        except Exception:
                            action = torch.argmax(fwd["action_dist_inputs"], dim=-1)
                        if csv_writer is not None:
                            try:
                                probs = torch.softmax(fwd["action_dist_inputs"][0], dim=-1).tolist()
                                action_probs[agent_id] = [round(p, 4) for p in probs]
                            except Exception:
                                action_probs[agent_id] = None
                    else:
                        action = fwd["action"]
                        action_probs[agent_id] = None
                    action = action[0].item() if isinstance(action, torch.Tensor) and action.dim() > 0 else (action.item() if isinstance(action, torch.Tensor) else int(action))
                    actions[agent_id] = int(action)
                except Exception:
                    actions[agent_id] = int(np.random.randint(0, 6))
                    action_probs[agent_id] = None

            # Pre-step state/map for the row (same convention as the AIF CSV logs: log the
            # state the action was chosen from, plus the resulting reward/outcome below).
            if csv_writer is not None:
                model_obs = aif_env_utils.env_obs_to_model_obs(obs, width=env.width)
                grid = env.render(mode="silent")
                map_str = "|".join("".join(row) for row in grid)

            a1, a2 = actions["agent_0"], actions["agent_1"]
            obs, reward, terminated, truncated, info = env.step((a1, a2))
            reward = float(reward)
            episode_reward += reward

            if csv_writer is not None:
                csv_writer.writerow({
                    "seed": seed,
                    "episode": episode,
                    "step": step,
                    "config_idx": config_idx,
                    "agent1_pos": model_obs["agent1_pos"],
                    "agent2_pos": model_obs["agent2_pos"],
                    "agent1_on_red_button": model_obs["agent1_on_red_button"],
                    "agent1_on_blue_button": model_obs["agent1_on_blue_button"],
                    "agent2_on_red_button": model_obs["agent2_on_red_button"],
                    "agent2_on_blue_button": model_obs["agent2_on_blue_button"],
                    "red_button_state": model_obs["red_button_state"],
                    "blue_button_state": model_obs["blue_button_state"],
                    "game_result": model_obs["game_result"],
                    "action1": a1,
                    "action1_name": action_names.get(a1, str(a1)),
                    "action2": a2,
                    "action2_name": action_names.get(a2, str(a2)),
                    "map": map_str,
                    "reward": reward,
                    "cumulative_reward": episode_reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "result": info.get("result", "neutral"),
                    "button_pressed": info.get("button_just_pressed", ""),
                    "pressed_by": info.get("button_pressed_by", ""),
                    "agent0_action_probs": action_probs.get("agent_0"),
                    "agent1_action_probs": action_probs.get("agent_1"),
                })

            vec = env_obs_to_ppo_vector(obs, width=env.width)
            if terminated or truncated:
                break
        results.append({
            "reward": episode_reward,
            "steps": step,
            "success": info.get("result") == "win",
        })
        if progress_callback:
            progress_callback(1)
    return results


def main():
    parser = argparse.ArgumentParser(description="Two PPO agents on RedBlueButton (same state as AIF)")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes-per-config", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint; if None, train first")
    parser.add_argument("--no-train", action="store_true", help="Only evaluate (requires --checkpoint)")
    parser.add_argument("--stats-output", type=str, default=None)
    parser.add_argument("--episode-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument(
        "--mode", type=str, choices=["pretrained", "online"], default="pretrained",
        help=(
            "PPO training protocol: 'pretrained' trains offline on domain-randomized maps "
            "with a generous budget, then freezes the policy for evaluation (default). "
            "'online' caps the training env-steps to match the AIF paradigms' total "
            "experience (episodes * max-steps) and trains on the same map schedule used "
            "for evaluation, as a budget-matched approximation of online learning."
        ),
    )
    parser.add_argument(
        "--train-config-mode", type=str,
        choices=["fixed", "domain_random", "matched_schedule"], default=None,
        help="Override training map-config strategy (default depends on --mode)",
    )
    parser.add_argument(
        "--train-steps-budget", type=int, default=None,
        help="Override total training env-steps (default depends on --mode)",
    )
    args = parser.parse_args()

    if not RAY_AVAILABLE:
        print(f"Error: {IMPORT_ERR}")
        sys.exit(1)

    seeds_to_run = [args.seed] if args.seed is not None else list(range(args.seeds))
    num_episodes = args.episodes
    episodes_per_config = args.episodes_per_config
    max_steps = args.max_steps

    training_meta = None
    if args.checkpoint or args.no_train:
        if not args.checkpoint:
            print("Error: --checkpoint required when using --no-train")
            sys.exit(1)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_abs():
            checkpoint_path = project_root / checkpoint_path
    else:
        checkpoint_path, training_meta = train_ppo(
            episodes_per_seed=num_episodes,
            episodes_per_config=episodes_per_config,
            max_steps=max_steps,
            seed=seeds_to_run[0],
            verbose=args.verbose,
            mode=args.mode,
            config_mode=args.train_config_mode,
            train_steps_budget=args.train_steps_budget,
        )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    algo = Algorithm.from_checkpoint(str(checkpoint_path))

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = (
        log_dir / f"two_ppo_agents_{args.mode}_seeds{len(seeds_to_run)}_ep{num_episodes}_{timestamp}.csv"
    )
    csv_fieldnames = [
        "seed", "episode", "step", "config_idx",
        "agent1_pos", "agent2_pos",
        "agent1_on_red_button", "agent1_on_blue_button",
        "agent2_on_red_button", "agent2_on_blue_button",
        "red_button_state", "blue_button_state", "game_result",
        "action1", "action1_name", "action2", "action2_name",
        "map", "reward", "cumulative_reward", "terminated", "truncated",
        "result", "button_pressed", "pressed_by",
        "agent0_action_probs", "agent1_action_probs",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer_obj = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, extrasaction="ignore")
    csv_writer_obj.writeheader()
    print(f"  CSV log: {csv_path}")

    all_results = []
    seed_summaries = []
    total_episodes = len(seeds_to_run) * num_episodes
    try:
        with tqdm(total=total_episodes, desc="Total", unit="ep", position=0) as pbar:
            for seed in seeds_to_run:
                results = run_seed_experiment(
                    algo, seed, num_episodes, episodes_per_config, max_steps,
                    progress_callback=pbar.update, csv_writer=csv_writer_obj,
                )
                all_results.extend(results)
                n = len(results)
                successes = sum(1 for r in results if r["success"])
                seed_summaries.append({
                    "seed": seed,
                    "successes": successes,
                    "total": n,
                    "success_rate": 100.0 * successes / max(1, n),
                    "avg_reward": float(np.mean([r["reward"] for r in results])),
                    "avg_steps": float(np.mean([r["steps"] for r in results])),
                    "first_half_wins": sum(1 for r in results[: n // 2] if r["success"]),
                    "second_half_wins": sum(1 for r in results[n // 2 :] if r["success"]),
                })
    finally:
        csv_file.close()

    algo.stop()

    seed_summaries_ser = [
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in s.items()}
        for s in seed_summaries
    ]
    stats = {
        "paradigm": f"ppo_{args.mode}",
        "ppo_mode": args.mode,
        "training_budget": training_meta,
        "n_seeds": len(seeds_to_run),
        "n_episodes_per_seed": num_episodes,
        "episodes_per_config": episodes_per_config,
        "max_steps": max_steps,
        "total_episodes": len(all_results),
        "total_successes": sum(1 for r in all_results if r["success"]),
        "success_rate": float(100 * sum(1 for r in all_results if r["success"]) / max(1, len(all_results))),
        "mean_reward": float(np.mean([r["reward"] for r in all_results])),
        "std_reward": float(np.std([r["reward"] for r in all_results])),
        "mean_steps": float(np.mean([r["steps"] for r in all_results])),
        "std_steps": float(np.std([r["steps"] for r in all_results])),
        "seed_summaries": seed_summaries_ser,
    }
    stats_path = (
        log_dir / f"two_ppo_agents_{args.mode}_seeds{len(seeds_to_run)}_ep{num_episodes}_{timestamp}_stats.json"
    )
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print(f"TWO PPO AGENTS ({args.mode.upper()}) - RESULTS (same state as AIF)")
    print("=" * 80)
    if training_meta:
        print(
            f"Training budget: config_mode={training_meta['config_mode']} "
            f"iterations={training_meta['num_iterations']} "
            f"total_env_steps={training_meta['total_training_env_steps']}"
        )
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Mean reward: {stats['mean_reward']:+.2f}")
    print(f"Mean steps: {stats['mean_steps']:.1f}")
    print(f"CSV log: {csv_path}")
    print(f"Stats: {stats_path}")
    if args.stats_output:
        print(f"Stats (comparison): {args.stats_output}")
    try:
        ray.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
