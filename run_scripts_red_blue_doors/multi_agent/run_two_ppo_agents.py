"""
Run two PPO agents on TwoAgentRedBlueButton with the same state as AIF agents.

Uses Ray RLlib PPO (PPOConfig). Observation is the same as the AIF model observation:
env_obs_to_model_obs(env_obs, width) flattened to a 10-D vector, so PPO sees the same
state as the Fully Collective AIF agent. Both PPO agents receive this joint observation.
Runs the same protocol as the AIF scripts (seeds, episodes, episodes_per_config, configs)
and supports --stats-output for comparison.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import numpy as np
import json
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
    """

    def __init__(self, config=None, **kwargs):
        super().__init__()
        if config is not None and hasattr(config, "env_config"):
            kwargs.update(getattr(config, "env_config", {}))
        elif isinstance(config, dict):
            kwargs.update(config)
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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


# -----------------------------------------------------------------------------
# Train PPO
# -----------------------------------------------------------------------------

def train_ppo(seeds, episodes_per_seed, episodes_per_config, max_steps, seed=0, checkpoint_dir=None, verbose=True):
    if not RAY_AVAILABLE:
        raise RuntimeError(f"Ray RLlib not available: {IMPORT_ERR}")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=2)

    # We train on a single fixed config (or could sample); evaluation will use varying configs.
    rng = np.random.default_rng(seed)
    train_config = generate_random_config(rng)
    env_config = {
        "width": 3,
        "height": 3,
        "red_button_pos": train_config["red_pos"],
        "blue_button_pos": train_config["blue_pos"],
        "agent1_start_pos": (0, 0),
        "agent2_start_pos": (2, 2),
        "max_steps": max_steps,
    }
    env_instance = RedBlueButtonPPOWrapper(env_config)

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
            train_batch_size=min(2000, 64 * 50),
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
        checkpoint_dir = project_root / "logs" / f"ppo_redblue_two_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    algo = config.build_algo()
    num_iterations = max(50, (episodes_per_seed * 2) // 4)
    if verbose:
        print(f"Training PPO for {num_iterations} iterations (batch from 4 envs)...")
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
    return checkpoint_path


# -----------------------------------------------------------------------------
# Evaluate with same protocol as AIF (seeds, episodes, configs)
# -----------------------------------------------------------------------------

def run_seed_experiment(algo, seed, num_episodes, episodes_per_config, max_steps, progress_callback=None):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    results = []
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    configs = [generate_random_config(rng) for _ in range(num_configs)]
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
                    else:
                        action = fwd["action"]
                    action = action[0].item() if isinstance(action, torch.Tensor) and action.dim() > 0 else (action.item() if isinstance(action, torch.Tensor) else int(action))
                    actions[agent_id] = int(action)
                except Exception:
                    actions[agent_id] = int(np.random.randint(0, 6))
            a1, a2 = actions["agent_0"], actions["agent_1"]
            obs, reward, terminated, truncated, info = env.step((a1, a2))
            reward = float(reward)
            episode_reward += reward
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
    args = parser.parse_args()

    if not RAY_AVAILABLE:
        print(f"Error: {IMPORT_ERR}")
        sys.exit(1)

    seeds_to_run = [args.seed] if args.seed is not None else list(range(args.seeds))
    num_episodes = args.episodes
    episodes_per_config = args.episodes_per_config
    max_steps = args.max_steps

    if args.checkpoint or args.no_train:
        if not args.checkpoint:
            print("Error: --checkpoint required when using --no-train")
            sys.exit(1)
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_abs():
            checkpoint_path = project_root / checkpoint_path
    else:
        checkpoint_path = train_ppo(
            seeds=len(seeds_to_run),
            episodes_per_seed=num_episodes,
            episodes_per_config=episodes_per_config,
            max_steps=max_steps,
            seed=seeds_to_run[0],
            verbose=args.verbose,
        )

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    algo = Algorithm.from_checkpoint(str(checkpoint_path))

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []
    seed_summaries = []
    total_episodes = len(seeds_to_run) * num_episodes
    with tqdm(total=total_episodes, desc="Total", unit="ep", position=0) as pbar:
        for seed in seeds_to_run:
            results = run_seed_experiment(
                algo, seed, num_episodes, episodes_per_config, max_steps,
                progress_callback=pbar.update,
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

    algo.stop()

    seed_summaries_ser = [
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in s.items()}
        for s in seed_summaries
    ]
    stats = {
        "paradigm": "ppo",
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
    stats_path = log_dir / f"two_ppo_agents_seeds{len(seeds_to_run)}_ep{num_episodes}_{timestamp}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("TWO PPO AGENTS - RESULTS (same state as AIF)")
    print("=" * 80)
    print(f"Success rate: {stats['success_rate']:.1f}%")
    print(f"Mean reward: {stats['mean_reward']:+.2f}")
    print(f"Mean steps: {stats['mean_steps']:.1f}")
    print(f"Stats: {stats_path}")
    if args.stats_output:
        print(f"Stats (comparison): {args.stats_output}")
    try:
        ray.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
