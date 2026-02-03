"""
Run two independent Active Inference agents in Overcooked environment.

Each agent uses a single-agent generative model and selects only its own action.
The other agent is treated as part of the environment.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import json
import csv
from datetime import datetime
import os

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning, module="pkg_resources")

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import Active Inference agent
from agents.ActiveInference.agent import Agent

# Import Independent paradigm model (use Independent's __init__.py directly)
# Add the Independent directory to path to import directly
independent_dir = Path(__file__).parent.parent / "generative_models" / "MA_ActiveInference" / "Overcooked" / "cramped_room" / "Independent"
sys.path.insert(0, str(independent_dir.parent))
from Independent import A_fn, B_fn, C_fn, D_fn, model_init, env_utils

# Initialize pygame and patch image loading if PNG support is missing
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Use dummy video driver
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'  # Hide pygame support prompt
try:
    import pygame
    pygame.mixer.quit()  # Don't need audio
    pygame.init()
    pygame.display.init()
    screen = pygame.display.set_mode((1, 1), pygame.HIDDEN)
    
    # Patch pygame.image.load if PNG support is missing
    if not pygame.image.get_extended():
        from PIL import Image
        import io
        original_load = pygame.image.load
        
        def patched_load(file):
            """Load image using PIL if pygame doesn't support PNG."""
            try:
                return original_load(file)
            except pygame.error as e:
                if "BMP" in str(e) or "not a Windows BMP" in str(e):
                    # Try loading with PIL and converting to pygame surface
                    pil_image = Image.open(file)
                    mode = pil_image.mode
                    size = pil_image.size
                    data = pil_image.tobytes()
                    # Use frombytes (newer API) or fromstring (older API)
                    if hasattr(pygame.image, 'frombytes'):
                        if mode == 'RGBA':
                            return pygame.image.frombytes(data, size, 'RGBA')
                        elif mode == 'RGB':
                            return pygame.image.frombytes(data, size, 'RGB')
                        else:
                            pil_image = pil_image.convert('RGB')
                            data = pil_image.tobytes()
                            return pygame.image.frombytes(data, size, 'RGB')
                    else:
                        # Fallback to fromstring for older pygame
                        if mode == 'RGBA':
                            return pygame.image.fromstring(data, size, 'RGBA')
                        elif mode == 'RGB':
                            return pygame.image.fromstring(data, size, 'RGB')
                        else:
                            pil_image = pil_image.convert('RGB')
                            data = pil_image.tobytes()
                            return pygame.image.fromstring(data, size, 'RGB')
                else:
                    raise
        
        pygame.image.load = patched_load
except Exception as e:
    print(f"Warning: Could not initialize pygame: {e}")
    pass

# Import environment
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv
from overcooked_ai_py.mdp.actions import Action


ACTION_NAMES = {
    0: "NORTH",
    1: "SOUTH",
    2: "EAST",
    3: "WEST",
    4: "STAY",
    5: "INTERACT"
}


def create_independent_agent(agent_idx, seed=None):
    """
    Create an independent Active Inference agent.
    
    Args:
        agent_idx: Agent index (0 or 1)
        seed: Random seed for reproducibility
    
    Returns:
        Agent instance
    """
    if seed is not None:
        np.random.seed(seed + agent_idx)
    
    # Environment parameters
    env_params = {
        "width": model_init.GRID_WIDTH,
        "height": model_init.GRID_HEIGHT,
    }
    
    # State factors and sizes
    state_factors = list(model_init.states.keys())
    state_sizes = {f: len(v) for f, v in model_init.states.items()}
    
    # Observation labels
    observation_labels = model_init.observations
    
    # Create agent
    # NOTE:
    # - For the RedBlueButton experiments, the functional AIF agent was tuned with a
    #   slightly longer policy length and higher precisions.
    # - In Overcooked, very low precisions (gamma/alpha) plus short policies tend to
    #   make q_pi almost uniform (high entropy) and the agents get stuck in
    #   dithering behaviours (e.g. WEST/EAST loops) without committing.
    # - Here we use more "decisive" settings closer to the original ones.
    agent = Agent(
        A_fn=A_fn,
        B_fn=B_fn,
        C_fn=C_fn,
        D_fn=D_fn,
        state_factors=state_factors,
        state_sizes=state_sizes,
        observation_labels=observation_labels,
        env_params=env_params,
        observation_state_dependencies=model_init.observation_state_dependencies,
        actions=list(range(model_init.N_ACTIONS)),
        gamma=4.0,  # Policy precision (higher = sharper q_pi)
        alpha=4.0,  # Action precision (higher = more decisive actions)
        policy_len=2,
        inference_horizon=2,
        action_selection="stochastic",
        sampling_mode="full",
        inference_algorithm="VANILLA",
        num_iter=16,
        dF_tol=0.001,
    )
    
    return agent


def run_episode(env, agent0, agent1, episode_num, max_steps, csv_writer=None, seed=None):
    """
    Run a single episode with two independent agents.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        agent0: Agent 0 instance
        agent1: Agent 1 instance
        episode_num: Episode number
        max_steps: Maximum steps per episode
        csv_writer: Optional CSV writer for logging
        seed: Random seed
    
    Returns:
        Dictionary with episode results
    """
    # Reset environment
    observations, infos = env.reset()
    
    # Get initial state for D config (match prior to true start positions/orientations)
    state = infos["agent_0"]["state"]
    
    config0 = env_utils.get_D_config_from_state(state, agent_idx=0)
    config1 = env_utils.get_D_config_from_state(state, agent_idx=1)
    
    agent0.reset(config=config0)
    agent1.reset(config=config1)
    
    episode_reward = 0.0
    episode_length = 0
    prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
    
    # Track previous positions for collision detection
    prev_pos0 = None
    prev_pos1 = None
    
    for step in range(1, max_steps + 1):
        # Convert environment observations to model observations
        # Use previous step's reward info for soup delivery detection
        obs0_model = env_utils.env_obs_to_model_obs(state, agent_idx=0, reward_info=prev_reward_info)
        obs1_model = env_utils.env_obs_to_model_obs(state, agent_idx=1, reward_info=prev_reward_info)
        
        # Store current positions before actions
        curr_pos0 = obs0_model["agent_pos"]
        curr_pos1 = obs1_model["agent_pos"]
        
        # Get actions from agents (model actions are already indices 0-5)
        action0_idx = int(agent0.step(obs0_model))
        action1_idx = int(agent1.step(obs1_model))
        
        # Print beliefs to terminal if verbose
        if hasattr(env, "_verbose") and env._verbose:
            print(f"\n{'═'*80}")
            print(f"  EPISODE {episode_num} │ STEP {step}")
            print(f"{'═'*80}")
            
            # Print observations compactly
            print(f"\n📥 OBSERVATIONS:")
            obs0_str = ", ".join([f"{k}={v}" for k, v in obs0_model.items()])
            obs1_str = ", ".join([f"{k}={v}" for k, v in obs1_model.items()])
            print(f"   Agent 0: {obs0_str}")
            print(f"   Agent 1: {obs1_str}")
            
            # Print state factor beliefs in a compact table format
            print(f"\n🧠 STATE BELIEFS:")
            print(f"   {'Factor':<20} {'Agent 0':<30} {'Agent 1':<30}")
            print(f"   {'─'*20} {'─'*30} {'─'*30}")
            
            qs0 = agent0.get_state_beliefs()
            qs1 = agent1.get_state_beliefs()
            
            for factor in agent0.state_factors:
                # Agent 0 beliefs
                probs0 = qs0[factor]
                map0 = int(np.argmax(probs0))
                max_prob0 = float(np.max(probs0))
                entropy0 = -np.sum(probs0 * np.log(probs0 + 1e-16))
                belief0_str = f"{map0} ({max_prob0:.2f}) H={entropy0:.2f}"
                
                # Agent 1 beliefs
                probs1 = qs1[factor]
                map1 = int(np.argmax(probs1))
                max_prob1 = float(np.max(probs1))
                entropy1 = -np.sum(probs1 * np.log(probs1 + 1e-16))
                belief1_str = f"{map1} ({max_prob1:.2f}) H={entropy1:.2f}"
                
                print(f"   {factor:<20} {belief0_str:<30} {belief1_str:<30}")
            
            # Print policy beliefs in a compact format
            print(f"\n🎯 POLICY BELIEFS:")
            
            for agent_idx, agent in enumerate([agent0, agent1]):
                q_pi = agent.get_policy_posterior()
                top_policies = agent.get_top_policies(top_k=3)
                policy_entropy = -np.sum(q_pi * np.log(q_pi + 1e-16))
                
                print(f"   Agent {agent_idx} (entropy: {policy_entropy:.3f}):")
                for rank, (pol, prob, idx) in enumerate(top_policies, 1):
                    pol_str = "→".join([ACTION_NAMES.get(int(a), str(a))[:1] for a in pol])
                    bar = "█" * int(prob * 20)  # Visual bar
                    print(f"      #{rank} [{pol_str:>8}] {bar:<20} {prob:.3f}")
            
            # Print utility and info_gain for each policy (when available)
            for agent_idx, agent in enumerate([agent0, agent1]):
                details = agent.get_last_policy_details()
                if not details:
                    continue
                print(f"\n   📊 Agent {agent_idx} — utility & info_gain per policy:")
                # Sort by prob descending
                details = sorted(details, key=lambda d: d["prob"], reverse=True)
                print(f"      {'Policy':<12} {'Utility':>10} {'InfoGain':>10} {'G':>10} {'Prob':>8}")
                print(f"      {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
                for d in details:
                    pol_str = "→".join([ACTION_NAMES.get(int(a), str(a))[0] for a in d["policy"]])
                    print(f"      {pol_str:<12} {d['utility']:>10.4f} {d['info_gain']:>10.4f} {d['G']:>10.4f} {d['prob']:>8.4f}")
            
            # Print selected actions prominently
            print(f"\n⚡ ACTIONS:")
            print(f"   Agent 0 → {ACTION_NAMES.get(action0_idx, 'UNKNOWN'):<10} [{action0_idx}]")
            print(f"   Agent 1 → {ACTION_NAMES.get(action1_idx, 'UNKNOWN'):<10} [{action1_idx}]")
            print(f"{'─'*80}")

        # Optional: log policy posteriors (q_pi) and state factor beliefs (qs) each step
        if hasattr(env, "_policy_log_fh") and env._policy_log_fh is not None:
            def _serialize_top_policies(agent, top_k):
                top = agent.get_top_policies(top_k=top_k)
                return [
                    {"policy_idx": int(idx), "policy": [int(a) for a in pol], "prob": float(prob)}
                    for (pol, prob, idx) in top
                ]
            
            def _serialize_state_beliefs(agent):
                """Serialize state factor beliefs (qs) to JSON-serializable format."""
                qs = agent.get_state_beliefs()
                return {
                    factor: {
                        "probabilities": [float(p) for p in qs[factor]],
                        "map_state": int(np.argmax(qs[factor])),  # Most likely state
                        "entropy": float(-np.sum(qs[factor] * np.log(qs[factor] + 1e-16)))  # Entropy of belief
                    }
                    for factor in agent.state_factors
                }

            top_k = getattr(env, "_policy_log_top_k", 5)
            include_full = bool(getattr(env, "_policy_log_full_q_pi", True))
            include_state_beliefs = bool(getattr(env, "_policy_log_state_beliefs", True))

            rec0 = {
                "episode": int(episode_num),
                "step": int(step),
                "agent": 0,
                "obs": {k: int(v) for k, v in obs0_model.items()},
                "action": int(action0_idx),
                "action_name": ACTION_NAMES.get(action0_idx, str(action0_idx)),
                "top_policies": _serialize_top_policies(agent0, top_k),
            }
            rec1 = {
                "episode": int(episode_num),
                "step": int(step),
                "agent": 1,
                "obs": {k: int(v) for k, v in obs1_model.items()},
                "action": int(action1_idx),
                "action_name": ACTION_NAMES.get(action1_idx, str(action1_idx)),
                "top_policies": _serialize_top_policies(agent1, top_k),
            }
            
            # Add full policy posterior if requested
            if include_full:
                rec0["q_pi"] = [float(x) for x in agent0.get_policy_posterior()]
                rec1["q_pi"] = [float(x) for x in agent1.get_policy_posterior()]
            
            # Add state factor beliefs if requested
            if include_state_beliefs:
                rec0["state_beliefs"] = _serialize_state_beliefs(agent0)
                rec1["state_beliefs"] = _serialize_state_beliefs(agent1)

            env._policy_log_fh.write(json.dumps(rec0) + "\n")
            env._policy_log_fh.write(json.dumps(rec1) + "\n")
            env._policy_log_fh.flush()
        
        # Step environment (expects action indices 0-5)
        actions = {
            "agent_0": action0_idx,
            "agent_1": action1_idx
        }
        
        observations, rewards, terminated, truncated, infos = env.step(actions)
        
        # Update state for next iteration
        state = infos["agent_0"]["state"]
        
        # Get new positions after step
        new_obs0 = env_utils.env_obs_to_model_obs(state, agent_idx=0, reward_info=prev_reward_info)
        new_obs1 = env_utils.env_obs_to_model_obs(state, agent_idx=1, reward_info=prev_reward_info)
        new_pos0 = new_obs0["agent_pos"]
        new_pos1 = new_obs1["agent_pos"]
        
        # Detect collisions (action attempted but position didn't change)
        collision0 = False
        collision1 = False
        if prev_pos0 is not None:
            # Check if agent tried to move but stayed in place
            if action0_idx in [0, 1, 2, 3]:  # Movement action
                if curr_pos0 == new_pos0:
                    collision0 = True
        if prev_pos1 is not None:
            if action1_idx in [0, 1, 2, 3]:  # Movement action
                if curr_pos1 == new_pos1:
                    collision1 = True
        
        # Update previous positions for next iteration
        prev_pos0 = new_pos0
        prev_pos1 = new_pos1
        
        # Update reward info for next step (sparse rewards indicate soup delivery)
        prev_reward_info = {
            "sparse_reward_by_agent": [
                infos["agent_0"].get("sparse_reward", 0),
                infos["agent_1"].get("sparse_reward", 0)
            ]
        }
        
        # Update episode statistics
        episode_reward += rewards["agent_0"]  # Shared reward
        episode_length = step
        
        # Print reward if verbose (after step is complete)
        if hasattr(env, "_verbose") and env._verbose:
            collision_msg = ""
            if collision0 or collision1:
                collision_msg = " ⚠️ COLLISION DETECTED: "
                if collision0:
                    collision_msg += f"Agent 0 blocked "
                if collision1:
                    collision_msg += f"Agent 1 blocked "
            print(f"   💰 Step reward: {rewards['agent_0']:.2f} │ Cumulative: {episode_reward:.2f}{collision_msg}")
            if terminated["__all__"] or truncated["__all__"]:
                print(f"\n   ✅ Episode complete! Total reward: {episode_reward:.2f}, Length: {episode_length}")
            print()
        
        # Log step if CSV writer provided
        if csv_writer is not None:
            csv_writer.writerow({
                "episode": episode_num,
                "step": step,
                "agent0_action": action0_idx,
                "agent0_action_name": ACTION_NAMES.get(action0_idx, str(action0_idx)),
                "agent1_action": action1_idx,
                "agent1_action_name": ACTION_NAMES.get(action1_idx, str(action1_idx)),
                "reward": rewards["agent_0"],
                "cumulative_reward": episode_reward,
                "terminated": terminated["__all__"],
                "truncated": truncated["__all__"],
            })
            # Note: flushing is handled at file-level by OS; if you want hard flush,
            # run with --log_policy_beliefs (JSONL is flushed every step).
        
        # Check if episode is done
        if terminated["__all__"] or truncated["__all__"]:
            break
    
    return {
        "episode": episode_num,
        "reward": episode_reward,
        "length": episode_length,
        "success": episode_reward > 0,  # Success if any soup was delivered
    }


def main():
    parser = argparse.ArgumentParser(description="Run two independent Active Inference agents in Overcooked")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--layout", type=str, default="cramped_room", help="Layout name")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--verbose", action="store_true", help="Print progress and show beliefs at each step")
    parser.add_argument("--log_policy_beliefs", action="store_true", help="Write per-step q_pi/policy beliefs and state factor beliefs to JSONL")
    parser.add_argument("--policy_top_k", type=int, default=5, help="Top-k policies to include in policy belief log")
    parser.add_argument("--policy_full_q_pi", action="store_true", help="Include full q_pi vector in policy belief log")
    parser.add_argument("--log_state_beliefs", action="store_true", help="Include state factor beliefs (qs) in belief log (enabled by default when --log_policy_beliefs is set)")
    parser.add_argument("--no_log_state_beliefs", action="store_true", help="Disable state factor beliefs logging even when --log_policy_beliefs is set")
    
    args = parser.parse_args()
    
    # Create log directory
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV log file
    csv_path = log_dir / f"two_independent_agents_{timestamp}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "step", "agent0_action", "agent0_action_name",
        "agent1_action", "agent1_action_name", "reward", "cumulative_reward",
        "terminated", "truncated"
    ])
    csv_writer.writeheader()

    # Optional: create policy belief log (JSONL)
    policy_log_fh = None
    policy_log_path = None
    if args.log_policy_beliefs:
        policy_log_path = log_dir / f"two_independent_agents_policy_beliefs_{timestamp}.jsonl"
        policy_log_fh = open(policy_log_path, "w")
    
    # Create environment
    if args.verbose:
        print(f"Creating Overcooked environment: layout={args.layout}, horizon={args.max_steps}")
    
    env = OvercookedMultiAgentEnv(
        config={"layout": args.layout, "horizon": args.max_steps}
    )

    # Attach logging config to env (simple way to thread into run_episode without refactor)
    env._policy_log_fh = policy_log_fh
    env._policy_log_top_k = int(args.policy_top_k)
    env._policy_log_full_q_pi = bool(args.policy_full_q_pi)
    # Default to True if logging is enabled (unless explicitly disabled with --no_log_state_beliefs)
    env._policy_log_state_beliefs = not args.no_log_state_beliefs if policy_log_fh is not None else False
    env._verbose = args.verbose  # Attach verbose flag to env for use in run_episode
    
    # Create agents
    if args.verbose:
        print("Creating two independent Active Inference agents...")
    
    agent0 = create_independent_agent(agent_idx=0, seed=args.seed)
    agent1 = create_independent_agent(agent_idx=1, seed=args.seed if args.seed is None else args.seed + 1000)
    
    # Run episodes
    if args.verbose:
        print(f"\n{'='*80}")
        print(f"Running {args.num_episodes} episodes")
        print(f"{'='*80}\n")
    
    episode_results = []
    
    for episode in range(args.num_episodes):
        result = run_episode(
            env, agent0, agent1, episode, args.max_steps,
            csv_writer=csv_writer, seed=args.seed
        )
        episode_results.append(result)
        
        if args.verbose and (episode + 1) % 10 == 0:
            avg_reward = np.mean([r["reward"] for r in episode_results[-10:]])
            avg_length = np.mean([r["length"] for r in episode_results[-10:]])
            success_rate = np.mean([r["success"] for r in episode_results[-10:]])
            print(f"Episode {episode + 1}/{args.num_episodes}: "
                  f"Reward={avg_reward:.2f}, Length={avg_length:.1f}, Success={success_rate:.2%}")
    
    csv_file.close()
    if policy_log_fh is not None:
        policy_log_fh.close()
    
    # Calculate statistics
    total_rewards = [r["reward"] for r in episode_results]
    episode_lengths = [r["length"] for r in episode_results]
    successes = [r["success"] for r in episode_results]
    
    stats = {
        "num_episodes": args.num_episodes,
        "max_steps_per_episode": args.max_steps,
        "layout": args.layout,
        "seed": args.seed,
        "total_reward_mean": float(np.mean(total_rewards)),
        "total_reward_std": float(np.std(total_rewards)),
        "total_reward_min": float(np.min(total_rewards)),
        "total_reward_max": float(np.max(total_rewards)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "total_soups_delivered": float(np.sum(total_rewards) / 20.0),  # Assuming 20 points per soup
    }
    
    # Save statistics
    stats_path = log_dir / f"two_independent_agents_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("EPISODE SUMMARY")
    print(f"{'='*80}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Layout: {args.layout}")
    print(f"\nRewards:")
    print(f"  Mean: {stats['total_reward_mean']:.2f} ± {stats['total_reward_std']:.2f}")
    print(f"  Range: [{stats['total_reward_min']:.2f}, {stats['total_reward_max']:.2f}]")
    print(f"\nEpisode Lengths:")
    print(f"  Mean: {stats['episode_length_mean']:.1f} ± {stats['episode_length_std']:.1f}")
    print(f"\nSuccess Rate: {stats['success_rate']:.2%}")
    print(f"Total Soups Delivered: {stats['total_soups_delivered']:.1f}")
    print(f"\nLogs saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Stats: {stats_path}")
    if policy_log_path is not None:
        print(f"  Beliefs log (JSONL): {policy_log_path}")
        print(f"    - State factor beliefs (qs): {'included' if args.log_state_beliefs else 'excluded'}")
        print(f"    - Policy beliefs (q_pi): {'full' if args.policy_full_q_pi else 'top-' + str(args.policy_top_k)}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
