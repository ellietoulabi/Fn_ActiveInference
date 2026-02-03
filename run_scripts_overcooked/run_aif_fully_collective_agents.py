"""
Run two Active Inference agents in Overcooked environment using FullyCollective paradigm.

FullyCollective: One centralized agent sees the full joint state and outputs
a joint action (a1, a2) encoded as a single integer. This agent controls both agents.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import json
import csv
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import Active Inference agent
from agents.ActiveInference.agent import Agent

# Import FullyCollective paradigm model
fully_collective_dir = Path(__file__).parent.parent / "generative_models" / "MA_ActiveInference" / "Overcooked" / "cramped_room" / "FullyCollective"
sys.path.insert(0, str(fully_collective_dir.parent))
from FullyCollective import A_fn, B_fn, C_fn, D_fn, model_init, env_utils

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


def create_fully_collective_agent(seed=None):
    """
    Create a fully collective Active Inference agent.
    
    This agent sees the full joint state and outputs joint actions.
    
    Args:
        seed: Random seed for reproducibility
    
    Returns:
        Agent instance
    """
    if seed is not None:
        np.random.seed(seed)
    
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
    
    # Create agent with joint action space (0-35)
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
        actions=list(range(model_init.N_JOINT_ACTIONS)),  # Joint actions: 0-35
        gamma=4.0,  # Policy precision
        alpha=4.0,  # Action precision
        policy_len=4,
        inference_horizon=4,
        action_selection="deterministic",
        sampling_mode="full",
        inference_algorithm="VANILLA",
        num_iter=16,
        dF_tol=0.001,
    )
    
    return agent


def run_episode(env, agent, episode_num, max_steps, csv_writer=None, seed=None):
    """
    Run a single episode with a fully collective agent.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        agent: FullyCollective agent instance
        episode_num: Episode number
        max_steps: Maximum steps per episode
        csv_writer: Optional CSV writer for logging
        seed: Random seed
    
    Returns:
        Dictionary with episode results
    """
    # Reset environment
    observations, infos = env.reset()
    
    # Get initial state for D config
    state = infos["agent_0"]["state"]
    
    # Reset agent with initial configuration
    config = env_utils.get_D_config_from_mdp(env.mdp, state)
    agent.reset(config=config)
    
    episode_reward = 0.0
    episode_length = 0
    prev_reward_info = {"sparse_reward_by_agent": [0, 0]}
    
    # Track previous positions for collision detection
    prev_pos1 = None
    prev_pos2 = None
    
    for step in range(1, max_steps + 1):
        # Convert environment observations to joint model observations
        obs_model = env_utils.env_obs_to_model_obs(state, reward_info=prev_reward_info)
        
        # Store current positions before actions
        curr_pos1 = obs_model["agent1_pos"]
        curr_pos2 = obs_model["agent2_pos"]
        
        # Get joint action from centralized agent
        joint_action_idx = int(agent.step(obs_model))
        
        # Decode joint action to individual actions
        a1_idx, a2_idx = env_utils.decode_joint_action(joint_action_idx)
        
        # Print beliefs to terminal if verbose
        if hasattr(env, "_verbose") and env._verbose:
            print(f"\n{'═'*80}")
            print(f"  EPISODE {episode_num} │ STEP {step}")
            print(f"{'═'*80}")
            
            # Print observations compactly
            print(f"\n📥 OBSERVATIONS:")
            obs_str = ", ".join([f"{k}={v}" for k, v in obs_model.items()])
            print(f"   Joint: {obs_str}")
            
            # Print state factor beliefs in a compact table format
            print(f"\n🧠 STATE BELIEFS:")
            print(f"   {'Factor':<25} {'Belief':<30}")
            print(f"   {'─'*25} {'─'*30}")
            
            qs = agent.get_state_beliefs()
            
            for factor in agent.state_factors:
                probs = qs[factor]
                map_state = int(np.argmax(probs))
                max_prob = float(np.max(probs))
                entropy = -np.sum(probs * np.log(probs + 1e-16))
                belief_str = f"{map_state} ({max_prob:.2f}) H={entropy:.2f}"
                
                print(f"   {factor:<25} {belief_str:<30}")
            
            # Print policy beliefs in a compact format
            print(f"\n🎯 POLICY BELIEFS:")
            q_pi = agent.get_policy_posterior()
            top_policies = agent.get_top_policies(top_k=3)
            policy_entropy = -np.sum(q_pi * np.log(q_pi + 1e-16))
            
            print(f"   Centralized Agent (entropy: {policy_entropy:.3f}):")
            for rank, (pol, prob, idx) in enumerate(top_policies, 1):
                # Decode first action of policy to show joint action
                if len(pol) > 0:
                    ja1, ja2 = env_utils.decode_joint_action(int(pol[0]))
                    pol_str = f"{ACTION_NAMES.get(ja1, str(ja1))[:1]}+{ACTION_NAMES.get(ja2, str(ja2))[:1]}"
                    if len(pol) > 1:
                        ja1_2, ja2_2 = env_utils.decode_joint_action(int(pol[1]))
                        pol_str += f"→{ACTION_NAMES.get(ja1_2, str(ja1_2))[:1]}+{ACTION_NAMES.get(ja2_2, str(ja2_2))[:1]}"
                else:
                    pol_str = "N/A"
                bar = "█" * int(prob * 20)  # Visual bar
                print(f"      #{rank} [{pol_str:>10}] {bar:<20} {prob:.3f}")
            
            # Print selected joint action prominently
            print(f"\n⚡ JOINT ACTION:")
            print(f"   Joint action index: {joint_action_idx}")
            print(f"   Agent 0 → {ACTION_NAMES.get(a1_idx, 'UNKNOWN'):<10} [{a1_idx}]")
            print(f"   Agent 1 → {ACTION_NAMES.get(a2_idx, 'UNKNOWN'):<10} [{a2_idx}]")
            print(f"{'─'*80}")

        # Optional: log policy posteriors (q_pi) and state factor beliefs (qs) each step
        if hasattr(env, "_policy_log_fh") and env._policy_log_fh is not None:
            def _serialize_top_policies(agent, top_k):
                top = agent.get_top_policies(top_k=top_k)
                return [
                    {
                        "policy_idx": int(idx),
                        "policy": [int(a) for a in pol],
                        "joint_actions": [env_utils.decode_joint_action(int(a)) for a in pol],
                        "prob": float(prob)
                    }
                    for (pol, prob, idx) in top
                ]
            
            def _serialize_state_beliefs(agent):
                """Serialize state factor beliefs (qs) to JSON-serializable format."""
                qs = agent.get_state_beliefs()
                return {
                    factor: {
                        "probabilities": [float(p) for p in qs[factor]],
                        "map_state": int(np.argmax(qs[factor])),
                        "entropy": float(-np.sum(qs[factor] * np.log(qs[factor] + 1e-16)))
                    }
                    for factor in agent.state_factors
                }

            top_k = getattr(env, "_policy_log_top_k", 5)
            include_full = bool(getattr(env, "_policy_log_full_q_pi", True))
            include_state_beliefs = bool(getattr(env, "_policy_log_state_beliefs", True))

            rec = {
                "episode": int(episode_num),
                "step": int(step),
                "obs": {k: int(v) for k, v in obs_model.items()},
                "joint_action": int(joint_action_idx),
                "action0": int(a1_idx),
                "action1": int(a2_idx),
                "action0_name": ACTION_NAMES.get(a1_idx, str(a1_idx)),
                "action1_name": ACTION_NAMES.get(a2_idx, str(a2_idx)),
                "top_policies": _serialize_top_policies(agent, top_k),
            }
            
            # Add full policy posterior if requested
            if include_full:
                rec["q_pi"] = [float(x) for x in agent.get_policy_posterior()]
            
            # Add state factor beliefs if requested
            if include_state_beliefs:
                rec["state_beliefs"] = _serialize_state_beliefs(agent)

            env._policy_log_fh.write(json.dumps(rec) + "\n")
            env._policy_log_fh.flush()
        
        # Step environment (expects action indices 0-5 for each agent)
        actions = {
            "agent_0": a1_idx,
            "agent_1": a2_idx
        }
        
        observations, rewards, terminated, truncated, infos = env.step(actions)
        
        # Update state for next iteration
        state = infos["agent_0"]["state"]
        
        # Get new positions after step
        new_obs = env_utils.env_obs_to_model_obs(state, reward_info=prev_reward_info)
        new_pos1 = new_obs["agent1_pos"]
        new_pos2 = new_obs["agent2_pos"]
        
        # Detect collisions (action attempted but position didn't change)
        collision0 = False
        collision1 = False
        if prev_pos1 is not None:
            if a1_idx in [0, 1, 2, 3]:  # Movement action
                if curr_pos1 == new_pos1:
                    collision0 = True
        if prev_pos2 is not None:
            if a2_idx in [0, 1, 2, 3]:  # Movement action
                if curr_pos2 == new_pos2:
                    collision1 = True
        
        # Update previous positions for next iteration
        prev_pos1 = new_pos1
        prev_pos2 = new_pos2
        
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
                "joint_action": joint_action_idx,
                "agent0_action": a1_idx,
                "agent0_action_name": ACTION_NAMES.get(a1_idx, str(a1_idx)),
                "agent1_action": a2_idx,
                "agent1_action_name": ACTION_NAMES.get(a2_idx, str(a2_idx)),
                "reward": rewards["agent_0"],
                "cumulative_reward": episode_reward,
                "terminated": terminated["__all__"],
                "truncated": truncated["__all__"],
            })
        
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
    parser = argparse.ArgumentParser(description="Run fully collective Active Inference agent in Overcooked")
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
    csv_path = log_dir / f"fully_collective_agents_{timestamp}.csv"
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=[
        "episode", "step", "joint_action", "agent0_action", "agent0_action_name",
        "agent1_action", "agent1_action_name", "reward", "cumulative_reward",
        "terminated", "truncated"
    ])
    csv_writer.writeheader()

    # Optional: create policy belief log (JSONL)
    policy_log_fh = None
    policy_log_path = None
    if args.log_policy_beliefs:
        policy_log_path = log_dir / f"fully_collective_agents_policy_beliefs_{timestamp}.jsonl"
        policy_log_fh = open(policy_log_path, "w")
    
    # Create environment
    if args.verbose:
        print(f"Creating Overcooked environment: layout={args.layout}, horizon={args.max_steps}")
    
    env = OvercookedMultiAgentEnv(
        config={"layout": args.layout, "horizon": args.max_steps}
    )

    # Attach logging config to env
    env._policy_log_fh = policy_log_fh
    env._policy_log_top_k = int(args.policy_top_k)
    env._policy_log_full_q_pi = bool(args.policy_full_q_pi)
    env._policy_log_state_beliefs = not args.no_log_state_beliefs if policy_log_fh is not None else False
    env._verbose = args.verbose
    
    # Create centralized agent
    if args.verbose:
        print("Creating fully collective Active Inference agent...")
    
    agent = create_fully_collective_agent(seed=args.seed)
    
    # Run episodes
    if args.verbose:
        print(f"\n{'='*80}")
        print(f"Running {args.num_episodes} episodes")
        print(f"{'='*80}\n")
    
    episode_results = []
    
    for episode in range(args.num_episodes):
        result = run_episode(
            env, agent, episode, args.max_steps,
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
    stats_path = log_dir / f"fully_collective_agents_stats_{timestamp}.json"
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
