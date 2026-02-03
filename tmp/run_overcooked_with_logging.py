"""
Run agents on Overcooked environment with detailed step-by-step logging.

This script can run trained PPO agents from checkpoints or random agents,
and logs detailed information about each step including:
- Actions taken by each agent
- Rewards received
- State visualization (map)
- Player positions and objects held
- Other important state information
"""

import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
import os

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np

# Import RLlib for loading trained agents
try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.algorithm import Algorithm
    import ray
except ImportError:
    print("Warning: Ray RLlib not installed. Can only run random agents.")
    ray = None
    Algorithm = None

# Import environment (this sets up the path for overcooked_ai imports)
from environments.overcooked_ma_gym import OvercookedMultiAgentEnv

# Now we can import Action
from overcooked_ai_py.mdp.actions import Action


def get_state_info(env, state):
    """Extract detailed state information from Overcooked state."""
    info = {
        "player_positions": [],
        "player_objects": [],
        "pot_states": [],
        "onion_dispensers": [],
        "tomato_dispensers": [],
        "serving_locations": [],
        "terrain": None,
        "map_string": None
    }
    
    # Get player positions and objects
    try:
        if hasattr(state, 'players'):
            for player in state.players:
                try:
                    pos = player.position
                    held_object = None
                    if player.has_object():
                        try:
                            obj = player.get_object()
                            held_object = {
                                "name": obj.name if hasattr(obj, 'name') else str(type(obj).__name__),
                                "is_cooked": getattr(obj, 'is_cooked', False),
                                "is_ready": getattr(obj, 'is_ready', False),
                            }
                        except Exception:
                            held_object = {"name": "unknown"}
                    
                    info["player_positions"].append({
                        "x": int(pos[0]),
                        "y": int(pos[1])
                    })
                    info["player_objects"].append(held_object)
                except Exception:
                    pass
    except Exception:
        pass
    
    # Get pot states
    try:
        if hasattr(state, 'pot_states') and state.pot_states:
            for pot_pos, pot_state in state.pot_states.items():
                try:
                    info["pot_states"].append({
                        "position": {"x": int(pot_pos[0]), "y": int(pot_pos[1])},
                        "state": str(pot_state) if pot_state else "empty"
                    })
                except Exception:
                    pass
    except Exception:
        pass
    
    # Get object locations
    try:
        if hasattr(state, 'objects') and state.objects:
            for obj_pos, obj in state.objects.items():
                try:
                    obj_info = {
                        "position": {"x": int(obj_pos[0]), "y": int(obj_pos[1])},
                        "name": obj.name if hasattr(obj, 'name') else str(type(obj).__name__)
                    }
                    if hasattr(obj, 'is_cooked'):
                        obj_info["is_cooked"] = obj.is_cooked
                    if hasattr(obj, 'is_ready'):
                        obj_info["is_ready"] = obj.is_ready
                    # Note: This would overwrite previous objects at same position
                    # Consider using a list if multiple objects can be at same position
                except Exception:
                    pass
    except Exception:
        pass
    
    # Get terrain and map visualization
    if hasattr(env, 'mdp') and env.mdp:
        try:
            if hasattr(env.mdp, 'terrain_mtx') and env.mdp.terrain_mtx is not None:
                terrain = env.mdp.terrain_mtx
                if hasattr(terrain, 'tolist'):
                    info["terrain"] = terrain.tolist()
                elif isinstance(terrain, (list, tuple)):
                    info["terrain"] = list(terrain)
                else:
                    info["terrain"] = str(terrain)
        except Exception:
            pass
        
        try:
            # Get map string representation
            info["map_string"] = env.mdp.state_string(state)
        except Exception:
            pass
    
    return info


def action_to_string(action_idx):
    """Convert action index to human-readable string."""
    action_names = {
        0: "NORTH",
        1: "SOUTH", 
        2: "EAST",
        3: "WEST",
        4: "STAY",
        5: "INTERACT"
    }
    return action_names.get(action_idx, f"UNKNOWN({action_idx})")


def run_episode_with_logging(env, agents, episode_num, log_dir, verbose=True):
    """
    Run a single episode with detailed logging.
    
    Args:
        env: OvercookedMultiAgentEnv instance
        agents: Dict mapping agent_id to agent (or None for random)
        episode_num: Episode number
        log_dir: Directory to save logs
        verbose: Whether to print to console
    """
    # Reset environment
    observations, infos = env.reset()
    
    episode_log = {
        "episode": episode_num,
        "layout": env.layout,
        "horizon": env.horizon,
        "steps": []
    }
    
    step_count = 0
    total_reward = 0.0
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Episode {episode_num}")
        print(f"{'='*80}")
        print(f"Layout: {env.layout}")
        print(f"Initial state:")
        if hasattr(env, 'mdp'):
            try:
                print(env.mdp.state_string(env.base_env.state))
            except:
                pass
    
    while step_count < env.horizon:
        # Get actions from agents
        actions = {}
        for agent_id in ["agent_0", "agent_1"]:
            if agents and agent_id in agents and agents[agent_id] is not None:
                # Use trained agent
                obs = observations[agent_id]
                if hasattr(agents[agent_id], 'compute_single_action'):
                    # RLlib agent
                    action = agents[agent_id].compute_single_action(obs, explore=False)
                    if isinstance(action, tuple):
                        action = action[0]
                    actions[agent_id] = int(action)
                else:
                    # Custom agent with choose_action method
                    actions[agent_id] = agents[agent_id].choose_action(obs)
            else:
                # Random action
                actions[agent_id] = env.action_space[agent_id].sample()
        
        # Get state before step
        state_before = env.base_env.state
        state_info_before = get_state_info(env, state_before)
        
        # Step environment
        observations_next, rewards, terminated, truncated, infos = env.step(actions)
        
        # Get state after step
        state_after = env.base_env.state
        state_info_after = get_state_info(env, state_after)
        
        # Log step information
        step_log = {
            "step": step_count,
            "actions": {
                "agent_0": {
                    "action_idx": int(actions["agent_0"]),
                    "action_name": action_to_string(actions["agent_0"])
                },
                "agent_1": {
                    "action_idx": int(actions["agent_1"]),
                    "action_name": action_to_string(actions["agent_1"])
                }
            },
            "rewards": {
                "agent_0": float(rewards["agent_0"]),
                "agent_1": float(rewards["agent_1"]),
                "total": float(rewards["agent_0"])  # Shared reward
            },
            "state_before": state_info_before,
            "state_after": state_info_after,
            "terminated": terminated["__all__"],
            "truncated": truncated["__all__"]
        }
        
        # Add map string if available
        if hasattr(env, 'mdp'):
            try:
                step_log["map_string_before"] = env.mdp.state_string(state_before)
                step_log["map_string_after"] = env.mdp.state_string(state_after)
            except:
                pass
        
        episode_log["steps"].append(step_log)
        total_reward += rewards["agent_0"]
        step_count += 1
        
        # Print step info if verbose
        if verbose:
            print(f"\nStep {step_count}:")
            print(f"  Actions: agent_0={action_to_string(actions['agent_0'])}, "
                  f"agent_1={action_to_string(actions['agent_1'])}")
            print(f"  Reward: {rewards['agent_0']:.4f}")
            print(f"  Agent 0 position: {state_info_after['player_positions'][0]}")
            print(f"  Agent 1 position: {state_info_after['player_positions'][1]}")
            if state_info_after['player_objects'][0]:
                print(f"  Agent 0 holding: {state_info_after['player_objects'][0]}")
            if state_info_after['player_objects'][1]:
                print(f"  Agent 1 holding: {state_info_after['player_objects'][1]}")
            if step_log.get("map_string_after"):
                print(f"  Map after step:")
                print(step_log["map_string_after"])
        
        # Check if done
        if terminated["__all__"] or truncated["__all__"]:
            if verbose:
                print(f"\nEpisode finished at step {step_count}")
                if "episode" in infos["agent_0"]:
                    ep_info = infos["agent_0"]["episode"]
                    print(f"  Episode sparse reward: {ep_info.get('ep_sparse_r', 'N/A')}")
                    print(f"  Episode shaped reward: {ep_info.get('ep_shaped_r', 'N/A')}")
            break
        
        observations = observations_next
    
    episode_log["total_reward"] = float(total_reward)
    episode_log["episode_length"] = step_count
    
    # Save episode log
    log_file = log_dir / f"episode_{episode_num:04d}.json"
    with open(log_file, 'w') as f:
        json.dump(episode_log, f, indent=2)
    
    # Also save human-readable text log
    text_log_file = log_dir / f"episode_{episode_num:04d}.txt"
    with open(text_log_file, 'w') as f:
        f.write(f"Episode {episode_num}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Layout: {episode_log['layout']}\n")
        f.write(f"Total Reward: {episode_log['total_reward']:.4f}\n")
        f.write(f"Episode Length: {episode_log['episode_length']}\n")
        f.write(f"\n")
        
        for step_log in episode_log["steps"]:
            f.write(f"\nStep {step_log['step']}:\n")
            f.write(f"  Actions:\n")
            f.write(f"    Agent 0: {step_log['actions']['agent_0']['action_name']} "
                   f"({step_log['actions']['agent_0']['action_idx']})\n")
            f.write(f"    Agent 1: {step_log['actions']['agent_1']['action_name']} "
                   f"({step_log['actions']['agent_1']['action_idx']})\n")
            f.write(f"  Rewards: {step_log['rewards']['total']:.4f}\n")
            f.write(f"  Agent 0 position: {step_log['state_after']['player_positions'][0]}\n")
            f.write(f"  Agent 1 position: {step_log['state_after']['player_positions'][1]}\n")
            if step_log['state_after']['player_objects'][0]:
                f.write(f"  Agent 0 holding: {step_log['state_after']['player_objects'][0]}\n")
            if step_log['state_after']['player_objects'][1]:
                f.write(f"  Agent 1 holding: {step_log['state_after']['player_objects'][1]}\n")
            if step_log.get('map_string_after'):
                f.write(f"  Map:\n{step_log['map_string_after']}\n")
            f.write(f"\n")
    
    if verbose:
        print(f"\nEpisode {episode_num} completed:")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Episode length: {step_count}")
        print(f"  Logs saved to: {log_file} and {text_log_file}")
    
    return episode_log


def load_ppo_agents(checkpoint_path):
    """Load trained PPO agents from checkpoint."""
    if ray is None or Algorithm is None:
        raise ImportError("Ray RLlib not installed. Cannot load trained agents.")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Load algorithm from checkpoint
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Get policies
    agents = {}
    for policy_id in ["agent_0", "agent_1"]:
        if policy_id in algo.config.policies:
            agents[policy_id] = algo.get_policy(policy_id)
        else:
            # Try to get default policy
            agents[policy_id] = algo.get_policy("default_policy")
    
    return agents, algo


def main():
    parser = argparse.ArgumentParser(description="Run agents on Overcooked with detailed logging")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to PPO checkpoint (if None, uses random agents)")
    parser.add_argument("--layout", type=str, default="cramped_room", 
                       help="Layout name")
    parser.add_argument("--horizon", type=int, default=400, 
                       help="Max steps per episode")
    parser.add_argument("--num_episodes", type=int, default=1, 
                       help="Number of episodes to run")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Directory to save logs (default: logs/overcooked_<timestamp>)")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Print detailed information to console")
    parser.add_argument("--quiet", action="store_true",
                       help="Quiet mode (opposite of verbose)")
    
    args = parser.parse_args()
    
    if args.quiet:
        args.verbose = False
    
    # Create log directory
    if args.log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"logs/overcooked_{timestamp}")
    else:
        log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("OVERCOOKED AGENT RUNNING WITH DETAILED LOGGING")
    print("="*80)
    print(f"Layout: {args.layout}")
    print(f"Horizon: {args.horizon}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Log directory: {log_dir}")
    
    # Load agents if checkpoint provided
    agents = None
    algo = None
    if args.checkpoint:
        print(f"\nLoading agents from checkpoint: {args.checkpoint}")
        try:
            agents, algo = load_ppo_agents(args.checkpoint)
            print("✓ Agents loaded successfully")
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            print("  Falling back to random agents")
            agents = None
    else:
        print("\nUsing random agents")
    
    # Create environment
    env = OvercookedMultiAgentEnv(
        layout=args.layout,
        horizon=args.horizon
    )
    
    # Run episodes
    all_episodes = []
    for episode in range(args.num_episodes):
        episode_log = run_episode_with_logging(
            env, agents, episode, log_dir, verbose=args.verbose
        )
        all_episodes.append(episode_log)
    
    # Save summary
    summary = {
        "layout": args.layout,
        "horizon": args.horizon,
        "num_episodes": args.num_episodes,
        "checkpoint": args.checkpoint,
        "episodes": [
            {
                "episode": ep["episode"],
                "total_reward": ep["total_reward"],
                "episode_length": ep["episode_length"]
            }
            for ep in all_episodes
        ],
        "average_reward": float(np.mean([ep["total_reward"] for ep in all_episodes])),
        "average_length": float(np.mean([ep["episode_length"] for ep in all_episodes]))
    }
    
    summary_file = log_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Average reward: {summary['average_reward']:.4f}")
    print(f"  Average length: {summary['average_length']:.1f}")
    print(f"  Summary saved to: {summary_file}")
    print(f"  All logs saved to: {log_dir}")
    print(f"{'='*80}\n")
    
    # Cleanup
    env.close()
    if algo:
        algo.stop()
    if ray and ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
