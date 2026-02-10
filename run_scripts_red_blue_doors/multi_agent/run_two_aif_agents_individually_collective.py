"""
Run two Active Inference agents in TwoAgentRedBlueButton using the Individually
Collective paradigm.

Individually Collective: each agent is an AIF agent that thinks it's deciding for
both (shared joint model) but only takes its own action component:
full joint inference via step(), then marginalize q_pi to P(a1) or P(a2) and execute
that. Coordination via shared model structure; supports heterogeneous partners.

Runs across multiple seeds, episodes, and changing map configurations.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

import numpy as np
import csv
import json
from datetime import datetime
import argparse
from tqdm import tqdm
from environments.RedBlueButton.TwoAgentRedBlueButton import TwoAgentRedBlueButtonEnv
from generative_models.MA_ActiveInference.RedBlueButton.IndividuallyCollective import (
    A_fn, B_fn, C_fn, D_fn, model_init, env_utils
)
from agents.ActiveInference.agent import Agent


# =============================================================================
# Helper function for verbose step printing (similar to single-agent style)
# =============================================================================

def _state_belief_table(agent):
    """One line per factor: MAP, Prob, Entropy."""
    qs = agent.get_state_beliefs()
    lines = []
    for factor in agent.state_factors:
        p = np.array(qs[factor], dtype=float)
        map_idx = int(np.argmax(p))
        map_prob = float(p[map_idx])
        entropy = float(-np.sum(p * np.log(p + 1e-16)))
        lines.append(f"   {factor:<18} {map_idx:<6} {map_prob:<8.3f} {entropy:<8.3f}")
    return "\n".join(lines)


def _top_policies_bar(agent, action_names, top_k=5):
    """Top-k policies with bar for one agent."""
    lines = []
    q_pi = agent.get_policy_posterior()
    entropy = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
    lines.append(f"   Entropy: {entropy:.3f}")
    for (pol, prob, idx) in agent.get_top_policies(top_k=top_k):
        a1, a2 = env_utils.decode_joint_action(int(pol[0]))
        pol_str = f"{action_names.get(a1, str(a1))[:2]}→{action_names.get(a2, str(a2))[:2]}"
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        lines.append(f"   [{pol_str:>8}] {bar} {prob:.3f}")
    return "\n".join(lines)


def print_step_info(step, env_obs, agent1, agent2, joint_action1_idx, joint_action2_idx, action1, action2, reward, cumulative_reward, info, env, action_names, env_utils):
    """Compact step info: obs, both agents' state beliefs + policy bars, executed actions, outcome."""
    print(f"\n{'═'*80}")
    print(f"  EPISODE (step) │ STEP {step}")
    print(f"{'═'*80}")
    print("\n📥 OBSERVATIONS (joint):")
    a1_pos = env_obs.get("agent1_position", (0, 0))
    a2_pos = env_obs.get("agent2_position", (0, 0))
    print(f"   agent1_pos={a1_pos} on_red={env_obs.get('agent1_on_red_button', 0)} on_blue={env_obs.get('agent1_on_blue_button', 0)}")
    print(f"   agent2_pos={a2_pos} on_red={env_obs.get('agent2_on_red_button', 0)} on_blue={env_obs.get('agent2_on_blue_button', 0)}")
    print(f"   red_pressed={env_obs.get('red_button_pressed', 0)} blue_pressed={env_obs.get('blue_button_pressed', 0)} game_result={env_obs.get('win_lose_neutral', 0)}")
    print("\n🧠 AGENT 1 STATE BELIEFS:")
    print(f"   {'Factor':<18} {'MAP':<6} {'Prob':<8} {'Entropy':<8}")
    print(f"   {'─'*18} {'─'*6} {'─'*8} {'─'*8}")
    print(_state_belief_table(agent1))
    print("\n🧠 AGENT 2 STATE BELIEFS:")
    print(f"   {'Factor':<18} {'MAP':<6} {'Prob':<8} {'Entropy':<8}")
    print(f"   {'─'*18} {'─'*6} {'─'*8} {'─'*8}")
    print(_state_belief_table(agent2))
    print("\n🎯 AGENT 1 POLICY (top 5):")
    print(_top_policies_bar(agent1, action_names, 5))
    print("\n🎯 AGENT 2 POLICY (top 5):")
    print(_top_policies_bar(agent2, action_names, 5))
    print("\n⚡ EXECUTED (each agent's own component):")
    print(f"   Agent 1 → {action_names.get(action1, str(action1))} [{action1}]   Agent 2 → {action_names.get(action2, str(action2))} [{action2}]")
    print(f"\n📈 OUTCOME: reward={reward:+.3f}  cumulative={cumulative_reward:+.3f}  result={info.get('result', 'neutral')}")
    if info.get("button_just_pressed"):
        print(f"   Button pressed: {info['button_just_pressed']} by {info.get('button_pressed_by', '')}")
    print(f"{'─'*80}")


# =============================================================================
# Helper functions for logging
# =============================================================================

def serialize_beliefs(qs_dict):
    """
    Serialize state beliefs to a readable string format (similar to single-agent style).
    
    Format: "factor1=idx1:prob1;factor2=idx2:prob2;..." where idx is MAP state and prob is its probability.
    
    Args:
        qs_dict: dict mapping factor names to belief arrays
    
    Returns:
        String representation: "agent_pos=0:0.95;red_button_pos=3:0.12;..."
    """
    parts = []
    for factor, belief in sorted(qs_dict.items()):
        belief_arr = np.array(belief)
        map_idx = int(np.argmax(belief_arr))
        map_prob = float(belief_arr[map_idx])
        parts.append(f"{factor}={map_idx}:{map_prob:.4f}")
    return ";".join(parts)


def serialize_policies(q_pi, top_k=3):
    """
    Serialize policy posterior to a readable string format (similar to single-agent style).
    
    Format: "policy_idx1:prob1;policy_idx2:prob2;..." showing top-k policies.
    
    Args:
        q_pi: array of policy posterior probabilities
        top_k: number of top policies to include
    
    Returns:
        String representation: "15:0.205;9:0.057;11:0.054;..."
    """
    q_pi_arr = np.array(q_pi)
    top_indices = np.argsort(q_pi_arr)[-top_k:][::-1]  # Top k, highest first
    parts = []
    for idx in top_indices:
        prob = float(q_pi_arr[idx])
        parts.append(f"{idx}:{prob:.4f}")
    return ";".join(parts)


def serialize_state_beliefs_for_json(agent):
    """Full state factor beliefs for JSONL."""
    qs = agent.get_state_beliefs()
    out = {}
    for factor in agent.state_factors:
        p = np.array(qs[factor], dtype=float)
        out[factor] = {
            "probabilities": p.tolist(),
            "map_state": int(np.argmax(p)),
            "entropy": float(-np.sum(p * np.log(p + 1e-16))),
        }
    return out


def serialize_top_policies_for_json(agent, top_k):
    """Top-k policies with decoded joint actions for JSONL."""
    top = agent.get_top_policies(top_k=top_k)
    return [
        {
            "policy_idx": int(idx),
            "policy": [int(a) for a in pol],
            "joint_actions": [env_utils.decode_joint_action(int(a)) for a in pol],
            "prob": float(prob),
        }
        for (pol, prob, idx) in top
    ]


def _validate_ma_model(env):
    """
    Sanity checks to catch common A/B/C/D interface mismatches early.
    Raises AssertionError with a helpful message if something is inconsistent.
    """
    # Basic key consistency
    state_factors = list(model_init.states.keys())
    obs_modalities = list(model_init.observations.keys())

    d_config = env_utils.get_D_config_from_env(env)
    D = D_fn(d_config)
    assert set(D.keys()) == set(state_factors), (
        f"D_fn keys mismatch.\nExpected: {state_factors}\nGot: {sorted(D.keys())}"
    )
    for f in state_factors:
        assert len(D[f]) == len(model_init.states[f]), f"D[{f}] has wrong length"
        s = float(np.sum(D[f]))
        assert np.isfinite(s) and abs(s - 1.0) < 1e-6, f"D[{f}] not normalized (sum={s})"

    # A_fn output shape / normalization
    map_state = {f: int(np.argmax(np.array(D[f]))) for f in state_factors}
    A = A_fn(map_state)
    assert set(A.keys()) == set(obs_modalities), (
        f"A_fn keys mismatch.\nExpected: {obs_modalities}\nGot: {sorted(A.keys())}"
    )
    for m in obs_modalities:
        assert len(A[m]) == len(model_init.observations[m]), f"A[{m}] has wrong length"
        s = float(np.sum(A[m]))
        assert np.isfinite(s) and abs(s - 1.0) < 1e-6, f"A[{m}] not normalized (sum={s})"

    # B_fn output keys / normalization (joint action)
    joint_action = env_utils.encode_joint_action(0, 0)
    qs_next = B_fn(D, action=joint_action, width=env.width, height=env.height)
    assert set(qs_next.keys()) == set(state_factors), (
        f"B_fn keys mismatch.\nExpected: {state_factors}\nGot: {sorted(qs_next.keys())}"
    )
    for f in state_factors:
        assert len(qs_next[f]) == len(D[f]), f"B[{f}] has wrong length"
        s = float(np.sum(qs_next[f]))
        assert np.isfinite(s) and abs(s - 1.0) < 1e-6, f"B[{f}] not normalized (sum={s})"

    # C_fn interface sanity
    dummy_obs = {m: 0 for m in obs_modalities}
    prefs = C_fn(dummy_obs)
    assert isinstance(prefs, dict), "C_fn must return a dict"


def create_aif_agent(agent_id, env):
    """Create an Active Inference agent using the joint model."""
    
    state_factors = list(model_init.states.keys())
    state_sizes = {factor: len(values) for factor, values in model_init.states.items()}
    
    # Joint action space: 36 actions (6x6)
    joint_actions = list(range(model_init.N_JOINT_ACTIONS))
    
    agent = Agent(
        A_fn=A_fn,
        B_fn=B_fn,
        C_fn=C_fn,
        D_fn=D_fn,
        state_factors=state_factors,
        state_sizes=state_sizes,
        observation_labels=model_init.observations,
        env_params={'width': 3, 'height': 3},
        observation_state_dependencies=model_init.observation_state_dependencies,
        actions=joint_actions,
        policy_len=2,  # 36 policies; len=2 for two-step lookahead (fair comparison across paradigms)
        gamma=2.0,  # Policy precision
        alpha=1.0,  # Action precision
        num_iter=16,
    )
    
    # Get config from environment (same for both agents in this paradigm)
    d_config = env_utils.get_D_config_from_env(env)
    agent.reset(config=d_config)
    
    return agent


def reset_agent_beliefs(agent, env):
    """Reset agent beliefs for new episode (preserving button position beliefs)."""
    d_config = env_utils.get_D_config_from_env(env)
    # Preserve button-position beliefs across episodes within the same config.
    agent.reset(config=d_config, keep_factors=['red_button_pos', 'blue_button_pos'])


def run_episode(
    env,
    agent1,
    agent2,
    episode_num,
    max_steps=50,
    verbose=False,
    csv_writer=None,
    episode_progress=False,
    show_beliefs=False,
    show_policies=False,
    config_idx=0,
    policy_log_fh=None,
    log_policy_top_k=5,
    log_full_q_pi=False,
    log_state_beliefs=False,
    print_steps=False,
):
    """Run one episode with two AIF agents using IndividuallyCollective paradigm."""
    obs, _ = env.reset()
    reset_agent_beliefs(agent1, env)
    reset_agent_beliefs(agent2, env)

    if verbose:
        print(f"\n{'='*80}")
        print(f"EPISODE {episode_num}")
        print(f"{'='*80}")
        print(f"Environment: Red at {env.red_button}, Blue at {env.blue_button}")
        print("\nInitial state:")
        env.render()

    episode_reward = 0.0
    outcome = "timeout"
    step = 0
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "PRESS", 5: "NOOP"}
    step_iter = range(1, max_steps + 1)
    if episode_progress and not verbose:
        step_iter = tqdm(step_iter, desc=f"Ep {episode_num}", leave=False, position=2, unit="step")

    for step in step_iter:
        if isinstance(obs, dict) and "agent_0" in obs:
            joint_obs = env_utils.merge_env_obs_for_collective(obs)
        else:
            joint_obs = obs
        # Perspective observations: each agent sees (my_pos, other_pos) so they get different inputs and can form different beliefs.
        model_obs_1 = env_utils.env_obs_to_model_obs_for_agent(joint_obs, env.width, agent_id=1)
        model_obs_2 = env_utils.env_obs_to_model_obs_for_agent(joint_obs, env.width, agent_id=2)

        joint_action1_idx = int(agent1.step(model_obs_1))
        joint_action2_idx = int(agent2.step(model_obs_2))
        # With perspective obs, each agent's policy is over (my_action, other_action), so "my" is always component 0.
        action1 = int(env_utils.sample_my_component(agent1.get_policy_posterior(), agent1.policies, 0))
        action2 = int(env_utils.sample_my_component(agent2.get_policy_posterior(), agent2.policies, 0))
        actions = (action1, action2)

        grid = env.render(mode="silent")
        map_str = "|".join(["".join(row) for row in grid])

        next_obs, reward, terminated, truncated, info = env.step(actions)
        if isinstance(reward, dict):
            reward = reward.get("agent_0", reward.get("agent_1", 0.0))
        reward = float(reward)
        done = terminated or truncated
        episode_reward += reward

        if verbose:
            print_step_info(
                step, joint_obs, agent1, agent2, joint_action1_idx, joint_action2_idx,
                action1, action2, reward, episode_reward, info, env, action_names, env_utils,
            )
        elif print_steps:
            a1n = action_names.get(action1, str(action1))
            a2n = action_names.get(action2, str(action2))
            print(f"  Ep {episode_num} Step {step}: A1={a1n} A2={a2n} r={reward:+.2f} cum={episode_reward:+.2f} {info.get('result', 'neutral')}")

        if csv_writer is not None:
            # Log in world frame (same as agent1's observation)
            row = {
                "episode": episode_num,
                "step": step,
                "config_idx": config_idx,
                "agent1_pos": model_obs_1["agent1_pos"],
                "agent2_pos": model_obs_1["agent2_pos"],
                "agent1_on_red_button": model_obs_1["agent1_on_red_button"],
                "agent1_on_blue_button": model_obs_1["agent1_on_blue_button"],
                "agent2_on_red_button": model_obs_1["agent2_on_red_button"],
                "agent2_on_blue_button": model_obs_1["agent2_on_blue_button"],
                "red_button_state": model_obs_1["red_button_state"],
                "blue_button_state": model_obs_1["blue_button_state"],
                "game_result": model_obs_1["game_result"],
                "joint_action1": joint_action1_idx,
                "joint_action2": joint_action2_idx,
                "action1": action1,
                "action1_name": action_names.get(action1, str(action1)),
                "action2": action2,
                "action2_name": action_names.get(action2, str(action2)),
                "map": map_str,
                "reward": reward,
                "cumulative_reward": episode_reward,
                "terminated": terminated,
                "truncated": truncated,
                "result": info.get("result", "neutral"),
                "button_pressed": info.get("button_just_pressed", ""),
                "pressed_by": info.get("button_pressed_by", ""),
                "agent1_beliefs": serialize_beliefs(agent1.get_state_beliefs()),
                "agent1_policies": serialize_policies(agent1.get_policy_posterior()),
                "agent2_beliefs": serialize_beliefs(agent2.get_state_beliefs()),
                "agent2_policies": serialize_policies(agent2.get_policy_posterior()),
            }
            csv_writer.writerow(row)

        if policy_log_fh is not None:
            log_entry = {
                "episode": episode_num,
                "step": step,
                "obs": dict(model_obs_1),  # world frame for log
                "action": {"action1": action1, "action2": action2},
                "agent1": {"top_policies": serialize_top_policies_for_json(agent1, log_policy_top_k)},
                "agent2": {"top_policies": serialize_top_policies_for_json(agent2, log_policy_top_k)},
            }
            if log_full_q_pi:
                log_entry["agent1"]["q_pi"] = agent1.get_policy_posterior().tolist()
                log_entry["agent2"]["q_pi"] = agent2.get_policy_posterior().tolist()
            if log_state_beliefs:
                log_entry["agent1"]["state_beliefs"] = serialize_state_beliefs_for_json(agent1)
                log_entry["agent2"]["state_beliefs"] = serialize_state_beliefs_for_json(agent2)
            policy_log_fh.write(json.dumps(log_entry) + "\n")
            policy_log_fh.flush()

        obs = next_obs
        if done:
            outcome = info.get("result", "neutral")
            break

    if verbose:
        status = "✅ WIN" if outcome == "win" else "❌ FAIL"
        print(f"\nResult: {status} - {outcome} (steps: {step}, total reward: {episode_reward:+.2f})")

    return {
        "outcome": outcome,
        "reward": episode_reward,
        "steps": step,
        "success": outcome == "win",
    }


def generate_random_config(rng, grid_width=3, grid_height=3):
    """Generate random button positions (avoiding agent start positions)."""
    available_positions = []
    for x in range(grid_width):
        for y in range(grid_height):
            if (x, y) not in [(0, 0), (grid_width-1, grid_height-1)]:
                available_positions.append((x, y))
    
    rng.shuffle(available_positions)
    red_pos = available_positions[0]
    blue_pos = available_positions[1]
    
    return {
        'red_pos': red_pos,
        'blue_pos': blue_pos,
    }


def run_seed_experiment(seed, num_episodes, episodes_per_config, max_steps,
                        verbose=False, csv_writer=None, episode_progress=False,
                        show_beliefs=False, show_policies=False,
                        policy_log_fh=None, log_policy_top_k=5, log_full_q_pi=False,                         log_state_beliefs=False,
                        print_steps=False, progress_callback=None):
    """Run experiment for a single seed."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    results = []
    configs = []
    num_configs = (num_episodes + episodes_per_config - 1) // episodes_per_config
    for _ in range(num_configs):
        configs.append(generate_random_config(rng))
    env = None
    agent1 = None
    agent2 = None
    episode_iter = range(1, num_episodes + 1)
    for episode in tqdm(episode_iter, disable=verbose, desc=f"Seed {seed}", leave=True, unit="ep", position=1):
        config_idx = (episode - 1) // episodes_per_config
        config = configs[config_idx]
        if (episode - 1) % episodes_per_config == 0 or env is None:
            env = TwoAgentRedBlueButtonEnv(
                width=3,
                height=3,
                red_button_pos=config["red_pos"],
                blue_button_pos=config["blue_pos"],
                agent1_start_pos=(0, 0),
                agent2_start_pos=(2, 2),
                max_steps=max_steps,
            )
            _validate_ma_model(env)
            agent1 = create_aif_agent(1, env)
            agent2 = create_aif_agent(2, env)
            if verbose:
                print(f"\n{'='*80}")
                print(f"SEED {seed} - CONFIG {config_idx + 1}")
                print(f"Red at {config['red_pos']}, Blue at {config['blue_pos']}")
                print(f"Episodes {episode}-{min(episode + episodes_per_config - 1, num_episodes)}")
                print(f"{'='*80}")
        result = run_episode(
            env,
            agent1,
            agent2,
            episode,
            max_steps=max_steps,
            verbose=verbose,
            csv_writer=csv_writer,
            episode_progress=episode_progress,
            show_beliefs=show_beliefs,
            show_policies=show_policies,
            config_idx=config_idx,
            policy_log_fh=policy_log_fh,
            log_policy_top_k=log_policy_top_k,
            log_full_q_pi=log_full_q_pi,
            log_state_beliefs=log_state_beliefs,
            print_steps=print_steps,
        )
        result["seed"] = seed
        result["config_idx"] = config_idx
        results.append(result)
        if progress_callback is not None:
            progress_callback(1)

        # Print progress every 100 episodes
        if episode % 100 == 0:
            recent = results[-100:]
            recent_wins = sum(1 for r in recent if r['success'])
            recent_rate = 100.0 * recent_wins / max(1, len(recent))
            print(f"  Seed {seed}, Episode {episode}/{num_episodes}: "
                  f"Last 100 win rate: {recent_rate:.1f}% ({recent_wins}/100)")
    
    return results, configs


def main():
    print("="*80)
    print("TWO ACTIVE INFERENCE AGENTS - INDIVIDUALLY COLLECTIVE PARADIGM")
    print("="*80)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds to run (if --seed not provided)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed to run (overrides --seeds if provided)")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes-per-config", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--verbose", action="store_true", help="Print full step info (beliefs, policies) each step")
    parser.add_argument("--print-steps", action="store_true", help="Print one line per step (episode, step, actions, reward, cumulative) when not using --verbose")
    parser.add_argument("--plots", action="store_true", help="Generate and save plots at the end")
    parser.add_argument(
        "--episode-progress",
        action="store_true",
        help="Show per-episode progress over steps (tqdm); off by default to keep logs light",
    )
    parser.add_argument(
        "--show-beliefs",
        action="store_true",
        help="When verbose, print top beliefs per factor each step",
    )
    parser.add_argument(
        "--show-policies",
        action="store_true",
        help="When verbose, print top policy posterior entries each step",
    )
    parser.add_argument("--log-policy-beliefs", action="store_true", help="Write JSONL with per-step policy (and optional state) beliefs for both agents")
    parser.add_argument("--policy-top-k", type=int, default=5, help="Number of top policies to log in JSONL (default: 5)")
    parser.add_argument("--log-full-q-pi", action="store_true", help="In JSONL, include full policy posterior q_pi for both agents")
    parser.add_argument("--log-state-beliefs", action="store_true", help="In JSONL, include full state beliefs per factor for both agents")
    parser.add_argument("--stats-output", type=str, default=None, help="Also write stats JSON to this path (for comparison scripts)")
    args = parser.parse_args()

    # Parameters
    # If --seed is provided, run only that seed; otherwise use --seeds
    if args.seed is not None:
        SEEDS_TO_RUN = [args.seed]
        NUM_SEEDS_FOR_FILENAME = 1  # For filename generation
    else:
        SEEDS_TO_RUN = list(range(args.seeds))
        NUM_SEEDS_FOR_FILENAME = args.seeds
    
    NUM_SEEDS = len(SEEDS_TO_RUN)
    NUM_EPISODES_PER_SEED = args.episodes
    EPISODES_PER_CONFIG = args.episodes_per_config
    MAX_STEPS = args.max_steps
    VERBOSE = args.verbose
    GENERATE_PLOTS = args.plots
    
    # Setup logging
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"two_aif_agents_individually_collective_seeds{NUM_SEEDS_FOR_FILENAME}_ep{NUM_EPISODES_PER_SEED}_{timestamp}.csv"
    csv_path = log_dir / csv_filename
    csv_fieldnames = [
        "seed", "episode", "step", "config_idx",
        "agent1_pos", "agent2_pos",
        "agent1_on_red_button", "agent1_on_blue_button",
        "agent2_on_red_button", "agent2_on_blue_button",
        "red_button_state", "blue_button_state", "game_result",
        "joint_action1", "joint_action2", "action1", "action1_name", "action2", "action2_name",
        "map", "reward", "cumulative_reward", "terminated", "truncated",
        "result", "button_pressed", "pressed_by",
        "agent1_beliefs", "agent1_policies", "agent2_beliefs", "agent2_policies",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, extrasaction="ignore")
    csv_writer.writeheader()

    jsonl_path = None
    jsonl_fh = None
    if args.log_policy_beliefs:
        jsonl_filename = f"two_aif_agents_individually_collective_seeds{NUM_SEEDS_FOR_FILENAME}_ep{NUM_EPISODES_PER_SEED}_{timestamp}_policy_log.jsonl"
        jsonl_path = log_dir / jsonl_filename
        jsonl_fh = open(jsonl_path, "w")

    print(f"\nExperiment Parameters:")
    print(f"  Seeds to run: {SEEDS_TO_RUN}")
    print(f"  Number of seeds: {NUM_SEEDS}")
    print(f"  Episodes per seed: {NUM_EPISODES_PER_SEED}")
    print(f"  Episodes per config: {EPISODES_PER_CONFIG}")
    print(f"  Max steps per episode: {MAX_STEPS}")
    print(f"  CSV log: {csv_path}")
    if jsonl_path:
        print(f"  JSONL log: {jsonl_path}")
    
    # Run experiments across all seeds
    all_results = []
    seed_summaries = []
    total_episodes = NUM_SEEDS * NUM_EPISODES_PER_SEED
    with tqdm(total=total_episodes, desc="Total", unit="ep", leave=True, position=0) as pbar:
        try:
            for seed_idx, seed in enumerate(SEEDS_TO_RUN):
                print(f"\n{'='*80}")
                print(f"SEED {seed} ({seed_idx + 1}/{NUM_SEEDS})")
                print(f"{'='*80}")

                # Add seed to CSV rows
                class SeedCSVWriter:
                    def __init__(self, writer, seed):
                        self.writer = writer
                        self.seed = seed

                    def writerow(self, row):
                        row['seed'] = self.seed
                        self.writer.writerow(row)

                seed_csv_writer = SeedCSVWriter(csv_writer, seed)

                results, configs = run_seed_experiment(
                    seed=seed,
                    num_episodes=NUM_EPISODES_PER_SEED,
                    episodes_per_config=EPISODES_PER_CONFIG,
                    max_steps=MAX_STEPS,
                    verbose=VERBOSE,
                    csv_writer=seed_csv_writer,
                    episode_progress=args.episode_progress,
                    show_beliefs=args.show_beliefs and VERBOSE,
                    show_policies=args.show_policies and VERBOSE,
                    policy_log_fh=jsonl_fh,
                    log_policy_top_k=args.policy_top_k,
                    log_full_q_pi=args.log_full_q_pi,
                    log_state_beliefs=args.log_state_beliefs,
                    print_steps=args.print_steps,
                    progress_callback=pbar.update,
                )

                all_results.extend(results)

                # Compute seed summary
                successes = sum(1 for r in results if r['success'])
                success_rate = 100 * successes / len(results)
                avg_reward = np.mean([r['reward'] for r in results])
                avg_steps = np.mean([r['steps'] for r in results])

                # Learning curve: first half vs second half
                mid = len(results) // 2
                first_half_wins = sum(1 for r in results[:mid] if r['success'])
                second_half_wins = sum(1 for r in results[mid:] if r['success'])

                seed_summaries.append({
                    'seed': seed,
                    'successes': successes,
                    'total': len(results),
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'avg_steps': avg_steps,
                    'first_half_wins': first_half_wins,
                    'second_half_wins': second_half_wins,
                })

                print(f"\nSeed {seed} Summary:")
                print(f"  Success rate: {successes}/{len(results)} ({success_rate:.1f}%)")
                print(f"  Average reward: {avg_reward:+.2f}")
                print(f"  Average steps: {avg_steps:.1f}")
                print(f"  Learning: First half {first_half_wins}/{mid}, Second half {second_half_wins}/{mid}")
        finally:
            csv_file.close()
            if jsonl_fh is not None:
                jsonl_fh.close()

    seed_summaries_serializable = [
        {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in s.items()}
        for s in seed_summaries
    ]
    stats = {
        "paradigm": "individually_collective",
        "n_seeds": NUM_SEEDS,
        "n_episodes_per_seed": NUM_EPISODES_PER_SEED,
        "episodes_per_config": EPISODES_PER_CONFIG,
        "max_steps": MAX_STEPS,
        "total_episodes": len(all_results),
        "total_successes": sum(1 for r in all_results if r["success"]),
        "success_rate": float(100 * sum(1 for r in all_results if r["success"]) / max(1, len(all_results))),
        "mean_reward": float(np.mean([r["reward"] for r in all_results])),
        "std_reward": float(np.std([r["reward"] for r in all_results])),
        "mean_steps": float(np.mean([r["steps"] for r in all_results])),
        "std_steps": float(np.std([r["steps"] for r in all_results])),
        "seed_summaries": seed_summaries_serializable,
    }
    stats_filename = f"two_aif_agents_individually_collective_seeds{NUM_SEEDS_FOR_FILENAME}_ep{NUM_EPISODES_PER_SEED}_{timestamp}_stats.json"
    stats_path = log_dir / stats_filename
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"\n✓ Logs saved:")
    print(f"    CSV:   {csv_path}")
    if jsonl_path:
        print(f"    JSONL: {jsonl_path}")
    print(f"    Stats: {stats_path}")

    # Overall summary
    print("\n" + "="*80)
    print("OVERALL RESULTS SUMMARY")
    print("="*80)
    
    total_successes = sum(s['successes'] for s in seed_summaries)
    total_episodes = sum(s['total'] for s in seed_summaries)
    overall_success_rate = 100 * total_successes / total_episodes
    overall_avg_reward = np.mean([r['reward'] for r in all_results])
    overall_avg_steps = np.mean([r['steps'] for r in all_results])
    
    print(f"\nTotal episodes: {total_episodes}")
    print(f"Total successes: {total_successes} ({overall_success_rate:.1f}%)")
    print(f"Overall average reward: {overall_avg_reward:+.2f}")
    print(f"Overall average steps: {overall_avg_steps:.1f}")
    
    print(f"\n{'Seed':<6} {'Success Rate':<18} {'Avg Reward':<12} {'1st Half':<10} {'2nd Half':<10}")
    print("-" * 65)
    for s in seed_summaries:
        mid = s['total'] // 2
        print(f"{s['seed']:<6} {s['successes']:>4}/{s['total']:<4} ({s['success_rate']:>5.1f}%)  "
              f"{s['avg_reward']:>+7.2f}     "
              f"{s['first_half_wins']:>3}/{mid:<5}   "
              f"{s['second_half_wins']:>3}/{mid:<5}")
    
    # Learning improvement
    total_first = sum(s['first_half_wins'] for s in seed_summaries)
    total_second = sum(s['second_half_wins'] for s in seed_summaries)
    mid = NUM_EPISODES_PER_SEED // 2
    total_episodes_first_half = NUM_SEEDS * mid
    total_episodes_second_half = NUM_SEEDS * mid
    print(f"\nLearning Progress:")
    if total_episodes_first_half > 0:
        print(f"  First half: {total_first}/{total_episodes_first_half} ({100*total_first/total_episodes_first_half:.1f}%)")
    else:
        print(f"  First half: {total_first}/0 (N/A)")
    if total_episodes_second_half > 0:
        print(f"  Second half: {total_second}/{total_episodes_second_half} ({100*total_second/total_episodes_second_half:.1f}%)")
    else:
        print(f"  Second half: {total_second}/0 (N/A)")
    
    # Generate plots (optional)
    if not GENERATE_PLOTS:
        print("\n(plots disabled; pass --plots to generate figures)")
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE")
        print("="*80)
        return

    # Make matplotlib robust in headless environments
    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"\n{'='*80}")
    print("GENERATING PLOTS")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Two Active Inference Agents - Individually Collective Paradigm', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Success rate over episodes (learning curve)
    ax = axes[0, 0]
    episodes = list(range(1, NUM_EPISODES_PER_SEED + 1))
    success_per_episode = []
    for ep_idx in range(NUM_EPISODES_PER_SEED):
        ep_results = [all_results[seed_idx * NUM_EPISODES_PER_SEED + ep_idx]
                     for seed_idx in range(NUM_SEEDS)]
        success_per_episode.append(np.mean([1 if r['success'] else 0 for r in ep_results]))
    
    window = 50
    if len(success_per_episode) >= window:
        success_ma = np.convolve(success_per_episode, np.ones(window)/window, mode='valid')
        ax.plot(episodes[window-1:], success_ma, '-', linewidth=2, color='purple', 
                label=f'MA-{window}')
    ax.scatter(episodes, success_per_episode, alpha=0.1, s=5, color='purple')
    
    num_configs = NUM_EPISODES_PER_SEED // EPISODES_PER_CONFIG
    for i in range(1, num_configs):
        ax.axvline(i * EPISODES_PER_CONFIG + 0.5, color='red', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Learning Curve (Averaged Across Seeds)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    # Plot 2: Success rate by seed
    ax = axes[0, 1]
    seeds = [s['seed'] for s in seed_summaries]
    rates = [s['success_rate'] for s in seed_summaries]
    colors = plt.cm.Purples(np.linspace(0.4, 0.8, len(seeds)))
    ax.bar(seeds, rates, color=colors)
    ax.axhline(np.mean(rates), color='red', linestyle='--', label=f'Mean: {np.mean(rates):.1f}%')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate by Seed')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Reward distribution
    ax = axes[1, 0]
    rewards = [r['reward'] for r in all_results]
    ax.hist(rewards, bins=20, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(rewards):+.2f}')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Frequency')
    ax.set_title('Reward Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: First half vs second half comparison
    ax = axes[1, 1]
    first_half_rates = [100 * s['first_half_wins'] / (s['total'] // 2) for s in seed_summaries]
    second_half_rates = [100 * s['second_half_wins'] / (s['total'] // 2) for s in seed_summaries]
    
    x = np.arange(len(seeds))
    width = 0.35
    ax.bar(x - width/2, first_half_rates, width, label='First Half', color='lavender', edgecolor='purple')
    ax.bar(x + width/2, second_half_rates, width, label='Second Half', color='darkviolet', edgecolor='purple')
    ax.set_xlabel('Seed')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Learning Progress (First vs Second Half)')
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"two_aif_agents_individually_collective_seeds{NUM_SEEDS}_ep{NUM_EPISODES_PER_SEED}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {plot_path}")
    
    plt.close(fig)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()

