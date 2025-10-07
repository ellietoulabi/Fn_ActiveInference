"""Profile get_expected_obs_from_beliefs in detail."""

import time
import numpy as np
from generative_models.SA_ActiveInference.RedBlueButton import A, B, D, model_init, env_utils
from environments.RedBlueButton.SingleAgentRedBlueButton import SingleAgentRedBlueButtonEnv

# Setup
env = SingleAgentRedBlueButtonEnv(width=3, height=3, max_steps=10)
env.reset()
d_config = env_utils.get_D_config_from_env(env)
qs = D.D_fn(d_config)
qs_next = B.B_fn(qs, action=1, width=3, height=3, B_NOISE_LEVEL=0.0)

state_factors = list(model_init.states.keys())
state_sizes = {k: len(v) for k, v in model_init.states.items()}

# Manual implementation with timing
print("Manual get_expected_obs_from_beliefs with timing:")
print("=" * 70)

import itertools

t0 = time.time()

map_indices = {f: int(np.argmax(qs_next[f])) for f in state_factors}
print(f"1. Compute MAP indices: {(time.time()-t0)*1000:.2f}ms")

t1 = time.time()
SKIP_MODALITIES = {'button_just_pressed'}
ENTROPY_THRESHOLD = 0.01
dynamic_factors = set()
for f in state_factors:
    q_f = qs_next[f]
    entropy = -np.sum(q_f * np.log(q_f + 1e-16))
    if entropy > ENTROPY_THRESHOLD:
        dynamic_factors.add(f)

print(f"2. Find dynamic factors: {(time.time()-t1)*1000:.2f}ms, dynamic={dynamic_factors}")

t2 = time.time()
all_deps = set()
for modality, deps in model_init.observation_state_dependencies.items():
    if modality not in SKIP_MODALITIES:
        for dep in deps:
            if dep in dynamic_factors:
                all_deps.add(dep)

dep_list = sorted(all_deps)
print(f"3. Find all deps: {(time.time()-t2)*1000:.2f}ms, deps={dep_list}")

t3 = time.time()
dep_ranges = [range(len(qs_next[dep])) for dep in dep_list] if dep_list else [[0]]
print(f"4. Build dep ranges: {(time.time()-t3)*1000:.2f}ms")

t4 = time.time()
likelihood_cache = []
prob_cache = []

num_combos = 0
for combo in itertools.product(*dep_ranges):
    num_combos += 1
    joint_prob = 1.0
    state_indices = map_indices.copy()
    if dep_list:  # Only if we have dynamic factors
        for dep, idx in zip(dep_list, combo):
            joint_prob *= qs_next[dep][idx]
            state_indices[dep] = int(idx)
    
    if joint_prob <= 1e-16:
        continue
    
    obs_likelihoods = A.A_fn(state_indices)
    likelihood_cache.append(obs_likelihoods)
    prob_cache.append(joint_prob)

print(f"5. Enumerate & call A_fn: {(time.time()-t4)*1000:.2f}ms, combos={num_combos}, cached={len(likelihood_cache)}")

t5 = time.time()
qo_dict = {}
for modality, deps in model_init.observation_state_dependencies.items():
    if modality in SKIP_MODALITIES:
        continue
    
    num_obs = len(model_init.observations[modality])
    qo_m = np.zeros(num_obs)
    
    for obs_lik, joint_prob in zip(likelihood_cache, prob_cache):
        p_o_m = obs_lik[modality]
        qo_m += joint_prob * p_o_m
    
    qo_dict[modality] = qo_m / np.maximum(np.sum(qo_m), 1e-8)

print(f"6. Marginalize modalities: {(time.time()-t5)*1000:.2f}ms")

t6 = time.time()
if 'button_just_pressed' in model_init.observation_state_dependencies:
    p_on_red = qo_dict['on_red_button'][1]
    p_on_blue = qo_dict['on_blue_button'][1]
    p_just_pressed = min(1.0, p_on_red + p_on_blue)
    qo_dict['button_just_pressed'] = np.array([1.0 - p_just_pressed, p_just_pressed])

print(f"7. Approximate button_just_pressed: {(time.time()-t6)*1000:.2f}ms")

print(f"\nTOTAL: {(time.time()-t0)*1000:.2f}ms")


