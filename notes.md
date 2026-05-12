# Notes



```
  XXPXX
  O1  O
  X  2X
  XDXSX
```


## Status: 
 - semantic policies work




## TODO
- [ ] make sure everything is correct
- [ ] action-level policy
- [ ] dynamic policy

### model_init.py
- [x] states
- [x] observations
- [x] state–state dependencies
- [x] state–obs dependencies
- [x] add other position and other held object (`InDe`)

---

### A.py
- [x] soup delivered observation depends only on ck_delivered state
- [ ] other position and held observations (`InDe`)

---

### B.py
- [x] drop on full counter handled
- [ ] pickup from counter
- [x] model counters when agent interacts with them
- [ ] other position and held transitions (`InDe`)
- [ ] joint policy transitions (`InCo`)

---

### C.py
- [ ] (not implemented yet) (`InDe`)

---

### D.py
- [ ] other position and held initial states (`InDe`)

---

### Policies
- [x] joint policies (`InCo`)
- [x] Fixed Predefined set of policies saved in a file
- [ ] think about if you need to change Active Inference implementation
- [ ] list all policy ideas (why james' doesnt work)
- [ ] change the gen model
- [ ] run
- [ ] REMEMBER to save results to discuss them in
---

## Parameters

```python
A_NOISE_LEVEL = 0.001
B_NOISE_LEVEL = 0.0

INTERACT_SUCCESS_PROB = 1.0
```












ind changes:
generative_models/.../IndependentWithSemanticPoliciesActionLevel/
B.py — Refactored to true independence:

Removed all joint/interleaved action decoding (_try_interleaved_step, _decode_joint_pair, etc.)
B_other_pos, B_other_orientation, B_other_held now return identity transitions (other agent is pure environment, not planned over)
_try_primitive_policy_step extracts only self_action from the action tuple (no other_action)
B_fn and B_fn_primitive_step no longer accept or use other_action
B_self_pos handles collision avoidance using q_other_pos (observed, not planned)
model_init.py — Cleaned up action/state spaces:

Removed "noop" from DESTINATIONS and SEMANTIC_DEST_TARGET_POSE → N_ACTIONS reduced from 22 to 20
Removed joint pair machinery (N_INTERLEAVED_STEP_ACTIONS, etc.)
Updated state_state_dependencies to reflect independent (ego-only) transitions
A.py — Fixed entropy:

A_NOISE_LEVEL: 0.001 → 0.01 (restores working value; old value caused all belief entropies to fall below ENTROPY_THRESHOLD, making every factor non-dynamic → zero info gain → uniform policies)
D.py — Fixed initial beliefs:

Removed hard-coded default start positions (DEFAULT_START_GRID_XY, DEFAULT_OTHER_START_GRID_XY)
D_fn(None) now returns uniform priors for self_pos and other_pos (genuine uncertainty), point-mass for orientation and task state
D_fn(config) still returns point-mass beliefs when actual positions are known
__init__.py, C.py, env_utils.py — Docstrings updated to reflect Independent paradigm

agents/IndependentActiveInferenceWithDynamicPolicies/agent.py
Added use_action_for_state_inference: bool = False parameter to __init__
Added conditional logic in infer_states to optionally use the agent's previous action as a prior when computing beliefs
run_scripts_overcooked/run_independent_semantic_action_level.py
Fixed imports to pull from IndependentWithSemanticPoliciesActionLevel (not IC)
Removed other_action from the ego agent's action tuple: agent_0.action = (PPS, a0_prim) instead of (PPS, a0_prim, a1_prim)
agent_0.reset() / agent_1.reset() called without config= argument → triggers uniform position priors from D_fn(None)
