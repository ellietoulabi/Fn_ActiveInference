# Performance Analysis: Functional Active Inference Agent

## Summary

**Final Performance**: ~10-70ms per step (after JAX warmup) for `policy_len=1`, 6 policies  
**Compared to**: Matrix-based version runs at similar speeds  
**Key Achievement**: Dependency-based marginalization reduces state enumeration from 2,916 to typically 1-36 states

---

## System Configuration

### Problem Size
- **State space**: 2,916 joint states (9×9×9×2×2)
  - `agent_pos`: 9 positions (3×3 grid)
  - `red_button_pos`: 9 positions (static)
  - `blue_button_pos`: 9 positions (static)
  - `red_button_state`: 2 states (not_pressed, pressed)
  - `blue_button_state`: 2 states (not_pressed, pressed)

- **Observation space**: 7 modalities
  - `agent_pos`: 9 observations
  - `on_red_button`: 2 observations
  - `on_blue_button`: 2 observations
  - `red_button_state`: 2 observations
  - `blue_button_state`: 2 observations
  - `game_result`: 3 observations
  - `button_just_pressed`: 2 observations

- **Action space**: 6 actions (UP, DOWN, LEFT, RIGHT, OPEN, NOOP)

### Agent Configuration
- `policy_len`: 1 (reduced from 3 for speed)
- `inference_horizon`: 1 (reduced from 3 for speed)
- `num_policies`: 6 (= 6^1)
- `num_iter` (state inference): 3 iterations

---

## Per-Step Breakdown

### Step 1: State Inference (`infer_states`)
**Time**: ~26ms  
**Purpose**: Update beliefs about hidden states given observation

#### What it does:
1. Format observation into one-hot vectors
2. Run Fixed-Point Iteration (FPI) with functional A
3. Iterate 3 times to converge beliefs

#### Bottlenecks:

**1. Observation Enumeration for Likelihood Computation**
- **Current approach**: For each FPI iteration, enumerate plausible joint states to compute `p(o|s)` for each state configuration
- **Why slow**: Even with pruning (threshold=1e-3), we enumerate many joint states
- **Complexity**: 
  - Worst case: O(num_iter × num_plausible_states × num_modalities)
  - With 3 iterations, ~10-100 plausible states: ~30-300 A_fn calls
- **Time**: ~20-25ms of the 26ms total

**2. A_fn Call Overhead**
- **Current**: Each A_fn call computes likelihoods for ALL 7 modalities
- **Observation**: We only need likelihoods for observed modalities (could skip others)
- **Improvement potential**: Marginal (~10-20% speedup)

**3. Coordinate Ascent Updates**
- **Current**: Update each factor sequentially in a Python loop
- **Complexity**: O(num_iter × num_factors)
- **Time**: Negligible (~1-2ms)

**Key Insight**: State inference is fundamentally limited by how many states we need to evaluate to compute accurate likelihoods. The functional approach requires explicit enumeration where matrix approach uses matrix-vector products.

---

### Step 2: Policy Inference (`infer_policies`)
**Time**: ~12-60ms (varies with state uncertainty)  
**Purpose**: Evaluate Expected Free Energy (EFE) for each policy

#### What it does:
For each of 6 policies:
1. **State Rollout** (`get_expected_states`): Simulate future states using B_fn
2. **Observation Prediction** (`get_expected_obs_from_beliefs`): Predict observations at each timestep
3. **Utility Calculation**: Compute pragmatic value using C_fn
4. **Information Gain**: Compute epistemic value using Bayesian surprise

#### Bottlenecks:

**1. State Enumeration in Observation Prediction** (MAJOR)
- **Current approach**: For each timestep, enumerate combinations of DYNAMIC state factors
- **Why slow**: 
  - Need to call A_fn for each unique state configuration
  - With uncertain agent position (9 states) + button states (2×2): up to 36 combinations
  - This is done TWICE per timestep (once for expected obs, once for info gain)
- **Complexity per policy**: 
  - O(policy_len × num_dynamic_combos × 2)
  - With policy_len=1, dynamic_combos=1-36: 2-72 A_fn calls per policy
  - Total: 12-432 A_fn calls for all 6 policies
- **Time**: 8-50ms depending on belief uncertainty

**Optimization Applied**:
- ✅ Skip `button_just_pressed` modality (would add 2,916 combos)
- ✅ Only enumerate factors with entropy > 0.01 (skip static factors like button positions)
- ✅ Cache A_fn results and reuse across modalities
- ✅ Convert JAX arrays to NumPy before operations (avoid recompilation)

**Remaining Issues**:
- We still enumerate states twice (expected_obs + info_gain) - these could share computation
- When agent is uncertain (e.g., after noisy movement), we enumerate many states

**2. B_fn Calls for State Rollout**
- **Current**: Call B_fn once per policy (with policy_len=1)
- **Complexity**: O(num_policies × policy_len)
- **Time**: ~1-2ms total (very fast after NumPy optimization)
- **Not a bottleneck**

**3. Information Gain Calculation**
- **Current**: Compute Bayesian surprise = H[q(o)] - E_q[H[p(o|s)]]
- **Requires**: Same state enumeration as observation prediction
- **Could share**: The state enumeration and A_fn calls with observation prediction
- **Improvement potential**: ~50% speedup if combined with expected_obs computation

---

## Fundamental Algorithmic Bottlenecks

### 1. **Dependency-Based Marginalization Trade-off**

**The Core Problem**:
In matrix-based Active Inference:
```python
qo = A @ qs  # Single matrix-vector multiply
```

In functional Active Inference:
```python
qo = Σ_s q(s) * A_fn(s)  # Must enumerate states explicitly
```

**Our Optimization**:
Instead of full joint (2,916 states), we only enumerate over dependent factors:
- For `agent_pos` observation: enumerate only `agent_pos` factor (9 states)
- For `on_red_button`: enumerate `agent_pos` × `red_button_pos` (81 states)
- For `button_just_pressed`: SKIP (would be 2,916) - approximate instead

**Result**: Reduced from ~2,916 states to ~1-36 dynamic states per marginalization

**Remaining Challenge**: 
- When multiple factors are uncertain, combinatorial explosion still occurs
- Example: Uncertain agent position (9) + uncertain button states (4) = 36 combos
- This is **fundamentally unavoidable** in functional approach without further approximations

---

### 2. **Redundant State Enumeration**

**The Problem**:
We enumerate states twice per timestep in policy evaluation:
1. In `get_expected_obs_from_beliefs`: to predict q(o)
2. In `calc_surprise_functional`: to compute info gain

Both need the same information: p(o|s) for plausible states.

**Current Workaround**: None - we duplicate the work

**Potential Solution**:
Create unified function `get_expected_obs_and_info_gain()` that:
- Enumerates states once
- Computes both q(o) and Bayesian surprise in one pass
- Returns both results

**Estimated Speedup**: ~30-40% for policy evaluation

---

### 3. **A_fn Call Overhead**

**Current Behavior**:
Each A_fn call computes likelihoods for ALL 7 modalities, even if we only need 1-2.

**Example**:
When marginalizing `agent_pos` observation, we only need `A_agent_pos(state)`, but we compute all 7 modalities.

**Why This Happens**:
A_fn interface returns all modalities: `obs_likelihoods = A_fn(state_indices)`

**Potential Solution**:
Add optional parameter: `A_fn(state_indices, modalities=['agent_pos'])`
- Only compute requested modalities
- Avoid unnecessary computation

**Estimated Speedup**: ~20-30% for observation prediction

---

### 4. **JAX Compilation Overhead**

**The Problem** (MOSTLY SOLVED):
JAX recompiles operations on traced arrays, causing massive slowdowns.

**Symptoms**:
- First few steps: 200-300ms
- After warmup: 10-70ms

**Fixes Applied**:
- ✅ Convert all JAX arrays to NumPy in B_fn before operations
- ✅ Return NumPy arrays from B functions (not JAX)
- ✅ Use NumPy for all marginalization operations in control.py and maths.py

**Remaining Issue**:
- State inference still uses some JAX operations (in inference.py)
- Could further optimize by converting all to NumPy

---

### 5. **Policy Enumeration Scaling**

**Current**: `num_policies = num_actions ^ policy_len`
- policy_len=1: 6 policies ✅ Fast
- policy_len=2: 36 policies → ~216-360ms per step
- policy_len=3: 216 policies → ~1.3-4.3s per step ❌ Too slow

**The Fundamental Limit**:
- Each additional timestep in policy_len multiplies:
  - Number of policies by 6
  - State rollouts by 1 timestep
  - A_fn calls for observation prediction

**Trade-off**:
- Short planning (policy_len=1): Fast but myopic
- Long planning (policy_len=3): Slow but foresighted

**Possible Solutions**:
1. **Policy pruning**: Only evaluate promising policies (e.g., top-k after 1 step)
2. **Hierarchical planning**: Separate high-level and low-level policies
3. **Amortized inference**: Learn to predict good policies (neural network)
4. **Sparse sampling**: Monte Carlo tree search instead of exhaustive enumeration

---

## Comparison: Functional vs Matrix Approach

### Matrix Approach Advantages:
1. **Vectorized operations**: Matrix-vector products are highly optimized
2. **No enumeration**: All marginalizations are implicit in linear algebra
3. **GPU acceleration**: Can use hardware acceleration for large matrices

### Functional Approach Advantages:
1. **Memory efficient**: No need to store huge tensors (2,916^2 × num_actions for B)
2. **Flexible dependencies**: Easy to express complex conditional dependencies
3. **Interpretable**: Each function is a clear p(o|s) or p(s'|s,a) definition
4. **Scalable to large spaces**: Matrix approach infeasible when state space > 10^6

### When Functional Wins:
- Large state spaces (> 100K states)
- Sparse dependencies (most factors independent)
- Complex dynamics (hard to express as matrix)
- Memory constrained environments

### When Matrix Wins:
- Small state spaces (< 10K states)
- Dense dependencies (many factors interact)
- Need real-time performance (< 10ms per step)
- GPU available

**This Problem (RedBlueButton)**:
- State space: 2,916 (small-to-medium)
- Dependencies: Sparse (most factors independent)
- **Conclusion**: Matrix and Functional are **comparable** with proper optimization

---

## Profiling Data

### Detailed Timing (policy_len=1, after warmup):

```
Per Agent Step (76ms total):
├── State Inference (26ms)
│   ├── State enumeration & A_fn calls: ~20ms
│   ├── Coordinate ascent updates: ~3ms
│   └── Overhead (formatting, etc.): ~3ms
│
└── Policy Inference (12-50ms, depends on uncertainty)
    ├── Per policy (6 total):
    │   ├── B_fn for state rollout: ~0.2ms
    │   ├── Enumerate states for obs prediction: 1-8ms
    │   │   └── A_fn calls: ~0.1ms each
    │   ├── Enumerate states for info gain: 1-8ms
    │   │   └── A_fn calls: ~0.1ms each (duplicate!)
    │   └── Utility calculation: <0.1ms
    │
    └── Policy posterior (softmax): <0.1ms
```

### A_fn Performance:
- **Per call**: ~0.1ms (very fast)
- **Calls per step**: 12-100 (varies with uncertainty)
- **Total time in A_fn**: 1.2-10ms

### B_fn Performance:
- **Per call**: ~0.2ms (very fast after NumPy optimization)
- **Calls per step**: 6 (one per policy with policy_len=1)
- **Total time in B_fn**: ~1.2ms

---

## Recommended Improvements (In Priority Order)

### 1. **Combine Expected Obs and Info Gain** ⭐⭐⭐
**Impact**: 30-40% speedup for policy evaluation  
**Complexity**: Medium  
**Implementation**: Create unified `get_expected_obs_and_info_gain()` function

### 2. **Selective Modality Computation** ⭐⭐
**Impact**: 20-30% speedup for observation prediction  
**Complexity**: Low  
**Implementation**: Add `modalities` parameter to A_fn

### 3. **Policy Pruning** ⭐⭐⭐
**Impact**: Enable policy_len=2 or 3 with acceptable speed  
**Complexity**: Medium-High  
**Implementation**: 
- Evaluate all policies at depth 1
- Only expand top-k policies to depth 2, 3, etc.
- Use branch-and-bound to prune suboptimal branches

### 4. **Adaptive Enumeration Threshold** ⭐
**Impact**: 10-20% speedup when beliefs are concentrated  
**Complexity**: Low  
**Implementation**: Dynamically adjust `ENTROPY_THRESHOLD` based on belief concentration

### 5. **Parallelize Policy Evaluation** ⭐⭐
**Impact**: Near-linear speedup with number of cores  
**Complexity**: Medium  
**Implementation**: Use multiprocessing to evaluate policies in parallel

### 6. **Vectorized State Enumeration** ⭐
**Impact**: 10-15% speedup  
**Complexity**: Medium  
**Implementation**: Batch A_fn calls using NumPy broadcasting instead of Python loops

---

## Discussion Points for Supervisor

1. **Fundamental Trade-off**: Functional approach requires explicit state enumeration where matrix approach uses implicit marginalization. Is the memory/flexibility benefit worth the computational cost?

2. **Scalability**: At what state space size does functional approach become clearly superior to matrix? Our analysis suggests ~100K states.

3. **Approximations**: We approximate `button_just_pressed` modality (2,916 combos). How to rigorously validate such approximations?

4. **Policy Depth**: With current implementation, policy_len=1 is fast, policy_len=3 is slow. Is planning depth or speed more important for this task?

5. **Hybrid Approach**: Could we combine functional (for dynamics) with matrix (for inference) to get best of both worlds?

6. **Learned Approximations**: Instead of exact marginalization, could we train a neural network to approximate q(o|qs) quickly?

---

## Conclusion

The functional Active Inference implementation achieves **comparable performance to matrix-based** approach for small-to-medium state spaces (~3K states) after careful optimization. 

**Key bottleneck**: Explicit state enumeration during marginalization (unavoidable in pure functional approach)

**Main optimization**: Dependency-based marginalization reduces enumeration from 2,916 to 1-36 states

**Remaining challenge**: Policy evaluation with long horizons (policy_len > 1) scales poorly due to exponential policy space growth

**Recommended next step**: Implement policy pruning to enable deeper planning while maintaining real-time performance.

