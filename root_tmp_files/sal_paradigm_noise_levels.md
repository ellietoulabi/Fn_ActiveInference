# SAL cramped_room — noise levels (IND / IC / FC)

Comparison of observation and transition noise in the three **Semantic Action Level** generative models:

- `IndependentWithSemanticPoliciesActionLevel/` (IND)
- `IndividuallyCollectiveWithSemanticPoliciesActionLevel/` (IC)
- `FullyCollectiveWithSemanticPoliciesActionLevel/` (FC)

## Summary

| Noise type | IND | IC | FC |
|------------|-----|----|----|
| **`A_NOISE_LEVEL`** (observation likelihood) | **0.01** | **0.001** | **0.005** |
| **`B_NOISE_LEVEL`** (transition uniform mix-in) | **0.0** (default, unused) | **0.0** | **0.0** |
| **Counter obs** (`ctr_*_obs`) | **0** (deterministic) | **0** | **0** |
| **`INTERACT_SUCCESS_PROB`** (B dynamics) | **1.0** | **1.0** | **1.0** |

**Takeaway:** only **A (observation) noise** differs across paradigms. IND is noisiest, IC is quietest, FC is intermediate.

## Observation noise (`A_NOISE_LEVEL`)

All three use the same `_noisy_categorical` helper in `A.py`:

- True outcome: `P = 1 − ε`
- Each wrong outcome: `P = ε / (n − 1)`

| Paradigm | ε | Example: 6-way `self_pos_obs` |
|----------|---|-------------------------------|
| IND | 0.01 | true **0.99**, each wrong **0.002** |
| IC | 0.001 | true **0.999**, each wrong **0.0002** |
| FC | 0.005 | true **0.995**, each wrong **0.001** |

**Modalities using `A_NOISE_LEVEL`:**

- `self_pos_obs`, `self_orientation_obs`, `self_held_obs`
- `other_pos_obs`, `other_orientation_obs`, `other_held_obs`
- `pot_state_obs`
- `soup_delivered_obs` (IND uses custom binary logic with the same ε; IC/FC use `_noisy_categorical`)

**Deterministic (noise = 0):** all `ctr_{1,3,10,14,17}_obs` counter observations.

### FC rationale

FC sets `A_NOISE_LEVEL = 0.005` (not IC’s 0.001) so post-inference beliefs retain enough entropy for epistemic value / information-gain during joint policy evaluation. At 0.001, beliefs collapsed and all 400 joint semantic pairs received identical EFE → near-uniform `q_pi`.

See comment in `FullyCollectiveWithSemanticPoliciesActionLevel/A.py`.

## Transition noise (`B_NOISE_LEVEL`)

`B_fn(..., B_NOISE_LEVEL=0.0)` in all three. Uniform mixing is applied only if `B_NOISE_LEVEL > 0`. Agents and runners never pass a nonzero value, so **B noise is effectively off**.

## Interaction stochasticity

`INTERACT_SUCCESS_PROB = 1.0` in all three `model_init.py` files → INTERACT transitions in B are deterministic (no failed-interact noise).

## Relative ordering

| Comparison | Ratio |
|------------|-------|
| IND vs IC | IND is **10×** noisier (0.01 vs 0.001) |
| FC vs IC | FC is **5×** noisier (0.005 vs 0.001) |
| IND vs FC | IND is **2×** noisier (0.01 vs 0.005) |

## Source files

| Paradigm | A | B | model_init |
|----------|---|---|------------|
| IND | `IndependentWithSemanticPoliciesActionLevel/A.py` | `.../B.py` | `.../model_init.py` |
| IC | `IndividuallyCollectiveWithSemanticPoliciesActionLevel/A.py` | `.../B.py` | `.../model_init.py` |
| FC | `FullyCollectiveWithSemanticPoliciesActionLevel/A.py` | `.../B.py` | `.../model_init.py` |

**Note:** The non–action-level `IndividuallyCollectiveWithSemanticPolicies/` package uses the same IC values (`A_NOISE_LEVEL = 0.001`). SAL runners use the `*ActionLevel` packages above.
