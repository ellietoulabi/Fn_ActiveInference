## Policy generation (ego-first interleaved)

This repo defines a small, fixed policy set in `policy.py` to reduce the Active Inference policy space.

### Policy representation

A policy is a horizon-3 list of `(actor, action)` pairs:

- **actor**: who executes the step
  - `SELF = 0`
  - `OTHER = 1`
- **action**: one of the 6 primitive actions
  - `NORTH=0, SOUTH=1, EAST=2, WEST=3, STAY=4, INTERACT=5`

So one policy looks like:

```text
[(SELF, EAST), (OTHER, INTERACT), (SELF, WEST)]
```

### Ego-first constraint

The generator enforces an “ego-first” rule:

- **Step 1**: `actor` must be `SELF`
- **Steps 2–3**: `actor` can be `SELF` or `OTHER`

This guarantees that the action you *actually execute in the environment at the current timestep* is always a SELF action, while still allowing the agent to reason about the OTHER agent’s possible moves in imagined rollouts.

### Enumeration logic and policy count

The generator enumerates the full cartesian product of allowed step options:

- Step 1 options: `SELF × 6 actions` → \(6\)
- Step 2 options: `(SELF or OTHER) × 6 actions` → \(12\)
- Step 3 options: `(SELF or OTHER) × 6 actions` → \(12\)

Total:


`6 × 12 × 12 = 864`


