# Subgoal-match report — does destination choice track the recipe, with no reward for doing so?

## The question

Overcooked's reward (`C`) pays out only on final soup delivery — nothing for fetching an onion,
depositing it, fetching a dish, or picking up the finished soup. If an agent's destination choice
nonetheless tracks the objectively correct next recipe step most of the time, that's a direct,
quantified behavioral sign of subgoal-emergent sequencing with no reward shaping behind it.

## Method

For every step in an episode's **productive phase** (from the start up to and including the
episode's *last* delivery — the window before all three paradigms hit their shared "leftover item,
nothing left to do" absorbing state), the canonical next move is derived purely from the agent's own
belief about what it's holding (`self_held`) and the pot's fill level (`pot_state`):

| Holding | Pot | Canonical next destination |
|---|---|---|
| nothing | not full (0–2 onions) | an onion (`onion1` or `onion2`) |
| nothing | full (3 onions) | `dish` |
| an onion | not full | `pot` (deposit) |
| a dish | full | `pot` (pick up the soup) |
| soup | — | `serve` |

Two belief states have no single correct answer and are excluded from scoring: holding a spare onion
while the pot's already full (**TRAP**), and holding a dish while the pot isn't ready yet (**WAIT**).
The agent's actual chosen destination is then bucketed into one of 5 roles (`onion`, `dish`, `pot`,
`serve`, `counter`) and checked against the canonical role. This exactly reproduces the methodology
already on record in `ai/stage3_fc_ic_ind_collaboration_report.md` §8 — re-derived independently here
from the raw CSV+JSONL logs, not copied, and cross-checked to match it (see Provenance below).

## Headline numbers

| Paradigm | Mean match rate (per-seed) | SD | Range | Pooled match rate | Independence baseline | n seeds |
|---|---|---|---|---|---|---|
| Independent | **49.5%** | 10.1% | 27.1%–69.7% | 47.3% | 23.1% | 28/30 |
| Individually Collective | **74.8%** | 6.9% | 59.0%–88.1% | 75.1% | 29.2% | 30/30 |
| Fully Collective | **75.4%** | 5.6% | 63.8%–87.3% | 76.4% | 29.7% | 29/30 |

*(2 IND seeds, 1 FC seed have zero deliveries, so no "productive phase" to score — excluded, not
counted as failures.)*

**Every paradigm tracks the recipe far above chance.** The independence baseline — the match rate
you'd expect if destination choice had nothing to do with the required role, computed from each
paradigm's own actual marginal distribution of chosen destinations — is 23–30%. All three paradigms
land 20–48 points above that. Since nothing in the reward function distinguishes a correct
intermediate move from an irrelevant one, this gap is the signature of subgoal-emergent behavior.

**Statistical separation** (Mann-Whitney U, per-seed rates):
- Independent vs. Individually Collective: p = 3.8×10⁻¹⁰
- Independent vs. Fully Collective: p = 3.5×10⁻¹⁰
- Individually Collective vs. Fully Collective: p = 0.81 (**not distinguishable** — consistent with
  every other metric in this project where IC and FC land statistically tied)

## Where, specifically, does each paradigm succeed or fail?

The single average number hides *which* recipe transition is driving the gap. Breaking it down by
what was actually required (row) vs. what was actually chosen (column), IND's failure is concentrated
in two specific places, not spread evenly:

**Independent**
| required → chosen | onion | dish | pot | serve | counter |
|---|---|---|---|---|---|
| onion needed | 51.4% | 18.3% | 4.5% | 3.7% | 22.1% |
| dish needed (pot full) | 49.2% | 28.3% | 2.0% | 3.1% | 17.3% |
| pot needed (deposit/collect) | 10.1% | 3.1% | **45.7%** | 4.5% | 36.6% |
| serve needed (holding soup) | 5.1% | 5.7% | 4.7% | **59.9%** | 24.6% |

**Fully Collective**
| required → chosen | onion | dish | pot | serve | counter |
|---|---|---|---|---|---|
| onion needed | 69.6% | 25.5% | 1.1% | 0.7% | 3.1% |
| dish needed (pot full) | 67.0% | 27.8% | 0.9% | 0.3% | 4.0% |
| pot needed | 1.6% | 0.1% | **93.4%** | 0.7% | 4.2% |
| serve needed | 0.0% | 0.0% | 0.0% | **97.7%** | 2.3% |

**Individually Collective**
| required → chosen | onion | dish | pot | serve | counter |
|---|---|---|---|---|---|
| onion needed | 66.3% | 26.8% | 0.6% | 0.6% | 5.7% |
| dish needed (pot full) | 62.8% | 31.5% | 0.9% | 0.6% | 4.3% |
| pot needed | 2.0% | 0.7% | **90.0%** | 1.0% | 6.3% |
| serve needed | 0.0% | 0.0% | 0.0% | **98.6%** | 1.4% |

**Two concrete findings from this breakdown:**

1. **IND's biggest specific weakness is "pot" and "serve"** — only 45.7% and 59.9% respectively,
   vastly below IC/FC's 90–98%. IND sends the agent to a *counter* instead of the pot 36.6% of the
   time when it should be depositing/collecting — the single largest source of IND's overall gap.
2. **All three paradigms share the exact same blind spot**: once the pot fills up, an empty-handed
   agent should switch to fetching a *dish*, but every paradigm still goes for another *onion* most
   of the time instead (IND 49.2%, IC 62.8%, FC 67.0% — worse, if anything, for the two joint-modeling
   paradigms). This is not an IND-specific failure; it's a structural property of the current
   preference model shared by all three, worth naming as a limitation in its own right.

## Per-seed spread (see `subgoal_match_distribution.png`)

IND's per-seed range (27%–70%) is roughly double the width of IC's or FC's (59–88%, 64–87%) — IND is
not just lower on average, it's also far less consistent seed-to-seed. IC and FC are both tight and
overlapping, visually confirming the p=0.81 result: there is no reading of this plot in which one
looks like it beats the other.

## Interpretation and honest limits

This is a **behavioral signature** consistent with subgoal-emergent sequencing, not a mechanism-level
proof. The Red-Blue-Button toy task demonstrated *why* mean-field variational factorization produces
this kind of behavior (partially-completed states become more probable under the marginal posterior,
provably, from the belief geometry). Replicating that exact mechanistic argument for Overcooked would
require per-candidate-policy utility/info-gain decomposition, which isn't retained in these logs (only
top-5 policy indices/probabilities per step) — what's shown here is the observable consequence for
this environment, not the mechanism. `counter` is always scored as a miss even when visiting one is a
reasonable move (staging an item, freeing a hand), so these match rates are a conservative floor, not
an inflated number.

## Provenance

Computed directly from `thesis_logs/03_ma_overcooked/sal_{ind,fc,ic}_30seed*/` (CSV + JSONL, all 30
seeds per paradigm, episode_seeds 76–105), independently re-derived from raw belief/action logs rather
than taken from any prior write-up. Cross-checked against `ai/stage3_fc_ic_ind_collaboration_report.md`
§8 — per-seed mean/SD/range and the IND-vs-IC significance test match within rounding. The
required-vs-chosen confusion matrix and the pooled/baseline numbers above are new, not present in the
prior report.
