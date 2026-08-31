"""
Preference model (C) for FullyCollective paradigm.

Preferences are expressed over JOINT observations.
Keep close to SA conventions: prefer win, avoid lose, small shaping for pressed buttons.

USE_INTERMEDIATE_REWARD: temporary experiment flag (2026-08-13, ai/02-debug.md
"reward-shaping asymmetry" entry). Set False to strip button-press shaping
entirely (win/lose only), isolating how much of FC/IC's advantage over
Independent comes from this shaping term vs. other structural differences
(joint modeling, collision-aware transitions). Default True preserves the
original, previously-reported behavior exactly.
"""

USE_INTERMEDIATE_REWARD = False


def C_fn(observation_indices):
    prefs = {}
    for modality, obs_idx in observation_indices.items():
        if modality == "game_result":
            if obs_idx == 1:  # win
                prefs[modality] = 4.0
            elif obs_idx == 2:  # lose
                # Loss aversion, deliberately asymmetric with the +4.0 win reward
                # (2026-08-14, ai/02-debug.md "cross-timestep termination not
                # tracked in EFE rollout"). Root cause: get_expected_states rolls
                # a policy_len=2 policy forward with no memory that step 1 may
                # already have ended the episode, so a policy that presses blue
                # before red at step 1 can have its correctly-computed loss
                # utility (~-3.2 at the old -4.0 magnitude) almost fully canceled
                # by an imagined "recover by also pressing red at step 2" branch
                # that predicts a further +win at step 2 -- physically impossible,
                # since the real environment ends the episode the instant blue is
                # pressed before red. The real fix is making the rollout
                # termination-aware; this is a deliberate, requested mitigation
                # in the meantime: at policy_len=2, a false "recovery" step can
                # contribute at most ~4.6 (win=4.0 plus the two small shaping
                # terms below), so scaling the loss penalty to -40.0 leaves a
                # large margin even in that worst case -- confirmed by
                # re-evaluating the exact motivating scenario (see debug notes):
                # the previously-winning catastrophic policy's G went from
                # -1.01 (best in the whole 1296-policy space) to +34.2 (soundly
                # rejected), with no change needed to win reward or any
                # intermediate/shaping term.
                prefs[modality] = -40.0
            else:
                prefs[modality] = 0.0
        elif modality == "red_button_state":
            prefs[modality] = (0.5 if obs_idx == 1 else 0.0) if USE_INTERMEDIATE_REWARD else 0.0
        elif modality == "blue_button_state":
            prefs[modality] = (0.2 if obs_idx == 1 else 0.0) if USE_INTERMEDIATE_REWARD else 0.0
        else:
            prefs[modality] = 0.0
    return prefs


