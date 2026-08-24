"""
Two-agent OPSRL for TwoAgentRedBlueButton.

This is an isolated *subclass* of OPSRLAgent, not a modification of
agents/OPSRL/agent.py. That file is the exact class behind the already-
reported Stage-1 single-agent nine-agent table (see ai/02-debug.md, "OPSRL
bug in the exact script behind the reported Stage-1 table"), so per this
project's isolate-before-editing-shared-code convention (the same one that
produced agents/ActiveInferenceRedBlueButton/, generative_models/*Independent,
etc.), MA-specific behavior lives here instead of touching the parent file.
Every single-agent call site (run_nonstationary_opsrl.py,
compare_nine_agents.py) is completely unaffected by this file.

Each MAOPSRLAgent is one physical agent's own, fully independent OPSRL
brain -- its own posterior (N_sas/M_sa), own Q-table, own Thompson-sampled
exploration. Two of these are run side by side against one shared
TwoAgentRedBlueButtonEnv, each learning only from the single team reward the
environment returns each step. This mirrors the Independent AIF paradigm's
philosophy (the partner is folded into the environment's dynamics, not
co-planned or communicated with) -- which is why the observation each OPSRL
brain is given is deliberately the same set of factors Independent AIF and
the MAPPO comparator use as each agent's own, ego-relative observation (see
run_two_aif_agents_independent.py's two_agent_obs_to_single_agent_obs and
run_two_ppo_agents.py's OWN_OBS_KEYS): my position, the partner's position,
whether *I* am on the red/blue button, and both buttons' pressed state.
`game_result` and `button_just_pressed` are deliberately left out of the
discrete state, matching the existing single-agent OPSRLAgent._obs_to_state,
which already omits both -- OPSRL receives win/lose information through the
reward signal directly rather than needing it duplicated as a state factor,
and button_just_pressed is a transient one-step flag with no bearing on
optimal control.
"""
import numpy as np
import gymnasium.spaces as spaces

from agents.OPSRL.agent import OPSRLAgent


def _vectorized_backward_induction(Q, V, R, P, horizon, gamma):
    """
    Numerically identical to agents.OPSRL.utils.backward_induction_in_place
    (verified to ~1e-15 against it on randomized instances), but vectorized
    over (state, action) via a single matmul per stage instead of a Python-
    level double loop over s and a.

    This matters here specifically because MAOPSRLAgent's joint
    (my_pos, other_pos, on_red, on_blue, red_pressed, blue_pressed) state
    space is 9x larger than single-agent OPSRLAgent's (1296 vs 144 states on
    a 3x3 grid, from adding the partner's position) -- at that size the
    shared utility's O(H*S*A) Python-loop cost turns a sub-second call into
    multiple seconds, and this is called twice (once per agent) at the start
    of *every* episode. Kept local to this file rather than added to
    agents/OPSRL/utils.py to avoid touching code shared with Stage 1's
    already-reported single-agent results.
    """
    S, A = Q.shape[1], Q.shape[2]
    if R.ndim == 3:
        R = np.mean(R, axis=2)
    if P.ndim == 4:
        P = np.mean(P, axis=3)

    # Terminal-stage Q was previously never assigned here (only V was, via
    # np.max(R, axis=1) directly) -- see the identical fix and comment in
    # agents/OPSRL/utils.py::backward_induction_in_place, ai/02-debug.md
    # 2026-08-22 OPSRL entry. This file's own docstring claim of being
    # "numerically identical... verified to ~1e-15" against that shared
    # utility was true as far as it went -- both left Q[horizon-1] at zero
    # identically, so a diff-based check could never have caught this.
    Q[horizon - 1] = R
    V[horizon - 1, :] = np.max(R, axis=1)
    for hh in reversed(range(horizon - 1)):
        Q[hh] = R + gamma * (P.reshape(S * A, S) @ V[hh + 1]).reshape(S, A)
        V[hh] = Q[hh].max(axis=1)


def _vectorized_backward_induction_sd(Q, V, R, P, gamma):
    """Stage-dependent counterpart of _vectorized_backward_induction, same
    verified-identical relationship to agents.OPSRL.utils.backward_induction_sd."""
    H, S, A = Q.shape
    if R.ndim == 4:
        R = np.mean(R, axis=3)
    if P.ndim == 5:
        P = np.mean(P, axis=4)

    # See the identical fix and comment in _vectorized_backward_induction above.
    Q[H - 1] = R[H - 1, :, :]
    V[H - 1, :] = np.max(R[H - 1, :, :], axis=1)
    for hh in reversed(range(H - 1)):
        Q[hh] = R[hh] + gamma * (P[hh].reshape(S * A, S) @ V[hh + 1]).reshape(S, A)
        V[hh] = Q[hh].max(axis=1)


class _ActionSpaceShim:
    """
    Minimal stand-in for the `env` argument OPSRLAgent.__init__ reads at
    construction time (only .action_space, .width, .height are touched
    there). MAOPSRLAgent never calls self.env.reset()/.step() itself -- the
    run script owns the one shared TwoAgentRedBlueButtonEnv and drives both
    agents through it directly, exactly like the AIF and PPO two-agent
    runners already do. TwoAgentRedBlueButtonEnv's own .action_space is a
    spaces.Tuple of two Discrete(6)s (one per physical agent), which would
    fail OPSRLAgent's own `isinstance(..., spaces.Discrete)` assertion, so
    this shim exposes the single per-agent Discrete(6) space instead.
    """

    def __init__(self, width=3, height=3, n_actions=6):
        self.width = width
        self.height = height
        self.action_space = spaces.Discrete(n_actions)


class MAOPSRLAgent(OPSRLAgent):
    """OPSRL brain for one physical agent in TwoAgentRedBlueButton."""

    def __init__(self, width=3, height=3, n_actions=6, **kwargs):
        shim = _ActionSpaceShim(width=width, height=height, n_actions=n_actions)
        super().__init__(env=shim, **kwargs)

        # Parent __init__ sized N_sas/M_sa/V/Q from single-agent factors only
        # (n_positions * 2*2*2*2 = 144 states for a 3x3 grid). Widen to also
        # include the partner's position and re-run reset() so every array is
        # (re-)allocated at the correct, larger shape. reset() only reads
        # self.n_states/self.n_actions/self.horizon/self.prior_transition/
        # self.scale_prior_{reward,transition}, all already set by this
        # point, so calling it again post-hoc is safe.
        self.n_positions = width * height
        self.n_states = self.n_positions * self.n_positions * 2 * 2 * 2 * 2
        self.reset()

    def _obs_to_state(self, obs):
        """
        Convert this agent's own ego-relative observation dict to a discrete
        state integer.

        obs keys:
            my_pos, other_pos   -- flat grid indices, 0..n_positions-1
            on_red_button       -- is *this* agent on the red button
            on_blue_button      -- is *this* agent on the blue button
            red_button_pressed
            blue_button_pressed
        """
        if obs is None:
            return None
        my_pos = int(obs["my_pos"])
        other_pos = int(obs["other_pos"])
        on_red = int(obs.get("on_red_button", 0))
        on_blue = int(obs.get("on_blue_button", 0))
        red_pressed = int(obs.get("red_button_pressed", 0))
        blue_pressed = int(obs.get("blue_button_pressed", 0))

        state = my_pos
        state = state * self.n_positions + other_pos
        state = state * 2 + on_red
        state = state * 2 + on_blue
        state = state * 2 + red_pressed
        state = state * 2 + blue_pressed
        return state

    def resample_and_plan(self):
        """
        Draw one Thompson sample of the transition/reward posterior and run
        backward induction to get this episode's Q-table.

        This is exactly the preamble OPSRLAgent._run_episode() runs before
        interacting with its own single-agent env -- duplicated here (rather
        than factored into a shared method on OPSRLAgent) so this file never
        has to touch agents/OPSRL/agent.py. compare_nine_agents.py's
        run_opsrl_episode duplicates the identical preamble for the same
        underlying reason: _run_episode() owns its env's reset/step loop
        internally, which doesn't work once two agents must be stepped
        jointly through one shared TwoAgentRedBlueButtonEnv instance.

        Call once at the start of every episode, before stepping through it
        with _get_action()/_update().
        """
        B = self.thompson_samples

        M_sab_zero = np.repeat(self.M_sa[..., 0, np.newaxis], B, -1)
        M_sab_one = np.repeat(self.M_sa[..., 1, np.newaxis], B, -1)
        N_sasb = np.repeat(self.N_sas[..., np.newaxis], B, axis=-1)

        # Sample rewards from Beta distribution
        R_samples = self.rng.beta(M_sab_zero, M_sab_one)

        # Sample transitions from Dirichlet (via Gamma)
        P_samples = self.rng.gamma(N_sasb)
        P_samples = P_samples + 1e-10  # avoid exact zeros
        # Normalize over the "next state" axis, which is always second-to-last
        # regardless of stage_dependent -- the Thompson-sample axis B was
        # appended last by np.repeat(..., axis=-1) above. The former
        # stage_dependent=False branch used sum(-1) (the B axis, size 1 here),
        # which is a no-op division (P_samples / P_samples == 1.0 everywhere)
        # rather than real normalization -- see ai/02-debug.md, 2026-08-22
        # OPSRL entry, and the identical fix in agents/OPSRL/agent.py.
        sums = P_samples.sum(-2, keepdims=True)
        P_samples = P_samples / sums

        # Denormalize rewards back to [-1, 1] range
        R_samples = 2.0 * R_samples - 1.0

        if self.stage_dependent:
            _vectorized_backward_induction_sd(
                self.Q, self.V, R_samples, P_samples, self.gamma
            )
        else:
            _vectorized_backward_induction(
                self.Q, self.V, R_samples, P_samples,
                self.horizon, self.gamma,
            )

    def compute_recommended_policy(self):
        """
        Same computation as OPSRLAgent.fit()'s tail: derive Q_policy/V_policy
        from the posterior mean (no exploration bonus) rather than a fresh
        Thompson sample. Exposed standalone since MAOPSRLAgent's episodes are
        driven step-by-step by the run script rather than by fit(). Optional
        -- only needed if the eval-time greedy policy() method is used
        instead of the exploring _get_action() used during online training.
        """
        R_hat = self.M_sa[..., 0] / (self.M_sa[..., 0] + self.M_sa[..., 1])
        R_hat = 2.0 * R_hat - 1.0
        P_hat = self.N_sas / self.N_sas.sum(-1, keepdims=True)
        if self.stage_dependent:
            _vectorized_backward_induction_sd(
                self.Q_policy, self.V_policy, R_hat, P_hat, self.gamma
            )
        else:
            _vectorized_backward_induction(
                self.Q_policy, self.V_policy, R_hat, P_hat,
                self.horizon, self.gamma,
            )
