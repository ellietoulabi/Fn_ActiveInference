import numpy as np
from . import utils
from . import control
from . import inference


class FunctionalAgent:
    def __init__(
        self,
        A_funcs,              # dict of modality -> function(state_tuple, …) -> obs dist
        B_func,               # functional B: apply_B(qs, action, width, height)
        C_funcs,              # dict of modality -> preference function returning vector
        D_funcs,              # list/array of priors (factorized)
        policies=None,        # list of action sequences
        actions=None,         # list of available actions
        decode_table=None,    # joint-state decode table
        env_params=None,      # dict with width, height, etc.
        gamma=16.0,
        alpha=16.0,
        policy_len=1,
        inference_horizon=1,
        action_selection="deterministic",
        sampling_mode="full",
        inference_algorithm="VANILLA",
    ):
        """
        Functional Active Inference Agent (no matrices)
        """

        # Functional model
        self.A_funcs = A_funcs
        self.B_func = B_func
        self.C_funcs = C_funcs
        self.D = D_funcs
        self.decode_table = decode_table
        self.env_params = env_params

        # Actions & policies
        self.actions = list(range(len(actions))) if actions is not None else list(range(6))
        self.policies = np.array(policies, dtype=object) if policies is not None else self._construct_policies(policy_len)

        # Hyperparameters
        self.gamma = gamma
        self.alpha = alpha
        self.policy_len = policy_len
        self.inference_horizon = inference_horizon
        self.action_selection = action_selection
        self.sampling_mode = sampling_mode
        self.inference_algorithm = inference_algorithm
        self.inference_params = self._get_default_params()

        # Flags
        self.use_utility = True
        self.use_states_info_gain = True
        self.E = None  # policy prior

        # Internal state
        self.qs = None
        self.q_pi = None
        self.action = None
        self.prev_actions = None
        self.curr_timestep = 0

        self.reset()

    # ---------------------------
    # Reset
    # ---------------------------
    def reset(self):
        """Reset beliefs and counters"""
        self.curr_timestep = 0
        self.qs = self.D.copy()
        self.q_pi = np.ones(len(self.policies)) / len(self.policies)
        self.action = None
        self.prev_actions = None

    # ---------------------------
    # Inference over states
    # ---------------------------
    def infer_states(self, obs):
        if self.inference_algorithm != "VANILLA":
            raise NotImplementedError("Only VANILLA inference supported")

        if self.action is not None:
            prior = control.get_expected_state(self.qs, self.action, self.env_params)
        else:
            prior = self.D.copy()

        # Call functional inference
        self.qs = inference.vanilla_fpi_update_posterior_states(
            self.A_funcs,
            obs,
            prior,
            self.qs,
            self.decode_table,
            self.env_params["obs_sizes"],
            **self.inference_params,
        )
        return self.qs

    # ---------------------------
    # Inference over policies
    # ---------------------------
    def infer_policies(self):
        if self.inference_algorithm != "VANILLA":
            raise NotImplementedError("Only VANILLA inference supported")

        q_pi, G = control.vanilla_fpi_update_posterior_policies(
            self.qs,
            self.A_funcs,
            self.B_func,
            self.C_funcs,
            self.policies,
            self.decode_table,
            self.env_params,
            use_utility=self.use_utility,
            use_states_info_gain=self.use_states_info_gain,
            E=self.E,
            gamma=self.gamma,
        )
        self.q_pi = q_pi
        return self.q_pi, G

    # ---------------------------
    # Action selection
    # ---------------------------
    def sample_action(self):
        if self.sampling_mode == "marginal":
            self.action = control.sample_action(
                self.q_pi, self.policies, self.action_selection, self.alpha, self.actions
            )
        elif self.sampling_mode == "full":
            self.action = control.sample_policy(
                self.q_pi, self.policies, self.action_selection, self.alpha
            )
        else:
            raise NotImplementedError(f"Sampling mode {self.sampling_mode} not implemented")
        return self.action

    # ---------------------------
    # Step time
    # ---------------------------
    def step_time(self):
        if self.prev_actions is None:
            self.prev_actions = [self.action]
        else:
            self.prev_actions.append(self.action)
        self.curr_timestep += 1
        return self.curr_timestep

    # ---------------------------
    # Helpers
    # ---------------------------
    def _construct_policies(self, policy_len):
        return utils.construct_policies(self.actions, policy_len)

    def _get_default_params(self):
        if self.inference_algorithm == "VANILLA":
            return {"num_iter": 10, "dF": 1.0, "dF_tol": 0.001}
        return None
