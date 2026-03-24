"""
ActiveInferenceFixedPolicies agent.

This package exists so we can extend Active Inference behavior without changing
the core `agents/ActiveInference` implementation.

Change vs core:
- The `infer_states` fast path also supports modalities that map to a single
  hidden factor via `observation_state_dependencies`, even if the modality name
  does not follow the "{factor}_obs" convention (e.g. soup_delivered_obs -> ck_delivered).
"""

import numpy as np

from agents.ActiveInference.agent import Agent as _BaseAgent
from agents.ActiveInference import control, inference


class Agent(_BaseAgent):
    @staticmethod
    def _get_expected_states_structured(B_fn, qs_current, policy, env_params):
        """
        Roll out beliefs for policies whose steps may be non-scalars
        (e.g. tuple step-actions such as (a0, a1)).
        """
        if np.isscalar(policy):
            policy = [int(policy)]

        qs_pred = []
        qs_t = qs_current
        for action in policy:
            action_for_b = int(action) if np.isscalar(action) else action
            qs_next = B_fn(qs_t, action_for_b, **env_params)
            qs_pred.append(qs_next)
            qs_t = qs_next
        return qs_pred

    def infer_states(self, obs_dict):
        if self.inference_algorithm != "VANILLA":
            raise NotImplementedError("Only VANILLA inference supported")

        # Get prior
        if self.action is not None:
            prior_dict = control.get_expected_state(
                self.B_fn, self.qs, self.action, self.env_params
            )
        else:
            prior_dict = {factor: self.qs[factor].copy() for factor in self.state_factors}

        # Fast path setup
        map_idx = {f: int(np.argmax(prior_dict[f])) for f in self.state_factors}
        qs_fast = {f: prior_dict[f].copy() for f in self.state_factors}

        direct_modalities = []

        # 1) Standard "{factor}_obs" direct observations
        for modality in obs_dict.keys():
            if not modality.endswith("_obs"):
                continue
            factor = modality[: -len("_obs")]
            if factor in self.state_sizes and modality in self.observation_labels:
                if len(self.observation_labels[modality]) == self.state_sizes[factor]:
                    direct_modalities.append((factor, modality))

        # 2) Single-dependency modalities (e.g. soup_delivered_obs -> ck_delivered)
        if self.observation_state_dependencies is not None:
            for modality in obs_dict.keys():
                deps = self.observation_state_dependencies.get(modality, None)
                if not deps or len(deps) != 1:
                    continue
                factor = deps[0]
                if factor not in self.state_sizes:
                    continue
                if modality not in self.observation_labels:
                    continue
                if len(self.observation_labels[modality]) != self.state_sizes[factor]:
                    continue
                if any(f == factor for f, _ in direct_modalities):
                    continue
                direct_modalities.append((factor, modality))

        # Bayes update each directly observed factor
        for factor, modality in direct_modalities:
            obs_idx = int(obs_dict[modality])
            S = self.state_sizes[factor]
            like = np.zeros(S, dtype=float)
            s_idx = map_idx.copy()
            for v in range(S):
                s_idx[factor] = int(v)
                p_o = self.A_fn(s_idx)[modality]
                if 0 <= obs_idx < len(p_o):
                    like[v] = float(p_o[obs_idx])
            # IMPORTANT: With one-hot priors (hard zeros), Bayes rule cannot move mass
            # to the observed state. For *direct sensor* modalities (factor_obs),
            # we clamp to the likelihood (i.e., let the observation dominate).
            #
            # For nonstandard single-dependency modalities (e.g. soup_delivered_obs),
            # we still use a lightly softened prior.
            if modality == f"{factor}_obs":
                post = like
            else:
                prior = np.asarray(qs_fast[factor], dtype=float)
                if prior.size:
                    eps = 1e-3
                    prior = (1.0 - eps) * prior + eps * (np.ones_like(prior) / float(prior.size))
                post = prior * like
            z = float(post.sum())
            qs_fast[factor] = post / (z if z > 0 else 1.0)

        # Skip VI if all informative modalities are covered by the fast path
        informative_modalities = []
        for m in obs_dict.keys():
            if not m.endswith("_obs"):
                continue
            if m[: -len("_obs")] in self.state_sizes:
                informative_modalities.append(m)
                continue
            if self.observation_state_dependencies is not None:
                deps = self.observation_state_dependencies.get(m, None)
                if deps and len(deps) == 1 and deps[0] in self.state_sizes:
                    informative_modalities.append(m)

        if informative_modalities and all(
            any(m == dm for _, dm in direct_modalities) for m in informative_modalities
        ):
            self.qs = qs_fast
        else:
            self.qs = inference.vanilla_fpi_update_posterior_states(
                self.A_fn,
                obs_dict,
                prior_dict,
                self.state_factors,
                self.state_sizes,
                num_iter=16,
                dF_tol=self.dF_tol,
                debug=False,
            )

        return self.qs

    def infer_policies(self):
        """
        Same as base Agent.infer_policies, but uses a structured-action-compatible
        rollout for policy steps (supports tuple actions).
        """
        if self.inference_algorithm != "VANILLA":
            raise NotImplementedError("Only VANILLA inference supported")

        original_get_expected_states = control.get_expected_states
        control.get_expected_states = self._get_expected_states_structured
        try:
            result = control.vanilla_fpi_update_posterior_policies(
                self.qs,
                self.A_fn,
                self.B_fn,
                self.C_fn,
                self.policies,
                self.env_params,
                self.state_factors,
                self.state_sizes,
                self.observation_labels,
                observation_state_dependencies=self.observation_state_dependencies,
                use_utility=self.use_utility,
                use_states_info_gain=self.use_states_info_gain,
                E=self.E,
                gamma=self.gamma,
                return_policy_details=True,
            )
        finally:
            control.get_expected_states = original_get_expected_states

        if len(result) == 3:
            self.q_pi, G, self._last_policy_details = result
        else:
            self.q_pi, G = result
            self._last_policy_details = None

        return self.q_pi, G

