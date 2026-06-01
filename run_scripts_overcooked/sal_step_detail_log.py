"""
Rich per-step logs: map snapshots, state beliefs (qs), and policy beliefs (q_pi).

Used by ind / ic / fc sweep runners for:
  - stdout (--log-steps): human-readable map + belief tables + policy list
  - JSONL (--log-jsonl): machine-readable full q_pi and state_beliefs per step
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def _repo_logs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "logs"


def make_jsonl_path(
    log_dir: Path | str,
    paradigm: str,
    *,
    episode_seed: int,
    agent0_seed: int | None = None,
    agent1_seed: int | None = None,
    brain_seed: int | None = None,
) -> Path:
    from datetime import datetime

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if paradigm == "fc":
        name = "sal_fc_ep{}_brain{}_{}.jsonl".format(int(episode_seed), int(brain_seed), ts)
    else:
        name = "sal_{}_ep{}_a0_{}_a1_{}_{}.jsonl".format(
            paradigm,
            int(episode_seed),
            int(agent0_seed),
            int(agent1_seed),
            ts,
        )
    return log_dir / name


def map_lines(state, model_init, render_grid_fn) -> list[str]:
    return list(render_grid_fn(state, model_init))


def map_cell(lines: list[str]) -> str:
    """Single-cell friendly map (Excel / JSON); rows separated by newlines."""
    return "\n".join(lines)


def serialize_state_beliefs(agent) -> dict[str, Any]:
    import numpy as np

    qs = agent.get_state_beliefs()
    out: dict[str, Any] = {}
    for factor in agent.state_factors:
        p = np.asarray(qs[factor], dtype=float).reshape(-1)
        out[factor] = {
            "probabilities": [float(x) for x in p],
            "map_state": int(np.argmax(p)) if p.size else 0,
            "entropy": float(-np.sum(p * np.log(p + 1e-16))) if p.size else 0.0,
        }
    return out


def serialize_top_policies(
    agent,
    top_k: int,
    policy_label_fn: Callable[[int, Any], str],
) -> list[dict[str, Any]]:
    top = agent.get_top_policies(top_k=int(top_k))
    return [
        {
            "policy_idx": int(pidx),
            "label": policy_label_fn(int(pidx), agent),
            "prob": float(prob),
        }
        for (_pol, prob, pidx) in top
    ]


def serialize_q_pi(agent, policy_label_fn: Callable[[int, Any], str]) -> list[dict[str, Any]]:
    import numpy as np

    q_pi = np.asarray(agent.get_policy_posterior(), dtype=float).reshape(-1)
    return [
        {
            "policy_idx": int(i),
            "label": policy_label_fn(int(i), agent),
            "prob": float(q_pi[i]),
        }
        for i in range(int(q_pi.size))
    ]


def print_map(title: str, lines: list[str]) -> None:
    print("    {}:".format(title), flush=True)
    for row in lines:
        print("      {}".format(row), flush=True)


def print_policy_beliefs_stdout(
    title: str,
    agent,
    *,
    policy_label_fn: Callable[[int, Any], str],
    fmt_policy: Callable[[Any], str] | None = None,
    top_k: int = 5,
    label_width: int = 24,
    bar_width: int = 20,
) -> None:
    """Original runner style: entropy line, BEST, ranked list with bar chart."""
    import numpy as np

    q_pi = np.asarray(agent.get_policy_posterior(), dtype=float).reshape(-1)
    heading = title if title.endswith(":") else "{}:".format(title)
    print("    {}".format(heading), flush=True)
    if q_pi.size == 0:
        print("      (empty posterior)", flush=True)
        return
    H = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
    print("      entropy {:.3f}:".format(H), flush=True)
    top = agent.get_top_policies(top_k=int(top_k))
    if top:
        best_pol, best_prob, best_idx = top[0]
        if fmt_policy is not None:
            best_lbl = fmt_policy(best_pol)
        else:
            best_lbl = policy_label_fn(int(best_idx), agent)
        print(
            "      BEST: idx={} p={:.3f}  {}".format(
                int(best_idx), float(best_prob), best_lbl
            ),
            flush=True,
        )
    for rank, (pol, prob, pidx) in enumerate(top, 1):
        if fmt_policy is not None:
            lbl = fmt_policy(pol)
        else:
            lbl = policy_label_fn(int(pidx), agent)
        bar = "█" * int(float(prob) * int(bar_width))
        print(
            "        #{:d} [{:>{w}}] {:<20} {:.3f}".format(
                rank, lbl, bar, float(prob), w=int(label_width)
            ),
            flush=True,
        )


def print_policy_beliefs_indexed(
    title: str,
    agent,
    policy_label_fn: Callable[[int, Any], str],
    *,
    top_k: int = 20,
    full: bool = False,
) -> None:
    """Supplemental listing by policy index (used when --log-full-q-pi)."""
    import numpy as np

    q_pi = np.asarray(agent.get_policy_posterior(), dtype=float).reshape(-1)
    if q_pi.size == 0:
        return
    H = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
    n_show = int(q_pi.size) if full else min(int(top_k), int(q_pi.size))
    print(
        "    {} (by index, {} of {}): entropy {:.4f}".format(
            title, n_show, int(q_pi.size), H
        ),
        flush=True,
    )
    order = np.argsort(-q_pi)
    for rank, pidx in enumerate(order[:n_show], 1):
        p = float(q_pi[int(pidx)])
        lbl = policy_label_fn(int(pidx), agent)
        print(
            "      #{:4d}  idx={:4d}  p={:.6f}  {}".format(rank, int(pidx), p, lbl),
            flush=True,
        )


def print_agent_beliefs(
    agent,
    *,
    np_mod,
    model_init,
    agent_label: str,
    belief_table_fn: Callable[..., str],
    policy_label_fn: Callable[[int, Any], str],
    policy_top_k: int,
    policy_full: bool,
    state_belief_title: str | None = None,
    policy_belief_title: str | None = None,
    fmt_policy: Callable[[Any], str] | None = None,
    policy_label_width: int = 24,
) -> None:
    qs = agent.get_state_beliefs()
    print(
        belief_table_fn(
            np_mod,
            qs,
            model_init,
            title=state_belief_title or "Beliefs {}".format(agent_label),
        ),
        flush=True,
    )
    pol_title = policy_belief_title or "Policy beliefs {}".format(agent_label)
    print_policy_beliefs_stdout(
        pol_title,
        agent,
        policy_label_fn=policy_label_fn,
        fmt_policy=fmt_policy,
        top_k=policy_top_k,
        label_width=policy_label_width,
    )
    if policy_full:
        print_policy_beliefs_indexed(
            pol_title,
            agent,
            policy_label_fn,
            top_k=policy_top_k,
            full=True,
        )


class StepJsonlLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def open_jsonl(log_dir, paradigm: str, **seed_kwargs) -> StepJsonlLog:
    path = make_jsonl_path(log_dir, paradigm, **seed_kwargs)
    return StepJsonlLog(path)


def build_agent_payload(
    agent,
    *,
    policy_label_fn: Callable[[int, Any], str],
    policy_top_k: int,
    include_full_q_pi: bool,
    selected_policy_idx: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "state_beliefs": serialize_state_beliefs(agent),
        "top_policies": serialize_top_policies(agent, policy_top_k, policy_label_fn),
    }
    if selected_policy_idx is not None:
        payload["selected_policy_idx"] = int(selected_policy_idx)
    if include_full_q_pi:
        payload["q_pi"] = serialize_q_pi(agent, policy_label_fn)
    if extra:
        payload.update(extra)
    return payload


def write_ind_step(
    jsonl: StepJsonlLog | None,
    *,
    step: int,
    episode_seed: int,
    agent0_seed: int,
    agent1_seed: int,
    map_before: list[str],
    map_after: list[str],
    agent_0,
    agent_1,
    policy_label_fn_0: Callable[[int, Any], str],
    policy_label_fn_1: Callable[[int, Any], str],
    pol_idx_0: int,
    pol_idx_1: int,
    a0_prim: int,
    a1_prim: int,
    a0_prim_name: str,
    a1_prim_name: str,
    reward_a0: float,
    reward_a1: float,
    cumulative_reward_a0: float,
    cumulative_reward_a1: float,
    terminated: bool,
    truncated: bool,
    policy_top_k: int,
    include_full_q_pi: bool,
    meta_0: dict | None = None,
    meta_1: dict | None = None,
) -> None:
    if jsonl is None:
        return
    rec = {
        "paradigm": "ind",
        "step": int(step),
        "episode_seed": int(episode_seed),
        "agent0_seed": int(agent0_seed),
        "agent1_seed": int(agent1_seed),
        "map_before": map_cell(map_before),
        "map_after": map_cell(map_after),
        "agent_0": build_agent_payload(
            agent_0,
            policy_label_fn=policy_label_fn_0,
            policy_top_k=policy_top_k,
            include_full_q_pi=include_full_q_pi,
            selected_policy_idx=pol_idx_0,
            extra={
                "primitive": int(a0_prim),
                "primitive_name": a0_prim_name,
                "semantic_destination": (meta_0 or {}).get("destination"),
                "semantic_mode": (meta_0 or {}).get("mode"),
            },
        ),
        "agent_1": build_agent_payload(
            agent_1,
            policy_label_fn=policy_label_fn_1,
            policy_top_k=policy_top_k,
            include_full_q_pi=include_full_q_pi,
            selected_policy_idx=pol_idx_1,
            extra={
                "primitive": int(a1_prim),
                "primitive_name": a1_prim_name,
                "semantic_destination": (meta_1 or {}).get("destination"),
                "semantic_mode": (meta_1 or {}).get("mode"),
            },
        ),
        "reward_a0": float(reward_a0),
        "reward_a1": float(reward_a1),
        "cumulative_reward_a0": float(cumulative_reward_a0),
        "cumulative_reward_a1": float(cumulative_reward_a1),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    jsonl.write(rec)


def write_ic_step(
    jsonl: StepJsonlLog | None,
    *,
    step: int,
    episode_seed: int,
    agent0_seed: int,
    agent1_seed: int,
    map_before: list[str],
    map_after: list[str],
    agent_0,
    agent_1,
    joint_label_fn: Callable[[int, Any], str],
    n_semantic: int,
    pol_idx_0: int,
    pol_idx_1: int,
    ego0_sem: int,
    ego1_sem: int,
    a0_prim: int,
    a1_prim: int,
    a0_prim_name: str,
    a1_prim_name: str,
    reward_a0: float,
    reward_a1: float,
    cumulative_reward_a0: float,
    cumulative_reward_a1: float,
    terminated: bool,
    truncated: bool,
    policy_top_k: int,
    include_full_q_pi: bool,
    action_names: dict,
) -> None:
    if jsonl is None:
        return
    d0, m0 = _dest_mode_from_action_names(action_names, ego0_sem)
    d1, m1 = _dest_mode_from_action_names(action_names, ego1_sem)
    rec = {
        "paradigm": "ic",
        "step": int(step),
        "episode_seed": int(episode_seed),
        "agent0_seed": int(agent0_seed),
        "agent1_seed": int(agent1_seed),
        "n_semantic": int(n_semantic),
        "map_before": map_cell(map_before),
        "map_after": map_cell(map_after),
        "agent_0": build_agent_payload(
            agent_0,
            policy_label_fn=joint_label_fn,
            policy_top_k=policy_top_k,
            include_full_q_pi=include_full_q_pi,
            selected_policy_idx=pol_idx_0,
            extra={
                "ego_semantic_idx": int(ego0_sem),
                "joint_semantic_label": joint_label_fn(int(pol_idx_0), agent_0),
                "semantic_destination": d0,
                "semantic_mode": m0,
                "primitive": int(a0_prim),
                "primitive_name": a0_prim_name,
            },
        ),
        "agent_1": build_agent_payload(
            agent_1,
            policy_label_fn=joint_label_fn,
            policy_top_k=policy_top_k,
            include_full_q_pi=include_full_q_pi,
            selected_policy_idx=pol_idx_1,
            extra={
                "ego_semantic_idx": int(ego1_sem),
                "joint_semantic_label": joint_label_fn(int(pol_idx_1), agent_1),
                "semantic_destination": d1,
                "semantic_mode": m1,
                "primitive": int(a1_prim),
                "primitive_name": a1_prim_name,
            },
        ),
        "reward_a0": float(reward_a0),
        "reward_a1": float(reward_a1),
        "cumulative_reward_a0": float(cumulative_reward_a0),
        "cumulative_reward_a1": float(cumulative_reward_a1),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    jsonl.write(rec)


def write_fc_step(
    jsonl: StepJsonlLog | None,
    *,
    step: int,
    episode_seed: int,
    brain_seed: int,
    map_before: list[str],
    map_after: list[str],
    brain,
    joint_label_fn: Callable[[int, Any], str],
    joint_pairs: list,
    joint_idx: int,
    a_self_sem: int,
    a_other_sem: int,
    a0_prim: int,
    a1_prim: int,
    a0_prim_name: str,
    a1_prim_name: str,
    action_names: dict,
    reward_a0: float,
    reward_a1: float,
    cumulative_reward_a0: float,
    cumulative_reward_a1: float,
    terminated: bool,
    truncated: bool,
    policy_top_k: int,
    include_full_q_pi: bool,
) -> None:
    if jsonl is None:
        return
    name_s = action_names.get(int(a_self_sem), str(int(a_self_sem)))
    name_o = action_names.get(int(a_other_sem), str(int(a_other_sem)))
    rec = {
        "paradigm": "fc",
        "step": int(step),
        "episode_seed": int(episode_seed),
        "brain_seed": int(brain_seed),
        "map_before": map_cell(map_before),
        "map_after": map_cell(map_after),
        "brain": build_agent_payload(
            brain,
            policy_label_fn=joint_label_fn,
            policy_top_k=policy_top_k,
            include_full_q_pi=include_full_q_pi,
            selected_policy_idx=int(joint_idx),
            extra={
                "joint_semantic_self": name_s,
                "joint_semantic_other": name_o,
                "a0_semantic_idx": int(a_self_sem),
                "a1_semantic_idx": int(a_other_sem),
                "a0_primitive": int(a0_prim),
                "a0_primitive_name": a0_prim_name,
                "a1_primitive": int(a1_prim),
                "a1_primitive_name": a1_prim_name,
                "joint_pairs_count": len(joint_pairs),
            },
        ),
        "reward_a0": float(reward_a0),
        "reward_a1": float(reward_a1),
        "cumulative_reward_a0": float(cumulative_reward_a0),
        "cumulative_reward_a1": float(cumulative_reward_a1),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }
    jsonl.write(rec)


def _dest_mode_from_action_names(action_names: dict, sem_idx: int) -> tuple[str, str]:
    name = str(action_names.get(int(sem_idx), ""))
    if ":" in name:
        d, m = name.split(":", 1)
        return d, m
    return name, ""
