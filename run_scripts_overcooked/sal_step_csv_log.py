"""
Per-step CSV logs for semantic-action-level Overcooked runners (ind / ic / fc).

CSV files open directly in Excel. One row per env primitive step.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


def _repo_logs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "logs"


def make_log_path(
    log_dir: Path | str,
    paradigm: str,
    *,
    episode_seed: int,
    agent0_seed: int | None = None,
    agent1_seed: int | None = None,
    brain_seed: int | None = None,
) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if paradigm == "fc":
        name = "sal_fc_ep{}_brain{}_{}.csv".format(int(episode_seed), int(brain_seed), ts)
    else:
        name = "sal_{}_ep{}_a0_{}_a1_{}_{}.csv".format(
            paradigm,
            int(episode_seed),
            int(agent0_seed),
            int(agent1_seed),
            ts,
        )
    return log_dir / name


def _policy_stats(agent, fmt_policy) -> dict[str, Any]:
    import numpy as np

    q_pi = np.asarray(agent.get_policy_posterior(), dtype=float).reshape(-1)
    if q_pi.size == 0:
        return {
            "policy_idx": "",
            "q_pi_entropy": "",
            "top_policy_prob": "",
            "top_policy_plan": "",
        }
    H = float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
    top = agent.get_top_policies(top_k=1)
    if top:
        pol, prob, pidx = top[0]
        return {
            "policy_idx": int(pidx),
            "q_pi_entropy": round(H, 6),
            "top_policy_prob": round(float(prob), 6),
            "top_policy_plan": fmt_policy(pol),
        }
    return {
        "policy_idx": int(np.argmax(q_pi)),
        "q_pi_entropy": round(H, 6),
        "top_policy_prob": round(float(np.max(q_pi)), 6),
        "top_policy_plan": "",
    }


def semantic_dest_mode(action_names: dict, idx: int) -> tuple[str, str]:
    """Parse model_init ACTION_NAMES entry 'dst:mode' into destination and mode."""
    name = str(action_names.get(int(idx), ""))
    if ":" in name:
        d, m = name.split(":", 1)
        return d, m
    return name, ""


def _meta_cols(meta, prefix: str) -> dict[str, str]:
    if not meta:
        return {
            "{}_semantic_destination".format(prefix): "",
            "{}_semantic_mode".format(prefix): "",
        }
    return {
        "{}_semantic_destination".format(prefix): str(meta.get("destination", "")),
        "{}_semantic_mode".format(prefix): str(meta.get("mode", "")),
    }


class StepCsvLog:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=fieldnames, extrasaction="ignore")
        self._writer.writeheader()
        self._fh.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


IND_FIELDS = [
    "paradigm",
    "episode_seed",
    "agent0_seed",
    "agent1_seed",
    "step",
    "a0_policy_idx",
    "a0_semantic_destination",
    "a0_semantic_mode",
    "a0_primitive",
    "a0_primitive_name",
    "a0_q_pi_entropy",
    "a0_top_policy_prob",
    "a0_top_policy_plan",
    "a1_policy_idx",
    "a1_semantic_destination",
    "a1_semantic_mode",
    "a1_primitive",
    "a1_primitive_name",
    "a1_q_pi_entropy",
    "a1_top_policy_prob",
    "a1_top_policy_plan",
    "reward_a0",
    "reward_a1",
    "cumulative_reward_a0",
    "cumulative_reward_a1",
    "terminated",
    "truncated",
]

IC_FIELDS = [
    "paradigm",
    "episode_seed",
    "agent0_seed",
    "agent1_seed",
    "step",
    "a0_joint_policy_idx",
    "a0_joint_semantic_label",
    "a0_ego_semantic_idx",
    "a0_semantic_destination",
    "a0_semantic_mode",
    "a0_primitive",
    "a0_primitive_name",
    "a0_q_pi_entropy",
    "a0_top_policy_prob",
    "a1_joint_policy_idx",
    "a1_joint_semantic_label",
    "a1_ego_semantic_idx",
    "a1_semantic_destination",
    "a1_semantic_mode",
    "a1_primitive",
    "a1_primitive_name",
    "a1_q_pi_entropy",
    "a1_top_policy_prob",
    "reward_a0",
    "reward_a1",
    "cumulative_reward_a0",
    "cumulative_reward_a1",
    "terminated",
    "truncated",
]

FC_FIELDS = [
    "paradigm",
    "episode_seed",
    "brain_seed",
    "step",
    "joint_policy_idx",
    "joint_semantic_self",
    "joint_semantic_other",
    "a0_semantic_idx",
    "a1_semantic_idx",
    "a0_primitive",
    "a0_primitive_name",
    "a1_primitive",
    "a1_primitive_name",
    "brain_q_pi_entropy",
    "brain_top_policy_prob",
    "reward_a0",
    "reward_a1",
    "cumulative_reward_a0",
    "cumulative_reward_a1",
    "terminated",
    "truncated",
]


def open_ind_log(log_dir, episode_seed, agent0_seed, agent1_seed) -> StepCsvLog:
    path = make_log_path(
        log_dir,
        "ind",
        episode_seed=episode_seed,
        agent0_seed=agent0_seed,
        agent1_seed=agent1_seed,
    )
    return StepCsvLog(path, IND_FIELDS)


def open_ic_log(log_dir, episode_seed, agent0_seed, agent1_seed) -> StepCsvLog:
    path = make_log_path(
        log_dir,
        "ic",
        episode_seed=episode_seed,
        agent0_seed=agent0_seed,
        agent1_seed=agent1_seed,
    )
    return StepCsvLog(path, IC_FIELDS)


def open_fc_log(log_dir, episode_seed, brain_seed) -> StepCsvLog:
    path = make_log_path(
        log_dir,
        "fc",
        episode_seed=episode_seed,
        brain_seed=brain_seed,
    )
    return StepCsvLog(path, FC_FIELDS)
