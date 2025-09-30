import numpy as np
from functools import lru_cache


class ModalityA:
    """
    Matrix-like adapter for a functional likelihood A_m.

    Exposes:
      - shape: (num_outcomes, *state_sizes)
      - __getitem__((slice(None), i1, i2, ..., ik)) -> np.ndarray (num_outcomes,)

    The wrapped function f must accept k integer indices (one per state factor)
    and return a 1D probability vector of length num_outcomes.
    """

    def __init__(self, outcome_count: int, state_sizes: list[int], f_prob_vector):
        self.shape = (int(outcome_count), *[int(s) for s in state_sizes])
        self._f = f_prob_vector
        self._O = int(outcome_count)

    @lru_cache(maxsize=None)
    def _eval(self, state_idx_tuple: tuple[int, ...]) -> np.ndarray:
        probs = self._f(*state_idx_tuple)
        arr = np.asarray(probs, dtype=np.float64)
        if arr.shape != (self._O,):
            raise ValueError(
                f"Functional A returned shape {arr.shape}, expected ({self._O},)"
            )
        total = arr.sum()
        if not (total == 0.0 or np.isclose(total, 1.0)):
            arr = arr / total
        return arr

    def __getitem__(self, index):
        # Expect (slice over outcomes, i1, i2, ..., ik)
        if not isinstance(index, tuple) or len(index) < 2:
            raise IndexError("Expected tuple index: (slice(None), i1, i2, ..., ik)")
        first, *state_idx = index
        if not isinstance(first, slice):
            raise IndexError("First index must be a slice over outcomes")
        return self._eval(tuple(state_idx))


def build_functional_A(state_sizes: list[int]):
    """
    Build an object-array A compatible with pymdp-style APIs for the RedBlueButton model.

    State factor order (indices expected by functions):
      0: agent_pos           (0..S-1)
      1: red_button_pos      (0..S-1)
      2: blue_button_pos     (0..S-1)
      3: red_button_state    (0:not_pressed, 1:pressed)
      4: blue_button_state   (0:not_pressed, 1:pressed)
      5: goal_context        (0:red_then_blue, 1:blue_then_red)

    Modalities included (outcome sizes):
      - agent_pos           (S)
      - on_red_button       (2)
      - on_blue_button      (2)
      - red_button_state    (2)
      - blue_button_state   (2)
      - game_result         (3)  [neutral, win, lose]

    Returns
    -------
    np.ndarray[dtype=object]
        Array of ModalityA objects, one per modality, in the order above.
    """

    S = int(state_sizes[0])

    def f_agent_pos(i_agent, *rest):
        v = np.zeros(S, dtype=np.float64)
        v[i_agent] = 1.0
        return v

    def f_on_red_button(i_agent, i_red_pos, *rest):
        on = float(i_agent == i_red_pos)
        return np.array([1.0 - on, on], dtype=np.float64)

    def f_on_blue_button(i_agent, _i_red_pos, i_blue_pos, *rest):
        on = float(i_agent == i_blue_pos)
        return np.array([1.0 - on, on], dtype=np.float64)

    def f_red_button_state(_i_agent, _i_red_pos, _i_blue_pos, red_state, *rest):
        return np.array([1.0, 0.0], dtype=np.float64) if red_state == 0 else np.array([0.0, 1.0], dtype=np.float64)

    def f_blue_button_state(_i_agent, _i_red_pos, _i_blue_pos, _red_state, blue_state, *rest):
        return np.array([1.0, 0.0], dtype=np.float64) if blue_state == 0 else np.array([0.0, 1.0], dtype=np.float64)

    def f_game_result(_i_agent, _i_red_pos, _i_blue_pos, red_state, blue_state, goal_ctx):
        both_pressed = (red_state == 1 and blue_state == 1)
        if not both_pressed:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)  # neutral
        # both pressed: win/lose depends on goal context
        if goal_ctx == 0:  # red_then_blue
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)  # win
        else:  # blue_then_red
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)  # lose

    modalities = [
        ModalityA(S, state_sizes, f_agent_pos),
        ModalityA(2, state_sizes, f_on_red_button),
        ModalityA(2, state_sizes, f_on_blue_button),
        ModalityA(2, state_sizes, f_red_button_state),
        ModalityA(2, state_sizes, f_blue_button_state),
        ModalityA(3, state_sizes, f_game_result),
    ]

    return np.array(modalities, dtype=object)


# Convenience export with names
def build_named_functional_A(state_sizes: list[int]):
    A_array = build_functional_A(state_sizes)
    names = [
        "agent_pos",
        "on_red_button",
        "on_blue_button",
        "red_button_state",
        "blue_button_state",
        "game_result",
    ]
    return {name: A_array[idx] for idx, name in enumerate(names)}

#here