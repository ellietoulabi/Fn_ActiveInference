"""
Utility functions for OPSRL agent: backward induction algorithms.
"""
import numpy as np


def backward_induction_in_place(Q, V, R, P, horizon, gamma, v_max):
    """
    Backward induction for finite-horizon MDP with stationary transitions.
    
    Updates Q and V arrays in place.
    
    Parameters
    ----------
    Q : ndarray, shape (H, S, A)
        Q-function to update
    V : ndarray, shape (H, S)
        Value function to update
    R : ndarray, shape (S, A) or (S, A, B)
        Reward function. If shape is (S, A, B), B is the number of samples.
    P : ndarray, shape (S, A, S) or (S, A, S, B)
        Transition probabilities. If shape is (S, A, S, B), B is the number of samples.
    horizon : int
        Horizon length
    gamma : float
        Discount factor
    v_max : float
        Maximum possible value (for clipping)
    """
    S, A = Q.shape[1], Q.shape[2]
    
    # Handle multiple samples by averaging R and P
    if R.ndim == 3:
        R = np.mean(R, axis=2)  # Average over samples: (S, A, B) -> (S, A)
    if P.ndim == 4:
        P = np.mean(P, axis=3)  # Average over samples: (S, A, S, B) -> (S, A, S)
    
    # Initialize V at horizon
    V[horizon - 1, :] = np.max(R, axis=1)
    
    # Backward induction
    for hh in reversed(range(horizon - 1)):
        for s in range(S):
            for a in range(A):
                Q[hh, s, a] = R[s, a] + gamma * np.dot(P[s, a, :], V[hh + 1, :])
            V[hh, s] = np.max(Q[hh, s, :])


def backward_induction_sd(Q, V, R, P, gamma, v_max):
    """
    Backward induction for stage-dependent MDP.
    
    Updates Q and V arrays in place.
    
    Parameters
    ----------
    Q : ndarray, shape (H, S, A)
        Q-function to update
    V : ndarray, shape (H, S)
        Value function to update
    R : ndarray, shape (H, S, A) or (H, S, A, B)
        Stage-dependent reward function
    P : ndarray, shape (H, S, A, S) or (H, S, A, S, B)
        Stage-dependent transition probabilities
    gamma : float
        Discount factor
    v_max : float
        Maximum possible value (for clipping)
    """
    H, S, A = Q.shape
    
    # Handle multiple samples by averaging R and P
    if R.ndim == 4:
        R = np.mean(R, axis=3)  # Average over samples: (H, S, A, B) -> (H, S, A)
    if P.ndim == 5:
        P = np.mean(P, axis=4)  # Average over samples: (H, S, A, S, B) -> (H, S, A, S)
    
    # Initialize V at horizon
    V[H - 1, :] = np.max(R[H - 1, :, :], axis=1)
    
    # Backward induction
    for hh in reversed(range(H - 1)):
        for s in range(S):
            for a in range(A):
                Q[hh, s, a] = R[hh, s, a] + gamma * np.dot(P[hh, s, a, :], V[hh + 1, :])
            V[hh, s] = np.max(Q[hh, s, :])

