"""Two agents with signaling (Results, "Signaling can remove suboptimal equilibria";
SI section S8.1).

A strategy is a pair (eps, eps_switch): the exploration rate an agent uses until its
competitor reaches site d, and the rate it switches to after that signal. If both
agents reach site d on the same step, both keep their original rates.
"""

import numpy as np
from scipy.sparse import identity, kron
from scipy.sparse.linalg import splu
from tqdm import tqdm

from .single_agent import state_sites, transition_matrix
from .two_agent import reward_shares


def signaling_payoff_grid(d, eps_grid, gamma, R0=1.0, Rd=1.0, progress=False):
    """Both agents' payoffs for every pair of signaling strategies on a grid.

    Returns U1 and U2 of shape (m, m, m, m), indexed [eps1, eps1_switch, eps2,
    eps2_switch], where m = len(eps_grid).

    The joint chain has a transient phase, before either agent has reached site d,
    and three terminal phases, in which both rates are fixed (Section S8.1). Each
    terminal phase is an ordinary two-agent chain, so its values are solved once per
    pair of rates. The transient phase enters only through the discounted number of
    times the chain enters each terminal state from the start, which needs one
    solve per pair of initial rates. The payoff is then a constant plus a term that
    depends only on eps1_switch and a term that depends only on eps2_switch.
    """
    m = len(eps_grid)
    P = [transition_matrix(d, eps) for eps in eps_grid]
    n = P[0].shape[0]

    shares = reward_shares(d)
    rewards = np.column_stack(
        [R0 * shares[:, 0] + Rd * shares[:, 1], R0 * shares[:, 2] + Rd * shares[:, 3]]
    )

    # Phase of each joint state: 0 neither agent at d, 1 only agent 1, 2 only agent 2,
    # 3 both. The start state, index 0, is in phase 0.
    at_d = state_sites(d) == d
    phase = np.add.outer(at_d, 2 * at_d).ravel()
    phase_idx = [np.flatnonzero(phase == k) for k in range(4)]
    transient = phase_idx[0]

    # Terminal values V[i, j] for fixed rates (eps_grid[i], eps_grid[j]), kept only on
    # the states of phases 1-3, where the chain enters them.
    eye = identity(n * n, format="csc")
    terminal = [np.empty((m, m, len(idx), 2)) for idx in phase_idx[1:]]
    for i in tqdm(range(m), desc="terminal phases", disable=not progress):
        for j in range(m):
            K = kron(P[i], P[j], format="csc")
            V = splu(eye - gamma * K).solve(rewards)
            for k, idx in enumerate(phase_idx[1:]):
                terminal[k][i, j] = V[idx]
    V1, V2, V3 = terminal

    start = np.zeros(len(transient))
    start[0] = 1.0
    eye0 = identity(len(transient), format="csc")

    U = np.empty((2, m, m, m, m))
    for b1 in tqdm(range(m), desc="transient phase", disable=not progress):
        for b2 in range(m):
            K0 = kron(P[b1], P[b2], format="csr")[transient]
            # Discounted visits to each transient state from the start state.
            w = splu((eye0 - gamma * K0[:, transient]).tocsc()).solve(start, trans="T")
            entry1, entry2, entry3 = (
                gamma * (K0[:, idx].T @ w) for idx in phase_idx[1:]
            )

            # Agent 1 first: agent 2 switches, so V1 varies with eps2_switch.
            by_eps2_switch = np.einsum("jsp,s->pj", V1[b1], entry1)
            # Agent 2 first: agent 1 switches, so V2 varies with eps1_switch.
            by_eps1_switch = np.einsum("isp,s->pi", V2[:, b2], entry2)
            constant = w @ rewards[transient] + entry3 @ V3[b1, b2]

            U[:, b1, :, b2, :] = (
                constant[:, None, None]
                + by_eps1_switch[:, :, None]
                + by_eps2_switch[:, None, :]
            )

    return U[0], U[1]
