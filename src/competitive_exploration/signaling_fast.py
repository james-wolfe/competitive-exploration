"""The signaling game on fine grids, without the four-index payoff arrays.

signaling.signaling_payoff_grid returns each agent's payoff U[i, a, j, b] for every
profile of agent 1's initial and post-signal rates (i, a) and agent 2's (j, b). As its
docstring notes, the payoff separates:

    U[i, a, j, b] = C[i, j] + A[i, j, a] + B[i, j, b],

where C collects the transient phase and the phase in which both agents reach site d
together, A the phase in which agent 2 reaches d first (so agent 1 switches to rate a),
and B the phase in which agent 1 reaches d first (so agent 2 switches to rate b). The
arrays C, A and B take m^3 memory, where m is the number of rates, instead of the m^4 of
U, which makes grids of several hundred rates feasible.

signaling_summary_fast returns the same five quantities as
equilibria.signaling_summary, with the same definitions and tolerances:
- a Nash equilibrium is a profile at which each agent's payoff is within the tolerance
  of the largest payoff it can obtain by changing both of its rates
  (equilibria.nash_4d);
- a locally stable profile is one at which each agent's payoff is within the
  tolerance of the largest payoff in the 3 x 3 block of one-step changes of its own
  two rates (equilibria.locally_stable_4d).
Both are found without enumerating all m^4 profiles. Nash equilibria: agent 1's best
payoff against (j, b) is max_i (C[i, j] + max_a A[i, j, a] + B[i, j, b]), so only the
pairs (i, j) at which some (a, b) can satisfy both agents need to be examined.
Locally stable profiles: a change of the post-signal rate alone changes only A (for
agent 1) or B (for agent 2), so a locally stable profile has a at a local maximum of
A[i, j, :] and b at a local maximum of B[i, j, :], up to the tolerance; only those
profiles are checked against the full 3 x 3 neighborhoods.
"""

import numpy as np
from scipy.sparse import identity, kron
from scipy.sparse.linalg import splu
from tqdm import tqdm

from .single_agent import state_sites, transition_matrix
from .two_agent import reward_shares


def signaling_components(d, eps_grid, gamma, progress=False):
    """Payoff components C, A, B of the signaling game, for the near reward alone
    (index 0 of the first axis: R0 = 1, Rd = 0) and the distant reward alone (index 1:
    R0 = 0, Rd = 1). Shapes: C (2, 2, m, m), A and B (2, 2, m, m, m), indexed
    [reward, agent, i, j, (a or b)]. The payoff for rewards (R0, Rd) is R0 times the
    first and Rd times the second. The construction is that of
    signaling.signaling_payoff_grid, with both reward vectors solved together."""
    m = len(eps_grid)
    P = [transition_matrix(d, eps) for eps in eps_grid]
    n = P[0].shape[0]

    shares = reward_shares(d)
    # Columns: agent 1 and agent 2 for the near reward, then for the distant reward.
    rewards = shares[:, [0, 2, 1, 3]]

    at_d = state_sites(d) == d
    phase = np.add.outer(at_d, 2 * at_d).ravel()
    phase_idx = [np.flatnonzero(phase == k) for k in range(4)]
    transient = phase_idx[0]

    eye = identity(n * n, format="csc")
    terminal = [np.empty((m, m, len(idx), 4)) for idx in phase_idx[1:]]
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

    C = np.empty((4, m, m))
    A = np.empty((4, m, m, m))
    B = np.empty((4, m, m, m))
    for b1 in tqdm(range(m), desc="transient phase", disable=not progress):
        for b2 in range(m):
            K0 = kron(P[b1], P[b2], format="csr")[transient]
            w = splu((eye0 - gamma * K0[:, transient]).tocsc()).solve(start, trans="T")
            entry1, entry2, entry3 = (
                gamma * (K0[:, idx].T @ w) for idx in phase_idx[1:]
            )
            B[:, b1, b2, :] = np.einsum("jsp,s->pj", V1[b1], entry1)
            A[:, b1, b2, :] = np.einsum("isp,s->pi", V2[:, b2], entry2)
            C[:, b1, b2] = w @ rewards[transient] + entry3 @ V3[b1, b2]

    split = lambda X: X.reshape((2, 2) + X.shape[1:])
    return split(C), split(A), split(B)


def combine(components, R0, Rd):
    """C, A, B for rewards (R0, Rd) from the output of signaling_components."""
    return tuple(R0 * X[0] + Rd * X[1] for X in components)


def _close(x, target, rtol, atol):
    """np.isclose(x, target), written out so that it broadcasts without copies."""
    return np.abs(x - target) <= atol + rtol * np.abs(target)


def _local_max_1d(x, tol):
    """x[..., k] is within tol of its neighbors x[..., k - 1] and x[..., k + 1]."""
    ok = np.ones(x.shape, dtype=bool)
    ok[..., 1:] &= x[..., 1:] >= x[..., :-1] - tol
    ok[..., :-1] &= x[..., :-1] >= x[..., 1:] - tol
    return ok


def _nash(C, A, B, rtol, atol):
    """Number of Nash profiles and the smallest collective payoff among them."""
    (c1, c2), (A1, A2), (B1, B2) = C, A, B
    X = c1[:, :, None] + A1.max(axis=2)[:, :, None] + B1  # [i, j, b]: agent 1 best over a
    M1 = X.max(axis=0)                                     # [j, b]: agent 1 best over (i, a)
    Y = c2[:, :, None] + B2.max(axis=2)[:, :, None] + A2   # [i, j, a]: agent 2 best over b
    M2 = Y.max(axis=1)                                     # [i, a]: agent 2 best over (j, b)
    slack = 1e-12 * (np.abs(M1).max() + np.abs(M2).max())  # rounding of the decomposition (~1e-16)
    near1 = (X >= M1[None] - (atol + rtol * np.abs(M1[None])) - slack).any(axis=2)
    near2 = (Y >= M2[:, None] - (atol + rtol * np.abs(M2[:, None])) - slack).any(axis=1)
    count, worst = 0, np.inf
    for i, j in np.argwhere(near1 & near2):
        U1 = c1[i, j] + A1[i, j][:, None] + B1[i, j][None, :]  # [a, b]
        U2 = c2[i, j] + A2[i, j][:, None] + B2[i, j][None, :]
        ok = _close(U1, M1[j][None, :], rtol, atol) & _close(U2, M2[i][:, None], rtol, atol)
        if ok.any():
            count += int(ok.sum())
            worst = min(worst, float((U1 + U2)[ok].min()))
    return count, worst


def _local(C, A, B, rtol, atol, chunk=2_000_000):
    """Number of locally stable profiles and the largest collective payoff among them."""
    (c1, c2), (A1, A2), (B1, B2) = C, A, B
    m = c1.shape[0]
    bound = max(np.abs(c1).max() + np.abs(A1).max() + np.abs(B1).max(),
                np.abs(c2).max() + np.abs(A2).max() + np.abs(B2).max())
    loose = atol + rtol * bound + 1e-12 * bound  # covers rounding of the decomposition (~1e-16)
    P1 = _local_max_1d(A1, loose)  # [i, j, a]
    P2 = _local_max_1d(B2, loose)  # [i, j, b]

    ia = np.argwhere(P1).astype(np.int32)  # rows sorted by (i, j)
    ib = np.argwhere(P2).astype(np.int32)
    del P1, P2
    key_a = ia[:, 0].astype(np.int64) * m + ia[:, 1]
    key_b = ib[:, 0].astype(np.int64) * m + ib[:, 1]
    na = np.bincount(key_a, minlength=m * m)
    nb = np.bincount(key_b, minlength=m * m)
    first_a = np.concatenate([[0], np.cumsum(na)[:-1]])
    first_b = np.concatenate([[0], np.cumsum(nb)[:-1]])
    # Candidate profiles are the products, for each (i, j), of the a's in ia and the b's in
    # ib. Group the keys (i, j) into batches of about `chunk` profiles each.
    keys = np.flatnonzero((na > 0) & (nb > 0))
    batch = np.cumsum(na[keys] * nb[keys]) // chunk

    def neighborhood_max(value, center, lo, hi):
        best = center.copy()
        for d0 in (-1, 0, 1):
            for d1 in (-1, 0, 1):
                if d0 or d1:
                    best = np.maximum(best, value(d0, d1))
        return best

    count, best_local = 0, -np.inf
    for g in np.unique(batch):
        ks = keys[batch == g]
        lo, hi = first_a[ks[0]], first_a[ks[-1]] + na[ks[-1]]
        rep = nb[key_a[lo:hi]]  # zero for keys without b's
        r = lo + np.repeat(np.arange(hi - lo), rep)
        o = np.arange(len(r)) - np.repeat(np.cumsum(rep) - rep, rep)
        I, J, a = ia[r, 0], ia[r, 1], ia[r, 2]
        b = ib[first_b[key_a[r]] + o, 2]

        def u1(di, da):
            ii, aa = I + di, a + da
            ok = (ii >= 0) & (ii < m) & (aa >= 0) & (aa < m)
            ii, aa = np.clip(ii, 0, m - 1), np.clip(aa, 0, m - 1)
            return np.where(ok, c1[ii, J] + A1[ii, J, aa] + B1[ii, J, b], -np.inf)

        def u2(dj, db):
            jj, bb = J + dj, b + db
            ok = (jj >= 0) & (jj < m) & (bb >= 0) & (bb < m)
            jj, bb = np.clip(jj, 0, m - 1), np.clip(bb, 0, m - 1)
            return np.where(ok, c2[I, jj] + A2[I, jj, a] + B2[I, jj, bb], -np.inf)

        U1, U2 = u1(0, 0), u2(0, 0)
        stable = _close(U1, neighborhood_max(u1, U1, 0, m), rtol, atol)
        stable &= _close(U2, neighborhood_max(u2, U2, 0, m), rtol, atol)
        if stable.any():
            count += int(stable.sum())
            best_local = max(best_local, float((U1 + U2)[stable].max()))
    return count, best_local


def signaling_summary_fast(C, A, B, tol=1e-12):
    """Same output as equilibria.signaling_summary(U1, U2, tol) for the payoffs
    U[k, i, a, j, b] = C[k, i, j] + A[k, i, j, a] + B[k, i, j, b]: worst Nash and best
    locally stable collective payoffs, the numbers of each, and whether a strict
    dilemma remains."""
    nash_count, worst_nash = _nash(C, A, B, tol, tol)
    local_count, best_local = _local(C, A, B, tol, tol)
    worst_nash = worst_nash if nash_count else np.nan
    best_local = best_local if local_count else np.nan
    dilemma = bool(
        nash_count
        and local_count
        and not np.isclose(worst_nash, best_local, rtol=tol, atol=tol)
    )
    return worst_nash, best_local, nash_count, local_count, dilemma
