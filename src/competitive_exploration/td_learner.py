"""Direct simulation of the epsilon-greedy temporal-difference learner (Model,
"Interpretation as a temporal-difference or Q-learning problem")."""

import numpy as np
from tqdm import tqdm


def run_episode(d, steps, eps, gamma, R0=1.0, Rd=1.0, rng=None):
    """Simulate one agent for a number of steps, starting at site 0.

    With probability 1 - eps the agent moves to the site among its own and its
    neighbors' with the highest value estimate Q (ties broken at random); otherwise it
    moves to a random neighbor. On arriving at a reward site it sets Q to the
    steady-state value r / (1 - gamma); at an empty site it applies the Bellman update
    Q(s) = gamma * max Q over the neighbors of s and s itself. Returns the visited
    sites, including the starting site.
    """
    rng = np.random.default_rng() if rng is None else rng
    reward = np.zeros(d + 1)
    reward[0], reward[d] = R0, Rd

    Q = np.zeros(d + 1)
    Q[0] = R0 / (1 - gamma)
    x = 0
    trajectory = [x]

    for _ in range(steps):
        if rng.random() < 1 - eps:
            candidates = np.array([s for s in (x, x - 1, x + 1) if 0 <= s <= d])
            values = Q[candidates]
            x = int(rng.choice(candidates[values == values.max()]))
        else:
            x = int(
                rng.choice([1] if x == 0 else [d - 1] if x == d else [x - 1, x + 1])
            )

        if reward[x] > 0:
            Q[x] = reward[x] / (1 - gamma)
        else:
            Q[x] = gamma * Q[max(x - 1, 0) : x + 2].max()
        trajectory.append(x)

    return np.array(trajectory)


def simulate_rewards(d, T, eps_values, n_runs, gamma, R0=1.0, Rd=1.0, seed=1):
    """Mean and standard error, across runs, of the reward collected over T steps."""
    rng = np.random.default_rng(seed)
    means, standard_errors = [], []
    for eps in tqdm(eps_values, desc="TD learner"):
        totals = np.empty(n_runs)
        for k in range(n_runs):
            trajectory = run_episode(d, T, float(eps), gamma, R0, Rd, rng)
            totals[k] = R0 * np.sum(trajectory == 0) + Rd * np.sum(trajectory == d)
        means.append(totals.mean())
        standard_errors.append(totals.std(ddof=1) / np.sqrt(n_runs))
    return np.array(means), np.array(standard_errors)
