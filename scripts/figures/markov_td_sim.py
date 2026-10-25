"""Figure S1: the Markov chain reproduces the mean reward of simulated TD learners."""

import matplotlib.pyplot as plt
import numpy as np

from competitive_exploration.paths import DATA_DIR
from competitive_exploration.plotting import save_figure, use_paper_style

use_paper_style()
data = np.load(DATA_DIR / "td_learner_sim.npz")

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
ax.plot(data["chain_eps"], data["expected_reward"], label="Markov chain expectation")
ax.errorbar(
    data["sim_eps"],
    data["mean_reward"],
    yerr=2 * data["standard_error"],
    fmt="o",
    capsize=4,
    label="TD learner simulation",
)
ax.set_xlabel(r"Exploration rate $\varepsilon$")
ax.set_ylabel(rf"Reward over $T = {data['T']}$ steps")
ax.legend()
ax.grid(True)

save_figure(fig, "markov_td_sim.pdf")
