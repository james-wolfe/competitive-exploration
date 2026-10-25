# Generated data

The files described here are archived on Zenodo at
[doi:10.5281/zenodo.23627789](https://doi.org/10.5281/zenodo.23627789), as
`competitive-exploration-data.zip`. Unpack it so that the `.npz` files sit in
`data/generated/` of the code repository,
[github.com/james-wolfe/competitive-exploration](https://github.com/james-wolfe/competitive-exploration)
(see its README).

Every file in `data/generated/` is written by the script of the same name in
`scripts/generate/` (the variants of the sweep are written by
`sweep_nash_social_dilemma.py fine`, `... small_d` and `... fine401`, and
`signaling_fine401.npz` by `signaling_fine401.py`). No external data are used.
Each `.npz` archive stores its parameter grids next to the results and loads with
`numpy.load` without pickling. `timings.txt` records how long each script took.

Payoff matrices follow one convention: `Z[i, j]` is the payoff to an agent with
exploration rate `eps_grid[i]` whose competitor has rate `eps_grid[j]`.

| File | Contents | Used in |
|---|---|---|
| `single_agent_rewards.npz` | A lone agent's discounted reward from each site, against its rate | Fig. 2A |
| `opt_eps_single.npz` | A lone agent's optimal rate over (gamma, d, R) | Figs. 2B, 2C, S3 |
| `two_agent_payoffs.npz` | Player 2's payoff from each site over 1001 x 1001 rate pairs, d = 4, R = 2.1 | Fig. 3 |
| `two_agent_local_br.npz` | Payoffs, Nash equilibria, and local best-response basins in four regimes | Figs. 4, S8 |
| `sweep_nash_social_dilemma.npz` | Payoff matrices, equilibrium types, collective payoffs, strict and Pareto dilemmas over (gamma, d, R), 41 rates | Fig. S6 (via `local_br_outcomes.npz`); checks |
| `sweep_nash_social_dilemma_fine.npz` | The same summaries with 101 rates | Figs. S4, S5 |
| `sweep_nash_social_dilemma_small_d.npz` | The same summaries for d = 1-5, with rates down to 1e-8 | Fig. S5 |
| `sweep_nash_social_dilemma_fine401.npz` | The same summaries with 401 rates plus 1e-8, ..., 1e-2 (407 rates), d = 1-30 | Figs. 5, 6, S7; Tables 2, S2 |
| `signaling_fine401.npz` | Signaling outcome for each strict dilemma whose worst Nash equilibrium has both agents exploring, on the 407-rate grid, d = 2-30 (about 350 CPU-hours; see the README) | Fig. 6, Table S2 |
| `locally_stable_points.npz` | Locally stable points with and without signaling at d = 4, R = 2.1 | Table 3 |
| `dilemma_kinds.npz` | Which kinds of dilemma (collective and Pareto) occur over nine discount factors | Table S1 |
| `td_learner_sim.npz` | Mean reward of simulated TD learners and the Markov chain expectation | Fig. S1 |
| `threshold_sweep.npz` | Threshold reward ratio R*(d, T) and the rate the optimum jumps to there, d = 1-40 and 85 horizons T = 1e2-1e9, plus the three paper horizons | Fig. S2 |
| `opt_eps_poisson.npz` | Optimal rate when the distance is random | Fig. S3 |
| `opt_eps_poisson_comparison.npz` | Random-distance optimum minus fixed-distance optimum | Fig. S3 |
| `local_max_eps_single.npz` | Rate at the positive local maximum of a lone agent's reward curve | Fig. S4 |
| `local_br_outcomes.npz` | Mean collective payoff reached by local best-response dynamics | Fig. S6 |
| `two_site_local_br.npz` | Payoffs and local best-response basins in the two-site model | Fig. S8 |
| `hitting_stats.npz` | Hitting probabilities and times: formulas and simulation | Fig. S9 |
