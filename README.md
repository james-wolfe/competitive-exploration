# Competitive exploration

Code for *Competitive exploration in structured environments* by James P. Wolfe and
Joshua B. Plotkin ([arXiv:2610.34643](https://arxiv.org/abs/2610.34643)). It computes
every figure and table in the paper from the model, using exact Markov-chain calculations,
plus simulations that check them.

## Setup

The code needs Python 3.12 or later. The figures are typeset with LaTeX, so a LaTeX
installation (for example TeX Live or MacTeX) must be on the path.

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Both install the pinned package versions and the code in `src/` as the package
`competitive_exploration`.

## Reproducing the paper

```bash
make
```

runs four steps, which can also be run one at a time:

| Command | What it does | Output |
|---|---|---|
| `make data` | Runs the model and simulations, except the slow signaling step below | `data/generated/` |
| `make figures` | Draws every figure | `figures/*.pdf` |
| `make tables` | Builds the tables, shows each as a DataFrame, and prints its LaTeX as laid out in the paper | `tables/*.tex`, `notebooks/tables.ipynb` |
| `make checks` | Checks numerical claims in the text, and checks the fast signaling code against the full computation | `notebooks/checks.ipynb` |

A data file is computed only if it is missing, never because a script is newer, so
existing or downloaded files are used as they are. To recompute one step, delete its
file and run `make data`. `make -j` runs independent steps in parallel. Scripts that take
`--workers` use all cores by default; set `WORKERS=N` to change that.

The figure notebooks, `notebooks/main_figs.ipynb` and `notebooks/sup_figs.ipynb`, are the
reference version of the figure code. After editing a figure cell, run
`make figure-scripts` to copy it to its script in `scripts/figures/`; `make figures`
stops with an error while any script differs from its cell.

### Using the archived data

The generated data are archived on Zenodo at
[doi:10.5281/zenodo.23627789](https://doi.org/10.5281/zenodo.23627789). To draw the figures and
tables without recomputing anything, unpack the archive so that the `.npz` files sit in
`data/generated/`, then run

```bash
make figures tables checks
```

`make` alone also works: with every data file present, `make data` does nothing.

### Compute time

On a laptop (Apple M3, 8 cores, 24 GB of memory), `make data` takes about 3 hours. The
steps that take longest are:

| Step | Time |
|---|---|
| `sweep_nash_social_dilemma_fine401.npz` (407 exploration rates, about 6 CPU-hours, on 8 cores) | 1.5 h |
| `td_learner_sim.npz` (5000 simulated learners at each of 21 rates) | 36 min |
| `sweep_nash_social_dilemma_fine.npz` | 27 min |
| `dilemma_kinds.npz` | 9 min |
| `sweep_nash_social_dilemma.npz`, `two_agent_payoffs.npz`, `threshold_sweep.npz` | about 2 min each |
| the other 11 steps | about 1 min or less each |

`make figures` and `make tables` take under 2 minutes, and `make checks` a few minutes.

One step is not part of `make data` because it is slow: `signaling_fine401.npz`
(signaling on the 407-rate grid; Fig. 6 and Table S2) takes about 350 CPU-hours and
about 5 GB of memory per worker (we ran it with 16 workers on a computing cluster, in
21.8 hours). Take it from the data archive; `make figures`, `make tables`, and
`make checks` stop with a message if it is missing. To recompute it, run
`make signaling-data WORKERS=N`, choosing N so that N x 5 GB fits in memory (without
`WORKERS`, it uses one worker). It saves each (gamma, d) cell to
`data/generated/signaling_fine401_checkpoint/`, so an interrupted run resumes where it
stopped.

The time of each step is appended to `data/generated/timings.txt` when it runs.

## Where each result comes from

Each figure is drawn by the script in `scripts/figures/` with the same name as its PDF.
The notebooks `notebooks/main_figs.ipynb` and `notebooks/sup_figs.ipynb` contain the
same scripts, in the order of the paper, with their output. Each data file is written by
the script of the same name in `scripts/generate/`; `data/README.md` describes them.

| Paper | Script or notebook | Data |
|---|---|---|
| Figure 1 | `lattice_schematic.py` | |
| Figure 2 | `single_agent_optimal_exploration.py` | `single_agent_rewards.npz`, `opt_eps_single.npz` |
| Figure 3 | `payoff_matrix_and_slices.py` | `two_agent_payoffs.npz` |
| Figure 4 | `four_regimes_payoff_and_localBR.py` | `two_agent_local_br.npz` |
| Figure 5 | `nash_social_dilemma_classification.py` | `sweep_nash_social_dilemma_fine401.npz` |
| Figure 6 | `signaling_dilemma_summary_sampled_refined.py` | `sweep_nash_social_dilemma_fine401.npz`, `signaling_fine401.npz` |
| Figure S1 | `markov_td_sim.py` | `td_learner_sim.npz` |
| Figure S2 | `threshold_reward_ratio.py` | `threshold_sweep.npz` |
| Figure S3 | `opt_eps_poisson.py` | `opt_eps_poisson.npz`, `opt_eps_poisson_comparison.npz` |
| Figure S4 | `sym_local_nash_location.py` | `sweep_nash_social_dilemma_fine.npz`, `local_max_eps_single.npz` |
| Figure S5 | `worst_equilibrium_fitness_percentages.py` | `sweep_nash_social_dilemma_fine.npz`, `sweep_nash_social_dilemma_small_d.npz` |
| Figure S6 | `local_br_mean_fitness_ratio.py` | `local_br_outcomes.npz` |
| Figure S7 | `global_social_optimum_classification.py` | `sweep_nash_social_dilemma_fine401.npz` |
| Figure S8 | `two_site_model.py` | `two_site_local_br.npz`, `two_agent_local_br.npz` |
| Figure S9 | `hitting_stats.py` | `hitting_stats.npz` |
| Table 1 | Model parameters, no data | |
| Table 2 | `notebooks/tables.ipynb` | `sweep_nash_social_dilemma_fine401.npz` |
| Table 3 | `notebooks/tables.ipynb` | `locally_stable_points.npz` |
| Table S1 | `notebooks/tables.ipynb` | `dilemma_kinds.npz` |
| Table S2 | `notebooks/tables.ipynb` | `sweep_nash_social_dilemma_fine401.npz`, `signaling_fine401.npz` |
| Numbers in the text and captions | `notebooks/checks.ipynb` | several |

## Repository layout

```text
src/competitive_exploration/   the model: single agent, two agents, signaling, equilibria
scripts/generate/              one script per data file
scripts/figures/               one script per figure
scripts/sync_figure_scripts.py copies the figure cells of the notebooks to scripts/figures/
notebooks/                     figures, tables, and checks as notebooks
data/generated/                generated data (archived separately)
figures/, tables/              outputs
```

The package modules follow the paper:

| Module | Paper |
|---|---|
| `single_agent.py` | Model, "Markov chain formulation"; Methods, "The single-agent transition matrix" |
| `two_agent.py` | Model, "Two competing agents"; Methods, "The two-agent transition matrix and reward vectors" |
| `equilibria.py`, `local_br.py` | Results, "Two competing foragers play a game with multiple equilibria" and "Competition produces a social dilemma"; Methods, "Numerical computation of equilibria" |
| `signaling.py`, `signaling_fast.py` | Results, "Signaling can remove suboptimal equilibria"; SI Section S8 |
| `td_learner.py` | Model, "Interpretation as a temporal-difference or $Q$-learning problem" |
| `poisson.py` | SI Section S1 |
| `two_site.py` | SI Section S7 |
| `plotting.py` | shared figure style |

## Citation

James P. Wolfe and Joshua B. Plotkin, *Competitive exploration in structured
environments*, [arXiv:2610.34643](https://arxiv.org/abs/2610.34643) (2026).

The version of the code used for the paper is archived on Zenodo at
[doi:10.5281/zenodo.23671168](https://doi.org/10.5281/zenodo.23671168).

## License

The code is released under the MIT License (see `LICENSE`).
