# Reproduces the data, figures, and tables of "Competitive exploration in structured
# environments". Run `make` for everything, or `make data`, `make figures`,
# `make tables`, `make checks`. See README.md. `make data` builds every data file except
# the slow signaling computation (SLOW_FILES below), which is taken from the data archive
# or built with `make signaling-data`. The figure notebooks are the reference version of
# the figure code: after editing them, `make figure-scripts` copies each figure
# cell to its script in scripts/figures/, and `make figures` refuses to run until they match.

PYTHON ?= python
# Parallel processes for the scripts that take --workers (default: all cores).
WORKERS ?= $(shell getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
DATA := data/generated
GENERATE := scripts/generate
TIMINGS := $(DATA)/timings.txt

DATA_FILES := $(addprefix $(DATA)/, \
	single_agent_rewards.npz \
	opt_eps_single.npz \
	two_agent_payoffs.npz \
	two_agent_local_br.npz \
	sweep_nash_social_dilemma.npz \
	locally_stable_points.npz \
	dilemma_kinds.npz \
	td_learner_sim.npz \
	threshold_sweep.npz \
	opt_eps_poisson.npz \
	opt_eps_poisson_comparison.npz \
	sweep_nash_social_dilemma_fine.npz \
	local_max_eps_single.npz \
	sweep_nash_social_dilemma_small_d.npz \
	sweep_nash_social_dilemma_fine401.npz \
	local_br_outcomes.npz \
	hitting_stats.npz \
	two_site_local_br.npz)

# Slow: signaling_fine401.npz takes about 350 CPU-hours (21.8 h with 16 workers) and
# about 5 GB of memory per worker. `make data` never builds these; download them from the
# data archive, or build them with `make signaling-data`. That uses one worker unless
# WORKERS=N is given; choose N so that N x 5 GB fits in memory.
SLOW_FILES := $(addprefix $(DATA)/, \
	signaling_fine401.npz)

ALL_DATA := $(DATA_FILES) $(SLOW_FILES)

# Execute a notebook in place, with its kernel running the same Python as $(PYTHON).
RUN_NOTEBOOK = PATH="$$(dirname "$$($(PYTHON) -c 'import sys; print(sys.executable)')"):$$PATH" \
	$(PYTHON) -m nbconvert --to notebook --execute --inplace

FIGURES := $(patsubst scripts/figures/%.py,figures/%.pdf,$(wildcard scripts/figures/*.py))

.PHONY: all data signaling-data figures tables checks figure-scripts check-figure-scripts

all: data figures tables checks

data: $(DATA_FILES)

signaling-data: BUILD_SLOW := 1
signaling-data: WORKERS := 1
signaling-data: $(SLOW_FILES)

figures: check-figure-scripts $(FIGURES)

check-figure-scripts:
	@$(PYTHON) scripts/sync_figure_scripts.py --check

figure-scripts:
	$(PYTHON) scripts/sync_figure_scripts.py

tables: | $(ALL_DATA)
	$(RUN_NOTEBOOK) notebooks/tables.ipynb

checks: | $(ALL_DATA)
	$(RUN_NOTEBOOK) notebooks/checks.ipynb

figures/%.pdf: scripts/figures/%.py src/competitive_exploration/plotting.py $(ALL_DATA)
	MPLBACKEND=Agg $(PYTHON) $<

# Run a generation script and append its wall-clock time to $(TIMINGS).
define generate
	@echo "$(PYTHON) $(1)"
	@start=$$(date +%s); $(PYTHON) $(1) && \
		echo "$(notdir $@) $$(( $$(date +%s) - start )) s" >> $(TIMINGS)
endef

# Each script writes the file of the same name; some read earlier outputs. Data files
# are built only when missing, never because a script is newer, so downloaded data
# are used as is. To rerun a step, delete its output.
$(DATA)/%.npz: | $(GENERATE)/%.py
	$(call generate,$(GENERATE)/$*.py)

$(DATA)/sweep_nash_social_dilemma_fine.npz:
	$(call generate,$(GENERATE)/sweep_nash_social_dilemma.py fine)

$(DATA)/sweep_nash_social_dilemma_small_d.npz:
	$(call generate,$(GENERATE)/sweep_nash_social_dilemma.py small_d)

$(DATA)/sweep_nash_social_dilemma_fine401.npz:
	$(call generate,$(GENERATE)/sweep_nash_social_dilemma.py fine401 --workers $(WORKERS))

# Slow (see SLOW_FILES): refuses to run unless called through `make signaling-data`.
$(DATA)/signaling_fine401.npz: | $(DATA)/sweep_nash_social_dilemma_fine401.npz
	@if [ -z "$(BUILD_SLOW)" ]; then \
		echo "$@ is missing. It takes about 350 CPU-hours: download it from the data" \
			"archive, or build it with 'make signaling-data WORKERS=N'." >&2; \
		exit 1; \
	fi
	$(call generate,$(GENERATE)/signaling_fine401.py --workers $(WORKERS) --checkpoint $(DATA)/signaling_fine401_checkpoint)

$(DATA)/local_br_outcomes.npz: | $(DATA)/sweep_nash_social_dilemma.npz
$(DATA)/opt_eps_poisson_comparison.npz: | $(DATA)/opt_eps_single.npz
$(DATA)/local_max_eps_single.npz: | $(DATA)/sweep_nash_social_dilemma_fine.npz
