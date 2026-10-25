"""Write every figure cell of the figure notebooks to its script in scripts/figures/.

The notebooks are the reference version of the figure code. A figure is a markdown cell
such as "## Figure S2" naming its script in backticks
("`scripts/figures/threshold_reward_ratio.py`"), followed by the code cell, and the
script is exactly that code cell. With --check, report stale or orphaned scripts and
exit with an error instead of writing.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("notebooks/main_figs.ipynb", "notebooks/sup_figs.ipynb")
HEADER = re.compile(r"## Figure \w+\n`(scripts/figures/\w+\.py)`")


def figure_cells():
    """Yield (script path, script text) for every figure cell, in notebook order."""
    for notebook in NOTEBOOKS:
        cells = json.loads((ROOT / notebook).read_text())["cells"]
        for header, cell in zip(cells, cells[1:]):
            match = HEADER.match("".join(header["source"]))
            if header["cell_type"] == "markdown" and match:
                if cell["cell_type"] != "code":
                    sys.exit(f"{notebook}: no code cell after the header for {match[1]}")
                yield ROOT / match[1], "".join(cell["source"]).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    check = parser.parse_args().check

    expected = dict(figure_cells())
    stale = [p for p, text in expected.items() if not p.exists() or p.read_text() != text]
    orphans = sorted(set((ROOT / "scripts/figures").glob("*.py")) - set(expected))
    if check:
        problems = [f"out of sync: {p.relative_to(ROOT)}" for p in stale]
        problems += [f"no notebook cell: {p.relative_to(ROOT)}" for p in orphans]
        if problems:
            problems.append("Run `make figure-scripts` after editing the notebooks.")
            sys.exit("\n".join(problems))
        print(f"All {len(expected)} figure scripts match the notebooks.")
        return
    for path in stale:
        path.write_text(expected[path])
        print(f"wrote {path.relative_to(ROOT)}")
    for path in orphans:
        print(f"warning: {path.relative_to(ROOT)} has no notebook cell")


if __name__ == "__main__":
    main()
