"""docs/validation/compatibility.md must match the code that generates it."""

from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "docs"))
import compatibility_table as table  # noqa: E402

DOC = ROOT / "docs" / "validation" / "compatibility.md"
SYMBOL = {"✓": "works", "–": "rejected", "✗": "error"}


def parsed():
    """{(row, column): status} for both tables in the page."""
    cells, columns = {}, None
    for line in DOC.read_text().splitlines():
        if not line.startswith("| ") or line.startswith("| ---"):
            continue
        parts = [part.strip() for part in line.strip("|").split("|")]
        if parts[0] == "":
            columns = [column.strip("`") for column in parts[1:]]
            continue
        for column, value in zip(columns, parts[1:]):
            cells[parts[0], column] = SYMBOL[re.sub(r"<sup>\d+</sup>", "", value)]
    return cells


def test_page_lists_every_family_sampler_and_modifier():
    cells = parsed()
    assert {row for row, _ in cells} == set(table.FAMILIES)
    assert {column for _, column in cells} == set(table.SAMPLERS) | set(table.MODIFIERS)


def test_no_combination_fails_without_a_reason():
    assert "error" not in parsed().values()


# Every family's own sampler, plus a spread of accepted and refused cells.
SPOT_CHECKS = [(family, None, None) for family in table.FAMILIES] + [
    ("Gaussian", "hybrid", None),
    ("Blurring", "composed", None),
    ("MDLM (masked)", "euler", None),
    ("Autoregressive", "ancestral", None),
    ("D3PM (absorbing)", "entropy_bounded", None),
    ("GIDD", "uniform_entropy_bounded", None),
    ("SEDD (absorbing)", None, "self-conditioning"),
    ("D3PM (absorbing)", None, "token feedback"),
    ("Spherical", None, "block diffusion"),
    ("CANDI hybrid", None, "self-conditioning"),
]


def test_every_family_has_a_working_sampler():
    cells = parsed()
    for family in table.FAMILIES:
        assert any(cells[family, sampler] == "works" for sampler in table.SAMPLERS), family


@pytest.mark.parametrize("family, sampler, modifier", SPOT_CHECKS)
def test_spot_checked_cells_still_hold(family, sampler, modifier):
    status, _ = table.outcome(
        table.FAMILIES[family], sampler, table.MODIFIERS[modifier] if modifier else None
    )
    if sampler is None and modifier is None:
        # The preset's own sampler, with a prompt (unconditional presets refuse it).
        assert status in {"works", "rejected"}
        return
    assert status == parsed()[family, sampler or modifier]
