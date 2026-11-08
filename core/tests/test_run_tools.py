"""The agent-facing tools: manifest check, run status, archiver and doctor."""

import json
import os
import sys
import tarfile
import time

import pytest

from core.analysis import load_cases
from core.analysis.status import case_status, main as status_main
import core.sweeps.nested_tuning as tuning
from core.tests.helpers import ROOT, load_module
from core.tests.test_nested_tuning import spec


def finished_run(tmp_path, monkeypatch):
    """One case run end to end with a fake worker, as test_nested_tuning does."""
    settings = spec(tmp_path)

    def execute(stage, payload, output):
        if stage == "train":
            (output / "last.ckpt").parent.mkdir(parents=True, exist_ok=True)
            (output / "last.ckpt").write_bytes(b"weights" * 100)
            return {"config": settings["config"], "checkpoint": str(output / "last.ckpt")}
        return {"score": payload["parameters"]["algorithm.predictor.temperature"]}

    monkeypatch.setattr(tuning, "execute", execute)
    tuning.run_case(settings, tmp_path / "runs", concurrency=1)
    return settings, tmp_path / "runs"


def age(path, minutes):
    """Backdate every file under ``path``."""
    then = time.time() - minutes * 60
    for p in [path, *path.rglob("*")]:
        os.utime(p, (then, then))


def test_check_reports_every_bad_case_and_protocol_conflicts(tmp_path, monkeypatch):
    good = spec(tmp_path)
    overlap = {**spec(tmp_path), "name": "overlap"}
    overlap["evaluation"] = {**overlap["evaluation"], "test_indices": [1, 2]}
    rows, errors = tuning.check_manifest([good, overlap, good])
    assert [row[0] for row in rows] == ["example"]
    assert rows[0][1:] == (2, 1, 2, 4, "new")
    assert any("overlap" in e and "disjoint" in e for e in errors)
    assert any("Duplicate" in e for e in errors)

    settings, runs = finished_run(tmp_path, monkeypatch)
    rows, errors = tuning.check_manifest([settings], runs)
    assert not errors and rows[0][-1] == "resumes"
    changed = {**settings, "evaluation": {**settings["evaluation"], "test_indices": [8, 9]}}
    rows, errors = tuning.check_manifest([changed], runs)
    assert not rows and "a resume would refuse it" in errors[0]


def test_check_cli_exits_nonzero_on_errors(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "manifest.json"
    bad = {**spec(tmp_path), "inference": []}
    for specs, code in (([spec(tmp_path)], 0), ([bad], 1)):
        manifest.write_text(json.dumps(specs))
        monkeypatch.setattr(sys, "argv", ["nested_tuning", str(manifest), "--check"])
        with pytest.raises(SystemExit) as exit:
            tuning.main()
        assert exit.value.code == code
    assert "ERROR example" in capsys.readouterr().out


def test_status_states(tmp_path, monkeypatch, capsys):
    _, runs = finished_run(tmp_path, monkeypatch)
    case = runs / "example"
    assert case_status(case)["state"] == "done"
    assert case_status(case)["trained"] == "2/2" and case_status(case)["policies"] == "1/1"

    (case / "final" / "result.json").unlink()
    assert case_status(case)["state"] == "running"  # files just changed
    age(case, 60)
    assert case_status(case)["state"] == "partial"
    candidate = next((case / "training" / "candidates").iterdir())
    (candidate / "failure.json").write_text("{}")
    age(case, 60)
    assert case_status(case)["state"] == "failed"

    fresh = runs / "fresh"
    fresh.mkdir()
    (fresh / "protocol.json").write_text(json.dumps({"spec": {"name": "fresh", "inference": []}}))
    age(fresh, 60)
    assert case_status(fresh)["state"] == "not started"
    status_main([str(runs)])
    out = capsys.readouterr().out
    assert "1 failed" in out and "1 not started" in out


def test_archive_drops_checkpoints_keeps_mtimes_and_verifies(tmp_path, monkeypatch, capsys):
    _, runs = finished_run(tmp_path, monkeypatch)
    archive = load_module(ROOT / "scripts/archive_runs.py")
    analysis = tmp_path / "analyze.py"
    analysis.write_text(
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from core.analysis import load_cases\n"
        "args = sys.argv[1:]\n"
        "out = Path(args[args.index('--output') + 1]); out.mkdir(parents=True)\n"
        "cases = load_cases(args[0])\n"
        "(out / 'cases.json').write_text(json.dumps("
        "[[c.name, c.selected, c.test, sorted(c.policies)] for c in cases]))\n"
    )
    stamp = (runs / "example" / "protocol.json").stat().st_mtime
    output = tmp_path / "out" / "runs.tar.gz"
    archive.main([str(runs), "--output", str(output), "--verify", str(analysis)])
    assert "verified" in capsys.readouterr().out

    with tarfile.open(output) as tar:
        names = tar.getnames()
        assert not [n for n in names if n.startswith("runs/") and n.endswith((".ckpt", ".lock"))]
        assert "provenance/uv.lock" in names
        assert int(tar.getmember("runs/runs/example/protocol.json").mtime) == int(stamp)
        index = json.loads(tar.extractfile("index.json").read())
        sums = tar.extractfile("SHA256SUMS").read().decode().splitlines()
        tar.extractall(tmp_path / "restored")
    assert index == [{
        "root": "runs", "case": "example", "path": "example", "state": "done", "task": "",
        "recipe": "", "selected": "native", "test": 1.0,
    }]
    assert {line.split("  ")[1] for line in sums} == set(names) - {"SHA256SUMS"} - {
        n for n in names if (tmp_path / "restored" / n).is_dir()
    }
    restored = load_cases(tmp_path / "restored" / "runs" / "runs")
    assert [(c.name, c.selected, c.test) for c in restored] == [("example", "native", 1.0)]
    with pytest.raises(SystemExit):
        archive.main([str(runs), "--output", str(output)])  # refuses to overwrite


def test_doctor_reports_each_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SDDC_STORAGE_ROOT", str(tmp_path / "storage"))
    doctor = load_module(ROOT / "scripts/doctor.py")
    code = doctor.main([])
    out = capsys.readouterr().out
    for name in ("python", "torch", "storage root", "tokenizers", "configs compose"):
        assert f"] {name}" in out
    assert "[ok  ] tokenizers" in out and "[ok  ] configs compose" in out
    assert code == (1 if "[fail]" in out else 0)
