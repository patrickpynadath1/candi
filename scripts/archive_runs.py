"""Pack nested-tuning run trees into a portable archive without checkpoints.

    uv run --locked --extra tuning --extra scoring python scripts/archive_runs.py \
        RUN_ROOT [RUN_ROOT ...] --output runs-NAME.tar.gz \
        [--verify experiments/<example>/analyze.py] [--dry-run]

Keeps every file of each case except bulky runtime state: checkpoints and weights
(*.ckpt, *.pt, *.safetensors, *.bin), W&B and Ray directories, driver locks and
temporary files, and any file over --max-file-mb. File mtimes are preserved
(block-encoder-decoder's analysis derives training hours from them).

Layout inside the archive:
    runs/<root name>/...   the kept files, at their paths relative to each root
    index.json             one entry per case: root, state, task, recipe, selected policy, test score
    summary.csv            the same as a table
    provenance/            git revision, uncommitted diff, uv.lock, pyproject.toml, versions, command
    SHA256SUMS             a checksum for every file above

--verify runs the analysis script on the original roots and on the extracted
archive and fails unless both write the same results. Restore with
``tar -xzf ARCHIVE`` and point analyze.py at ``runs/<root name>``.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from core.analysis.results import load_case  # noqa: E402
from core.analysis.status import case_status  # noqa: E402

DROP_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".bin", ".tmp", ".lock"}
DROP_DIRS = {"wandb", "ray_results", "__pycache__", ".ray"}
VERSIONED = ["torch", "lightning", "hydra-core", "omegaconf", "optuna", "ray", "sacrebleu"]


def kept(path, limit):
    """Why ``path`` is dropped, or None when it is kept."""
    if set(path.parts) & DROP_DIRS:
        return "runtime directory"
    if path.suffix in DROP_SUFFIXES or path.name == ".driver.lock":
        return f"{path.suffix or path.name} file"
    if path.stat().st_size > limit:
        return f"over {limit // 2**20} MiB"
    return None


def collect(roots, limit):
    keep, drop = [], []
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            reason = kept(path, limit)
            (drop if reason else keep).append((root, path, reason))
    return keep, drop


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return b""


def provenance(argv):
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in VERSIONED:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            pass
    files = {
        "provenance/git-revision.txt": git("rev-parse", "HEAD"),
        "provenance/git-status.txt": git("status", "--porcelain"),
        "provenance/uncommitted.patch": git("diff", "HEAD", "--binary"),
        "provenance/environment.json": json.dumps({
            "python": sys.version,
            "platform": platform.platform(),
            "packages": versions,
            "command": argv,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, indent=2).encode(),
    }
    for name in ("uv.lock", "pyproject.toml"):
        if (REPO / name).exists():
            files[f"provenance/{name}"] = (REPO / name).read_bytes()
    return files


def index(roots):
    entries = []
    for root in roots:
        for protocol in sorted(root.rglob("protocol.json")):
            case, status = load_case(protocol.parent), case_status(protocol.parent)
            entries.append({
                "root": root.name,
                "case": case.name,
                "path": str(protocol.parent.relative_to(root)),
                "state": status["state"],
                "task": case.task,
                "recipe": case.recipe,
                "selected": case.selected,
                "test": case.test,
            })
    return entries


def summary_csv(entries):
    out = io.StringIO()
    fields = ["root", "case", "state", "task", "recipe", "selected", "test"]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(entries)
    return out.getvalue().encode()


def add_bytes(tar, name, data, sums):
    info = tarfile.TarInfo(name)
    info.size, info.mtime = len(data), time.time()
    tar.addfile(info, io.BytesIO(data))
    sums[name] = hashlib.sha256(data).hexdigest()


def write(output, roots, keep, argv):
    sums = {}
    entries = index(roots)
    with tarfile.open(output, "w:gz") as tar:
        for root, path, _ in keep:
            name = f"runs/{root.name}/{path.relative_to(root).as_posix()}"
            tar.add(path, arcname=name, recursive=False)  # keeps mtime and mode
            sums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        add_bytes(tar, "index.json", json.dumps(entries, indent=2).encode(), sums)
        add_bytes(tar, "summary.csv", summary_csv(entries), sums)
        for name, data in provenance(argv).items():
            add_bytes(tar, name, data, sums)
        add_bytes(tar, "SHA256SUMS", "".join(f"{h}  {n}\n" for n, h in sorted(sums.items())).encode(), {})
    return entries


def analysis_outputs(directory):
    """Every file an analysis wrote, parsed when it is JSON."""
    results = {}
    for path in sorted(p for p in Path(directory).rglob("*") if p.is_file()):
        text = path.read_text()
        try:
            results[path.name] = json.loads(text)
        except ValueError:
            results[path.name] = text
    return results


def verify(script, roots, archive, scratch):
    restored = scratch / "restored"
    with tarfile.open(archive) as tar:
        # The data filter refuses absolute paths and links out of the target.
        options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(restored, **options)
    runs = []
    for label, dirs in (("original", roots), ("archive", [restored / "runs" / r.name for r in roots])):
        out = scratch / label
        command = [sys.executable, str(script), *map(str, dirs), "--output", str(out)]
        done = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f"{label} analysis failed:\n{done.stdout}\n{done.stderr}")
        runs.append(analysis_outputs(out))
    if not runs[0]:
        raise SystemExit("the analysis wrote no files; nothing to compare")
    if runs[0] != runs[1]:
        different = sorted(k for k in set(runs[0]) | set(runs[1]) if runs[0].get(k) != runs[1].get(k))
        raise SystemExit(f"archive analysis differs from the original in: {', '.join(different)}")
    return sorted(runs[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="nested_tuning --output directories")
    parser.add_argument("--output", type=Path, required=True, help="archive path (.tar.gz)")
    parser.add_argument("--verify", type=Path, help="experiments/<example>/analyze.py to compare on")
    parser.add_argument("--max-file-mb", type=float, default=200)
    parser.add_argument("--dry-run", action="store_true", help="list what would be kept and dropped")
    parser.add_argument("--force", action="store_true", help="overwrite an existing archive")
    args = parser.parse_args(argv)
    roots = [root.resolve() for root in args.roots]
    for root in roots:
        if not root.is_dir():
            parser.error(f"{root} is not a directory")
    if len({root.name for root in roots}) != len(roots):
        parser.error("run roots need distinct directory names")
    keep, drop = collect(roots, int(args.max_file_mb * 2**20))
    size = lambda items: sum(path.stat().st_size for _, path, _ in items) / 2**20  # noqa: E731
    print(f"keep {len(keep)} files ({size(keep):.1f} MiB); drop {len(drop)} files ({size(drop):.1f} MiB)")
    reasons = {}
    for _, _, reason in drop:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        print(f"  dropped {count}: {reason}")
    if args.dry_run:
        return
    output = args.output.resolve()
    if output.exists() and not args.force:
        parser.error(f"{output} exists; pass --force to overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = write(output, roots, keep, sys.argv if argv is None else argv)
    states = {}
    for entry in entries:
        states[entry["state"]] = states.get(entry["state"], 0) + 1
    print(f"wrote {output} ({output.stat().st_size / 2**20:.1f} MiB): {len(entries)} cases, "
          + ", ".join(f"{n} {s}" for s, n in sorted(states.items())))
    if args.verify:
        scratch = Path(tempfile.mkdtemp(prefix=".verify-", dir=output.parent))
        try:
            names = verify(args.verify.resolve(), roots, output, scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        print(f"verified: {args.verify} gives the same {', '.join(names)} from the archive")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
