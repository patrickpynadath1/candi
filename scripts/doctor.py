"""Check that this checkout can run SDDC, one line per check.

    uv run --locked --extra tuning --extra scoring python scripts/doctor.py [--smoke] [--tests]

Statuses: ok, warn (works, but something optional is missing), fail (fix before
running experiments) and skip. Exits 1 when any check fails. The checks only read,
apart from a probe file in the storage root; --smoke trains a 2-step CPU model
under $SDDC_STORAGE_ROOT/doctor/ and evaluates it, and --tests runs a quick subset
of the test suite.
"""

import argparse
import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
QUICK_TESTS = [
    "core/tests/test_run_preparation.py",
    "core/tests/test_mdlm.py",
    "core/tests/test_agent_skills.py",
]
# (import name, extra that provides it); None means a core dependency.
PACKAGES = [
    ("torch", None), ("lightning", None), ("hydra", None), ("omegaconf", None),
    ("ray", None), ("optuna", "tuning"), ("sacrebleu", "scoring"),
    ("wandb", "tracking"), ("maze_dataset", "mazes"),
]
results = []


def report(status, name, detail=""):
    results.append(status)
    print(f"[{status:4}] {name}" + (f": {detail}" if detail else ""))


def storage_root():
    return Path(os.environ.get("SDDC_STORAGE_ROOT", Path.home() / ".cache" / "sddc")).expanduser()


def check_platform():
    major, minor = sys.version_info[:2]
    if (major, minor) == (3, 10):
        report("ok", "python", platform.python_version())
    else:
        report("fail", "python", f"{platform.python_version()}; the lock targets 3.10 (.python-version)")
    system, machine = platform.system(), platform.machine()
    if (system, machine) in {("Linux", "x86_64"), ("Windows", "AMD64")}:
        report("ok", "platform", f"{system} {machine}")
    else:
        report("warn", "platform", f"{system} {machine}: `uv sync --locked` cannot install the "
               "CUDA 12.1 torch wheels here. Run torch-free tests with `uv run --no-project`, "
               "and send GPU work to a server (docs/guides/remote-execution.md)")
    report("ok" if shutil.which("uv") else "warn", "uv", shutil.which("uv") or "not on PATH")


def check_packages():
    for name, extra in PACKAGES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            hint = f"install with --extra {extra}" if extra else "run `uv sync --locked`"
            report("warn" if extra else "fail", name, f"missing; {hint}")
            continue
        report("ok", name, getattr(module, "__version__", ""))
        if name == "torch":
            if module.cuda.is_available():
                kind = "ROCm" if getattr(module.version, "hip", None) else "CUDA"
                report("ok", "gpu", f"{kind}, {module.cuda.device_count()} device(s): {module.cuda.get_device_name(0)}")
            else:
                report("warn", "gpu", "none visible; CPU runs and tests only")


def check_storage():
    root = storage_root()
    source = "SDDC_STORAGE_ROOT" if "SDDC_STORAGE_ROOT" in os.environ else "default"
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root):
            pass
        free = shutil.disk_usage(root).free / 2**30
        report("ok" if free > 20 else "warn", "storage root", f"{root} ({source}), {free:.0f} GiB free")
    except OSError as error:
        report("fail", "storage root", f"{root} ({source}) is not writable: {error}")
    data = root / "data_cache"
    for name, path, hint in [
        ("text8 data", data / "text8" / "raw", "uv run python scripts/prepare_data.py text8"),
        ("TinyStories cohort", data / "tinystories_bpe" / "allowed_ids.json",
         "uv run python scripts/prepare_data.py tinystories --parquet DIR"),
    ]:
        report("ok" if path.exists() else "warn", name,
               str(path) if path.exists() else f"missing (needed for that task); {hint}")
    parquet = os.environ.get("TINYSTORIES_DATA_DIR")
    report("ok" if parquet and Path(parquet).is_dir() else "warn", "TINYSTORIES_DATA_DIR",
           parquet or "unset (needed for TinyStories tasks; see docs/tasks/tinystories.md)")


def check_tokenizers():
    sys.path.insert(0, str(REPO))
    from core.data.cache import file_sha256

    folder = REPO / "artifacts" / "tokenizers" / "tinystories-bpe"
    bad = []
    for line in (folder / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        path = folder / name
        if not path.exists() or file_sha256(path) != digest:
            bad.append(name)
    report("fail" if bad else "ok", "tokenizers", f"checksum mismatch: {', '.join(bad)}" if bad else "SHA256SUMS match")


def check_compose():
    try:
        sys.path.insert(0, str(REPO))
        from core.sweeps.configuration import compose

        config = compose("smoke")
        report("ok", "configs compose", f"smoke -> {config.algorithm._target_}")
    except Exception as error:  # report any composition failure as one line
        report("fail", "configs compose", f"{type(error).__name__}: {error}")


def run(name, command, cwd=REPO, timeout=900):
    started = time.time()
    done = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    seconds = time.time() - started
    if done.returncode:
        tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-15:])
        report("fail", name, f"exit {done.returncode} after {seconds:.0f}s\n{tail}")
        return False
    report("ok", name, f"{seconds:.0f}s")
    return True


def smoke():
    out = storage_root() / "doctor" / time.strftime("smoke-%Y%m%d-%H%M%S")
    train = [sys.executable, "main.py", "+experiment=smoke", f"hydra.run.dir={out}/train"]
    if not run("smoke train (2 CPU steps)", train):
        return
    checkpoints = sorted((out / "train").rglob("last.ckpt"))
    if not checkpoints:
        report("fail", "smoke checkpoint", f"no last.ckpt under {out}/train")
        return
    evaluate = [sys.executable, "main.py", "+experiment=smoke", "mode=evaluate",
                f"evaluation.checkpoint={checkpoints[0]}", f"hydra.run.dir={out}/evaluate"]
    run("smoke evaluate", evaluate)
    print(f"       outputs: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="train and evaluate the 2-step CPU smoke model")
    parser.add_argument("--tests", action="store_true", help="run a quick subset of the test suite")
    args = parser.parse_args(argv)
    check_platform()
    check_packages()
    check_storage()
    check_tokenizers()
    have_torch = importlib.util.find_spec("torch") is not None
    if have_torch:
        check_compose()
    else:
        report("skip", "configs compose", "needs torch")
    if args.smoke:
        smoke() if have_torch else report("skip", "smoke", "needs torch")
    if args.tests:
        tests = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *QUICK_TESTS]
        run("quick tests", tests) if have_torch else report("skip", "quick tests", "needs torch")
    failed = results.count("fail")
    print(f"\n{results.count('ok')} ok, {results.count('warn')} warn, {failed} fail, {results.count('skip')} skip")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
