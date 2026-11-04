"""Submit and manage CANDI jobs on the persistent server Ray cluster from the laptop.

Run without the project environment (the lock pins CUDA PyTorch):

    uv run --no-project --python 3.10 --with 'ray[default]==2.59.0' \
        python scripts/cluster/candi_remote.py submit --name smoke -- \
        python main.py +experiment=smoke

Requires the SSH tunnel from ``candi_remote.py tunnel`` and the cluster token at
``~/.config/candi/ray_token``. Every command runs on the server through
``scripts/cluster/entrypoint.sh`` from a persistent source snapshot.
"""

import argparse
from fnmatch import fnmatch
import hashlib
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
HOST = os.environ.get("CANDI_REMOTE_HOST", "ruqi6000")
DASHBOARD_PORT = int(os.environ.get("CANDI_REMOTE_DASHBOARD_PORT", "8275"))
ADDRESS = os.environ.get("CANDI_REMOTE_ADDRESS", f"http://127.0.0.1:{DASHBOARD_PORT}")
STORAGE_ROOT = os.environ.get("CANDI_REMOTE_STORAGE_ROOT", "/home/patrick/.cache/duo")
TOKEN_PATH = Path(
    os.environ.get("CANDI_REMOTE_TOKEN", "~/.config/candi/ray_token")
).expanduser()
# Ray also honours .gitignore; these keep large or secret files out even if untracked.
EXCLUDES = [
    ".git",
    ".venv",
    ".env",
    ".env.*",
    "core/outputs",
    "core/data_cache",
    "results",
    "wandb",
    "*.ckpt",
    "*.pt",
    "*.safetensors",
    "*.sqlite",
]
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _excluded(path):
    """Gitignore-style match, as Ray applies EXCLUDES: anchored if the pattern has a slash."""
    parts = PurePosixPath(path).parts
    for pattern in EXCLUDES:
        if "/" in pattern:
            if path == pattern or path.startswith(pattern + "/"):
                return True
        elif any(fnmatch(part, pattern) for part in parts):
            return True
    return False


def upload_files(repo=REPO):
    """Tracked and untracked, non-ignored files that Ray will upload."""
    output = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout.decode()
    files = sorted({f for f in output.split("\0") if f and not _excluded(f)})
    return [f for f in files if (repo / f).is_file()]


def source_identity(repo=REPO):
    """Return (revision, content hash, lock hash) for the files being uploaded."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    digest = hashlib.sha256()
    size = 0
    for name in upload_files(repo):
        data = (repo / name).read_bytes()
        size += len(data)
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    if size > MAX_UPLOAD_BYTES:
        raise SystemExit(f"Upload would be {size / 2**20:.0f} MiB; check EXCLUDES/.gitignore")
    lock = hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest()[:12]
    return revision, digest.hexdigest()[:12], lock


def runtime_env(revision, content, lock, storage_root=STORAGE_ROOT):
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    return {
        "working_dir": str(REPO),
        "excludes": EXCLUDES,
        # Workers use the same locked environment the entrypoint builds.
        "py_executable": f"{storage_root}/envs/{lock}/bin/python",
        "env_vars": {
            "CANDI_STORAGE_ROOT": storage_root,
            "CANDI_SOURCE_ID": f"{revision[:12]}-{content}",
            "CANDI_SOURCE_REV": revision,
            "CANDI_SOURCE_DIRTY": "1" if dirty else "0",
            "CANDI_LOCK": lock,
            "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",
        },
    }


def client():
    if not TOKEN_PATH.is_file():
        raise SystemExit(
            f"Missing {TOKEN_PATH}; copy it with: candi_remote.py fetch-token"
        )
    # Ray reads these at import time.
    os.environ.setdefault("RAY_AUTH_MODE", "token")
    os.environ.setdefault("RAY_AUTH_TOKEN_PATH", str(TOKEN_PATH))
    from ray.job_submission import JobSubmissionClient

    return JobSubmissionClient(ADDRESS)


def submit(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("Give the command to run after --")
    revision, content, lock = source_identity()
    env = runtime_env(revision, content, lock)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    submission_id = f"{args.name}-{stamp}" if args.name else None
    entrypoint = shlex.join(["bash", "scripts/cluster/entrypoint.sh", *command])
    job = client().submit_job(
        entrypoint=entrypoint,
        submission_id=submission_id,
        runtime_env=env,
        # The driver only coordinates; GPUs are reserved by worker tasks.
        entrypoint_num_cpus=0,
        metadata={
            "source_rev": revision,
            "source_id": env["env_vars"]["CANDI_SOURCE_ID"],
            "source_dirty": env["env_vars"]["CANDI_SOURCE_DIRTY"],
            "lock": lock,
        },
    )
    print(job)
    if env["env_vars"]["CANDI_SOURCE_DIRTY"] == "1":
        print("note: uncommitted changes were included in this snapshot", file=sys.stderr)
    if args.follow:
        _follow(job)


def _follow(job):
    import asyncio

    async def tail():
        async for lines in client().tail_job_logs(job):
            print(lines, end="")

    asyncio.run(tail())
    print(f"\n{job}: {client().get_job_status(job)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("submit", help="Upload a snapshot and run a command on the server")
    p.add_argument("--name", help="Submission ID prefix; a timestamp is appended")
    p.add_argument("--follow", action="store_true", help="Stream logs until the job ends")
    p.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("status", "logs", "stop", "info"):
        p = commands.add_parser(name)
        p.add_argument("job")
        if name == "logs":
            p.add_argument("--follow", action="store_true")
    commands.add_parser("list")
    commands.add_parser("gpus", help="Advisory GPU use: nvidia-smi plus Ray reservations")
    commands.add_parser("tunnel", help=f"Forward port {DASHBOARD_PORT} from {HOST} (blocks)")
    commands.add_parser("fetch-token", help=f"Copy the cluster token from {HOST}")
    args = parser.parse_args()

    if args.action == "submit":
        submit(args)
    elif args.action == "status":
        print(client().get_job_status(args.job))
    elif args.action == "info":
        print(client().get_job_info(args.job))
    elif args.action == "logs":
        _follow(args.job) if args.follow else print(client().get_job_logs(args.job))
    elif args.action == "stop":
        print(client().stop_job(args.job))
    elif args.action == "list":
        for job in sorted(client().list_jobs(), key=lambda j: j.start_time or 0):
            print(job.submission_id, job.status, (job.metadata or {}).get("source_id", ""))
    elif args.action == "gpus":
        script = (
            "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
            "--format=csv; echo; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory "
            "--format=csv; echo; ray status"
        )
        args.name, args.follow, args.command = "gpus", True, ["bash", "-c", script]
        submit(args)
    elif args.action == "tunnel":
        os.execvp("ssh", ["ssh", "-N", "-L", f"{DASHBOARD_PORT}:127.0.0.1:{DASHBOARD_PORT}", HOST])
    elif args.action == "fetch-token":
        TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        token = subprocess.run(
            ["ssh", HOST, f"cat {STORAGE_ROOT}/ray/rj.token"],
            check=True,
            capture_output=True,
        ).stdout
        TOKEN_PATH.touch(mode=0o600)
        TOKEN_PATH.write_bytes(token)
        print(f"Saved {TOKEN_PATH}")


if __name__ == "__main__":
    main()
