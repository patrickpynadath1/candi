"""Start an allocation-scoped Ray cluster and submit existing campaign runners."""

import argparse
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def authenticate(directory):
    os.environ["RAY_AUTH_MODE"] = "token"
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(directory / "auth_token")
    os.environ.pop("RAY_AUTH_TOKEN", None)


def prepare_ray_worker():
    """Ray tasks are independent jobs, not ranks of the surrounding Slurm step."""
    if "SLURM_JOB_ID" in os.environ:
        os.environ["CANDI_SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]
    for key in list(os.environ):
        if key.startswith("SLURM_") or key in {
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "GROUP_RANK",
            "NODE_RANK",
        }:
            del os.environ[key]
    # Ray's GPU assignment and authentication must remain intact.


def node_command(args, head, address):
    command = [
        str(Path(sys.executable).parent / "ray"),
        "start",
        "--num-cpus",
        str(args.cpus),
        "--num-gpus",
        str(args.gpus),
        "--disable-usage-stats",
        "--block",
    ]
    if head:
        command += [
            "--head",
            "--node-ip-address",
            address,
            "--port",
            str(args.port),
            "--dashboard-host",
            "0.0.0.0",
            "--dashboard-port",
            str(args.dashboard_port),
            "--include-dashboard=true",
            "--temp-dir",
            f"/tmp/candi-ray-{os.getuid()}-{os.environ['SLURM_JOB_ID']}-{os.environ.get('SLURM_RESTART_COUNT', '0')}",
        ]
    else:
        command += ["--address", f"{address}:{args.port}"]
    return command


def start(args):
    job = os.environ["SLURM_JOB_ID"]
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # A requeue gets a distinct record/token; stale clients cannot submit to it.
    directory = directory / f"{job}-{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    directory.mkdir(mode=0o700, exist_ok=False)
    token = directory / "auth_token"
    with token.open("x", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(secrets.token_hex(32))
    authenticate(directory)
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True
    ).split()
    ip = args.head_ip or socket.gethostbyname(nodes[0])
    info = dict(
        job_id=job,
        head=nodes[0],
        address=f"{ip}:{args.port}",
        dashboard=f"http://{ip}:{args.dashboard_port}",
        nodes=nodes,
        gpus_per_node=args.gpus,
        cpus_per_node=args.cpus,
        python=sys.executable,
        source=str(Path.cwd()),
        state="starting",
    )
    record = directory / "cluster.json"
    save(record, info)
    print(f"Connection record: {record}", flush=True)
    children = []
    logs = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for index, node in enumerate(nodes):
            if stopping:
                break
            # Slurm owns each process tree. Never use global `ray stop`.
            log = (directory / f"{node}.log").open("a")
            logs.append(log)
            command = [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                f"--nodelist={node}",
                f"--cpus-per-task={args.cpus}",
                f"--gres=gpu:{args.gpus}",
                *node_command(args, index == 0, ip),
            ]
            children.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            )
            if index == 0:
                from ray.job_submission import JobSubmissionClient

                deadline = time.monotonic() + args.startup_timeout
                while not stopping:
                    try:
                        JobSubmissionClient(info["dashboard"]).get_version()
                        break
                    except Exception:
                        if (
                            children[0].poll() is not None
                            or time.monotonic() > deadline
                        ):
                            raise RuntimeError(
                                f"Ray head failed to become ready; see {log.name}"
                            )
                        time.sleep(2)
        if not stopping:
            import ray

            ray.init(address=info["address"])
            deadline = time.monotonic() + args.startup_timeout
            while not stopping:
                alive = [n for n in ray.nodes() if n["Alive"]]
                if len(alive) == len(nodes) and all(
                    n["Resources"].get("GPU", 0) == args.gpus for n in alive
                ):
                    info["state"] = "ready"
                    break
                if (
                    any(p.poll() is not None for p in children)
                    or time.monotonic() > deadline
                ):
                    raise RuntimeError(
                        "Not all allocated Ray workers registered their GPUs"
                    )
                time.sleep(2)
            ray.shutdown()
            save(record, info)
    except Exception as exc:
        info.update(state="failed", error=str(exc))
        save(record, info)
        print(
            f"{exc}; retaining allocation for inspection.", file=sys.stderr, flush=True
        )
    try:
        while not stopping:
            if info["state"] == "ready" and any(p.poll() is not None for p in children):
                info.update(
                    state="degraded", error="A Ray Slurm step exited; inspect node logs"
                )
                save(record, info)
            time.sleep(2)
    finally:
        info["state"] = "stopped"
        save(record, info)
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        for log in logs:
            log.close()


def entrypoint(info, args):
    module = "core.sweeps.smoke" if args.smoke else "core.sweeps.nested_tuning"
    command = [
        info["python"],
        "-m",
        module,
        str(args.manifest.resolve()),
        "--output",
        str(args.output.resolve()),
        "--address",
        info["address"],
    ]
    if args.smoke:
        command += [
            "--train-steps",
            str(args.train_steps),
            "--sampler-steps",
            str(args.sampler_steps),
            "--examples",
            str(args.examples),
        ]
        if args.no_compile:
            command.append("--no-compile")
    else:
        command += [
            "--case-concurrency",
            str(args.case_concurrency),
            "--concurrency",
            "1",
        ]
    return "cd " + shlex.quote(info["source"]) + " && exec " + shlex.join(command)


def submit(args):
    info = json.loads(args.cluster.read_text())
    if info["state"] != "ready":
        raise ValueError(f"Cluster is not ready: {info['state']}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    authenticate(args.cluster.resolve().parent)
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(info["dashboard"])
    # Source and environment are shared, frozen paths, not an upload of the repo/cache.
    env = {key: os.environ[key] for key in ("RAY_AUTH_MODE", "RAY_AUTH_TOKEN_PATH")}
    env.update(
        PYTHONPATH=info["source"], OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false"
    )
    submission = client.submit_job(
        submission_id=args.id,
        entrypoint=entrypoint(info, args),
        entrypoint_num_cpus=0,
        runtime_env={
            "py_executable": info["python"],
            "env_vars": env,
            "worker_process_setup_hook": "core.cluster.slurm.prepare_ray_worker",
        },
    )
    print(submission)


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def status(args):
    info = json.loads(args.cluster.read_text())
    print(json.dumps(info, indent=2))
    subprocess.run(
        ["squeue", "-j", info["job_id"], "-o", "%.18i %.9T %.10L %R"], check=True
    )
    authenticate(args.cluster.resolve().parent)
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient(info["dashboard"])
    for job in client.list_jobs():
        print(job.submission_id, job.status, job.message or "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("start")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--gpus", type=positive, required=True)
    p.add_argument("--cpus", type=positive, required=True)
    p.add_argument("--port", type=int, default=6379)
    p.add_argument("--dashboard-port", type=int, default=8265)
    p.add_argument("--head-ip")
    p.add_argument("--startup-timeout", type=positive, default=300)
    p = commands.add_parser("submit")
    p.add_argument("--cluster", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--case-concurrency", type=positive, default=30)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--train-steps", type=positive, default=1)
    p.add_argument("--sampler-steps", type=positive, default=1)
    p.add_argument("--examples", type=positive, default=1)
    p.add_argument("--no-compile", action="store_true")
    p = commands.add_parser("status")
    p.add_argument("--cluster", type=Path, required=True)
    args = parser.parse_args()
    {"start": start, "submit": submit, "status": status}[args.command](args)


if __name__ == "__main__":
    main()
